"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,  # (3, 224, 224)
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            anomaly_score_num_nn=1,
            featuresampler=patchcore.sampler.IdentitySampler(),
            nn_method=patchcore.common.FaissNN(False, 4),
            mode='resnet',
            **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.mode = mode

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})
        #######
        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, mode
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)  # [512,1024]
        self.forward_modules["feature_aggregator"] = feature_aggregator
        # if self.mode == 'swin3':
        #     preprocessing = patchcore.common.Myprocess(feature_dimensions)
        #     self.forward_modules["preprocessing"] = preprocessing

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing
        #########
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                # print(len(features))
                # 1568
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]
        # class 'torch.Tensor'
        # print(type(features[0]))
        # torch.Size([2, 512, 28, 28]) torch.Size([2, 1024, 14, 14])
        # print(features[0].size(), features[1].size())
        patch_shapes = []
        if self.mode == 'swin1':
            layer2 = 14
            layer3 = 7
            features[0] = features[0].reshape(features[0].shape[0], layer2, layer2, features[0].shape[2])
            features[1] = features[1].reshape(features[1].shape[0], layer3, layer3, features[1].shape[2])
            # print(features[0].size(), features[1].size())
            features[0] = features[0].permute(0, -1, 1, 2)
            features[1] = features[1].permute(0, -1, 1, 2)
            features = [
                self.patch_maker.patchify(x, return_spatial_info=True) for x in features
            ]  # [(,[28,28])(,[14,14])]
            # print("features", features[0][0].size())
            patch_shapes = [x[1] for x in features]  # [[28, 28], [14, 14]]
            # print("patch_shapes", patch_shapes)
            features = [x[0] for x in features]
            # torch.Size([2, 784, 512, 3, 3]) torch.Size([2, 196, 1024, 3, 3])
            # print("features", features[0].size(), features[1].size())
            ref_num_patches = patch_shapes[0]  # [28,28]
            for i in range(1, len(features)):
                _features = features[i]  # torch.Size([2, 784, 512, 3, 3]) torch.Size([2, 196, 1024, 3, 3])
                patch_dims = patch_shapes[i]  # [28,28] [14,14]
                # TODO(pgehler): Add comments
                _features = _features.reshape(
                    _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                )  # torch.Size([2, 28 , 28, 512, 3, 3]) torch.Size([2, 14, 14, 1024, 3, 3])
                _features = _features.permute(0, -3, -2, -1, 1, 2)
                # torch.Size([2, 512 , 3, 3 , 28, 28])  torch.Size([2, 1024, 3, 3, 14, 14])
                perm_base_shape = _features.shape
                _features = _features.reshape(-1, *_features.shape[-2:])
                # torch.Size([18432, 14, 14])
                _features = F.interpolate(
                    _features.unsqueeze(1),  # [18432 , 1 , 14, 14]
                    size=(ref_num_patches[0], ref_num_patches[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                # torch.Size([18432, 1, 28, 28])
                _features = _features.squeeze(1)
                # torch.Size([18432, 28, 28])
                _features = _features.reshape(
                    *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                )
                # torch.Size([2, 1024, 3, 3, 28, 28])
                _features = _features.permute(0, -2, -1, 1, 2, 3)
                # torch.Size([2, 28, 28, 1024, 3, 3])
                _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
                # torch.Size([2, 784, 1024, 3, 3])
                features[i] = _features
            features = [x.reshape(-1, *x.shape[-3:]) for x in features]
            # torch.Size([1568, 512, 3, 3]) torch.Size([1568, 1024, 3, 3])
            # print(features[0].size(), features[1].size())
            # As different feature backbones & patching provide differently
            # sized features, these are brought into the correct form here.
            features = self.forward_modules["preprocessing"](features)
            # torch.Size([1568, 2, 1024])
            # print(features.size())
            features = self.forward_modules["preadapt_aggregator"](features)
            # torch.Size([1568, 1024])
            # print(features.size())
        elif self.mode == 'swin2':
            # layer1 = 28
            layer2 = 14
            layer3 = 7
            # features[0] = features[0].reshape(features[0].shape[0], layer1, layer1, features[0].shape[2])
            # features[1] = features[1].reshape(features[1].shape[0], layer2, layer2, features[1].shape[2])
            # features[2] = features[2].reshape(features[2].shape[0], layer3, layer3, features[2].shape[2])
            # patch_shapes = [[layer1, layer1], [layer2, layer2], [layer3, layer3]]  # [[28, 28], [14, 14],[7,7]]
            patch_shapes = [[layer2, layer2], [layer3, layer3]]  # [[28, 28], [14, 14],[7,7]]
            ref_num_patches = patch_shapes[0]  # [28,28]
            # torch.Size([2, 784,256]) torch.Size([2, 196,512]) torch.Size([2, 49,1024])
            for i in range(1, len(features)):
                _features = features[i]
                patch_dims = patch_shapes[i]
                # print("1", _features.size())
                _features = _features.reshape(
                    _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                )  # torch.Size([2, 14, 14, 512])
                # print("t1", _features.size())
                _features = _features.permute(0, -1, 1, 2)  # torch.Size([2, 512, 14, 14])
                # print("t", _features.size())
                perm_base_shape = _features.shape
                _features = _features.reshape(-1, *_features.shape[-2:])  # torch.Size([1024, 14,14])
                # print("2", _features.size())
                _features = F.interpolate(
                    _features.unsqueeze(1),  # [1024 , 1 , 14, 14]
                    size=(ref_num_patches[0], ref_num_patches[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                # print("3", _features.size())
                _features = _features.squeeze(1)  # torch.Size([1024,28,28])
                # print("4", _features.size())
                _features = _features.reshape(
                    *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                )  # [2 , 512 , 28, 28]
                # print("5", _features.size())
                _features = _features.permute(0, -2, -1, 1)  # [2 ,28, 28,512]
                # print("6", _features.size())
                _features = _features.reshape(len(_features), -1, *_features.shape[-1:])  # [2 ,784,512]
                # print("7", _features.size())
                features[i] = _features
            features = [x.reshape(-1, *x.shape[-1:]) for x in features]
            # print(features[0].size(), features[1].size(), features[2].size())
            # print(features[0].size(), features[1].size())
            # torch.Size([1568, 256]) torch.Size([1568, 512]) torch.Size([1568, 1024])
            # features = torch.cat(features, dim=1)
            # print("8", features.size())
            features = self.forward_modules["preprocessing"](features)
            # torch.Size([1568, 2, 1024])
            # print(features.size())
            features = self.forward_modules["preadapt_aggregator"](features)
            # print(features.size())
        elif self.mode == 'swin3':
            # layer1 = 28
            layer2 = 14
            layer3 = 7
            layer4 = 7
            # features[0] = features[0].reshape(features[0].shape[0], layer1, layer1, features[0].shape[2])
            # features[1] = features[1].reshape(features[1].shape[0], layer2, layer2, features[1].shape[2])
            # features[2] = features[2].reshape(features[2].shape[0], layer3, layer3, features[2].shape[2])
            # patch_shapes = [[layer1, layer1], [layer2, layer2], [layer3, layer3]]  # [[28, 28], [14, 14],[7,7]]
            patch_shapes = [[layer2, layer2], [layer3, layer3], [layer4, layer4]]  # [[28, 28], [14, 14],[7,7]]
            ref_num_patches = patch_shapes[0]  # [28,28]
            # torch.Size([2, 784,256]) torch.Size([2, 196,512]) torch.Size([2, 49,1024])
            for i in range(1, len(features)):
                _features = features[i]
                patch_dims = patch_shapes[i]
                # print("1", _features.size())
                _features = _features.reshape(
                    _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                )  # torch.Size([2, 14, 14, 512])
                # print("t1", _features.size())
                _features = _features.permute(0, -1, 1, 2)  # torch.Size([2, 512, 14, 14])
                # print("t", _features.size())
                perm_base_shape = _features.shape
                _features = _features.reshape(-1, *_features.shape[-2:])  # torch.Size([1024, 14,14])
                # print("2", _features.size())
                _features = F.interpolate(
                    _features.unsqueeze(1),  # [1024 , 1 , 14, 14]
                    size=(ref_num_patches[0], ref_num_patches[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                # print("3", _features.size())
                _features = _features.squeeze(1)  # torch.Size([1024,28,28])
                # print("4", _features.size())
                _features = _features.reshape(
                    *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                )  # [2 , 512 , 28, 28]
                # print("5", _features.size())
                _features = _features.permute(0, -2, -1, 1)  # [2 ,28, 28,512]
                # print("6", _features.size())
                _features = _features.reshape(len(_features), -1, *_features.shape[-1:])  # [2 ,784,512]
                # print("7", _features.size())
                features[i] = _features
            features = [x.reshape(-1, *x.shape[-1:]) for x in features]
            # print(features[0].size(), features[1].size(), features[2].size())
            # print(features[0].size(), features[1].size())
            # torch.Size([1568, 256]) torch.Size([1568, 512]) torch.Size([1568, 1024])
            # features = torch.cat(features, dim=1)
            # print("8", features.size())
            features = self.forward_modules["preprocessing"](features)
            # torch.Size([1568, 2, 1024])
            # print(features.size())
            features = self.forward_modules["preadapt_aggregator"](features)
            # print(features.size())
        else:
            features = [
                self.patch_maker.patchify(x, return_spatial_info=True) for x in features
            ]  # [(,[28,28])(,[14,14])]
            # print("features", features[0][0].size())
            patch_shapes = [x[1] for x in features]  # [[28, 28], [14, 14]]
            # print("patch_shapes", patch_shapes)
            features = [x[0] for x in features]
            # torch.Size([2, 784, 512, 3, 3]) torch.Size([2, 196, 1024, 3, 3])
            # print("features", features[0].size(), features[1].size())
            ref_num_patches = patch_shapes[0]  # [28,28]
            for i in range(1, len(features)):
                _features = features[i]  # torch.Size([2, 784, 512, 3, 3]) torch.Size([2, 196, 1024, 3, 3])
                patch_dims = patch_shapes[i]  # [28,28] [14,14]
                # TODO(pgehler): Add comments
                _features = _features.reshape(
                    _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                )  # torch.Size([2, 28 , 28, 512, 3, 3]) torch.Size([2, 14, 14, 1024, 3, 3])
                _features = _features.permute(0, -3, -2, -1, 1, 2)
                # torch.Size([2, 512 , 3, 3 , 28, 28])  torch.Size([2, 1024, 3, 3, 14, 14])
                perm_base_shape = _features.shape
                _features = _features.reshape(-1, *_features.shape[-2:])
                # torch.Size([18432, 14, 14])
                _features = F.interpolate(
                    _features.unsqueeze(1),  # [18432 , 1 , 14, 14]
                    size=(ref_num_patches[0], ref_num_patches[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                # torch.Size([18432, 1, 28, 28])
                _features = _features.squeeze(1)
                # torch.Size([18432, 28, 28])
                _features = _features.reshape(
                    *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                )
                # torch.Size([2, 1024, 3, 3, 28, 28])
                _features = _features.permute(0, -2, -1, 1, 2, 3)
                # torch.Size([2, 28, 28, 1024, 3, 3])
                _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
                # torch.Size([2, 784, 1024, 3, 3])
                features[i] = _features
            features = [x.reshape(-1, *x.shape[-3:]) for x in features]
            # torch.Size([1568, 512, 3, 3]) torch.Size([1568, 1024, 3, 3])
            # print(features[0].size(), features[1].size())
            # As different feature backbones & patching provide differently
            # sized features, these are brought into the correct form here.
            features = self.forward_modules["preprocessing"](features)
            # torch.Size([1568, 2, 1024])
            # print(features.size())
            features = self.forward_modules["preadapt_aggregator"](features)
            # torch.Size([1568, 1024])
            # print(features.size())
        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
                input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]  # 对应之前迭代的字典
                features.append(_image_to_features(image))
        # print(type(features))  # 105 list
        # print(type(features[0]))  # 1568/784 list
        # print(type(features[0][0]))  # 1024 ndarray
        features = np.concatenate(features, axis=0)  # 104*1568+784=163856,1024
        features = self.featuresampler.run(features)  # [16385,1024]ndarray

        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    # print(image["image_path"])
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())  # [1,1]
                    masks_gt.extend(image["mask"].numpy().tolist())  # {list:2}{list:1}{list:224}{list:224}
                    image = image["image"]
                _scores, _masks = self._predict(image)
                # print(_scores, len(_masks))
                for score, mask in zip(_scores, _masks):
                    scores.append(score)  # [7.426983,7.4016705]
                    masks.append(mask)  # {list:2}{ndarray:(224,224)}
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        # torch.Size([2, 3, 224, 224])
        images = images.to(torch.float).to(self.device)
        # print(images.size())
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]  # 2
        # print(batchsize)
        with torch.no_grad():
            # {list:1568}{list:1024} [[28,28][14,14]]
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            # print(type(features), patch_shapes)
            features = np.asarray(features)  # ndarray(1568,1024)
            # print(features.shape)
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]  # ndarray(1568,1)
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )  # (2, 784)
            # print(image_scores.shape)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)  # (2, 784, 1)
            # print(image_scores.shape)
            image_scores = self.patch_maker.score(image_scores)  # [7.426983  7.4016705]
            # print(image_scores)
            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )  # (2, 784)
            # print(patch_scores.shape)
            scales = patch_shapes[0]  # [28,28]
            # print(scales)
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])  # (2, 28, 28)
            # print(patch_scores.shape)
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)  # {list:2}{ndarray:(224,224)}
            # print(len(masks))
        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
            self,
            load_path: str,
            device: torch.device,
            nn_method: patchcore.common.FaissNN(False, 4),
            prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize  # 3
        self.stride = stride  # 1

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        # print("patchsize:", self.patchsize)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        # print("unfolded_features:", unfolded_features.size())
        number_of_total_patches = []
        for s in features.shape[-2:]:  # ([28,28])
            n_patches = (
                                s + 2 * padding - 1 * (self.patchsize - 1) - 1
                        ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
            # print("n_patches", n_patches)  # 28
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )  # [2,512,3,3,784]
        # print("unfolded_features:", unfolded_features.size())
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        # print("unfolded_features:", unfolded_features.size())  # [2, 784, 512, 3, 3]
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        # print(x.size(), x.ndim)
        # torch.Size([2, 784, 1]) 3
        while x.ndim > 1:  # torch.Size([2, 784, 1]) torch.Size([2, 784])
            # print(x.size())
            x = torch.max(x, dim=-1).values
        # print(x.size())
        # torch.Size([2])
        if was_numpy:
            return x.numpy()
        return x
