import copy
import os
import pickle
from typing import List
from typing import Union

import faiss
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F


class FaissNN(object):
    def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
        """FAISS Nearest neighbourhood search.

        Args:
            on_gpu: If set true, nearest neighbour searches are done on GPU.
            num_workers: Number of workers to use with FAISS for similarity search.
        """
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None

    def _gpu_cloner_options(self):
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        if self.on_gpu:
            # For the non-gpu faiss python package, there is no GpuClonerOptions
            # so we can not make a default in the function header.
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):  # dimension:1024
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
            )
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        """
        Adds features to the FAISS search index.

        Args:
            features: Array of size NxD.[16385,1024]
        """
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])  # {IndexFlatL2}d=1024
        self._train(self.search_index, features)
        self.search_index.add(features)

    def _train(self, _index, _features):
        pass

    def run(
            self,
            n_nearest_neighbours,
            query_features: np.ndarray,
            index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns distances and indices of nearest neighbour search.

        Args:
            query_features: Features to retrieve.[1568,1024]
            index_features: [optional] Index features to search in.
        """
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)

        # Build a search index just for this search.
        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


class ApproximateFaissNN(FaissNN):
    def _train(self, index, features):
        index.train(features)

    def _gpu_cloner_options(self):
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        return cloner

    def _create_index(self, dimension):
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dimension),
            dimension,
            512,  # n_centroids
            64,  # sub-quantizers
            8,
        )  # nbits per code
        return self._index_to_gpu(index)


class _BaseMerger:
    def __init__(self):
        """Merges feature embedding by name."""

    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxC
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(
            axis=-1
        )


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxCWH
        return features.reshape(len(features), -1)


class Myprocess(torch.nn.Module):
    def __init__(self, input_dims):
        super(Myprocess, self).__init__()
        self.preprocessing_modules = torch.nn.ModuleList()
        for i in range(0, len(input_dims) - 1):
            module = MeanMapper(input_dims[i])
            self.preprocessing_modules.append(module)
        module = MeanMapper(input_dims[-1])
        self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        # print(_features[0].size())
        # torch.Size([1568, 1024])
        return torch.cat(_features, dim=1)


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims  # [512,1024]
        self.output_dim = output_dim  # 1024

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        # print(_features[0].size())
        # torch.Size([1568, 1024])
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim  # 1024

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)  # torch.Size([1568, 1, 4608]) torch.Size([1568, 1, 9216])
        # print(features.size())
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)  # 自适应池化


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        # torch.Size([1568, 1, 2048])
        # print(features.size())
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        # torch.Size([1568, 1, 1024])
        # print(features.size())
        return features.reshape(len(features), -1)


class RescaleSegmentor:
    def __init__(self, device, target_size=224):
        self.device = device
        self.target_size = target_size
        # print(self.target_size)
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):
        # patch_scores (2, 28, 28)
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)  # torch.Size([2, 1, 28, 28])
            # print(_scores.size())
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )  # torch.Size([2, 1, 224, 224])
            # print(_scores.size())
            _scores = _scores.squeeze(1)  # torch.Size([2, 224, 224])
            # print(_scores.size())
            patch_scores = _scores.cpu().numpy()  # (2, 224, 224)
            # print(patch_scores.shape)

        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device, mode):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        self.mode = mode
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if self.mode == 'resnet':
                if "." in extract_layer:
                    extract_block, extract_idx = extract_layer.split(".")
                    network_layer = backbone.__dict__["_modules"][extract_block]
                    if extract_idx.isnumeric():
                        extract_idx = int(extract_idx)
                        network_layer = network_layer[extract_idx]
                    else:
                        network_layer = network_layer.__dict__["_modules"][extract_idx]
                else:
                    network_layer = backbone.__dict__["_modules"][extract_layer]

                if isinstance(network_layer, torch.nn.Sequential):
                    self.backbone.hook_handles.append(
                        network_layer[-1].register_forward_hook(forward_hook)
                    )
                else:
                    self.backbone.hook_handles.append(
                        network_layer.register_forward_hook(forward_hook)
                    )
            elif self.mode == 'swin1':
                if extract_layer == 'layer2':
                    backbone.layers.__dict__["_modules"]["1"].register_forward_hook(forward_hook)
                elif extract_layer == 'layer3':
                    backbone.layers.__dict__["_modules"]["2"].register_forward_hook(forward_hook)
            elif self.mode == 'swin2':
                if extract_layer == 'layer2':
                    backbone.layers.__dict__["_modules"]["1"].register_forward_hook(forward_hook)
                elif extract_layer == 'layer3':
                    backbone.layers.__dict__["_modules"]["2"].register_forward_hook(forward_hook)
                elif extract_layer == 'layer1':
                    backbone.layers.__dict__["_modules"]["0"].register_forward_hook(forward_hook)
            elif self.mode == 'swin3':
                if extract_layer == 'layer2':
                    backbone.layers.__dict__["_modules"]["1"].register_forward_hook(forward_hook)
                elif extract_layer == 'layer3':
                    backbone.layers.__dict__["_modules"]["2"].register_forward_hook(forward_hook)
                elif extract_layer == 'layer4':
                    backbone.layers.__dict__["_modules"]["3"].register_forward_hook(forward_hook)

        self.to(self.device)

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        if self.mode == 'swin1' or self.mode == 'swin2' or self.mode == 'swin3':
            return [_output[layer].shape[2] for layer in self.layers_to_extract_from]
        else:
            return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        # self.raise_exception_to_break = copy.deepcopy(
        #     layer_name == last_layer_to_extract
        # )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        # print(output.size())
        # if self.raise_exception_to_break:
        #     raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass


class NearestNeighbourScorer(object):
    def __init__(self, n_nearest_neighbours: int, nn_method=FaissNN(False, 4)) -> None:
        """
        Neearest-Neighbourhood Anomaly Scorer class.

        Args:
            n_nearest_neighbours: [int] Number of nearest neighbours used to
                determine anomalous pixels.
            nn_method: Nearest neighbour search method.
        """
        self.feature_merger = ConcatMerger()

        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method

        self.imagelevel_nn = lambda query: self.nn_method.run(
            n_nearest_neighbours, query
        )
        self.pixelwise_nn = lambda query, index: self.nn_method.run(1, query, index)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        """Calls the fit function of the nearest neighbour method.

        Args:
            detection_features: [list of np.arrays]
                [[bs x d_i] for i in n] Contains a list of
                np.arrays for all training images corresponding to respective
                features VECTORS (or maps, but will be resized) produced by
                some backbone network which should be used for image-level
                anomaly detection.
        """
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )  # ndarray[16385,1024]
        self.nn_method.fit(self.detection_features)

    def predict(
            self, query_features: List[np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts anomaly score.

        Searches for nearest neighbours of test images in all
        support training images.

        Args:
             detection_query_features: [dict of np.arrays] List of np.arrays
                 corresponding to the test features generated by
                 some backbone network.
        """
        query_features = self.feature_merger.merge(
            query_features,
        )  # (1568,1024)
        # print(query_features.shape)
        # (1568,1)distance (1568,1)ID
        query_distances, query_nns = self.imagelevel_nn(query_features)  # distance id
        # print(query_distances.shape,query_nns.shape)
        # print(query_distances,query_nns)
        # (1568,1)
        anomaly_scores = np.mean(query_distances, axis=-1)
        # print(query_distances.shape)
        return anomaly_scores, query_distances, query_nns

    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
            self,
            save_folder: str,
            save_features_separately: bool = False,
            prepend: str = "",
    ) -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )
