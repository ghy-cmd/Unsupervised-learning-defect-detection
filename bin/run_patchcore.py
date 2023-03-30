import contextlib
import logging
import os
import sys
import numpy as np

import torch.utils.data

import patchcore.utils
import patchcore.sampler
import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.metrics


def patch_core():
    backbone_names = Backbone
    backbone_names = [backbone_names]
    layers_to_extract_from = Layers
    # layers_to_extract_from = ('layer1', 'layer2', 'layer3')
    layers_to_extract_from_coll = [layers_to_extract_from]

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
                backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            # 加载骨干网络
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = patchcore.common.FaissNN(_FAISS_ON_GPU, _FAISS_NUM_WORKERS)

            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=_PRETRAIN_EMBED_DIMENSION,
                target_embed_dimension=_TARGET_EMBED_DIMENSION,
                patchsize=_PATCHSIZE,
                featuresampler=sampler,
                anomaly_scorer_num_nn=_ANOMALY_SCORER_NUM_NN,
                nn_method=nn_method,
                mode=Mode
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    # print(backbone_names, layers_to_extract_from, layers_to_extract_from_coll)
    return "get_patchcore", get_patchcore


def sampler():
    def get_sampler(device, sampler_type, percentage):
        if sampler_type == "identity":
            return patchcore.sampler.IdentitySampler()
        elif sampler_type == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif sampler_type == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return "get_sampler", get_sampler


def dataset():
    dataset_info = ["patchcore.datasets.mvtec", "MVTecDataset"]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(data_path, subdatasets, resize, train_val_split,
                        imagesize, seed, augment, batch_size,
                        num_workers):
        dataloaders = []
        # print(dataset_library.__dict__[dataset_info[1]])
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
                chanel=CHANEL,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,  # 批大小
                shuffle=False,  # 是否打乱
                num_workers=num_workers,  # 使用多少个子进程来导入数据
                pin_memory=True,  # 在数据返回前，是否将数据复制到CUDA内存中
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = 'mvtec'
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }
            dataloaders.append(dataloader_dict)
            '''
            [{
            'training': <torch.utils.data.dataloader.DataLoader object at 0x7fb3fbd75910>,
            'validation': None,
            'testing': <torch.utils.data.dataloader.DataLoader object at 0x7fb3fbd75a00>
            },{...},...]
            '''
        return dataloaders

    return "get_dataloaders", get_dataloaders


'''
False:mvtec,optical
true:altex,sdd
'''
CHANEL = False
Backbone = ('wideresnet50')  # vit_swin_base/wideresnet50
Layers = ('layer2', 'layer3')
Mode = "resnet"  # swin1/resnet
'''
/home/guihaoyue_bishe/mvtec
/home/guihaoyue_bishe/data/data_detection
/home/guihaoyue_bishe/data/data_detection/SDD
/home/guihaoyue_bishe/data/data_detection/Optical
'''
_DATA_PATH = '/home/guihaoyue_bishe/data/data_detection/Optical'
_RESULT_PATH = "/home/guihaoyue_bishe/data/res_test"
'''
MVTecAD_Results
SDD_Results
Optical_Results
'''
_LOG_PROJECT = "Optical_Results"
_LOG_GROUP = "resnet_layer2and3_10"  # swin1_layer2and3/resnet_layer2and3
'''
["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
               "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper", ]
["AITEX"]
["SDD"]
["optical_class"]
'''
_CLASSNAMES = ["optical_class"]
_RISIZE = 256
_TRAIN_VAL_SPLIT = 1.0
_IMAGESIZE = 224
_SEED = 0
_AUGMENT = False
_BATCH_SIZE = 2
_NUM_WORKERS = 8
_GPU = [8]
# "approx_greedy_coreset"/"greedy_coreset"/"identity"
_SAMPLER_TYPE = "approx_greedy_coreset"
_SAMPLER_PERCENTAGE = 0.1
_FAISS_ON_GPU = False
_FAISS_NUM_WORKERS = 8
_PRETRAIN_EMBED_DIMENSION = 1024
_TARGET_EMBED_DIMENSION = 1024
_PATCHSIZE = 3
_ANOMALY_SCORER_NUM_NN = 5
_SAVE_SEGMENTATION_IMAGES = False
_SAVE_PATCHCORE_MODEL = True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger("run_patchcore")
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    LOGGER.info("BEGIN")
    methods = (patch_core(), sampler(), dataset())
    LOGGER.info(methods)
    methods = {key: item for (key, item) in methods}
    LOGGER.info(methods)  # 将tuple变为dist
    # 构建保存路径
    run_save_path = patchcore.utils.create_storage_folder(_RESULT_PATH, _LOG_PROJECT, _LOG_GROUP)
    # 构建train_dataloader,test_dataloader,val_dataloader
    list_of_dataloaders = methods["get_dataloaders"](_DATA_PATH, _CLASSNAMES, _RISIZE, _TRAIN_VAL_SPLIT, _IMAGESIZE,
                                                     _SEED, _AUGMENT, _BATCH_SIZE, _NUM_WORKERS)
    # 选择GPU cuda:1
    device = patchcore.utils.set_torch_device(_GPU)

    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )
        # 固定seed以获得再现性
        patchcore.utils.fix_seeds(_SEED, device)
        # dataset_name:mvtec_bottle
        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()  # 释放未被占用缓存
            # (3, 224, 224)
            imagesize = dataloaders["training"].dataset.imagesize
            # 下采样（有待考证）
            sampler = methods["get_sampler"](device, _SAMPLER_TYPE, _SAMPLER_PERCENTAGE)
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                # 提取特征，下采样，初始化faissNN，index.add(features)
                PatchCore.fit(dataloaders["training"])

            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"]
                )  # scores:{list:83} segmentations:{list:83}{ndarray(224,224)} labels_gt:{list:83} masks_gt:{
                # list:83-1-224-224}
                # print(segmentations.shape)
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])  # ndarray(1,83)
            min_scores = scores.min(axis=-1).reshape(-1, 1)  # min_scores:[[1.0355244]]
            max_scores = scores.max(axis=-1).reshape(-1, 1)  # max_scores:[[12.152414]]
            scores = (scores - min_scores) / (max_scores - min_scores)  # ndarray(1,83)
            scores = np.mean(scores, axis=0)  # ndarray(83)对不同patchcore分数求平均

            segmentations = np.array(aggregator["segmentations"])  # ndarray(1,83,224,224)
            min_scores = (
                segmentations.reshape(len(segmentations), -1)  # ndarray(1,4164608)
                .min(axis=-1)  # ndarray(1,1)
                .reshape(-1, 1, 1, 1)  # ndarray(1,1,1,1)
            )  # [[[[0.05737968]]]]
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )  # [[[[10.577863]]]]
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)  # ndarray(1,83,224,224)
            segmentations = np.mean(segmentations, axis=0)  # ndarray(83,224,224)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]  # list{83} TRUE/FALSE

            # (Optional) Plot example images.
            if _SAVE_SEGMENTATION_IMAGES:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]


                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)


                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()


                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics.")  # ndarray(83) list{83} TRUE/FALSE
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )["auroc"]  # 1.0

            # Compute PRO score & PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )  # 像素级分数
            full_pixel_auroc = pixel_scores["auroc"]  # 0.9848

            # Compute PRO score & PW Auroc only images with anomalies只计算异常图分数
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            if _SAVE_PATCHCORE_MODEL:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )
