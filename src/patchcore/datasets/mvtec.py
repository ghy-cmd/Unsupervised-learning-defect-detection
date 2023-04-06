import os
import PIL.Image
import torch
import random

from torch.utils.data import Dataset
from enum import Enum
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecDataset(Dataset):
    def __init__(
            self, data_path, classname,
            resize, imagesize, split,
            train_val_split=1.0, chanel=False, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.split = split  # 三种模式
        self.classnames_to_use = [classname]
        self.train_val_split = train_val_split  # 训练及测试集比例
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        '''
        imgpaths_per_class
        {'bottle':
            {'good':
                ['/home/guihaoyue_bishe/mvtec/bottle/train/good/000.png',
                ...]
            }
        }
        {'bottle':
            {'good':
                ['/home/guihaoyue_bishe/mvtec/bottle/test/good/000.png',
                ...]
            }
            {'contamination': [...,...]}
            {'broken_large': [...,...]}
            {'broken_small': [...,...]}
        }
        data_to_iterate
        [['bottle', 
        'broken_large', 
        '/home/guihaoyue_bishe/mvtec/bottle/test/broken_large/000.png', 
        '/home/guihaoyue_bishe/mvtec/bottle/ground_truth/broken_large/000_mask.png'
        ],......,]
        '''
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = transforms.Compose([
            transforms.Resize(resize),  # 256
            transforms.CenterCrop(imagesize),  # 224
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])

        self.imagesize = (3, imagesize, imagesize)

        self.chanel = chanel

    def __getitem__(self, index):
        '''
        'bottle'
        'good'
        '/home/guihaoyue_bishe/mvtec/bottle/train/good/014.png'
        None
        '''
        classname, anomaly, image_path, mask_path = self.data_to_iterate[index]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        if image.shape[0] == 1:
            print(image_path)
        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            if self.chanel:
                mask = mask.convert('L')
                threshold_value = 128  # 阈值
                mask = mask.point(lambda x: 0 if x < threshold_value else 255, '1')  # 二值化
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])
        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            # 'bottle/train/good/014.png'
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            # '/home/guihaoyue_bishe/mvtec/bottle/train'
            classpath = os.path.join(self.data_path, classname,
                                     self.split.value)
            # '/home/guihaoyue_bishe/mvtec/bottle/ground_truth'
            maskpath = os.path.join(self.data_path, classname,
                                    "ground_truth")
            anomaly_types = os.listdir(classpath)  # ['good']

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                # '/home/guihaoyue_bishe/mvtec/bottle/train/good'
                anomaly_path = os.path.join(classpath, anomaly)
                # ['000.png',......]
                anomaly_files = sorted(os.listdir(anomaly_path))
                # {'bottle': {'good': ['/home/guihaoyue_bishe/mvtec/bottle/train/good/000.png',...]}}
                if self.split == DatasetSplit.TRAIN and len(anomaly_files) > 5000:
                    num_files_to_sample = 1000
                    sampled_files = random.sample(anomaly_files, num_files_to_sample)
                    imgpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_path, x) for x in sampled_files
                    ]
                else:
                    imgpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_path, x) for x in anomaly_files
                    ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])  # 所有张数
                    train_val_split_idx = int(n_images * self.train_val_split)  # 分界点
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        data_to_iterate = []
        # {'bottle': {'good': ['/home/guihaoyue_bishe/mvtec/bottle/train/good/000.png',...]}}
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    # ['bottle', 'good', '/home/guihaoyue_bishe/mvtec/bottle/train/good/000.png']
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
