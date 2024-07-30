import os
import pickle
import glob
import json
import numpy as np
from torch.utils.data import ConcatDataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from timm.data import create_transform
from PIL import Image

from .augmentations.autoaugment import CIFAR10Policy, ImageNetPolicy

mean_std_dict = {
  "cifar10"       : {
    "mean" : (0.4914, 0.4822, 0.4465),
    "std"  : (0.2470, 0.2435, 0.2616),
  },
    "cifar100"    : {
    "mean" : (0.5071, 0.4867, 0.4408),
    "std"  : (0.2675, 0.2565, 0.2761),
  },
  "tinyimagenet"  : {
    "mean" : (0.4803, 0.4481, 0.3976),
    "std"  : (0.2764, 0.2689, 0.2816),
  },
  "pathmnist"		: {
    "mean" : (0.7405, 0.5330, 0.7058),
    "std"  : (0.1237, 0.1768, 0.1244),
  },
  "dermamnist"		: {
    "mean" : (0.7631, 0.5381, 0.5614),
    "std"  : (0.1366, 0.1543, 0.1692),
  },
  "octmnist"		: {
    "mean" : (0.1889,),
    "std"  : (0.1963,),
  },
  "pneumoniamnist"	: {
    "mean" : (0.5719,),
    "std"  : (0.1684,),
  },
  "breastmnist"		: {
    "mean" : (0.3276,),
    "std"  : (0.2057,),
  },
  "bloodmnist"		: {
    "mean" : (0.7943, 0.6597, 0.6962),
    "std"  : (0.2156, 0.2416, 0.1179),
  },
  "tissuemnist"		: {
    "mean" : (0.1020,),
    "std"  : (0.1000,),
  },
  "organamnist"		: {
    "mean" : (0.4678,),
    "std"  : (0.2975,),
  },
  "organcmnist"		: {
    "mean" : (0.4932,),
    "std"  : (0.2839,),
  },
  "organsmnist"		: {
    "mean" : (0.4950,),
    "std"  : (0.2828,),
  },
}

def get_data_folder():
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder


class DatasetPlusInstance(VisionDataset):
    """Custom dataset generated from synthetic Diffusion generated images"""
    
    def __init__(self,
                 cfg,
                 transform=None,
                 target_transform=None,
                 num_classes=None):
        self.transform = transform
        self.target_transform = target_transform
        
        self.data = []
        self.targets = []
        
        data_folder = get_data_folder()
        dataset_folder = os.path.join(data_folder, cfg.SYNTHETIC_DATA.DIR_NAME)

        if not os.path.exists(dataset_folder):
            raise NotImplementedError(f"File not found. Path not valid: {dataset_folder}.")

        # Load label dictionary
        label_names = {}
        
        try:
            if cfg.DATASET.TYPE.TRAIN == "cifar10":
                for i in range(10):
                    label_names[str(i)] = str(i)
            elif cfg.DATASET.TYPE.TRAIN == "cifar100":
                for i in range(100):
                    label_names[str(i)] = str(i)
            elif cfg.DATASET.TYPE.TRAIN == "tinyimagenet":
                for i in range(200):
                    label_names[str(i)] = str(i)
            elif cfg.DATASET.TYPE.TRAIN in ["pathmnist", "dermamnist", "octmnist", "pneumoniamnist", "breastmnist", "bloodmnist", "tissuemnist", "organamnist", "organcmnist", "organsmnist"]:
                for i in range(num_classes):
                    label_names[str(i)] = str(i)
            else:
                raise NotImplementedError(cfg.DATASET.TYPE.TRAIN)
        except:
            raise NotImplementedError(cfg.DATASET.TYPE.TRAIN)
        
        # create dataset from GAN generated images
        for class_idx in range(len(label_names)):
            dir_path = os.path.join(dataset_folder, label_names[str(class_idx)])
            img_path_list = glob.glob(f"{dir_path}/*.png")
            if cfg.SYNTHETIC_DATA.PERCENT > 1.0 or cfg.SYNTHETIC_DATA.PERCENT <= 0.0:
                raise ValueError()
            sub_img_paths = np.random.choice(img_path_list, int(cfg.SYNTHETIC_DATA.PERCENT * len(img_path_list)), replace=False)
            
            for img_path in sub_img_paths:
                self.data.append(img_path)
                self.targets.append(class_idx)
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    def __len__(self) -> int:
        return len(self.data)


def get_datasetplus_transform(cfg, in_channels=3):
    if cfg.DATASET.IN_NORMALIZATION and in_channels == 3:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # ImageNet normalization
    else:
        mean, std = mean_std_dict[cfg.DATASET.TYPE.TRAIN]["mean"], mean_std_dict[cfg.DATASET.TYPE.TRAIN]["std"]

    train_transform = []
    test_transform = []
    
    if not cfg.DATASET.AUGMENTATION.TT_BENCHMARK:
        train_transform += [
            transforms.Resize(cfg.DATASET.IMG_SIZE),
            transforms.RandomCrop(cfg.DATASET.IMG_SIZE, padding=cfg.DATASET.IMG_SIZE//8),
            transforms.RandomHorizontalFlip(),
            ]
    
    if cfg.DATASET.AUGMENTATION.AUTOAUGMENT:
        if "cifar" in cfg.DATASET.TYPE.TRAIN:
            train_transform.append(CIFAR10Policy())
        elif "imagenet" in cfg.DATASET.TYPE.TRAIN:
            train_transform.append(ImageNetPolicy())
        else:
            raise NotImplementedError("AutoAugment")
        
    if not cfg.DATASET.AUGMENTATION.AUGMIX and not cfg.DATASET.AUGMENTATION.TT_BENCHMARK:
        train_transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

    if cfg.DATASET.AUGMENTATION.TT_BENCHMARK:
        train_transform += [transform for transform in create_transform(
            input_size=(in_channels, cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE),
            is_training=True,
            use_prefetcher=True if cfg.DATASET.AUGMENTATION.AUGMIX else False,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5-inc1",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            interpolation="bicubic",
            separate=True,
            mean=mean,
            std=std,
        )]
        if cfg.DATASET.AUGMENTATION.AUGMIX:
            train_transform += [
                transforms.Lambda(lambda x: np.transpose(x, (1, 2, 0))),
                transforms.ToPILImage()
            ]
    
    test_transform += [
        transforms.Resize(cfg.DATASET.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    
    if in_channels < 3:
        train_transform  += [
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
        test_transform += [
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)
    
    return train_transform, test_transform


def get_datasetplus_dataset(train_real_set, cfg, num_classes=None, in_channels=3):
    train_transform, _ = get_datasetplus_transform(cfg, in_channels)
    train_fake_set = DatasetPlusInstance(
        cfg=cfg, transform=train_transform, num_classes=num_classes,
    )
    if cfg.SYNTHETIC_DATA.TRAINING == "pretraining":
        train_set = ConcatDataset([train_fake_set])
    elif cfg.SYNTHETIC_DATA.TRAINING == "mixed":
        train_set = ConcatDataset([train_real_set, train_fake_set])
    else:
        raise NotImplementedError(f"{cfg.SYNTHETIC_DATA.TRAINING} is not implemeted as training type.")
    num_data = len(train_set)
    
    return train_set, num_data
