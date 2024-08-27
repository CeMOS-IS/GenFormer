import os
import numpy as np
from torchvision import datasets, transforms
from timm.data import create_transform

import medmnist
from medmnist import INFO
from robust_minisets import BloodMNISTC, BreastMNISTC, DermaMNISTC, OCTMNISTC, OrganAMNISTC, OrganCMNISTC, OrganSMNISTC, PathMNISTC, PneumoniaMNISTC, TissueMNISTC
from .utils import get_data_folder

def get_medmnist_dataset(cfg):
    base_folder = get_data_folder()
    info = INFO[cfg.DATASET.TYPE.TRAIN]
    in_channels = info["n_channels"]
    num_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])
    train_transform, test_transform = get_medmnist_transform(cfg, in_channels)
    
    DataClassInstance = normal2instance(DataClass)
    train_set = DataClassInstance(
        root=base_folder, download=True, split="train", transform=train_transform
    )
    test_set = DataClass(
        root=base_folder, download=True, split="test", transform=test_transform
    )
    train_set.labels = train_set.labels.flatten()
    test_set.labels = test_set.labels.flatten()
    num_data = len(train_set)
    
    return train_set, test_set, num_data, num_classes, in_channels



def get_medmnist_c_dataset(cfg):
    base_folder = get_data_folder()
    info = INFO[cfg.DATASET.TYPE.TRAIN]
    _, test_transform = get_medmnist_transform(cfg, 3)

    dataset_c = data_dict[cfg.DATASET.TYPE.TRAIN]['C-Set']
    test_set = dataset_c(
        split='test', transform=test_transform, download=True, root=base_folder
    )
    num_data = len(test_set)
    
    return _, test_set, num_data

def normal2instance(DataClass):
    class MedMNISTInstance(DataClass):
        """MedMNISTInstance Dataset."""

        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            return img, target

    return MedMNISTInstance


data_dict= {
  "pathmnist"		: {
    "mean" : (0.7405, 0.5330, 0.7058),
    "std"  : (0.1237, 0.1768, 0.1244),
    "C-Set" : PathMNISTC,
  },
  "dermamnist"		: {
    "mean" : (0.7631, 0.5381, 0.5614),
    "std"  : (0.1366, 0.1543, 0.1692),
    "C-Set" : DermaMNISTC
  },
  "octmnist"		: {
    "mean" : (0.1889,),
    "std"  : (0.1963,),
    "C-Set" : OCTMNISTC
  },
  "pneumoniamnist"	: {
    "mean" : (0.5719,),
    "std"  : (0.1684,),
    "C-Set" : PneumoniaMNISTC
  },
  "breastmnist"		: {
    "mean" : (0.3276,),
    "std"  : (0.2057,),
    "C-Set" : BreastMNISTC
  },
  "bloodmnist"		: {
    "mean" : (0.7943, 0.6597, 0.6962),
    "std"  : (0.2156, 0.2416, 0.1179),
    "C-Set" : BloodMNISTC
  },
  "tissuemnist"		: {
    "mean" : (0.1020,),
    "std"  : (0.1000,),
    "C-Set" : TissueMNISTC
  },
  "organamnist"		: {
    "mean" : (0.4678,),
    "std"  : (0.2975,),
    "C-Set" : OrganAMNISTC
  },
  "organcmnist"		: {
    "mean" : (0.4932,),
    "std"  : (0.2839,),
    "C-Set" :OrganCMNISTC
  },
  "organsmnist"		: {
    "mean" : (0.4950,),
    "std"  : (0.2828,),
    "C-Set" : OrganSMNISTC
  },
}

def get_medmnist_transform(cfg, in_channels=3):

    if cfg.DATASET.IN_NORMALIZATION and in_channels == 3:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # ImageNet normalization
    else:
        mean, std = data_dict[cfg.DATASET.TYPE.TRAIN]["mean"], data_dict[cfg.DATASET.TYPE.TRAIN]["std"] # MedMNIST normalization

    train_transform = []
    test_transform = []

    if not cfg.DATASET.AUGMENTATION.TT_BENCHMARK:
        train_transform += [
            transforms.Resize(cfg.DATASET.IMG_SIZE),
            transforms.RandomCrop(cfg.DATASET.IMG_SIZE, padding=cfg.DATASET.IMG_SIZE//8),
            transforms.RandomHorizontalFlip(),
            ]

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
