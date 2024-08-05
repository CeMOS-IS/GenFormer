import os
import numpy as np
from torchvision import transforms
from timm.data import create_transform

from robust_minisets import TinyImageNet, TinyImageNetC, TinyImageNetA, TinyImageNetR, TinyImageNetv2
from .augmentations.autoaugment import ImageNetPolicy


def get_tinyimagenet_dataset(cfg):
    base_folder = get_data_folder()
    train_transform, test_transform = get_tinyimagenet_transform(cfg)
    train_set = TinyImageNet(
        split='train', transform=train_transform, download=True, root=base_folder
    )
    test_set = TinyImageNet(
        split='test', transform=test_transform, download=True, root=base_folder
    )
    num_data = len(train_set)
    train_set.labels = train_set.labels.flatten()
    test_set.labels = test_set.labels.flatten()
    return train_set, test_set, num_data


def get_tinyimagenet_test_dataset(cfg):
    base_folder = get_data_folder()
    _, test_transform = get_tinyimagenet_transform(cfg)
    
    if cfg.DATASET.TYPE.TEST == "tinyimagenet-v2":
        test_set = TinyImageNetv2(
            split='test', transform=test_transform, download=True, root=base_folder
        )
    elif cfg.DATASET.TYPE.TEST == "tinyimagenet-a":
        test_set = TinyImageNetA(
            split='test', transform=test_transform, download=True, root=base_folder
        )
    elif cfg.DATASET.TYPE.TEST == "tinyimagenet-r":
        test_set = TinyImageNetR(
            split='test', transform=test_transform, download=True, root=base_folder
        )
    num_data = len(test_set)
    test_set.labels = test_set.labels.flatten()
    return test_set, num_data


def get_tinyimagenet_c_dataset(cfg):
    base_folder = get_data_folder()
    _, test_transform = get_tinyimagenet_transform(cfg)

    test_set = TinyImageNetC(
        split='test', transform=test_transform, download=True, root=base_folder
    )
    num_data = len(test_set)
    test_set.labels = test_set.labels.flatten()
    return _, test_set, num_data


def get_data_folder():
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder

def get_tinyimagenet_transform(cfg):
    if cfg.DATASET.IN_NORMALIZATION:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # ImageNet normalization
    else:
        mean, std = (0.4803, 0.4481, 0.3976), (0.2764, 0.2689, 0.2816) # TinyImageNet normalization

    train_transform = []
    test_transform = []
    
    if not cfg.DATASET.AUGMENTATION.TT_BENCHMARK:
        train_transform += [
            transforms.Resize(cfg.DATASET.IMG_SIZE),
            transforms.RandomCrop(cfg.DATASET.IMG_SIZE, padding=cfg.DATASET.IMG_SIZE//8),
            transforms.RandomHorizontalFlip(),
            ]
    
    if cfg.DATASET.AUGMENTATION.AUTOAUGMENT:
        train_transform.append(ImageNetPolicy())
        
    if not cfg.DATASET.AUGMENTATION.AUGMIX and not cfg.DATASET.AUGMENTATION.TT_BENCHMARK:
        train_transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

    if cfg.DATASET.AUGMENTATION.TT_BENCHMARK:
        train_transform += [transform for transform in create_transform(
            input_size=(cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE),
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
    
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)
    
    return train_transform, test_transform
