import os
import numpy as np
from torchvision import datasets, transforms
from timm.data import create_transform

from robust_minisets import CIFAR10C, CIFAR100C, CIFAR10_1
from .augmentations.autoaugment import CIFAR10Policy
from .utils import get_data_folder


def get_cifar10_dataset(cfg):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    base_folder = get_data_folder()
    train_transform, test_transform = get_cifar_transform(cfg, mean, std)
    train_set = datasets.CIFAR10(
        root=base_folder, download=True, train=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root=base_folder, download=True, train=False, transform=test_transform
    )
    num_data = len(train_set)
    
    return train_set, test_set, num_data


def get_cifar101_dataset(cfg):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    base_folder = get_data_folder()
    _, test_transform = get_cifar_transform(cfg, mean, std)

    test_set = CIFAR10_1(
            split='test', transform=test_transform, download=True, root=base_folder
        )
    
    num_data = len(test_set)
    return test_set, num_data


def get_cifar10_c_dataset(cfg):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    base_folder = get_data_folder()
    _, test_transform = get_cifar_transform(cfg, mean, std)

    test_set = CIFAR10C(
        split='test', transform=test_transform, download=True, root=base_folder
    )
    num_data = len(test_set)
    
    return _, test_set, num_data


def get_cifar100_dataset(cfg):
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    base_folder = get_data_folder()
    train_transform, test_transform = get_cifar_transform(cfg, mean, std)
    train_set = datasets.CIFAR100(
        root=base_folder, download=True, train=True, transform=train_transform
    )
    test_set = datasets.CIFAR100(
        root=base_folder, download=True, train=False, transform=test_transform
    )
    num_data = len(train_set)
    
    return train_set, test_set, num_data


def get_cifar100_c_dataset(cfg):
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    base_folder = get_data_folder()
    _, test_transform = get_cifar_transform(cfg, mean, std)

    test_set = CIFAR100C(
        split='test', transform=test_transform, download=True, root=base_folder
    )
    num_data = len(test_set)
    
    return _, test_set, num_data


def get_cifar_transform(cfg, mean, std):
    if cfg.DATASET.IN_NORMALIZATION:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # ImageNet normalization
    else:
        mean, std = mean, std #(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761) # CIFAR100 normalization

    train_transform = []
    test_transform = []

    if not cfg.DATASET.AUGMENTATION.TT_BENCHMARK:
        train_transform += [
            transforms.Resize(cfg.DATASET.IMG_SIZE),
            transforms.RandomCrop(cfg.DATASET.IMG_SIZE, padding=cfg.DATASET.IMG_SIZE//8),
            transforms.RandomHorizontalFlip(),
            ]

    if cfg.DATASET.AUGMENTATION.AUTOAUGMENT:
        train_transform.append(CIFAR10Policy())

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
