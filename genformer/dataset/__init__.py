import torch

from .cifar import *
from .tinyimagenet import *
from .medmnist import get_medmnist_dataset, get_medmnist_c_dataset
from .augmentations.augmix import get_augmix_dataset
from .datasetplus import get_datasetplus_dataset
from .utils import seed_worker, get_dataloaders

def get_dataset(cfg, test=False):
    # set dataloader deterministic
    g = torch.Generator()
    if cfg.SEED.DETERMINISTIC:
        g.manual_seed(cfg.SEED.SEED)
    
    if test:
        dataset = cfg.DATASET.TYPE.TEST
    else:
        dataset = cfg.DATASET.TYPE.TRAIN

    
    if dataset == "tinyimagenet":
        train_set, test_set, num_data = get_tinyimagenet_dataset(cfg)
        if cfg.DATASET.SYNTHETIC_DATA:
            train_set, num_data = get_datasetplus_dataset(train_set, cfg)
        if cfg.DATASET.AUGMENTATION.AUGMIX:
            train_set = get_augmix_dataset(train_set, cfg)
        train_loader, val_loader = get_dataloaders(
            train_set=train_set,
            test_set=test_set,
            cfg=cfg,
            worker_init_fn=seed_worker,
            generator=g,
        )
        in_channels = 3
        num_classes = 200

    
    elif dataset in ["tinyimagenet-v2", "tinyimagenet-a", "tinyimagenet-r"]:
        test_set, num_data = get_tinyimagenet_test_dataset(cfg)
        train_loader, val_loader = get_dataloaders(
            train_set=None,
            test_set=test_set,
            cfg=cfg,
            worker_init_fn=seed_worker,
            generator=g,
        )
        in_channels = 3
        num_classes = 200    

    elif dataset == "tinyimagenet-c":
        _, test_set_clean, _ = get_tinyimagenet_dataset(cfg)
        _, test_set_cor, _ = get_tinyimagenet_c_dataset(cfg)
        in_channels = 3
        num_classes = 200
        return test_set_clean, test_set_cor, in_channels, num_classes
    





    elif dataset == "cifar10":
        train_set, test_set, num_data = get_cifar10_dataset(cfg)
        if cfg.DATASET.SYNTHETIC_DATA:
            train_set, num_data = get_datasetplus_dataset(train_set, cfg)
        if cfg.DATASET.AUGMENTATION.AUGMIX:
            train_set = get_augmix_dataset(train_set, cfg)
        train_loader, val_loader = get_dataloaders(
            train_set=train_set,
            test_set=test_set,
            cfg=cfg,
            worker_init_fn=seed_worker,
            generator=g,
        )
        in_channels = 3
        num_classes = 10
    
    elif dataset == "cifar10.1":
        test_set, num_data = get_cifar101_dataset(cfg)
        train_loader, val_loader = get_dataloaders(
            train_set=None,
            test_set=test_set,
            cfg=cfg,
            worker_init_fn=seed_worker,
            generator=g,
        )
        in_channels = 3
        num_classes = 10   

    elif dataset == "cifar10-c":
        _, test_set_clean, _ = get_cifar10_dataset(cfg)
        _, test_set_cor, _ = get_cifar10_c_dataset(cfg)
        in_channels = 3
        num_classes = 10
        return test_set_clean, test_set_cor, in_channels, num_classes


    elif dataset == "cifar100":
        train_set, test_set, num_data = get_cifar100_dataset(cfg)
        if cfg.DATASET.SYNTHETIC_DATA:
            train_set, num_data = get_datasetplus_dataset(train_set, cfg)
        if cfg.DATASET.AUGMENTATION.AUGMIX:
            train_set = get_augmix_dataset(train_set, cfg)
        train_loader, val_loader = get_dataloaders(
            train_set=train_set,
            test_set=test_set,
            cfg=cfg,
            worker_init_fn=seed_worker,
            generator=g,
        )
        in_channels = 3
        num_classes = 100

    elif dataset == "cifar100-c":
        _, test_set_clean, _ = get_cifar100_dataset(cfg)
        _, test_set_cor, _ = get_cifar100_c_dataset(cfg)
        in_channels = 3
        num_classes = 100
        return test_set_clean, test_set_cor, in_channels, num_classes
    



    
    elif dataset in ["pathmnist", "dermamnist", "octmnist", "pneumoniamnist", 
                     "breastmnist", "bloodmnist", "tissuemnist", "organamnist", 
                     "organcmnist", "organsmnist"]:
        train_set, test_set, num_data, num_classes, in_channels = get_medmnist_dataset(cfg)
        if cfg.DATASET.SYNTHETIC_DATA:
            train_set, num_data = get_datasetplus_dataset(train_set, cfg, num_classes, in_channels)
        if cfg.DATASET.AUGMENTATION.AUGMIX:
            train_set = get_augmix_dataset(train_set, cfg)
        train_loader, val_loader = get_dataloaders(
            train_set=train_set,
            test_set=test_set,
            cfg=cfg,
            worker_init_fn=seed_worker,
            generator=g,
        )
        in_channels = 3
        num_classes = num_classes

    elif dataset in ["pathmnist-c", "dermamnist-c", "octmnist-c", "pneumoniamnist-c", 
                     "breastmnist-c", "bloodmnist-c", "tissuemnist-c", "organamnist-c", 
                     "organcmnist-c", "organsmnist-c"]:
        _, test_set_clean, num_data, num_classes, in_channels = get_medmnist_dataset(cfg)
        _, test_set_cor, _ = get_medmnist_c_dataset(cfg)
        in_channels = 3
        num_classes = num_classes
        return test_set_clean, test_set_cor, in_channels, num_classes
    
    else:
        raise NotImplementedError(dataset)

    return train_loader, val_loader, num_data, in_channels, num_classes
