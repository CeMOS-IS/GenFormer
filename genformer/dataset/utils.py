import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_folder():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    two_down = os.path.dirname(os.path.dirname(this_dir))
    data_folder = os.path.join(two_down, "data")
    os.makedirs(data_folder, exist_ok=True)
    return data_folder
    
    
def sev_scheduler(cfg, epoch):
    # linear increasing/decreasing severity (partwise constant)
    if cfg.DATASET.AUGMENTATION.SCHEDULER == "linear":
        return (np.clip((epoch - cfg.SCHEDULER.START_EPOCH)/(cfg.SCHEDULER.END_EPOCH - cfg.SCHEDULER.START_EPOCH), 0, 1) *
                (cfg.AUGMIX.MAX_SEVERITY - cfg.AUGMIX.MIN_SEVERITY) + cfg.AUGMIX.MIN_SEVERITY)
        
    # periodically increasing and decreasing severity w/ constant amplitude
    elif cfg.DATASET.AUGMENTATION.SCHEDULER == "cosine":
        return ((-np.cos(2 * np.pi * epoch / cfg.SCHEDULER.PERIOD_DURATION) + 1) *
                ((cfg.AUGMIX.MAX_SEVERITY - cfg.AUGMIX.MIN_SEVERITY) / 2) + cfg.AUGMIX.MIN_SEVERITY)
        
    # periodically increasing and decreasing severity w/ increasing amplitude
    elif cfg.DATASET.AUGMENTATION.SCHEDULER == "linear_cosine":
        return (np.clip((epoch - cfg.SCHEDULER.START_EPOCH)/(cfg.SCHEDULER.END_EPOCH - cfg.SCHEDULER.START_EPOCH), 0, 1) *
                (-np.cos(2 * np.pi * epoch / cfg.SCHEDULER.PERIOD_DURATION) + 1) * 
                ((cfg.AUGMIX.MAX_SEVERITY - cfg.AUGMIX.MIN_SEVERITY) / 2) + cfg.AUGMIX.MIN_SEVERITY)
        
    # linear decreasing severity and alternating cycles of augmentation and no augmentation
    elif cfg.DATASET.AUGMENTATION.SCHEDULER == "cycle":
        return ((epoch % (cfg.SOLVER.EPOCHS / cfg.SCHEDULER.CYCLES) < cfg.SOLVER.EPOCHS/cfg.SCHEDULER.CYCLES/2) *
                np.clip(((cfg.SCHEDULER.END_EPOCH - cfg.SCHEDULER.START_EPOCH) - (epoch - cfg.SCHEDULER.START_EPOCH)) /
                        (cfg.SCHEDULER.END_EPOCH - cfg.SCHEDULER.START_EPOCH), 0, 1) *
                (cfg.AUGMIX.MAX_SEVERITY - cfg.AUGMIX.MIN_SEVERITY) + cfg.AUGMIX.MIN_SEVERITY)
        
    # cosine annealing of severity
    elif cfg.DATASET.AUGMENTATION.SCHEDULER == "cosine_annealing":
        return ((np.cos(np.pi * np.clip(((cfg.SCHEDULER.END_EPOCH - cfg.SCHEDULER.START_EPOCH) - (epoch - cfg.SCHEDULER.START_EPOCH)) /
                        (cfg.SCHEDULER.END_EPOCH - cfg.SCHEDULER.START_EPOCH), 0, 1)) + 1) * 
                ((cfg.AUGMIX.MAX_SEVERITY - cfg.AUGMIX.MIN_SEVERITY) / 2) + cfg.AUGMIX.MIN_SEVERITY)
        
    # constant severity
    else:
        return cfg.AUGMIX.MAX_SEVERITY
    

def get_dataloaders(train_set, test_set, cfg, worker_init_fn, generator):
    train_loader = None
    if not train_set is None:
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.DATASET.BATCH_SIZE.TRAIN,
            shuffle=True,
            pin_memory=True,
            num_workers=cfg.DATASET.NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.DATASET.BATCH_SIZE.TEST,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.DATASET.NUM_WORKERS,
    )
    return train_loader, test_loader
