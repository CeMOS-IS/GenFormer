import os
import argparse
import json
import time
import torch

from genformer.models import cifar_model_dict, tinyimagenet_model_dict
from genformer.dataset import get_dataset
from genformer.engine.utils import log_msg, set_deterministic, load_checkpoint
from genformer.engine.cfg import CFG as cfg
from genformer.engine.cfg import save_cfg
from genformer.engine import trainer_dict


def main(resume=False, opts=None):
    # Set seed and cuda settings according to cfg
    set_deterministic(cfg)
    
    # Prepare namings and log paths
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.MODEL.TYPE
    tags = cfg.EXPERIMENT.TAG.split(",")

    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    log_path = os.path.join(cfg.LOG.PREFIX, experiment_name, time.strftime("%Y%m%d-%H%M%S"))

    if not os.path.exists(log_path):
            os.makedirs(log_path)
            
    # Initialize WANDB loggers
    if cfg.LOG.WANDB:
        try:
            import wandb
            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags, config=save_cfg(cfg), dir=log_path)      
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False
            
    # cfg & loggers
    cfg.freeze()
    save_cfg(cfg, show=True)
    
    # init dataloader & models
    train_loader, val_loader, num_data, in_channels, num_classes = get_dataset(cfg)

    # vanilla
    if any(dataset in cfg.DATASET.TYPE.TRAIN for dataset in ["tinyimagenet"]):
        model = tinyimagenet_model_dict[cfg.MODEL.TYPE](
            pretrained=cfg.MODEL.PRETRAINED,
            in_channels=in_channels,
            num_classes=num_classes,
            cfg=cfg,)
    elif any(dataset in cfg.DATASET.TYPE.TRAIN for dataset in ["cifar10", "cifar100", "mnist"]):
        model = cifar_model_dict[cfg.MODEL.TYPE](
            pretrained=cfg.MODEL.PRETRAINED,
            in_channels=in_channels,
            num_classes=num_classes,
            cfg=cfg,)
    else:
        raise NotImplementedError(f"{cfg.MODEL.TYPE} is not implemeted for {cfg.DATASET.TYPE.TRAIN}.")
    
    if cfg.MODEL.WEIGHTS:
        try:
            model.load_state_dict(load_checkpoint(cfg.MODEL.WEIGHTS)["model"])
        except:
            raise Exception(f"No suitable pretrained model: {model.__class__.__name__}")


    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        model, train_loader, val_loader, cfg, log_path, num_classes
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for image classification.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    main(args.resume, args.opts)