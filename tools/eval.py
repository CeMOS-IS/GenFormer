import argparse
import torch

from genformer.models import cifar_model_dict, tinyimagenet_model_dict

from genformer.dataset import get_dataset
from genformer.engine.utils import load_checkpoint, validate, validate_mCE, set_deterministic
from genformer.engine.cfg import CFG as cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="pretrain")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cifar100",
        choices=[
            "cifar10","cifar10.1", "cifar100", "cifar10-c", "cifar100-c", 
            "tinyimagenet", "tinyimagenet-c", "tinyimagenet-v2", "tinyimagenet-a", "tinyimagenet-r",
            "eurosat", "eurosat-c",
            "pathmnist", "dermamnist", "octmnist", "pneumoniamnist", "breastmnist", "bloodmnist", 
            "tissuemnist", "organamnist", "organcmnist", "organsmnist",
            "pathmnist-c", "dermamnist-c", "octmnist-c", "pneumoniamnist-c", "breastmnist-c", "bloodmnist-c", 
            "tissuemnist-c", "organamnist-c", "organcmnist-c", "organsmnist-c"
            ],
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=128)
    parser.add_argument("-nd", "--non-deterministic", default=True, action="store_false")
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    
    cfg.DATASET.TYPE.TEST = args.dataset
    cfg.DATASET.BATCH_SIZE.TEST = args.batch_size
    cfg.SEED.DETERMINISTIC = args.non_deterministic
    ckpt = args.ckpt
    
    set_deterministic(cfg)
        
    # init dataloader & models
    if "-c" in args.dataset:
        test_set_clean, test_set_cor, in_channels, num_classes = get_dataset(cfg, test=True)
    else:
        _, test_loader, _, in_channels, num_classes = get_dataset(cfg, test=True)        


    # choose Model
    if any(dataset in cfg.DATASET.TYPE.TEST for dataset in ["tinyimagenet", "tinyimagenet-c"]):
        model = tinyimagenet_model_dict[cfg.MODEL.TYPE](
            pretrained=True if ckpt == "pretrain" else False,
            in_channels=in_channels,
            num_classes=num_classes,
            cfg=cfg,)
        
    elif any(dataset in cfg.DATASET.TYPE.TEST for dataset in ["cifar10", "cifar10.1", "cifar10-c", "cifar100", "cifar100-c", "mnist"]):
        model = cifar_model_dict[cfg.MODEL.TYPE](
            pretrained=True if ckpt == "pretrain" else False,
            in_channels=in_channels,
            num_classes=num_classes,
            cfg=cfg,)
        
    else:
        raise NotImplementedError(f"{cfg.MODEL.TYPE} is not implemeted for {cfg.DATASET.TYPE.TEST}.")
    
    # load ckpt
    if args.ckpt != "pretrain":
        try:
            model.load_state_dict(load_checkpoint(args.ckpt)["model"])
        except:
            checkpoint = load_checkpoint(args.ckpt)
            test_err = checkpoint["test_err"] if "test_err" in checkpoint else 100
            ema_err = checkpoint["ema_err"] if "ema_err" in checkpoint else 100
            ema_state = "ema_state" if "ema_state" in checkpoint else "model_state"
            best_state = "model_state" if test_err <= ema_err else ema_state
            model.load_state_dict(checkpoint[best_state])
            
        
    
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    

    # evaluate
    if "-c" in args.dataset:
        validate_mCE(test_set_clean, test_set_cor, model, test_set_cor.info['corruption_dict']['test'], cfg, ckpt)
    else:
        k = 5 if num_classes > 5 else 1 
        top1, _, _ = validate(test_loader, model, cfg, k)
                
