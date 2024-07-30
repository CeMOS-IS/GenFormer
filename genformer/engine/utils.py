import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import time
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, model, cfg, k):
    batch_time, losses, top1, topk = [AverageMeter() for _ in range(4)]
    ce_loss = nn.CrossEntropyLoss()
    nll_loss = nn.NLLLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.squeeze().cuda(non_blocking=True)
            output = model(image)
            if len(output.shape) > 2:
                # averaged logits, CE loss
                output = output.mean(dim=1)
                loss = ce_loss(output, target)
            else:
                loss = ce_loss(output, target)
            acc1, acck = accuracy(output, target, _topk=(1, k))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            topk.update(acck[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-{k:d}:{topk.avg:.3f}".format(
                top1=top1, k=k, topk=topk
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    return top1.avg, topk.avg, losses.avg



def validate_mCE(test_set_clean, test_set_cor, model, distortions, cfg, ckpt):
    model_path = os.path.dirname(ckpt)
    dataset = cfg.DATASET.TYPE.TEST
    baseline_model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        f"../../references/{dataset.split('-')[0]}_models/alexnet/results.txt")
    
    # set dataloader deterministic
    g = torch.Generator()
    if cfg.SEED.DETERMINISTIC:
        g.manual_seed(cfg.SEED.SEED)

    with open(baseline_model_path) as file:
        CE_baseline = json.load(file)
    CE_log = {}
    
    n_dist = len(distortions)
    model.eval()
    
    test_loader = DataLoader(test_set_clean,
                             batch_size=cfg.DATASET.BATCH_SIZE.TEST,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=cfg.DATASET.NUM_WORKERS,
                             worker_init_fn=seed_worker,
                             generator=g,
                             )
    len_dataset, counter = len(test_set_clean), 0

    CE_log["CE_clean"] = error_rate(model, test_loader, "clean",n_dist=n_dist, cfg=cfg)
    mCE_abs = AverageMeter()
    mCE_rel = AverageMeter()
    relative_mCE = AverageMeter()

    for idx, distortion in enumerate(distortions):
        CE_abs = AverageMeter()
        CE_rel = AverageMeter()
        CE_log[f"CE_{distortion}"] = {}
        for sev_lvl in range(5):
            test_set = Subset(test_set_cor, range(counter,(counter+len_dataset)))
            test_loader = DataLoader(test_set,
                                    batch_size=cfg.DATASET.BATCH_SIZE.TEST,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=cfg.DATASET.NUM_WORKERS,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    )
            CE_abs.update(error_rate(model, test_loader, distortion, sev_lvl+1, idx+1, n_dist=n_dist, cfg=cfg))
            CE_log[f"CE_{distortion}"][f"severity {sev_lvl+1}"] = CE_abs.val
            counter += len_dataset
        CE_rel.update(CE_abs.sum / CE_baseline[f"CE_{distortion}"]["sum"] * 100.0)
        
        CE_log[f"CE_{distortion}"]["sum"] = CE_abs.sum
        CE_log[f"CE_{distortion}"]["abs"] = CE_abs.avg
        CE_log[f"CE_{distortion}"]["rel"] = CE_rel.avg
        CE_log[f"CE_{distortion}"]["Relative CE"] = (
            (CE_abs.sum - CE_log["CE_clean"]) / (CE_baseline[f"CE_{distortion}"]["sum"] - CE_baseline["CE_clean"]) * 100.0
            )
        
        mCE_abs.update(CE_abs.avg)
        mCE_rel.update(CE_rel.avg)
        relative_mCE.update(CE_log[f"CE_{distortion}"]["Relative CE"])
    
    CE_log["CE_clean"] = CE_log.pop("CE_clean")
    CE_log["mCE"] = {}
    CE_log["mCE"]["abs"] = mCE_abs.avg
    CE_log["mCE"]["rel"] = mCE_rel.avg
    
    CE_log["Relative mCE"] = relative_mCE.avg

    for k_out, v_out in CE_log.items():
        if isinstance(v_out, dict):
            msg = f"{k_out}:"
            for k_in, v_in in v_out.items():
                msg = msg + f"\n    {k_in} : {v_in:.3f}"
        else:
            msg = f"{k_out} : {v_out:.3f}"
        print(log_msg(msg, "RESULT"))

    with open(os.path.join(model_path, "results.txt"), 'w') as file:
        file.write(json.dumps(CE_log, indent = 4))

    return CE_log

def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 37,
        "RESULT": 33,
        "WARNING": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return cfg.SOLVER.LR


def cosine_lr(step, total_steps, lr_max, lr_min):
    # Compute learning rate according to cosine annealing schedule
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def accuracy(output, target, _topk=(1,)):
    with torch.no_grad():
        maxk = max(_topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in _topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    
def error_rate(model, dataloader, distortion_name=None, k=0, idx=0, n_dist=15, cfg=None):
    num_iter = len(dataloader)
    pbar = tqdm(range(num_iter))
    # Calculate error
    correct = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, (image, target) in enumerate(dataloader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(image)
            if len(output.shape) > 2:
                # averaged logits, CE loss
                output = output.mean(dim=1)

            acc1 = accuracy(output, target)
            
            batch_size = target.size(0)
            correct.update(acc1[0].item(), batch_size)
            msg = f"[{idx}/{n_dist}] Distortion: {distortion_name}, Severity Level: {k} | CE: {100 - correct.avg:.3f}"
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    
    return 100 - correct.avg
            

def set_deterministic(cfg):
    deterministic = cfg.SEED.DETERMINISTIC
    seed = cfg.SEED.SEED
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        msg = f"Mode set to deterministic with seed set as {seed}."
        print(log_msg(msg, "INFO"))
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        msg = f"Mode set to non-deterministic with seed set as {seed}."
        print(log_msg(msg, "INFO"))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")
