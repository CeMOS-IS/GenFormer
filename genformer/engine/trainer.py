import os
import time
import copy
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from timm.data import Mixup
from collections import OrderedDict

from .loss import Loss
from .utils import (
    AverageMeter,
    accuracy,
    validate,
    validate_mCE,
    save_checkpoint,
    load_checkpoint,
    log_msg,
)
from .cfg import save_cfg
from genformer.dataset import get_dataset
from genformer.dataset.utils import sev_scheduler


class BaseTrainer(object):
    def __init__(self, model, train_loader, val_loader, cfg, log_path, num_classes):
        self.cfg = cfg
        self.model = model
        self.val_model = copy.deepcopy(model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_lr_scheduler()
        self.criterion = Loss(self.cfg)
        

        # init augmentation strategies
        self.cfg.defrost()
        if not self.cfg.DATASET.AUGMENTATION.CUTMIX:
            self.cfg.CUTMIX.ALPHA = 0.0
        if not self.cfg.DATASET.AUGMENTATION.MIXUP:
            self.cfg.MIXUP.ALPHA = 0.0
        self.cfg.freeze()
        self.cutmix_mixup = Mixup(
                mixup_alpha=self.cfg.MIXUP.ALPHA,
                cutmix_alpha=self.cfg.CUTMIX.ALPHA,
                label_smoothing=self.cfg.LOSS.LABEL_SMOOTHING,
                num_classes=num_classes,
        )
        
        # helper variables
        self.best_acc = -1
        self.better_acc = False
        self.k = 5 if num_classes > 5 else 1        

        
        # init loggers
        self.log_path = log_path
        save_cfg(cfg, self.log_path)
        if self.cfg.LOG.WANDB:
            global wandb
            import wandb



    def init_optimizer(self):
        if self.cfg.SOLVER.TYPE.upper() == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.cfg.SOLVER.BASE_LR,
                momentum=self.cfg.SGD.MOMENTUM,
                weight_decay=self.cfg.SGD.WEIGHT_DECAY,
                nesterov=self.cfg.SGD.NESTEROV,)
        elif self.cfg.SOLVER.TYPE.upper() == "ADAM":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.cfg.SOLVER.BASE_LR,
                betas=self.cfg.ADAM.BETAS,
                weight_decay=self.cfg.ADAM.WEIGHT_DECAY,)
        elif self.cfg.SOLVER.TYPE.upper() == "ADAMW":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.SOLVER.BASE_LR,
                betas=self.cfg.ADAMW.BETAS,
                weight_decay=self.cfg.ADAMW.WEIGHT_DECAY,)
        else:
            raise NotImplementedError(self.cfg.SOLVER.TYPE)
        return optimizer


    
    def init_lr_scheduler(self):
        if self.cfg.SOLVER.LR_TYPE == "multistep":
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[stage * len(self.train_loader) for stage in self.cfg.MULTISTEP.DECAY_STAGES],
                gamma=self.cfg.MULTISTEP.DECAY_RATE,)
        elif self.cfg.SOLVER.LR_TYPE == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.SOLVER.EPOCHS * len(self.train_loader),
                eta_min=self.cfg.COSINE.MIN_LR,)
        elif self.cfg.SOLVER.LR_TYPE == "warmup_cosine":
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.cfg.SOLVER.BASE_LR,
                total_steps=self.cfg.SOLVER.EPOCHS * len(self.train_loader),
                pct_start=self.cfg.WARMUP_COSINE.WARMUP_EPOCHS / self.cfg.SOLVER.EPOCHS,
                div_factor=self.cfg.SOLVER.BASE_LR / self.cfg.WARMUP_COSINE.START_LR,
                final_div_factor=self.cfg.WARMUP_COSINE.START_LR / self.cfg.WARMUP_COSINE.MIN_LR,)
        else:
            raise NotImplementedError(self.cfg.SOLVER.LR_TYPE)

    


    def log(self, epoch, log_dict):
        # wandb log
        if self.cfg.LOG.WANDB:
            wandb.log(log_dict)
        if log_dict["test_acc"] > self.best_acc:
            self.best_acc = log_dict["test_acc"]
            if self.cfg.LOG.WANDB:
                wandb.run.summary["best_acc"] = self.best_acc
        # worklog.txt
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            lines = [
                "-" * 25 + os.linesep,
                "epoch: {}".format(epoch) + os.linesep,
                "lr: {:.5f}".format(float(log_dict["learning_rate"])) + os.linesep,
            ]
            for k, v in log_dict.items():
                lines.append("{}: {:.2f}".format(k, v) + os.linesep)
            lines.append("-" * 25 + os.linesep)
            writer.writelines(lines)

    

    def train(self, resume=False):
        epoch = 1
        self.model = self.model.cuda()
        if not resume is None:
            state = load_checkpoint(os.path.join(resume, "latest"))
            epoch = state["epoch"] + 1
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1

        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))


    def train_epoch(self, epoch):
        lr = self.scheduler.get_last_lr()[0]
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            f"top{self.k}": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter))

        # train loops
        self.model.train()
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
        pbar.close()

        # validate
        test_acc, test_acc_topk, test_loss = validate(self.val_loader, self.model, self.cfg, self.k)
        
        # saving checkpoint
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        save_checkpoint(state, os.path.join(self.log_path, "latest"))
    
        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(
                state, os.path.join(self.log_path, "epoch_{}".format(epoch))
            )

        # update the best
        if test_acc >= self.best_acc:
            self.better_acc = True
            save_checkpoint(state, os.path.join(self.log_path, "best"))


        # update augmentation severity according to schedule
        if self.cfg.DATASET.AUGMENTATION.AUGMIX:
            self.train_loader.dataset.severity_student = sev_scheduler(self.cfg, epoch)

        # log
        log_dict = OrderedDict(
            {
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "test_acc": test_acc,
                f"test_acc_top{self.k}": test_acc_topk,
                "best_acc": test_acc if test_acc > self.best_acc else self.best_acc,
                "test_loss": test_loss,
                "epoch": epoch,
                "learning_rate": lr,
                "augmix_severity": 0 if self.cfg.DATASET.AUGMENTATION.AUGMIX != True else self.train_loader.dataset.severity_student,
            }
        )
        self.log(epoch, log_dict)

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad(set_to_none=True)
        train_start_time = time.time()
        image, target = data
        target = target.squeeze()
        
        # CutMix / MixUp option
        if self.cfg.DATASET.AUGMENTATION.CUTMIX or self.cfg.DATASET.AUGMENTATION.MIXUP:
            image, target_cm = self.cutmix_mixup(image, target)
        else:
            target_cm = target
            
        image = image.float() if not isinstance(image, list) else [_.float() for _ in image]
        image = image.cuda(non_blocking=True) if not (self.cfg.DATASET.AUGMENTATION.AUGMIX and self.cfg.AUGMIX.JSD) else torch.cat(image, 0).cuda()  
                    
        target = target.cuda(non_blocking=True)
        target_cm = target_cm.cuda(non_blocking=True)
        train_meters["data_time"].update(time.time() - train_start_time)
        
        logits = self.model(image)
        logits, loss = self.criterion(logits, target_cm)

        # backward
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        
        # collect info
        batch_size = image.size(0) if not isinstance(image, list) else image[0].size(0)
        acc1, acck = accuracy(logits, target, _topk=(1, self.k))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters[f"top{self.k}"].update(acck[0], batch_size)
        
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-{:d}:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            self.k,
            train_meters[f"top{self.k}"].avg,
        )
        return msg
