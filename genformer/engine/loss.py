import torch
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy

from .utils import log_msg

class NLLLoss():
    """Custom negative log likelihood loss with Label Smoothing"""
    def __init__(self, label_smoothing=0.0):
        self.label_smoothing = label_smoothing
    
    def __call__(self, probs, target):
        # probs.shape[-1] refers to number of classes
        target = F.one_hot(target, num_classes=probs.shape[-1]).float()
        target = (1 - self.label_smoothing) * target + self.label_smoothing / probs.shape[-1]
        loss = -(target * probs.log()).sum(dim=1)
        return loss.mean()

class CELoss():
    """Custom cross entropy loss"""
    def __init__(self, cfg):
        self.loss_fcn = torch.nn.CrossEntropyLoss(label_smoothing=cfg.LOSS.LABEL_SMOOTHING)
            
    def __call__(self, logits, target, **kwargs):
        loss = self.loss_fcn(logits, target)
        return logits, loss
    
class CMLoss():
    """Custom CutMix/MixUp Cross Entropy loss"""
    def __init__(self, cfg):
        self.loss_fcn = SoftTargetCrossEntropy()
        
    def __call__(self, logits, target, **kwargs):
        loss = self.loss_fcn(logits, target)
        return logits, loss

class ACLoss():
    """Custom AugMix consistency loss"""
    def __init__(self, cfg):
        self.loss_fcn = CELoss(cfg)
        
    def __call__(self, logits, target, **kwargs):
        logits_clean, logits_aug1, logits_aug2 = torch.split(
                logits, int(logits.shape[0] / 3))
        
        # Cross-entropy is only computed on clean images
        loss = self.loss_fcn(logits_clean, target)[1]
            
        p_clean, p_aug1, p_aug2 = (
            F.softmax(logits_clean, dim=1),
            F.softmax(logits_aug1, dim=1),
            F.softmax(logits_aug2, dim=1),
        )
                    
        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        loss += 12 * (F.kl_div(p_mixture, p_clean, reduction="batchmean") +
                      F.kl_div(p_mixture, p_aug1, reduction="batchmean") +
                      F.kl_div(p_mixture, p_aug2, reduction="batchmean")) / 3
        
        logits = torch.stack((logits_clean, logits_aug1, logits_aug2))
        logits = torch.mean(logits, dim=0)
        return logits, loss

class Loss():
    """Custom loss class dependent on config file"""
    def __init__(self, cfg):
        self.loss_fcn = self.init_loss_fcn(cfg)
    
    def init_loss_fcn(self, cfg):
        if cfg.DATASET.AUGMENTATION.AUGMIX and cfg.AUGMIX.JSD:
            return ACLoss(cfg)
        elif cfg.DATASET.AUGMENTATION.CUTMIX or cfg.DATASET.AUGMENTATION.MIXUP:
            return CMLoss(cfg)
        else:
            return CELoss(cfg)
    
    def __call__(self, logits, target, **kwargs):
        return self.loss_fcn(logits, target, **kwargs)
