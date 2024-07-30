import os
from yacs.config import CfgNode as CN
from genformer.engine.utils import log_msg


# Globale config object (example usage from engine.cfg import CFG as cfg)
CFG = CN()

# -------------------------------- Basic project settings -------------------------------- #

# Seed
CFG.SEED = CN()
CFG.SEED.DETERMINISTIC = False
CFG.SEED.SEED = 42

# WandB-Experiments
CFG.EXPERIMENT = CN()
CFG.EXPERIMENT.NAME = ""    
CFG.EXPERIMENT.PROJECT = ""
CFG.EXPERIMENT.TAG = "default"

# ------------------------------------ Dataset options ----------------------------------- #

# Dataset
CFG.DATASET = CN()
CFG.DATASET.IMG_SIZE = 32
CFG.DATASET.TYPE = CN()
CFG.DATASET.TYPE.TRAIN = "cifar100" # Train and validation dataset
CFG.DATASET.TYPE.TEST = "cifar100-c" # Test dataset for evaluation (eval.py)
CFG.DATASET.SYNTHETIC_DATA = False # Training with synthetic images
CFG.DATASET.TRAIN_SUBSET = None # Path to subset of training dataset
CFG.DATASET.TEST_SUBSET = [] # List of paths to subsets of test dataset
CFG.DATASET.IN_NORMALIZATION = True # Normalization mean and std of ImageNet if True, else dataset specific
CFG.DATASET.BATCH_SIZE = CN()
CFG.DATASET.BATCH_SIZE.TRAIN = 64
CFG.DATASET.BATCH_SIZE.TEST = 64
CFG.DATASET.NUM_WORKERS = 12

# Augmentation settings (Cropping and  Flipping is implemented by default)
CFG.DATASET.AUGMENTATION = CN()
CFG.DATASET.AUGMENTATION.AUGMIX = False
CFG.DATASET.AUGMENTATION.AUTOAUGMENT = False
CFG.DATASET.AUGMENTATION.CUTMIX = False
CFG.DATASET.AUGMENTATION.MIXUP = False
CFG.DATASET.AUGMENTATION.TT_BENCHMARK = False # Training with composition of ColorJitter, RandAugment, Random Erasing (adapted from https://github.com/lkhl/tiny-transformers)
CFG.DATASET.AUGMENTATION.SCHEDULER = None # AugMix Severity Scheduler select from {"linear", "cosine", "linear_cosine", "cycle", "cosine_annealing"}

# Dataset+ settings
CFG.SYNTHETIC_DATA = CN()
CFG.SYNTHETIC_DATA.DIR_NAME = None
CFG.SYNTHETIC_DATA.PERCENT = 1.0 # Percentage of synthetic images drawn from CFG.SYNTHETIC_DATA.DIR_NAME
CFG.SYNTHETIC_DATA.TRAINING = "mixed" # Training Type select from {"mixed", "pretraining"}; mixed: combination of real and fake, pretraining: only fake

# ----------------------------------- Training options ----------------------------------- #

# Solver
CFG.SOLVER = CN()
CFG.SOLVER.TYPE = "SGD" # Optimizer select from {"SGD", "ADAM"}
CFG.SOLVER.TRAINER = "base" # Trainer select from {"base"}
CFG.SOLVER.EPOCHS = 240
CFG.SOLVER.LR_TYPE = "" # LR scheduler select from {"multistep", "cosine", "warmup_cosine"}
CFG.SOLVER.BASE_LR = 0.05

# Optimizer
CFG.SGD = CN()
CFG.SGD.MOMENTUM = 0.9
CFG.SGD.NESTEROV = False
CFG.SGD.WEIGHT_DECAY = 0.0001

CFG.ADAM = CN()
CFG.ADAM.BETAS = [0.9, 0.999]
CFG.ADAM.WEIGHT_DECAY = 0.0001

CFG.ADAMW = CN()
CFG.ADAMW.BETAS = [0.9, 0.999]
CFG.ADAMW.WEIGHT_DECAY = 0.05


# LR
CFG.MULTISTEP = CN()
CFG.MULTISTEP.DECAY_STAGES = None
CFG.MULTISTEP.DECAY_RATE = None

CFG.COSINE = CN()
CFG.COSINE.MIN_LR = 5e-5

CFG.WARMUP_COSINE = CN()
CFG.WARMUP_COSINE.START_LR = 0.00004
CFG.WARMUP_COSINE.MIN_LR = 0.00001
CFG.WARMUP_COSINE.WARMUP_EPOCHS = 5

# Loss
CFG.LOSS = CN()
CFG.LOSS.LABEL_SMOOTHING = 0.0

# Log
CFG.LOG = CN()
CFG.LOG.SAVE_CHECKPOINT_FREQ = 40 # frequency of permanent checkpoints
CFG.LOG.VAL_MCE_DETAILED = False # only relevant for CIFAR-100 and Tiny ImageNet: corrupted evaluation is performed regardless of clean accuracy
CFG.LOG.PREFIX = "./output" # log directory
CFG.LOG.WANDB = False

# --------------------------------- Model specifications --------------------------------- #
CFG.MODEL = CN()
CFG.MODEL.TYPE = None
CFG.MODEL.PRETRAINED = False # If no weights specified, latest ckpt is taken from download_ckpts/{dataset}_models/{model}
CFG.MODEL.WEIGHTS = None # specified weights are loaded

# General Transformer specifications
CFG.MODEL.TRANSFORMER = CN()
CFG.MODEL.TRANSFORMER.PATCH_SIZE = None
CFG.MODEL.TRANSFORMER.PATCH_STRIDE = None
CFG.MODEL.TRANSFORMER.PATCH_PADDING = None
CFG.MODEL.TRANSFORMER.HIDDEN_DIM = None
CFG.MODEL.TRANSFORMER.DEPTH = None
CFG.MODEL.TRANSFORMER.NUM_HEADS = None
CFG.MODEL.TRANSFORMER.MLP_RATIO = None

CFG.MODEL.TRANSFORMER.LN_EPS = None
CFG.MODEL.TRANSFORMER.DROP_RATE = None
CFG.MODEL.TRANSFORMER.DROP_PATH_RATE = None
CFG.MODEL.TRANSFORMER.ATTENTION_DROP_RATE = None

CFG.MODEL.TRANSFORMER.RETURN_ATTN_SCORES = False

# General CNN specifications
CFG.MODEL.CNN = CN()
CFG.MODEL.CNN.DEPTH = 18

# CONVIT CFG
CFG.MODEL.CONVIT = CN()
CFG.MODEL.CONVIT.LOCAL_LAYERS = 10
CFG.MODEL.CONVIT.LOCALITY_STRENGTH = 1.0

# DEIT CFG
CFG.MODEL.DEIT = CN()
CFG.MODEL.DEIT.ENABLE_LOGIT = False

# PVT CFG
CFG.MODEL.PVT = CN()
CFG.MODEL.PVT.SR_RATIO = [8, 4, 2, 1]

CFG.MODEL.RESNET = CN()
CFG.MODEL.RESNET.TRANS_FUN = "basic_transform"
CFG.MODEL.RESNET.NUM_GROUPS = 1
CFG.MODEL.RESNET.WIDTH_PER_GROUP = 64

# --------------------------------- Augmentation Methods --------------------------------- #

# AugMix CFG
CFG.AUGMIX = CN()
CFG.AUGMIX.MIXTURE_WIDTH = 3
CFG.AUGMIX.MIXTURE_DEPTH = -1
CFG.AUGMIX.MIN_SEVERITY = 0.0
CFG.AUGMIX.MAX_SEVERITY = 3.0
CFG.AUGMIX.ALL_OPS = False
CFG.AUGMIX.JSD = True

# CutMix CFG
CFG.CUTMIX = CN()
CFG.CUTMIX.ALPHA = 1.0

# MixUp CFG
CFG.MIXUP = CN()
CFG.MIXUP.ALPHA = 0.8

# Scheduler CFG
CFG.SCHEDULER = CN()
CFG.SCHEDULER.START_EPOCH = 0 # start epoch of severity scheduling
CFG.SCHEDULER.END_EPOCH = 240 # end epoch of severity scheduling
CFG.SCHEDULER.PERIOD_DURATION = 4 # period duration of scheduling [epochs] (only "cosine", "linear_cosine")
CFG.SCHEDULER.CYCLES = 8 # number of periodical cycles of cycle scheduler (only "cycle")



# Method to show/save config
def save_cfg(cfg, save_path=None, show=False):
    dump_cfg = CN()
    dump_cfg.SEED = cfg.SEED
    dump_cfg.EXPERIMENT = cfg.EXPERIMENT
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.MODEL = cfg.MODEL
    dump_cfg.SOLVER = cfg.SOLVER
    dump_cfg.LOSS = cfg.LOSS
    dump_cfg.LOG = cfg.LOG
    if cfg.DATASET.SYNTHETIC_DATA:
        dump_cfg.update({"SYNTHETIC_DATA": cfg.get("SYNTHETIC_DATA")})
    if cfg.DATASET.AUGMENTATION.AUGMIX:
        dump_cfg.update({"AUGMIX": cfg.get("AUGMIX")})
    if cfg.DATASET.AUGMENTATION.CUTMIX:
        dump_cfg.update({"CUTMIX": cfg.get("CUTMIX")})
    if cfg.DATASET.AUGMENTATION.MIXUP:
        dump_cfg.update({"MIXUP": cfg.get("MIXUP")})
    if cfg.DATASET.AUGMENTATION.SCHEDULER:
        dump_cfg.update({"SCHEDULER": cfg.get("SCHEDULER")})
    if cfg.SOLVER.TYPE.upper() in cfg:
        dump_cfg.update({cfg.SOLVER.TYPE.upper(): cfg.get(cfg.SOLVER.TYPE.upper())})
    if cfg.SOLVER.LR_TYPE.upper() in cfg:
        dump_cfg.update({cfg.SOLVER.LR_TYPE.upper(): cfg.get(cfg.SOLVER.LR_TYPE.upper())})
    if cfg.MODEL.TYPE in cfg:
        dump_cfg.update({cfg.MODEL.TYPE: cfg.get(cfg.MODEL.TYPE)})
        
    if show:
        print(log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO"))
        
    if save_path:
        with open(os.path.join(save_path, "config.yaml"), 'w') as file:
            file.write(dump_cfg.dump())
    else:
        return dump_cfg