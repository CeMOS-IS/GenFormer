import torch.nn as nn
from abc import ABCMeta, abstractmethod


class BaseTransformerModel(nn.Module, metaclass=ABCMeta):
    """
    Base class for Transformer models.
    Attributes:
        - self.features (List[Tensor]): the features in each block.
        - self.feature_dims (List[int]): the dimension of features in each block.
        - self.distill_logits (Tensor|None): the logits of the distillation token, only for DeiT.
    """

    def __init__(self, cfg):
        super().__init__()
        # Base configs for Transformers
        self.img_size = cfg.DATASET.IMG_SIZE
        
        self.patch_size = cfg.MODEL.TRANSFORMER.PATCH_SIZE
        self.patch_stride = cfg.MODEL.TRANSFORMER.PATCH_STRIDE
        self.patch_padding = cfg.MODEL.TRANSFORMER.PATCH_PADDING
        self.hidden_dim = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
        self.depth = cfg.MODEL.TRANSFORMER.DEPTH
        self.num_heads = cfg.MODEL.TRANSFORMER.NUM_HEADS
        self.mlp_ratio = cfg.MODEL.TRANSFORMER.MLP_RATIO
        self.ln_eps = cfg.MODEL.TRANSFORMER.LN_EPS
        self.drop_rate = cfg.MODEL.TRANSFORMER.DROP_RATE
        self.drop_path_rate = cfg.MODEL.TRANSFORMER.DROP_PATH_RATE
        self.attn_drop_rate = cfg.MODEL.TRANSFORMER.ATTENTION_DROP_RATE

        # Calculate the dimension of features in each block
        if isinstance(self.hidden_dim, int):
            assert isinstance(self.depth, int)
            self.feature_dims = [self.hidden_dim] * self.depth
        elif isinstance(self.hidden_dim, (list, tuple)):
            assert isinstance(self.depth, (list, tuple))
            assert len(self.hidden_dim) == len(self.depth)
            self.feature_dims = sum([[self.hidden_dim[i]] * d for i, d in enumerate(self.depth)], [])
        else:
            raise ValueError
        self.features = list()
        self.distill_logits = None

    def complexity(self):
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'params': f'{round(params/1e6, 2)}M'}


class BaseConvModel(nn.Module, metaclass=ABCMeta):
    """
    Base class for conv models.

    Attributes:
        - self.features (List[Tensor]): the features in each stage.
        - self.feature_dims (List[int]): the dimension of features in each stage.
    """

    def __init__(self, cfg):
        super(BaseConvModel, self).__init__()
        self.depth = cfg.MODEL.CNN.DEPTH
        self.img_size = cfg.DATASET.IMG_SIZE
        self.features = list()
        self.feature_dims = None

    def complexity(self):
        params = sum(p.numel() for p in self.parameters())
        return {'params': f'{round(params/1e6, 2)}M'}
