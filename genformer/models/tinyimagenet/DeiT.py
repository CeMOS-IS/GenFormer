# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

"""
Modified from the official implementation of DeiT.
https://github.com/facebookresearch/deit/blob/main/models.py
"""

import os
import torch
import torch.nn as nn
import timm
from timm.models.layers import trunc_normal_

from genformer.models.utils import ckpt_path
from genformer.engine.utils import load_checkpoint
from genformer.models.base import BaseTransformerModel
from genformer.models.common import (
    PatchEmbedding,
    TransformerLayer,
    layernorm,
)


class DeiT(BaseTransformerModel):

    def __init__(self, in_channels:int=None, num_classes:int=None, cfg:dict=None, **kwargs):
        super().__init__(cfg=cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.return_attn_scores = cfg.MODEL.TRANSFORMER.RETURN_ATTN_SCORES


        self.patch_embed = PatchEmbedding(img_size=self.img_size, patch_size=self.patch_size, in_channels=self.in_channels, out_channels=self.hidden_dim)
        self.num_patches = self.patch_embed.num_patches
        self.num_tokens = 1 + cfg.MODEL.DEIT.ENABLE_LOGIT
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, self.hidden_dim))
        self.pe_dropout = nn.Dropout(p=self.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.layers = nn.ModuleList([TransformerLayer(
            in_channels=self.hidden_dim,
            num_heads=self.num_heads,
            qkv_bias=True,
            mlp_ratio=self.mlp_ratio,
            eps=self.ln_eps,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=dpr[i],
            return_attn_scores=self.return_attn_scores) for i in range(self.depth)])


        self.norm = layernorm(self.hidden_dim, self.ln_eps)
        self.apply(self._init_weights)

        self.head = nn.Linear(self.hidden_dim, self.num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.distill_logits = None

        self.distill_token = None
        self.distill_head = None
        if cfg.MODEL.DEIT.ENABLE_LOGIT:
            self.distill_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            self.distill_head = nn.Linear(self.hidden_dim, self.num_classes)
            nn.init.zeros_(self.distill_head.weight)
            nn.init.constant_(self.distill_head.bias, 0)
            trunc_normal_(self.distill_token, std=.02)

    def _feature_hook(self, module, inputs, outputs):
        feat_size = int(self.num_patches ** 0.5)
        x = outputs[:, self.num_tokens:].view(outputs.size(0), feat_size, feat_size, self.hidden_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        self.features.append(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x, tsne=False):
        if self.return_attn_scores:
            attn_dict = dict()
        
        x = self.patch_embed(x)
        if self.num_tokens == 1:
            x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1)
        else:
            x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), self.distill_token.repeat(x.size(0), 1, 1), x], dim=1)
        x = self.pe_dropout(x + self.pos_embed)
        
        if self.return_attn_scores:
            for i, layer in enumerate(self.layers):
                x, attn_scores = layer(x)
                attn_dict[
                    f"transformer_layer_{i}_att"
                ] = attn_scores
        else:
            for layer in self.layers:
                    x = layer(x)

        x = self.norm(x)
        logits = self.head(x[:, 0])
        
        feats = {}
        feats["pooled_feat"] = x[:, 0]

        if self.num_tokens == 1:
            if tsne:
                return logits, feats
            else:
                if self.return_attn_scores:
                    return logits, attn_dict
                else:
                    return logits

        self.distill_logits = None
        self.distill_logits = self.distill_head(x[:, 1])
        
        if self.training:
            return logits, feats
        else:
            return (logits + self.distill_logits) / 2, feats
        
        
def deit(pretrained=False, **kwargs):
    model = DeiT(**kwargs)
    cfg = kwargs["cfg"]
    if pretrained:
        try:
            model.load_state_dict(load_checkpoint(os.path.join(
                ckpt_path(), f"{cfg.DATASET.TYPE.TRAIN}_models", "deit", "student_best"))["model"])
        except:
            try:
                checkpoint = load_checkpoint(os.path.join(
                    ckpt_path(), f"{cfg.DATASET.TYPE.TRAIN}_models", "deit", "model.pyth"))
                test_err = checkpoint["test_err"] if "test_err" in checkpoint else 100
                ema_err = checkpoint["ema_err"] if "ema_err" in checkpoint else 100
                ema_state = "ema_state" if "ema_state" in checkpoint else "model_state"
                best_state = "model_state" if test_err <= ema_err else ema_state
                model.load_state_dict(checkpoint[best_state])
            except:
                raise Exception(f"No suitable pretrained model for: {model.__class__.__name__}")
    return model

def deit_in1k(pretrained=False, **kwargs):
    model = timm.create_model("deit_tiny_patch16_224",
                                pretrained=pretrained,
                                num_classes=kwargs["num_classes"])
    return model


if __name__ == "__main__":
    from torchsummary import summary
    from genformer.engine.cfg import CFG as cfg
    
    img_size = 224
    in_channels = 3
    num_classes = 4
    
    cfg.DATASET.IMG_SIZE = img_size
    
    cfg.MODEL.TRANSFORMER.PATCH_SIZE = 16
    cfg.MODEL.TRANSFORMER.HIDDEN_DIM = 192
    cfg.MODEL.TRANSFORMER.DEPTH = 12
    cfg.MODEL.TRANSFORMER.NUM_HEADS = 3
    cfg.MODEL.TRANSFORMER.MLP_RATIO = 4
    cfg.MODEL.TRANSFORMER.LN_EPS = 1e-6
    cfg.MODEL.TRANSFORMER.DROP_RATE = 0.0
    cfg.MODEL.TRANSFORMER.DROP_PATH_RATE = 0.1
    cfg.MODEL.TRANSFORMER.ATTENTION_DROP_RATE = 0.0
    
    x = torch.randn(2, in_channels, img_size, img_size).cuda()
    net = deit(pretrained=False, in_channels=in_channels, num_classes=num_classes, cfg=cfg).cuda()
    net_1k = deit_in1k(pretrained=True, in_channels=in_channels, num_classes=num_classes, cfg=cfg).cuda()
    import time

    a = time.time()
    logit = net(x)
    b = time.time()
    print(b - a)
    print(logit.shape)
    print(net.complexity())
    summary(net, (in_channels, img_size, img_size))
