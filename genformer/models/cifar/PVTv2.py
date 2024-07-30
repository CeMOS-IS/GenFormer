
"""
Modified from the official implementation of PVTv2.
https://github.com/whai362/PVT/blob/v2/classification/pvt_v2.py
"""

import os
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from genformer.models.utils import ckpt_path
from genformer.engine.utils import load_checkpoint
from genformer.models.common import layernorm
from genformer.models.base import BaseTransformerModel


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, eps=1e-6, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = layernorm(dim, eps)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = layernorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, eps=1e-6, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim, eps)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, eps=eps, linear=linear)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, eps)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, eps=1e-6):
        super().__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = layernorm(embed_dim, eps)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PVTv2(BaseTransformerModel):

    def __init__(self, in_channels:int=None, num_classes:int=None, cfg:dict=None, **kwargs):
        super().__init__(cfg=cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.sr_ratio = cfg.MODEL.PVT.SR_RATIO
        self.num_stages = len(self.hidden_dim)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depth))]  # stochastic depth decay rule
        cur = 0

        for i in range(self.num_stages):
            patch_embed = OverlapPatchEmbed(img_size=self.img_size if i == 0 else self.img_size // (2 ** (i + 1)),
                                            patch_size=self.patch_size[i],
                                            stride=self.patch_stride[i],
                                            in_chans=self.in_channels if i == 0 else self.hidden_dim[i - 1],
                                            embed_dim=self.hidden_dim[i],
                                            eps=self.ln_eps)

            block = nn.ModuleList([Block(
                dim=self.hidden_dim[i], num_heads=self.num_heads[i], mlp_ratio=self.mlp_ratio[i], qkv_bias=True, qk_scale=None,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[cur + j], norm_layer=layernorm,
                sr_ratio=self.sr_ratio[i], eps=self.ln_eps, linear=False)
                for j in range(self.depth[i])])
            norm = layernorm(self.hidden_dim[i], self.ln_eps)
            cur += self.depth[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        layers = [[m for m in getattr(self, f'block{i + 1}')] for i in range(self.num_stages)]
        layers = sum(layers, [])

        # classification head
        self.head = nn.Linear(self.hidden_dim[-1], self.num_classes)
        self.apply(self._init_weights)

    def _feature_hook(self, module, inp, out):
        _, H, W = inp
        feat = out.view(out.size(0), H, W, out.size(-1))
        feat = feat.permute(0, 3, 1, 2).contiguous()
        self.features.append(feat)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x, tsne=False):
        x = self.forward_features(x)
        x = self.head(x)
        feats = {}
        if tsne:
            return x, feats
        else:
            return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
    

def pvtv2(pretrained=False, **kwargs):
    model = PVTv2(**kwargs)
    cfg = kwargs["cfg"]
    if pretrained:
        try:
            model.load_state_dict(load_checkpoint(os.path.join(
                ckpt_path(), f"cifar{kwargs['num_classes']}_models", "pvtv2", "student_best"))["model"])
        except:
            raise Exception(f"No suitable pretrained model for: {model.__class__.__name__}")
    return model


if __name__ == "__main__":
    from genformer.engine.cfg import CFG as cfg
    
    img_size = 32
    in_channels = 3
    num_classes = 100
    
    cfg.DATASET.IMG_SIZE = img_size
    
    cfg.MODEL.TRANSFORMER.PATCH_SIZE = [7, 3, 3, 3]
    cfg.MODEL.TRANSFORMER.PATCH_STRIDE = [4, 2, 2, 2]
    cfg.MODEL.TRANSFORMER.HIDDEN_DIM = [32, 64, 160, 256]
    cfg.MODEL.TRANSFORMER.DEPTH = [2, 2, 2, 2]
    cfg.MODEL.TRANSFORMER.NUM_HEADS = [1, 2, 5, 8]
    cfg.MODEL.TRANSFORMER.MLP_RATIO = [8, 8, 4, 4]
    cfg.MODEL.TRANSFORMER.LN_EPS = 1e-6
    cfg.MODEL.TRANSFORMER.DROP_RATE = 0.0
    cfg.MODEL.TRANSFORMER.DROP_PATH_RATE = 0.1
    cfg.MODEL.TRANSFORMER.ATTENTION_DROP_RATE = 0.0
    
    cfg.MODEL.PVT.SR_RATIO = [8, 4, 2, 1]
    
    x = torch.randn(2, in_channels, img_size, img_size).cuda()
    net = pvtv2(in_channels=in_channels, num_classes=num_classes, cfg=cfg).cuda()
    import time

    a = time.time()
    logit, feats = net(x)
    b = time.time()
    print(b - a)
    print(logit.shape)
    print(net.complexity())
