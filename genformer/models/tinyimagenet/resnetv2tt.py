#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/models/resnet.py
"""

import os
import torch
from torch.nn import Module
import timm

from genformer.models.utils import ckpt_path
from genformer.engine.utils import load_checkpoint
from genformer.models.base import BaseConvModel
from genformer.models.common import (
    activation,
    conv2d,
    gap2d,
    init_weights,
    linear,
    norm2d,
    pool2d,
)


_IN_STAGE_DS = {
    18: (2, 2, 2, 2), 34: (3, 4, 6, 3),
    50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}


def get_trans_fun(name):
    trans_funs = {
        "basic_transform": BasicTransform,
        "bottleneck_transform": BottleneckTransform,
    }
    err_str = "Transformation function '{}' not supported"
    assert name in trans_funs.keys(), err_str.format(name)
    return trans_funs[name]


class ResHead(Module):

    def __init__(self, w_in, num_classes):
        super(ResHead, self).__init__()
        self.avg_pool = gap2d(w_in)
        self.fc = linear(w_in, num_classes, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicTransform(Module):

    def __init__(self, w_in, w_out, stride, w_b=None, groups=1):
        err_str = "Basic transform does not support w_b and groups options"
        assert w_b is None and groups == 1, err_str
        super(BasicTransform, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out)
        self.a_af = activation()
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = norm2d(w_out)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class BottleneckTransform(Module):

    def __init__(self, w_in, w_out, stride, w_b, groups):
        super(BottleneckTransform, self).__init__()
        (s1, s3) = (stride, 1)
        self.a = conv2d(w_in, w_b, 1, stride=s1)
        self.a_bn = norm2d(w_b)
        self.a_af = activation()
        self.b = conv2d(w_b, w_b, 3, stride=s3, groups=groups)
        self.b_bn = norm2d(w_b)
        self.b_af = activation()
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBlock(Module):

    def __init__(self, w_in, w_out, stride, trans_fun, w_b=None, groups=1):
        super(ResBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = trans_fun(w_in, w_out, stride, w_b, groups)
        self.af = activation()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))


class ResStage(Module):

    def __init__(self, w_in, w_out, stride, d, w_b=None, groups=1, _trans_fun="basic_transform"):
        super(ResStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_fun = get_trans_fun(_trans_fun)
            res_block = ResBlock(b_w_in, w_out, b_stride, trans_fun, w_b, groups)
            self.add_module("b{}".format(i + 1), res_block)
        self.out_channels = w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class ResStemCifar(Module):

    def __init__(self, w_in, w_out):
        super(ResStemCifar, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResStemIN(Module):

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResNet(BaseConvModel):

    def __init__(self, in_channels:int=None, num_classes:int=None, cfg:dict=None, **kwargs):
        super().__init__(cfg=cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.num_groups = cfg.MODEL.RESNET.NUM_GROUPS
        self.group_width = cfg.MODEL.RESNET.WIDTH_PER_GROUP
        self.trans_fun = cfg.MODEL.RESNET.TRANS_FUN
        
        if self.depth in _IN_STAGE_DS:
            self._construct_imagenet()
        else:
            self._construct_cifar()
        layers = [m for m in self.modules() if isinstance(m, ResStage)]
        feature_dims = [m.out_channels for m in layers]
        self.apply(init_weights)

    def _feature_hook(self, module, inputs, outputs):
        self.features.append(outputs)

    def _construct_cifar(self):
        err_str = "Model depth should be of the format 6n + 2 for cifar"
        assert (self.depth - 2) % 6 == 0, err_str
        d = int((self.depth - 2) / 6)
        self.stem = ResStemCifar(self.in_channels, 16)
        self.s1 = ResStage(16, 16, stride=1, d=d, _trans_fun=self.trans_fun)
        self.s2 = ResStage(16, 32, stride=2, d=d, _trans_fun=self.trans_fun)
        self.s3 = ResStage(32, 64, stride=2, d=d, _trans_fun=self.trans_fun)
        self.head = ResHead(64, self.num_classes)

    def _construct_imagenet(self):
        g, gw = self.num_groups, self.group_width
        (d1, d2, d3, d4) = _IN_STAGE_DS[self.depth]
        w_b = gw * g
        self.stem = ResStemIN(self.in_channels, 64)
        if self.trans_fun == 'bottleneck_transform':
            self.s1 = ResStage(64, 256, stride=1, d=d1, w_b=w_b, groups=g, _trans_fun=self.trans_fun)
            self.s2 = ResStage(256, 512, stride=2, d=d2, w_b=w_b * 2, groups=g, _trans_fun=self.trans_fun)
            self.s3 = ResStage(512, 1024, stride=2, d=d3, w_b=w_b * 4, groups=g, _trans_fun=self.trans_fun)
            self.s4 = ResStage(1024, 2048, stride=2, d=d4, w_b=w_b * 8, groups=g, _trans_fun=self.trans_fun)
            self.head = ResHead(2048, self.num_classes)
        else:
            self.s1 = ResStage(64, 64, stride=1, d=d1)
            self.s2 = ResStage(64, 128, stride=2, d=d2)
            self.s3 = ResStage(128, 256, stride=2, d=d3)
            self.s4 = ResStage(256, 512, stride=2, d=d4)
            self.head = ResHead(512, self.num_classes)

    def forward(self, x, tsne=False):
        for module in self.children():
            x = module(x)
        feats = {}
        if tsne:
            return x, feats
        else:
            return x


def ResNetv2TT(pretrained=False, **kwargs):
    model = ResNet(**kwargs)
    cfg = kwargs["cfg"]
    if pretrained:
        try:
            model.load_state_dict(load_checkpoint(os.path.join(
                ckpt_path(), f"{cfg.DATASET.TYPE.TRAIN}_models", "resnet18", "student_best"))["model"])
        except:
            raise Exception(f"No suitable pretrained model for: {model.__class__.__name__}")
    return model

def ResNetv2TT_in1k(pretrained=False, **kwargs):
    model = timm.create_model("resnet18",
                                pretrained=pretrained,
                                num_classes=kwargs["num_classes"])
    return model


if __name__ == "__main__":
    from genformer.engine.cfg import CFG as cfg
    
    img_size = 64
    in_channels = 3
    num_classes = 200
    
    cfg.DATASET.IMG_SIZE = img_size
    
    x = torch.randn(2, in_channels, img_size, img_size).cuda()
    net = ResNetv2TT(in_channels=in_channels, num_classes=num_classes, cfg=cfg).cuda()
    import time

    a = time.time()
    logit = net(x)
    b = time.time()
    print(b - a)
    print(logit.shape)
    print(net.complexity())