from .ConViT import convit
from .DeiT import deit, deit_in1k
from .PVT import pvt
from .PVTv2 import pvtv2
from .resnet import (
    resnet8,
    resnet14,
    resnet20,
    resnet32,
    resnet44,
    resnet56,
    resnet110,
    resnet8x4,
    resnet32x4,
)
from .resnetv2 import (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
)
from .resnetv2tt import ResNetv2TT

cifar_model_dict = {
    "convit"        : convit,
    "deit"          : deit,
    "deit_in1k"     : deit_in1k,
    "pvt"           : pvt,
    "pvtv2"         : pvtv2,
    "resnet8"       : resnet8,
    "resnet14"      : resnet14,
    "resnet20"      : resnet20,
    "resnet32"      : resnet32,
    "resnet44"      : resnet44,
    "resnet56"      : resnet56,
    "resnet110"     : resnet110,
    "resnet8x4"     : resnet8x4,
    "resnet32x4"    : resnet32x4,
    "ResNet18"      : ResNet18,
    "ResNet34"      : ResNet34,
    "ResNet50"      : ResNet50,
    "ResNet101"     : ResNet101,
    "ResNet152"     : ResNet152,
    "ResNetv2TT"    : ResNetv2TT,
}
