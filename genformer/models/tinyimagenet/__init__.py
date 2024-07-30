from .ConViT import convit
from .DeiT import deit
from .PVT import pvt
from .PVTv2 import pvtv2
from .resnetv2 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .resnetv2tt import ResNetv2TT, ResNetv2TT_in1k

tinyimagenet_model_dict = {
    "convit"            : convit,
    "deit"              : deit,
    "pvt"               : pvt,
    "pvtv2"             : pvtv2,
    "ResNetv2TT"        : ResNetv2TT,
    "ResNetv2TT_in1k"   : ResNetv2TT_in1k,
    "ResNet18"          : ResNet18,
    "ResNet34"          : ResNet34,
    "ResNet50"          : ResNet50,
    "ResNet101"         : ResNet101,
    "ResNet152"         : ResNet152,
}
