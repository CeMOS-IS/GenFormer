import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

def layernorm(w_in, eps):
    return nn.LayerNorm(w_in, eps)


class MultiheadAttention(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads,
                 qkv_bias=False,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 qk_scale=None,
                 return_attn_scores=False):
        super().__init__()
        assert out_channels % num_heads == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.return_attn_scores = return_attn_scores

        self.norm_factor = qk_scale if qk_scale else (out_channels // num_heads) ** -0.5
        self.qkv_transform = nn.Linear(in_channels, out_channels * 3, bias=qkv_bias)
        self.projection = nn.Linear(out_channels, out_channels)
        self.attention_dropout = nn.Dropout(attn_drop_rate)
        self.projection_dropout = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        N, L, _ = x.shape
        x = self.qkv_transform(x).view(N, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = x[0], x[1], x[2]

        qk = query @ key.transpose(-1, -2) * self.norm_factor
        attn_scores = F.softmax(qk, dim=-1) # Attention scores
        qk = self.attention_dropout(attn_scores)

        out = qk @ value
        out = out.transpose(1, 2).contiguous().view(N, L, self.out_channels)
        out = self.projection(out)
        out = self.projection_dropout(out)
        
        if self.in_channels != self.out_channels:
            out = out + value.squeeze(1)

        if self.return_attn_scores:
            return out, attn_scores
        return out


class MLP(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 drop_rate=0.,
                 hidden_ratio=1.):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = int(in_channels * hidden_ratio)
        self.fc1 = nn.Linear(in_channels, self.hidden_channels)
        self.fc2 = nn.Linear(self.hidden_channels, out_channels)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 num_heads,
                 qkv_bias=False,
                 out_channels=None,
                 mlp_ratio=1.,
                 eps=1e-6,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qk_scale=None,
                 return_attn_scores=False):
        super(TransformerLayer, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.return_attn_scores = return_attn_scores
        
        self.norm1 = layernorm(in_channels, eps)
        self.attn = MultiheadAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            qk_scale=qk_scale,
            return_attn_scores=self.return_attn_scores)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = layernorm(out_channels, eps)
        self.mlp = MLP(
            in_channels=out_channels,
            out_channels=out_channels,
            drop_rate=drop_rate,
            hidden_ratio=mlp_ratio)

    def forward(self, x):
        norm1 = self.norm1(x)
        if self.return_attn_scores:
            attn_out, attn_scores = self.attn(norm1)
        else:
            attn_out = self.attn(norm1)            
        if self.in_channels == self.out_channels:
            x = x + self.drop_path(attn_out)
        else:
            x = self.attn(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if self.return_attn_scores:
            return x, attn_scores
        else:
            return x


class PatchEmbedding(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_channels=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size and W == self.img_size
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def conv2d(w_in, w_out, k, *, stride=1, groups=1, bias=False):
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    s, p, g, b = stride, (k - 1) // 2, groups, bias
    return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, groups=g, bias=b)


def norm2d(w_in):
    return nn.BatchNorm2d(num_features=w_in, eps=1e-5, momentum=0.1)


def pool2d(_w_in, k, *, stride=1):
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    return nn.MaxPool2d(k, stride=stride, padding=(k - 1) // 2)


def gap2d(_w_in):
    return nn.AdaptiveAvgPool2d((1, 1))


def linear(w_in, w_out, *, bias=False):
    return nn.Linear(w_in, w_out, bias=bias)


def activation(activation_fun="relu"):
    activation_fun = activation_fun.lower()
    if activation_fun == "relu":
        return nn.ReLU(inplace=True)
    elif activation_fun == "silu" or activation_fun == "swish":
        try:
            return torch.nn.SiLU()
        except AttributeError:
            return SiLU()
    elif activation_fun == "gelu":
        return torch.nn.GELU()
    else:
        raise AssertionError("Unknown MODEL.ACTIVATION_FUN: " + activation_fun)


def conv2d_cx(cx, w_in, w_out, k, *, stride=1, groups=1, bias=False):
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    flops += k * k * w_in * w_out * h * w // groups + (w_out * h * w if bias else 0)
    params += k * k * w_in * w_out // groups + (w_out if bias else 0)
    acts += w_out * h * w
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def norm2d_cx(cx, w_in):
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    params += 2 * w_in
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def pool2d_cx(cx, w_in, k, *, stride=1):
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    acts += w_in * h * w
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def gap2d_cx(cx, _w_in):
    flops, params, acts = cx["flops"], cx["params"], cx["acts"]
    return {"h": 1, "w": 1, "flops": flops, "params": params, "acts": acts}


def layernorm_cx(cx, w_in):
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    params += 2 * w_in
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def linear_cx(cx, w_in, w_out, *, bias=False, num_locations=1):
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    flops += w_in * w_out * num_locations + (w_out * num_locations if bias else 0)
    params += w_in * w_out + (w_out if bias else 0)
    acts += w_out * num_locations
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


class SiLU(nn.Module):

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SE(nn.Module):

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = gap2d(w_in)
        self.f_ex = nn.Sequential(
            conv2d(w_in, w_se, 1, bias=True),
            activation(),
            conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx = gap2d_cx(cx, w_in)
        cx = conv2d_cx(cx, w_in, w_se, 1, bias=True)
        cx = conv2d_cx(cx, w_se, w_in, 1, bias=True)
        cx["h"], cx["w"] = h, w
        return cx


def adjust_block_compatibility(ws, bs, gs):
    assert len(ws) == len(bs) == len(gs)
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))
    assert all(b < 1 or b % 1 == 0 for b in bs)
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, int(b)) if b > 1 else g for g, b in zip(gs, bs)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))
    return ws, bs, gs


def init_weights(m, zero_init_gamma=False):
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = hasattr(m, "final_bn") and m.final_bn and zero_init_gamma
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x
