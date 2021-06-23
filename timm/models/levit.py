""" LeViT

Paper: `LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference`
    - https://arxiv.org/abs/2104.01136

@article{graham2021levit,
  title={LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference},
  author={Benjamin Graham and Alaaeldin El-Nouby and Hugo Touvron and Pierre Stock and Armand Joulin and Herv\'e J\'egou and Matthijs Douze},
  journal={arXiv preprint arXiv:22104.01136},
  year={2021}
}

Adapted from official impl at https://github.com/facebookresearch/LeViT, original copyright bellow.

This version combines both conv/linear models and fixes torchscript compatibility.

Modifications by/coyright Copyright 2021 Ross Wightman
"""

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# Modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Copyright 2020 Ross Wightman, Apache-2.0 License
import itertools
from copy import deepcopy
from functools import partial
from typing import Dict

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN
from .helpers import build_model_with_cfg, overlay_external_default_cfg
from .layers import to_ntuple, get_act_layer
from .vision_transformer import trunc_normal_
from .registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.0.c', 'classifier': ('head.l', 'head_dist.l'),
        **kwargs
    }


default_cfgs = dict(
    levit_128s=_cfg(
        url='https://dl.fbaipublicfiles.com/LeViT/LeViT-128S-96703c44.pth'
    ),
    levit_128=_cfg(
        url='https://dl.fbaipublicfiles.com/LeViT/LeViT-128-b88c2750.pth'
    ),
    levit_192=_cfg(
        url='https://dl.fbaipublicfiles.com/LeViT/LeViT-192-92712e41.pth'
    ),
    levit_256=_cfg(
        url='https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth'
    ),
    levit_384=_cfg(
        url='https://dl.fbaipublicfiles.com/LeViT/LeViT-384-9bdaf2e2.pth'
    ),
)

model_cfgs = dict(
    levit_128s=dict(
        embed_dim=(128, 256, 384), key_dim=16, num_heads=(4, 6, 8), depth=(2, 3, 4)),
    levit_128=dict(
        embed_dim=(128, 256, 384), key_dim=16, num_heads=(4, 8, 12), depth=(4, 4, 4)),
    levit_192=dict(
        embed_dim=(192, 288, 384), key_dim=32, num_heads=(3, 5, 6), depth=(4, 4, 4)),
    levit_256=dict(
        embed_dim=(256, 384, 512), key_dim=32, num_heads=(4, 6, 8), depth=(4, 4, 4)),
    levit_384=dict(
        embed_dim=(384, 512, 768), key_dim=32, num_heads=(6, 9, 12), depth=(4, 4, 4)),
)

__all__ = ['Levit']


@register_model
def levit_128s(pretrained=False, use_conv=False, **kwargs):
    return create_levit(
        'levit_128s', pretrained=pretrained, use_conv=use_conv, **kwargs)


@register_model
def levit_128(pretrained=False, use_conv=False, **kwargs):
    return create_levit(
        'levit_128', pretrained=pretrained, use_conv=use_conv, **kwargs)


@register_model
def levit_192(pretrained=False, use_conv=False, **kwargs):
    return create_levit(
        'levit_192', pretrained=pretrained, use_conv=use_conv, **kwargs)


@register_model
def levit_256(pretrained=False, use_conv=False, **kwargs):
    return create_levit(
        'levit_256', pretrained=pretrained, use_conv=use_conv, **kwargs)


@register_model
def levit_384(pretrained=False, use_conv=False, **kwargs):
    return create_levit(
        'levit_384', pretrained=pretrained, use_conv=use_conv, **kwargs)


class ConvNorm(nn.Sequential):
    def __init__(
            self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1), w.size(0), w.shape[2:], stride=self.c.stride,
            padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class LinearNorm(nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', nn.Linear(a, b, bias=False))
        bn = nn.BatchNorm1d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        x = self.c(x)
        return self.bn(x.flatten(0, 1)).reshape_as(x)


class NormLinear(nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', nn.BatchNorm1d(a))
        l = nn.Linear(a, b, bias=bias)
        trunc_normal_(l.weight, std=std)
        if bias:
            nn.init.constant_(l.bias, 0)
        self.add_module('l', l)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def stem_b16(in_chs, out_chs, activation, resolution=224):
    return nn.Sequential(
        ConvNorm(in_chs, out_chs // 8, 3, 2, 1, resolution=resolution),
        activation(),
        ConvNorm(out_chs // 8, out_chs // 4, 3, 2, 1, resolution=resolution // 2),
        activation(),
        ConvNorm(out_chs // 4, out_chs // 2, 3, 2, 1, resolution=resolution // 4),
        activation(),
        ConvNorm(out_chs // 2, out_chs, 3, 2, 1, resolution=resolution // 8))


class Residual(nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(
                x.size(0), 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class Subsample(nn.Module):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[:, ::self.stride, ::self.stride]
        return x.reshape(B, -1, C)


class Attention(nn.Module):
    ab: Dict[str, torch.Tensor]

    def __init__(
            self, dim, key_dim, num_heads=8, attn_ratio=4, act_layer=None, resolution=14, use_conv=False):
        super().__init__()

        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.use_conv = use_conv
        ln_layer = ConvNorm if self.use_conv else LinearNorm
        h = self.dh + nh_kd * 2
        self.qkv = ln_layer(dim, h, resolution=resolution)
        self.proj = nn.Sequential(
            act_layer(),
            ln_layer(self.dh, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))
        self.ab = {}

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.ab:
            self.ab = {}  # clear ab cache

    def get_attention_biases(self, device: torch.device) -> torch.Tensor:
        if self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.ab:
                self.ab[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.ab[device_key]

    def forward(self, x):  # x (B,C,H,W)
        if self.use_conv:
            B, C, H, W = x.shape
            q, k, v = self.qkv(x).view(B, self.num_heads, -1, H * W).split([self.key_dim, self.key_dim, self.d], dim=2)

            attn = (q.transpose(-2, -1) @ k) * self.scale + self.get_attention_biases(x.device)
            attn = attn.softmax(dim=-1)

            x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        else:
            B, N, C = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            attn = q @ k.transpose(-2, -1) * self.scale + self.get_attention_biases(x.device)
            attn = attn.softmax(dim=-1)

            x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class AttentionSubsample(nn.Module):
    ab: Dict[str, torch.Tensor]

    def __init__(
            self, in_dim, out_dim, key_dim, num_heads=8, attn_ratio=2,
            act_layer=None, stride=2, resolution=14, resolution_=7, use_conv=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_ ** 2
        self.use_conv = use_conv
        if self.use_conv:
            ln_layer = ConvNorm
            sub_layer = partial(nn.AvgPool2d, kernel_size=1, padding=0)
        else:
            ln_layer = LinearNorm
            sub_layer = partial(Subsample, resolution=resolution)

        h = self.dh + nh_kd
        self.kv = ln_layer(in_dim, h, resolution=resolution)
        self.q = nn.Sequential(
            sub_layer(stride=stride),
            ln_layer(in_dim, nh_kd, resolution=resolution_))
        self.proj = nn.Sequential(
            act_layer(),
            ln_layer(self.dh, out_dim, resolution=resolution_))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(range(resolution_), range(resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N_, N))
        self.ab = {}  # per-device attention_biases cache

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.ab:
            self.ab = {}  # clear ab cache

    def get_attention_biases(self, device: torch.device) -> torch.Tensor:
        if self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.ab:
                self.ab[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.ab[device_key]

    def forward(self, x):
        if self.use_conv:
            B, C, H, W = x.shape
            k, v = self.kv(x).view(B, self.num_heads, -1, H * W).split([self.key_dim, self.d], dim=2)
            q = self.q(x).view(B, self.num_heads, self.key_dim, self.resolution_2)

            attn = (q.transpose(-2, -1) @ k) * self.scale + self.get_attention_biases(x.device)
            attn = attn.softmax(dim=-1)

            x = (v @ attn.transpose(-2, -1)).reshape(B, -1, self.resolution_, self.resolution_)
        else:
            B, N, C = x.shape
            k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.d], dim=3)
            k = k.permute(0, 2, 1, 3)  # BHNC
            v = v.permute(0, 2, 1, 3)  # BHNC
            q = self.q(x).view(B, self.resolution_2, self.num_heads, self.key_dim).permute(0, 2, 1, 3)

            attn = q @ k.transpose(-2, -1) * self.scale + self.get_attention_biases(x.device)
            attn = attn.softmax(dim=-1)

            x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


class Levit(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage

    NOTE: distillation is defaulted to True since pretrained weights use it, will cause problems
    w/ train scripts that don't take tuple outputs,
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=(192,),
            key_dim=64,
            depth=(12,),
            num_heads=(3,),
            attn_ratio=2,
            mlp_ratio=2,
            hybrid_backbone=None,
            down_ops=None,
            act_layer='hard_swish',
            attn_act_layer='hard_swish',
            distillation=True,
            use_conv=False,
            drop_rate=0.,
            drop_path_rate=0.):
        super().__init__()
        act_layer = get_act_layer(act_layer)
        attn_act_layer = get_act_layer(attn_act_layer)
        if isinstance(img_size, tuple):
            # FIXME origin impl passes single img/res dim through whole hierarchy,
            # not sure this model will be used enough to spend time fixing it.
            assert img_size[0] == img_size[1]
            img_size = img_size[0]
        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        N = len(embed_dim)
        assert len(depth) == len(num_heads) == N
        key_dim = to_ntuple(N)(key_dim)
        attn_ratio = to_ntuple(N)(attn_ratio)
        mlp_ratio = to_ntuple(N)(mlp_ratio)
        down_ops = down_ops or (
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ('Subsample', key_dim[0], embed_dim[0] // key_dim[0], 4, 2, 2),
            ('Subsample', key_dim[0], embed_dim[1] // key_dim[1], 4, 2, 2),
            ('',)
        )
        self.distillation = distillation
        self.use_conv = use_conv
        ln_layer = ConvNorm if self.use_conv else LinearNorm

        self.patch_embed = hybrid_backbone or stem_b16(in_chans, embed_dim[0], activation=act_layer)

        self.blocks = []
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(
                    Residual(
                        Attention(
                            ed, kd, nh, attn_ratio=ar, act_layer=attn_act_layer,
                            resolution=resolution, use_conv=use_conv),
                        drop_path_rate))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(nn.Sequential(
                            ln_layer(ed, h, resolution=resolution),
                            act_layer(),
                            ln_layer(h, ed, bn_weight_init=0, resolution=resolution),
                        ), drop_path_rate))
            if do[0] == 'Subsample':
                # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3], act_layer=attn_act_layer, stride=do[5],
                        resolution=resolution, resolution_=resolution_, use_conv=use_conv))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(nn.Sequential(
                            ln_layer(embed_dim[i + 1], h, resolution=resolution),
                            act_layer(),
                            ln_layer(h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution),
                        ), drop_path_rate))
        self.blocks = nn.Sequential(*self.blocks)

        # Classifier head
        self.head = NormLinear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distillation:
            self.head_dist = NormLinear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def get_classifier(self):
        if self.head_dist is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool='', distillation=None):
        self.num_classes = num_classes
        self.head = NormLinear(self.embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        if distillation is not None:
            self.distillation = distillation
        if self.distillation:
            self.head_dist = NormLinear(self.embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head_dist = None

    def forward_features(self, x):
        x = self.patch_embed(x)
        if not self.use_conv:
            x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        x = x.mean((-2, -1)) if self.use_conv else x.mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x), self.head_dist(x)
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                # during inference, return the average of both classifier predictions
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    D = model.state_dict()
    for k in state_dict.keys():
        if k in D and D[k].ndim == 4 and state_dict[k].ndim == 2:
            state_dict[k] = state_dict[k][:, :, None, None]
    return state_dict


def create_levit(variant, pretrained=False, default_cfg=None, fuse=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model_cfg = dict(**model_cfgs[variant], **kwargs)
    model = build_model_with_cfg(
        Levit, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        **model_cfg)
    #if fuse:
    #    utils.replace_batchnorm(model)
    return model

