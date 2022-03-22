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

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
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
from .helpers import build_model_with_cfg, checkpoint_seq
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

    levit_256d=_cfg(url='', classifier='head.l'),
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

    levit_256d=dict(
        embed_dim=(256, 384, 512), key_dim=32, num_heads=(4, 6, 8), depth=(4, 8, 6)),
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


@register_model
def levit_256d(pretrained=False, use_conv=False, **kwargs):
    return create_levit(
        'levit_256d', pretrained=pretrained, use_conv=use_conv, distilled=False, **kwargs)


class ConvNorm(nn.Sequential):
    def __init__(
            self, in_chs, out_chs, kernel_size=1, stride=1, pad=0, dilation=1,
            groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', nn.Conv2d(in_chs, out_chs, kernel_size, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_chs))

        nn.init.constant_(self.bn.weight, bn_weight_init)

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
    def __init__(self, in_features, out_features, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', nn.Linear(in_features, out_features, bias=False))
        self.add_module('bn', nn.BatchNorm1d(out_features))

        nn.init.constant_(self.bn.weight, bn_weight_init)

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
    def __init__(self, in_features, out_features, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', nn.BatchNorm1d(in_features))
        self.add_module('l', nn.Linear(in_features, out_features, bias=bias))

        trunc_normal_(self.l.weight, std=std)
        if self.l.bias is not None:
            nn.init.constant_(self.l.bias, 0)

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
        ln_layer = ConvNorm if use_conv else LinearNorm
        self.use_conv = use_conv
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.key_attn_dim = key_dim * num_heads
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = int(attn_ratio * key_dim) * num_heads

        self.qkv = ln_layer(dim, self.val_attn_dim + self.key_attn_dim * 2, resolution=resolution)
        self.proj = nn.Sequential(
            act_layer(),
            ln_layer(self.val_attn_dim, dim, bn_weight_init=0, resolution=resolution)
        )

        self.attention_biases = nn.Parameter(torch.zeros(num_heads, resolution ** 2))
        pos = torch.stack(torch.meshgrid(torch.arange(resolution), torch.arange(resolution))).flatten(1)
        rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * resolution) + rel_pos[1]
        self.register_buffer('attention_bias_idxs', rel_pos)
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
            q, k, v = self.qkv(x).view(
                B, self.num_heads, -1, H * W).split([self.key_dim, self.key_dim, self.val_dim], dim=2)

            attn = (q.transpose(-2, -1) @ k) * self.scale + self.get_attention_biases(x.device)
            attn = attn.softmax(dim=-1)

            x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        else:
            B, N, C = x.shape
            q, k, v = self.qkv(x).view(
                B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 3, 1)
            v = v.permute(0, 2, 1, 3)

            attn = q @ k * self.scale + self.get_attention_biases(x.device)
            attn = attn.softmax(dim=-1)

            x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        x = self.proj(x)
        return x


class AttentionSubsample(nn.Module):
    ab: Dict[str, torch.Tensor]

    def __init__(
            self, in_dim, out_dim, key_dim, num_heads=8, attn_ratio=2,
            act_layer=None, stride=2, resolution=14, resolution_out=7, use_conv=False):
        super().__init__()
        self.stride = stride
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.key_attn_dim = key_dim * num_heads
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = self.val_dim * self.num_heads
        self.resolution = resolution
        self.resolution_out_area = resolution_out ** 2

        self.use_conv = use_conv
        if self.use_conv:
            ln_layer = ConvNorm
            sub_layer = partial(nn.AvgPool2d, kernel_size=1, padding=0)
        else:
            ln_layer = LinearNorm
            sub_layer = partial(Subsample, resolution=resolution)

        self.kv = ln_layer(in_dim, self.val_attn_dim + self.key_attn_dim, resolution=resolution)
        self.q = nn.Sequential(
            sub_layer(stride=stride),
            ln_layer(in_dim, self.key_attn_dim, resolution=resolution_out)
        )
        self.proj = nn.Sequential(
            act_layer(),
            ln_layer(self.val_attn_dim, out_dim, resolution=resolution_out)
        )

        self.attention_biases = nn.Parameter(torch.zeros(num_heads, self.resolution ** 2))
        k_pos = torch.stack(torch.meshgrid(torch.arange(resolution), torch.arange(resolution))).flatten(1)
        q_pos = torch.stack(torch.meshgrid(
            torch.arange(0, resolution, step=stride),
            torch.arange(0, resolution, step=stride))).flatten(1)
        rel_pos = (q_pos[..., :, None] - k_pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * resolution) + rel_pos[1]
        self.register_buffer('attention_bias_idxs', rel_pos)

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
            k, v = self.kv(x).view(B, self.num_heads, -1, H * W).split([self.key_dim, self.val_dim], dim=2)
            q = self.q(x).view(B, self.num_heads, self.key_dim, self.resolution_out_area)

            attn = (q.transpose(-2, -1) @ k) * self.scale + self.get_attention_biases(x.device)
            attn = attn.softmax(dim=-1)

            x = (v @ attn.transpose(-2, -1)).reshape(B, -1, self.resolution, self.resolution)
        else:
            B, N, C = x.shape
            k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
            k = k.permute(0, 2, 3, 1)  # BHCN
            v = v.permute(0, 2, 1, 3)  # BHNC
            q = self.q(x).view(B, self.resolution_out_area, self.num_heads, self.key_dim).permute(0, 2, 1, 3)

            attn = q @ k * self.scale + self.get_attention_biases(x.device)
            attn = attn.softmax(dim=-1)

            x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
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
            use_conv=False,
            global_pool='avg',
            drop_rate=0.,
            drop_path_rate=0.):
        super().__init__()
        act_layer = get_act_layer(act_layer)
        attn_act_layer = get_act_layer(attn_act_layer)
        ln_layer = ConvNorm if use_conv else LinearNorm
        self.use_conv = use_conv
        if isinstance(img_size, tuple):
            # FIXME origin impl passes single img/res dim through whole hierarchy,
            # not sure this model will be used enough to spend time fixing it.
            assert img_size[0] == img_size[1]
            img_size = img_size[0]
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.grad_checkpointing = False

        num_stages = len(embed_dim)
        assert len(depth) == len(num_heads) == num_stages
        key_dim = to_ntuple(num_stages)(key_dim)
        attn_ratio = to_ntuple(num_stages)(attn_ratio)
        mlp_ratio = to_ntuple(num_stages)(mlp_ratio)
        down_ops = down_ops or (
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ('Subsample', key_dim[0], embed_dim[0] // key_dim[0], 4, 2, 2),
            ('Subsample', key_dim[0], embed_dim[1] // key_dim[1], 4, 2, 2),
            ('',)
        )

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
                resolution_out = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3], act_layer=attn_act_layer, stride=do[5],
                        resolution=resolution, resolution_out=resolution_out, use_conv=use_conv))
                resolution = resolution_out
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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None, distillation=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = NormLinear(self.embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if not self.use_conv:
            x = x.flatten(2).transpose(1, 2)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=(-2, -1)) if self.use_conv else x.mean(dim=1)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class LevitDistilled(Levit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_dist = NormLinear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.distilled_training = False  # must set this True to train w/ distillation token

    @torch.jit.ignore
    def get_classifier(self):
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=None, distillation=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = NormLinear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = NormLinear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def forward_head(self, x):
        if self.global_pool == 'avg':
            x = x.mean(dim=(-2, -1)) if self.use_conv else x.mean(dim=1)
        x, x_dist = self.head(x), self.head_dist(x)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train/finetune, inference average the classifier predictions
            return (x + x_dist) / 2


def checkpoint_filter_fn(state_dict, model):
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    D = model.state_dict()
    for k in state_dict.keys():
        if k in D and D[k].ndim == 4 and state_dict[k].ndim == 2:
            state_dict[k] = state_dict[k][:, :, None, None]
    return state_dict


def create_levit(variant, pretrained=False, distilled=True, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model_cfg = dict(**model_cfgs[variant], **kwargs)
    model = build_model_with_cfg(
        LevitDistilled if distilled else Levit, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **model_cfg)
    return model

