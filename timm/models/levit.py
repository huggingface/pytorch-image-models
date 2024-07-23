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
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN
from timm.layers import to_ntuple, to_2tuple, get_act_layer, DropPath, trunc_normal_, ndgrid
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['Levit']


class ConvNorm(nn.Module):
    def __init__(
            self, in_chs, out_chs, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.linear = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_chs)

        nn.init.constant_(self.bn.weight, bn_weight_init)

    @torch.no_grad()
    def fuse(self):
        c, bn = self.linear, self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1), w.size(0), w.shape[2:], stride=self.linear.stride,
            padding=self.linear.padding, dilation=self.linear.dilation, groups=self.linear.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        return self.bn(self.linear(x))


class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, bn_weight_init=1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

        nn.init.constant_(self.bn.weight, bn_weight_init)

    @torch.no_grad()
    def fuse(self):
        l, bn = self.linear, self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        x = self.linear(x)
        return self.bn(x.flatten(0, 1)).reshape_as(x)


class NormLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, std=0.02, drop=0.):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.drop = nn.Dropout(drop)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        trunc_normal_(self.linear.weight, std=std)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self.bn, self.linear
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.linear.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.linear.bias
        m = nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        return self.linear(self.drop(self.bn(x)))


class Stem8(nn.Sequential):
    def __init__(self, in_chs, out_chs, act_layer):
        super().__init__()
        self.stride = 8

        self.add_module('conv1', ConvNorm(in_chs, out_chs // 4, 3, stride=2, padding=1))
        self.add_module('act1', act_layer())
        self.add_module('conv2', ConvNorm(out_chs // 4, out_chs // 2, 3, stride=2, padding=1))
        self.add_module('act2', act_layer())
        self.add_module('conv3', ConvNorm(out_chs // 2, out_chs, 3, stride=2, padding=1))


class Stem16(nn.Sequential):
    def __init__(self, in_chs, out_chs, act_layer):
        super().__init__()
        self.stride = 16

        self.add_module('conv1', ConvNorm(in_chs, out_chs // 8, 3, stride=2, padding=1))
        self.add_module('act1', act_layer())
        self.add_module('conv2', ConvNorm(out_chs // 8, out_chs // 4, 3, stride=2, padding=1))
        self.add_module('act2', act_layer())
        self.add_module('conv3', ConvNorm(out_chs // 4, out_chs // 2, 3, stride=2, padding=1))
        self.add_module('act3', act_layer())
        self.add_module('conv4', ConvNorm(out_chs // 2, out_chs, 3, stride=2, padding=1))


class Downsample(nn.Module):
    def __init__(self, stride, resolution, use_pool=False):
        super().__init__()
        self.stride = stride
        self.resolution = to_2tuple(resolution)
        self.pool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False) if use_pool else None

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution[0], self.resolution[1], C)
        if self.pool is not None:
            x = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        else:
            x = x[:, ::self.stride, ::self.stride]
        return x.reshape(B, -1, C)


class Attention(nn.Module):
    attention_bias_cache: Dict[str, torch.Tensor]

    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4.,
            resolution=14,
            use_conv=False,
            act_layer=nn.SiLU,
    ):
        super().__init__()
        ln_layer = ConvNorm if use_conv else LinearNorm
        resolution = to_2tuple(resolution)

        self.use_conv = use_conv
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.key_attn_dim = key_dim * num_heads
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = int(attn_ratio * key_dim) * num_heads

        self.qkv = ln_layer(dim, self.val_attn_dim + self.key_attn_dim * 2)
        self.proj = nn.Sequential(OrderedDict([
            ('act', act_layer()),
            ('ln', ln_layer(self.val_attn_dim, dim, bn_weight_init=0))
        ]))

        self.attention_biases = nn.Parameter(torch.zeros(num_heads, resolution[0] * resolution[1]))
        pos = torch.stack(ndgrid(torch.arange(resolution[0]), torch.arange(resolution[1]))).flatten(1)
        rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * resolution[1]) + rel_pos[1]
        self.register_buffer('attention_bias_idxs', rel_pos, persistent=False)
        self.attention_bias_cache = {}

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # clear ab cache

    def get_attention_biases(self, device: torch.device) -> torch.Tensor:
        if torch.jit.is_tracing() or self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]

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


class AttentionDownsample(nn.Module):
    attention_bias_cache: Dict[str, torch.Tensor]

    def __init__(
            self,
            in_dim,
            out_dim,
            key_dim,
            num_heads=8,
            attn_ratio=2.0,
            stride=2,
            resolution=14,
            use_conv=False,
            use_pool=False,
            act_layer=nn.SiLU,
    ):
        super().__init__()
        resolution = to_2tuple(resolution)

        self.stride = stride
        self.resolution = resolution
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.key_attn_dim = key_dim * num_heads
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = self.val_dim * self.num_heads
        self.scale = key_dim ** -0.5
        self.use_conv = use_conv

        if self.use_conv:
            ln_layer = ConvNorm
            sub_layer = partial(
                nn.AvgPool2d,
                kernel_size=3 if use_pool else 1, padding=1 if use_pool else 0, count_include_pad=False)
        else:
            ln_layer = LinearNorm
            sub_layer = partial(Downsample, resolution=resolution, use_pool=use_pool)

        self.kv = ln_layer(in_dim, self.val_attn_dim + self.key_attn_dim)
        self.q = nn.Sequential(OrderedDict([
            ('down', sub_layer(stride=stride)),
            ('ln', ln_layer(in_dim, self.key_attn_dim))
        ]))
        self.proj = nn.Sequential(OrderedDict([
            ('act', act_layer()),
            ('ln', ln_layer(self.val_attn_dim, out_dim))
        ]))

        self.attention_biases = nn.Parameter(torch.zeros(num_heads, resolution[0] * resolution[1]))
        k_pos = torch.stack(ndgrid(torch.arange(resolution[0]), torch.arange(resolution[1]))).flatten(1)
        q_pos = torch.stack(ndgrid(
            torch.arange(0, resolution[0], step=stride),
            torch.arange(0, resolution[1], step=stride)
        )).flatten(1)
        rel_pos = (q_pos[..., :, None] - k_pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * resolution[1]) + rel_pos[1]
        self.register_buffer('attention_bias_idxs', rel_pos, persistent=False)

        self.attention_bias_cache = {}  # per-device attention_biases cache

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # clear ab cache

    def get_attention_biases(self, device: torch.device) -> torch.Tensor:
        if torch.jit.is_tracing() or self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]

    def forward(self, x):
        if self.use_conv:
            B, C, H, W = x.shape
            HH, WW = (H - 1) // self.stride + 1, (W - 1) // self.stride + 1
            k, v = self.kv(x).view(B, self.num_heads, -1, H * W).split([self.key_dim, self.val_dim], dim=2)
            q = self.q(x).view(B, self.num_heads, self.key_dim, -1)

            attn = (q.transpose(-2, -1) @ k) * self.scale + self.get_attention_biases(x.device)
            attn = attn.softmax(dim=-1)

            x = (v @ attn.transpose(-2, -1)).reshape(B, self.val_attn_dim, HH, WW)
        else:
            B, N, C = x.shape
            k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
            k = k.permute(0, 2, 3, 1)  # BHCN
            v = v.permute(0, 2, 1, 3)  # BHNC
            q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)

            attn = q @ k * self.scale + self.get_attention_biases(x.device)
            attn = attn.softmax(dim=-1)

            x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
        x = self.proj(x)
        return x


class LevitMlp(nn.Module):
    """ MLP for Levit w/ normalization + ability to switch btw conv and linear
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            use_conv=False,
            act_layer=nn.SiLU,
            drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        ln_layer = ConvNorm if use_conv else LinearNorm

        self.ln1 = ln_layer(in_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.ln2 = ln_layer(hidden_features, out_features, bn_weight_init=0)

    def forward(self, x):
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.ln2(x)
        return x


class LevitDownsample(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            key_dim,
            num_heads=8,
            attn_ratio=4.,
            mlp_ratio=2.,
            act_layer=nn.SiLU,
            attn_act_layer=None,
            resolution=14,
            use_conv=False,
            use_pool=False,
            drop_path=0.,
    ):
        super().__init__()
        attn_act_layer = attn_act_layer or act_layer

        self.attn_downsample = AttentionDownsample(
            in_dim=in_dim,
            out_dim=out_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            act_layer=attn_act_layer,
            resolution=resolution,
            use_conv=use_conv,
            use_pool=use_pool,
        )

        self.mlp = LevitMlp(
            out_dim,
            int(out_dim * mlp_ratio),
            use_conv=use_conv,
            act_layer=act_layer
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.attn_downsample(x)
        x = x + self.drop_path(self.mlp(x))
        return x


class LevitBlock(nn.Module):
    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4.,
            mlp_ratio=2.,
            resolution=14,
            use_conv=False,
            act_layer=nn.SiLU,
            attn_act_layer=None,
            drop_path=0.,
    ):
        super().__init__()
        attn_act_layer = attn_act_layer or act_layer

        self.attn = Attention(
            dim=dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            resolution=resolution,
            use_conv=use_conv,
            act_layer=attn_act_layer,
            )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = LevitMlp(
            dim,
            int(dim * mlp_ratio),
            use_conv=use_conv,
            act_layer=act_layer
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(x))
        x = x + self.drop_path2(self.mlp(x))
        return x


class LevitStage(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            key_dim,
            depth=4,
            num_heads=8,
            attn_ratio=4.0,
            mlp_ratio=4.0,
            act_layer=nn.SiLU,
            attn_act_layer=None,
            resolution=14,
            downsample='',
            use_conv=False,
            drop_path=0.,
    ):
        super().__init__()
        resolution = to_2tuple(resolution)

        if downsample:
            self.downsample = LevitDownsample(
                in_dim,
                out_dim,
                key_dim=key_dim,
                num_heads=in_dim // key_dim,
                attn_ratio=4.,
                mlp_ratio=2.,
                act_layer=act_layer,
                attn_act_layer=attn_act_layer,
                resolution=resolution,
                use_conv=use_conv,
                drop_path=drop_path,
            )
            resolution = [(r - 1) // 2 + 1 for r in resolution]
        else:
            assert in_dim == out_dim
            self.downsample = nn.Identity()

        blocks = []
        for _ in range(depth):
            blocks += [LevitBlock(
                out_dim,
                key_dim,
                num_heads=num_heads,
                attn_ratio=attn_ratio,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                attn_act_layer=attn_act_layer,
                resolution=resolution,
                use_conv=use_conv,
                drop_path=drop_path,
            )]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class Levit(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage

    NOTE: distillation is defaulted to True since pretrained weights use it, will cause problems
    w/ train scripts that don't take tuple outputs,
    """

    def __init__(
            self,
            img_size=224,
            in_chans=3,
            num_classes=1000,
            embed_dim=(192,),
            key_dim=64,
            depth=(12,),
            num_heads=(3,),
            attn_ratio=2.,
            mlp_ratio=2.,
            stem_backbone=None,
            stem_stride=None,
            stem_type='s16',
            down_op='subsample',
            act_layer='hard_swish',
            attn_act_layer=None,
            use_conv=False,
            global_pool='avg',
            drop_rate=0.,
            drop_path_rate=0.):
        super().__init__()
        act_layer = get_act_layer(act_layer)
        attn_act_layer = get_act_layer(attn_act_layer or act_layer)
        self.use_conv = use_conv
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = embed_dim[-1]
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []

        num_stages = len(embed_dim)
        assert len(depth) == num_stages
        num_heads = to_ntuple(num_stages)(num_heads)
        attn_ratio = to_ntuple(num_stages)(attn_ratio)
        mlp_ratio = to_ntuple(num_stages)(mlp_ratio)

        if stem_backbone is not None:
            assert stem_stride >= 2
            self.stem = stem_backbone
            stride = stem_stride
        else:
            assert stem_type in ('s16', 's8')
            if stem_type == 's16':
                self.stem = Stem16(in_chans, embed_dim[0], act_layer=act_layer)
            else:
                self.stem = Stem8(in_chans, embed_dim[0], act_layer=act_layer)
            stride = self.stem.stride
        resolution = tuple([i // p for i, p in zip(to_2tuple(img_size), to_2tuple(stride))])

        in_dim = embed_dim[0]
        stages = []
        for i in range(num_stages):
            stage_stride = 2 if i > 0 else 1
            stages += [LevitStage(
                in_dim,
                embed_dim[i],
                key_dim,
                depth=depth[i],
                num_heads=num_heads[i],
                attn_ratio=attn_ratio[i],
                mlp_ratio=mlp_ratio[i],
                act_layer=act_layer,
                attn_act_layer=attn_act_layer,
                resolution=resolution,
                use_conv=use_conv,
                downsample=down_op if stage_stride == 2 else '',
                drop_path=drop_path_rate
            )]
            stride *= stage_stride
            resolution = tuple([(r - 1) // stage_stride + 1 for r in resolution])
            self.feature_info += [dict(num_chs=embed_dim[i], reduction=stride, module=f'stages.{i}')]
            in_dim = embed_dim[i]
        self.stages = nn.Sequential(*stages)

        # Classifier head
        self.head = NormLinear(embed_dim[-1], num_classes, drop=drop_rate) if num_classes > 0 else nn.Identity()

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
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int , global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = NormLinear(
            self.num_features, num_classes, drop=self.drop_rate) if num_classes > 0 else nn.Identity()

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages), indices)

        # forward pass
        x = self.stem(x)
        B, C, H, W = x.shape
        if not self.use_conv:
            x = x.flatten(2).transpose(1, 2)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]
        for feat_idx, stage in enumerate(stages):
            x = stage(x)
            if feat_idx in take_indices:
                if self.use_conv:
                    intermediates.append(x)
                else:
                    intermediates.append(x.reshape(B, H, W, -1).permute(0, 3, 1, 2))
            H = (H + 2 - 1) // 2
            W = (W + 2 - 1) // 2

        if intermediates_only:
            return intermediates

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        self.stages = self.stages[:max_index + 1]  # truncate blocks w/ stem as idx 0
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        x = self.stem(x)
        if not self.use_conv:
            x = x.flatten(2).transpose(1, 2)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
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
    def get_classifier(self) -> nn.Module:
        return self.head, self.head_dist

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = NormLinear(
            self.num_features, num_classes, drop=self.drop_rate) if num_classes > 0 else nn.Identity()
        self.head_dist = NormLinear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=(-2, -1)) if self.use_conv else x.mean(dim=1)
        if pre_logits:
            return x
        x, x_dist = self.head(x), self.head_dist(x)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train/finetune, inference average the classifier predictions
            return (x + x_dist) / 2


def checkpoint_filter_fn(state_dict, model):
    if 'model' in state_dict:
        state_dict = state_dict['model']

    # filter out attn biases, should not have been persistent
    state_dict = {k: v for k, v in state_dict.items() if 'attention_bias_idxs' not in k}

    D = model.state_dict()
    out_dict = {}
    for ka, kb, va, vb in zip(D.keys(), state_dict.keys(), D.values(), state_dict.values()):
        if va.ndim == 4 and vb.ndim == 2:
            vb = vb[:, :, None, None]
        if va.shape != vb.shape:
            # head or first-conv shapes may change for fine-tune
            assert 'head' in ka or 'stem.conv1.linear' in ka
        out_dict[ka] = vb

    return out_dict


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

    # stride-8 stem experiments
    levit_384_s8=dict(
        embed_dim=(384, 512, 768), key_dim=32, num_heads=(6, 9, 12), depth=(4, 4, 4),
        act_layer='silu', stem_type='s8'),
    levit_512_s8=dict(
        embed_dim=(512, 640, 896), key_dim=64, num_heads=(8, 10, 14), depth=(4, 4, 4),
        act_layer='silu', stem_type='s8'),

    # wider experiments
    levit_512=dict(
        embed_dim=(512, 768, 1024), key_dim=64, num_heads=(8, 12, 16), depth=(4, 4, 4), act_layer='silu'),

    # deeper experiments
    levit_256d=dict(
        embed_dim=(256, 384, 512), key_dim=32, num_heads=(4, 6, 8), depth=(4, 8, 6), act_layer='silu'),
    levit_512d=dict(
        embed_dim=(512, 640, 768), key_dim=64, num_heads=(8, 10, 12), depth=(4, 8, 6), act_layer='silu'),
)


def create_levit(variant, cfg_variant=None, pretrained=False, distilled=True, **kwargs):
    is_conv = '_conv' in variant
    out_indices = kwargs.pop('out_indices', (0, 1, 2))
    if kwargs.get('features_only', False) and not is_conv:
        kwargs.setdefault('feature_cls', 'getter')
    if cfg_variant is None:
        if variant in model_cfgs:
            cfg_variant = variant
        elif is_conv:
            cfg_variant = variant.replace('_conv', '')

    model_cfg = dict(model_cfgs[cfg_variant], **kwargs)
    model = build_model_with_cfg(
        LevitDistilled if distilled else Levit,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **model_cfg,
    )
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1.linear', 'classifier': ('head.linear', 'head_dist.linear'),
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # weights in nn.Linear mode
    'levit_128s.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'levit_128.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'levit_192.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'levit_256.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'levit_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
    ),

    # weights in nn.Conv2d mode
    'levit_conv_128s.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        pool_size=(4, 4),
    ),
    'levit_conv_128.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        pool_size=(4, 4),
    ),
    'levit_conv_192.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        pool_size=(4, 4),
    ),
    'levit_conv_256.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        pool_size=(4, 4),
    ),
    'levit_conv_384.fb_dist_in1k': _cfg(
        hf_hub_id='timm/',
        pool_size=(4, 4),
    ),

    'levit_384_s8.untrained': _cfg(classifier='head.linear'),
    'levit_512_s8.untrained': _cfg(classifier='head.linear'),
    'levit_512.untrained': _cfg(classifier='head.linear'),
    'levit_256d.untrained': _cfg(classifier='head.linear'),
    'levit_512d.untrained': _cfg(classifier='head.linear'),

    'levit_conv_384_s8.untrained': _cfg(classifier='head.linear'),
    'levit_conv_512_s8.untrained': _cfg(classifier='head.linear'),
    'levit_conv_512.untrained': _cfg(classifier='head.linear'),
    'levit_conv_256d.untrained': _cfg(classifier='head.linear'),
    'levit_conv_512d.untrained': _cfg(classifier='head.linear'),
})


@register_model
def levit_128s(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_128s', pretrained=pretrained, **kwargs)


@register_model
def levit_128(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_128', pretrained=pretrained, **kwargs)


@register_model
def levit_192(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_192', pretrained=pretrained, **kwargs)


@register_model
def levit_256(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_256', pretrained=pretrained, **kwargs)


@register_model
def levit_384(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_384', pretrained=pretrained, **kwargs)


@register_model
def levit_384_s8(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_384_s8', pretrained=pretrained, **kwargs)


@register_model
def levit_512_s8(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_512_s8', pretrained=pretrained, distilled=False, **kwargs)


@register_model
def levit_512(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_512', pretrained=pretrained, distilled=False, **kwargs)


@register_model
def levit_256d(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_256d', pretrained=pretrained, distilled=False, **kwargs)


@register_model
def levit_512d(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_512d', pretrained=pretrained, distilled=False, **kwargs)


@register_model
def levit_conv_128s(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_conv_128s', pretrained=pretrained, use_conv=True, **kwargs)


@register_model
def levit_conv_128(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_conv_128', pretrained=pretrained, use_conv=True, **kwargs)


@register_model
def levit_conv_192(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_conv_192', pretrained=pretrained, use_conv=True, **kwargs)


@register_model
def levit_conv_256(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_conv_256', pretrained=pretrained, use_conv=True, **kwargs)


@register_model
def levit_conv_384(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_conv_384', pretrained=pretrained, use_conv=True, **kwargs)


@register_model
def levit_conv_384_s8(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_conv_384_s8', pretrained=pretrained, use_conv=True, **kwargs)


@register_model
def levit_conv_512_s8(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_conv_512_s8', pretrained=pretrained, use_conv=True, distilled=False, **kwargs)


@register_model
def levit_conv_512(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_conv_512', pretrained=pretrained, use_conv=True, distilled=False, **kwargs)


@register_model
def levit_conv_256d(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_conv_256d', pretrained=pretrained, use_conv=True, distilled=False, **kwargs)


@register_model
def levit_conv_512d(pretrained=False, **kwargs) -> Levit:
    return create_levit('levit_conv_512d', pretrained=pretrained, use_conv=True, distilled=False, **kwargs)

