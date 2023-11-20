""" DaViT: Dual Attention Vision Transformers

As described in https://arxiv.org/abs/2204.03645

Input size invariant transformer architecture that combines channel and spacial
attention in each block. The attention mechanisms used are linear in complexity.

DaViT model defs and weights adapted from https://github.com/dingmyu/davit, original copyright below

"""
# Copyright (c) 2022 Mingyu Ding
# All rights reserved.
# This source code is licensed under the MIT license
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, to_2tuple, trunc_normal_, Mlp, LayerNorm2d, get_norm_layer, use_fused_attn
from timm.layers import NormMlpClassifierHead, ClassifierHead
from ._builder import build_model_with_cfg
from ._features_fx import register_notrace_function
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['DaVit']


class ConvPosEnc(nn.Module):
    def __init__(self, dim: int, k: int = 3, act: bool = False):
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: Tensor):
        feat = self.proj(x)
        x = x + self.act(feat)
        return x


class Stem(nn.Module):
    """ Size-agnostic implementation of 2D image to patch embedding,
        allowing input size to be adjusted during model forward operation
    """

    def __init__(
            self,
            in_chs=3,
            out_chs=96,
            stride=4,
            norm_layer=LayerNorm2d,
    ):
        super().__init__()
        stride = to_2tuple(stride)
        self.stride = stride
        self.in_chs = in_chs
        self.out_chs = out_chs
        assert stride[0] == 4  # only setup for stride==4
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=7,
            stride=stride,
            padding=3,
        )
        self.norm = norm_layer(out_chs)

    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        x = F.pad(x, (0, (self.stride[1] - W % self.stride[1]) % self.stride[1]))
        x = F.pad(x, (0, 0, 0, (self.stride[0] - H % self.stride[0]) % self.stride[0]))
        x = self.conv(x)
        x = self.norm(x)
        return x


class Downsample(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            norm_layer=LayerNorm2d,
    ):
        super().__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs

        self.norm = norm_layer(in_chs)
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=2,
            stride=2,
            padding=0,
        )

    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = F.pad(x, (0, (2 - W % 2) % 2))
        x = F.pad(x, (0, 0, 0, (2 - H % 2) % 2))
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class ChannelBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            ffn=True,
            cpe_act=False,
    ):
        super().__init__()

        self.cpe1 = ConvPosEnc(dim=dim, k=3, act=cpe_act)
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2 = ConvPosEnc(dim=dim, k=3, act=cpe_act)

        if self.ffn:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
            )
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        else:
            self.norm2 = None
            self.mlp = None
            self.drop_path2 = None

    def forward(self, x: Tensor):
        B, C, H, W = x.shape

        x = self.cpe1(x).flatten(2).transpose(1, 2)

        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path1(cur)

        x = self.cpe2(x.transpose(1, 2).view(B, C, H, W))

        if self.mlp is not None:
            x = x.flatten(2).transpose(1, 2)
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
            x = x.transpose(1, 2).view(B, C, H, W)

        return x


def window_partition(x: Tensor, window_size: Tuple[int, int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows: Tensor, window_size: Tuple[int, int], H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor):
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = self.softmax(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SpatialBlock(nn.Module):
    r""" Windows Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self,
            dim,
            num_heads,
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            ffn=True,
            cpe_act=False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = to_2tuple(window_size)
        self.mlp_ratio = mlp_ratio

        self.cpe1 = ConvPosEnc(dim=dim, k=3, act=cpe_act)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = ConvPosEnc(dim=dim, k=3, act=cpe_act)
        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
            )
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        else:
            self.norm2 = None
            self.mlp = None
            self.drop_path1 = None

    def forward(self, x: Tensor):
        B, C, H, W = x.shape

        shortcut = self.cpe1(x).flatten(2).transpose(1, 2)

        x = self.norm1(shortcut)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # if pad_r > 0 or pad_b > 0:
        x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path1(x)

        x = self.cpe2(x.transpose(1, 2).view(B, C, H, W))

        if self.mlp is not None:
            x = x.flatten(2).transpose(1, 2)
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
            x = x.transpose(1, 2).view(B, C, H, W)

        return x


class DaVitStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            depth=1,
            downsample=True,
            attn_types=('spatial', 'channel'),
            num_heads=3,
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rates=(0, 0),
            norm_layer=LayerNorm2d,
            norm_layer_cl=nn.LayerNorm,
            ffn=True,
            cpe_act=False
    ):
        super().__init__()

        self.grad_checkpointing = False

        # downsample embedding layer at the beginning of each stage
        if downsample:
            self.downsample = Downsample(in_chs, out_chs, norm_layer=norm_layer)
        else:
            self.downsample = nn.Identity()

        '''
         repeating alternating attention blocks in each stage
         default: (spatial -> channel) x depth
         
         potential opportunity to integrate with a more general version of ByobNet/ByoaNet
         since the logic is similar
        '''
        stage_blocks = []
        for block_idx in range(depth):
            dual_attention_block = []
            for attn_idx, attn_type in enumerate(attn_types):
                if attn_type == 'spatial':
                    dual_attention_block.append(SpatialBlock(
                        dim=out_chs,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=drop_path_rates[block_idx],
                        norm_layer=norm_layer_cl,
                        ffn=ffn,
                        cpe_act=cpe_act,
                        window_size=window_size,
                    ))
                elif attn_type == 'channel':
                    dual_attention_block.append(ChannelBlock(
                        dim=out_chs,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=drop_path_rates[block_idx],
                        norm_layer=norm_layer_cl,
                        ffn=ffn,
                        cpe_act=cpe_act
                    ))
            stage_blocks.append(nn.Sequential(*dual_attention_block))
        self.blocks = nn.Sequential(*stage_blocks)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, x: Tensor):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class DaVit(nn.Module):
    r""" DaViT
        A PyTorch implementation of `DaViT: Dual Attention Vision Transformers`  - https://arxiv.org/abs/2204.03645
        Supports arbitrary input sizes and pyramid feature extraction
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks in each stage. Default: (1, 1, 3, 1)
        embed_dims (tuple(int)): Patch embedding dimension. Default: (96, 192, 384, 768)
        num_heads (tuple(int)): Number of attention heads in different layers. Default: (3, 6, 12, 24)
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(
            self,
            in_chans=3,
            depths=(1, 1, 3, 1),
            embed_dims=(96, 192, 384, 768),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer='layernorm2d',
            norm_layer_cl='layernorm',
            norm_eps=1e-5,
            attn_types=('spatial', 'channel'),
            ffn=True,
            cpe_act=False,
            drop_rate=0.,
            drop_path_rate=0.,
            num_classes=1000,
            global_pool='avg',
            head_norm_first=False,
    ):
        super().__init__()
        num_stages = len(embed_dims)
        assert num_stages == len(num_heads) == len(depths)
        norm_layer = partial(get_norm_layer(norm_layer), eps=norm_eps)
        norm_layer_cl = partial(get_norm_layer(norm_layer_cl), eps=norm_eps)
        self.num_classes = num_classes
        self.num_features = embed_dims[-1]
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []

        self.stem = Stem(in_chans, embed_dims[0], norm_layer=norm_layer)
        in_chs = embed_dims[0]

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        for stage_idx in range(num_stages):
            out_chs = embed_dims[stage_idx]
            stage = DaVitStage(
                in_chs,
                out_chs,
                depth=depths[stage_idx],
                downsample=stage_idx > 0,
                attn_types=attn_types,
                num_heads=num_heads[stage_idx],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path_rates=dpr[stage_idx],
                norm_layer=norm_layer,
                norm_layer_cl=norm_layer_cl,
                ffn=ffn,
                cpe_act=cpe_act,
            )
            in_chs = out_chs
            stages.append(stage)
            self.feature_info += [dict(num_chs=out_chs, reduction=2, module=f'stages.{stage_idx}')]

        self.stages = nn.Sequential(*stages)

        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default DaViT order, similar to ConvNeXt
        # FIXME generalize this structure to ClassifierHead
        if head_norm_first:
            self.norm_pre = norm_layer(self.num_features)
            self.head = ClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
            )
        else:
            self.norm_pre = nn.Identity()
            self.head = NormMlpClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
                norm_layer=norm_layer,
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
        for stage in self.stages:
            stage.set_grad_checkpointing(enable=enable)

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.head.reset(num_classes, global_pool=global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.head.global_pool(x)
        x = self.head.norm(x)
        x = self.head.flatten(x)
        x = self.head.drop(x)
        return x if pre_logits else self.head.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    """ Remap MSFT checkpoints -> timm """
    if 'head.fc.weight' in state_dict:
        return state_dict  # non-MSFT checkpoint

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    import re
    out_dict = {}
    for k, v in state_dict.items():
        k = re.sub(r'patch_embeds.([0-9]+)', r'stages.\1.downsample', k)
        k = re.sub(r'main_blocks.([0-9]+)', r'stages.\1.blocks', k)
        k = k.replace('downsample.proj', 'downsample.conv')
        k = k.replace('stages.0.downsample', 'stem')
        k = k.replace('head.', 'head.fc.')
        k = k.replace('norms.', 'head.norm.')
        k = k.replace('cpe.0', 'cpe1')
        k = k.replace('cpe.1', 'cpe2')
        out_dict[k] = v
    return out_dict


def _create_davit(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 1, 3, 1))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        DaVit,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)

    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
        **kwargs
    }


# TODO contact authors to get larger pretrained models
default_cfgs = generate_default_cfgs({
    # official microsoft weights from https://github.com/dingmyu/davit
    'davit_tiny.msft_in1k': _cfg(
        hf_hub_id='timm/'),
    'davit_small.msft_in1k': _cfg(
        hf_hub_id='timm/'),
    'davit_base.msft_in1k': _cfg(
        hf_hub_id='timm/'),
    'davit_large': _cfg(),
    'davit_huge': _cfg(),
    'davit_giant': _cfg(),
})


@register_model
def davit_tiny(pretrained=False, **kwargs) -> DaVit:
    model_args = dict(depths=(1, 1, 3, 1), embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24))
    return _create_davit('davit_tiny', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def davit_small(pretrained=False, **kwargs) -> DaVit:
    model_args = dict(depths=(1, 1, 9, 1), embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24))
    return _create_davit('davit_small', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def davit_base(pretrained=False, **kwargs) -> DaVit:
    model_args = dict(depths=(1, 1, 9, 1), embed_dims=(128, 256, 512, 1024), num_heads=(4, 8, 16, 32))
    return _create_davit('davit_base', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def davit_large(pretrained=False, **kwargs) -> DaVit:
    model_args = dict(depths=(1, 1, 9, 1), embed_dims=(192, 384, 768, 1536), num_heads=(6, 12, 24, 48))
    return _create_davit('davit_large', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def davit_huge(pretrained=False, **kwargs) -> DaVit:
    model_args = dict(depths=(1, 1, 9, 1), embed_dims=(256, 512, 1024, 2048), num_heads=(8, 16, 32, 64))
    return _create_davit('davit_huge', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def davit_giant(pretrained=False, **kwargs) -> DaVit:
    model_args = dict(depths=(1, 1, 12, 3), embed_dims=(384, 768, 1536, 3072), num_heads=(12, 24, 48, 96))
    return _create_davit('davit_giant', pretrained=pretrained, **dict(model_args, **kwargs))
