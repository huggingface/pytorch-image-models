""" DaViT: Dual Attention Vision Transformers

As described in https://arxiv.org/abs/2204.03645

Input size invariant transformer architecture that combines channel and spacial
attention in each block. The attention mechanisms used are linear in complexity.

DaViT model defs and weights adapted from https://github.com/dingmyu/davit, original copyright below

"""
# Copyright (c) 2022 Mingyu Ding
# All rights reserved.
# This source code is licensed under the MIT license

# FIXME remove unused imports

import itertools
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, overload, Tuple, TypeVar, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.checkpoint as checkpoint

from .features import FeatureInfo
from .fx_features import register_notrace_function, register_notrace_module
from .helpers import build_model_with_cfg, pretrained_cfg_for_features
from .layers import DropPath, to_2tuple, trunc_normal_, SelectAdaptivePool2d, ClassifierHead, Mlp
from .pretrained import generate_default_cfgs
from .registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

__all__ = ['DaViT']


class ConvPosEnc(nn.Module):
    def __init__(self, dim : int, k : int=3, act : bool=False, normtype : str='none'):

        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),
                              to_2tuple(1),
                              to_2tuple(k // 2),
                              groups=dim)
        self.normtype = normtype
        self.norm = nn.Identity()
        if self.normtype == 'batch':
            self.norm = nn.BatchNorm2d(dim)
        elif self.normtype == 'layer':
            self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU() if act else nn.Identity()

    def forward(self, x : Tensor):
        B, C, H, W = x.shape

        #feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(x)
        if self.normtype == 'batch':
            feat = self.norm(feat).flatten(2).transpose(1, 2)
        elif self.normtype == 'layer':
            feat = self.norm(feat.flatten(2).transpose(1, 2))
        else:
            feat = feat.flatten(2).transpose(1, 2)
        x = x + self.activation(feat).transpose(1, 2).view(B, C, H, W)
        return x
        

@register_notrace_module
class PatchEmbedOld(nn.Module):
    """ Size-agnostic implementation of 2D image to patch embedding,
        allowing input size to be adjusted during model forward operation
    """

    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            overlapped=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if patch_size[0] == 4:
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=(7, 7),
                stride=patch_size,
                padding=(3, 3))
            self.norm = nn.LayerNorm(embed_dim)
        if patch_size[0] == 2:
            kernel = 3 if overlapped else 2
            pad = 1 if overlapped else 0
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=to_2tuple(kernel),
                stride=patch_size,
                padding=to_2tuple(pad))
            self.norm = nn.LayerNorm(in_chans)

    
    def forward(self, x : Tensor):
        B, C, H, W = x.shape
        if self.norm.normalized_shape[0] == self.in_chans:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)

        if self.norm.normalized_shape[0] == self.embed_dim:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

@register_notrace_module
class PatchEmbed(nn.Module):
    """ Size-agnostic implementation of 2D image to patch embedding,
        allowing input size to be adjusted during model forward operation
    """

    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            overlapped=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if patch_size[0] == 4:
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=(7, 7),
                stride=patch_size,
                padding=(3, 3))
            self.norm = nn.LayerNorm(embed_dim)
        if patch_size[0] == 2:
            kernel = 3 if overlapped else 2
            pad = 1 if overlapped else 0
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=to_2tuple(kernel),
                stride=patch_size,
                padding=to_2tuple(pad))
            self.norm = nn.LayerNorm(in_chans)

    
    def forward(self, x : Tensor):
        B, C, H, W = x.shape
        if self.norm.normalized_shape[0] == self.in_chans:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        
        x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        
        x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)

        if self.norm.normalized_shape[0] == self.embed_dim:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x
        
class ChannelAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x : Tensor):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
        

class ChannelBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super().__init__()

        self.cpe1 = ConvPosEnc(dim=dim, k=3, act=cpe_act)
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2 = ConvPosEnc(dim=dim, k=3, act=cpe_act)
        
        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)


    def forward(self, x : Tensor):

        B, C, H, W = x.shape

        x = self.cpe1(x).flatten(2).transpose(1, 2)
        
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)

        x = self.cpe2(x.transpose(1, 2).view(B, C, H, W)).flatten(2).transpose(1, 2)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        x = x.transpose(1, 2).view(B, C, H, W)
        
        return x

def window_partition(x : Tensor, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

@register_notrace_function # reason: int argument is a Proxy
def window_reverse(windows : Tensor, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x : Tensor):
        B_, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
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

    def __init__(self, dim, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super().__init__()
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        self.cpe1 = ConvPosEnc(dim=dim, k=3, act=cpe_act)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2 = ConvPosEnc(dim=dim, k=3, act=cpe_act)
 
        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)


    def forward(self, x : Tensor):
        B, C, H, W = x.shape


        shortcut = self.cpe1(x).flatten(2).transpose(1, 2)
        x = self.norm1(shortcut)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        #if pad_r > 0 or pad_b > 0:
        x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        x = self.cpe2(x.transpose(1, 2).view(B, C, H, W)).flatten(2).transpose(1, 2)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            
        x = x.transpose(1, 2).view(B, C, H, W)
        
        return x

class SpatialBlockOld(nn.Module):
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

    def __init__(self, dim, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super().__init__()
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        self.cpe1 = ConvPosEnc(dim=dim, k=3, act=cpe_act)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2 = ConvPosEnc(dim=dim, k=3, act=cpe_act)
 
        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)


    def forward(self, x : Tensor, size: Tuple[int, int]):

        H, W = size
        B, L, C = x.shape

        shortcut = self.cpe1(x, size)
        x = self.norm1(shortcut)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        #if pad_r > 0 or pad_b > 0:
        x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        x = self.cpe2(x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size
        
class DaViTStage(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        depth = 1,
        patch_size = 4,
        overlapped_patch = False,
        attention_types = ('spatial', 'channel'),
        num_heads = 3,
        window_size = 7,
        mlp_ratio = 4,
        qkv_bias = True,
        drop_path_rates = (0, 0),
        norm_layer = nn.LayerNorm,
        ffn = True,
        cpe_act = False
    ):
        super().__init__()

        self.grad_checkpointing = False
        
        # patch embedding layer at the beginning of each stage
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chs,
            embed_dim=out_chs,
            overlapped=overlapped_patch
        )
        '''
         repeating alternating attention blocks in each stage
         default: (spatial -> channel) x depth
         
         potential opportunity to integrate with a more general version of ByobNet/ByoaNet
         since the logic is similar
        '''
        stage_blocks = []
        for block_idx in range(depth):
        
            dual_attention_block = []
            
            for attention_id, attention_type in enumerate(attention_types):
                if attention_type == 'spatial':
                    dual_attention_block.append(SpatialBlock(
                        dim=out_chs,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=drop_path_rates[len(attention_types) * block_idx + attention_id],
                        norm_layer=nn.LayerNorm,
                        ffn=ffn,
                        cpe_act=cpe_act,
                        window_size=window_size,
                    ))
                elif attention_type == 'channel':
                    dual_attention_block.append(ChannelBlock(
                        dim=out_chs,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=drop_path_rates[len(attention_types) * block_idx + attention_id],
                        norm_layer=nn.LayerNorm,
                        ffn=ffn,
                        cpe_act=cpe_act
                    ))
            
            stage_blocks.append(nn.Sequential(*dual_attention_block))
            
        self.blocks = nn.Sequential(*stage_blocks)
        
    def forward(self, x : Tensor):
        x = self.patch_embed(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class DaViT(nn.Module):
    r""" DaViT
        A PyTorch implementation of `DaViT: Dual Attention Vision Transformers`  - https://arxiv.org/abs/2204.03645
        Supports arbitrary input sizes and pyramid feature extraction
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks in each stage. Default: (1, 1, 3, 1)
        patch_size (int | tuple(int)): Patch size. Default: 4
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
        patch_size=4,
        embed_dims=(96, 192, 384, 768),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        attention_types=('spatial', 'channel'),
        ffn=True,
        overlapped_patch=False,
        cpe_act=False,
        drop_rate=0.,
        attn_drop_rate=0.,
        num_classes=1000,
        global_pool='avg'
    ):
        super().__init__()

        architecture = [[index] * item for index, item in enumerate(depths)]
        self.architecture = architecture
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_stages = len(self.embed_dims)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(attention_types) * len(list(itertools.chain(*self.architecture))))]
        assert self.num_stages == len(self.num_heads) == (sorted(list(itertools.chain(*self.architecture)))[-1] + 1)
        
        self.num_classes = num_classes
        self.num_features = embed_dims[-1]
        self.drop_rate=drop_rate
        self.grad_checkpointing = False
        self.feature_info = []

        stages = []
        
        for stage_id in range(self.num_stages):
            stage_drop_rates = dpr[len(attention_types) * sum(depths[:stage_id]):len(attention_types) * sum(depths[:stage_id + 1])]

            stage = DaViTStage(
                in_chans if stage_id == 0 else embed_dims[stage_id - 1],
                embed_dims[stage_id],
                depth = depths[stage_id],
                patch_size = patch_size if stage_id == 0 else 2,
                overlapped_patch = overlapped_patch,
                attention_types = attention_types,
                num_heads = num_heads[stage_id],
                window_size = window_size,
                mlp_ratio = mlp_ratio,
                qkv_bias = qkv_bias,
                drop_path_rates = stage_drop_rates,
                norm_layer = nn.LayerNorm,
                ffn = ffn,
                cpe_act = cpe_act
            )
            
            stages.append(stage)
            self.feature_info += [dict(num_chs=self.embed_dims[stage_id], reduction=2, module=f'stages.{stage_id}')]
        
        
        self.stages = nn.Sequential(*stages)
        
        self.norms = norm_layer(self.num_features)
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
 
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
        
    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc
        
    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is None:
            global_pool = self.head.global_pool.pool_type
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stages(x)
        # take final feature and norm
        x = self.norms(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #H, W = sizes[-1]
        #x = x.view(-1, H, W, self.embed_dims[-1]).permute(0, 3, 1, 2).contiguous()
        return x
    
    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)
        
    def forward_classifier(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
        
    def forward(self, x):
        return self.forward_classifier(x)

def checkpoint_filter_fn(state_dict, model):
    """ Remap MSFT checkpoints -> timm """
    if 'head.norm.weight' in state_dict:
        return state_dict  # non-MSFT checkpoint
    
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    import re
    out_dict = {}
    for k, v in state_dict.items():
        k = re.sub(r'patch_embeds.([0-9]+)', r'stages.\1.patch_embed', k)
        k = re.sub(r'main_blocks.([0-9]+)', r'stages.\1.blocks', k)
        k = k.replace('head.', 'head.fc.')
        k = k.replace('cpe.0', 'cpe1')
        k = k.replace('cpe.1', 'cpe2')
        out_dict[k] = v
    return out_dict
    

def _create_davit(variant, pretrained=False, **kwargs):

    

    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 1, 3, 1))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        DaViT,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)

    return model
    
    

def _cfg(url='', **kwargs): # not sure how this should be set up
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stages.0.patch_embed.proj', 'classifier': 'head.fc',
        **kwargs
    }



default_cfgs = generate_default_cfgs({
    # official microsoft weights from https://github.com/dingmyu/davit
    'davit_tiny.msft_in1k': _cfg(
        url="https://github.com/fffffgggg54/pytorch-image-models/releases/download/checkpoint/davit_tiny_ed28dd55.pth.tar"),
    'davit_small.msft_in1k': _cfg(
        url="https://github.com/fffffgggg54/pytorch-image-models/releases/download/checkpoint/davit_small_d1ecf281.pth.tar"),
    'davit_base.msft_in1k': _cfg(
        url="https://github.com/fffffgggg54/pytorch-image-models/releases/download/checkpoint/davit_base_67d9ac26.pth.tar"),
    'davit_large': _cfg(),
    'davit_huge': _cfg(),
    'davit_giant': _cfg(),
})



@register_model
def davit_tiny(pretrained=False, **kwargs):
    model_kwargs = dict(depths=(1, 1, 3, 1), embed_dims=(96, 192, 384, 768),
    num_heads=(3, 6, 12, 24), **kwargs)
    return _create_davit('davit_tiny', pretrained=pretrained, **model_kwargs)
    
@register_model
def davit_small(pretrained=False, **kwargs):
    model_kwargs = dict(depths=(1, 1, 9, 1), embed_dims=(96, 192, 384, 768),
    num_heads=(3, 6, 12, 24), **kwargs)
    return _create_davit('davit_small', pretrained=pretrained, **model_kwargs)
    
@register_model
def davit_base(pretrained=False, **kwargs):
    model_kwargs = dict(depths=(1, 1, 9, 1), embed_dims=(128, 256, 512, 1024),
    num_heads=(4, 8, 16, 32), **kwargs)
    return _create_davit('davit_base', pretrained=pretrained, **model_kwargs)


# TODO contact authors to get larger pretrained models
@register_model
def davit_large(pretrained=False, **kwargs):
    model_kwargs = dict(depths=(1, 1, 9, 1), embed_dims=(192, 384, 768, 1536),
    num_heads=(6, 12, 24, 48), **kwargs)
    return _create_davit('davit_large', pretrained=pretrained, **model_kwargs)
    
@register_model
def davit_huge(pretrained=False, **kwargs):
    model_kwargs = dict(depths=(1, 1, 9, 1), embed_dims=(256, 512, 1024, 2048),
    num_heads=(8, 16, 32, 64), **kwargs)
    return _create_davit('davit_huge', pretrained=pretrained, **model_kwargs)
    
@register_model
def davit_giant(pretrained=False, **kwargs):
    model_kwargs = dict(depths=(1, 1, 12, 3), embed_dims=(384, 768, 1536, 3072),
    num_heads=(12, 24, 48, 96), **kwargs)
    return _create_davit('davit_giant', pretrained=pretrained, **model_kwargs)
