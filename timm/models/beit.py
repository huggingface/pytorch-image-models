""" BEiT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)

Model from official source: https://github.com/microsoft/unilm/tree/master/beit

@inproceedings{beit,
title={{BEiT}: {BERT} Pre-Training of Image Transformers},
author={Hangbo Bao and Li Dong and Songhao Piao and Furu Wei},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=p-BhZSz59o4}
}

BEiT-v2 from https://github.com/microsoft/unilm/tree/master/beit2

@article{beitv2,
title={{BEiT v2}: Masked Image Modeling with Vector-Quantized Visual Tokenizers},
author={Zhiliang Peng and Li Dong and Hangbo Bao and Qixiang Ye and Furu Wei},
year={2022},
eprint={2208.06366},
archivePrefix={arXiv},
primaryClass={cs.CV}
}

At this point only the 1k fine-tuned classification weights and model configs have been added,
see original source above for pre-training models and procedure.

Modifications by / Copyright 2021 Ross Wightman, original copyrights below
"""
# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import math
from functools import partial
from typing import Callable, Final, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, Mlp, SwiGLU, LayerNorm, DropPath, trunc_normal_, use_fused_attn

from ._builder import build_model_with_cfg
from ._registry import generate_default_cfgs, register_model
from .vision_transformer import checkpoint_filter_fn

__all__ = ['Beit']


def gen_relative_position_index(window_size: Tuple[int, int]) -> torch.Tensor:
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
    # cls to token & token 2 cls & cls to cls
    # get pair-wise relative position index for each token inside the window
    window_area = window_size[0] * window_size[1]
    coords = torch.stack(torch.meshgrid(
        [torch.arange(window_size[0]),
         torch.arange(window_size[1])]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = torch.zeros(size=(window_area + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            window_size: Optional[Tuple[int, int]] = None,
            attn_head_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            self.register_buffer("relative_position_index", gen_relative_position_index(window_size))
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_rel_pos_bias(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, shared_rel_pos_bias: Optional[torch.Tensor] = None):
        B, N, C = x.shape

        qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim

        if self.fused_attn:
            rel_pos_bias = None
            if self.relative_position_bias_table is not None:
                rel_pos_bias = self._get_rel_pos_bias()
                if shared_rel_pos_bias is not None:
                    rel_pos_bias = rel_pos_bias + shared_rel_pos_bias
            elif shared_rel_pos_bias is not None:
                rel_pos_bias = shared_rel_pos_bias

            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=rel_pos_bias,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            if self.relative_position_bias_table is not None:
                attn = attn + self._get_rel_pos_bias()
            if shared_rel_pos_bias is not None:
                attn = attn + shared_rel_pos_bias

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = False,
            mlp_ratio: float = 4.,
            scale_mlp: bool = False,
            swiglu_mlp: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            window_size: Optional[Tuple[int, int]] = None,
            attn_head_dim: Optional[int] = None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        if swiglu_mlp:
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if init_values:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, shared_rel_pos_bias: Optional[torch.Tensor] = None):
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_distance, num_heads))
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.register_buffer("relative_position_index", gen_relative_position_index(window_size))

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_area + 1, self.window_area + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class Beit(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            qkv_bias: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            use_abs_pos_emb: bool = True,
            use_rel_pos_bias: bool = False,
            use_shared_rel_pos_bias: bool = False,
            head_init_scale: float = 0.001,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1
        self.grad_checkpointing = False

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.grid_size,
                num_heads=num_heads,
            )
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                mlp_ratio=mlp_ratio,
                scale_mlp=scale_mlp,
                swiglu_mlp=swiglu_mlp,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=self.patch_embed.grid_size if use_rel_pos_bias else None,
            )
            for i in range(depth)])

        use_fc_norm = self.global_pool == 'avg'
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed|rel_pos_bias',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, shared_rel_pos_bias=rel_pos_bias)
            else:
                x = blk(x, shared_rel_pos_bias=rel_pos_bias)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'beit_base_patch16_224.in22k_ft_in22k_in1k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22kto1k.pth',
        hf_hub_id='timm/'),
    'beit_base_patch16_384.in22k_ft_in22k_in1k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_384_pt22k_ft22kto1k.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
    'beit_base_patch16_224.in22k_ft_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22k.pth',
        hf_hub_id='timm/',
        num_classes=21841,
    ),
    'beit_large_patch16_224.in22k_ft_in22k_in1k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22kto1k.pth',
        hf_hub_id='timm/'),
    'beit_large_patch16_384.in22k_ft_in22k_in1k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_384_pt22k_ft22kto1k.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
    'beit_large_patch16_512.in22k_ft_in22k_in1k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_512_pt22k_ft22kto1k.pth',
        hf_hub_id='timm/',
        input_size=(3, 512, 512), crop_pct=1.0,
    ),
    'beit_large_patch16_224.in22k_ft_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth',
        hf_hub_id='timm/',
        num_classes=21841,
    ),

    'beitv2_base_patch16_224.in1k_ft_in22k_in1k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21kto1k.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_base_patch16_224.in1k_ft_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21k.pth',
        hf_hub_id='timm/',
        num_classes=21841, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_large_patch16_224.in1k_ft_in22k_in1k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21kto1k.pth',
        hf_hub_id='timm/',
        crop_pct=0.95, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_large_patch16_224.in1k_ft_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth',
        hf_hub_id='timm/',
        num_classes=21841, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
})


def _beit_checkpoint_filter_fn(state_dict, model):
    if 'module' in state_dict:
        # beit v2 didn't strip module
        state_dict = state_dict['module']
    return checkpoint_filter_fn(state_dict, model)


def _create_beit(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for BEiT models.')

    model = build_model_with_cfg(
        Beit, variant, pretrained,
        # FIXME an updated filter fn needed to interpolate rel pos emb if fine tuning to diff model sizes
        pretrained_filter_fn=_beit_checkpoint_filter_fn,
        **kwargs)
    return model


@register_model
def beit_base_patch16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1)
    model = _create_beit('beit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def beit_base_patch16_384(pretrained=False, **kwargs):
    model_args = dict(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1)
    model = _create_beit('beit_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def beit_large_patch16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5)
    model = _create_beit('beit_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def beit_large_patch16_384(pretrained=False, **kwargs):
    model_args = dict(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5)
    model = _create_beit('beit_large_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def beit_large_patch16_512(pretrained=False, **kwargs):
    model_args = dict(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5)
    model = _create_beit('beit_large_patch16_512', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def beitv2_base_patch16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5)
    model = _create_beit('beitv2_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def beitv2_large_patch16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5)
    model = _create_beit('beitv2_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
