""" ConViT Model

@article{d2021convit,
  title={ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases},
  author={d'Ascoli, St{\'e}phane and Touvron, Hugo and Leavitt, Matthew and Morcos, Ari and Biroli, Giulio and Sagun, Levent},
  journal={arXiv preprint arXiv:2103.10697},
  year={2021}
}

Paper link: https://arxiv.org/abs/2103.10697
Original code: https://github.com/facebookresearch/convit, original copyright below

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
"""
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
'''These modules are adapted from those of timm, see
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''

from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_, PatchEmbed, Mlp
from ._builder import build_model_with_cfg
from ._features_fx import register_notrace_module
from ._registry import register_model
from .vision_transformer_hybrid import HybridEmbed


__all__ = ['ConViT']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'fixed_input_size': True,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # ConViT
    'convit_tiny': _cfg(
        url="https://dl.fbaipublicfiles.com/convit/convit_tiny.pth"),
    'convit_small': _cfg(
        url="https://dl.fbaipublicfiles.com/convit/convit_small.pth"),
    'convit_base': _cfg(
        url="https://dl.fbaipublicfiles.com/convit/convit_base.pth")
}


@register_notrace_module  # reason: FX can't symbolically trace control flow in forward method
class GPSA(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
            locality_strength=1.,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.locality_strength = locality_strength

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        self.rel_indices: torch.Tensor = torch.zeros(1, 1, 1, 3)  # silly torchscript hack, won't work with None

    def forward(self, x):
        B, N, C = x.shape
        if self.rel_indices is None or self.rel_indices.shape[1] != N:
            self.rel_indices = self.get_rel_indices(N)
        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        pos_score = self.rel_indices.expand(B, -1, -1, -1)
        pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1, -1, 1, 1)
        attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map=False):
        attn_map = self.get_attention(x).mean(0)  # average over batch
        distances = self.rel_indices.squeeze()[:, :, -1] ** .5
        dist = torch.einsum('nm,hnm->h', (distances, attn_map)) / distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist

    def local_init(self):
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1  # max(1,1/locality_strength**.5)

        kernel_size = int(self.num_heads ** .5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.pos_proj.weight.data[position, 2] = -1
                self.pos_proj.weight.data[position, 1] = 2 * (h1 - center) * locality_distance
                self.pos_proj.weight.data[position, 0] = 2 * (h2 - center) * locality_distance
        self.pos_proj.weight.data *= self.locality_strength

    def get_rel_indices(self, num_patches: int) -> torch.Tensor:
        img_size = int(num_patches ** .5)
        rel_indices = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)
        device = self.qk.weight.device
        return rel_indices.to(device)


class MHSA(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def get_attention_map(self, x, return_map=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = int(N ** .5)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        distances = indd ** .5
        distances = distances.to(x.device)

        dist = torch.einsum('nm,hnm->h', (distances, attn_map)) / N
        if return_map:
            return dist, attn_map
        else:
            return dist

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            use_gpsa=True,
            locality_strength=1.,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_gpsa = use_gpsa
        if self.use_gpsa:
            self.attn = GPSA(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                locality_strength=locality_strength,
            )
        else:
            self.attn = MHSA(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            global_pool='token',
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=False,
            drop_rate=0.,
            pos_drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            hybrid_backbone=None,
            norm_layer=nn.LayerNorm,
            local_up_to_layer=3,
            locality_strength=1.,
            use_pos_embed=True,
    ):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        embed_dim *= num_heads
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.local_up_to_layer = local_up_to_layer
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                use_gpsa=i < local_up_to_layer,
                locality_strength=locality_strength,
            ) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        for n, m in self.named_modules():
            if hasattr(m, 'local_init'):
                m.local_init()

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
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'token', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        for u, blk in enumerate(self.blocks):
            if u == self.local_up_to_layer:
                x = torch.cat((cls_tokens, x), dim=1)
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_convit(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    return build_model_with_cfg(ConViT, variant, pretrained, **kwargs)


@register_model
def convit_tiny(pretrained=False, **kwargs):
    model_args = dict(
        local_up_to_layer=10, locality_strength=1.0, embed_dim=48,
        num_heads=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = _create_convit(variant='convit_tiny', pretrained=pretrained, **model_args)
    return model


@register_model
def convit_small(pretrained=False, **kwargs):
    model_args = dict(
        local_up_to_layer=10, locality_strength=1.0, embed_dim=48,
        num_heads=9, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = _create_convit(variant='convit_small', pretrained=pretrained, **model_args)
    return model


@register_model
def convit_base(pretrained=False, **kwargs):
    model_args = dict(
        local_up_to_layer=10, locality_strength=1.0, embed_dim=48,
        num_heads=16, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = _create_convit(variant='convit_base', pretrained=pretrained, **model_args)
    return model
