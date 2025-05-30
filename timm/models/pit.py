""" Pooling-based Vision Transformer (PiT) in PyTorch

A PyTorch implement of Pooling-based Vision Transformers as described in
'Rethinking Spatial Dimensions of Vision Transformers' - https://arxiv.org/abs/2103.16302

This code was adapted from the original version at https://github.com/naver-ai/pit, original copyright below.

Modifications for timm by / Copyright 2020 Ross Wightman
"""
# PiT
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

import math
import re
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import trunc_normal_, to_2tuple
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._registry import register_model, generate_default_cfgs
from .vision_transformer import Block


__all__ = ['PoolingVisionTransformer']  # model_registry will add each entrypoint fn to this


class SequentialTuple(nn.Sequential):
    """ This module exists to work around torchscript typing issues list -> list"""
    def __init__(self, *args):
        super(SequentialTuple, self).__init__(*args)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            base_dim,
            depth,
            heads,
            mlp_ratio,
            pool=None,
            proj_drop=.0,
            attn_drop=.0,
            drop_path_prob=None,
            norm_layer=None,
    ):
        super(Transformer, self).__init__()
        embed_dim = base_dim * heads

        self.pool = pool
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path_prob[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(depth)])

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, cls_tokens = x
        token_length = cls_tokens.shape[1]
        if self.pool is not None:
            x, cls_tokens = self.pool(x, cls_tokens)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)
        x = self.blocks(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = x.transpose(1, 2).reshape(B, C, H, W)

        return x, cls_tokens


class Pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride, padding_mode='zeros'):
        super(Pooling, self).__init__()

        self.conv = nn.Conv2d(
            in_feature,
            out_feature,
            kernel_size=stride + 1,
            padding=stride // 2,
            stride=stride,
            padding_mode=padding_mode,
            groups=in_feature,
        )
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        cls_token = self.fc(cls_token)
        return x, cls_token


class ConvEmbedding(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            img_size: int = 224,
            patch_size: int = 16,
            stride: int = 8,
            padding: int = 0,
    ):
        super(ConvEmbedding, self).__init__()
        padding = padding
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.height = math.floor((self.img_size[0] + 2 * padding - self.patch_size[0]) / stride + 1)
        self.width = math.floor((self.img_size[1] + 2 * padding - self.patch_size[1]) / stride + 1)
        self.grid_size = (self.height, self.width)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=patch_size,
            stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolingVisionTransformer(nn.Module):
    """ Pooling-based Vision Transformer

    A PyTorch implement of 'Rethinking Spatial Dimensions of Vision Transformers'
        - https://arxiv.org/abs/2103.16302
    """
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            stride: int = 8,
            stem_type: str = 'overlap',
            base_dims: Sequence[int] = (48, 48, 48),
            depth: Sequence[int] = (2, 6, 4),
            heads: Sequence[int] = (2, 4, 8),
            mlp_ratio: float = 4,
            num_classes=1000,
            in_chans=3,
            global_pool='token',
            distilled=False,
            drop_rate=0.,
            pos_drop_drate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
    ):
        super(PoolingVisionTransformer, self).__init__()
        assert global_pool in ('token',)

        self.base_dims = base_dims
        self.heads = heads
        embed_dim = base_dims[0] * heads[0]
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_tokens = 2 if distilled else 1
        self.feature_info = []

        self.patch_embed = ConvEmbedding(in_chans, embed_dim, img_size, patch_size, stride)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, self.patch_embed.height, self.patch_embed.width))
        self.cls_token = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_drate)

        transformers = []
        # stochastic depth decay rule
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depth)).split(depth)]
        prev_dim = embed_dim
        for i in range(len(depth)):
            pool = None
            embed_dim = base_dims[i] * heads[i]
            if i > 0:
                pool = Pooling(
                    prev_dim,
                    embed_dim,
                    stride=2,
                )
            transformers += [Transformer(
                base_dims[i],
                depth[i],
                heads[i],
                mlp_ratio,
                pool=pool,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_prob=dpr[i],
            )]
            prev_dim = embed_dim
            self.feature_info += [dict(num_chs=prev_dim, reduction=(stride - 1) * 2**i, module=f'transformers.{i}')]

        self.transformers = SequentialTuple(*transformers)
        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim

        # Classifier head
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        self.distilled_training = False  # must set this True to train w/ distillation token

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    def get_classifier(self) -> nn.Module:
        if self.head_dist is not None:
            return self.head, self.head_dist
        else:
            return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.head_dist is not None:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

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
        take_indices, max_index = feature_take_indices(len(self.transformers), indices)

        # forward pass
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        last_idx = len(self.transformers) - 1
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.transformers
        else:
            stages = self.transformers[:max_index + 1]

        for feat_idx, stage in enumerate(stages):
            x, cls_tokens = stage((x, cls_tokens))
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        if feat_idx == last_idx:
            cls_tokens = self.norm(cls_tokens)

        return cls_tokens, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.transformers), indices)
        self.transformers = self.transformers[:max_index + 1]  # truncate blocks w/ stem as idx 0
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x, cls_tokens = self.transformers((x, cls_tokens))
        cls_tokens = self.norm(cls_tokens)
        return cls_tokens

    def forward_head(self, x, pre_logits: bool = False) -> torch.Tensor:
        if self.head_dist is not None:
            assert self.global_pool == 'token'
            x, x_dist = x[:, 0], x[:, 1]
            x = self.head_drop(x)
            x_dist = self.head_drop(x)
            if not pre_logits:
                x = self.head(x)
                x_dist = self.head_dist(x_dist)
            if self.distilled_training and self.training and not torch.jit.is_scripting():
                # only return separate classification predictions when training in distilled mode
                return x, x_dist
            else:
                # during standard train / finetune, inference average the classifier predictions
                return (x + x_dist) / 2
        else:
            if self.global_pool == 'token':
                x = x[:, 0]
            x = self.head_drop(x)
            if not pre_logits:
                x = self.head(x)
            return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    """ preprocess checkpoints """
    out_dict = {}
    p_blocks = re.compile(r'pools\.(\d)\.')
    for k, v in state_dict.items():
        # FIXME need to update resize for PiT impl
        # if k == 'pos_embed' and v.shape != model.pos_embed.shape:
        #     # To resize pos embedding when using model at different size from pretrained weights
        #     v = resize_pos_embed(v, model.pos_embed)
        k = p_blocks.sub(lambda exp: f'transformers.{int(exp.group(1)) + 1}.pool.', k)
        out_dict[k] = v
    return out_dict


def _create_pit(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(range(3))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        PoolingVisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(feature_cls='hook', out_indices=out_indices),
        **kwargs,
    )
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.conv', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # deit models (FB weights)
    'pit_ti_224.in1k': _cfg(hf_hub_id='timm/'),
    'pit_xs_224.in1k': _cfg(hf_hub_id='timm/'),
    'pit_s_224.in1k': _cfg(hf_hub_id='timm/'),
    'pit_b_224.in1k': _cfg(hf_hub_id='timm/'),
    'pit_ti_distilled_224.in1k': _cfg(
        hf_hub_id='timm/',
        classifier=('head', 'head_dist')),
    'pit_xs_distilled_224.in1k': _cfg(
        hf_hub_id='timm/',
        classifier=('head', 'head_dist')),
    'pit_s_distilled_224.in1k': _cfg(
        hf_hub_id='timm/',
        classifier=('head', 'head_dist')),
    'pit_b_distilled_224.in1k': _cfg(
        hf_hub_id='timm/',
        classifier=('head', 'head_dist')),
})


@register_model
def pit_b_224(pretrained=False, **kwargs) -> PoolingVisionTransformer:
    model_args = dict(
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
    )
    return _create_pit('pit_b_224', pretrained, **dict(model_args, **kwargs))


@register_model
def pit_s_224(pretrained=False, **kwargs) -> PoolingVisionTransformer:
    model_args = dict(
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
    )
    return _create_pit('pit_s_224', pretrained, **dict(model_args, **kwargs))


@register_model
def pit_xs_224(pretrained=False, **kwargs) -> PoolingVisionTransformer:
    model_args = dict(
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
    )
    return _create_pit('pit_xs_224', pretrained, **dict(model_args, **kwargs))


@register_model
def pit_ti_224(pretrained=False, **kwargs) -> PoolingVisionTransformer:
    model_args = dict(
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
    )
    return _create_pit('pit_ti_224', pretrained, **dict(model_args, **kwargs))


@register_model
def pit_b_distilled_224(pretrained=False, **kwargs) -> PoolingVisionTransformer:
    model_args = dict(
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        distilled=True,
    )
    return _create_pit('pit_b_distilled_224', pretrained, **dict(model_args, **kwargs))


@register_model
def pit_s_distilled_224(pretrained=False, **kwargs) -> PoolingVisionTransformer:
    model_args = dict(
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        distilled=True,
    )
    return _create_pit('pit_s_distilled_224', pretrained, **dict(model_args, **kwargs))


@register_model
def pit_xs_distilled_224(pretrained=False, **kwargs) -> PoolingVisionTransformer:
    model_args = dict(
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        distilled=True,
    )
    return _create_pit('pit_xs_distilled_224', pretrained, **dict(model_args, **kwargs))


@register_model
def pit_ti_distilled_224(pretrained=False, **kwargs) -> PoolingVisionTransformer:
    model_args = dict(
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        distilled=True,
    )
    return _create_pit('pit_ti_distilled_224', pretrained, **dict(model_args, **kwargs))
