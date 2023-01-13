""" DeiT - Data-efficient Image Transformers

DeiT model defs and weights from https://github.com/facebookresearch/deit, original copyright below

paper: `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

paper: `DeiT III: Revenge of the ViT` - https://arxiv.org/abs/2204.07118

Modifications copyright 2021, Ross Wightman
"""
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from functools import partial

import torch
from torch import nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.vision_transformer import VisionTransformer, trunc_normal_, checkpoint_filter_fn
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model

__all__ = ['VisionTransformerDistilled']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),

    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    'deit3_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_224_1k.pth'),
    'deit3_small_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_medium_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_medium_224_1k.pth'),
    'deit3_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_224_1k.pth'),
    'deit3_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_large_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_224_1k.pth'),
    'deit3_large_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_huge_patch14_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_huge_224_1k.pth'),

    'deit3_small_patch16_224_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_224_21k.pth',
        crop_pct=1.0),
    'deit3_small_patch16_384_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_medium_patch16_224_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_medium_224_21k.pth',
        crop_pct=1.0),
    'deit3_base_patch16_224_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth',
        crop_pct=1.0),
    'deit3_base_patch16_384_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_large_patch16_224_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_224_21k.pth',
        crop_pct=1.0),
    'deit3_large_patch16_384_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_huge_patch14_224_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_huge_224_21k_v1.pth',
        crop_pct=1.0),
}


class VisionTransformerDistilled(VisionTransformer):
    """ Vision Transformer w/ Distillation Token and Head

    Distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, *args, **kwargs):
        weight_init = kwargs.pop('weight_init', '')
        super().__init__(*args, **kwargs, weight_init='skip')
        assert self.global_pool in ('token',)

        self.num_prefix_tokens = 2
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.distilled_training = False  # must set this True to train w/ distillation token

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        trunc_normal_(self.dist_token, std=.02)
        super().init_weights(mode=mode)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed|dist_token',
            blocks=[
                (r'^blocks\.(\d+)', None),
                (r'^norm', (99999,))]  # final norm w/ last block
        )

    @torch.jit.ignore
    def get_classifier(self):
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def forward_features(self, x) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat((
            self.cls_token.expand(x.shape[0], -1, -1),
            self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False) -> torch.Tensor:
        if pre_logits:
            return (x[:, 0] + x[:, 1]) / 2
        x, x_dist = self.head(x[:, 0]), self.head_dist(x[:, 1])
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train / finetune, inference average the classifier predictions
            return (x + x_dist) / 2


def _create_deit(variant, pretrained=False, distilled=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    model_cls = VisionTransformerDistilled if distilled else VisionTransformer
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        **kwargs)
    return model


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_deit('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_deit('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_deit('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_deit('deit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_deit(
        'deit_tiny_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_deit(
        'deit_small_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_deit(
        'deit_base_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_deit(
        'deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def deit3_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 small model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_small_patch16_384(pretrained=False, **kwargs):
    """ DeiT-3 small model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_small_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_medium_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 medium model @ 224x224 (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_medium_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_base_patch16_384(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_large_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 large model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_large_patch16_384(pretrained=False, **kwargs):
    """ DeiT-3 large model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_huge_patch14_224(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_huge_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_small_patch16_224_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 small model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-21k pretrained weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_small_patch16_224_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_small_patch16_384_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 small model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-21k pretrained weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_small_patch16_384_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_medium_patch16_224_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 medium model @ 224x224 (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_medium_patch16_224_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_base_patch16_224_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-21k pretrained weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_base_patch16_224_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_base_patch16_384_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-21k pretrained weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_base_patch16_384_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_large_patch16_224_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 large model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-21k pretrained weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_large_patch16_224_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_large_patch16_384_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 large model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-21k pretrained weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_large_patch16_384_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_huge_patch14_224_in21ft1k(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-21k pretrained weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_huge_patch14_224_in21ft1k', pretrained=pretrained, **model_kwargs)
    return model
