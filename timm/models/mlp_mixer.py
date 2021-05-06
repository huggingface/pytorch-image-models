""" MLP-Mixer in PyTorch

Official JAX impl: https://github.com/google-research/vision_transformer/blob/linen/vit_jax/models_mixer.py

Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601

@article{tolstikhin2021,
  title={MLP-Mixer: An all-MLP Architecture for Vision},
  author={Tolstikhin, Ilya and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner,
        Thomas and Yung, Jessica and Keysers, Daniel and Uszkoreit, Jakob and Lucic, Mario and Dosovitskiy, Alexey},
  journal={arXiv preprint arXiv:2105.01601},
  year={2021}
}

A thank you to paper authors for releasing code and weights.

Hacked together by / Copyright 2021 Ross Wightman
"""
import math
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg, overlay_external_default_cfg
from .layers import PatchEmbed, Mlp, GluMlp, DropPath, lecun_normal_
from .registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 0.875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'stem.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = dict(
    mixer_s32_224=_cfg(),
    mixer_s16_224=_cfg(),
    mixer_s16_glu_224=_cfg(),
    mixer_b32_224=_cfg(),
    mixer_b16_224=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224-76587d61.pth',
    ),
    mixer_b16_224_in21k=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224_in21k-617b3de2.pth',
        num_classes=21843
    ),
    mixer_l32_224=_cfg(),
    mixer_l16_224=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224-92f9adc4.pth',
    ),
    mixer_l16_224_in21k=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224_in21k-846aa33c.pth',
        num_classes=21843
    ),
)


class MixerBlock(nn.Module):

    def __init__(
            self, dim, seq_len, tokens_dim, channels_dim,
            mlp_layer=Mlp, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class MlpMixer(nn.Module):

    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            hidden_dim=512,
            tokens_dim=256,
            channels_dim=2048,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop=0.,
            drop_path=0.,
            nlhb=False,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.stem = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=hidden_dim)
        # FIXME drop_path (stochastic depth scaling rule?)
        self.blocks = nn.Sequential(*[
            MixerBlock(
                hidden_dim, self.stem.num_patches, tokens_dim, channels_dim,
                mlp_layer=mlp_layer, norm_layer=norm_layer, act_layer=act_layer, drop=drop, drop_path=drop_path)
            for _ in range(num_blocks)])
        self.norm = norm_layer(hidden_dim)
        self.head = nn.Linear(hidden_dim, self.num_classes)  # zero init

        self.init_weights(nlhb=nlhb)

    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        for n, m in self.named_modules():
            _init_weights(m, n, head_bias=head_bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


def _init_weights(m, n: str, head_bias: float = 0.):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    if isinstance(m, nn.Linear):
        if n.startswith('head'):
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, head_bias)
        else:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                if 'mlp' in n:
                    nn.init.normal_(m.bias, std=1e-6)
                else:
                    nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def _create_mixer(variant, pretrained=False, default_cfg=None, **kwargs):
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-2:]
    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)

    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for MLP-Mixer models.')

    model = build_model_with_cfg(
        MlpMixer, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        **kwargs)

    return model


@register_model
def mixer_s32_224(pretrained=False, **kwargs):
    """ Mixer-S/32 224x224
    """
    model_args = dict(patch_size=32, num_blocks=8, hidden_dim=512, tokens_dim=256, channels_dim=2048, **kwargs)
    model = _create_mixer('mixer_s32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_s16_224(pretrained=False, **kwargs):
    """ Mixer-S/16 224x224
    """
    model_args = dict(patch_size=16, num_blocks=8, hidden_dim=512, tokens_dim=256, channels_dim=2048, **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_s16_glu_224(pretrained=False, **kwargs):
    """ Mixer-S/16 224x224
    """
    model_args = dict(
        patch_size=16, num_blocks=8, hidden_dim=512, tokens_dim=256, channels_dim=1536,
        mlp_layer=GluMlp, act_layer=nn.SiLU, **kwargs)
    model = _create_mixer('mixer_s16_glu_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b32_224(pretrained=False, **kwargs):
    """ Mixer-B/32 224x224
    """
    model_args = dict(patch_size=32, num_blocks=12, hidden_dim=768, tokens_dim=384, channels_dim=3072, **kwargs)
    model = _create_mixer('mixer_b32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b16_224(pretrained=False, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    """
    model_args = dict(patch_size=16, num_blocks=12, hidden_dim=768, tokens_dim=384, channels_dim=3072, **kwargs)
    model = _create_mixer('mixer_b16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b16_224_in21k(pretrained=False, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-21k pretrained weights.
    """
    model_args = dict(patch_size=16, num_blocks=12, hidden_dim=768, tokens_dim=384, channels_dim=3072, **kwargs)
    model = _create_mixer('mixer_b16_224_in21k', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_l32_224(pretrained=False, **kwargs):
    """ Mixer-L/32 224x224.
    """
    model_args = dict(patch_size=32, num_blocks=24, hidden_dim=1024, tokens_dim=512, channels_dim=4096, **kwargs)
    model = _create_mixer('mixer_l32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_l16_224(pretrained=False, **kwargs):
    """ Mixer-L/16 224x224. ImageNet-1k pretrained weights.
    """
    model_args = dict(patch_size=16, num_blocks=24, hidden_dim=1024, tokens_dim=512, channels_dim=4096, **kwargs)
    model = _create_mixer('mixer_l16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_l16_224_in21k(pretrained=False, **kwargs):
    """ Mixer-L/16 224x224. ImageNet-21k pretrained weights.
    """
    model_args = dict(patch_size=16, num_blocks=24, hidden_dim=1024, tokens_dim=512, channels_dim=4096, **kwargs)
    model = _create_mixer('mixer_l16_224_in21k', pretrained=pretrained, **model_args)
    return model
