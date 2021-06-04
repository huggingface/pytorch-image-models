""" MLP-Mixer, ResMLP, and gMLP in PyTorch

This impl originally based on MLP-Mixer paper.

Official JAX impl: https://github.com/google-research/vision_transformer/blob/linen/vit_jax/models_mixer.py

Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601

@article{tolstikhin2021,
  title={MLP-Mixer: An all-MLP Architecture for Vision},
  author={Tolstikhin, Ilya and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner,
        Thomas and Yung, Jessica and Keysers, Daniel and Uszkoreit, Jakob and Lucic, Mario and Dosovitskiy, Alexey},
  journal={arXiv preprint arXiv:2105.01601},
  year={2021}
}

Also supporting preliminary (not verified) implementations of ResMlp, gMLP, and possibly more...

Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
@misc{touvron2021resmlp,
      title={ResMLP: Feedforward networks for image classification with data-efficient training},
      author={Hugo Touvron and Piotr Bojanowski and Mathilde Caron and Matthieu Cord and Alaaeldin El-Nouby and
        Edouard Grave and Armand Joulin and Gabriel Synnaeve and Jakob Verbeek and Hervé Jégou},
      year={2021},
      eprint={2105.03404},
}

Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
@misc{liu2021pay,
      title={Pay Attention to MLPs},
      author={Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},
      year={2021},
      eprint={2105.08050},
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
from .layers import PatchEmbed, Mlp, GluMlp, GatedMlp, DropPath, lecun_normal_, to_2tuple
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
    # Mixer ImageNet-21K-P pretraining
    mixer_b16_224_miil_in21k=_cfg(
    url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/mixer_b16_224_miil_in21k.pth',
    mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    mixer_b16_224_miil=_cfg(
    url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/mixer_b16_224_miil.pth',
    mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),

    gmixer_12_224=_cfg(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    gmixer_24_224=_cfg(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    resmlp_12_224=_cfg(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    resmlp_24_224=_cfg(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    resmlp_36_224=_cfg(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    gmlp_ti16_224=_cfg(),
    gmlp_s16_224=_cfg(),
    gmlp_b16_224=_cfg(),
)


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0), mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)


class ResBlock(nn.Module):
    """ Residual MLP block w/ LayerScale and Affine 'norm'

    Based on: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=4, mlp_layer=Mlp, norm_layer=Affine,
            act_layer=nn.GELU, init_values=1e-4, drop=0., drop_path=0.):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.linear_tokens = nn.Linear(seq_len, seq_len)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, drop=drop)
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        return x


class SpatialGatingUnit(nn.Module):
    """ Spatial Gating Unit

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(self, dim, seq_len, norm_layer=nn.LayerNorm):
        super().__init__()
        gate_dim = dim // 2
        self.norm = norm_layer(gate_dim)
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(-1, -2))
        return u * v.transpose(-1, -2)


class SpatialGatingBlock(nn.Module):
    """ Residual Block w/ Spatial Gating

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=4, mlp_layer=GatedMlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm = norm_layer(dim)
        sgu = partial(SpatialGatingUnit, seq_len=seq_len)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, gate_layer=sgu, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
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
            mlp_ratio=(0.5, 4.0),
            block_layer=MixerBlock,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem_norm=False,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.stem = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=hidden_dim,
            norm_layer=norm_layer if stem_norm else None)
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = nn.Sequential(*[
            block_layer(
                hidden_dim, self.stem.num_patches, mlp_ratio, mlp_layer=mlp_layer, norm_layer=norm_layer,
                act_layer=act_layer, drop=drop_rate, drop_path=drop_path_rate)
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
        elif n.endswith('gate.proj'):
            nn.init.normal_(m.weight, std=1e-4)
            nn.init.ones_(m.bias)
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


def _create_mixer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for MLP-Mixer models.')

    model = build_model_with_cfg(
        MlpMixer, variant, pretrained,
        default_cfg=default_cfgs[variant],
        **kwargs)
    return model


@register_model
def mixer_s32_224(pretrained=False, **kwargs):
    """ Mixer-S/32 224x224
    Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=8, hidden_dim=512, **kwargs)
    model = _create_mixer('mixer_s32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_s16_224(pretrained=False, **kwargs):
    """ Mixer-S/16 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=8, hidden_dim=512, **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b32_224(pretrained=False, **kwargs):
    """ Mixer-B/32 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=12, hidden_dim=768, **kwargs)
    model = _create_mixer('mixer_b32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b16_224(pretrained=False, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=12, hidden_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b16_224_in21k(pretrained=False, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-21k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=12, hidden_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224_in21k', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_l32_224(pretrained=False, **kwargs):
    """ Mixer-L/32 224x224.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=24, hidden_dim=1024, **kwargs)
    model = _create_mixer('mixer_l32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_l16_224(pretrained=False, **kwargs):
    """ Mixer-L/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=24, hidden_dim=1024, **kwargs)
    model = _create_mixer('mixer_l16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_l16_224_in21k(pretrained=False, **kwargs):
    """ Mixer-L/16 224x224. ImageNet-21k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=24, hidden_dim=1024, **kwargs)
    model = _create_mixer('mixer_l16_224_in21k', pretrained=pretrained, **model_args)
    return model

@register_model
def mixer_b16_224_miil(pretrained=False, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-21k pretrained weights.
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_args = dict(patch_size=16, num_blocks=12, hidden_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224_miil', pretrained=pretrained, **model_args)
    return model

@register_model
def mixer_b16_224_miil_in21k(pretrained=False, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_args = dict(patch_size=16, num_blocks=12, hidden_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224_miil_in21k', pretrained=pretrained, **model_args)
    return model

@register_model
def gmixer_12_224(pretrained=False, **kwargs):
    """ Glu-Mixer-12 224x224 (short & fat)
    Experiment by Ross Wightman, adding (Si)GLU to MLP-Mixer
    """
    model_args = dict(
        patch_size=20, num_blocks=12, hidden_dim=512, mlp_ratio=(1.0, 6.0),
        mlp_layer=GluMlp, act_layer=nn.SiLU, **kwargs)
    model = _create_mixer('gmixer_12_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmixer_24_224(pretrained=False, **kwargs):
    """ Glu-Mixer-24 224x224 (tall & slim)
    Experiment by Ross Wightman, adding (Si)GLU to MLP-Mixer
    """
    model_args = dict(
        patch_size=20, num_blocks=24, hidden_dim=384, mlp_ratio=(1.0, 6.0),
        mlp_layer=GluMlp, act_layer=nn.SiLU, **kwargs)
    model = _create_mixer('gmixer_24_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_12_224(pretrained=False, **kwargs):
    """ ResMLP-12
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16, num_blocks=12, hidden_dim=384, mlp_ratio=4, block_layer=ResBlock, norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_12_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_24_224(pretrained=False, **kwargs):
    """ ResMLP-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16, num_blocks=24, hidden_dim=384, mlp_ratio=4, block_layer=ResBlock, norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_24_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_36_224(pretrained=False, **kwargs):
    """ ResMLP-36
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16, num_blocks=36, hidden_dim=384, mlp_ratio=4, block_layer=ResBlock, norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_36_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmlp_ti16_224(pretrained=False, **kwargs):
    """ gMLP-Tiny
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, hidden_dim=128, mlp_ratio=6, block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_ti16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmlp_s16_224(pretrained=False, **kwargs):
    """ gMLP-Small
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, hidden_dim=256, mlp_ratio=6, block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_s16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmlp_b16_224(pretrained=False, **kwargs):
    """ gMLP-Base
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, hidden_dim=512, mlp_ratio=6, block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_b16_224', pretrained=pretrained, **model_args)
    return model
