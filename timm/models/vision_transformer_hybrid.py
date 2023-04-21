""" Hybrid Vision Transformer (ViT) in PyTorch

A PyTorch implement of the Hybrid Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

NOTE These hybrid model definitions depend on code in vision_transformer.py.
They were moved here to keep file sizes sane.

Hacked together by / Copyright 2020, Ross Wightman
"""
from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import StdConv2dSame, StdConv2d, to_2tuple
from ._registry import generate_default_cfgs, register_model, register_model_deprecations
from .resnet import resnet26d, resnet50d
from .resnetv2 import ResNetV2, create_resnetv2_stem
from .vision_transformer import _create_vision_transformer


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(
            self,
            backbone,
            img_size=224,
            patch_size=1,
            feature_size=None,
            in_chans=3,
            embed_dim=768,
            bias=True,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def _create_vision_transformer_hybrid(variant, backbone, pretrained=False, **kwargs):
    embed_layer = partial(HybridEmbed, backbone=backbone)
    kwargs.setdefault('patch_size', 1)  # default patch size for hybrid models if not set
    return _create_vision_transformer(variant, pretrained=pretrained, embed_layer=embed_layer, **kwargs)


def _resnetv2(layers=(3, 4, 9), **kwargs):
    """ ResNet-V2 backbone helper"""
    padding_same = kwargs.get('padding_same', True)
    stem_type = 'same' if padding_same else ''
    conv_layer = partial(StdConv2dSame, eps=1e-8) if padding_same else partial(StdConv2d, eps=1e-8)
    if len(layers):
        backbone = ResNetV2(
            layers=layers, num_classes=0, global_pool='', in_chans=kwargs.get('in_chans', 3),
            preact=False, stem_type=stem_type, conv_layer=conv_layer)
    else:
        backbone = create_resnetv2_stem(
            kwargs.get('in_chans', 3), stem_type=stem_type, preact=False, conv_layer=conv_layer)
    return backbone


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.backbone.stem.conv', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # hybrid in-1k models (weights from official JAX impl where they exist)
    'vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R_Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True,
        first_conv='patch_embed.backbone.conv'),
    'vit_tiny_r_s16_p8_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R_Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        first_conv='patch_embed.backbone.conv', input_size=(3, 384, 384), crop_pct=1.0, custom_load=True),
    'vit_small_r26_s32_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R26_S_32-i21k-300ep-lr_0.001-aug_light0-wd_0.03-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True,
    ),
    'vit_small_r26_s32_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R26_S_32-i21k-300ep-lr_0.001-aug_medium2-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0, custom_load=True),
    'vit_base_r26_s32_224.untrained': _cfg(),
    'vit_base_r50_s16_384.orig_in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_r50_s32_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R50_L_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True,
    ),
    'vit_large_r50_s32_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R50_L_32-i21k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0, custom_load=True,
    ),

    # hybrid in-21k models (weights from official Google JAX impl where they exist)
    'vit_tiny_r_s16_p8_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R_Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        num_classes=21843, crop_pct=0.9, first_conv='patch_embed.backbone.conv', custom_load=True),
    'vit_small_r26_s32_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R26_S_32-i21k-300ep-lr_0.001-aug_medium2-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        num_classes=21843, crop_pct=0.9, custom_load=True),
    'vit_base_r50_s16_224.orig_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth',
        hf_hub_id='timm/',
        num_classes=21843, crop_pct=0.9),
    'vit_large_r50_s32_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R50_L_32-i21k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        num_classes=21843, crop_pct=0.9, custom_load=True),

    # hybrid models (using timm resnet backbones)
    'vit_small_resnet26d_224.untrained': _cfg(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, first_conv='patch_embed.backbone.conv1.0'),
    'vit_small_resnet50d_s16_224.untrained': _cfg(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, first_conv='patch_embed.backbone.conv1.0'),
    'vit_base_resnet26d_224.untrained': _cfg(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, first_conv='patch_embed.backbone.conv1.0'),
    'vit_base_resnet50d_224.untrained': _cfg(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, first_conv='patch_embed.backbone.conv1.0'),
})


@register_model
def vit_tiny_r_s16_p8_224(pretrained=False, **kwargs):
    """ R+ViT-Ti/S16 w/ 8x8 patch hybrid @ 224 x 224.
    """
    backbone = _resnetv2(layers=(), **kwargs)
    model_args = dict(patch_size=8, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer_hybrid(
        'vit_tiny_r_s16_p8_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_tiny_r_s16_p8_384(pretrained=False, **kwargs):
    """ R+ViT-Ti/S16 w/ 8x8 patch hybrid @ 384 x 384.
    """
    backbone = _resnetv2(layers=(), **kwargs)
    model_args = dict(patch_size=8, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer_hybrid(
        'vit_tiny_r_s16_p8_384', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_r26_s32_224(pretrained=False, **kwargs):
    """ R26+ViT-S/S32 hybrid.
    """
    backbone = _resnetv2((2, 2, 2, 2), **kwargs)
    model_args = dict(embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer_hybrid(
        'vit_small_r26_s32_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_r26_s32_384(pretrained=False, **kwargs):
    """ R26+ViT-S/S32 hybrid.
    """
    backbone = _resnetv2((2, 2, 2, 2), **kwargs)
    model_args = dict(embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer_hybrid(
        'vit_small_r26_s32_384', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_r26_s32_224(pretrained=False, **kwargs):
    """ R26+ViT-B/S32 hybrid.
    """
    backbone = _resnetv2((2, 2, 2, 2), **kwargs)
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_hybrid(
        'vit_base_r26_s32_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_r50_s16_224(pretrained=False, **kwargs):
    """ R50+ViT-B/S16 hybrid from original paper (https://arxiv.org/abs/2010.11929).
    """
    backbone = _resnetv2((3, 4, 9), **kwargs)
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_hybrid(
        'vit_base_r50_s16_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_r50_s16_384(pretrained=False, **kwargs):
    """ R50+ViT-B/16 hybrid from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    backbone = _resnetv2((3, 4, 9), **kwargs)
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_hybrid(
        'vit_base_r50_s16_384', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_r50_s32_224(pretrained=False, **kwargs):
    """ R50+ViT-L/S32 hybrid.
    """
    backbone = _resnetv2((3, 4, 6, 3), **kwargs)
    model_args = dict(embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer_hybrid(
        'vit_large_r50_s32_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_r50_s32_384(pretrained=False, **kwargs):
    """ R50+ViT-L/S32 hybrid.
    """
    backbone = _resnetv2((3, 4, 6, 3), **kwargs)
    model_args = dict(embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer_hybrid(
        'vit_large_r50_s32_384', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_resnet26d_224(pretrained=False, **kwargs):
    """ Custom ViT small hybrid w/ ResNet26D stride 32. No pretrained weights.
    """
    backbone = resnet26d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[4])
    model_args = dict(embed_dim=768, depth=8, num_heads=8, mlp_ratio=3)
    model = _create_vision_transformer_hybrid(
        'vit_small_resnet26d_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_resnet50d_s16_224(pretrained=False, **kwargs):
    """ Custom ViT small hybrid w/ ResNet50D 3-stages, stride 16. No pretrained weights.
    """
    backbone = resnet50d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[3])
    model_args = dict(embed_dim=768, depth=8, num_heads=8, mlp_ratio=3)
    model = _create_vision_transformer_hybrid(
        'vit_small_resnet50d_s16_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_resnet26d_224(pretrained=False, **kwargs):
    """ Custom ViT base hybrid w/ ResNet26D stride 32. No pretrained weights.
    """
    backbone = resnet26d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[4])
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_hybrid(
        'vit_base_resnet26d_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_resnet50d_224(pretrained=False, **kwargs):
    """ Custom ViT base hybrid w/ ResNet50D stride 32. No pretrained weights.
    """
    backbone = resnet50d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[4])
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_hybrid(
        'vit_base_resnet50d_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


register_model_deprecations(__name__, {
    'vit_tiny_r_s16_p8_224_in21k': 'vit_tiny_r_s16_p8_224.augreg_in21k',
    'vit_small_r26_s32_224_in21k': 'vit_small_r26_s32_224.augreg_in21k',
    'vit_base_r50_s16_224_in21k': 'vit_base_r50_s16_224.orig_in21k',
    'vit_base_resnet50_224_in21k': 'vit_base_r50_s16_224.orig_in21k',
    'vit_large_r50_s32_224_in21k': 'vit_large_r50_s32_224.augreg_in21k',
    'vit_base_resnet50_384': 'vit_base_r50_s16_384.orig_in21k_ft_in1k'
})
