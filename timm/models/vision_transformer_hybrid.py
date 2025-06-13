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
from typing import Dict, Tuple, Type, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import StdConv2dSame, StdConv2d, ConvNormAct, to_ntuple, HybridEmbed

from ._builder import build_model_with_cfg
from ._registry import generate_default_cfgs, register_model, register_model_deprecations
from .resnet import resnet26d, resnet50d
from .resnetv2 import ResNetV2, create_resnetv2_stem
from .vision_transformer import VisionTransformer


class ConvStem(nn.Sequential):
    def __init__(
            self,
            in_chans: int = 3,
            depth: int = 3,
            channels: Union[int, Tuple[int, ...]] = 64,
            kernel_size: Union[int, Tuple[int, ...]] = 3,
            stride: Union[int, Tuple[int, ...]] = (2, 2, 2),
            padding: Union[str, int, Tuple[int, ...]] = "",
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        if isinstance(channels, int):
            # a default tiered channel strategy
            channels = tuple([channels // 2**i for i in range(depth)][::-1])

        kernel_size = to_ntuple(depth)(kernel_size)
        padding = to_ntuple(depth)(padding)
        assert depth == len(stride) == len(kernel_size) == len(channels)

        in_chs = in_chans
        for i in range(len(channels)):
            last_conv = i == len(channels) - 1
            self.add_module(f'{i}', ConvNormAct(
                in_chs,
                channels[i],
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i],
                bias=last_conv,
                apply_norm=not last_conv,
                apply_act=not last_conv,
                norm_layer=norm_layer,
                act_layer=act_layer,
            ))
            in_chs = channels[i]


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


def _convert_mobileclip(state_dict, model, prefix='image_encoder.model.'):
    out = {}
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, '')
        k = k.replace('patch_emb.', 'patch_embed.backbone.')
        k = k.replace('block.conv', 'conv')
        k = k.replace('block.norm', 'bn')
        k = k.replace('post_transformer_norm.', 'norm.')
        k = k.replace('pre_norm_mha.0', 'norm1')
        k = k.replace('pre_norm_mha.1', 'attn')
        k = k.replace('pre_norm_ffn.0', 'norm2')
        k = k.replace('pre_norm_ffn.1', 'mlp.fc1')
        k = k.replace('pre_norm_ffn.4', 'mlp.fc2')
        k = k.replace('qkv_proj.', 'qkv.')
        k = k.replace('out_proj.', 'proj.')
        k = k.replace('transformer.', 'blocks.')
        if k == 'pos_embed.pos_embed.pos_embed':
            k = 'pos_embed'
            v = v.squeeze(0)
        if 'classifier.proj' in k:
            bias_k = k.replace('classifier.proj', 'head.bias')
            k = k.replace('classifier.proj', 'head.weight')
            v = v.T
            out[bias_k] = torch.zeros(v.shape[0])
        out[k] = v
    return out


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: VisionTransformer,
        interpolation: str = 'bicubic',
        antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    from .vision_transformer import checkpoint_filter_fn as _filter_fn

    if 'image_encoder.model.patch_emb.0.block.conv.weight' in state_dict:
        state_dict = _convert_mobileclip(state_dict, model)

    return _filter_fn(state_dict, model, interpolation=interpolation, antialias=antialias)


def _create_vision_transformer_hybrid(variant, backbone, embed_args=None, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    embed_args = embed_args or {}
    embed_layer = partial(HybridEmbed, backbone=backbone, **embed_args)
    kwargs.setdefault('embed_layer', embed_layer)
    kwargs.setdefault('patch_size', 1)  # default patch size for hybrid models if not set
    return build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )


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
        #url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth',
        hf_hub_id='timm/',
        num_classes=0, crop_pct=0.9),
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

    'vit_base_mci_224.apple_mclip_lt': _cfg(
        hf_hub_id='apple/mobileclip_b_lt_timm',
        url='https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt',
        num_classes=512,
        mean=(0., 0., 0.), std=(1., 1., 1.), first_conv='patch_embed.backbone.0.conv',
    ),
    'vit_base_mci_224.apple_mclip': _cfg(
        hf_hub_id='apple/mobileclip_b_timm',
        url='https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_b.pt',
        num_classes=512,
        mean=(0., 0., 0.), std=(1., 1., 1.), first_conv='patch_embed.backbone.0.conv',
    ),
})


@register_model
def vit_tiny_r_s16_p8_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ R+ViT-Ti/S16 w/ 8x8 patch hybrid @ 224 x 224.
    """
    backbone = _resnetv2(layers=(), **kwargs)
    model_args = dict(patch_size=8, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer_hybrid(
        'vit_tiny_r_s16_p8_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_tiny_r_s16_p8_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ R+ViT-Ti/S16 w/ 8x8 patch hybrid @ 384 x 384.
    """
    backbone = _resnetv2(layers=(), **kwargs)
    model_args = dict(patch_size=8, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer_hybrid(
        'vit_tiny_r_s16_p8_384', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_r26_s32_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ R26+ViT-S/S32 hybrid.
    """
    backbone = _resnetv2((2, 2, 2, 2), **kwargs)
    model_args = dict(embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer_hybrid(
        'vit_small_r26_s32_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_r26_s32_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ R26+ViT-S/S32 hybrid.
    """
    backbone = _resnetv2((2, 2, 2, 2), **kwargs)
    model_args = dict(embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer_hybrid(
        'vit_small_r26_s32_384', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_r26_s32_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ R26+ViT-B/S32 hybrid.
    """
    backbone = _resnetv2((2, 2, 2, 2), **kwargs)
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_hybrid(
        'vit_base_r26_s32_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_r50_s16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ R50+ViT-B/S16 hybrid from original paper (https://arxiv.org/abs/2010.11929).
    """
    backbone = _resnetv2((3, 4, 9), **kwargs)
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_hybrid(
        'vit_base_r50_s16_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_r50_s16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ R50+ViT-B/16 hybrid from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    backbone = _resnetv2((3, 4, 9), **kwargs)
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_hybrid(
        'vit_base_r50_s16_384', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_r50_s32_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ R50+ViT-L/S32 hybrid.
    """
    backbone = _resnetv2((3, 4, 6, 3), **kwargs)
    model_args = dict(embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer_hybrid(
        'vit_large_r50_s32_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_r50_s32_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ R50+ViT-L/S32 hybrid.
    """
    backbone = _resnetv2((3, 4, 6, 3), **kwargs)
    model_args = dict(embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer_hybrid(
        'vit_large_r50_s32_384', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_resnet26d_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ Custom ViT small hybrid w/ ResNet26D stride 32. No pretrained weights.
    """
    backbone = resnet26d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[4])
    model_args = dict(embed_dim=768, depth=8, num_heads=8, mlp_ratio=3)
    model = _create_vision_transformer_hybrid(
        'vit_small_resnet26d_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_resnet50d_s16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ Custom ViT small hybrid w/ ResNet50D 3-stages, stride 16. No pretrained weights.
    """
    backbone = resnet50d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[3])
    model_args = dict(embed_dim=768, depth=8, num_heads=8, mlp_ratio=3)
    model = _create_vision_transformer_hybrid(
        'vit_small_resnet50d_s16_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_resnet26d_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ Custom ViT base hybrid w/ ResNet26D stride 32. No pretrained weights.
    """
    backbone = resnet26d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[4])
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_hybrid(
        'vit_base_resnet26d_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_resnet50d_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ Custom ViT base hybrid w/ ResNet50D stride 32. No pretrained weights.
    """
    backbone = resnet50d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[4])
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_hybrid(
        'vit_base_resnet50d_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_mci_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ Custom ViT base hybrid w/ ResNet50D stride 32. No pretrained weights.
    """
    backbone = ConvStem(
        channels=(768//4, 768//4, 768),
        stride=(4, 2, 2),
        kernel_size=(4, 2, 2),
        padding=0,
        in_chans=kwargs.get('in_chans', 3),
        act_layer=nn.GELU,
    )
    model_args = dict(embed_dim=768, depth=12, num_heads=12, no_embed_class=True)
    model = _create_vision_transformer_hybrid(
        'vit_base_mci_224', backbone=backbone, embed_args=dict(proj=False),
        pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


register_model_deprecations(__name__, {
    'vit_tiny_r_s16_p8_224_in21k': 'vit_tiny_r_s16_p8_224.augreg_in21k',
    'vit_small_r26_s32_224_in21k': 'vit_small_r26_s32_224.augreg_in21k',
    'vit_base_r50_s16_224_in21k': 'vit_base_r50_s16_224.orig_in21k',
    'vit_base_resnet50_224_in21k': 'vit_base_r50_s16_224.orig_in21k',
    'vit_large_r50_s32_224_in21k': 'vit_large_r50_s32_224.augreg_in21k',
    'vit_base_resnet50_384': 'vit_base_r50_s16_384.orig_in21k_ft_in1k'
})
