from functools import partial

import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ._builder import build_model_with_cfg
from ._builder import pretrained_cfg_for_features
from ._efficientnet_blocks import SqueezeExcite
from ._efficientnet_builder import decode_arch_def, resolve_act_layer, resolve_bn_args, round_channels
from ._registry import register_model, generate_default_cfgs
from .mobilenetv3 import MobileNetV3, MobileNetV3Features

__all__ = []  # model_registry will add each entrypoint fn to this


def _gen_hardcorenas(pretrained, variant, arch_def, **kwargs):
    """Creates a hardcorenas model

    Ref impl: https://github.com/Alibaba-MIIL/HardCoReNAS
    Paper: https://arxiv.org/abs/2102.11646

    """
    num_features = 1280
    se_layer = partial(SqueezeExcite, gate_layer='hard_sigmoid', force_act_layer=nn.ReLU, rd_round_fn=round_channels)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=num_features,
        stem_size=32,
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_layer=se_layer,
        **kwargs,
    )

    features_only = False
    model_cls = MobileNetV3
    kwargs_filter = None
    if model_kwargs.pop('features_only', False):
        features_only = True
        kwargs_filter = ('num_classes', 'num_features', 'global_pool', 'head_conv', 'head_bias', 'global_pool')
        model_cls = MobileNetV3Features
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        pretrained_strict=not features_only,
        kwargs_filter=kwargs_filter,
        **model_kwargs,
    )
    if features_only:
        model.default_cfg = pretrained_cfg_for_features(model.default_cfg)
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'hardcorenas_a.miil_green_in1k': _cfg(hf_hub_id='timm/'),
    'hardcorenas_b.miil_green_in1k': _cfg(hf_hub_id='timm/'),
    'hardcorenas_c.miil_green_in1k': _cfg(hf_hub_id='timm/'),
    'hardcorenas_d.miil_green_in1k': _cfg(hf_hub_id='timm/'),
    'hardcorenas_e.miil_green_in1k': _cfg(hf_hub_id='timm/'),
    'hardcorenas_f.miil_green_in1k': _cfg(hf_hub_id='timm/'),
})


@register_model
def hardcorenas_a(pretrained=False, **kwargs) -> MobileNetV3:
    """ hardcorenas_A """
    arch_def = [['ds_r1_k3_s1_e1_c16_nre'], ['ir_r1_k5_s2_e3_c24_nre', 'ir_r1_k5_s1_e3_c24_nre_se0.25'],
                ['ir_r1_k5_s2_e3_c40_nre', 'ir_r1_k5_s1_e6_c40_nre_se0.25'],
                ['ir_r1_k5_s2_e6_c80_se0.25', 'ir_r1_k5_s1_e6_c80_se0.25'],
                ['ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25'],
                ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25'], ['cn_r1_k1_s1_c960']]
    model = _gen_hardcorenas(pretrained=pretrained, variant='hardcorenas_a', arch_def=arch_def, **kwargs)
    return model


@register_model
def hardcorenas_b(pretrained=False, **kwargs) -> MobileNetV3:
    """ hardcorenas_B """
    arch_def = [['ds_r1_k3_s1_e1_c16_nre'],
                ['ir_r1_k5_s2_e3_c24_nre', 'ir_r1_k5_s1_e3_c24_nre_se0.25', 'ir_r1_k3_s1_e3_c24_nre'],
                ['ir_r1_k5_s2_e3_c40_nre', 'ir_r1_k5_s1_e3_c40_nre', 'ir_r1_k5_s1_e3_c40_nre'],
                ['ir_r1_k5_s2_e3_c80', 'ir_r1_k5_s1_e3_c80', 'ir_r1_k3_s1_e3_c80', 'ir_r1_k3_s1_e3_c80'],
                ['ir_r1_k5_s1_e3_c112', 'ir_r1_k3_s1_e3_c112', 'ir_r1_k3_s1_e3_c112', 'ir_r1_k3_s1_e3_c112'],
                ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k3_s1_e3_c192_se0.25'],
                ['cn_r1_k1_s1_c960']]
    model = _gen_hardcorenas(pretrained=pretrained, variant='hardcorenas_b', arch_def=arch_def, **kwargs)
    return model


@register_model
def hardcorenas_c(pretrained=False, **kwargs) -> MobileNetV3:
    """ hardcorenas_C """
    arch_def = [['ds_r1_k3_s1_e1_c16_nre'], ['ir_r1_k5_s2_e3_c24_nre', 'ir_r1_k5_s1_e3_c24_nre_se0.25'],
                ['ir_r1_k5_s2_e3_c40_nre', 'ir_r1_k5_s1_e3_c40_nre', 'ir_r1_k5_s1_e3_c40_nre',
                 'ir_r1_k5_s1_e3_c40_nre'],
                ['ir_r1_k5_s2_e4_c80', 'ir_r1_k5_s1_e6_c80_se0.25', 'ir_r1_k3_s1_e3_c80', 'ir_r1_k3_s1_e3_c80'],
                ['ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k3_s1_e3_c112', 'ir_r1_k3_s1_e3_c112', 'ir_r1_k3_s1_e3_c112'],
                ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k3_s1_e3_c192_se0.25'],
                ['cn_r1_k1_s1_c960']]
    model = _gen_hardcorenas(pretrained=pretrained, variant='hardcorenas_c', arch_def=arch_def, **kwargs)
    return model


@register_model
def hardcorenas_d(pretrained=False, **kwargs) -> MobileNetV3:
    """ hardcorenas_D """
    arch_def = [['ds_r1_k3_s1_e1_c16_nre'], ['ir_r1_k5_s2_e3_c24_nre_se0.25', 'ir_r1_k5_s1_e3_c24_nre_se0.25'],
                ['ir_r1_k5_s2_e3_c40_nre_se0.25', 'ir_r1_k5_s1_e4_c40_nre_se0.25', 'ir_r1_k3_s1_e3_c40_nre_se0.25'],
                ['ir_r1_k5_s2_e4_c80_se0.25', 'ir_r1_k3_s1_e3_c80_se0.25', 'ir_r1_k3_s1_e3_c80_se0.25',
                 'ir_r1_k3_s1_e3_c80_se0.25'],
                ['ir_r1_k3_s1_e4_c112_se0.25', 'ir_r1_k5_s1_e4_c112_se0.25', 'ir_r1_k3_s1_e3_c112_se0.25',
                 'ir_r1_k5_s1_e3_c112_se0.25'],
                ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25',
                 'ir_r1_k3_s1_e6_c192_se0.25'], ['cn_r1_k1_s1_c960']]
    model = _gen_hardcorenas(pretrained=pretrained, variant='hardcorenas_d', arch_def=arch_def, **kwargs)
    return model


@register_model
def hardcorenas_e(pretrained=False, **kwargs) -> MobileNetV3:
    """ hardcorenas_E """
    arch_def = [['ds_r1_k3_s1_e1_c16_nre'], ['ir_r1_k5_s2_e3_c24_nre_se0.25', 'ir_r1_k5_s1_e3_c24_nre_se0.25'],
                ['ir_r1_k5_s2_e6_c40_nre_se0.25', 'ir_r1_k5_s1_e4_c40_nre_se0.25', 'ir_r1_k5_s1_e4_c40_nre_se0.25',
                 'ir_r1_k3_s1_e3_c40_nre_se0.25'], ['ir_r1_k5_s2_e4_c80_se0.25', 'ir_r1_k3_s1_e6_c80_se0.25'],
                ['ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25',
                 'ir_r1_k5_s1_e3_c112_se0.25'],
                ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25',
                 'ir_r1_k3_s1_e6_c192_se0.25'], ['cn_r1_k1_s1_c960']]
    model = _gen_hardcorenas(pretrained=pretrained, variant='hardcorenas_e', arch_def=arch_def, **kwargs)
    return model


@register_model
def hardcorenas_f(pretrained=False, **kwargs) -> MobileNetV3:
    """ hardcorenas_F """
    arch_def = [['ds_r1_k3_s1_e1_c16_nre'], ['ir_r1_k5_s2_e3_c24_nre_se0.25', 'ir_r1_k5_s1_e3_c24_nre_se0.25'],
                ['ir_r1_k5_s2_e6_c40_nre_se0.25', 'ir_r1_k5_s1_e6_c40_nre_se0.25'],
                ['ir_r1_k5_s2_e6_c80_se0.25', 'ir_r1_k5_s1_e6_c80_se0.25', 'ir_r1_k3_s1_e3_c80_se0.25',
                 'ir_r1_k3_s1_e3_c80_se0.25'],
                ['ir_r1_k3_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25',
                 'ir_r1_k3_s1_e3_c112_se0.25'],
                ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k3_s1_e6_c192_se0.25',
                 'ir_r1_k3_s1_e6_c192_se0.25'], ['cn_r1_k1_s1_c960']]
    model = _gen_hardcorenas(pretrained=pretrained, variant='hardcorenas_f', arch_def=arch_def, **kwargs)
    return model
