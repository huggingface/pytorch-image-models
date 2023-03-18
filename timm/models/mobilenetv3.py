""" MobileNet V3

A PyTorch impl of MobileNet-V3, compatible with TF weights from official impl.

Paper: Searching for MobileNetV3 - https://arxiv.org/abs/1905.02244

Hacked together by / Copyright 2019, Ross Wightman
"""
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import SelectAdaptivePool2d, Linear, create_conv2d, get_norm_act_layer
from ._builder import build_model_with_cfg, pretrained_cfg_for_features
from ._efficientnet_blocks import SqueezeExcite
from ._efficientnet_builder import EfficientNetBuilder, decode_arch_def, efficientnet_init_weights, \
    round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from ._features import FeatureInfo, FeatureHooks
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['MobileNetV3', 'MobileNetV3Features']


class MobileNetV3(nn.Module):
    """ MobiletNet-V3

    Based on my EfficientNet implementation and building blocks, this model utilizes the MobileNet-v3 specific
    'efficient head', where global pooling is done before the head convolution without a final batch-norm
    layer before the classifier.

    Paper: `Searching for MobileNetV3` - https://arxiv.org/abs/1905.02244

    Other architectures utilizing MobileNet-V3 efficient head that are supported by this impl include:
      * HardCoRe-NAS - https://arxiv.org/abs/2102.11646 (defn in hardcorenas.py uses this class)
      * FBNet-V3 - https://arxiv.org/abs/2006.02049
      * LCNet - https://arxiv.org/abs/2109.15099
    """

    def __init__(
            self,
            block_args,
            num_classes=1000,
            in_chans=3,
            stem_size=16,
            fix_stem=False,
            num_features=1280,
            head_bias=True,
            pad_type='',
            act_layer=None,
            norm_layer=None,
            se_layer=None,
            se_from_exp=True,
            round_chs_fn=round_channels,
            drop_rate=0.,
            drop_path_rate=0.,
            global_pool='avg',
    ):
        super(MobileNetV3, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_act_layer(stem_size, inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=32,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            se_from_exp=se_from_exp,
            act_layer=act_layer,
            norm_layer=norm_layer,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
        )
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        num_pooled_chs = head_chs * self.global_pool.feat_mult()
        self.conv_head = create_conv2d(num_pooled_chs, self.num_features, 1, padding=pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1]
        layers.extend(self.blocks)
        layers.extend([self.global_pool, self.conv_head, self.act2])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^conv_stem|bn1',
            blocks=r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)'
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if pre_logits:
            return x
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class MobileNetV3Features(nn.Module):
    """ MobileNetV3 Feature Extractor

    A work-in-progress feature extraction module for MobileNet-V3 to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(
            self,
            block_args,
            out_indices=(0, 1, 2, 3, 4),
            feature_location='bottleneck',
            in_chans=3,
            stem_size=16,
            fix_stem=False,
            output_stride=32,
            pad_type='',
            round_chs_fn=round_channels,
            se_from_exp=True,
            act_layer=None,
            norm_layer=None,
            se_layer=None,
            drop_rate=0.,
            drop_path_rate=0.,
    ):
        super(MobileNetV3Features, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            se_from_exp=se_from_exp,
            act_layer=act_layer,
            norm_layer=norm_layer,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
            feature_location=feature_location,
        )
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = FeatureInfo(builder.features, out_indices)
        self._stage_out_idx = {v['stage']: i for i, v in enumerate(self.feature_info) if i in out_indices}

        efficientnet_init_weights(self)

        # Register feature extraction hooks with FeatureHooks helper
        self.feature_hooks = None
        if feature_location != 'bottleneck':
            hooks = self.feature_info.get_dicts(keys=('module', 'hook_type'))
            self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, x) -> List[torch.Tensor]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.feature_hooks is None:
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)  # add stem out
            for i, b in enumerate(self.blocks):
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(b, x)
                else:
                    x = b(x)
                if i + 1 in self._stage_out_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            out = self.feature_hooks.get_output(x.device)
            return list(out.values())


def _create_mnv3(variant, pretrained=False, **kwargs):
    features_only = False
    model_cls = MobileNetV3
    kwargs_filter = None
    if kwargs.pop('features_only', False):
        features_only = True
        kwargs_filter = ('num_classes', 'num_features', 'head_conv', 'head_bias', 'global_pool')
        model_cls = MobileNetV3Features
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        pretrained_strict=not features_only,
        kwargs_filter=kwargs_filter,
        **kwargs)
    if features_only:
        model.default_cfg = pretrained_cfg_for_features(model.default_cfg)
    return model


def _gen_mobilenet_v3_rw(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_nre_noskip'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c960'],  # hard-swish
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        head_bias=False,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_layer=partial(SqueezeExcite, gate_layer='hard_sigmoid'),
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_mobilenet_v3(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    if 'small' in variant:
        num_features = 1024
        if 'minimal' in variant:
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16'],
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24', 'ir_r1_k3_s1_e3.67_c24'],
                # stage 2, 28x28 in
                ['ir_r1_k3_s2_e4_c40', 'ir_r2_k3_s1_e6_c40'],
                # stage 3, 14x14 in
                ['ir_r2_k3_s1_e3_c48'],
                # stage 4, 14x14in
                ['ir_r3_k3_s2_e6_c96'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],
            ]
        else:
            act_layer = resolve_act_layer(kwargs, 'hard_swish')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16_se0.25_nre'],  # relu
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24_nre', 'ir_r1_k3_s1_e3.67_c24_nre'],  # relu
                # stage 2, 28x28 in
                ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r2_k5_s1_e6_c40_se0.25'],  # hard-swish
                # stage 3, 14x14 in
                ['ir_r2_k5_s1_e3_c48_se0.25'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r3_k5_s2_e6_c96_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],  # hard-swish
            ]
    else:
        num_features = 1280
        if 'minimal' in variant:
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16'],
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24', 'ir_r1_k3_s1_e3_c24'],
                # stage 2, 56x56 in
                ['ir_r3_k3_s2_e3_c40'],
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112'],
                # stage 5, 14x14in
                ['ir_r3_k3_s2_e6_c160'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],
            ]
        else:
            act_layer = resolve_act_layer(kwargs, 'hard_swish')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16_nre'],  # relu
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
                # stage 2, 56x56 in
                ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
                # stage 5, 14x14in
                ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],  # hard-swish
            ]
    se_layer = partial(SqueezeExcite, gate_layer='hard_sigmoid', force_act_layer=nn.ReLU, rd_round_fn=round_channels)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=num_features,
        stem_size=16,
        fix_stem=channel_multiplier < 0.75,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        se_layer=se_layer,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_fbnetv3(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """ FBNetV3
    Paper: `FBNetV3: Joint Architecture-Recipe Search using Predictor Pretraining`
        - https://arxiv.org/abs/2006.02049
    FIXME untested, this is a preliminary impl of some FBNet-V3 variants.
    """
    vl = variant.split('_')[-1]
    if vl in ('a', 'b'):
        stem_size = 16
        arch_def = [
            ['ds_r2_k3_s1_e1_c16'],
            ['ir_r1_k5_s2_e4_c24', 'ir_r3_k5_s1_e2_c24'],
            ['ir_r1_k5_s2_e5_c40_se0.25', 'ir_r4_k5_s1_e3_c40_se0.25'],
            ['ir_r1_k5_s2_e5_c72', 'ir_r4_k3_s1_e3_c72'],
            ['ir_r1_k3_s1_e5_c120_se0.25', 'ir_r5_k5_s1_e3_c120_se0.25'],
            ['ir_r1_k3_s2_e6_c184_se0.25', 'ir_r5_k5_s1_e4_c184_se0.25', 'ir_r1_k5_s1_e6_c224_se0.25'],
            ['cn_r1_k1_s1_c1344'],
        ]
    elif vl == 'd':
        stem_size = 24
        arch_def = [
            ['ds_r2_k3_s1_e1_c16'],
            ['ir_r1_k3_s2_e5_c24', 'ir_r5_k3_s1_e2_c24'],
            ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r4_k3_s1_e3_c40_se0.25'],
            ['ir_r1_k3_s2_e5_c72', 'ir_r4_k3_s1_e3_c72'],
            ['ir_r1_k3_s1_e5_c128_se0.25', 'ir_r6_k5_s1_e3_c128_se0.25'],
            ['ir_r1_k3_s2_e6_c208_se0.25', 'ir_r5_k5_s1_e5_c208_se0.25', 'ir_r1_k5_s1_e6_c240_se0.25'],
            ['cn_r1_k1_s1_c1440'],
        ]
    elif vl == 'g':
        stem_size = 32
        arch_def = [
            ['ds_r3_k3_s1_e1_c24'],
            ['ir_r1_k5_s2_e4_c40', 'ir_r4_k5_s1_e2_c40'],
            ['ir_r1_k5_s2_e4_c56_se0.25', 'ir_r4_k5_s1_e3_c56_se0.25'],
            ['ir_r1_k5_s2_e5_c104', 'ir_r4_k3_s1_e3_c104'],
            ['ir_r1_k3_s1_e5_c160_se0.25', 'ir_r8_k5_s1_e3_c160_se0.25'],
            ['ir_r1_k3_s2_e6_c264_se0.25', 'ir_r6_k5_s1_e5_c264_se0.25', 'ir_r2_k5_s1_e6_c288_se0.25'],
            ['cn_r1_k1_s1_c1728'],
        ]
    else:
        raise NotImplemented
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, round_limit=0.95)
    se_layer = partial(SqueezeExcite, gate_layer='hard_sigmoid', rd_round_fn=round_chs_fn)
    act_layer = resolve_act_layer(kwargs, 'hard_swish')
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=1984,
        head_bias=False,
        stem_size=stem_size,
        round_chs_fn=round_chs_fn,
        se_from_exp=False,
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        se_layer=se_layer,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_lcnet(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """ LCNet
    Essentially a MobileNet-V3 crossed with a MobileNet-V1

    Paper: `PP-LCNet: A Lightweight CPU Convolutional Neural Network` - https://arxiv.org/abs/2109.15099

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['dsa_r1_k3_s1_c32'],
        # stage 1, 112x112 in
        ['dsa_r2_k3_s2_c64'],
        # stage 2, 56x56 in
        ['dsa_r2_k3_s2_c128'],
        # stage 3, 28x28 in
        ['dsa_r1_k3_s2_c256', 'dsa_r1_k5_s1_c256'],
        # stage 4, 14x14in
        ['dsa_r4_k5_s1_c256'],
        # stage 5, 14x14in
        ['dsa_r2_k5_s2_c512_se0.25'],
        # 7x7
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=16,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_layer=partial(SqueezeExcite, gate_layer='hard_sigmoid', force_act_layer=nn.ReLU),
        num_features=1280,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_lcnet(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """ LCNet
    Essentially a MobileNet-V3 crossed with a MobileNet-V1

    Paper: `PP-LCNet: A Lightweight CPU Convolutional Neural Network` - https://arxiv.org/abs/2109.15099

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['dsa_r1_k3_s1_c32'],
        # stage 1, 112x112 in
        ['dsa_r2_k3_s2_c64'],
        # stage 2, 56x56 in
        ['dsa_r2_k3_s2_c128'],
        # stage 3, 28x28 in
        ['dsa_r1_k3_s2_c256', 'dsa_r1_k5_s1_c256'],
        # stage 4, 14x14in
        ['dsa_r4_k5_s1_c256'],
        # stage 5, 14x14in
        ['dsa_r2_k5_s2_c512_se0.25'],
        # 7x7
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=16,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_layer=partial(SqueezeExcite, gate_layer='hard_sigmoid', force_act_layer=nn.ReLU),
        num_features=1280,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
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
    'mobilenetv3_large_075.untrained': _cfg(url=''),
    'mobilenetv3_large_100.ra_in1k': _cfg(
        interpolation='bicubic',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth',
        hf_hub_id='timm/'),
    'mobilenetv3_large_100.miil_in21k_ft_in1k': _cfg(
        interpolation='bilinear', mean=(0., 0., 0.), std=(1., 1., 1.),
        origin_url='https://github.com/Alibaba-MIIL/ImageNet21K',
        paper_ids='arXiv:2104.10972v4',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mobilenetv3_large_100_1k_miil_78_0-66471c13.pth',
        hf_hub_id='timm/'),
    'mobilenetv3_large_100.miil_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mobilenetv3_large_100_in21k_miil-d71cc17b.pth',
        hf_hub_id='timm/',
        origin_url='https://github.com/Alibaba-MIIL/ImageNet21K',
        paper_ids='arXiv:2104.10972v4',
        interpolation='bilinear', mean=(0., 0., 0.), std=(1., 1., 1.), num_classes=11221),

    'mobilenetv3_small_050.lamb_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_050_lambc-4b7bbe87.pth',
        hf_hub_id='timm/',
        interpolation='bicubic'),
    'mobilenetv3_small_075.lamb_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_075_lambc-384766db.pth',
        hf_hub_id='timm/',
        interpolation='bicubic'),
    'mobilenetv3_small_100.lamb_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_100_lamb-266a294c.pth',
        hf_hub_id='timm/',
        interpolation='bicubic'),

    'mobilenetv3_rw.rmsp_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_100-35495452.pth',
        interpolation='bicubic'),

    'tf_mobilenetv3_large_075.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_075-150ee8b0.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_100.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_100-427764d5.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_minimal_100.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_minimal_100-8596ae28.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_075.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_075-da427f52.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_100.in1k': _cfg(
        url= 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_100-37f49e2b.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_minimal_100.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_minimal_100-922a7843.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),

    'fbnetv3_b.ra2_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_b_224-ead5d2a1.pth',
        hf_hub_id='timm/',
        test_input_size=(3, 256, 256), crop_pct=0.95),
    'fbnetv3_d.ra2_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_d_224-c98bce42.pth',
        hf_hub_id='timm/',
        test_input_size=(3, 256, 256), crop_pct=0.95),
    'fbnetv3_g.ra2_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_g_240-0b1df83b.pth',
        hf_hub_id='timm/',
        input_size=(3, 240, 240), test_input_size=(3, 288, 288), crop_pct=0.95, pool_size=(8, 8)),

    "lcnet_035.untrained": _cfg(),
    "lcnet_050.ra2_in1k": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_050-f447553b.pth',
        hf_hub_id='timm/',
        interpolation='bicubic',
    ),
    "lcnet_075.ra2_in1k": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_075-318cad2c.pth',
        hf_hub_id='timm/',
        interpolation='bicubic',
    ),
    "lcnet_100.ra2_in1k": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_100-a929038c.pth',
        hf_hub_id='timm/',
        interpolation='bicubic',
    ),
    "lcnet_150.untrained": _cfg(),
})


@register_model
def mobilenetv3_large_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_large_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_large_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_large_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_small_050(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_050', 0.50, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_small_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_small_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_rw(pretrained=False, **kwargs):
    """ MobileNet V3 """
    if pretrained:
        # pretrained model trained with non-default BN epsilon
        kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    model = _gen_mobilenet_v3_rw('mobilenetv3_rw', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_large_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_large_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_large_minimal_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_minimal_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_small_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_small_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_small_minimal_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_minimal_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def fbnetv3_b(pretrained=False, **kwargs):
    """ FBNetV3-B """
    model = _gen_fbnetv3('fbnetv3_b', pretrained=pretrained, **kwargs)
    return model


@register_model
def fbnetv3_d(pretrained=False, **kwargs):
    """ FBNetV3-D """
    model = _gen_fbnetv3('fbnetv3_d', pretrained=pretrained, **kwargs)
    return model


@register_model
def fbnetv3_g(pretrained=False, **kwargs):
    """ FBNetV3-G """
    model = _gen_fbnetv3('fbnetv3_g', pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_035(pretrained=False, **kwargs):
    """ PP-LCNet 0.35"""
    model = _gen_lcnet('lcnet_035', 0.35, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_050(pretrained=False, **kwargs):
    """ PP-LCNet 0.5"""
    model = _gen_lcnet('lcnet_050', 0.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_075(pretrained=False, **kwargs):
    """ PP-LCNet 1.0"""
    model = _gen_lcnet('lcnet_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_100(pretrained=False, **kwargs):
    """ PP-LCNet 1.0"""
    model = _gen_lcnet('lcnet_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_150(pretrained=False, **kwargs):
    """ PP-LCNet 1.5"""
    model = _gen_lcnet('lcnet_150', 1.5, pretrained=pretrained, **kwargs)
    return model


register_model_deprecations(__name__, {
    'mobilenetv3_large_100_miil': 'mobilenetv3_large_100.miil_in21k_ft_in1k',
    'mobilenetv3_large_100_miil_in21k': 'mobilenetv3_large_100.miil_in21k',
})
