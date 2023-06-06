""" The EfficientNet Family in PyTorch

An implementation of EfficienNet that covers variety of related models with efficient architectures:

* EfficientNet-V2
  - `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298

* EfficientNet (B0-B8, L2 + Tensorflow pretrained AutoAug/RandAug/AdvProp/NoisyStudent weight ports)
  - EfficientNet: Rethinking Model Scaling for CNNs - https://arxiv.org/abs/1905.11946
  - CondConv: Conditionally Parameterized Convolutions for Efficient Inference - https://arxiv.org/abs/1904.04971
  - Adversarial Examples Improve Image Recognition - https://arxiv.org/abs/1911.09665
  - Self-training with Noisy Student improves ImageNet classification - https://arxiv.org/abs/1911.04252

* MixNet (Small, Medium, and Large)
  - MixConv: Mixed Depthwise Convolutional Kernels - https://arxiv.org/abs/1907.09595

* MNasNet B1, A1 (SE), Small
  - MnasNet: Platform-Aware Neural Architecture Search for Mobile - https://arxiv.org/abs/1807.11626

* FBNet-C
  - FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable NAS - https://arxiv.org/abs/1812.03443

* Single-Path NAS Pixel1
  - Single-Path NAS: Designing Hardware-Efficient ConvNets - https://arxiv.org/abs/1904.02877

* TinyNet
    - Model Rubik's Cube: Twisting Resolution, Depth and Width for TinyNets - https://arxiv.org/abs/2010.14819
    - Definitions & weights borrowed from https://github.com/huawei-noah/CV-Backbones/tree/master/tinynet_pytorch

* And likely more...

The majority of the above models (EfficientNet*, MixNet, MnasNet) and original weights were made available
by Mingxing Tan, Quoc Le, and other members of their Google Brain team. Thanks for consistently releasing
the models and weights open source!

Hacked together by / Copyright 2019, Ross Wightman
"""
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import create_conv2d, create_classifier, get_norm_act_layer, GroupNormAct
from ._builder import build_model_with_cfg, pretrained_cfg_for_features
from ._efficientnet_blocks import SqueezeExcite
from ._efficientnet_builder import EfficientNetBuilder, decode_arch_def, efficientnet_init_weights, \
    round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from ._features import FeatureInfo, FeatureHooks
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['EfficientNet', 'EfficientNetFeatures']


class EfficientNet(nn.Module):
    """ EfficientNet

    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet-V2 Small, Medium, Large, XL & B0-B3
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-CondConv
      * MixNet S, M, L, XL
      * MnasNet A1, B1, and small
      * MobileNet-V2
      * FBNet C
      * Single-Path NAS Pixel1
      * TinyNet
    """

    def __init__(
            self,
            block_args,
            num_classes=1000,
            num_features=1280,
            in_chans=3,
            stem_size=32,
            fix_stem=False,
            output_stride=32,
            pad_type='',
            round_chs_fn=round_channels,
            act_layer=None,
            norm_layer=None,
            se_layer=None,
            drop_rate=0.,
            drop_path_rate=0.,
            global_pool='avg'
    ):
        super(EfficientNet, self).__init__()
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
            output_stride=output_stride,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            act_layer=act_layer,
            norm_layer=norm_layer,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
        )
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head + Pooling
        self.conv_head = create_conv2d(head_chs, self.num_features, 1, padding=pad_type)
        self.bn2 = norm_act_layer(self.num_features, inplace=True)
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.global_pool])
        layers.extend([nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^conv_stem|bn1',
            blocks=[
                (r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)', None),
                (r'conv_head|bn2', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class EfficientNetFeatures(nn.Module):
    """ EfficientNet Feature Extractor

    A work-in-progress feature extraction module for EfficientNet, to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(
            self,
            block_args,
            out_indices=(0, 1, 2, 3, 4),
            feature_location='bottleneck',
            in_chans=3,
            stem_size=32,
            fix_stem=False,
            output_stride=32,
            pad_type='',
            round_chs_fn=round_channels,
            act_layer=None,
            norm_layer=None,
            se_layer=None,
            drop_rate=0.,
            drop_path_rate=0.
    ):
        super(EfficientNetFeatures, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_act_layer(stem_size, inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
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


def _create_effnet(variant, pretrained=False, **kwargs):
    features_only = False
    model_cls = EfficientNet
    kwargs_filter = None
    if kwargs.pop('features_only', False):
        features_only = True
        kwargs_filter = ('num_classes', 'num_features', 'head_conv', 'global_pool')
        model_cls = EfficientNetFeatures
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        pretrained_strict=not features_only,
        kwargs_filter=kwargs_filter,
        **kwargs)
    if features_only:
        model.default_cfg = pretrained_cfg_for_features(model.default_cfg)
    return model


def _gen_mnasnet_a1(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a mnasnet-a1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r2_k3_s2_e6_c24'],
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25'],
        # stage 3, 28x28 in
        ['ir_r4_k3_s2_e6_c80'],
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_mnasnet_b1(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a mnasnet-b1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r3_k3_s2_e3_c24'],
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40'],
        # stage 3, 28x28 in
        ['ir_r3_k5_s2_e6_c80'],
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c96'],
        # stage 5, 14x14in
        ['ir_r4_k5_s2_e6_c192'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320_noskip']
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_mnasnet_small(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a mnasnet-b1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        ['ds_r1_k3_s1_c8'],
        ['ir_r1_k3_s2_e3_c16'],
        ['ir_r2_k3_s2_e6_c16'],
        ['ir_r4_k5_s2_e6_c32_se0.25'],
        ['ir_r3_k3_s1_e6_c32_se0.25'],
        ['ir_r3_k5_s2_e6_c88_se0.25'],
        ['ir_r1_k3_s1_e6_c144']
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=8,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_mobilenet_v2(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, fix_stem_head=False, pretrained=False, **kwargs):
    """ Generate MobileNet-V2 network
    Ref impl: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
    Paper: https://arxiv.org/abs/1801.04381
    """
    arch_def = [
        ['ds_r1_k3_s1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r3_k3_s2_e6_c32'],
        ['ir_r4_k3_s2_e6_c64'],
        ['ir_r3_k3_s1_e6_c96'],
        ['ir_r3_k3_s2_e6_c160'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier=depth_multiplier, fix_first_last=fix_stem_head),
        num_features=1280 if fix_stem_head else max(1280, round_chs_fn(1280)),
        stem_size=32,
        fix_stem=fix_stem_head,
        round_chs_fn=round_chs_fn,
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'relu6'),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_fbnetc(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """ FBNet-C

        Paper: https://arxiv.org/abs/1812.03443
        Ref Impl: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_modeldef.py

        NOTE: the impl above does not relate to the 'C' variant here, that was derived from paper,
        it was used to confirm some building block details
    """
    arch_def = [
        ['ir_r1_k3_s1_e1_c16'],
        ['ir_r1_k3_s2_e6_c24', 'ir_r2_k3_s1_e1_c24'],
        ['ir_r1_k5_s2_e6_c32', 'ir_r1_k5_s1_e3_c32', 'ir_r1_k5_s1_e6_c32', 'ir_r1_k3_s1_e6_c32'],
        ['ir_r1_k5_s2_e6_c64', 'ir_r1_k5_s1_e3_c64', 'ir_r2_k5_s1_e6_c64'],
        ['ir_r3_k5_s1_e6_c112', 'ir_r1_k5_s1_e3_c112'],
        ['ir_r4_k5_s2_e6_c184'],
        ['ir_r1_k3_s1_e6_c352'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=16,
        num_features=1984,  # paper suggests this, but is not 100% clear
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_spnasnet(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates the Single-Path NAS model from search targeted for Pixel1 phone.

    Paper: https://arxiv.org/abs/1904.02877

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r3_k3_s2_e3_c24'],
        # stage 2, 56x56 in
        ['ir_r1_k5_s2_e6_c40', 'ir_r3_k3_s1_e3_c40'],
        # stage 3, 28x28 in
        ['ir_r1_k5_s2_e6_c80', 'ir_r3_k3_s1_e3_c80'],
        # stage 4, 14x14in
        ['ir_r1_k5_s1_e6_c96', 'ir_r3_k5_s1_e3_c96'],
        # stage 5, 14x14in
        ['ir_r4_k5_s2_e6_c192'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320_noskip']
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnet(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, channel_divisor=8,
        group_size=None, pretrained=False, **kwargs):
    """Creates an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, divisor=channel_divisor)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, group_size=group_size),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        act_layer=resolve_act_layer(kwargs, 'swish'),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnet_edge(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, group_size=None, pretrained=False, **kwargs):
    """ Creates an EfficientNet-EdgeTPU model

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu
    """

    arch_def = [
        # NOTE `fc` is present to override a mismatch between stem channels and in chs not
        # present in other models
        ['er_r1_k3_s1_e4_c24_fc24_noskip'],
        ['er_r2_k3_s2_e8_c32'],
        ['er_r4_k3_s2_e8_c48'],
        ['ir_r5_k5_s2_e8_c96'],
        ['ir_r4_k5_s1_e8_c144'],
        ['ir_r2_k5_s2_e8_c192'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, group_size=group_size),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'relu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnet_condconv(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, experts_multiplier=1, pretrained=False, **kwargs):
    """Creates an EfficientNet-CondConv model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/condconv
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25_cc4'],
        ['ir_r4_k5_s2_e6_c192_se0.25_cc4'],
        ['ir_r1_k3_s1_e6_c320_se0.25_cc4'],
    ]
    # NOTE unlike official impl, this one uses `cc<x>` option where x is the base number of experts for each stage and
    # the expert_multiplier increases that on a per-model basis as with depth/channel multipliers
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, experts_multiplier=experts_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'swish'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnet_lite(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates an EfficientNet-Lite model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
      'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
      'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
      'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
      'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
      'efficientnet-lite4': (1.4, 1.8, 300, 0.3),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r2_k5_s2_e6_c40'],
        ['ir_r3_k3_s2_e6_c80'],
        ['ir_r3_k5_s1_e6_c112'],
        ['ir_r4_k5_s2_e6_c192'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, fix_first_last=True),
        num_features=1280,
        stem_size=32,
        fix_stem=True,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        act_layer=resolve_act_layer(kwargs, 'relu6'),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnetv2_base(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """ Creates an EfficientNet-V2 base model

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298
    """
    arch_def = [
        ['cn_r1_k3_s1_e1_c16_skip'],
        ['er_r2_k3_s2_e4_c32'],
        ['er_r2_k3_s2_e4_c48'],
        ['ir_r3_k3_s2_e4_c96_se0.25'],
        ['ir_r5_k3_s1_e6_c112_se0.25'],
        ['ir_r8_k3_s2_e6_c192_se0.25'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, round_limit=0.)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnetv2_s(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, group_size=None, rw=False, pretrained=False, **kwargs):
    """ Creates an EfficientNet-V2 Small model

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298

    NOTE: `rw` flag sets up 'small' variant to behave like my initial v2 small model,
        before ref the impl was released.
    """
    arch_def = [
        ['cn_r2_k3_s1_e1_c24_skip'],
        ['er_r4_k3_s2_e4_c48'],
        ['er_r4_k3_s2_e4_c64'],
        ['ir_r6_k3_s2_e4_c128_se0.25'],
        ['ir_r9_k3_s1_e6_c160_se0.25'],
        ['ir_r15_k3_s2_e6_c256_se0.25'],
    ]
    num_features = 1280
    if rw:
        # my original variant, based on paper figure differs from the official release
        arch_def[0] = ['er_r2_k3_s1_e1_c24']
        arch_def[-1] = ['ir_r15_k3_s2_e6_c272_se0.25']
        num_features = 1792

    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, group_size=group_size),
        num_features=round_chs_fn(num_features),
        stem_size=24,
        round_chs_fn=round_chs_fn,
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnetv2_m(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """ Creates an EfficientNet-V2 Medium model

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298
    """

    arch_def = [
        ['cn_r3_k3_s1_e1_c24_skip'],
        ['er_r5_k3_s2_e4_c48'],
        ['er_r5_k3_s2_e4_c80'],
        ['ir_r7_k3_s2_e4_c160_se0.25'],
        ['ir_r14_k3_s1_e6_c176_se0.25'],
        ['ir_r18_k3_s2_e6_c304_se0.25'],
        ['ir_r5_k3_s1_e6_c512_se0.25'],
    ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=1280,
        stem_size=24,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnetv2_l(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """ Creates an EfficientNet-V2 Large model

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298
    """

    arch_def = [
        ['cn_r4_k3_s1_e1_c32_skip'],
        ['er_r7_k3_s2_e4_c64'],
        ['er_r7_k3_s2_e4_c96'],
        ['ir_r10_k3_s2_e4_c192_se0.25'],
        ['ir_r19_k3_s1_e6_c224_se0.25'],
        ['ir_r25_k3_s2_e6_c384_se0.25'],
        ['ir_r7_k3_s1_e6_c640_se0.25'],
    ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=1280,
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnetv2_xl(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """ Creates an EfficientNet-V2 Xtra-Large model

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298
    """

    arch_def = [
        ['cn_r4_k3_s1_e1_c32_skip'],
        ['er_r8_k3_s2_e4_c64'],
        ['er_r8_k3_s2_e4_c96'],
        ['ir_r16_k3_s2_e4_c192_se0.25'],
        ['ir_r24_k3_s1_e6_c256_se0.25'],
        ['ir_r32_k3_s2_e6_c512_se0.25'],
        ['ir_r8_k3_s1_e6_c640_se0.25'],
    ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=1280,
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_mixnet_s(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MixNet Small model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
    Paper: https://arxiv.org/abs/1907.09595
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_a1.1_p1.1_s2_e6_c24', 'ir_r1_k3_a1.1_p1.1_s1_e3_c24'],  # relu
        # stage 2, 56x56 in
        ['ir_r1_k3.5.7_s2_e6_c40_se0.5_nsw', 'ir_r3_k3.5_a1.1_p1.1_s1_e6_c40_se0.5_nsw'],  # swish
        # stage 3, 28x28 in
        ['ir_r1_k3.5.7_p1.1_s2_e6_c80_se0.25_nsw', 'ir_r2_k3.5_p1.1_s1_e6_c80_se0.25_nsw'],  # swish
        # stage 4, 14x14in
        ['ir_r1_k3.5.7_a1.1_p1.1_s1_e6_c120_se0.5_nsw', 'ir_r2_k3.5.7.9_a1.1_p1.1_s1_e3_c120_se0.5_nsw'],  # swish
        # stage 5, 14x14in
        ['ir_r1_k3.5.7.9.11_s2_e6_c200_se0.5_nsw', 'ir_r2_k3.5.7.9_p1.1_s1_e6_c200_se0.5_nsw'],  # swish
        # 7x7
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=1536,
        stem_size=16,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_mixnet_m(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MixNet Medium-Large model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
    Paper: https://arxiv.org/abs/1907.09595
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c24'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3.5.7_a1.1_p1.1_s2_e6_c32', 'ir_r1_k3_a1.1_p1.1_s1_e3_c32'],  # relu
        # stage 2, 56x56 in
        ['ir_r1_k3.5.7.9_s2_e6_c40_se0.5_nsw', 'ir_r3_k3.5_a1.1_p1.1_s1_e6_c40_se0.5_nsw'],  # swish
        # stage 3, 28x28 in
        ['ir_r1_k3.5.7_s2_e6_c80_se0.25_nsw', 'ir_r3_k3.5.7.9_a1.1_p1.1_s1_e6_c80_se0.25_nsw'],  # swish
        # stage 4, 14x14in
        ['ir_r1_k3_s1_e6_c120_se0.5_nsw', 'ir_r3_k3.5.7.9_a1.1_p1.1_s1_e3_c120_se0.5_nsw'],  # swish
        # stage 5, 14x14in
        ['ir_r1_k3.5.7.9_s2_e6_c200_se0.5_nsw', 'ir_r3_k3.5.7.9_p1.1_s1_e6_c200_se0.5_nsw'],  # swish
        # 7x7
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, depth_trunc='round'),
        num_features=1536,
        stem_size=24,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_tinynet(
    variant, model_width=1.0, depth_multiplier=1.0, pretrained=False, **kwargs
):
    """Creates a TinyNet model.
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'], ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'], ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'], ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, depth_trunc='round'),
        num_features=max(1280, round_channels(1280, model_width, 8, None)),
        stem_size=32,
        fix_stem=True,
        round_chs_fn=partial(round_channels, multiplier=model_width),
        act_layer=resolve_act_layer(kwargs, 'swish'),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'mnasnet_050.untrained': _cfg(),
    'mnasnet_075.untrained': _cfg(),
    'mnasnet_100.rmsp_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_b1-74cb7081.pth',
        hf_hub_id='timm/'),
    'mnasnet_140.untrained': _cfg(),

    'semnasnet_050.untrained': _cfg(),
    'semnasnet_075.rmsp_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/semnasnet_075-18710866.pth',
        hf_hub_id='timm/'),
    'semnasnet_100.rmsp_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_a1-d9418771.pth',
        hf_hub_id='timm/'),
    'semnasnet_140.untrained': _cfg(),
    'mnasnet_small.lamb_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_small_lamb-aff75073.pth',
        hf_hub_id='timm/'),

    'mobilenetv2_035.untrained': _cfg(),
    'mobilenetv2_050.lamb_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_050-3d30d450.pth',
        hf_hub_id='timm/',
        interpolation='bicubic',
    ),
    'mobilenetv2_075.untrained': _cfg(),
    'mobilenetv2_100.ra_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_100_ra-b33bc2c4.pth',
        hf_hub_id='timm/'),
    'mobilenetv2_110d.ra_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_110d_ra-77090ade.pth',
        hf_hub_id='timm/'),
    'mobilenetv2_120d.ra_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_120d_ra-5987e2ed.pth',
        hf_hub_id='timm/'),
    'mobilenetv2_140.ra_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_140_ra-21a4e913.pth',
        hf_hub_id='timm/'),

    'fbnetc_100.rmsp_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetc_100-c345b898.pth',
        hf_hub_id='timm/',
        interpolation='bilinear'),
    'spnasnet_100.rmsp_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/spnasnet_100-048bc3f4.pth',
        hf_hub_id='timm/',
        interpolation='bilinear'),

    # NOTE experimenting with alternate attention
    'efficientnet_b0.ra_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth',
        hf_hub_id='timm/'),
    'efficientnet_b1.ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth',
        hf_hub_id='timm/',
        test_input_size=(3, 256, 256), crop_pct=1.0),
    'efficientnet_b2.ra_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth',
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8), test_input_size=(3, 288, 288), crop_pct=1.0),
    'efficientnet_b3.ra2_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth',
        hf_hub_id='timm/',
        input_size=(3, 288, 288), pool_size=(9, 9), test_input_size=(3, 320, 320), crop_pct=1.0),
    'efficientnet_b4.ra2_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b4_ra2_320-7eb33cd5.pth',
        hf_hub_id='timm/',
        input_size=(3, 320, 320), pool_size=(10, 10), test_input_size=(3, 384, 384), crop_pct=1.0),
    'efficientnet_b5.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 448, 448), pool_size=(14, 14), crop_pct=1.0, crop_mode='squash'),
    'efficientnet_b5.sw_in12k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 416, 416), pool_size=(13, 13), crop_pct=0.95, num_classes=11821),
    'efficientnet_b6.untrained': _cfg(
        url='', input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'efficientnet_b7.untrained': _cfg(
        url='', input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'efficientnet_b8.untrained': _cfg(
        url='', input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),
    'efficientnet_l2.untrained': _cfg(
        url='', input_size=(3, 800, 800), pool_size=(25, 25), crop_pct=0.961),

    # FIXME experimental
    'efficientnet_b0_gn.untrained': _cfg(),
    'efficientnet_b0_g8_gn.untrained': _cfg(),
    'efficientnet_b0_g16_evos.untrained': _cfg(),
    'efficientnet_b3_gn.untrained': _cfg(
        input_size=(3, 288, 288), pool_size=(9, 9), test_input_size=(3, 320, 320), crop_pct=1.0),
    'efficientnet_b3_g8_gn.untrained': _cfg(
        input_size=(3, 288, 288), pool_size=(9, 9), test_input_size=(3, 320, 320), crop_pct=1.0),

    'efficientnet_es.ra_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_es_ra-f111e99c.pth',
        hf_hub_id='timm/'),
    'efficientnet_em.ra2_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_em_ra2-66250f76.pth',
        hf_hub_id='timm/',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'efficientnet_el.ra_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_el-3b455510.pth',
        hf_hub_id='timm/',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),

    'efficientnet_es_pruned.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_es_pruned75-1b7248cf.pth',
        hf_hub_id='timm/'),
    'efficientnet_el_pruned.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_el_pruned70-ef2a2ccf.pth',
        hf_hub_id='timm/',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),

    'efficientnet_cc_b0_4e.untrained': _cfg(),
    'efficientnet_cc_b0_8e.untrained': _cfg(),
    'efficientnet_cc_b1_8e.untrained': _cfg(input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),

    'efficientnet_lite0.ra_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_lite0_ra-37913777.pth',
        hf_hub_id='timm/'),
    'efficientnet_lite1.untrained': _cfg(
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'efficientnet_lite2.untrained': _cfg(
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'efficientnet_lite3.untrained': _cfg(
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'efficientnet_lite4.untrained': _cfg(
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),

    'efficientnet_b1_pruned.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/effnetb1_pruned-bea43a3a.pth',
        hf_hub_id='timm/',
        input_size=(3, 240, 240), pool_size=(8, 8),
        crop_pct=0.882, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'efficientnet_b2_pruned.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/effnetb2_pruned-08c1b27c.pth',
        hf_hub_id='timm/',
        input_size=(3, 260, 260), pool_size=(9, 9),
        crop_pct=0.890, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'efficientnet_b3_pruned.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/effnetb3_pruned-59ecf72d.pth',
        hf_hub_id='timm/',
        input_size=(3, 300, 300), pool_size=(10, 10),
        crop_pct=0.904, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),

    'efficientnetv2_rw_t.ra2_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_t_agc-3620981a.pth',
        hf_hub_id='timm/',
        input_size=(3, 224, 224), test_input_size=(3, 288, 288), pool_size=(7, 7), crop_pct=1.0),
    'gc_efficientnetv2_rw_t.agc_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gc_efficientnetv2_rw_t_agc-927a0bde.pth',
        hf_hub_id='timm/',
        input_size=(3, 224, 224), test_input_size=(3, 288, 288), pool_size=(7, 7), crop_pct=1.0),
    'efficientnetv2_rw_s.ra2_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_v2s_ra2_288-a6477665.pth',
        hf_hub_id='timm/',
        input_size=(3, 288, 288), test_input_size=(3, 384, 384), pool_size=(9, 9), crop_pct=1.0),
    'efficientnetv2_rw_m.agc_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_rw_m_agc-3d90cb1e.pth',
        hf_hub_id='timm/',
        input_size=(3, 320, 320), test_input_size=(3, 416, 416), pool_size=(10, 10), crop_pct=1.0),

    'efficientnetv2_s.untrained': _cfg(
        input_size=(3, 288, 288), test_input_size=(3, 384, 384), pool_size=(9, 9), crop_pct=1.0),
    'efficientnetv2_m.untrained': _cfg(
        input_size=(3, 320, 320), test_input_size=(3, 416, 416), pool_size=(10, 10), crop_pct=1.0),
    'efficientnetv2_l.untrained': _cfg(
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),
    'efficientnetv2_xl.untrained': _cfg(
        input_size=(3, 384, 384), test_input_size=(3, 512, 512), pool_size=(12, 12), crop_pct=1.0),

    'tf_efficientnet_b0.ns_jft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_ns-c0e6a31c.pth',
        hf_hub_id='timm/',
        input_size=(3, 224, 224)),
    'tf_efficientnet_b1.ns_jft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_ns-99dd0c41.pth',
        hf_hub_id='timm/',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_b2.ns_jft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_ns-00306e48.pth',
        hf_hub_id='timm/',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'tf_efficientnet_b3.ns_jft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_ns-9d44bf68.pth',
        hf_hub_id='timm/',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_b4.ns_jft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ns-d6313a46.pth',
        hf_hub_id='timm/',
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'tf_efficientnet_b5.ns_jft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth',
        hf_hub_id='timm/',
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'tf_efficientnet_b6.ns_jft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ns-51548356.pth',
        hf_hub_id='timm/',
        input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'tf_efficientnet_b7.ns_jft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth',
        hf_hub_id='timm/',
        input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'tf_efficientnet_l2.ns_jft_in1k_475': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns_475-bebbd00a.pth',
        hf_hub_id='timm/',
        input_size=(3, 475, 475), pool_size=(15, 15), crop_pct=0.936),
    'tf_efficientnet_l2.ns_jft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns-df73bb44.pth',
        hf_hub_id='timm/',
        input_size=(3, 800, 800), pool_size=(25, 25), crop_pct=0.96),

    'tf_efficientnet_b0.ap_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_ap-f262efe1.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, input_size=(3, 224, 224)),
    'tf_efficientnet_b1.ap_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_ap-44ef0a3d.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_b2.ap_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_ap-2f8e7636.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'tf_efficientnet_b3.ap_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_ap-aad25bdd.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_b4.ap_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ap-dedb23e6.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'tf_efficientnet_b5.ap_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ap-9e82fae8.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'tf_efficientnet_b6.ap_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ap-4ffb161f.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'tf_efficientnet_b7.ap_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ap-ddb28fec.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'tf_efficientnet_b8.ap_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b8_ap-00e169fa.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),

    'tf_efficientnet_b5.ra_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ra-9a3e5369.pth',
        hf_hub_id='timm/',
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'tf_efficientnet_b7.ra_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ra-6c08e654.pth',
        hf_hub_id='timm/',
        input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'tf_efficientnet_b8.ra_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b8_ra-572d5dd9.pth',
        hf_hub_id='timm/',
        input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),

    'tf_efficientnet_b0.aa_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_aa-827b6e33.pth',
        hf_hub_id='timm/',
        input_size=(3, 224, 224)),
    'tf_efficientnet_b1.aa_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_aa-ea7a6ee0.pth',
        hf_hub_id='timm/',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_b2.aa_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_aa-60c94f97.pth',
        hf_hub_id='timm/',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'tf_efficientnet_b3.aa_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_aa-84b4657e.pth',
        hf_hub_id='timm/',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_b4.aa_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_aa-818f208c.pth',
        hf_hub_id='timm/',
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'tf_efficientnet_b5.aa_in1k': _cfg(
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_aa-99018a74.pth',
        hf_hub_id='timm/',
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'tf_efficientnet_b6.aa_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_aa-80ba17e4.pth',
        hf_hub_id='timm/',
        input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'tf_efficientnet_b7.aa_in1k': _cfg(
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_aa-076e3472.pth',
        hf_hub_id='timm/',
        input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),

    'tf_efficientnet_b0.in1k': _cfg(
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0-0af12548.pth',
        #hf_hub_id='timm/',
        input_size=(3, 224, 224)),
    'tf_efficientnet_b1.in1k': _cfg(
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1-5c1377c4.pth',
        #hf_hub_id='timm/',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_b2.in1k': _cfg(
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2-e393ef04.pth',
        #hf_hub_id='timm/',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'tf_efficientnet_b3.in1k': _cfg(
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3-e3bd6955.pth',
        #hf_hub_id='timm/',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_b4.in1k': _cfg(
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4-74ee3bed.pth',
        #hf_hub_id='timm/',
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'tf_efficientnet_b5.in1k': _cfg(
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5-c6949ce9.pth',
        #hf_hub_id='timm/',
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),


    'tf_efficientnet_es.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_es-ca1afbfe.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 224, 224), ),
    'tf_efficientnet_em.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_em-e78cfe58.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_el.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_el-5143854e.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),

    'tf_efficientnet_cc_b0_4e.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b0_4e-4362b6b2.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_efficientnet_cc_b0_8e.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b0_8e-66184a25.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_efficientnet_cc_b1_8e.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b1_8e-f7c79ae1.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),

    'tf_efficientnet_lite0.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite0-0aa007d2.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
    ),
    'tf_efficientnet_lite1.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite1-bde8b488.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882,
        interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
    ),
    'tf_efficientnet_lite2.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite2-dcccb7df.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890,
        interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
    ),
    'tf_efficientnet_lite3.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite3-b733e338.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904, interpolation='bilinear'),
    'tf_efficientnet_lite4.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite4-741542c3.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.920, interpolation='bilinear'),

    'tf_efficientnetv2_s.in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_s_21ft1k-d7dafa41.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), test_input_size=(3, 384, 384), pool_size=(10, 10), crop_pct=1.0),
    'tf_efficientnetv2_m.in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_m_21ft1k-bf41664a.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'tf_efficientnetv2_l.in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_l_21ft1k-60127a9d.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'tf_efficientnetv2_xl.in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_xl_in21ft1k-06c35c48.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 512, 512), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),

    'tf_efficientnetv2_s.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_s-eb54923e.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), test_input_size=(3, 384, 384), pool_size=(10, 10), crop_pct=1.0),
    'tf_efficientnetv2_m.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_m-cc09e0cd.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'tf_efficientnetv2_l.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_l-d664b728.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),

    'tf_efficientnetv2_s.in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_s_21k-6337ad01.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), num_classes=21843,
        input_size=(3, 300, 300), test_input_size=(3, 384, 384), pool_size=(10, 10), crop_pct=1.0),
    'tf_efficientnetv2_m.in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_m_21k-361418a2.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), num_classes=21843,
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'tf_efficientnetv2_l.in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_l_21k-91a19ec9.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), num_classes=21843,
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'tf_efficientnetv2_xl.in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_xl_in21k-fd7e8abf.pth',
        hf_hub_id='timm/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), num_classes=21843,
        input_size=(3, 384, 384), test_input_size=(3, 512, 512), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),

    'tf_efficientnetv2_b0.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b0-c7cc451f.pth',
        hf_hub_id='timm/',
        input_size=(3, 192, 192), test_input_size=(3, 224, 224), pool_size=(6, 6)),
    'tf_efficientnetv2_b1.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b1-be6e41b0.pth',
        hf_hub_id='timm/',
        input_size=(3, 192, 192), test_input_size=(3, 240, 240), pool_size=(6, 6), crop_pct=0.882),
    'tf_efficientnetv2_b2.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b2-847de54e.pth',
        hf_hub_id='timm/',
        input_size=(3, 208, 208), test_input_size=(3, 260, 260), pool_size=(7, 7), crop_pct=0.890),
    'tf_efficientnetv2_b3.in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 240, 240), test_input_size=(3, 300, 300), pool_size=(8, 8), crop_pct=0.9, crop_mode='squash'),
    'tf_efficientnetv2_b3.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b3-57773f13.pth',
        hf_hub_id='timm/',
        input_size=(3, 240, 240), test_input_size=(3, 300, 300), pool_size=(8, 8), crop_pct=0.904),
    'tf_efficientnetv2_b3.in21k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, num_classes=21843,
        input_size=(3, 240, 240), test_input_size=(3, 300, 300), pool_size=(8, 8), crop_pct=0.904),

    'mixnet_s.ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_s-a907afbc.pth',
        hf_hub_id='timm/'),
    'mixnet_m.ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_m-4647fc68.pth',
        hf_hub_id='timm/'),
    'mixnet_l.ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_l-5a9a2ed8.pth',
        hf_hub_id='timm/'),
    'mixnet_xl.ra_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_xl_ra-aac3c00c.pth',
        hf_hub_id='timm/'),
    'mixnet_xxl.untrained': _cfg(),

    'tf_mixnet_s.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_s-89d3354b.pth',
        hf_hub_id='timm/'),
    'tf_mixnet_m.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_m-0f4d8805.pth',
        hf_hub_id='timm/'),
    'tf_mixnet_l.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_l-6c92e0c8.pth',
        hf_hub_id='timm/'),

    "tinynet_a.in1k": _cfg(
        input_size=(3, 192, 192), pool_size=(6, 6),  # int(224 * 0.86)
        url='https://github.com/huawei-noah/CV-Backbones/releases/download/v1.2.0/tinynet_a.pth',
        hf_hub_id='timm/'),
    "tinynet_b.in1k": _cfg(
        input_size=(3, 188, 188), pool_size=(6, 6),  # int(224 * 0.84)
        url='https://github.com/huawei-noah/CV-Backbones/releases/download/v1.2.0/tinynet_b.pth',
        hf_hub_id='timm/'),
    "tinynet_c.in1k": _cfg(
        input_size=(3, 184, 184), pool_size=(6, 6),  # int(224 * 0.825)
        url='https://github.com/huawei-noah/CV-Backbones/releases/download/v1.2.0/tinynet_c.pth',
        hf_hub_id='timm/'),
    "tinynet_d.in1k": _cfg(
        input_size=(3, 152, 152), pool_size=(5, 5),  # int(224 * 0.68)
        url='https://github.com/huawei-noah/CV-Backbones/releases/download/v1.2.0/tinynet_d.pth',
        hf_hub_id='timm/'),
    "tinynet_e.in1k": _cfg(
        input_size=(3, 106, 106), pool_size=(4, 4),  # int(224 * 0.475)
        url='https://github.com/huawei-noah/CV-Backbones/releases/download/v1.2.0/tinynet_e.pth',
        hf_hub_id='timm/'),
})


@register_model
def mnasnet_050(pretrained=False, **kwargs) -> EfficientNet:
    """ MNASNet B1, depth multiplier of 0.5. """
    model = _gen_mnasnet_b1('mnasnet_050', 0.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def mnasnet_075(pretrained=False, **kwargs) -> EfficientNet:
    """ MNASNet B1, depth multiplier of 0.75. """
    model = _gen_mnasnet_b1('mnasnet_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def mnasnet_100(pretrained=False, **kwargs) -> EfficientNet:
    """ MNASNet B1, depth multiplier of 1.0. """
    model = _gen_mnasnet_b1('mnasnet_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mnasnet_b1(pretrained=False, **kwargs) -> EfficientNet:
    """ MNASNet B1, depth multiplier of 1.0. """
    return mnasnet_100(pretrained, **kwargs)


@register_model
def mnasnet_140(pretrained=False, **kwargs) -> EfficientNet:
    """ MNASNet B1,  depth multiplier of 1.4 """
    model = _gen_mnasnet_b1('mnasnet_140', 1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def semnasnet_050(pretrained=False, **kwargs) -> EfficientNet:
    """ MNASNet A1 (w/ SE), depth multiplier of 0.5 """
    model = _gen_mnasnet_a1('semnasnet_050', 0.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def semnasnet_075(pretrained=False, **kwargs) -> EfficientNet:
    """ MNASNet A1 (w/ SE),  depth multiplier of 0.75. """
    model = _gen_mnasnet_a1('semnasnet_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def semnasnet_100(pretrained=False, **kwargs) -> EfficientNet:
    """ MNASNet A1 (w/ SE), depth multiplier of 1.0. """
    model = _gen_mnasnet_a1('semnasnet_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mnasnet_a1(pretrained=False, **kwargs) -> EfficientNet:
    """ MNASNet A1 (w/ SE), depth multiplier of 1.0. """
    return semnasnet_100(pretrained, **kwargs)


@register_model
def semnasnet_140(pretrained=False, **kwargs) -> EfficientNet:
    """ MNASNet A1 (w/ SE), depth multiplier of 1.4. """
    model = _gen_mnasnet_a1('semnasnet_140', 1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def mnasnet_small(pretrained=False, **kwargs) -> EfficientNet:
    """ MNASNet Small,  depth multiplier of 1.0. """
    model = _gen_mnasnet_small('mnasnet_small', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_035(pretrained=False, **kwargs) -> EfficientNet:
    """ MobileNet V2 w/ 0.35 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_035', 0.35, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_050(pretrained=False, **kwargs) -> EfficientNet:
    """ MobileNet V2 w/ 0.5 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_050', 0.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_075(pretrained=False, **kwargs) -> EfficientNet:
    """ MobileNet V2 w/ 0.75 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_100(pretrained=False, **kwargs) -> EfficientNet:
    """ MobileNet V2 w/ 1.0 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_140(pretrained=False, **kwargs) -> EfficientNet:
    """ MobileNet V2 w/ 1.4 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_140', 1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_110d(pretrained=False, **kwargs) -> EfficientNet:
    """ MobileNet V2 w/ 1.1 channel, 1.2 depth multipliers"""
    model = _gen_mobilenet_v2(
        'mobilenetv2_110d', 1.1, depth_multiplier=1.2, fix_stem_head=True, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_120d(pretrained=False, **kwargs) -> EfficientNet:
    """ MobileNet V2 w/ 1.2 channel, 1.4 depth multipliers """
    model = _gen_mobilenet_v2(
        'mobilenetv2_120d', 1.2, depth_multiplier=1.4, fix_stem_head=True, pretrained=pretrained, **kwargs)
    return model


@register_model
def fbnetc_100(pretrained=False, **kwargs) -> EfficientNet:
    """ FBNet-C """
    if pretrained:
        # pretrained model trained with non-default BN epsilon
        kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    model = _gen_fbnetc('fbnetc_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def spnasnet_100(pretrained=False, **kwargs) -> EfficientNet:
    """ Single-Path NAS Pixel1"""
    model = _gen_spnasnet('spnasnet_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b0(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b1(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B1 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b2(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B2 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b2a(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B2 @ 288x288 w/ 1.0 test crop"""
    # WARN this model def is deprecated, different train/test res + test crop handled by default_cfg now
    return efficientnet_b2(pretrained=pretrained, **kwargs)


@register_model
def efficientnet_b3(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B3 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b3a(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B3 @ 320x320 w/ 1.0 test crop-pct """
    # WARN this model def is deprecated, different train/test res + test crop handled by default_cfg now
    return efficientnet_b3(pretrained=pretrained, **kwargs)


@register_model
def efficientnet_b4(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B4 """
    # NOTE for train, drop_rate should be 0.4, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b5(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B5 """
    # NOTE for train, drop_rate should be 0.4, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b5', channel_multiplier=1.6, depth_multiplier=2.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b6(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B6 """
    # NOTE for train, drop_rate should be 0.5, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b6', channel_multiplier=1.8, depth_multiplier=2.6, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b7(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B7 """
    # NOTE for train, drop_rate should be 0.5, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b7', channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b8(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B8 """
    # NOTE for train, drop_rate should be 0.5, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b8', channel_multiplier=2.2, depth_multiplier=3.6, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_l2(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-L2."""
    # NOTE for train, drop_rate should be 0.5, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_l2', channel_multiplier=4.3, depth_multiplier=5.3, pretrained=pretrained, **kwargs)
    return model


# FIXME experimental group cong / GroupNorm / EvoNorm experiments
@register_model
def efficientnet_b0_gn(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B0 + GroupNorm"""
    model = _gen_efficientnet(
        'efficientnet_b0_gn', norm_layer=partial(GroupNormAct, group_size=8), pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b0_g8_gn(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B0 w/ group conv + GroupNorm"""
    model = _gen_efficientnet(
        'efficientnet_b0_g8_gn', group_size=8, norm_layer=partial(GroupNormAct, group_size=8),
        pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b0_g16_evos(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B0 w/ group 16 conv + EvoNorm"""
    model = _gen_efficientnet(
        'efficientnet_b0_g16_evos', group_size=16, channel_divisor=16,
        pretrained=pretrained, **kwargs) #norm_layer=partial(EvoNorm2dS0, group_size=16),
    return model


@register_model
def efficientnet_b3_gn(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B3 w/ GroupNorm """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b3_gn', channel_multiplier=1.2, depth_multiplier=1.4, channel_divisor=16,
        norm_layer=partial(GroupNormAct, group_size=16), pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b3_g8_gn(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B3 w/ grouped conv + BN"""
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b3_g8_gn', channel_multiplier=1.2, depth_multiplier=1.4, group_size=8, channel_divisor=16,
        norm_layer=partial(GroupNormAct, group_size=16), pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_es(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Edge Small. """
    model = _gen_efficientnet_edge(
        'efficientnet_es', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_es_pruned(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Edge Small Pruned. For more info: https://github.com/DeGirum/pruned-models/releases/tag/efficientnet_v1.0"""
    model = _gen_efficientnet_edge(
        'efficientnet_es_pruned', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model

@register_model
def efficientnet_em(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Edge-Medium. """
    model = _gen_efficientnet_edge(
        'efficientnet_em', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_el(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Edge-Large. """
    model = _gen_efficientnet_edge(
        'efficientnet_el', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model

@register_model
def efficientnet_el_pruned(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Edge-Large pruned. For more info: https://github.com/DeGirum/pruned-models/releases/tag/efficientnet_v1.0"""
    model = _gen_efficientnet_edge(
        'efficientnet_el_pruned', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model

@register_model
def efficientnet_cc_b0_4e(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-CondConv-B0 w/ 8 Experts """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet_condconv(
        'efficientnet_cc_b0_4e', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_cc_b0_8e(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-CondConv-B0 w/ 8 Experts """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet_condconv(
        'efficientnet_cc_b0_8e', channel_multiplier=1.0, depth_multiplier=1.0, experts_multiplier=2,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_cc_b1_8e(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-CondConv-B1 w/ 8 Experts """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet_condconv(
        'efficientnet_cc_b1_8e', channel_multiplier=1.0, depth_multiplier=1.1, experts_multiplier=2,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_lite0(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Lite0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet_lite(
        'efficientnet_lite0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_lite1(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Lite1 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet_lite(
        'efficientnet_lite1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_lite2(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Lite2 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    model = _gen_efficientnet_lite(
        'efficientnet_lite2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_lite3(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Lite3 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    model = _gen_efficientnet_lite(
        'efficientnet_lite3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_lite4(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Lite4 """
    # NOTE for train, drop_rate should be 0.4, drop_path_rate should be 0.2
    model = _gen_efficientnet_lite(
        'efficientnet_lite4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b1_pruned(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B1 Pruned. The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    variant = 'efficientnet_b1_pruned'
    model = _gen_efficientnet(
        variant, channel_multiplier=1.0, depth_multiplier=1.1, pruned=True, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b2_pruned(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B2 Pruned. The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'efficientnet_b2_pruned', channel_multiplier=1.1, depth_multiplier=1.2, pruned=True,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b3_pruned(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B3 Pruned. The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'efficientnet_b3_pruned', channel_multiplier=1.2, depth_multiplier=1.4, pruned=True,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_rw_t(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2 Tiny (Custom variant, tiny not in paper). """
    model = _gen_efficientnetv2_s(
        'efficientnetv2_rw_t', channel_multiplier=0.8, depth_multiplier=0.9, rw=False, pretrained=pretrained, **kwargs)
    return model


@register_model
def gc_efficientnetv2_rw_t(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2 Tiny w/ Global Context Attn (Custom variant, tiny not in paper). """
    model = _gen_efficientnetv2_s(
        'gc_efficientnetv2_rw_t', channel_multiplier=0.8, depth_multiplier=0.9,
        rw=False, se_layer='gc', pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_rw_s(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2 Small (RW variant).
    NOTE: This is my initial (pre official code release) w/ some differences.
    See efficientnetv2_s and tf_efficientnetv2_s for versions that match the official w/ PyTorch vs TF padding
    """
    model = _gen_efficientnetv2_s('efficientnetv2_rw_s', rw=True, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_rw_m(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2 Medium (RW variant).
    """
    model = _gen_efficientnetv2_s(
        'efficientnetv2_rw_m', channel_multiplier=1.2, depth_multiplier=(1.2,) * 4 + (1.6,) * 2, rw=True,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_s(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2 Small. """
    model = _gen_efficientnetv2_s('efficientnetv2_s', pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_m(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2 Medium. """
    model = _gen_efficientnetv2_m('efficientnetv2_m', pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_l(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2 Large. """
    model = _gen_efficientnetv2_l('efficientnetv2_l', pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_xl(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2 Xtra-Large. """
    model = _gen_efficientnetv2_xl('efficientnetv2_xl', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b0(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B0. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b1(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B1. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b2(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B2. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b3(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B3. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b4(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B4. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b5(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B5. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b5', channel_multiplier=1.6, depth_multiplier=2.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b6(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B6. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b6', channel_multiplier=1.8, depth_multiplier=2.6, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b7(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B7. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b7', channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b8(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-B8. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b8', channel_multiplier=2.2, depth_multiplier=3.6, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_l2(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-L2 NoisyStudent. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_l2', channel_multiplier=4.3, depth_multiplier=5.3, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_es(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Edge Small. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_edge(
        'tf_efficientnet_es', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_em(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Edge-Medium. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_edge(
        'tf_efficientnet_em', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_el(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Edge-Large. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_edge(
        'tf_efficientnet_el', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_cc_b0_4e(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-CondConv-B0 w/ 4 Experts. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_condconv(
        'tf_efficientnet_cc_b0_4e', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_cc_b0_8e(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-CondConv-B0 w/ 8 Experts. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_condconv(
        'tf_efficientnet_cc_b0_8e', channel_multiplier=1.0, depth_multiplier=1.0, experts_multiplier=2,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_cc_b1_8e(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-CondConv-B1 w/ 8 Experts. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_condconv(
        'tf_efficientnet_cc_b1_8e', channel_multiplier=1.0, depth_multiplier=1.1, experts_multiplier=2,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_lite0(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Lite0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_lite1(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Lite1 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_lite2(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Lite2 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_lite3(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Lite3 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_lite4(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-Lite4 """
    # NOTE for train, drop_rate should be 0.4, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_s(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2 Small. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_s('tf_efficientnetv2_s', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_m(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2 Medium. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_m('tf_efficientnetv2_m', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_l(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2 Large. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_l('tf_efficientnetv2_l', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_xl(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2 Xtra-Large. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_xl('tf_efficientnetv2_xl', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_b0(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2-B0. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_base('tf_efficientnetv2_b0', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_b1(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2-B1. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_base(
        'tf_efficientnetv2_b1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_b2(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2-B2. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_base(
        'tf_efficientnetv2_b2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_b3(pretrained=False, **kwargs) -> EfficientNet:
    """ EfficientNet-V2-B3. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_base(
        'tf_efficientnetv2_b3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def mixnet_s(pretrained=False, **kwargs) -> EfficientNet:
    """Creates a MixNet Small model.
    """
    model = _gen_mixnet_s(
        'mixnet_s', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mixnet_m(pretrained=False, **kwargs) -> EfficientNet:
    """Creates a MixNet Medium model.
    """
    model = _gen_mixnet_m(
        'mixnet_m', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mixnet_l(pretrained=False, **kwargs) -> EfficientNet:
    """Creates a MixNet Large model.
    """
    model = _gen_mixnet_m(
        'mixnet_l', channel_multiplier=1.3, pretrained=pretrained, **kwargs)
    return model


@register_model
def mixnet_xl(pretrained=False, **kwargs) -> EfficientNet:
    """Creates a MixNet Extra-Large model.
    Not a paper spec, experimental def by RW w/ depth scaling.
    """
    model = _gen_mixnet_m(
        'mixnet_xl', channel_multiplier=1.6, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def mixnet_xxl(pretrained=False, **kwargs) -> EfficientNet:
    """Creates a MixNet Double Extra Large model.
    Not a paper spec, experimental def by RW w/ depth scaling.
    """
    model = _gen_mixnet_m(
        'mixnet_xxl', channel_multiplier=2.4, depth_multiplier=1.3, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mixnet_s(pretrained=False, **kwargs) -> EfficientNet:
    """Creates a MixNet Small model. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mixnet_s(
        'tf_mixnet_s', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mixnet_m(pretrained=False, **kwargs) -> EfficientNet:
    """Creates a MixNet Medium model. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mixnet_m(
        'tf_mixnet_m', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mixnet_l(pretrained=False, **kwargs) -> EfficientNet:
    """Creates a MixNet Large model. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mixnet_m(
        'tf_mixnet_l', channel_multiplier=1.3, pretrained=pretrained, **kwargs)
    return model


@register_model
def tinynet_a(pretrained=False, **kwargs) -> EfficientNet:
    model = _gen_tinynet('tinynet_a', 1.0, 1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tinynet_b(pretrained=False, **kwargs) -> EfficientNet:
    model = _gen_tinynet('tinynet_b', 0.75, 1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tinynet_c(pretrained=False, **kwargs) -> EfficientNet:
    model = _gen_tinynet('tinynet_c', 0.54, 0.85, pretrained=pretrained, **kwargs)
    return model


@register_model
def tinynet_d(pretrained=False, **kwargs) -> EfficientNet:
    model = _gen_tinynet('tinynet_d', 0.54, 0.695, pretrained=pretrained, **kwargs)
    return model


@register_model
def tinynet_e(pretrained=False, **kwargs) -> EfficientNet:
    model = _gen_tinynet('tinynet_e', 0.51, 0.6, pretrained=pretrained, **kwargs)
    return model


register_model_deprecations(__name__, {
    'tf_efficientnet_b0_ap': 'tf_efficientnet_b0.ap_in1k',
    'tf_efficientnet_b1_ap': 'tf_efficientnet_b1.ap_in1k',
    'tf_efficientnet_b2_ap': 'tf_efficientnet_b2.ap_in1k',
    'tf_efficientnet_b3_ap': 'tf_efficientnet_b3.ap_in1k',
    'tf_efficientnet_b4_ap': 'tf_efficientnet_b4.ap_in1k',
    'tf_efficientnet_b5_ap': 'tf_efficientnet_b5.ap_in1k',
    'tf_efficientnet_b6_ap': 'tf_efficientnet_b6.ap_in1k',
    'tf_efficientnet_b7_ap': 'tf_efficientnet_b7.ap_in1k',
    'tf_efficientnet_b8_ap': 'tf_efficientnet_b8.ap_in1k',
    'tf_efficientnet_b0_ns': 'tf_efficientnet_b0.ns_jft_in1k',
    'tf_efficientnet_b1_ns': 'tf_efficientnet_b1.ns_jft_in1k',
    'tf_efficientnet_b2_ns': 'tf_efficientnet_b2.ns_jft_in1k',
    'tf_efficientnet_b3_ns': 'tf_efficientnet_b3.ns_jft_in1k',
    'tf_efficientnet_b4_ns': 'tf_efficientnet_b4.ns_jft_in1k',
    'tf_efficientnet_b5_ns': 'tf_efficientnet_b5.ns_jft_in1k',
    'tf_efficientnet_b6_ns': 'tf_efficientnet_b6.ns_jft_in1k',
    'tf_efficientnet_b7_ns': 'tf_efficientnet_b7.ns_jft_in1k',
    'tf_efficientnet_l2_ns_475': 'tf_efficientnet_l2.ns_jft_in1k_475',
    'tf_efficientnet_l2_ns': 'tf_efficientnet_l2.ns_jft_in1k',
    'tf_efficientnetv2_s_in21ft1k': 'tf_efficientnetv2_s.in21k_ft_in1k',
    'tf_efficientnetv2_m_in21ft1k': 'tf_efficientnetv2_m.in21k_ft_in1k',
    'tf_efficientnetv2_l_in21ft1k': 'tf_efficientnetv2_l.in21k_ft_in1k',
    'tf_efficientnetv2_xl_in21ft1k': 'tf_efficientnetv2_xl.in21k_ft_in1k',
    'tf_efficientnetv2_s_in21k': 'tf_efficientnetv2_s.in21k',
    'tf_efficientnetv2_m_in21k': 'tf_efficientnetv2_m.in21k',
    'tf_efficientnetv2_l_in21k': 'tf_efficientnetv2_l.in21k',
    'tf_efficientnetv2_xl_in21k': 'tf_efficientnetv2_xl.in21k',
})
