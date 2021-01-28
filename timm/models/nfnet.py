""" Normalizer Free RegNet / ResNet (pre-activation) Models

Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
    - https://arxiv.org/abs/2101.08692

Hacked together by / copyright Ross Wightman, 2021.
"""
import math
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Tuple, Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .registry import register_model
from .layers import ClassifierHead, DropPath, AvgPool2dSame, ScaledStdConv2d, get_act_layer, get_attn, make_divisible


def _dcfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
        **kwargs
    }

# FIXME finish
default_cfgs = {
    'nf_regnet_b0': _dcfg(url=''),
    'nf_regnet_b1': _dcfg(url='', input_size=(3, 240, 240)),
    'nf_regnet_b2': _dcfg(url='', input_size=(3, 256, 256)),
    'nf_regnet_b3': _dcfg(url='', input_size=(3, 272, 272)),
    'nf_regnet_b4': _dcfg(url='', input_size=(3, 320, 320)),
    'nf_regnet_b5': _dcfg(url='', input_size=(3, 384, 384)),

    'nf_resnet26d': _dcfg(url='', first_conv='stem.conv1'),
    'nf_resnet50d': _dcfg(url='', first_conv='stem.conv1'),
    'nf_resnet101d': _dcfg(url='', first_conv='stem.conv1'),

    'nf_seresnet26d': _dcfg(url='', first_conv='stem.conv1'),
    'nf_seresnet50d': _dcfg(url='', first_conv='stem.conv1'),
    'nf_seresnet101d': _dcfg(url='', first_conv='stem.conv1'),

    'nf_ecaresnet26d': _dcfg(url='', first_conv='stem.conv1'),
    'nf_ecaresnet50d': _dcfg(url='', first_conv='stem.conv1'),
    'nf_ecaresnet101d': _dcfg(url='', first_conv='stem.conv1'),
}


@dataclass
class NfCfg:
    depths: Tuple[int, int, int, int]
    channels: Tuple[int, int, int, int]
    alpha: float = 0.2
    stem_type: str = '3x3'
    stem_chs: Optional[int] = None
    group_size: Optional[int] = 8
    attn_layer: Optional[str] = 'se'
    attn_kwargs: dict = field(default_factory=lambda: dict(reduction_ratio=0.5, divisor=8))
    attn_gain: float = 2.0  # NF correction gain to apply if attn layer is used
    width_factor: float = 0.75
    bottle_ratio: float = 2.25
    efficient: bool = True  # enables EfficientNet-like options that are used in paper 'nf_regnet_b*' models
    num_features: int = 1280  # num out_channels for final conv (when enabled in efficient mode)
    ch_div: int = 8  # round channels % 8 == 0 to keep tensor-core use optimal
    skipinit: bool = False
    act_layer: str = 'silu'


model_cfgs = dict(
    # EffNet influenced RegNet defs
    nf_regnet_b0=NfCfg(depths=(1, 3, 6, 6), channels=(48, 104, 208, 440), num_features=1280),
    nf_regnet_b1=NfCfg(depths=(2, 4, 7, 7), channels=(48, 104, 208, 440), num_features=1280),
    nf_regnet_b2=NfCfg(depths=(2, 4, 8, 8), channels=(56, 112, 232, 488), num_features=1416),
    nf_regnet_b3=NfCfg(depths=(2, 5, 9, 9), channels=(56, 128, 248, 528), num_features=1536),
    nf_regnet_b4=NfCfg(depths=(2, 6, 11, 11), channels=(64, 144, 288, 616), num_features=1792),
    nf_regnet_b5=NfCfg(depths=(3, 7, 14, 14), channels=(80, 168, 336, 704), num_features=2048),

    # ResNet (preact, D style deep stem/avg down) defs
    nf_resnet26d=NfCfg(
        depths=(2, 2, 2, 2), channels=(256, 512, 1024, 2048),
        stem_type='deep', stem_chs=64, width_factor=1.0, bottle_ratio=0.25, efficient=False, group_size=None,
        act_layer='relu', attn_layer=None,),
    nf_resnet50d=NfCfg(
        depths=(3, 4, 6, 3), channels=(256, 512, 1024, 2048),
        stem_type='deep', stem_chs=64, width_factor=1.0, bottle_ratio=0.25, efficient=False, group_size=None,
        act_layer='relu', attn_layer=None),
    nf_resnet101d=NfCfg(
        depths=(3, 4, 6, 3), channels=(256, 512, 1024, 2048),
        stem_type='deep', stem_chs=64, width_factor=1.0, bottle_ratio=0.25, efficient=False, group_size=None,
        act_layer='relu', attn_layer=None),


    nf_seresnet26d=NfCfg(
        depths=(2, 2, 2, 2), channels=(256, 512, 1024, 2048),
        stem_type='deep', stem_chs=64, width_factor=1.0, bottle_ratio=0.25, efficient=False, group_size=None,
        act_layer='relu', attn_layer='se', attn_kwargs=dict(reduction_ratio=0.25)),
    nf_seresnet50d=NfCfg(
        depths=(3, 4, 6, 3), channels=(256, 512, 1024, 2048),
        stem_type='deep', stem_chs=64, width_factor=1.0, bottle_ratio=0.25, efficient=False, group_size=None,
        act_layer='relu', attn_layer='se', attn_kwargs=dict(reduction_ratio=0.25)),
    nf_seresnet101d=NfCfg(
        depths=(3, 4, 6, 3), channels=(256, 512, 1024, 2048),
        stem_type='deep', stem_chs=64, width_factor=1.0, bottle_ratio=0.25, efficient=False, group_size=None,
        act_layer='relu', attn_layer='se', attn_kwargs=dict(reduction_ratio=0.25)),


    nf_ecaresnet26d=NfCfg(
        depths=(2, 2, 2, 2), channels=(256, 512, 1024, 2048),
        stem_type='deep', stem_chs=64, width_factor=1.0, bottle_ratio=0.25, efficient=False, group_size=None,
        act_layer='relu', attn_layer='eca', attn_kwargs=dict()),
    nf_ecaresnet50d=NfCfg(
        depths=(3, 4, 6, 3), channels=(256, 512, 1024, 2048),
        stem_type='deep', stem_chs=64, width_factor=1.0, bottle_ratio=0.25, efficient=False, group_size=None,
        act_layer='relu', attn_layer='eca', attn_kwargs=dict()),
    nf_ecaresnet101d=NfCfg(
        depths=(3, 4, 6, 3), channels=(256, 512, 1024, 2048),
        stem_type='deep', stem_chs=64, width_factor=1.0, bottle_ratio=0.25, efficient=False, group_size=None,
        act_layer='relu', attn_layer='eca', attn_kwargs=dict()),

)

# class NormFreeSiLU(nn.Module):
#     _K = 1. / 0.5595
#     def __init__(self, inplace=False):
#         super().__init__()
#         self.inplace = inplace
#
#     def forward(self, x):
#         return F.silu(x, inplace=self.inplace) * self._K
#
#
# class NormFreeReLU(nn.Module):
#     _K = (0.5 * (1. - 1. / math.pi)) ** -0.5
#
#     def __init__(self, inplace=False):
#         super().__init__()
#         self.inplace = inplace
#
#     def forward(self, x):
#         return F.relu(x, inplace=self.inplace) * self._K


class DownsampleAvg(nn.Module):
    def __init__(
            self, in_chs, out_chs, stride=1, dilation=1, first_dilation=None, conv_layer=ScaledStdConv2d):
        """ AvgPool Downsampling as in 'D' ResNet variants. Support for dilation."""
        super(DownsampleAvg, self).__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=1)

    def forward(self, x):
        return self.conv(self.pool(x))


class NormalizationFreeBlock(nn.Module):
    """Normalization-free pre-activation block.
    """

    def __init__(
            self, in_chs, out_chs=None, stride=1, dilation=1, first_dilation=None,
            alpha=1.0, beta=1.0, bottle_ratio=0.25, efficient=True, ch_div=1, group_size=None,
            attn_layer=None, attn_gain=2.0, act_layer=None, conv_layer=None, drop_path_rate=0., skipinit=False):
        super().__init__()
        first_dilation = first_dilation or dilation
        out_chs = out_chs or in_chs
        # EfficientNet-like models scale bottleneck from in_chs, otherwise scale from out_chs like ResNet
        mid_chs = make_divisible(in_chs * bottle_ratio if efficient else out_chs * bottle_ratio, ch_div)
        groups = 1
        if group_size is not None:
            # NOTE: not correcting the mid_chs % group_size, fix model def if broken. I want % ch_div == 0 to stand.
            groups = mid_chs // group_size
        self.alpha = alpha
        self.beta = beta
        self.attn_gain = attn_gain

        if in_chs != out_chs or stride != 1 or dilation != first_dilation:
            self.downsample = DownsampleAvg(
                in_chs, out_chs, stride=stride, dilation=dilation, first_dilation=first_dilation, conv_layer=conv_layer)
        else:
            self.downsample = None

        self.act1 = act_layer()
        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.act2 = act_layer(inplace=True)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        if attn_layer is not None:
            self.attn = attn_layer(mid_chs)
        else:
            self.attn = None
        self.act3 = act_layer()
        self.conv3 = conv_layer(mid_chs, out_chs, 1)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.skipinit_gain = nn.Parameter(torch.tensor(0.)) if skipinit else None

    def forward(self, x):
        out = self.act1(x) * self.beta

        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(out)

        # residual branch
        out = self.conv1(out)
        out = self.conv2(self.act2(out))
        if self.attn is not None:
            out = self.attn_gain * self.attn(out)
        out = self.conv3(self.act3(out))
        out = self.drop_path(out)
        if self.skipinit_gain is None:
            out = out * self.alpha + shortcut
        else:
            # this really slows things down for some reason, TBD
            out = out * self.alpha * self.skipinit_gain + shortcut
        return out


def create_stem(in_chs, out_chs, stem_type='', conv_layer=None):
    stem = OrderedDict()
    assert stem_type in ('', 'deep', '3x3', '7x7')
    if 'deep' in stem_type:
        # 3 deep 3x3  conv stack as in ResNet V1D models
        mid_chs = out_chs // 2
        stem['conv1'] = conv_layer(in_chs, mid_chs, kernel_size=3, stride=2)
        stem['conv2'] = conv_layer(mid_chs, mid_chs, kernel_size=3, stride=1)
        stem['conv3'] = conv_layer(mid_chs, out_chs, kernel_size=3, stride=1)
    elif '3x3' in stem_type:
        # 3x3 stem conv as in RegNet
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=3, stride=2)
    else:
        # 7x7 stem conv as in ResNet
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=7, stride=2)

    return nn.Sequential(stem)


_nonlin_gamma = dict(
    silu=.5595,
    relu=(0.5 * (1. - 1. / math.pi)) ** 0.5,
    identity=1.0
)


class NormalizerFreeNet(nn.Module):
    """ Normalizer-free ResNets and RegNets

    As described in `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
        - https://arxiv.org/abs/2101.08692

    This model aims to cover both the NFRegNet-Bx models as detailed in the paper's code snippets and
    the (preact) ResNet models described earlier in the paper.

    There are a few differences:
        * channels are rounded to be divisible by 8 by default (keep TC happy), this changes param counts
        * activation correcting gamma constants are moved into the ScaledStdConv as it has less performance
            impact in PyTorch when done with the weight scaling there. This likely wasn't a concern in the JAX impl.
        * skipinit is disabled by default, it seems to have a rather drastic impact on GPU memory use and throughput
            for what it is/does. Approx 8-10% throughput loss.
    """
    def __init__(self, cfg: NfCfg, num_classes=1000, in_chans=3, global_pool='avg', output_stride=32,
                 drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        act_layer = get_act_layer(cfg.act_layer)
        assert cfg.act_layer in _nonlin_gamma, f"Please add non-linearity constants for activation ({cfg.act_layer})."
        conv_layer = partial(ScaledStdConv2d, bias=True, gain=True, gamma=_nonlin_gamma[cfg.act_layer])
        attn_layer = partial(get_attn(cfg.attn_layer), **cfg.attn_kwargs) if cfg.attn_layer else None

        self.feature_info = []  # FIXME fill out feature info

        stem_chs = cfg.stem_chs or cfg.channels[0]
        stem_chs = make_divisible(stem_chs * cfg.width_factor, cfg.ch_div)
        self.stem = create_stem(in_chans, stem_chs, cfg.stem_type, conv_layer=conv_layer)

        prev_chs = stem_chs
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]
        net_stride = 2
        dilation = 1
        expected_var = 1.0
        stages = []
        for stage_idx, stage_depth in enumerate(cfg.depths):
            if net_stride >= output_stride:
                dilation *= 2
                stride = 1
            else:
                stride = 2
            net_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2

            blocks = []
            for block_idx in range(cfg.depths[stage_idx]):
                first_block = block_idx == 0 and stage_idx == 0
                out_chs = make_divisible(cfg.channels[stage_idx] * cfg.width_factor, cfg.ch_div)
                blocks += [NormalizationFreeBlock(
                    in_chs=prev_chs, out_chs=out_chs,
                    alpha=cfg.alpha,
                    beta=1. / expected_var ** 0.5,  # NOTE: beta used as multiplier in block
                    stride=stride if block_idx == 0 else 1,
                    dilation=dilation,
                    first_dilation=first_dilation,
                    group_size=cfg.group_size,
                    bottle_ratio=1. if cfg.efficient and first_block else cfg.bottle_ratio,
                    efficient=cfg.efficient,
                    ch_div=cfg.ch_div,
                    attn_layer=attn_layer,
                    attn_gain=cfg.attn_gain,
                    act_layer=act_layer,
                    conv_layer=conv_layer,
                    drop_path_rate=dpr[stage_idx][block_idx],
                    skipinit=cfg.skipinit,
                )]
                if block_idx == 0:
                    expected_var = 1.  # expected var is reset after first block of each stage
                expected_var += cfg.alpha ** 2   # Even if reset occurs, increment expected variance
                first_dilation = dilation
                prev_chs = out_chs
            stages += [nn.Sequential(*blocks)]
        self.stages = nn.Sequential(*stages)

        if cfg.efficient and cfg.num_features:
            # The paper NFRegNet models have an EfficientNet-like final head convolution.
            self.num_features = make_divisible(cfg.width_factor * cfg.num_features, cfg.ch_div)
            self.final_conv = conv_layer(prev_chs, self.num_features, 1)
        else:
            self.num_features = prev_chs
            self.final_conv = nn.Identity()
        self.final_act = act_layer()
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

        for n, m in self.named_modules():
            if 'fc' in n and isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                # as per discussion with paper authors, original in haiku is
                # hk.initializers.VarianceScaling(1.0, 'fan_in', 'normal')' w/ zero'd bias
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.final_conv(x)
        x = self.final_act(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _create_normfreenet(variant, pretrained=False, **kwargs):
    feature_cfg = dict(flatten_sequential=True)
    feature_cfg['feature_cls'] = 'hook'  # pre-act models need hooks to grab feat from act1 in bottleneck blocks

    return build_model_with_cfg(
        NormalizerFreeNet, variant, pretrained, model_cfg=model_cfgs[variant], default_cfg=default_cfgs[variant],
        feature_cfg=feature_cfg, **kwargs)


@register_model
def nf_regnet_b0(pretrained=False, **kwargs):
    return _create_normfreenet('nf_regnet_b0', pretrained=pretrained, **kwargs)


@register_model
def nf_regnet_b1(pretrained=False, **kwargs):
    return _create_normfreenet('nf_regnet_b1', pretrained=pretrained, **kwargs)


@register_model
def nf_regnet_b2(pretrained=False, **kwargs):
    return _create_normfreenet('nf_regnet_b2', pretrained=pretrained, **kwargs)


@register_model
def nf_regnet_b3(pretrained=False, **kwargs):
    return _create_normfreenet('nf_regnet_b3', pretrained=pretrained, **kwargs)


@register_model
def nf_regnet_b4(pretrained=False, **kwargs):
    return _create_normfreenet('nf_regnet_b4', pretrained=pretrained, **kwargs)


@register_model
def nf_regnet_b5(pretrained=False, **kwargs):
    return _create_normfreenet('nf_regnet_b5', pretrained=pretrained, **kwargs)


@register_model
def nf_resnet26d(pretrained=False, **kwargs):
    return _create_normfreenet('nf_resnet26d', pretrained=pretrained, **kwargs)


@register_model
def nf_resnet50d(pretrained=False, **kwargs):
    return _create_normfreenet('nf_resnet50d', pretrained=pretrained, **kwargs)


@register_model
def nf_seresnet26d(pretrained=False, **kwargs):
    return _create_normfreenet('nf_seresnet26d', pretrained=pretrained, **kwargs)


@register_model
def nf_seresnet50d(pretrained=False, **kwargs):
    return _create_normfreenet('nf_seresnet50d', pretrained=pretrained, **kwargs)


@register_model
def nf_ecaresnet26d(pretrained=False, **kwargs):
    return _create_normfreenet('nf_ecaresnet26d', pretrained=pretrained, **kwargs)


@register_model
def nf_ecaresnet50d(pretrained=False, **kwargs):
    return _create_normfreenet('nf_ecaresnet50d', pretrained=pretrained, **kwargs)
