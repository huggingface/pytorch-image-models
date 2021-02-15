""" Bring-Your-Own-Blocks Network

A flexible network w/ dataclass based config for stacking those NN blocks.

This model is currently used to implement the following networks:

GPU Efficient (ResNets) - gernet_l/m/s (original versions called genet, but this was already used (by SENet author)).
Paper: `Neural Architecture Design for GPU-Efficient Networks` - https://arxiv.org/abs/2006.14090
Code and weights: https://github.com/idstcv/GPU-Efficient-Networks, licensed Apache 2.0

RepVGG - repvgg_*
Paper: `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
Code and weights: https://github.com/DingXiaoH/RepVGG, licensed MIT

In all cases the models have been modified to fit within the design of ByobNet. I've remapped
the original weights and verified accuracies.

For GPU Efficient nets, I used the original names for the blocks since they were for the most part
the same as original residual blocks in ResNe(X)t, DarkNet, and other existing models. Note also some
changes introduced in RegNet were also present in the stem and bottleneck blocks for this model.

A significant number of different network archs can be implemented here, including variants of the
above nets that include attention.

Hacked together by / copyright Ross Wightman, 2021.
"""
import math
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Tuple, Dict, Optional, Union, Any, Callable
from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import ClassifierHead, ConvBnAct, DropPath, AvgPool2dSame, \
    create_conv2d, get_act_layer, get_attn, convert_norm_act, make_divisible
from .registry import register_model

__all__ = ['ByobNet', 'ByobCfg', 'BlocksCfg']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = {
    # GPU-Efficient (ResNet) weights
    'gernet_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-ger-weights/gernet_s-756b4751.pth'),
    'gernet_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-ger-weights/gernet_m-0873c53a.pth'),
    'gernet_l': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-ger-weights/gernet_l-f31e2e8d.pth',
        input_size=(3, 256, 256), pool_size=(8, 8)),

    # RepVGG weights
    'repvgg_a2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-repvgg-weights/repvgg_a2-c1ee6d2b.pth',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv')),
    'repvgg_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-repvgg-weights/repvgg_b0-80ac3f1b.pth',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv')),
    'repvgg_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-repvgg-weights/repvgg_b1-77ca2989.pth',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv')),
    'repvgg_b1g4': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-repvgg-weights/repvgg_b1g4-abde5d92.pth',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv')),
    'repvgg_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-repvgg-weights/repvgg_b2-25b7494e.pth',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv')),
    'repvgg_b2g4': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-repvgg-weights/repvgg_b2g4-165a85f2.pth',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv')),
    'repvgg_b3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-repvgg-weights/repvgg_b3-199bc50d.pth',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv')),
    'repvgg_b3g4': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-repvgg-weights/repvgg_b3g4-73c370bf.pth',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv')),
}


@dataclass
class BlocksCfg:
    type: Union[str, nn.Module]
    d: int  # block depth (number of block repeats in stage)
    c: int  # number of output channels for each block in stage
    s: int = 2  # stride of stage (first block)
    gs: Optional[Union[int, Callable]] = None  # group-size of blocks in stage, conv is depthwise if gs == 1
    br: float = 1.  # bottleneck-ratio of blocks in stage


@dataclass
class ByobCfg:
    blocks: Tuple[BlocksCfg, ...]
    downsample: str = 'conv1x1'
    stem_type: str = '3x3'
    stem_chs: int = 32
    width_factor: float = 1.0
    num_features: int = 0  # num out_channels for final conv, no final 1x1 conv if 0
    zero_init_last_bn: bool = True

    act_layer: str = 'relu'
    norm_layer: nn.Module = nn.BatchNorm2d
    attn_layer: Optional[str] = None
    attn_kwargs: dict = field(default_factory=lambda: dict())


def _rep_vgg_bcfg(d=(4, 6, 16, 1), wf=(1., 1., 1., 1.), groups=0):
    c = (64, 128, 256, 512)
    group_size = 0
    if groups > 0:
        group_size = lambda chs, idx: chs // groups if (idx + 1) % 2 == 0 else 0
    bcfg = tuple([BlocksCfg(type='rep', d=d, c=c * wf, gs=group_size) for d, c, wf in zip(d, c, wf)])
    return bcfg


model_cfgs = dict(

    gernet_l=ByobCfg(
        blocks=(
            BlocksCfg(type='basic', d=1, c=128, s=2, gs=0, br=1.),
            BlocksCfg(type='basic', d=2, c=192, s=2, gs=0, br=1.),
            BlocksCfg(type='bottle', d=6, c=640, s=2, gs=0, br=1 / 4),
            BlocksCfg(type='bottle', d=5, c=640, s=2, gs=1, br=3.),
            BlocksCfg(type='bottle', d=4, c=640, s=1, gs=1, br=3.),
        ),
        stem_chs=32,
        num_features=2560,
    ),
    gernet_m=ByobCfg(
        blocks=(
            BlocksCfg(type='basic', d=1, c=128, s=2, gs=0, br=1.),
            BlocksCfg(type='basic', d=2, c=192, s=2, gs=0, br=1.),
            BlocksCfg(type='bottle', d=6, c=640, s=2, gs=0, br=1 / 4),
            BlocksCfg(type='bottle', d=4, c=640, s=2, gs=1, br=3.),
            BlocksCfg(type='bottle', d=1, c=640, s=1, gs=1, br=3.),
        ),
        stem_chs=32,
        num_features=2560,
    ),
    gernet_s=ByobCfg(
        blocks=(
            BlocksCfg(type='basic', d=1, c=48, s=2, gs=0, br=1.),
            BlocksCfg(type='basic', d=3, c=48, s=2, gs=0, br=1.),
            BlocksCfg(type='bottle', d=7, c=384, s=2, gs=0, br=1 / 4),
            BlocksCfg(type='bottle', d=2, c=560, s=2, gs=1, br=3.),
            BlocksCfg(type='bottle', d=1, c=256, s=1, gs=1, br=3.),
        ),
        stem_chs=13,
        num_features=1920,
    ),

    repvgg_a2=ByobCfg(
        blocks=_rep_vgg_bcfg(d=(2, 4, 14, 1), wf=(1.5, 1.5, 1.5, 2.75)),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_b0=ByobCfg(
        blocks=_rep_vgg_bcfg(wf=(1., 1., 1., 2.5)),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_b1=ByobCfg(
        blocks=_rep_vgg_bcfg(wf=(2., 2., 2., 4.)),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_b1g4=ByobCfg(
        blocks=_rep_vgg_bcfg(wf=(2., 2., 2., 4.), groups=4),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_b2=ByobCfg(
        blocks=_rep_vgg_bcfg(wf=(2.5, 2.5, 2.5, 5.)),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_b2g4=ByobCfg(
        blocks=_rep_vgg_bcfg(wf=(2.5, 2.5, 2.5, 5.), groups=4),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_b3=ByobCfg(
        blocks=_rep_vgg_bcfg(wf=(3., 3., 3., 5.)),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_b3g4=ByobCfg(
        blocks=_rep_vgg_bcfg(wf=(3., 3., 3., 5.), groups=4),
        stem_type='rep',
        stem_chs=64,
    ),
)


def _na_args(cfg: dict):
    return dict(
        norm_layer=cfg.get('norm_layer', nn.BatchNorm2d),
        act_layer=cfg.get('act_layer', nn.ReLU))


def _ex_tuple(cfg: dict, *names):
    return tuple([cfg.get(n, None) for n in names])


def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class DownsampleAvg(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, dilation=1, apply_act=False, norm_layer=None, act_layer=None):
        """ AvgPool Downsampling as in 'D' ResNet variants."""
        super(DownsampleAvg, self).__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()
        self.conv = ConvBnAct(in_chs, out_chs, 1, apply_act=apply_act, norm_layer=norm_layer, act_layer=act_layer)

    def forward(self, x):
        return self.conv(self.pool(x))


def create_downsample(type, **kwargs):
    if type == 'avg':
        return DownsampleAvg(**kwargs)
    else:
        return ConvBnAct(kwargs.pop('in_chs'), kwargs.pop('out_chs'), kernel_size=1, **kwargs)


class BasicBlock(nn.Module):
    """ ResNet Basic Block - kxk + kxk
    """

    def __init__(
            self, in_chs, out_chs, kernel_size=3, stride=1, dilation=(1, 1), group_size=None, bottle_ratio=1.0,
            downsample='avg', linear_out=False, layer_cfg=None, drop_block=None, drop_path_rate=0.):
        super(BasicBlock, self).__init__()
        layer_cfg = layer_cfg or {}
        act_layer, attn_layer = _ex_tuple(layer_cfg, 'act_layer', 'attn_layer')
        layer_args = _na_args(layer_cfg)
        mid_chs = make_divisible(out_chs * bottle_ratio)
        groups = num_groups(group_size, mid_chs)

        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = create_downsample(
                downsample, in_chs=in_chs, out_chs=out_chs, stride=stride, dilation=dilation[0],
                apply_act=False, **layer_args)
        else:
            self.shortcut = nn.Identity()

        self.conv1_kxk = ConvBnAct(in_chs, mid_chs, kernel_size, stride=stride, dilation=dilation[0], **layer_args)
        self.conv2_kxk = ConvBnAct(
            mid_chs, out_chs, kernel_size, dilation=dilation[1], groups=groups,
            drop_block=drop_block, apply_act=False, **layer_args)
        self.attn = nn.Identity() if attn_layer is None else attn_layer(out_chs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.act = nn.Identity() if linear_out else act_layer(inplace=True)

    def init_weights(self, zero_init_last_bn=False):
        if zero_init_last_bn:
            nn.init.zeros_(self.conv2_kxk.bn.weight)

    def forward(self, x):
        shortcut = self.shortcut(x)

        # residual path
        x = self.conv1_kxk(x)
        x = self.conv2_kxk(x)
        x = self.attn(x)
        x = self.drop_path(x)

        x = self.act(x + shortcut)
        return x


class BottleneckBlock(nn.Module):
    """ ResNet-like Bottleneck Block - 1x1 - kxk - 1x1
    """

    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, dilation=(1, 1), bottle_ratio=1., group_size=None,
                 downsample='avg', linear_out=False, layer_cfg=None, drop_block=None, drop_path_rate=0.):
        super(BottleneckBlock, self).__init__()
        layer_cfg = layer_cfg or {}
        act_layer, attn_layer = _ex_tuple(layer_cfg, 'act_layer', 'attn_layer')
        layer_args = _na_args(layer_cfg)
        mid_chs = make_divisible(out_chs * bottle_ratio)
        groups = num_groups(group_size, mid_chs)

        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = create_downsample(
                downsample, in_chs=in_chs, out_chs=out_chs, stride=stride, dilation=dilation[0],
                apply_act=False, **layer_args)
        else:
            self.shortcut = nn.Identity()

        self.conv1_1x1 = ConvBnAct(in_chs, mid_chs, 1, **layer_args)
        self.conv2_kxk = ConvBnAct(
            mid_chs, mid_chs, kernel_size, stride=stride, dilation=dilation[0],
            groups=groups, drop_block=drop_block, **layer_args)
        self.attn = nn.Identity() if attn_layer is None else attn_layer(mid_chs)
        self.conv3_1x1 = ConvBnAct(mid_chs, out_chs, 1, apply_act=False, **layer_args)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.act = nn.Identity() if linear_out else act_layer(inplace=True)

    def init_weights(self, zero_init_last_bn=False):
        if zero_init_last_bn:
            nn.init.zeros_(self.conv3_1x1.bn.weight)

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1_1x1(x)
        x = self.conv2_kxk(x)
        x = self.attn(x)
        x = self.conv3_1x1(x)
        x = self.drop_path(x)

        x = self.act(x + shortcut)
        return x


class DarkBlock(nn.Module):
    """ DarkNet-like (1x1 + 3x3 w/ stride) block

    The GE-Net impl included a 1x1 + 3x3 block in their search space. It was not used in the feature models.
    This block is pretty much a DarkNet block (also DenseNet) hence the name. Neither DarkNet or DenseNet
    uses strides within the block (external 3x3 or maxpool downsampling is done in front of the block repeats).

    If one does want to use a lot of these blocks w/ stride, I'd recommend using the EdgeBlock (3x3 /w stride + 1x1)
    for more optimal compute.
    """

    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, dilation=(1, 1), bottle_ratio=1.0, group_size=None,
                 downsample='avg', linear_out=False, layer_cfg=None, drop_block=None, drop_path_rate=0.):
        super(DarkBlock, self).__init__()
        layer_cfg = layer_cfg or {}
        act_layer, attn_layer = _ex_tuple(layer_cfg, 'act_layer', 'attn_layer')
        layer_args = _na_args(layer_cfg)
        mid_chs = make_divisible(out_chs * bottle_ratio)
        groups = num_groups(group_size, mid_chs)

        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = create_downsample(
                downsample, in_chs=in_chs, out_chs=out_chs, stride=stride, dilation=dilation[0],
                apply_act=False, **layer_args)
        else:
            self.shortcut = nn.Identity()

        self.conv1_1x1 = ConvBnAct(in_chs, mid_chs, 1, **layer_args)
        self.conv2_kxk = ConvBnAct(
            mid_chs, out_chs, kernel_size, stride=stride, dilation=dilation[0],
            groups=groups,  drop_block=drop_block, apply_act=False, **layer_args)
        self.attn = nn.Identity() if attn_layer is None else attn_layer(out_chs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.act = nn.Identity() if linear_out else act_layer(inplace=True)

    def init_weights(self, zero_init_last_bn=False):
        if zero_init_last_bn:
            nn.init.zeros_(self.conv2_kxk.bn.weight)

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1_1x1(x)
        x = self.conv2_kxk(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = self.act(x + shortcut)
        return x


class EdgeBlock(nn.Module):
    """ EdgeResidual-like (3x3 + 1x1) block

    A two layer block like DarkBlock, but with the order of the 3x3 and 1x1 convs reversed.
    Very similar to the EfficientNet Edge-Residual block but this block it ends with activations, is
    intended to be used with either expansion or bottleneck contraction, and can use DW/group/non-grouped convs.

    FIXME is there a more common 3x3 + 1x1 conv block to name this after?
    """

    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, dilation=(1, 1), bottle_ratio=1.0, group_size=None,
                 downsample='avg', linear_out=False, layer_cfg=None, drop_block=None, drop_path_rate=0.):
        super(EdgeBlock, self).__init__()
        layer_cfg = layer_cfg or {}
        act_layer, attn_layer = _ex_tuple(layer_cfg, 'act_layer', 'attn_layer')
        layer_args = _na_args(layer_cfg)
        mid_chs = make_divisible(out_chs * bottle_ratio)
        groups = num_groups(group_size, mid_chs)

        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = create_downsample(
                downsample, in_chs=in_chs, out_chs=out_chs, stride=stride, dilation=dilation[0],
                apply_act=False, **layer_args)
        else:
            self.shortcut = nn.Identity()

        self.conv1_kxk = ConvBnAct(
            in_chs, mid_chs, kernel_size, stride=stride, dilation=dilation[0],
            groups=groups,  drop_block=drop_block, **layer_args)
        self.attn = nn.Identity() if attn_layer is None else attn_layer(out_chs)
        self.conv2_1x1 = ConvBnAct(mid_chs, out_chs, 1, apply_act=False, **layer_args)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.act = nn.Identity() if linear_out else act_layer(inplace=True)

    def init_weights(self, zero_init_last_bn=False):
        if zero_init_last_bn:
            nn.init.zeros_(self.conv2_1x1.bn.weight)

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1_kxk(x)
        x = self.attn(x)
        x = self.conv2_1x1(x)
        x = self.drop_path(x)
        x = self.act(x + shortcut)
        return x


class RepVggBlock(nn.Module):
    """ RepVGG Block.

    Adapted from impl at https://github.com/DingXiaoH/RepVGG

    This version does not currently support the deploy optimization. It is currently fixed in 'train' mode.
    """

    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, dilation=(1, 1), bottle_ratio=1.0, group_size=None,
                 downsample='', layer_cfg=None, drop_block=None, drop_path_rate=0.):
        super(RepVggBlock, self).__init__()
        layer_cfg = layer_cfg or {}
        act_layer, norm_layer, attn_layer = _ex_tuple(layer_cfg, 'act_layer', 'norm_layer', 'attn_layer')
        norm_layer = convert_norm_act(norm_layer=norm_layer, act_layer=act_layer)
        layer_args = _na_args(layer_cfg)
        groups = num_groups(group_size, in_chs)

        use_ident = in_chs == out_chs and stride == 1 and dilation[0] == dilation[1]
        self.identity = norm_layer(out_chs, apply_act=False) if use_ident else None
        self.conv_kxk = ConvBnAct(
            in_chs, out_chs, kernel_size, stride=stride, dilation=dilation[0],
            groups=groups, drop_block=drop_block, apply_act=False, **layer_args)
        self.conv_1x1 = ConvBnAct(in_chs, out_chs, 1, stride=stride, groups=groups, apply_act=False, **layer_args)
        self.attn = nn.Identity() if attn_layer is None else attn_layer(out_chs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. and use_ident else nn.Identity()
        self.act = act_layer(inplace=True)

    def init_weights(self, zero_init_last_bn=False):
        # NOTE this init overrides that base model init with specific changes for the block type
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, .1, .1)
                nn.init.normal_(m.bias, 0, .1)

    def forward(self, x):
        if self.identity is None:
            x = self.conv_1x1(x) + self.conv_kxk(x)
        else:
            identity = self.identity(x)
            x = self.conv_1x1(x) + self.conv_kxk(x)
            x = self.drop_path(x)  # not in the paper / official impl, experimental
            x = x + identity
        x = self.attn(x)  # no attn in the paper / official impl, experimental
        x = self.act(x)
        return x


_block_registry = dict(
    basic=BasicBlock,
    bottle=BottleneckBlock,
    dark=DarkBlock,
    edge=EdgeBlock,
    rep=RepVggBlock,
)


def register_block(block_type:str, block_fn: nn.Module):
    _block_registry[block_type] = block_fn


def create_block(block: Union[str, nn.Module], **kwargs):
    if isinstance(block, (nn.Module, partial)):
        return block(**kwargs)
    assert block in _block_registry, f'Unknown block type ({block}'
    return _block_registry[block](**kwargs)


def create_stem(in_chs, out_chs, stem_type='', layer_cfg=None):
    layer_cfg = layer_cfg or {}
    layer_args = _na_args(layer_cfg)
    assert stem_type in ('', 'deep', 'deep_tiered', '3x3', '7x7', 'rep')
    if 'deep' in stem_type:
        # 3 deep 3x3 conv stack
        stem = OrderedDict()
        stem_chs = (out_chs // 2, out_chs // 2)
        if 'tiered' in stem_type:
            stem_chs = (3 * stem_chs[0] // 4, stem_chs[1])
        norm_layer, act_layer = _ex_tuple(layer_args, 'norm_layer', 'act_layer')
        stem['conv1'] = create_conv2d(in_chs, stem_chs[0], kernel_size=3, stride=2)
        stem['conv2'] = create_conv2d(stem_chs[0], stem_chs[1], kernel_size=3, stride=1)
        stem['conv3'] = create_conv2d(stem_chs[1], out_chs, kernel_size=3, stride=1)
        norm_act_layer = convert_norm_act(norm_layer=norm_layer, act_layer=act_layer)
        stem['na'] = norm_act_layer(out_chs)
        stem = nn.Sequential(stem)
    elif '7x7' in stem_type:
        # 7x7 stem conv as in ResNet
        stem = ConvBnAct(in_chs, out_chs, 7, stride=2, **layer_args)
    elif 'rep' in stem_type:
        stem = RepVggBlock(in_chs, out_chs, stride=2, layer_cfg=layer_cfg)
    else:
        # 3x3 stem conv as in RegNet
        stem = ConvBnAct(in_chs, out_chs, 3, stride=2, **layer_args)

    return stem


class ByobNet(nn.Module):
    """ 'Bring-your-own-blocks' Net

    A flexible network backbone that allows building model stem + blocks via
    dataclass cfg definition w/ factory functions for module instantiation.

    Current assumption is that both stem and blocks are in conv-bn-act order (w/ block ending in act).
    """
    def __init__(self, cfg: ByobCfg, num_classes=1000, in_chans=3, global_pool='avg', output_stride=32,
                 zero_init_last_bn=True, drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        norm_layer = cfg.norm_layer
        act_layer = get_act_layer(cfg.act_layer)
        attn_layer = partial(get_attn(cfg.attn_layer), **cfg.attn_kwargs) if cfg.attn_layer else None
        layer_cfg = dict(norm_layer=norm_layer, act_layer=act_layer, attn_layer=attn_layer)

        stem_chs = int(round((cfg.stem_chs or cfg.blocks[0].c) * cfg.width_factor))
        self.stem = create_stem(in_chans, stem_chs, cfg.stem_type, layer_cfg=layer_cfg)

        self.feature_info = []
        depths = [bc.d for bc in cfg.blocks]
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_name = 'stem'
        prev_chs = stem_chs
        net_stride = 2
        dilation = 1
        stages = []
        for stage_idx, block_cfg in enumerate(cfg.blocks):
            stride = block_cfg.s
            if stride != 1:
                self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=prev_name))
            if net_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            net_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2

            blocks = []
            for block_idx in range(block_cfg.d):
                out_chs = make_divisible(block_cfg.c * cfg.width_factor)
                group_size = block_cfg.gs
                if isinstance(group_size, Callable):
                    group_size = group_size(out_chs, block_idx)
                block_kwargs = dict(  # Blocks used in this model must accept these arguments
                    in_chs=prev_chs,
                    out_chs=out_chs,
                    stride=stride if block_idx == 0 else 1,
                    dilation=(first_dilation, dilation),
                    group_size=group_size,
                    bottle_ratio=block_cfg.br,
                    downsample=cfg.downsample,
                    drop_path_rate=dpr[stage_idx][block_idx],
                    layer_cfg=layer_cfg,
                )
                blocks += [create_block(block_cfg.type, **block_kwargs)]
                first_dilation = dilation
                prev_chs = out_chs
            stages += [nn.Sequential(*blocks)]
            prev_name = f'stages.{stage_idx}'
        self.stages = nn.Sequential(*stages)

        if cfg.num_features:
            self.num_features = int(round(cfg.width_factor * cfg.num_features))
            self.final_conv = ConvBnAct(prev_chs, self.num_features, 1, **_na_args(layer_cfg))
        else:
            self.num_features = prev_chs
            self.final_conv = nn.Identity()
        self.feature_info += [dict(num_chs=self.num_features, reduction=net_stride, module='final_conv')]

        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.modules():
            # call each block's weight init for block-specific overrides to init above
            if hasattr(m, 'init_weights'):
                m.init_weights(zero_init_last_bn=zero_init_last_bn)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.final_conv(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _create_byobnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ByobNet, variant, pretrained,
        default_cfg=default_cfgs[variant],
        model_cfg=model_cfgs[variant],
        feature_cfg=dict(flatten_sequential=True),
        **kwargs)


@register_model
def gernet_l(pretrained=False, **kwargs):
    """ GEResNet-Large (GENet-Large from official impl)
    `Neural Architecture Design for GPU-Efficient Networks` - https://arxiv.org/abs/2006.14090
    """
    return _create_byobnet('gernet_l', pretrained=pretrained, **kwargs)


@register_model
def gernet_m(pretrained=False, **kwargs):
    """ GEResNet-Medium (GENet-Normal from official impl)
    `Neural Architecture Design for GPU-Efficient Networks` - https://arxiv.org/abs/2006.14090
    """
    return _create_byobnet('gernet_m', pretrained=pretrained, **kwargs)


@register_model
def gernet_s(pretrained=False, **kwargs):
    """ EResNet-Small (GENet-Small from official impl)
    `Neural Architecture Design for GPU-Efficient Networks` - https://arxiv.org/abs/2006.14090
    """
    return _create_byobnet('gernet_s', pretrained=pretrained, **kwargs)


@register_model
def repvgg_a2(pretrained=False, **kwargs):
    """ RepVGG-A2
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_a2', pretrained=pretrained, **kwargs)


@register_model
def repvgg_b0(pretrained=False, **kwargs):
    """ RepVGG-B0
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_b0', pretrained=pretrained, **kwargs)


@register_model
def repvgg_b1(pretrained=False, **kwargs):
    """ RepVGG-B1
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_b1', pretrained=pretrained, **kwargs)


@register_model
def repvgg_b1g4(pretrained=False, **kwargs):
    """ RepVGG-B1g4
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_b1g4', pretrained=pretrained, **kwargs)


@register_model
def repvgg_b2(pretrained=False, **kwargs):
    """ RepVGG-B2
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_b2', pretrained=pretrained, **kwargs)


@register_model
def repvgg_b2g4(pretrained=False, **kwargs):
    """ RepVGG-B2g4
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_b2g4', pretrained=pretrained, **kwargs)


@register_model
def repvgg_b3(pretrained=False, **kwargs):
    """ RepVGG-B3
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_b3', pretrained=pretrained, **kwargs)


@register_model
def repvgg_b3g4(pretrained=False, **kwargs):
    """ RepVGG-B3g4
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_b3g4', pretrained=pretrained, **kwargs)
