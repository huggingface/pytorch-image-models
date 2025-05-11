""" Bring-Your-Own-Blocks Network

A flexible network w/ dataclass based config for stacking those NN blocks.

This model is currently used to implement the following networks:

GPU Efficient (ResNets) - gernet_l/m/s (original versions called genet, but this was already used (by SENet author)).
Paper: `Neural Architecture Design for GPU-Efficient Networks` - https://arxiv.org/abs/2006.14090
Code and weights: https://github.com/idstcv/GPU-Efficient-Networks, licensed Apache 2.0

RepVGG - repvgg_*
Paper: `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
Code and weights: https://github.com/DingXiaoH/RepVGG, licensed MIT

MobileOne - mobileone_*
Paper: `MobileOne: An Improved One millisecond Mobile Backbone` - https://arxiv.org/abs/2206.04040
Code and weights: https://github.com/apple/ml-mobileone, licensed MIT

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
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Tuple, List, Dict, Optional, Union, Any, Callable, Sequence

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import (
    ClassifierHead, NormMlpClassifierHead, ConvNormAct, BatchNormAct2d, EvoNorm2dS0a,
    AttentionPool2d, RotAttentionPool2d, DropPath, AvgPool2dSame,
    create_conv2d, get_act_layer, get_norm_act_layer, get_attn, make_divisible, to_2tuple,
)
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import named_apply, checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['ByobNet', 'ByoModelCfg', 'ByoBlockCfg', 'create_byob_stem', 'create_block']


@dataclass
class ByoBlockCfg:
    type: Union[str, nn.Module]
    d: int  # block depth (number of block repeats in stage)
    c: int  # number of output channels for each block in stage
    s: int = 2  # stride of stage (first block)
    gs: Optional[Union[int, Callable]] = None  # group-size of blocks in stage, conv is depthwise if gs == 1
    br: float = 1.  # bottleneck-ratio of blocks in stage

    # NOTE: these config items override the model cfgs that are applied to all blocks by default
    attn_layer: Optional[str] = None
    attn_kwargs: Optional[Dict[str, Any]] = None
    self_attn_layer: Optional[str] = None
    self_attn_kwargs: Optional[Dict[str, Any]] = None
    block_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class ByoModelCfg:
    blocks: Tuple[Union[ByoBlockCfg, Tuple[ByoBlockCfg, ...]], ...]
    downsample: str = 'conv1x1'
    stem_type: str = '3x3'
    stem_pool: Optional[str] = 'maxpool'
    stem_chs: Union[int, List[int], Tuple[int, ...]] = 32
    width_factor: float = 1.0
    num_features: int = 0  # num out_channels for final conv, no final 1x1 conv if 0
    zero_init_last: bool = True  # zero init last weight (usually bn) in residual path
    fixed_input_size: bool = False  # model constrained to a fixed-input size / img_size must be provided on creation

    # layer config
    act_layer: str = 'relu'
    norm_layer: str = 'batchnorm'
    aa_layer: str = ''

    # Head config
    head_hidden_size: Optional[int] = None  # feat dim of MLP head or AttentionPool output
    head_type: str = 'classifier'

    # Block config
    # NOTE: these config items will be overridden by the block cfg (per-block) if they are set there
    attn_layer: Optional[str] = None
    attn_kwargs: dict = field(default_factory=lambda: dict())
    self_attn_layer: Optional[str] = None
    self_attn_kwargs: dict = field(default_factory=lambda: dict())
    block_kwargs: Dict[str, Any] = field(default_factory=lambda: dict())


def _rep_vgg_bcfg(d=(4, 6, 16, 1), wf=(1., 1., 1., 1.), groups=0):
    c = (64, 128, 256, 512)
    group_size = 0
    if groups > 0:
        group_size = lambda chs, idx: chs // groups if (idx + 1) % 2 == 0 else 0
    bcfg = tuple([ByoBlockCfg(type='rep', d=d, c=c * wf, gs=group_size) for d, c, wf in zip(d, c, wf)])
    return bcfg


def _mobileone_bcfg(d=(2, 8, 10, 1), wf=(1., 1., 1., 1.), se_blocks=(), num_conv_branches=1):
    c = (64, 128, 256, 512)
    prev_c = min(64, c[0] * wf[0])
    se_blocks = se_blocks or (0,) * len(d)
    bcfg = []
    for d, c, w, se in zip(d, c, wf, se_blocks):
        scfg = []
        for i in range(d):
            out_c = c * w
            bk = dict(num_conv_branches=num_conv_branches)
            ak = {}
            if i >= d - se:
                ak['attn_layer'] = 'se'
            scfg += [ByoBlockCfg(type='one', d=1, c=prev_c, gs=1, block_kwargs=bk, **ak)]  # depthwise block
            scfg += [ByoBlockCfg(
                type='one', d=1, c=out_c, gs=0, block_kwargs=dict(kernel_size=1, **bk), **ak)]  # pointwise block
            prev_c = out_c
        bcfg += [scfg]
    return bcfg


def interleave_blocks(
        types: Tuple[str, str], d,
        every: Union[int, List[int]] = 1,
        first: bool = False,
        **kwargs,
) -> Tuple[ByoBlockCfg]:
    """ interleave 2 block types in stack
    """
    assert len(types) == 2
    if isinstance(every, int):
        every = list(range(0 if first else every, d, every + 1))
        if not every:
            every = [d - 1]
    set(every)
    blocks = []
    for i in range(d):
        block_type = types[1] if i in every else types[0]
        blocks += [ByoBlockCfg(type=block_type, d=1, **kwargs)]
    return tuple(blocks)


def expand_blocks_cfg(stage_blocks_cfg: Union[ByoBlockCfg, Sequence[ByoBlockCfg]]) -> List[ByoBlockCfg]:
    if not isinstance(stage_blocks_cfg, Sequence):
        stage_blocks_cfg = (stage_blocks_cfg,)
    block_cfgs = []
    for i, cfg in enumerate(stage_blocks_cfg):
        block_cfgs += [replace(cfg, d=1) for _ in range(cfg.d)]
    return block_cfgs


def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


@dataclass
class LayerFn:
    conv_norm_act: Callable = ConvNormAct
    norm_act: Callable = BatchNormAct2d
    act: Callable = nn.ReLU
    attn: Optional[Callable] = None
    self_attn: Optional[Callable] = None


class DownsampleAvg(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dilation: int = 1,
            apply_act: bool = False,
            layers: LayerFn = None,
    ):
        """ AvgPool Downsampling as in 'D' ResNet variants."""
        super(DownsampleAvg, self).__init__()
        layers = layers or LayerFn()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()
        self.conv = layers.conv_norm_act(in_chs, out_chs, 1, apply_act=apply_act)

    def forward(self, x):
        return self.conv(self.pool(x))


def create_shortcut(
        downsample_type: str,
        in_chs: int,
        out_chs: int,
        stride: int,
        dilation: Tuple[int, int],
        layers: LayerFn,
        **kwargs,
):
    assert downsample_type in ('avg', 'conv1x1', '')
    if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
        if not downsample_type:
            return None  # no shortcut
        elif downsample_type == 'avg':
            return DownsampleAvg(in_chs, out_chs, stride=stride, dilation=dilation[0], **kwargs)
        else:
            return layers.conv_norm_act(in_chs, out_chs, kernel_size=1, stride=stride, dilation=dilation[0], **kwargs)
    else:
        return nn.Identity()  # identity shortcut


class BasicBlock(nn.Module):
    """ ResNet Basic Block - kxk + kxk
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            group_size: Optional[int] = None,
            bottle_ratio: float = 1.0,
            downsample: str = 'avg',
            attn_last: bool = True,
            linear_out: bool = False,
            layers: LayerFn = None,
            drop_block: Callable = None,
            drop_path_rate: float = 0.,
    ):
        super(BasicBlock, self).__init__()
        layers = layers or LayerFn()
        mid_chs = make_divisible(out_chs * bottle_ratio)
        groups = num_groups(group_size, mid_chs)

        self.shortcut = create_shortcut(
            downsample, in_chs, out_chs,
            stride=stride, dilation=dilation, apply_act=False, layers=layers,
        )

        self.conv1_kxk = layers.conv_norm_act(in_chs, mid_chs, kernel_size, stride=stride, dilation=dilation[0])
        self.attn = nn.Identity() if attn_last or layers.attn is None else layers.attn(mid_chs)
        self.conv2_kxk = layers.conv_norm_act(
            mid_chs, out_chs, kernel_size,
            dilation=dilation[1], groups=groups, drop_layer=drop_block, apply_act=False,
        )
        self.attn_last = nn.Identity() if not attn_last or layers.attn is None else layers.attn(out_chs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.act = nn.Identity() if linear_out else layers.act(inplace=True)

    def init_weights(self, zero_init_last: bool = False):
        if zero_init_last and self.shortcut is not None and getattr(self.conv2_kxk.bn, 'weight', None) is not None:
            nn.init.zeros_(self.conv2_kxk.bn.weight)
        for attn in (self.attn, self.attn_last):
            if hasattr(attn, 'reset_parameters'):
                attn.reset_parameters()

    def forward(self, x):
        shortcut = x
        x = self.conv1_kxk(x)
        x = self.attn(x)
        x = self.conv2_kxk(x)
        x = self.attn_last(x)
        x = self.drop_path(x)
        if self.shortcut is not None:
            x = x + self.shortcut(shortcut)
        return self.act(x)


class BottleneckBlock(nn.Module):
    """ ResNet-like Bottleneck Block - 1x1 - kxk - 1x1
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            bottle_ratio: float = 1.,
            group_size: Optional[int] = None,
            downsample: str = 'avg',
            attn_last: bool = False,
            linear_out: bool = False,
            extra_conv: bool = False,
            bottle_in: bool = False,
            layers: LayerFn = None,
            drop_block: Callable = None,
            drop_path_rate: float = 0.,
    ):
        super(BottleneckBlock, self).__init__()
        layers = layers or LayerFn()
        mid_chs = make_divisible((in_chs if bottle_in else out_chs) * bottle_ratio)
        groups = num_groups(group_size, mid_chs)

        self.shortcut = create_shortcut(
            downsample, in_chs, out_chs,
            stride=stride, dilation=dilation, apply_act=False, layers=layers,
        )

        self.conv1_1x1 = layers.conv_norm_act(in_chs, mid_chs, 1)
        self.conv2_kxk = layers.conv_norm_act(
            mid_chs, mid_chs, kernel_size,
            stride=stride, dilation=dilation[0], groups=groups, drop_layer=drop_block,
        )
        if extra_conv:
            self.conv2b_kxk = layers.conv_norm_act(
                mid_chs, mid_chs, kernel_size, dilation=dilation[1], groups=groups)
        else:
            self.conv2b_kxk = nn.Identity()
        self.attn = nn.Identity() if attn_last or layers.attn is None else layers.attn(mid_chs)
        self.conv3_1x1 = layers.conv_norm_act(mid_chs, out_chs, 1, apply_act=False)
        self.attn_last = nn.Identity() if not attn_last or layers.attn is None else layers.attn(out_chs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.act = nn.Identity() if linear_out else layers.act(inplace=True)

    def init_weights(self, zero_init_last: bool = False):
        if zero_init_last and self.shortcut is not None and getattr(self.conv3_1x1.bn, 'weight', None) is not None:
            nn.init.zeros_(self.conv3_1x1.bn.weight)
        for attn in (self.attn, self.attn_last):
            if hasattr(attn, 'reset_parameters'):
                attn.reset_parameters()

    def forward(self, x):
        shortcut = x
        x = self.conv1_1x1(x)
        x = self.conv2_kxk(x)
        x = self.conv2b_kxk(x)
        x = self.attn(x)
        x = self.conv3_1x1(x)
        x = self.attn_last(x)
        x = self.drop_path(x)
        if self.shortcut is not None:
            x = x + self.shortcut(shortcut)
        return self.act(x)


class DarkBlock(nn.Module):
    """ DarkNet-like (1x1 + 3x3 w/ stride) block

    The GE-Net impl included a 1x1 + 3x3 block in their search space. It was not used in the feature models.
    This block is pretty much a DarkNet block (also DenseNet) hence the name. Neither DarkNet or DenseNet
    uses strides within the block (external 3x3 or maxpool downsampling is done in front of the block repeats).

    If one does want to use a lot of these blocks w/ stride, I'd recommend using the EdgeBlock (3x3 /w stride + 1x1)
    for more optimal compute.
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            bottle_ratio: float = 1.0,
            group_size: Optional[int] = None,
            downsample: str = 'avg',
            attn_last: bool = True,
            linear_out: bool = False,
            layers: LayerFn = None,
            drop_block: Callable = None,
            drop_path_rate: float = 0.,
    ):
        super(DarkBlock, self).__init__()
        layers = layers or LayerFn()
        mid_chs = make_divisible(out_chs * bottle_ratio)
        groups = num_groups(group_size, mid_chs)

        self.shortcut = create_shortcut(
            downsample, in_chs, out_chs,
            stride=stride, dilation=dilation, apply_act=False, layers=layers,
        )

        self.conv1_1x1 = layers.conv_norm_act(in_chs, mid_chs, 1)
        self.attn = nn.Identity() if attn_last or layers.attn is None else layers.attn(mid_chs)
        self.conv2_kxk = layers.conv_norm_act(
            mid_chs, out_chs, kernel_size,
            stride=stride, dilation=dilation[0], groups=groups, drop_layer=drop_block, apply_act=False,
        )
        self.attn_last = nn.Identity() if not attn_last or layers.attn is None else layers.attn(out_chs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.act = nn.Identity() if linear_out else layers.act(inplace=True)

    def init_weights(self, zero_init_last: bool = False):
        if zero_init_last and self.shortcut is not None and getattr(self.conv2_kxk.bn, 'weight', None) is not None:
            nn.init.zeros_(self.conv2_kxk.bn.weight)
        for attn in (self.attn, self.attn_last):
            if hasattr(attn, 'reset_parameters'):
                attn.reset_parameters()

    def forward(self, x):
        shortcut = x
        x = self.conv1_1x1(x)
        x = self.attn(x)
        x = self.conv2_kxk(x)
        x = self.attn_last(x)
        x = self.drop_path(x)
        if self.shortcut is not None:
            x = x + self.shortcut(shortcut)
        return self.act(x)


class EdgeBlock(nn.Module):
    """ EdgeResidual-like (3x3 + 1x1) block

    A two layer block like DarkBlock, but with the order of the 3x3 and 1x1 convs reversed.
    Very similar to the EfficientNet Edge-Residual block but this block it ends with activations, is
    intended to be used with either expansion or bottleneck contraction, and can use DW/group/non-grouped convs.

    FIXME is there a more common 3x3 + 1x1 conv block to name this after?
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            bottle_ratio: float = 1.0,
            group_size: Optional[int] = None,
            downsample: str = 'avg',
            attn_last: bool = False,
            linear_out: bool = False,
            layers: LayerFn = None,
            drop_block: Callable = None,
            drop_path_rate: float = 0.,
    ):
        super(EdgeBlock, self).__init__()
        layers = layers or LayerFn()
        mid_chs = make_divisible(out_chs * bottle_ratio)
        groups = num_groups(group_size, mid_chs)

        self.shortcut = create_shortcut(
            downsample, in_chs, out_chs,
            stride=stride, dilation=dilation, apply_act=False, layers=layers,
        )
        self.conv1_kxk = layers.conv_norm_act(
            in_chs, mid_chs, kernel_size,
            stride=stride, dilation=dilation[0], groups=groups, drop_layer=drop_block,
        )
        self.attn = nn.Identity() if attn_last or layers.attn is None else layers.attn(mid_chs)
        self.conv2_1x1 = layers.conv_norm_act(mid_chs, out_chs, 1, apply_act=False)
        self.attn_last = nn.Identity() if not attn_last or layers.attn is None else layers.attn(out_chs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.act = nn.Identity() if linear_out else layers.act(inplace=True)

    def init_weights(self, zero_init_last: bool = False):
        if zero_init_last and self.shortcut is not None and getattr(self.conv2_1x1.bn, 'weight', None) is not None:
            nn.init.zeros_(self.conv2_1x1.bn.weight)
        for attn in (self.attn, self.attn_last):
            if hasattr(attn, 'reset_parameters'):
                attn.reset_parameters()

    def forward(self, x):
        shortcut = x
        x = self.conv1_kxk(x)
        x = self.attn(x)
        x = self.conv2_1x1(x)
        x = self.attn_last(x)
        x = self.drop_path(x)
        if self.shortcut is not None:
            x = x + self.shortcut(shortcut)
        return self.act(x)


class RepVggBlock(nn.Module):
    """ RepVGG Block.

    Adapted from impl at https://github.com/DingXiaoH/RepVGG
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            bottle_ratio: float = 1.0,
            group_size: Optional[int] = None,
            downsample: str = '',
            layers: LayerFn = None,
            drop_block: Callable = None,
            drop_path_rate: float = 0.,
            inference_mode: bool = False
    ):
        super(RepVggBlock, self).__init__()
        self.groups = groups = num_groups(group_size, in_chs)
        layers = layers or LayerFn()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_chs,
                out_channels=out_chs,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            self.reparam_conv = None
            use_ident = in_chs == out_chs and stride == 1 and dilation[0] == dilation[1]
            self.identity = layers.norm_act(out_chs, apply_act=False) if use_ident else None
            self.conv_kxk = layers.conv_norm_act(
                in_chs, out_chs, kernel_size,
                stride=stride, dilation=dilation[0], groups=groups, drop_layer=drop_block, apply_act=False,
            )
            self.conv_1x1 = layers.conv_norm_act(in_chs, out_chs, 1, stride=stride, groups=groups, apply_act=False)
            self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. and use_ident else nn.Identity()

        self.attn = nn.Identity() if layers.attn is None else layers.attn(out_chs)
        self.act = layers.act(inplace=True)

    def init_weights(self, zero_init_last: bool = False):
        # NOTE this init overrides that base model init with specific changes for the block type
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, .1, .1)
                nn.init.normal_(m.bias, 0, .1)
        if hasattr(self.attn, 'reset_parameters'):
            self.attn.reset_parameters()

    def forward(self, x):
        if self.reparam_conv is not None:
            return self.act(self.attn(self.reparam_conv(x)))

        if self.identity is None:
            x = self.conv_1x1(x) + self.conv_kxk(x)
        else:
            identity = self.identity(x)
            x = self.conv_1x1(x) + self.conv_kxk(x)
            x = self.drop_path(x)  # not in the paper / official impl, experimental
            x += identity
        x = self.attn(x)  # no attn in the paper / official impl, experimental
        return self.act(x)

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.reparam_conv is not None:
            return

        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.conv_kxk.conv.in_channels,
            out_channels=self.conv_kxk.conv.out_channels,
            kernel_size=self.conv_kxk.conv.kernel_size,
            stride=self.conv_kxk.conv.stride,
            padding=self.conv_kxk.conv.padding,
            dilation=self.conv_kxk.conv.dilation,
            groups=self.conv_kxk.conv.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for name, para in self.named_parameters():
            if 'reparam_conv' in name:
                continue
            para.detach_()
        self.__delattr__('conv_kxk')
        self.__delattr__('conv_1x1')
        self.__delattr__('identity')
        self.__delattr__('drop_path')

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83
        """
        # get weights and bias of scale branch
        kernel_1x1 = 0
        bias_1x1 = 0
        if self.conv_1x1 is not None:
            kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.conv_1x1)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.conv_kxk.conv.kernel_size[0] // 2
            kernel_1x1 = torch.nn.functional.pad(kernel_1x1, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.identity is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.identity)

        # get weights and bias of conv branches
        kernel_conv, bias_conv = self._fuse_bn_tensor(self.conv_kxk)

        kernel_final = kernel_conv + kernel_1x1 + kernel_identity
        bias_final = bias_conv + bias_1x1 + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95
        """
        if isinstance(branch, ConvNormAct):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                in_chs = self.conv_kxk.conv.in_channels
                input_dim = in_chs // self.groups
                kernel_size = self.conv_kxk.conv.kernel_size
                kernel_value = torch.zeros_like(self.conv_kxk.conv.weight)
                for i in range(in_chs):
                    kernel_value[i, i % input_dim, kernel_size[0] // 2, kernel_size[1] // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class MobileOneBlock(nn.Module):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            bottle_ratio: float = 1.0,  # unused
            group_size: Optional[int] = None,
            downsample: str = '',  # unused
            inference_mode: bool = False,
            num_conv_branches: int = 1,
            layers: LayerFn = None,
            drop_block: Callable = None,
            drop_path_rate: float = 0.,
    ) -> None:
        """ Construct a MobileOneBlock module.
        """
        super(MobileOneBlock, self).__init__()
        self.num_conv_branches = num_conv_branches
        self.groups = groups = num_groups(group_size, in_chs)
        layers = layers or LayerFn()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_chs,
                out_channels=out_chs,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=True)
        else:
            self.reparam_conv = None

            # Re-parameterizable skip connection
            use_ident = in_chs == out_chs and stride == 1 and dilation[0] == dilation[1]
            self.identity = layers.norm_act(out_chs, apply_act=False) if use_ident else None

            # Re-parameterizable conv branches
            convs = []
            for _ in range(self.num_conv_branches):
                convs.append(layers.conv_norm_act(
                    in_chs, out_chs, kernel_size=kernel_size,
                    stride=stride, groups=groups, apply_act=False))
            self.conv_kxk = nn.ModuleList(convs)

            # Re-parameterizable scale branch
            self.conv_scale = None
            if kernel_size > 1:
                self.conv_scale = layers.conv_norm_act(
                    in_chs, out_chs, kernel_size=1,
                    stride=stride, groups=groups, apply_act=False)
            self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. and use_ident else nn.Identity()

        self.attn = nn.Identity() if layers.attn is None else layers.attn(out_chs)
        self.act = layers.act(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.reparam_conv is not None:
            return self.act(self.attn(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.identity is not None:
            identity_out = self.identity(x)

        # Scale branch output
        scale_out = 0
        if self.conv_scale is not None:
            scale_out = self.conv_scale(x)

        # Other branches
        out = scale_out
        for ck in self.conv_kxk:
            out += ck(x)
        out = self.drop_path(out)
        out += identity_out

        return self.act(self.attn(out))

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.reparam_conv is not None:
            return

        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.conv_kxk[0].conv.in_channels,
            out_channels=self.conv_kxk[0].conv.out_channels,
            kernel_size=self.conv_kxk[0].conv.kernel_size,
            stride=self.conv_kxk[0].conv.stride,
            padding=self.conv_kxk[0].conv.padding,
            dilation=self.conv_kxk[0].conv.dilation,
            groups=self.conv_kxk[0].conv.groups,
            bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for name, para in self.named_parameters():
            if 'reparam_conv' in name:
                continue
            para.detach_()
        self.__delattr__('conv_kxk')
        self.__delattr__('conv_scale')
        self.__delattr__('identity')
        self.__delattr__('drop_path')

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.conv_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.conv_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.conv_kxk[0].conv.kernel_size[0] // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.identity is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.identity)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.conv_kxk[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95
        """
        if isinstance(branch, ConvNormAct):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                in_chs = self.conv_kxk[0].conv.in_channels
                input_dim = in_chs // self.groups
                kernel_size = self.conv_kxk[0].conv.kernel_size
                kernel_value = torch.zeros_like(self.conv_kxk[0].conv.weight)
                for i in range(in_chs):
                    kernel_value[i, i % input_dim, kernel_size[0] // 2, kernel_size[1] // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class SelfAttnBlock(nn.Module):
    """ ResNet-like Bottleneck Block - 1x1 - optional kxk - self attn - 1x1
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            bottle_ratio: float = 1.,
            group_size: Optional[int] = None,
            downsample: str = 'avg',
            extra_conv: bool = False,
            linear_out: bool = False,
            bottle_in: bool = False,
            post_attn_na: bool = True,
            feat_size: Optional[Tuple[int, int]] = None,
            layers: LayerFn = None,
            drop_block: Callable = None,
            drop_path_rate: float = 0.,
    ):
        super(SelfAttnBlock, self).__init__()
        assert layers is not None
        mid_chs = make_divisible((in_chs if bottle_in else out_chs) * bottle_ratio)
        groups = num_groups(group_size, mid_chs)

        self.shortcut = create_shortcut(
            downsample, in_chs, out_chs,
            stride=stride, dilation=dilation, apply_act=False, layers=layers,
        )

        self.conv1_1x1 = layers.conv_norm_act(in_chs, mid_chs, 1)
        if extra_conv:
            self.conv2_kxk = layers.conv_norm_act(
                mid_chs, mid_chs, kernel_size,
                stride=stride, dilation=dilation[0], groups=groups, drop_layer=drop_block,
            )
            stride = 1  # striding done via conv if enabled
        else:
            self.conv2_kxk = nn.Identity()
        opt_kwargs = {} if feat_size is None else dict(feat_size=feat_size)
        # FIXME need to dilate self attn to have dilated network support, moop moop
        self.self_attn = layers.self_attn(mid_chs, stride=stride, **opt_kwargs)
        self.post_attn = layers.norm_act(mid_chs) if post_attn_na else nn.Identity()
        self.conv3_1x1 = layers.conv_norm_act(mid_chs, out_chs, 1, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.act = nn.Identity() if linear_out else layers.act(inplace=True)

    def init_weights(self, zero_init_last: bool = False):
        if zero_init_last and self.shortcut is not None and getattr(self.conv3_1x1.bn, 'weight', None) is not None:
            nn.init.zeros_(self.conv3_1x1.bn.weight)
        if hasattr(self.self_attn, 'reset_parameters'):
            self.self_attn.reset_parameters()

    def forward(self, x):
        shortcut = x
        x = self.conv1_1x1(x)
        x = self.conv2_kxk(x)
        x = self.self_attn(x)
        x = self.post_attn(x)
        x = self.conv3_1x1(x)
        x = self.drop_path(x)
        if self.shortcut is not None:
            x = x + self.shortcut(shortcut)
        return self.act(x)


_block_registry = dict(
    basic=BasicBlock,
    bottle=BottleneckBlock,
    dark=DarkBlock,
    edge=EdgeBlock,
    rep=RepVggBlock,
    one=MobileOneBlock,
    self_attn=SelfAttnBlock,
)


def register_block(block_type:str, block_fn: nn.Module):
    _block_registry[block_type] = block_fn


def create_block(block: Union[str, nn.Module], **kwargs):
    if isinstance(block, (nn.Module, partial)):
        return block(**kwargs)
    assert block in _block_registry, f'Unknown block type ({block}'
    return _block_registry[block](**kwargs)


class Stem(nn.Sequential):

    def __init__(
            self,
            in_chs: int,
            out_chs: Union[int, List[int], Tuple[int, ...]],
            kernel_size: int = 3,
            stride: int = 4,
            pool: str = 'maxpool',
            num_rep: int = 3,
            num_act: Optional[int] = None,
            chs_decay: float = 0.5,
            layers: LayerFn = None,
    ):
        super().__init__()
        assert stride in (2, 4)
        layers = layers or LayerFn()

        if isinstance(out_chs, (list, tuple)):
            num_rep = len(out_chs)
            stem_chs = out_chs
        else:
            stem_chs = [round(out_chs * chs_decay ** i) for i in range(num_rep)][::-1]

        self.stride = stride
        self.feature_info = []  # track intermediate features
        prev_feat = ''
        stem_strides = [2] + [1] * (num_rep - 1)
        if stride == 4 and not pool:
            # set last conv in stack to be strided if stride == 4 and no pooling layer
            stem_strides[-1] = 2

        num_act = num_rep if num_act is None else num_act
        # if num_act < num_rep, first convs in stack won't have bn + act
        stem_norm_acts = [False] * (num_rep - num_act) + [True] * num_act
        prev_chs = in_chs
        curr_stride = 1
        last_feat_idx = -1
        for i, (ch, s, na) in enumerate(zip(stem_chs, stem_strides, stem_norm_acts)):
            layer_fn = layers.conv_norm_act if na else create_conv2d
            conv_name = f'conv{i + 1}'
            if i > 0 and s > 1:
                last_feat_idx = i - 1
                self.feature_info.append(dict(num_chs=prev_chs, reduction=curr_stride, module=prev_feat, stage=0))
            self.add_module(conv_name, layer_fn(prev_chs, ch, kernel_size=kernel_size, stride=s))
            prev_chs = ch
            curr_stride *= s
            prev_feat = conv_name

        if pool:
            pool = pool.lower()
            assert pool in ('max', 'maxpool', 'avg', 'avgpool', 'max2', 'avg2')
            last_feat_idx = i
            self.feature_info.append(dict(num_chs=prev_chs, reduction=curr_stride, module=prev_feat, stage=0))
            if pool == 'max2':
                self.add_module('pool', nn.MaxPool2d(2))
            elif pool == 'avg2':
                self.add_module('pool', nn.AvgPool2d(2))
            elif 'max' in pool:
                self.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            elif 'avg' in pool:
                self.add_module('pool', nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False))
            curr_stride *= 2
            prev_feat = 'pool'

        self.last_feat_idx = last_feat_idx if last_feat_idx >= 0 else None
        self.feature_info.append(dict(num_chs=prev_chs, reduction=curr_stride, module=prev_feat, stage=0))
        assert curr_stride == stride

    def forward_intermediates(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        intermediate: Optional[torch.Tensor] = None
        for i, m in enumerate(self):
            x = m(x)
            if self.last_feat_idx is not None and i == self.last_feat_idx:
                intermediate = x
        return x, intermediate


def create_byob_stem(
        in_chs: int,
        out_chs: int,
        stem_type: str = '',
        pool_type: str = '',
        feat_prefix: str = 'stem',
        layers: LayerFn = None,
):
    layers = layers or LayerFn()
    assert stem_type in ('', 'quad', 'quad2', 'tiered', 'deep', 'rep', 'one', '7x7', '3x3')
    if 'quad' in stem_type:
        # based on NFNet stem, stack of 4 3x3 convs
        num_act = 2 if 'quad2' in stem_type else None
        stem = Stem(in_chs, out_chs, num_rep=4, num_act=num_act, pool=pool_type, layers=layers)
    elif 'tiered' in stem_type:
        # 3x3 stack of 3 convs as in my ResNet-T
        stem = Stem(in_chs, (3 * out_chs // 8, out_chs // 2, out_chs), pool=pool_type, layers=layers)
    elif 'deep' in stem_type:
        # 3x3 stack of 3 convs as in ResNet-D
        stem = Stem(in_chs, out_chs, num_rep=3, chs_decay=1.0, pool=pool_type, layers=layers)
    elif 'rep' in stem_type:
        stem = RepVggBlock(in_chs, out_chs, stride=2, layers=layers)
    elif 'one' in stem_type:
        stem = MobileOneBlock(in_chs, out_chs, kernel_size=3, stride=2, layers=layers)
    elif '7x7' in stem_type:
        # 7x7 stem conv as in ResNet
        if pool_type:
            stem = Stem(in_chs, out_chs, 7, num_rep=1, pool=pool_type, layers=layers)
        else:
            stem = layers.conv_norm_act(in_chs, out_chs, 7, stride=2)
    else:
        if isinstance(out_chs, (tuple, list)):
            stem = Stem(in_chs, out_chs, 3, pool=pool_type, layers=layers)
        else:
            # 3x3 stem conv as in RegNet is the default
            if pool_type:
                stem = Stem(in_chs, out_chs, 3, num_rep=1, pool=pool_type, layers=layers)
            else:
                stem = layers.conv_norm_act(in_chs, out_chs, 3, stride=2)

    if isinstance(stem, Stem):
        feature_info = [dict(f, module='.'.join([feat_prefix, f['module']])) for f in stem.feature_info]
    else:
        feature_info = [dict(num_chs=out_chs, reduction=2, module=feat_prefix, stage=0)]
    return stem, feature_info


def reduce_feat_size(feat_size, stride=2):
    return None if feat_size is None else tuple([s // stride for s in feat_size])


def override_kwargs(block_kwargs, model_kwargs):
    """ Override model level attn/self-attn/block kwargs w/ block level

    NOTE: kwargs are NOT merged across levels, block_kwargs will fully replace model_kwargs
    for the block if set to anything that isn't None.

    i.e. an empty block_kwargs dict will remove kwargs set at model level for that block
    """
    out_kwargs = block_kwargs if block_kwargs is not None else model_kwargs
    return out_kwargs or {}  # make sure None isn't returned


def update_block_kwargs(block_kwargs: Dict[str, Any], block_cfg: ByoBlockCfg, model_cfg: ByoModelCfg, ):
    layer_fns = block_kwargs['layers']

    # override attn layer / args with block local config
    attn_set = block_cfg.attn_layer is not None
    if attn_set or block_cfg.attn_kwargs is not None:
        # override attn layer config
        if attn_set and not block_cfg.attn_layer:
            # empty string for attn_layer type will disable attn for this block
            attn_layer = None
        else:
            attn_kwargs = override_kwargs(block_cfg.attn_kwargs, model_cfg.attn_kwargs)
            attn_layer = block_cfg.attn_layer or model_cfg.attn_layer
            attn_layer = partial(get_attn(attn_layer), **attn_kwargs) if attn_layer is not None else None
        layer_fns = replace(layer_fns, attn=attn_layer)

    # override self-attn layer / args with block local cfg
    self_attn_set = block_cfg.self_attn_layer is not None
    if self_attn_set or block_cfg.self_attn_kwargs is not None:
        # override attn layer config
        if self_attn_set and not block_cfg.self_attn_layer:  # attn_layer == ''
            # empty string for self_attn_layer type will disable attn for this block
            self_attn_layer = None
        else:
            self_attn_kwargs = override_kwargs(block_cfg.self_attn_kwargs, model_cfg.self_attn_kwargs)
            self_attn_layer = block_cfg.self_attn_layer or model_cfg.self_attn_layer
            self_attn_layer = partial(get_attn(self_attn_layer), **self_attn_kwargs) \
                if self_attn_layer is not None else None
        layer_fns = replace(layer_fns, self_attn=self_attn_layer)

    block_kwargs['layers'] = layer_fns

    # add additional block_kwargs specified in block_cfg or model_cfg, precedence to block if set
    block_kwargs.update(override_kwargs(block_cfg.block_kwargs, model_cfg.block_kwargs))


def create_byob_stages(
        cfg: ByoModelCfg,
        drop_path_rate: float,
        output_stride: int,
        stem_feat: Dict[str, Any],
        feat_size: Optional[int] = None,
        layers: Optional[LayerFn] = None,
        block_kwargs_fn: Optional[Callable] = update_block_kwargs,
):

    layers = layers or LayerFn()
    feature_info = []
    block_cfgs = [expand_blocks_cfg(s) for s in cfg.blocks]
    depths = [sum([bc.d for bc in stage_bcs]) for stage_bcs in block_cfgs]
    dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
    dilation = 1
    net_stride = stem_feat['reduction']
    prev_chs = stem_feat['num_chs']
    prev_feat = stem_feat
    stages = []
    for stage_idx, stage_block_cfgs in enumerate(block_cfgs):
        stride = stage_block_cfgs[0].s
        if stride != 1 and prev_feat:
            feature_info.append(prev_feat)
        if net_stride >= output_stride and stride > 1:
            dilation *= stride
            stride = 1
        net_stride *= stride
        first_dilation = 1 if dilation in (1, 2) else 2

        blocks = []
        for block_idx, block_cfg in enumerate(stage_block_cfgs):
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
                layers=layers,
            )
            if block_cfg.type in ('self_attn',):
                # add feat_size arg for blocks that support/need it
                block_kwargs['feat_size'] = feat_size
            block_kwargs_fn(block_kwargs, block_cfg=block_cfg, model_cfg=cfg)
            blocks += [create_block(block_cfg.type, **block_kwargs)]
            first_dilation = dilation
            prev_chs = out_chs
            if stride > 1 and block_idx == 0:
                feat_size = reduce_feat_size(feat_size, stride)

        stages += [nn.Sequential(*blocks)]
        prev_feat = dict(num_chs=prev_chs, reduction=net_stride, module=f'stages.{stage_idx}', stage=stage_idx + 1)

    feature_info.append(prev_feat)
    return nn.Sequential(*stages), feature_info, feat_size


def get_layer_fns(cfg: ByoModelCfg, allow_aa: bool = True):
    act = get_act_layer(cfg.act_layer)
    norm_act = get_norm_act_layer(norm_layer=cfg.norm_layer, act_layer=act)
    if cfg.aa_layer and allow_aa:
        conv_norm_act = partial(ConvNormAct, norm_layer=cfg.norm_layer, act_layer=act, aa_layer=cfg.aa_layer)
    else:
        conv_norm_act = partial(ConvNormAct, norm_layer=cfg.norm_layer, act_layer=act)
    attn = partial(get_attn(cfg.attn_layer), **cfg.attn_kwargs) if cfg.attn_layer else None
    self_attn = partial(get_attn(cfg.self_attn_layer), **cfg.self_attn_kwargs) if cfg.self_attn_layer else None
    layer_fn = LayerFn(conv_norm_act=conv_norm_act, norm_act=norm_act, act=act, attn=attn, self_attn=self_attn)
    return layer_fn


class ByobNet(nn.Module):
    """ 'Bring-your-own-blocks' Net

    A flexible network backbone that allows building model stem + blocks via
    dataclass cfg definition w/ factory functions for module instantiation.

    Current assumption is that both stem and blocks are in conv-bn-act order (w/ block ending in act).
    """
    def __init__(
            self,
            cfg: ByoModelCfg,
            num_classes: int = 1000,
            in_chans: int = 3,
            global_pool: Optional[str] = None,
            output_stride: int = 32,
            img_size: Optional[Union[int, Tuple[int, int]]] = None,
            drop_rate: float = 0.,
            drop_path_rate: float =0.,
            zero_init_last: bool = True,
            **kwargs,
    ):
        """
        Args:
            cfg: Model architecture configuration.
            num_classes: Number of classifier classes.
            in_chans: Number of input channels.
            global_pool: Global pooling type.
            output_stride: Output stride of network, one of (8, 16, 32).
            img_size: Image size for fixed image size models (i.e. self-attn).
            drop_rate: Classifier dropout rate.
            drop_path_rate: Stochastic depth drop-path rate.
            zero_init_last: Zero-init last weight of residual path.
            **kwargs: Extra kwargs overlayed onto cfg.
        """
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        cfg = replace(cfg, **kwargs)  # overlay kwargs onto cfg
        stem_layers = get_layer_fns(cfg, allow_aa=False)  # keep aa off for stem-layers
        stage_layers = get_layer_fns(cfg)
        if cfg.fixed_input_size:
            assert img_size is not None, 'img_size argument is required for fixed input size model'
        feat_size = to_2tuple(img_size) if img_size is not None else None

        self.feature_info = []
        if isinstance(cfg.stem_chs, (list, tuple)):
            stem_chs = [int(round(c * cfg.width_factor)) for c in cfg.stem_chs]
        else:
            stem_chs = int(round((cfg.stem_chs or cfg.blocks[0].c) * cfg.width_factor))
        self.stem, stem_feat = create_byob_stem(
            in_chs=in_chans,
            out_chs=stem_chs,
            stem_type=cfg.stem_type,
            pool_type=cfg.stem_pool,
            layers=stem_layers,
        )
        self.feature_info.extend(stem_feat[:-1])
        feat_size = reduce_feat_size(feat_size, stride=stem_feat[-1]['reduction'])

        self.stages, stage_feat, feat_size = create_byob_stages(
            cfg,
            drop_path_rate,
            output_stride,
            stem_feat[-1],
            layers=stage_layers,
            feat_size=feat_size,
        )
        self.feature_info.extend(stage_feat[:-1])
        reduction = stage_feat[-1]['reduction']

        prev_chs = stage_feat[-1]['num_chs']
        if cfg.num_features:
            self.num_features = int(round(cfg.width_factor * cfg.num_features))
            self.final_conv = stage_layers.conv_norm_act(prev_chs, self.num_features, 1)
        else:
            self.num_features = prev_chs
            self.final_conv = nn.Identity()
        self.feature_info += [
            dict(num_chs=self.num_features, reduction=reduction, module='final_conv', stage=len(self.stages))]
        self.stage_ends = [f['stage'] for f in self.feature_info]

        self.head_hidden_size = self.num_features
        assert cfg.head_type in ('', 'classifier', 'mlp', 'attn_abs', 'attn_rot')
        if cfg.head_type == 'mlp':
            if global_pool is None:
                global_pool = 'avg'
            self.head = NormMlpClassifierHead(
                self.num_features,
                num_classes,
                hidden_size=cfg.head_hidden_size,
                pool_type=global_pool,
                norm_layer=cfg.norm_layer,
                act_layer=cfg.act_layer,
                drop_rate=self.drop_rate,
            )
            self.head_hidden_size = self.head.hidden_size
        elif cfg.head_type == 'attn_abs':
            if global_pool is None:
                global_pool = 'token'
            assert global_pool in ('', 'token')
            self.head = AttentionPool2d(
                self.num_features,
                embed_dim=cfg.head_hidden_size,
                out_features=num_classes,
                feat_size=feat_size,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
                qkv_separate=True,
            )
            self.head_hidden_size = self.head.embed_dim
        elif cfg.head_type =='attn_rot':
            if global_pool is None:
                global_pool = 'token'
            assert global_pool in ('', 'token')
            self.head = RotAttentionPool2d(
                self.num_features,
                embed_dim=cfg.head_hidden_size,
                out_features=num_classes,
                ref_feat_size=feat_size,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
                qkv_separate=True,
            )
            self.head_hidden_size = self.head.embed_dim
        else:
            if global_pool is None:
                global_pool = 'avg'
            assert cfg.head_hidden_size is None
            self.head = ClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
            )
        self.global_pool = global_pool

        # init weights
        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',
            blocks=[
                (r'^stages\.(\d+)' if coarse else r'^stages\.(\d+)\.(\d+)', None),
                (r'^final_conv', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            exclude_final_conv: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
            exclude_final_conv: Exclude final_conv from last intermediate
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stage_ends), indices)
        take_indices = [self.stage_ends[i] for i in take_indices]
        max_index = self.stage_ends[max_index]
        # forward pass
        feat_idx = 0  # stem is index 0
        if hasattr(self.stem, 'forward_intermediates'):
            # returns last intermediate features in stem (before final stride in stride > 2 stems)
            x, x_inter = self.stem.forward_intermediates(x)
        else:
            x, x_inter = self.stem(x), None
        if feat_idx in take_indices:
            intermediates.append(x if x_inter is None else x_inter)
        last_idx = self.stage_ends[-1]
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index]
        for stage in stages:
            feat_idx += 1
            x = stage(x)
            if not exclude_final_conv and feat_idx == last_idx:
                # default feature_info for this model uses final_conv as the last feature output (if present)
                x = self.final_conv(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        if exclude_final_conv and feat_idx == last_idx:
            x = self.final_conv(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.stage_ends), indices)
        max_index = self.stage_ends[max_index]
        self.stages = self.stages[:max_index]  # truncate blocks w/ stem as idx 0
        if max_index < self.stage_ends[-1]:
            self.final_conv = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices


    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        x = self.final_conv(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module, name='', zero_init_last=False):
    if isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights(zero_init_last=zero_init_last)


model_cfgs = dict(
    gernet_l=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='basic', d=1, c=128, s=2, gs=0, br=1.),
            ByoBlockCfg(type='basic', d=2, c=192, s=2, gs=0, br=1.),
            ByoBlockCfg(type='bottle', d=6, c=640, s=2, gs=0, br=1 / 4),
            ByoBlockCfg(type='bottle', d=5, c=640, s=2, gs=1, br=3.),
            ByoBlockCfg(type='bottle', d=4, c=640, s=1, gs=1, br=3.),
        ),
        stem_chs=32,
        stem_pool=None,
        num_features=2560,
    ),
    gernet_m=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='basic', d=1, c=128, s=2, gs=0, br=1.),
            ByoBlockCfg(type='basic', d=2, c=192, s=2, gs=0, br=1.),
            ByoBlockCfg(type='bottle', d=6, c=640, s=2, gs=0, br=1 / 4),
            ByoBlockCfg(type='bottle', d=4, c=640, s=2, gs=1, br=3.),
            ByoBlockCfg(type='bottle', d=1, c=640, s=1, gs=1, br=3.),
        ),
        stem_chs=32,
        stem_pool=None,
        num_features=2560,
    ),
    gernet_s=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='basic', d=1, c=48, s=2, gs=0, br=1.),
            ByoBlockCfg(type='basic', d=3, c=48, s=2, gs=0, br=1.),
            ByoBlockCfg(type='bottle', d=7, c=384, s=2, gs=0, br=1 / 4),
            ByoBlockCfg(type='bottle', d=2, c=560, s=2, gs=1, br=3.),
            ByoBlockCfg(type='bottle', d=1, c=256, s=1, gs=1, br=3.),
        ),
        stem_chs=13,
        stem_pool=None,
        num_features=1920,
    ),

    repvgg_a0=ByoModelCfg(
        blocks=_rep_vgg_bcfg(d=(2, 4, 14, 1), wf=(0.75, 0.75, 0.75, 2.5)),
        stem_type='rep',
        stem_chs=48,
    ),
    repvgg_a1=ByoModelCfg(
        blocks=_rep_vgg_bcfg(d=(2, 4, 14, 1), wf=(1, 1, 1, 2.5)),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_a2=ByoModelCfg(
        blocks=_rep_vgg_bcfg(d=(2, 4, 14, 1), wf=(1.5, 1.5, 1.5, 2.75)),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_b0=ByoModelCfg(
        blocks=_rep_vgg_bcfg(wf=(1., 1., 1., 2.5)),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_b1=ByoModelCfg(
        blocks=_rep_vgg_bcfg(wf=(2., 2., 2., 4.)),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_b1g4=ByoModelCfg(
        blocks=_rep_vgg_bcfg(wf=(2., 2., 2., 4.), groups=4),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_b2=ByoModelCfg(
        blocks=_rep_vgg_bcfg(wf=(2.5, 2.5, 2.5, 5.)),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_b2g4=ByoModelCfg(
        blocks=_rep_vgg_bcfg(wf=(2.5, 2.5, 2.5, 5.), groups=4),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_b3=ByoModelCfg(
        blocks=_rep_vgg_bcfg(wf=(3., 3., 3., 5.)),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_b3g4=ByoModelCfg(
        blocks=_rep_vgg_bcfg(wf=(3., 3., 3., 5.), groups=4),
        stem_type='rep',
        stem_chs=64,
    ),
    repvgg_d2se=ByoModelCfg(
        blocks=_rep_vgg_bcfg(d=(8, 14, 24, 1), wf=(2.5, 2.5, 2.5, 5.)),
        stem_type='rep',
        stem_chs=64,
        attn_layer='se',
        attn_kwargs=dict(rd_ratio=0.0625, rd_divisor=1),
    ),

    # 4 x conv stem w/ 2 act, no maxpool, 2,4,6,4 repeats, group size 32 in first 3 blocks
    # DW convs in last block, 2048 pre-FC, silu act
    resnet51q=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=4, c=512, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=6, c=1536, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=4, c=1536, s=2, gs=1, br=1.0),
        ),
        stem_chs=128,
        stem_type='quad2',
        stem_pool=None,
        num_features=2048,
        act_layer='silu',
    ),

    # 4 x conv stem w/ 4 act, no maxpool, 1,4,6,4 repeats, edge block first, group size 32 in next 2 blocks
    # DW convs in last block, 4 conv for each bottle block, 2048 pre-FC, silu act
    resnet61q=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='edge', d=1, c=256, s=1, gs=0, br=1.0, block_kwargs=dict()),
            ByoBlockCfg(type='bottle', d=4, c=512, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=6, c=1536, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=4, c=1536, s=2, gs=1, br=1.0),
        ),
        stem_chs=128,
        stem_type='quad',
        stem_pool=None,
        num_features=2048,
        act_layer='silu',
        block_kwargs=dict(extra_conv=True),
    ),

    # A series of ResNeXt-26 models w/ one of none, GC, SE, ECA, BAT attn, group size 32, SiLU act,
    # and a tiered stem w/ maxpool
    resnext26ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=1024, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=2048, s=2, gs=32, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        act_layer='silu',
    ),
    gcresnext26ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=1024, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=2048, s=2, gs=32, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        act_layer='silu',
        attn_layer='gca',
    ),
    seresnext26ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=1024, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=2048, s=2, gs=32, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        act_layer='silu',
        attn_layer='se',
    ),
    eca_resnext26ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=1024, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=2048, s=2, gs=32, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        act_layer='silu',
        attn_layer='eca',
    ),
    bat_resnext26ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=1024, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=2048, s=2, gs=32, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        act_layer='silu',
        attn_layer='bat',
        attn_kwargs=dict(block_size=8)
    ),

    # ResNet-32 (2, 3, 3, 2) models w/ no attn, no groups, SiLU act, no pre-fc feat layer, tiered stem w/o maxpool
    resnet32ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=3, c=512, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=3, c=1536, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=1536, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        num_features=0,
        act_layer='silu',
    ),

    # ResNet-33 (2, 3, 3, 2) models w/ no attn, no groups, SiLU act, 1280 pre-FC feat, tiered stem w/o maxpool
    resnet33ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=3, c=512, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=3, c=1536, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=1536, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        num_features=1280,
        act_layer='silu',
    ),

    # A series of ResNet-33 (2, 3, 3, 2) models w/ one of GC, SE, ECA attn, no groups, SiLU act, 1280 pre-FC feat
    # and a tiered stem w/ no maxpool
    gcresnet33ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=3, c=512, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=3, c=1536, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=1536, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        num_features=1280,
        act_layer='silu',
        attn_layer='gca',
    ),
    seresnet33ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=3, c=512, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=3, c=1536, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=1536, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        num_features=1280,
        act_layer='silu',
        attn_layer='se',
    ),
    eca_resnet33ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=3, c=512, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=3, c=1536, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=1536, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        num_features=1280,
        act_layer='silu',
        attn_layer='eca',
    ),

    gcresnet50t=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, br=0.25),
            ByoBlockCfg(type='bottle', d=4, c=512, s=2, br=0.25),
            ByoBlockCfg(type='bottle', d=6, c=1024, s=2, br=0.25),
            ByoBlockCfg(type='bottle', d=3, c=2048, s=2, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        attn_layer='gca',
    ),

    gcresnext50ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=4, c=512, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=6, c=1024, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=3, c=2048, s=2, gs=32, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        act_layer='silu',
        attn_layer='gca',
    ),

    # experimental models, closer to a RegNetZ than a ResNet. Similar to EfficientNets but w/ groups instead of DW
    regnetz_b16=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=48, s=2, gs=16, br=3),
            ByoBlockCfg(type='bottle', d=6, c=96, s=2, gs=16, br=3),
            ByoBlockCfg(type='bottle', d=12, c=192, s=2, gs=16, br=3),
            ByoBlockCfg(type='bottle', d=2, c=288, s=2, gs=16, br=3),
        ),
        stem_chs=32,
        stem_pool='',
        downsample='',
        num_features=1536,
        act_layer='silu',
        attn_layer='se',
        attn_kwargs=dict(rd_ratio=0.25),
        block_kwargs=dict(bottle_in=True, linear_out=True),
    ),
    regnetz_c16=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=48, s=2, gs=16, br=4),
            ByoBlockCfg(type='bottle', d=6, c=96, s=2, gs=16, br=4),
            ByoBlockCfg(type='bottle', d=12, c=192, s=2, gs=16, br=4),
            ByoBlockCfg(type='bottle', d=2, c=288, s=2, gs=16, br=4),
        ),
        stem_chs=32,
        stem_pool='',
        downsample='',
        num_features=1536,
        act_layer='silu',
        attn_layer='se',
        attn_kwargs=dict(rd_ratio=0.25),
        block_kwargs=dict(bottle_in=True, linear_out=True),
    ),
    regnetz_d32=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=64, s=1, gs=32, br=4),
            ByoBlockCfg(type='bottle', d=6, c=128, s=2, gs=32, br=4),
            ByoBlockCfg(type='bottle', d=12, c=256, s=2, gs=32, br=4),
            ByoBlockCfg(type='bottle', d=3, c=384, s=2, gs=32, br=4),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        downsample='',
        num_features=1792,
        act_layer='silu',
        attn_layer='se',
        attn_kwargs=dict(rd_ratio=0.25),
        block_kwargs=dict(bottle_in=True, linear_out=True),
    ),
    regnetz_d8=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=64, s=1, gs=8, br=4),
            ByoBlockCfg(type='bottle', d=6, c=128, s=2, gs=8, br=4),
            ByoBlockCfg(type='bottle', d=12, c=256, s=2, gs=8, br=4),
            ByoBlockCfg(type='bottle', d=3, c=384, s=2, gs=8, br=4),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        downsample='',
        num_features=1792,
        act_layer='silu',
        attn_layer='se',
        attn_kwargs=dict(rd_ratio=0.25),
        block_kwargs=dict(bottle_in=True, linear_out=True),
    ),
    regnetz_e8=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=96, s=1, gs=8, br=4),
            ByoBlockCfg(type='bottle', d=8, c=192, s=2, gs=8, br=4),
            ByoBlockCfg(type='bottle', d=16, c=384, s=2, gs=8, br=4),
            ByoBlockCfg(type='bottle', d=3, c=512, s=2, gs=8, br=4),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        downsample='',
        num_features=2048,
        act_layer='silu',
        attn_layer='se',
        attn_kwargs=dict(rd_ratio=0.25),
        block_kwargs=dict(bottle_in=True, linear_out=True),
    ),

    # experimental EvoNorm configs
    regnetz_b16_evos=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=48, s=2, gs=16, br=3),
            ByoBlockCfg(type='bottle', d=6, c=96, s=2, gs=16, br=3),
            ByoBlockCfg(type='bottle', d=12, c=192, s=2, gs=16, br=3),
            ByoBlockCfg(type='bottle', d=2, c=288, s=2, gs=16, br=3),
        ),
        stem_chs=32,
        stem_pool='',
        downsample='',
        num_features=1536,
        act_layer='silu',
        norm_layer=partial(EvoNorm2dS0a, group_size=16),
        attn_layer='se',
        attn_kwargs=dict(rd_ratio=0.25),
        block_kwargs=dict(bottle_in=True, linear_out=True),
    ),
    regnetz_c16_evos=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=48, s=2, gs=16, br=4),
            ByoBlockCfg(type='bottle', d=6, c=96, s=2, gs=16, br=4),
            ByoBlockCfg(type='bottle', d=12, c=192, s=2, gs=16, br=4),
            ByoBlockCfg(type='bottle', d=2, c=288, s=2, gs=16, br=4),
        ),
        stem_chs=32,
        stem_pool='',
        downsample='',
        num_features=1536,
        act_layer='silu',
        norm_layer=partial(EvoNorm2dS0a, group_size=16),
        attn_layer='se',
        attn_kwargs=dict(rd_ratio=0.25),
        block_kwargs=dict(bottle_in=True, linear_out=True),
    ),
    regnetz_d8_evos=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=64, s=1, gs=8, br=4),
            ByoBlockCfg(type='bottle', d=6, c=128, s=2, gs=8, br=4),
            ByoBlockCfg(type='bottle', d=12, c=256, s=2, gs=8, br=4),
            ByoBlockCfg(type='bottle', d=3, c=384, s=2, gs=8, br=4),
        ),
        stem_chs=64,
        stem_type='deep',
        stem_pool='',
        downsample='',
        num_features=1792,
        act_layer='silu',
        norm_layer=partial(EvoNorm2dS0a, group_size=16),
        attn_layer='se',
        attn_kwargs=dict(rd_ratio=0.25),
        block_kwargs=dict(bottle_in=True, linear_out=True),
    ),

    mobileone_s0=ByoModelCfg(
        blocks=_mobileone_bcfg(wf=(0.75, 1.0, 1.0, 2.), num_conv_branches=4),
        stem_type='one',
        stem_chs=48,
    ),
    mobileone_s1=ByoModelCfg(
        blocks=_mobileone_bcfg(wf=(1.5, 1.5, 2.0, 2.5)),
        stem_type='one',
        stem_chs=64,
    ),
    mobileone_s2=ByoModelCfg(
        blocks=_mobileone_bcfg(wf=(1.5, 2.0, 2.5, 4.0)),
        stem_type='one',
        stem_chs=64,
    ),
    mobileone_s3=ByoModelCfg(
        blocks=_mobileone_bcfg(wf=(2.0, 2.5, 3.0, 4.0)),
        stem_type='one',
        stem_chs=64,
    ),
    mobileone_s4=ByoModelCfg(
        blocks=_mobileone_bcfg(wf=(3.0, 3.5, 3.5, 4.0), se_blocks=(0, 0, 5, 1)),
        stem_type='one',
        stem_chs=64,
    ),

    resnet50_clip=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, br=0.25),
            ByoBlockCfg(type='bottle', d=4, c=512, s=2, br=0.25),
            ByoBlockCfg(type='bottle', d=6, c=1024, s=2, br=0.25),
            ByoBlockCfg(type='bottle', d=3, c=2048, s=2, br=0.25),
        ),
        stem_chs=(32, 32, 64),
        stem_type='',
        stem_pool='avg2',
        downsample='avg',
        aa_layer='avg',
        head_type='attn_abs',
    ),
    resnet101_clip=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, br=0.25),
            ByoBlockCfg(type='bottle', d=4, c=512, s=2, br=0.25),
            ByoBlockCfg(type='bottle', d=23, c=1024, s=2, br=0.25),
            ByoBlockCfg(type='bottle', d=3, c=2048, s=2, br=0.25),
        ),
        stem_chs=(32, 32, 64),
        stem_type='',
        stem_pool='avg2',
        downsample='avg',
        aa_layer='avg',
        head_type='attn_abs',
    ),
    resnet50x4_clip=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=4, c=256, s=1, br=0.25),
            ByoBlockCfg(type='bottle', d=6, c=512, s=2, br=0.25),
            ByoBlockCfg(type='bottle', d=10, c=1024, s=2, br=0.25),
            ByoBlockCfg(type='bottle', d=6, c=2048, s=2, br=0.25),
        ),
        width_factor=1.25,
        stem_chs=(32, 32, 64),
        stem_type='',
        stem_pool='avg2',
        downsample='avg',
        aa_layer='avg',
        head_type='attn_abs',
    ),
    resnet50x16_clip=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=6, c=256, s=1, br=0.25),
            ByoBlockCfg(type='bottle', d=8, c=512, s=2, br=0.25),
            ByoBlockCfg(type='bottle', d=18, c=1024, s=2, br=0.25),
            ByoBlockCfg(type='bottle', d=8, c=2048, s=2, br=0.25),
        ),
        width_factor=1.5,
        stem_chs=(32, 32, 64),
        stem_type='',
        stem_pool='avg2',
        downsample='avg',
        aa_layer='avg',
        head_type='attn_abs',
    ),
    resnet50x64_clip=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, br=0.25),
            ByoBlockCfg(type='bottle', d=15, c=512, s=2, br=0.25),
            ByoBlockCfg(type='bottle', d=36, c=1024, s=2, br=0.25),
            ByoBlockCfg(type='bottle', d=10, c=2048, s=2, br=0.25),
        ),
        width_factor=2.0,
        stem_chs=(32, 32, 64),
        stem_type='',
        stem_pool='avg2',
        downsample='avg',
        aa_layer='avg',
        head_type='attn_abs',
    ),

    resnet50_mlp=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, br=0.25),
            ByoBlockCfg(type='bottle', d=4, c=512, s=2, br=0.25),
            ByoBlockCfg(type='bottle', d=6, c=1024, s=2, br=0.25),
            ByoBlockCfg(type='bottle', d=3, c=2048, s=2, br=0.25),
        ),
        stem_chs=(32, 32, 64),
        stem_type='',
        stem_pool='avg2',
        downsample='avg',
        aa_layer='avg',
        head_hidden_size=1024,
        head_type='mlp',
    ),

    test_byobnet=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='edge', d=1, c=32, s=2, gs=0, br=0.5),
            ByoBlockCfg(type='dark', d=1, c=64, s=2, gs=0, br=0.5),
            ByoBlockCfg(type='basic', d=1, c=128, s=2, gs=32, br=0.25),
            ByoBlockCfg(type='bottle', d=1, c=256, s=2, gs=64, br=0.25),
        ),
        stem_chs=24,
        downsample='avg',
        stem_pool='',
        act_layer='relu',
        attn_layer='se',
        attn_kwargs=dict(rd_ratio=0.25),
    ),
)
for k in ('resnet50_clip', 'resnet101_clip', 'resnet50x4_clip', 'resnet50x16_clip', 'resnet50x64_clip'):
    model_cfgs[k + '_gap'] = replace(model_cfgs[k], head_type='classifier')


def _convert_openai_clip(
        state_dict: Dict[str, torch.Tensor],
        model: ByobNet,
        prefix: str = 'visual.',
) -> Dict[str, torch.Tensor]:
    model_has_attn_pool = isinstance(model.head, (RotAttentionPool2d, AttentionPool2d))
    import re

    def _stage_sub(m):
        stage_idx = int(m.group(1)) - 1
        layer_idx, layer_type, layer_id = int(m.group(2)), m.group(3), int(m.group(4))
        prefix_str = f'stages.{stage_idx}.{layer_idx}.'
        id_map = {1: 'conv1_1x1.', 2: 'conv2_kxk.', 3: 'conv3_1x1.'}
        suffix_str = id_map[layer_id] + layer_type
        return prefix_str + suffix_str

    def _down_sub(m):
        stage_idx = int(m.group(1)) - 1
        layer_idx, layer_id = int(m.group(2)), int(m.group(3))
        return f'stages.{stage_idx}.{layer_idx}.shortcut.' + ('conv.conv' if layer_id == 0 else 'conv.bn')

    out_dict = {}
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = re.sub(rf'{prefix}conv([0-9])', r'stem.conv\1.conv', k)
        k = re.sub(rf'{prefix}bn([0-9])', r'stem.conv\1.bn', k)
        k = re.sub(rf'{prefix}layer([0-9])\.([0-9]+)\.([a-z]+)([0-9])', _stage_sub, k)
        k = re.sub(rf'{prefix}layer([0-9])\.([0-9]+)\.downsample\.([0-9])', _down_sub, k)
        if k.startswith(f'{prefix}attnpool'):
            if not model_has_attn_pool:
                continue
            k = k.replace(prefix + 'attnpool', 'head')  #'attn_pool')
            k = k.replace('positional_embedding', 'pos_embed')
            k = k.replace('q_proj', 'q')
            k = k.replace('k_proj', 'k')
            k = k.replace('v_proj', 'v')
            k = k.replace('c_proj', 'proj')
        out_dict[k] = v

    return out_dict


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: ByobNet
):
    if 'visual.conv1.weight' in state_dict:
        state_dict = _convert_openai_clip(state_dict, model)
    return state_dict


def _create_byobnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ByobNet, variant, pretrained,
        model_cfg=model_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True),
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
        **kwargs
    }


def _cfgr(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 256, 256), 'pool_size': (8, 8),
        'crop_pct': 0.9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1.conv', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # GPU-Efficient (ResNet) weights
    'gernet_s.idstcv_in1k': _cfg(hf_hub_id='timm/'),
    'gernet_m.idstcv_in1k': _cfg(hf_hub_id='timm/'),
    'gernet_l.idstcv_in1k': _cfg(hf_hub_id='timm/', input_size=(3, 256, 256), pool_size=(8, 8)),

    # RepVGG weights
    'repvgg_a0.rvgg_in1k': _cfg(
        hf_hub_id='timm/',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv'), license='mit'),
    'repvgg_a1.rvgg_in1k': _cfg(
        hf_hub_id='timm/',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv'), license='mit'),
    'repvgg_a2.rvgg_in1k': _cfg(
        hf_hub_id='timm/',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv'), license='mit'),
    'repvgg_b0.rvgg_in1k': _cfg(
        hf_hub_id='timm/',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv'), license='mit'),
    'repvgg_b1.rvgg_in1k': _cfg(
        hf_hub_id='timm/',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv'), license='mit'),
    'repvgg_b1g4.rvgg_in1k': _cfg(
        hf_hub_id='timm/',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv'), license='mit'),
    'repvgg_b2.rvgg_in1k': _cfg(
        hf_hub_id='timm/',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv'), license='mit'),
    'repvgg_b2g4.rvgg_in1k': _cfg(
        hf_hub_id='timm/',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv'), license='mit'),
    'repvgg_b3.rvgg_in1k': _cfg(
        hf_hub_id='timm/',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv'), license='mit'),
    'repvgg_b3g4.rvgg_in1k': _cfg(
        hf_hub_id='timm/',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv'), license='mit'),
    'repvgg_d2se.rvgg_in1k': _cfg(
        hf_hub_id='timm/',
        first_conv=('stem.conv_kxk.conv', 'stem.conv_1x1.conv'), license='mit',
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=1.0,
    ),

    # experimental ResNet configs
    'resnet51q.ra2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet51q_ra2-d47dcc76.pth',
        first_conv='stem.conv1', input_size=(3, 256, 256), pool_size=(8, 8),
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'resnet61q.ra2_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet61q_ra2-6afc536c.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),

    # ResNeXt-26 models with different attention in Bottleneck blocks
    'resnext26ts.ra2_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/resnext26ts_256_ra2-8bbd9106.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'seresnext26ts.ch_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/seresnext26ts_256-6f0d74a3.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'gcresnext26ts.ch_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/gcresnext26ts_256-e414378b.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'eca_resnext26ts.ch_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/eca_resnext26ts_256-5a1d030f.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'bat_resnext26ts.ch_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/bat_resnext26ts_256-fa6fd595.pth',
        min_input_size=(3, 256, 256)),

    # ResNet-32 / 33 models with different attention in Bottleneck blocks
    'resnet32ts.ra2_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/resnet32ts_256-aacf5250.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'resnet33ts.ra2_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/resnet33ts_256-e91b09a4.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'gcresnet33ts.ra2_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/gcresnet33ts_256-0e0cd345.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'seresnet33ts.ra2_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/seresnet33ts_256-f8ad44d9.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'eca_resnet33ts.ra2_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/eca_resnet33ts_256-8f98face.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),

    'gcresnet50t.ra2_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/gcresnet50t_256-96374d1c.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),

    'gcresnext50ts.ch_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/gcresnext50ts_256-3e0f515e.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),

    # custom `timm` specific RegNetZ inspired models w/ different sizing from paper
    'regnetz_b16.ra3_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/regnetz_b_raa-677d9606.pth',
        first_conv='stem.conv', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 224, 224), pool_size=(7, 7), crop_pct=0.94, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'regnetz_c16.ra3_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/regnetz_c_rab2_256-a54bf36a.pth',
        first_conv='stem.conv', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        crop_pct=0.94, test_input_size=(3, 320, 320), test_crop_pct=1.0),
    'regnetz_d32.ra3_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/regnetz_d_rab_256-b8073a89.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=0.95, test_input_size=(3, 320, 320)),
    'regnetz_d8.ra3_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/regnetz_d8_bh-afc03c55.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=0.94, test_input_size=(3, 320, 320), test_crop_pct=1.0),
    'regnetz_e8.ra3_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/regnetz_e8_bh-aace8e6e.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=0.94, test_input_size=(3, 320, 320), test_crop_pct=1.0),

    'regnetz_b16_evos.untrained': _cfgr(
        first_conv='stem.conv', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 224, 224), pool_size=(7, 7), crop_pct=0.95, test_input_size=(3, 288, 288)),
    'regnetz_c16_evos.ch_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/regnetz_c16_evos_ch-d8311942.pth',
        first_conv='stem.conv', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        crop_pct=0.95, test_input_size=(3, 320, 320)),
    'regnetz_d8_evos.ch_in1k': _cfgr(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/regnetz_d8_evos_ch-2bc12646.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=0.95, test_input_size=(3, 320, 320), test_crop_pct=1.0),

    'mobileone_s0.apple_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.875,
        first_conv=('stem.conv_kxk.0.conv', 'stem.conv_scale.conv'),
    ),
    'mobileone_s1.apple_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.9,
        first_conv=('stem.conv_kxk.0.conv', 'stem.conv_scale.conv'),
    ),
    'mobileone_s2.apple_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.9,
        first_conv=('stem.conv_kxk.0.conv', 'stem.conv_scale.conv'),
    ),
    'mobileone_s3.apple_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.9,
        first_conv=('stem.conv_kxk.0.conv', 'stem.conv_scale.conv'),
    ),
    'mobileone_s4.apple_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.9,
        first_conv=('stem.conv_kxk.0.conv', 'stem.conv_scale.conv'),
    ),

    # original attention pool head variants
    'resnet50_clip.openai': _cfgr(
        hf_hub_id='timm/',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=1024, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        fixed_input_size=True, input_size=(3, 224, 224), pool_size=(7, 7),
        classifier='head.proj',
    ),
    'resnet101_clip.openai': _cfgr(
        hf_hub_id='timm/',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=512, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        fixed_input_size=True, input_size=(3, 224, 224), pool_size=(7, 7),
        classifier='head.proj',
    ),
    'resnet50x4_clip.openai': _cfgr(
        hf_hub_id='timm/',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=640, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        fixed_input_size=True, input_size=(3, 288, 288), pool_size=(9, 9),
        classifier='head.proj',
    ),
    'resnet50x16_clip.openai': _cfgr(
        hf_hub_id='timm/',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=768, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        fixed_input_size=True, input_size=(3, 384, 384), pool_size=(12, 12),
        classifier='head.proj',
    ),
    'resnet50x64_clip.openai': _cfgr(
        hf_hub_id='timm/',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=1024, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        fixed_input_size=True, input_size=(3, 448, 448), pool_size=(14, 14),
        classifier='head.proj',
    ),
    'resnet50_clip.cc12m': _cfgr(
        hf_hub_id='timm/',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=1024, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        fixed_input_size=True, input_size=(3, 224, 224), pool_size=(7, 7),
        classifier='head.proj',
    ),
    'resnet50_clip.yfcc15m': _cfgr(
        hf_hub_id='timm/',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=1024, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        fixed_input_size=True, input_size=(3, 224, 224), pool_size=(7, 7),
        classifier='head.proj',
    ),
    'resnet101_clip.yfcc15m': _cfgr(
        hf_hub_id='timm/',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=512, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        fixed_input_size=True, input_size=(3, 224, 224), pool_size=(7, 7),
        classifier='head.proj',
    ),

    # avg-pool w/ optional standard classifier head variants
    'resnet50_clip_gap.openai': _cfgr(
        hf_hub_id='timm/resnet50_clip.openai',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=0, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 224, 224), pool_size=(7, 7),
    ),
    'resnet101_clip_gap.openai': _cfgr(
        hf_hub_id='timm/resnet101_clip.openai',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=0, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 224, 224), pool_size=(7, 7),
    ),
    'resnet50x4_clip_gap.openai': _cfgr(
        hf_hub_id='timm/resnet50x4_clip.openai',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=0, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 288, 288), pool_size=(9, 9),
    ),
    'resnet50x16_clip_gap.openai': _cfgr(
        hf_hub_id='timm/resnet50x16_clip.openai',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=0, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 384, 384), pool_size=(12, 12),
    ),
    'resnet50x64_clip_gap.openai': _cfgr(
        hf_hub_id='timm/resnet50x64_clip.openai',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=0, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 448, 448), pool_size=(14, 14),
    ),
    'resnet50_clip_gap.cc12m': _cfgr(
        hf_hub_id='timm/resnet50_clip.cc12m',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=0, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 224, 224), pool_size=(7, 7),
    ),
    'resnet50_clip_gap.yfcc15m': _cfgr(
        hf_hub_id='timm/resnet50_clip.yfcc15m',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=0, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 224, 224), pool_size=(7, 7),
    ),
    'resnet101_clip_gap.yfcc15m': _cfgr(
        hf_hub_id='timm/resnet101_clip.yfcc15m',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=0, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 224, 224), pool_size=(7, 7),
    ),

    'resnet50_mlp.untrained': _cfgr(
        input_size=(3, 256, 256), pool_size=(8, 8),
    ),

    'test_byobnet.r160_in1k': _cfgr(
        hf_hub_id='timm/',
        first_conv='stem.conv',
        input_size=(3, 160, 160), crop_pct=0.95, pool_size=(5, 5),
    ),
})


@register_model
def gernet_l(pretrained=False, **kwargs) -> ByobNet:
    """ GEResNet-Large (GENet-Large from official impl)
    `Neural Architecture Design for GPU-Efficient Networks` - https://arxiv.org/abs/2006.14090
    """
    return _create_byobnet('gernet_l', pretrained=pretrained, **kwargs)


@register_model
def gernet_m(pretrained=False, **kwargs) -> ByobNet:
    """ GEResNet-Medium (GENet-Normal from official impl)
    `Neural Architecture Design for GPU-Efficient Networks` - https://arxiv.org/abs/2006.14090
    """
    return _create_byobnet('gernet_m', pretrained=pretrained, **kwargs)


@register_model
def gernet_s(pretrained=False, **kwargs) -> ByobNet:
    """ EResNet-Small (GENet-Small from official impl)
    `Neural Architecture Design for GPU-Efficient Networks` - https://arxiv.org/abs/2006.14090
    """
    return _create_byobnet('gernet_s', pretrained=pretrained, **kwargs)


@register_model
def repvgg_a0(pretrained=False, **kwargs) -> ByobNet:
    """ RepVGG-A0
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_a0', pretrained=pretrained, **kwargs)


@register_model
def repvgg_a1(pretrained=False, **kwargs) -> ByobNet:
    """ RepVGG-A1
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_a1', pretrained=pretrained, **kwargs)


@register_model
def repvgg_a2(pretrained=False, **kwargs) -> ByobNet:
    """ RepVGG-A2
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_a2', pretrained=pretrained, **kwargs)


@register_model
def repvgg_b0(pretrained=False, **kwargs) -> ByobNet:
    """ RepVGG-B0
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_b0', pretrained=pretrained, **kwargs)


@register_model
def repvgg_b1(pretrained=False, **kwargs) -> ByobNet:
    """ RepVGG-B1
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_b1', pretrained=pretrained, **kwargs)


@register_model
def repvgg_b1g4(pretrained=False, **kwargs) -> ByobNet:
    """ RepVGG-B1g4
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_b1g4', pretrained=pretrained, **kwargs)


@register_model
def repvgg_b2(pretrained=False, **kwargs) -> ByobNet:
    """ RepVGG-B2
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_b2', pretrained=pretrained, **kwargs)


@register_model
def repvgg_b2g4(pretrained=False, **kwargs) -> ByobNet:
    """ RepVGG-B2g4
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_b2g4', pretrained=pretrained, **kwargs)


@register_model
def repvgg_b3(pretrained=False, **kwargs) -> ByobNet:
    """ RepVGG-B3
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_b3', pretrained=pretrained, **kwargs)


@register_model
def repvgg_b3g4(pretrained=False, **kwargs) -> ByobNet:
    """ RepVGG-B3g4
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_b3g4', pretrained=pretrained, **kwargs)


@register_model
def repvgg_d2se(pretrained=False, **kwargs) -> ByobNet:
    """ RepVGG-D2se
    `Making VGG-style ConvNets Great Again` - https://arxiv.org/abs/2101.03697
    """
    return _create_byobnet('repvgg_d2se', pretrained=pretrained, **kwargs)


@register_model
def resnet51q(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('resnet51q', pretrained=pretrained, **kwargs)


@register_model
def resnet61q(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('resnet61q', pretrained=pretrained, **kwargs)


@register_model
def resnext26ts(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('resnext26ts', pretrained=pretrained, **kwargs)


@register_model
def gcresnext26ts(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('gcresnext26ts', pretrained=pretrained, **kwargs)


@register_model
def seresnext26ts(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('seresnext26ts', pretrained=pretrained, **kwargs)


@register_model
def eca_resnext26ts(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('eca_resnext26ts', pretrained=pretrained, **kwargs)


@register_model
def bat_resnext26ts(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('bat_resnext26ts', pretrained=pretrained, **kwargs)


@register_model
def resnet32ts(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('resnet32ts', pretrained=pretrained, **kwargs)


@register_model
def resnet33ts(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('resnet33ts', pretrained=pretrained, **kwargs)


@register_model
def gcresnet33ts(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('gcresnet33ts', pretrained=pretrained, **kwargs)


@register_model
def seresnet33ts(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('seresnet33ts', pretrained=pretrained, **kwargs)


@register_model
def eca_resnet33ts(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('eca_resnet33ts', pretrained=pretrained, **kwargs)


@register_model
def gcresnet50t(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('gcresnet50t', pretrained=pretrained, **kwargs)


@register_model
def gcresnext50ts(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('gcresnext50ts', pretrained=pretrained, **kwargs)


@register_model
def regnetz_b16(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('regnetz_b16', pretrained=pretrained, **kwargs)


@register_model
def regnetz_c16(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('regnetz_c16', pretrained=pretrained, **kwargs)


@register_model
def regnetz_d32(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('regnetz_d32', pretrained=pretrained, **kwargs)


@register_model
def regnetz_d8(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('regnetz_d8', pretrained=pretrained, **kwargs)


@register_model
def regnetz_e8(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('regnetz_e8', pretrained=pretrained, **kwargs)


@register_model
def regnetz_b16_evos(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('regnetz_b16_evos', pretrained=pretrained, **kwargs)


@register_model
def regnetz_c16_evos(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('regnetz_c16_evos', pretrained=pretrained, **kwargs)


@register_model
def regnetz_d8_evos(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('regnetz_d8_evos', pretrained=pretrained, **kwargs)


@register_model
def mobileone_s0(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('mobileone_s0', pretrained=pretrained, **kwargs)


@register_model
def mobileone_s1(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('mobileone_s1', pretrained=pretrained, **kwargs)


@register_model
def mobileone_s2(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('mobileone_s2', pretrained=pretrained, **kwargs)


@register_model
def mobileone_s3(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('mobileone_s3', pretrained=pretrained, **kwargs)


@register_model
def mobileone_s4(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('mobileone_s4', pretrained=pretrained, **kwargs)


@register_model
def resnet50_clip(pretrained=False, **kwargs) -> ByobNet:
    """ OpenAI Modified ResNet-50 CLIP image tower
    """
    return _create_byobnet('resnet50_clip', pretrained=pretrained, **kwargs)


@register_model
def resnet101_clip(pretrained=False, **kwargs) -> ByobNet:
    """ OpenAI Modified ResNet-101 CLIP image tower
    """
    return _create_byobnet('resnet101_clip', pretrained=pretrained, **kwargs)


@register_model
def resnet50x4_clip(pretrained=False, **kwargs) -> ByobNet:
    """ OpenAI Modified ResNet-50x4 CLIP image tower
    """
    return _create_byobnet('resnet50x4_clip', pretrained=pretrained, **kwargs)


@register_model
def resnet50x16_clip(pretrained=False, **kwargs) -> ByobNet:
    """ OpenAI Modified ResNet-50x16 CLIP image tower
    """
    return _create_byobnet('resnet50x16_clip', pretrained=pretrained, **kwargs)


@register_model
def resnet50x64_clip(pretrained=False, **kwargs) -> ByobNet:
    """ OpenAI Modified ResNet-50x64 CLIP image tower
    """
    return _create_byobnet('resnet50x64_clip', pretrained=pretrained, **kwargs)


@register_model
def resnet50_clip_gap(pretrained=False, **kwargs) -> ByobNet:
    """ OpenAI Modified ResNet-50 CLIP image tower w/ avg pool (no attention pool)
    """
    return _create_byobnet('resnet50_clip_gap', pretrained=pretrained, **kwargs)


@register_model
def resnet101_clip_gap(pretrained=False, **kwargs) -> ByobNet:
    """ OpenAI Modified ResNet-101 CLIP image tower w/ avg pool (no attention pool)
    """
    return _create_byobnet('resnet101_clip_gap', pretrained=pretrained, **kwargs)


@register_model
def resnet50x4_clip_gap(pretrained=False, **kwargs) -> ByobNet:
    """ OpenAI Modified ResNet-50x4 CLIP image tower w/ avg pool (no attention pool)
    """
    return _create_byobnet('resnet50x4_clip_gap', pretrained=pretrained, **kwargs)


@register_model
def resnet50x16_clip_gap(pretrained=False, **kwargs) -> ByobNet:
    """ OpenAI Modified ResNet-50x16 CLIP image tower w/ avg pool (no attention pool)
    """
    return _create_byobnet('resnet50x16_clip_gap', pretrained=pretrained, **kwargs)


@register_model
def resnet50x64_clip_gap(pretrained=False, **kwargs) -> ByobNet:
    """ OpenAI Modified ResNet-50x64 CLIP image tower w/ avg pool (no attention pool)
    """
    return _create_byobnet('resnet50x64_clip_gap', pretrained=pretrained, **kwargs)


@register_model
def resnet50_mlp(pretrained=False, **kwargs) -> ByobNet:
    """
    """
    return _create_byobnet('resnet50_mlp', pretrained=pretrained, **kwargs)


@register_model
def test_byobnet(pretrained=False, **kwargs) -> ByobNet:
    """ Minimal test ResNet (BYOB based) model.
    """
    return _create_byobnet('test_byobnet', pretrained=pretrained, **kwargs)
