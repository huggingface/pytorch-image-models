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
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Tuple, List, Dict, Optional, Union, Any, Callable, Sequence

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import ClassifierHead, ConvNormAct, BatchNormAct2d, DropPath, AvgPool2dSame, \
    create_conv2d, get_act_layer, get_norm_act_layer, get_attn, make_divisible, to_2tuple, EvoNorm2dS0a
from ._builder import build_model_with_cfg
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
    stem_chs: int = 32
    width_factor: float = 1.0
    num_features: int = 0  # num out_channels for final conv, no final 1x1 conv if 0
    zero_init_last: bool = True  # zero init last weight (usually bn) in residual path
    fixed_input_size: bool = False  # model constrained to a fixed-input size / img_size must be provided on creation

    act_layer: str = 'relu'
    norm_layer: str = 'batchnorm'

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
        x = self.conv2_kxk(x)
        x = self.attn(x)
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

    This version does not currently support the deploy optimization. It is currently fixed in 'train' mode.
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
    ):
        super(RepVggBlock, self).__init__()
        layers = layers or LayerFn()
        groups = num_groups(group_size, in_chs)

        use_ident = in_chs == out_chs and stride == 1 and dilation[0] == dilation[1]
        self.identity = layers.norm_act(out_chs, apply_act=False) if use_ident else None
        self.conv_kxk = layers.conv_norm_act(
            in_chs, out_chs, kernel_size,
            stride=stride, dilation=dilation[0], groups=groups, drop_layer=drop_block, apply_act=False,
        )
        self.conv_1x1 = layers.conv_norm_act(in_chs, out_chs, 1, stride=stride, groups=groups, apply_act=False)
        self.attn = nn.Identity() if layers.attn is None else layers.attn(out_chs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. and use_ident else nn.Identity()
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
        if self.identity is None:
            x = self.conv_1x1(x) + self.conv_kxk(x)
        else:
            identity = self.identity(x)
            x = self.conv_1x1(x) + self.conv_kxk(x)
            x = self.drop_path(x)  # not in the paper / official impl, experimental
            x = x + identity
        x = self.attn(x)  # no attn in the paper / official impl, experimental
        return self.act(x)


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
            out_chs: int,
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
        for i, (ch, s, na) in enumerate(zip(stem_chs, stem_strides, stem_norm_acts)):
            layer_fn = layers.conv_norm_act if na else create_conv2d
            conv_name = f'conv{i + 1}'
            if i > 0 and s > 1:
                self.feature_info.append(dict(num_chs=prev_chs, reduction=curr_stride, module=prev_feat))
            self.add_module(conv_name, layer_fn(prev_chs, ch, kernel_size=kernel_size, stride=s))
            prev_chs = ch
            curr_stride *= s
            prev_feat = conv_name

        if pool and 'max' in pool.lower():
            self.feature_info.append(dict(num_chs=prev_chs, reduction=curr_stride, module=prev_feat))
            self.add_module('pool', nn.MaxPool2d(3, 2, 1))
            curr_stride *= 2
            prev_feat = 'pool'

        self.feature_info.append(dict(num_chs=prev_chs, reduction=curr_stride, module=prev_feat))
        assert curr_stride == stride


def create_byob_stem(
        in_chs: int,
        out_chs: int,
        stem_type: str = '',
        pool_type: str = '',
        feat_prefix: str = 'stem',
        layers: LayerFn = None,
):
    layers = layers or LayerFn()
    assert stem_type in ('', 'quad', 'quad2', 'tiered', 'deep', 'rep', '7x7', '3x3')
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
    elif '7x7' in stem_type:
        # 7x7 stem conv as in ResNet
        if pool_type:
            stem = Stem(in_chs, out_chs, 7, num_rep=1, pool=pool_type, layers=layers)
        else:
            stem = layers.conv_norm_act(in_chs, out_chs, 7, stride=2)
    else:
        # 3x3 stem conv as in RegNet is the default
        if pool_type:
            stem = Stem(in_chs, out_chs, 3, num_rep=1, pool=pool_type, layers=layers)
        else:
            stem = layers.conv_norm_act(in_chs, out_chs, 3, stride=2)

    if isinstance(stem, Stem):
        feature_info = [dict(f, module='.'.join([feat_prefix, f['module']])) for f in stem.feature_info]
    else:
        feature_info = [dict(num_chs=out_chs, reduction=2, module=feat_prefix)]
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
        prev_feat = dict(num_chs=prev_chs, reduction=net_stride, module=f'stages.{stage_idx}')

    feature_info.append(prev_feat)
    return nn.Sequential(*stages), feature_info


def get_layer_fns(cfg: ByoModelCfg):
    act = get_act_layer(cfg.act_layer)
    norm_act = get_norm_act_layer(norm_layer=cfg.norm_layer, act_layer=act)
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
            global_pool: str = 'avg',
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
        layers = get_layer_fns(cfg)
        if cfg.fixed_input_size:
            assert img_size is not None, 'img_size argument is required for fixed input size model'
        feat_size = to_2tuple(img_size) if img_size is not None else None

        self.feature_info = []
        stem_chs = int(round((cfg.stem_chs or cfg.blocks[0].c) * cfg.width_factor))
        self.stem, stem_feat = create_byob_stem(in_chans, stem_chs, cfg.stem_type, cfg.stem_pool, layers=layers)
        self.feature_info.extend(stem_feat[:-1])
        feat_size = reduce_feat_size(feat_size, stride=stem_feat[-1]['reduction'])

        self.stages, stage_feat = create_byob_stages(
            cfg,
            drop_path_rate,
            output_stride,
            stem_feat[-1],
            layers=layers,
            feat_size=feat_size,
        )
        self.feature_info.extend(stage_feat[:-1])

        prev_chs = stage_feat[-1]['num_chs']
        if cfg.num_features:
            self.num_features = int(round(cfg.width_factor * cfg.num_features))
            self.final_conv = layers.conv_norm_act(prev_chs, self.num_features, 1)
        else:
            self.num_features = prev_chs
            self.final_conv = nn.Identity()
        self.feature_info += [
            dict(num_chs=self.num_features, reduction=stage_feat[-1]['reduction'], module='final_conv')]

        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=self.drop_rate,
        )

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
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        x = self.final_conv(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

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
)


def _create_byobnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ByobNet, variant, pretrained,
        model_cfg=model_cfgs[variant],
        feature_cfg=dict(flatten_sequential=True),
        **kwargs)


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
})


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


@register_model
def resnet51q(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('resnet51q', pretrained=pretrained, **kwargs)


@register_model
def resnet61q(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('resnet61q', pretrained=pretrained, **kwargs)


@register_model
def resnext26ts(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('resnext26ts', pretrained=pretrained, **kwargs)


@register_model
def gcresnext26ts(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('gcresnext26ts', pretrained=pretrained, **kwargs)


@register_model
def seresnext26ts(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('seresnext26ts', pretrained=pretrained, **kwargs)


@register_model
def eca_resnext26ts(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('eca_resnext26ts', pretrained=pretrained, **kwargs)


@register_model
def bat_resnext26ts(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('bat_resnext26ts', pretrained=pretrained, **kwargs)


@register_model
def resnet32ts(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('resnet32ts', pretrained=pretrained, **kwargs)


@register_model
def resnet33ts(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('resnet33ts', pretrained=pretrained, **kwargs)


@register_model
def gcresnet33ts(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('gcresnet33ts', pretrained=pretrained, **kwargs)


@register_model
def seresnet33ts(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('seresnet33ts', pretrained=pretrained, **kwargs)


@register_model
def eca_resnet33ts(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('eca_resnet33ts', pretrained=pretrained, **kwargs)


@register_model
def gcresnet50t(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('gcresnet50t', pretrained=pretrained, **kwargs)


@register_model
def gcresnext50ts(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('gcresnext50ts', pretrained=pretrained, **kwargs)


@register_model
def regnetz_b16(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('regnetz_b16', pretrained=pretrained, **kwargs)


@register_model
def regnetz_c16(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('regnetz_c16', pretrained=pretrained, **kwargs)


@register_model
def regnetz_d32(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('regnetz_d32', pretrained=pretrained, **kwargs)


@register_model
def regnetz_d8(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('regnetz_d8', pretrained=pretrained, **kwargs)


@register_model
def regnetz_e8(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('regnetz_e8', pretrained=pretrained, **kwargs)


@register_model
def regnetz_b16_evos(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('regnetz_b16_evos', pretrained=pretrained, **kwargs)


@register_model
def regnetz_c16_evos(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('regnetz_c16_evos', pretrained=pretrained, **kwargs)


@register_model
def regnetz_d8_evos(pretrained=False, **kwargs):
    """
    """
    return _create_byobnet('regnetz_d8_evos', pretrained=pretrained, **kwargs)
