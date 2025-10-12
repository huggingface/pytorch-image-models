"""RegNet X, Y, Z, and more

Paper: `Designing Network Design Spaces` - https://arxiv.org/abs/2003.13678
Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py

Paper: `Fast and Accurate Model Scaling` - https://arxiv.org/abs/2103.06877
Original Impl: None

Based on original PyTorch impl linked above, but re-wrote to use my own blocks (adapted from ResNet here)
and cleaned up with more descriptive variable names.

Weights from original pycls impl have been modified:
* first layer from BGR -> RGB as most PyTorch models are
* removed training specific dict entries from checkpoints and keep model state_dict only
* remap names to match the ones here

Supports weight loading from torchvision and classy-vision (incl VISSL SEER)

A number of custom timm model definitions additions including:
* stochastic depth, gradient checkpointing, layer-decay, configurable dilation
* a pre-activation 'V' variant
* only known RegNet-Z model definitions with pretrained weights

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Type

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import ClassifierHead, AvgPool2dSame, ConvNormAct, SEModule, DropPath, GroupNormAct, calculate_drop_path_rates
from timm.layers import get_act_layer, get_norm_act_layer, create_conv2d, make_divisible
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq, named_apply
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['RegNet', 'RegNetCfg']  # model_registry will add each entrypoint fn to this


@dataclass
class RegNetCfg:
    """RegNet architecture configuration."""
    depth: int = 21
    w0: int = 80
    wa: float = 42.63
    wm: float = 2.66
    group_size: int = 24
    bottle_ratio: float = 1.
    se_ratio: float = 0.
    group_min_ratio: float = 0.
    stem_width: int = 32
    downsample: Optional[str] = 'conv1x1'
    linear_out: bool = False
    preact: bool = False
    num_features: int = 0
    act_layer: Union[str, Callable] = 'relu'
    norm_layer: Union[str, Callable] = 'batchnorm'


def quantize_float(f: float, q: int) -> int:
    """Converts a float to the closest non-zero int divisible by q.

    Args:
        f: Input float value.
        q: Quantization divisor.

    Returns:
        Quantized integer value.
    """
    return int(round(f / q) * q)


def adjust_widths_groups_comp(
        widths: List[int],
        bottle_ratios: List[float],
        groups: List[int],
        min_ratio: float = 0.
) -> Tuple[List[int], List[int]]:
    """Adjusts the compatibility of widths and groups.

    Args:
        widths: List of channel widths.
        bottle_ratios: List of bottleneck ratios.
        groups: List of group sizes.
        min_ratio: Minimum ratio for divisibility.

    Returns:
        Tuple of adjusted widths and groups.
    """
    bottleneck_widths = [int(w * b) for w, b in zip(widths, bottle_ratios)]
    groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_widths)]
    if min_ratio:
        # torchvision uses a different rounding scheme for ensuring bottleneck widths divisible by group widths
        bottleneck_widths = [make_divisible(w_bot, g, min_ratio) for w_bot, g in zip(bottleneck_widths, groups)]
    else:
        bottleneck_widths = [quantize_float(w_bot, g) for w_bot, g in zip(bottleneck_widths, groups)]
    widths = [int(w_bot / b) for w_bot, b in zip(bottleneck_widths, bottle_ratios)]
    return widths, groups


def generate_regnet(
        width_slope: float,
        width_initial: int,
        width_mult: float,
        depth: int,
        group_size: int,
        quant: int = 8
) -> Tuple[List[int], int, List[int]]:
    """Generates per block widths from RegNet parameters.

    Args:
        width_slope: Slope parameter for width progression.
        width_initial: Initial width.
        width_mult: Width multiplier.
        depth: Network depth.
        group_size: Group convolution size.
        quant: Quantization factor.

    Returns:
        Tuple of (widths, num_stages, groups).
    """
    assert width_slope >= 0 and width_initial > 0 and width_mult > 1 and width_initial % quant == 0
    # TODO dWr scaling?
    # depth = int(depth * (scale ** 0.1))
    # width_scale = scale ** 0.4  # dWr scale, exp 0.8 / 2, applied to both group and layer widths
    widths_cont = torch.arange(depth, dtype=torch.float32) * width_slope + width_initial
    width_exps = torch.round(torch.log(widths_cont / width_initial) / math.log(width_mult))
    widths = torch.round((width_initial * torch.pow(width_mult, width_exps)) / quant) * quant
    num_stages, max_stage = len(torch.unique(widths)), int(width_exps.max().item()) + 1
    groups = torch.tensor([group_size for _ in range(num_stages)], dtype=torch.int32)
    return widths.int().tolist(), num_stages, groups.tolist()


def downsample_conv(
        in_chs: int,
        out_chs: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None,
        preact: bool = False,
        device=None,
        dtype=None,
) -> nn.Module:
    """Create convolutional downsampling module.

    Args:
        in_chs: Input channels.
        out_chs: Output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        dilation: Convolution dilation.
        norm_layer: Normalization layer.
        preact: Use pre-activation.

    Returns:
        Downsampling module.
    """
    dd = {'device': device, 'dtype': dtype}
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    dilation = dilation if kernel_size > 1 else 1
    if preact:
        return create_conv2d(
            in_chs,
            out_chs,
            kernel_size,
            stride=stride,
            dilation=dilation,
            **dd,
        )
    else:
        return ConvNormAct(
            in_chs,
            out_chs,
            kernel_size,
            stride=stride,
            dilation=dilation,
            norm_layer=norm_layer,
            apply_act=False,
            **dd,
        )


def downsample_avg(
        in_chs: int,
        out_chs: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None,
        preact: bool = False,
        device=None,
        dtype=None,
) -> nn.Sequential:
    """Create average pool downsampling module.

    AvgPool Downsampling as in 'D' ResNet variants. This is not in RegNet space but I might experiment.

    Args:
        in_chs: Input channels.
        out_chs: Output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        dilation: Convolution dilation.
        norm_layer: Normalization layer.
        preact: Use pre-activation.

    Returns:
        Sequential downsampling module.
    """
    dd = {'device': device, 'dtype': dtype}
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    pool = nn.Identity()
    if stride > 1 or dilation > 1:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
    if preact:
        conv = create_conv2d(in_chs, out_chs, 1, stride=1, **dd)
    else:
        conv = ConvNormAct(in_chs, out_chs, 1, stride=1, norm_layer=norm_layer, apply_act=False, **dd)
    return nn.Sequential(*[pool, conv])


def create_shortcut(
        downsample_type: Optional[str],
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int,
        dilation: Tuple[int, int] = (1, 1),
        norm_layer: Optional[Type[nn.Module]] = None,
        preact: bool = False,
        device=None,
        dtype=None,
) -> Optional[nn.Module]:
    """Create shortcut connection for residual blocks.

    Args:
        downsample_type: Type of downsampling ('avg', 'conv1x1', or None).
        in_chs: Input channels.
        out_chs: Output channels.
        kernel_size: Kernel size for conv downsampling.
        stride: Stride for downsampling.
        dilation: Dilation rates.
        norm_layer: Normalization layer.
        preact: Use pre-activation.

    Returns:
        Shortcut module or None.
    """
    dd = {'device': device, 'dtype': dtype}
    assert downsample_type in ('avg', 'conv1x1', '', None)
    if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
        dargs = dict(stride=stride, dilation=dilation[0], norm_layer=norm_layer, preact=preact, **dd)
        if not downsample_type:
            return None  # no shortcut, no downsample
        elif downsample_type == 'avg':
            return downsample_avg(in_chs, out_chs, **dargs)
        else:
            return downsample_conv(in_chs, out_chs, kernel_size=kernel_size, **dargs)
    else:
        return nn.Identity()  # identity shortcut (no downsample)


class Bottleneck(nn.Module):
    """RegNet Bottleneck block.

    This is almost exactly the same as a ResNet Bottleneck. The main difference is the SE block is moved from
    after conv3 to after conv2. Otherwise, it's just redefining the arguments for groups/bottleneck channels.
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            bottle_ratio: float = 1,
            group_size: int = 1,
            se_ratio: float = 0.25,
            downsample: str = 'conv1x1',
            linear_out: bool = False,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path_rate: float = 0.,
            device=None,
            dtype=None,
    ):
        """Initialize RegNet Bottleneck block.

        Args:
            in_chs: Input channels.
            out_chs: Output channels.
            stride: Convolution stride.
            dilation: Dilation rates for conv2 and shortcut.
            bottle_ratio: Bottleneck ratio (reduction factor).
            group_size: Group convolution size.
            se_ratio: Squeeze-and-excitation ratio.
            downsample: Shortcut downsampling type.
            linear_out: Use linear activation for output.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            drop_block: Drop block layer.
            drop_path_rate: Stochastic depth drop rate.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        act_layer = get_act_layer(act_layer)
        bottleneck_chs = int(round(out_chs * bottle_ratio))
        groups = bottleneck_chs // group_size

        cargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        self.conv1 = ConvNormAct(in_chs, bottleneck_chs, kernel_size=1, **cargs, **dd)
        self.conv2 = ConvNormAct(
            bottleneck_chs,
            bottleneck_chs,
            kernel_size=3,
            stride=stride,
            dilation=dilation[0],
            groups=groups,
            drop_layer=drop_block,
            **cargs,
            **dd,
        )
        if se_ratio:
            se_channels = int(round(in_chs * se_ratio))
            self.se = SEModule(bottleneck_chs, rd_channels=se_channels, act_layer=act_layer, **dd)
        else:
            self.se = nn.Identity()
        self.conv3 = ConvNormAct(bottleneck_chs, out_chs, kernel_size=1, apply_act=False, **cargs, **dd)
        self.act3 = nn.Identity() if linear_out else act_layer()
        self.downsample = create_shortcut(
            downsample,
            in_chs,
            out_chs,
            kernel_size=1,
            stride=stride,
            dilation=dilation,
            norm_layer=norm_layer,
            **dd,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def zero_init_last(self) -> None:
        """Zero-initialize the last batch norm in the block."""
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        if self.downsample is not None:
            # NOTE stuck with downsample as the attr name due to weight compatibility
            # now represents the shortcut, no shortcut if None, and non-downsample shortcut == nn.Identity()
            x = self.drop_path(x) + self.downsample(shortcut)
        x = self.act3(x)
        return x


class PreBottleneck(nn.Module):
    """Pre-activation RegNet Bottleneck block.

    Similar to Bottleneck but with pre-activation normalization.
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            bottle_ratio: float = 1,
            group_size: int = 1,
            se_ratio: float = 0.25,
            downsample: str = 'conv1x1',
            linear_out: bool = False,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path_rate: float = 0.,
            device=None,
            dtype=None,
    ):
        """Initialize pre-activation RegNet Bottleneck block.

        Args:
            in_chs: Input channels.
            out_chs: Output channels.
            stride: Convolution stride.
            dilation: Dilation rates for conv2 and shortcut.
            bottle_ratio: Bottleneck ratio (reduction factor).
            group_size: Group convolution size.
            se_ratio: Squeeze-and-excitation ratio.
            downsample: Shortcut downsampling type.
            linear_out: Use linear activation for output.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            drop_block: Drop block layer.
            drop_path_rate: Stochastic depth drop rate.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        bottleneck_chs = int(round(out_chs * bottle_ratio))
        groups = bottleneck_chs // group_size

        self.norm1 = norm_act_layer(in_chs, **dd)
        self.conv1 = create_conv2d(in_chs, bottleneck_chs, kernel_size=1, **dd)
        self.norm2 = norm_act_layer(bottleneck_chs, **dd)
        self.conv2 = create_conv2d(
            bottleneck_chs,
            bottleneck_chs,
            kernel_size=3,
            stride=stride,
            dilation=dilation[0],
            groups=groups,
            **dd,
        )
        if se_ratio:
            se_channels = int(round(in_chs * se_ratio))
            self.se = SEModule(bottleneck_chs, rd_channels=se_channels, act_layer=act_layer, **dd)
        else:
            self.se = nn.Identity()
        self.norm3 = norm_act_layer(bottleneck_chs, **dd)
        self.conv3 = create_conv2d(bottleneck_chs, out_chs, kernel_size=1, **dd)
        self.downsample = create_shortcut(
            downsample,
            in_chs,
            out_chs,
            kernel_size=1,
            stride=stride,
            dilation=dilation,
            preact=True,
            **dd,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def zero_init_last(self) -> None:
        """Zero-initialize the last batch norm (no-op for pre-activation)."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.norm1(x)
        shortcut = x
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.norm3(x)
        x = self.conv3(x)
        if self.downsample is not None:
            # NOTE stuck with downsample as the attr name due to weight compatibility
            # now represents the shortcut, no shortcut if None, and non-downsample shortcut == nn.Identity()
            x = self.drop_path(x) + self.downsample(shortcut)
        return x


class RegStage(nn.Module):
    """RegNet stage (sequence of blocks with the same output shape).

    A stage consists of multiple bottleneck blocks with the same output dimensions.
    """

    def __init__(
            self,
            depth: int,
            in_chs: int,
            out_chs: int,
            stride: int,
            dilation: int,
            drop_path_rates: Optional[List[float]] = None,
            block_fn: Type[nn.Module] = Bottleneck,
            **block_kwargs,
    ):
        """Initialize RegNet stage.

        Args:
            depth: Number of blocks in stage.
            in_chs: Input channels.
            out_chs: Output channels.
            stride: Stride for first block.
            dilation: Dilation rate.
            drop_path_rates: Drop path rates for each block.
            block_fn: Block class to use.
            **block_kwargs: Additional block arguments.
        """
        super().__init__()
        self.grad_checkpointing = False

        first_dilation = 1 if dilation in (1, 2) else 2
        for i in range(depth):
            block_stride = stride if i == 0 else 1
            block_in_chs = in_chs if i == 0 else out_chs
            block_dilation = (first_dilation, dilation)
            dpr = drop_path_rates[i] if drop_path_rates is not None else 0.
            name = "b{}".format(i + 1)
            self.add_module(
                name,
                block_fn(
                    block_in_chs,
                    out_chs,
                    stride=block_stride,
                    dilation=block_dilation,
                    drop_path_rate=dpr,
                    **block_kwargs,
                )
            )
            first_dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all blocks in the stage.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.children(), x)
        else:
            for block in self.children():
                x = block(x)
        return x


class RegNet(nn.Module):
    """RegNet-X, Y, and Z Models.

    Paper: https://arxiv.org/abs/2003.13678
    Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
    """

    def __init__(
            self,
            cfg: RegNetCfg,
            in_chans: int = 3,
            num_classes: int = 1000,
            output_stride: int = 32,
            global_pool: str = 'avg',
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            zero_init_last: bool = True,
            device=None,
            dtype=None,
            **kwargs,
    ):
        """Initialize RegNet model.

        Args:
            cfg: Model architecture configuration.
            in_chans: Number of input channels.
            num_classes: Number of classifier classes.
            output_stride: Output stride of network, one of (8, 16, 32).
            global_pool: Global pooling type.
            drop_rate: Dropout rate.
            drop_path_rate: Stochastic depth drop-path rate.
            zero_init_last: Zero-init last weight of residual path.
            kwargs: Extra kwargs overlayed onto cfg.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)
        cfg = replace(cfg, **kwargs)  # update cfg with extra passed kwargs

        # Construct the stem
        stem_width = cfg.stem_width
        na_args = dict(act_layer=cfg.act_layer, norm_layer=cfg.norm_layer)
        if cfg.preact:
            self.stem = create_conv2d(in_chans, stem_width, 3, stride=2, **dd)
        else:
            self.stem = ConvNormAct(in_chans, stem_width, 3, stride=2, **na_args, **dd)
        self.feature_info = [dict(num_chs=stem_width, reduction=2, module='stem')]

        # Construct the stages
        prev_width = stem_width
        curr_stride = 2
        per_stage_args, common_args = self._get_stage_args(
            cfg,
            output_stride=output_stride,
            drop_path_rate=drop_path_rate,
        )
        assert len(per_stage_args) == 4
        block_fn = PreBottleneck if cfg.preact else Bottleneck
        for i, stage_args in enumerate(per_stage_args):
            stage_name = "s{}".format(i + 1)
            self.add_module(
                stage_name,
                RegStage(
                    in_chs=prev_width,
                    block_fn=block_fn,
                    **stage_args,
                    **common_args,
                    **dd,
                )
            )
            prev_width = stage_args['out_chs']
            curr_stride *= stage_args['stride']
            self.feature_info += [dict(num_chs=prev_width, reduction=curr_stride, module=stage_name)]

        # Construct the head
        if cfg.num_features:
            self.final_conv = ConvNormAct(prev_width, cfg.num_features, kernel_size=1, **na_args, **dd)
            self.num_features = cfg.num_features
        else:
            final_act = cfg.linear_out or cfg.preact
            self.final_conv = get_act_layer(cfg.act_layer)() if final_act else nn.Identity()
            self.num_features = prev_width
        self.head_hidden_size = self.num_features
        self.head = ClassifierHead(
            in_features=self.num_features,
            num_classes=num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            **dd,
        )

        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    def _get_stage_args(
            self,
            cfg: RegNetCfg,
            default_stride: int = 2,
            output_stride: int = 32,
            drop_path_rate: float = 0.
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate stage arguments from configuration.

        Args:`
            cfg: RegNet configuration.
            default_stride: Default stride for stages.
            output_stride: Target output stride.
            drop_path_rate: Stochastic depth rate.

        Returns:
            Tuple of (per_stage_args, common_args).
        """
        # Generate RegNet ws per block
        widths, num_stages, stage_gs = generate_regnet(cfg.wa, cfg.w0, cfg.wm, cfg.depth, cfg.group_size)

        # Convert to per stage format
        stage_widths, stage_depths = torch.unique(torch.tensor(widths), return_counts=True)
        stage_widths, stage_depths = stage_widths.tolist(), stage_depths.tolist()
        stage_br = [cfg.bottle_ratio for _ in range(num_stages)]
        stage_strides = []
        stage_dilations = []
        net_stride = 2
        dilation = 1
        for _ in range(num_stages):
            if net_stride >= output_stride:
                dilation *= default_stride
                stride = 1
            else:
                stride = default_stride
                net_stride *= stride
            stage_strides.append(stride)
            stage_dilations.append(dilation)
        stage_dpr = calculate_drop_path_rates(drop_path_rate, stage_depths, stagewise=True)
        # Adjust the compatibility of ws and gws
        stage_widths, stage_gs = adjust_widths_groups_comp(
            stage_widths, stage_br, stage_gs, min_ratio=cfg.group_min_ratio)
        arg_names = ['out_chs', 'stride', 'dilation', 'depth', 'bottle_ratio', 'group_size', 'drop_path_rates']
        per_stage_args = [
            dict(zip(arg_names, params)) for params in
            zip(stage_widths, stage_strides, stage_dilations, stage_depths, stage_br, stage_gs, stage_dpr)
        ]
        common_args = dict(
            downsample=cfg.downsample,
            se_ratio=cfg.se_ratio,
            linear_out=cfg.linear_out,
            act_layer=cfg.act_layer,
            norm_layer=cfg.norm_layer,
        )
        return per_stage_args, common_args

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        """Group parameters for optimization."""
        return dict(
            stem=r'^stem',
            blocks=r'^s(\d+)' if coarse else r'^s(\d+)\.b(\d+)',
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing."""
        for s in list(self.children())[1:-1]:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """Get the classifier head."""
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Reset the classifier head.

        Args:
            num_classes: Number of classes for new classifier.
            global_pool: Global pooling type.
        """
        self.num_classes = num_classes
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(5, indices)

        # forward pass
        feat_idx = 0
        x = self.stem(x)
        if feat_idx in take_indices:
            intermediates.append(x)

        layer_names = ('s1', 's2', 's3', 's4')
        if stop_early:
            layer_names = layer_names[:max_index]
        for n in layer_names:
            feat_idx += 1
            x = getattr(self, n)(x)  # won't work with torchscript, but keeps code reasonable, FML
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        if feat_idx == 4:
            x = self.final_conv(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ) -> List[int]:
        """Prune layers not required for specified intermediates.

        Args:
            indices: Indices of intermediate layers to keep.
            prune_norm: Whether to prune normalization layer.
            prune_head: Whether to prune the classifier head.

        Returns:
            List of indices that were kept.
        """
        take_indices, max_index = feature_take_indices(5, indices)
        layer_names = ('s1', 's2', 's3', 's4')
        layer_names = layer_names[max_index:]
        for n in layer_names:
            setattr(self, n, nn.Identity())
        if max_index < 4:
            self.final_conv = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction layers.

        Args:
            x: Input tensor.

        Returns:
            Feature tensor.
        """
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.final_conv(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass through classifier head.

        Args:
            x: Input features.
            pre_logits: Return features before final linear layer.

        Returns:
            Classification logits or features.
        """
        return self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output logits.
        """
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module: nn.Module, name: str = '', zero_init_last: bool = False) -> None:
    """Initialize module weights.

    Args:
        module: PyTorch module to initialize.
        name: Module name.
        zero_init_last: Zero-initialize last layer weights.
    """
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
    elif zero_init_last and hasattr(module, 'zero_init_last'):
        module.zero_init_last()


def _filter_fn(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Filter and remap state dict keys for compatibility.

    Args:
        state_dict: Raw state dictionary.

    Returns:
        Filtered state dictionary.
    """
    state_dict = state_dict.get('model', state_dict)
    replaces = [
        ('f.a.0', 'conv1.conv'),
        ('f.a.1', 'conv1.bn'),
        ('f.b.0', 'conv2.conv'),
        ('f.b.1', 'conv2.bn'),
        ('f.final_bn', 'conv3.bn'),
        ('f.se.excitation.0', 'se.fc1'),
        ('f.se.excitation.2', 'se.fc2'),
        ('f.se', 'se'),
        ('f.c.0', 'conv3.conv'),
        ('f.c.1', 'conv3.bn'),
        ('f.c', 'conv3.conv'),
        ('proj.0', 'downsample.conv'),
        ('proj.1', 'downsample.bn'),
        ('proj', 'downsample.conv'),
    ]
    if 'classy_state_dict' in state_dict:
        # classy-vision & vissl (SEER) weights
        import re
        state_dict = state_dict['classy_state_dict']['base_model']['model']
        out = {}
        for k, v in state_dict['trunk'].items():
            k = k.replace('_feature_blocks.conv1.stem.0', 'stem.conv')
            k = k.replace('_feature_blocks.conv1.stem.1', 'stem.bn')
            k = re.sub(
                r'^_feature_blocks.res\d.block(\d)-(\d+)',
                lambda x: f's{int(x.group(1))}.b{int(x.group(2)) + 1}', k)
            k = re.sub(r's(\d)\.b(\d+)\.bn', r's\1.b\2.downsample.bn', k)
            for s, r in replaces:
                k = k.replace(s, r)
            out[k] = v
        for k, v in state_dict['heads'].items():
            if 'projection_head' in k or 'prototypes' in k:
                continue
            k = k.replace('0.clf.0', 'head.fc')
            out[k] = v
        return out
    if 'stem.0.weight' in state_dict:
        # torchvision weights
        import re
        out = {}
        for k, v in state_dict.items():
            k = k.replace('stem.0', 'stem.conv')
            k = k.replace('stem.1', 'stem.bn')
            k = re.sub(
                r'trunk_output.block(\d)\.block(\d+)\-(\d+)',
                lambda x: f's{int(x.group(1))}.b{int(x.group(3)) + 1}', k)
            for s, r in replaces:
                k = k.replace(s, r)
            k = k.replace('fc.', 'head.fc.')
            out[k] = v
        return out
    return state_dict


# Model FLOPS = three trailing digits * 10^8
model_cfgs = dict(
    # RegNet-X
    regnetx_002=RegNetCfg(w0=24, wa=36.44, wm=2.49, group_size=8, depth=13),
    regnetx_004=RegNetCfg(w0=24, wa=24.48, wm=2.54, group_size=16, depth=22),
    regnetx_004_tv=RegNetCfg(w0=24, wa=24.48, wm=2.54, group_size=16, depth=22, group_min_ratio=0.9),
    regnetx_006=RegNetCfg(w0=48, wa=36.97, wm=2.24, group_size=24, depth=16),
    regnetx_008=RegNetCfg(w0=56, wa=35.73, wm=2.28, group_size=16, depth=16),
    regnetx_016=RegNetCfg(w0=80, wa=34.01, wm=2.25, group_size=24, depth=18),
    regnetx_032=RegNetCfg(w0=88, wa=26.31, wm=2.25, group_size=48, depth=25),
    regnetx_040=RegNetCfg(w0=96, wa=38.65, wm=2.43, group_size=40, depth=23),
    regnetx_064=RegNetCfg(w0=184, wa=60.83, wm=2.07, group_size=56, depth=17),
    regnetx_080=RegNetCfg(w0=80, wa=49.56, wm=2.88, group_size=120, depth=23),
    regnetx_120=RegNetCfg(w0=168, wa=73.36, wm=2.37, group_size=112, depth=19),
    regnetx_160=RegNetCfg(w0=216, wa=55.59, wm=2.1, group_size=128, depth=22),
    regnetx_320=RegNetCfg(w0=320, wa=69.86, wm=2.0, group_size=168, depth=23),

    # RegNet-Y
    regnety_002=RegNetCfg(w0=24, wa=36.44, wm=2.49, group_size=8, depth=13, se_ratio=0.25),
    regnety_004=RegNetCfg(w0=48, wa=27.89, wm=2.09, group_size=8, depth=16, se_ratio=0.25),
    regnety_006=RegNetCfg(w0=48, wa=32.54, wm=2.32, group_size=16, depth=15, se_ratio=0.25),
    regnety_008=RegNetCfg(w0=56, wa=38.84, wm=2.4, group_size=16, depth=14, se_ratio=0.25),
    regnety_008_tv=RegNetCfg(w0=56, wa=38.84, wm=2.4, group_size=16, depth=14, se_ratio=0.25, group_min_ratio=0.9),
    regnety_016=RegNetCfg(w0=48, wa=20.71, wm=2.65, group_size=24, depth=27, se_ratio=0.25),
    regnety_032=RegNetCfg(w0=80, wa=42.63, wm=2.66, group_size=24, depth=21, se_ratio=0.25),
    regnety_040=RegNetCfg(w0=96, wa=31.41, wm=2.24, group_size=64, depth=22, se_ratio=0.25),
    regnety_064=RegNetCfg(w0=112, wa=33.22, wm=2.27, group_size=72, depth=25, se_ratio=0.25),
    regnety_080=RegNetCfg(w0=192, wa=76.82, wm=2.19, group_size=56, depth=17, se_ratio=0.25),
    regnety_080_tv=RegNetCfg(w0=192, wa=76.82, wm=2.19, group_size=56, depth=17, se_ratio=0.25, group_min_ratio=0.9),
    regnety_120=RegNetCfg(w0=168, wa=73.36, wm=2.37, group_size=112, depth=19, se_ratio=0.25),
    regnety_160=RegNetCfg(w0=200, wa=106.23, wm=2.48, group_size=112, depth=18, se_ratio=0.25),
    regnety_320=RegNetCfg(w0=232, wa=115.89, wm=2.53, group_size=232, depth=20, se_ratio=0.25),
    regnety_640=RegNetCfg(w0=352, wa=147.48, wm=2.4, group_size=328, depth=20, se_ratio=0.25),
    regnety_1280=RegNetCfg(w0=456, wa=160.83, wm=2.52, group_size=264, depth=27, se_ratio=0.25),
    regnety_2560=RegNetCfg(w0=640, wa=230.83, wm=2.53, group_size=373, depth=27, se_ratio=0.25),
    #regnety_2560=RegNetCfg(w0=640, wa=124.47, wm=2.04, group_size=848, depth=27, se_ratio=0.25),

    # Experimental
    regnety_040_sgn=RegNetCfg(
        w0=96, wa=31.41, wm=2.24, group_size=64, depth=22, se_ratio=0.25,
        act_layer='silu', norm_layer=partial(GroupNormAct, group_size=16)),

    # regnetv = 'preact regnet y'
    regnetv_040=RegNetCfg(
        depth=22, w0=96, wa=31.41, wm=2.24, group_size=64, se_ratio=0.25, preact=True, act_layer='silu'),
    regnetv_064=RegNetCfg(
        depth=25, w0=112, wa=33.22, wm=2.27, group_size=72, se_ratio=0.25, preact=True, act_layer='silu',
        downsample='avg'),

    # RegNet-Z (unverified)
    regnetz_005=RegNetCfg(
        depth=21, w0=16, wa=10.7, wm=2.51, group_size=4, bottle_ratio=4.0, se_ratio=0.25,
        downsample=None, linear_out=True, num_features=1024, act_layer='silu',
    ),
    regnetz_040=RegNetCfg(
        depth=28, w0=48, wa=14.5, wm=2.226, group_size=8, bottle_ratio=4.0, se_ratio=0.25,
        downsample=None, linear_out=True, num_features=0, act_layer='silu',
    ),
    regnetz_040_h=RegNetCfg(
        depth=28, w0=48, wa=14.5, wm=2.226, group_size=8, bottle_ratio=4.0, se_ratio=0.25,
        downsample=None, linear_out=True, num_features=1536, act_layer='silu',
    ),
)


def _create_regnet(variant: str, pretrained: bool, **kwargs) -> RegNet:
    """Create a RegNet model.

    Args:
        variant: Model variant name.
        pretrained: Load pretrained weights.
        **kwargs: Additional model arguments.

    Returns:
        RegNet model instance.
    """
    return build_model_with_cfg(
        RegNet, variant, pretrained,
        model_cfg=model_cfgs[variant],
        pretrained_filter_fn=_filter_fn,
        **kwargs)


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    """Create default configuration dictionary.

    Args:
        url: Model weight URL.
        **kwargs: Additional configuration options.

    Returns:
        Configuration dictionary.
    """
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'test_input_size': (3, 288, 288), 'crop_pct': 0.95, 'test_crop_pct': 1.0,
        'interpolation': 'bicubic', 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
        'license': 'apache-2.0', **kwargs
    }


def _cfgpyc(url: str = '', **kwargs) -> Dict[str, Any]:
    """Create pycls configuration dictionary.

    Args:
        url: Model weight URL.
        **kwargs: Additional configuration options.

    Returns:
        Configuration dictionary.
    """
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
        'license': 'mit', 'origin_url': 'https://github.com/facebookresearch/pycls', **kwargs
    }


def _cfgtv2(url: str = '', **kwargs) -> Dict[str, Any]:
    """Create torchvision v2 configuration dictionary.

    Args:
        url: Model weight URL.
        **kwargs: Additional configuration options.

    Returns:
        Configuration dictionary.
    """
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.965, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
        'license': 'bsd-3-clause', 'origin_url': 'https://github.com/pytorch/vision', **kwargs
    }


default_cfgs = generate_default_cfgs({
    # timm trained models
    'regnety_032.ra_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/regnety_032_ra-7f2439f9.pth'),
    'regnety_040.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-tpu-weights/regnety_040_ra3-670e1166.pth'),
    'regnety_064.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-tpu-weights/regnety_064_ra3-aa26dc7d.pth'),
    'regnety_080.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-tpu-weights/regnety_080_ra3-1fdc4344.pth'),
    'regnety_120.sw_in12k_ft_in1k': _cfg(hf_hub_id='timm/'),
    'regnety_160.sw_in12k_ft_in1k': _cfg(hf_hub_id='timm/'),
    'regnety_160.lion_in12k_ft_in1k': _cfg(hf_hub_id='timm/'),

    # timm in12k pretrain
    'regnety_120.sw_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821),
    'regnety_160.sw_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821),

    # timm custom arch (v and z guess) + trained models
    'regnety_040_sgn.untrained': _cfg(url=''),
    'regnetv_040.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-tpu-weights/regnetv_040_ra3-c248f51f.pth',
        first_conv='stem'),
    'regnetv_064.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-tpu-weights/regnetv_064_ra3-530616c2.pth',
        first_conv='stem'),

    'regnetz_005.untrained': _cfg(url=''),
    'regnetz_040.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-tpu-weights/regnetz_040_ra3-9007edf5.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 320, 320)),
    'regnetz_040_h.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-tpu-weights/regnetz_040h_ra3-f594343b.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 320, 320)),

    # used in DeiT for distillation (from Facebook DeiT GitHub repository)
    'regnety_160.deit_in1k': _cfg(
        hf_hub_id='timm/', url='https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth'),

    'regnetx_004_tv.tv2_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_x_400mf-62229a5f.pth'),
    'regnetx_008.tv2_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_x_800mf-94a99ebd.pth'),
    'regnetx_016.tv2_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_x_1_6gf-a12f2b72.pth'),
    'regnetx_032.tv2_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_x_3_2gf-7071aa85.pth'),
    'regnetx_080.tv2_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_x_8gf-2b70d774.pth'),
    'regnetx_160.tv2_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_x_16gf-ba3796d7.pth'),
    'regnetx_320.tv2_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_x_32gf-6eb8fdc6.pth'),

    'regnety_004.tv2_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_y_400mf-e6988f5f.pth'),
    'regnety_008_tv.tv2_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_y_800mf-58fc7688.pth'),
    'regnety_016.tv2_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_y_1_6gf-0d7bc02a.pth'),
    'regnety_032.tv2_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_y_3_2gf-9180c971.pth'),
    'regnety_080_tv.tv2_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_y_8gf-dc2b1b54.pth'),
    'regnety_160.tv2_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_y_16gf-3e4a00f9.pth'),
    'regnety_320.tv2_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_y_32gf-8db6d4b5.pth'),

    'regnety_160.swag_ft_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_y_16gf_swag-43afe44d.pth', license='cc-by-nc-4.0',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    'regnety_320.swag_ft_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_y_32gf_swag-04fdfa75.pth', license='cc-by-nc-4.0',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    'regnety_1280.swag_ft_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_y_128gf_swag-c8ce3e52.pth', license='cc-by-nc-4.0',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),

    'regnety_160.swag_lc_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_y_16gf_lc_swag-f3ec0043.pth', license='cc-by-nc-4.0'),
    'regnety_320.swag_lc_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_y_32gf_lc_swag-e1583746.pth', license='cc-by-nc-4.0'),
    'regnety_1280.swag_lc_in1k': _cfgtv2(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/regnet_y_128gf_lc_swag-cbe8ce12.pth', license='cc-by-nc-4.0'),

    'regnety_320.seer_ft_in1k': _cfgtv2(
        hf_hub_id='timm/',
        license='seer-license', origin_url='https://github.com/facebookresearch/vissl',
        url='https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet32_finetuned_in1k_model_final_checkpoint_phase78.torch',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    'regnety_640.seer_ft_in1k': _cfgtv2(
        hf_hub_id='timm/',
        license='seer-license', origin_url='https://github.com/facebookresearch/vissl',
        url='https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet64_finetuned_in1k_model_final_checkpoint_phase78.torch',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    'regnety_1280.seer_ft_in1k': _cfgtv2(
        hf_hub_id='timm/',
        license='seer-license', origin_url='https://github.com/facebookresearch/vissl',
        url='https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet128_finetuned_in1k_model_final_checkpoint_phase78.torch',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    'regnety_2560.seer_ft_in1k': _cfgtv2(
        hf_hub_id='timm/',
        license='seer-license', origin_url='https://github.com/facebookresearch/vissl',
        url='https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet256_finetuned_in1k_model_final_checkpoint_phase38.torch',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),

    'regnety_320.seer': _cfgtv2(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet32d/seer_regnet32gf_model_iteration244000.torch',
        num_classes=0, license='seer-license', origin_url='https://github.com/facebookresearch/vissl'),
    'regnety_640.seer': _cfgtv2(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet64/seer_regnet64gf_model_final_checkpoint_phase0.torch',
        num_classes=0, license='seer-license', origin_url='https://github.com/facebookresearch/vissl'),
    'regnety_1280.seer': _cfgtv2(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_ig1b_regnet128Gf_cnstant_bs32_node16_sinkhorn10_proto16k_syncBN64_warmup8k/model_final_checkpoint_phase0.torch',
        num_classes=0, license='seer-license', origin_url='https://github.com/facebookresearch/vissl'),
    # FIXME invalid weight <-> model match, mistake on their end
    #'regnety_2560.seer': _cfgtv2(
    #    url='https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_ig1b_cosine_rg256gf_noBNhead_wd1e5_fairstore_bs16_node64_sinkhorn10_proto16k_apex_syncBN64_warmup8k/model_final_checkpoint_phase0.torch',
    #    num_classes=0, license='other', origin_url='https://github.com/facebookresearch/vissl'),

    'regnetx_002.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnetx_004.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnetx_006.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnetx_008.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnetx_016.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnetx_032.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnetx_040.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnetx_064.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnetx_080.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnetx_120.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnetx_160.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnetx_320.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),

    'regnety_002.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnety_004.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnety_006.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnety_008.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnety_016.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnety_032.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnety_040.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnety_064.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnety_080.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnety_120.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnety_160.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
    'regnety_320.pycls_in1k': _cfgpyc(hf_hub_id='timm/'),
})


@register_model
def regnetx_002(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetX-200MF"""
    return _create_regnet('regnetx_002', pretrained, **kwargs)


@register_model
def regnetx_004(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetX-400MF"""
    return _create_regnet('regnetx_004', pretrained, **kwargs)


@register_model
def regnetx_004_tv(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetX-400MF w/ torchvision group rounding"""
    return _create_regnet('regnetx_004_tv', pretrained, **kwargs)


@register_model
def regnetx_006(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetX-600MF"""
    return _create_regnet('regnetx_006', pretrained, **kwargs)


@register_model
def regnetx_008(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetX-800MF"""
    return _create_regnet('regnetx_008', pretrained, **kwargs)


@register_model
def regnetx_016(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetX-1.6GF"""
    return _create_regnet('regnetx_016', pretrained, **kwargs)


@register_model
def regnetx_032(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetX-3.2GF"""
    return _create_regnet('regnetx_032', pretrained, **kwargs)


@register_model
def regnetx_040(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetX-4.0GF"""
    return _create_regnet('regnetx_040', pretrained, **kwargs)


@register_model
def regnetx_064(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetX-6.4GF"""
    return _create_regnet('regnetx_064', pretrained, **kwargs)


@register_model
def regnetx_080(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetX-8.0GF"""
    return _create_regnet('regnetx_080', pretrained, **kwargs)


@register_model
def regnetx_120(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetX-12GF"""
    return _create_regnet('regnetx_120', pretrained, **kwargs)


@register_model
def regnetx_160(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetX-16GF"""
    return _create_regnet('regnetx_160', pretrained, **kwargs)


@register_model
def regnetx_320(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetX-32GF"""
    return _create_regnet('regnetx_320', pretrained, **kwargs)


@register_model
def regnety_002(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-200MF"""
    return _create_regnet('regnety_002', pretrained, **kwargs)


@register_model
def regnety_004(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-400MF"""
    return _create_regnet('regnety_004', pretrained, **kwargs)


@register_model
def regnety_006(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-600MF"""
    return _create_regnet('regnety_006', pretrained, **kwargs)


@register_model
def regnety_008(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-800MF"""
    return _create_regnet('regnety_008', pretrained, **kwargs)


@register_model
def regnety_008_tv(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-800MF w/ torchvision group rounding"""
    return _create_regnet('regnety_008_tv', pretrained, **kwargs)


@register_model
def regnety_016(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-1.6GF"""
    return _create_regnet('regnety_016', pretrained, **kwargs)


@register_model
def regnety_032(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-3.2GF"""
    return _create_regnet('regnety_032', pretrained, **kwargs)


@register_model
def regnety_040(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-4.0GF"""
    return _create_regnet('regnety_040', pretrained, **kwargs)


@register_model
def regnety_064(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-6.4GF"""
    return _create_regnet('regnety_064', pretrained, **kwargs)


@register_model
def regnety_080(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-8.0GF"""
    return _create_regnet('regnety_080', pretrained, **kwargs)


@register_model
def regnety_080_tv(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-8.0GF w/ torchvision group rounding"""
    return _create_regnet('regnety_080_tv', pretrained, **kwargs)


@register_model
def regnety_120(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-12GF"""
    return _create_regnet('regnety_120', pretrained, **kwargs)


@register_model
def regnety_160(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-16GF"""
    return _create_regnet('regnety_160', pretrained, **kwargs)


@register_model
def regnety_320(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-32GF"""
    return _create_regnet('regnety_320', pretrained, **kwargs)


@register_model
def regnety_640(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-64GF"""
    return _create_regnet('regnety_640', pretrained, **kwargs)


@register_model
def regnety_1280(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-128GF"""
    return _create_regnet('regnety_1280', pretrained, **kwargs)


@register_model
def regnety_2560(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-256GF"""
    return _create_regnet('regnety_2560', pretrained, **kwargs)


@register_model
def regnety_040_sgn(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetY-4.0GF w/ GroupNorm """
    return _create_regnet('regnety_040_sgn', pretrained, **kwargs)


@register_model
def regnetv_040(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetV-4.0GF (pre-activation)"""
    return _create_regnet('regnetv_040', pretrained, **kwargs)


@register_model
def regnetv_064(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetV-6.4GF (pre-activation)"""
    return _create_regnet('regnetv_064', pretrained, **kwargs)


@register_model
def regnetz_005(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetZ-500MF
    NOTE: config found in https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/models/regnet.py
    but it's not clear it is equivalent to paper model as not detailed in the paper.
    """
    return _create_regnet('regnetz_005', pretrained, zero_init_last=False, **kwargs)


@register_model
def regnetz_040(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetZ-4.0GF
    NOTE: config found in https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/models/regnet.py
    but it's not clear it is equivalent to paper model as not detailed in the paper.
    """
    return _create_regnet('regnetz_040', pretrained, zero_init_last=False, **kwargs)


@register_model
def regnetz_040_h(pretrained: bool = False, **kwargs) -> RegNet:
    """RegNetZ-4.0GF
    NOTE: config found in https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/models/regnet.py
    but it's not clear it is equivalent to paper model as not detailed in the paper.
    """
    return _create_regnet('regnetz_040_h', pretrained, zero_init_last=False, **kwargs)


register_model_deprecations(__name__, {
    'regnetz_040h': 'regnetz_040_h',
})
