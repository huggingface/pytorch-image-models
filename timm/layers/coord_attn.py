""" Coordinate Attention and Variants

Coordinate Attention decomposes channel attention into two 1D feature encoding processes
to capture long-range dependencies with precise positional information. This module includes
the original implementation along with simplified and other variants.

Papers / References:
- Coordinate Attention: `Coordinate Attention for Efficient Mobile Network Design` - https://arxiv.org/abs/2103.02907
- Efficient Local Attention: `Rethinking Local Perception in Lightweight Vision Transformer` - https://arxiv.org/abs/2403.01123

Hacked together by / Copyright 2025 Ross Wightman
"""
from typing import Optional, Type, Union

import torch
from torch import nn

from .create_act import create_act_layer
from .helpers import make_divisible
from .norm import GroupNorm1


class CoordAttn(nn.Module):
    def __init__(
            self,
            channels: int,
            rd_ratio: float = 1. / 16,
            rd_channels: Optional[int] = None,
            rd_divisor: int = 8,
            se_factor: float = 2/3,
            bias: bool = False,
            act_layer: Type[nn.Module] = nn.Hardswish,
            norm_layer: Optional[Type[nn.Module]] = nn.BatchNorm2d,
            gate_layer: Union[str, Type[nn.Module]] = 'sigmoid',
            has_skip: bool = False,
            device=None,
            dtype=None,
    ):
        """Coordinate Attention module for spatial feature recalibration.

        Introduced in "Coordinate Attention for Efficient Mobile Network Design" (CVPR 2021).
        Decomposes channel attention into two 1D feature encoding processes along the height and
        width axes to capture long-range dependencies with precise positional information.

        Args:
            channels: Number of input channels.
            rd_ratio: Reduction ratio for bottleneck channel calculation.
            rd_channels: Explicit number of bottleneck channels, overrides rd_ratio if set.
            rd_divisor: Divisor for making bottleneck channels divisible.
            se_factor: Applied to rd_ratio for final channel count (keeps params similar to SE).
            bias: Whether to use bias in convolution layers.
            act_layer: Activation module class for bottleneck.
            norm_layer: Normalization module class, None for no normalization.
            gate_layer: Gate activation, either 'sigmoid', 'hardsigmoid', or a module class.
            has_skip: Whether to add residual skip connection to output.
            device: Device to place tensors on.
            dtype: Data type for tensors.
        """

        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.has_skip = has_skip
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio * se_factor, rd_divisor, round_limit=0.)

        self.conv1 = nn.Conv2d(channels, rd_channels, kernel_size=1, stride=1, padding=0, bias=bias, **dd)
        self.bn1 = norm_layer(rd_channels, **dd) if norm_layer is not None else nn.Identity()
        self.act = act_layer()

        self.conv_h = nn.Conv2d(rd_channels, channels, kernel_size=1, stride=1, padding=0, bias=bias, **dd)
        self.conv_w = nn.Conv2d(rd_channels, channels, kernel_size=1, stride=1, padding=0, bias=bias, **dd)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        identity = x

        N, C, H, W = x.size()

        # Strip pooling
        x_h = x.mean(3, keepdim=True)
        x_w = x.mean(2, keepdim=True)

        x_w = x_w.transpose(-1, -2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.transpose(-1, -2)

        a_h = self.gate(self.conv_h(x_h))
        a_w = self.gate(self.conv_w(x_w))

        out = identity * a_w * a_h
        if self.has_skip:
            out = out + identity

        return out


class SimpleCoordAttn(nn.Module):
    """Simplified Coordinate Attention variant.

    Uses
     * linear layers instead of convolutions
     * no norm
     * additive pre-gating re-combination
    for reduced complexity while maintaining the core coordinate attention mechanism
    of separate height and width attention.
    """

    def __init__(
            self,
            channels: int,
            rd_ratio: float = 0.25,
            rd_channels: Optional[int] = None,
            rd_divisor: int = 8,
            se_factor: float = 2 / 3,
            bias: bool = True,
            act_layer: Type[nn.Module] = nn.SiLU,
            gate_layer: Union[str, Type[nn.Module]] = 'sigmoid',
            has_skip: bool = False,
            device=None,
            dtype=None,
    ):
        """
        Args:
            channels: Number of input channels.
            rd_ratio: Reduction ratio for bottleneck channel calculation.
            rd_channels: Explicit number of bottleneck channels, overrides rd_ratio if set.
            rd_divisor: Divisor for making bottleneck channels divisible.
            se_factor: Applied to rd_ratio for final channel count (keeps param similar to SE)
            bias: Whether to use bias in linear layers.
            act_layer: Activation module class for bottleneck.
            gate_layer: Gate activation, either 'sigmoid', 'hardsigmoid', or a module class.
            has_skip: Whether to add residual skip connection to output.
            device: Device to place tensors on.
            dtype: Data type for tensors.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.has_skip = has_skip

        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio * se_factor, rd_divisor, round_limit=0.)

        self.fc1 = nn.Linear(channels, rd_channels, bias=bias, **dd)
        self.act = act_layer()
        self.fc_h = nn.Linear(rd_channels, channels, bias=bias, **dd)
        self.fc_w = nn.Linear(rd_channels, channels, bias=bias, **dd)

        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        identity = x

        # Strip pooling
        x_h = x.mean(dim=3)  # (N, C, H)
        x_w = x.mean(dim=2)  # (N, C, W)

        # Shared bottleneck projection
        x_h = self.act(self.fc1(x_h.transpose(1, 2)))  # (N, H, rd_c)
        x_w = self.act(self.fc1(x_w.transpose(1, 2)))  # (N, W, rd_c)

        # Separate attention heads
        a_h = self.fc_h(x_h).transpose(1, 2).unsqueeze(-1)  # (N, C, H, 1)
        a_w = self.fc_w(x_w).transpose(1, 2).unsqueeze(-2)  # (N, C, 1, W)

        out = identity * self.gate(a_h + a_w)
        if self.has_skip:
            out = out + identity

        return out


class EfficientLocalAttn(nn.Module):
    """Efficient Local Attention.

    Lightweight alternative to Coordinate Attention that preserves spatial
    information without channel reduction. Uses 1D depthwise convolutions
    and GroupNorm for better generalization.

    Paper: https://arxiv.org/abs/2403.01123
    """

    def __init__(
            self,
            channels: int,
            kernel_size: int = 7,
            bias: bool = False,
            act_layer: Type[nn.Module] = nn.SiLU,
            gate_layer: Union[str, Type[nn.Module]] = 'sigmoid',
            norm_layer: Optional[Type[nn.Module]] = GroupNorm1,
            has_skip: bool = False,
            device=None,
            dtype=None,
    ):
        """
        Args:
            channels: Number of input channels.
            kernel_size: Kernel size for 1D depthwise convolutions.
            bias: Whether to use bias in convolution layers.
            act_layer: Activation module class applied after normalization.
            gate_layer: Gate activation, either 'sigmoid', 'hardsigmoid', or a module class.
            norm_layer: Normalization module class, None for no normalization.
            has_skip: Whether to add residual skip connection to output.
            device: Device to place tensors on.
            dtype: Data type for tensors.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.has_skip = has_skip

        self.conv_h = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=(kernel_size // 2, 0),
            groups=channels,
            bias=bias,
            **dd
        )
        self.conv_w = nn.Conv2d(
            channels, channels,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=(0, kernel_size // 2),
            groups=channels,
            bias=bias,
            **dd
        )
        if norm_layer is not None:
            self.norm_h = norm_layer(channels, **dd)
            self.norm_w = norm_layer(channels, **dd)
        else:
            self.norm_h = nn.Identity()
            self.norm_w = nn.Identity()
        self.act = act_layer()
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        identity = x

        # Strip pooling: (N, C, H, W) -> (N, C, H) and (N, C, W)
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True)

        # 1D conv + norm + act
        x_h = self.act(self.norm_h(self.conv_h(x_h)))  # (N, C, H, 1)
        x_w = self.act(self.norm_w(self.conv_w(x_w)))  # (N, C, 1, W)

        # Generate attention maps
        a_h = self.gate(x_h)  # (N, C, H, 1)
        a_w = self.gate(x_w)  # (N, C, 1, W)

        out = identity * a_h * a_w
        if self.has_skip:
            out = out + identity

        return out


class StripAttn(nn.Module):
    """Minimal Strip Attention.

    Lightweight spatial attention using strip pooling with optional learned refinement.
    """

    def __init__(
            self,
            channels: int,
            use_conv: bool = True,
            kernel_size: int = 3,
            bias: bool = False,
            gate_layer: Union[str, Type[nn.Module]] = 'sigmoid',
            has_skip: bool = False,
            device=None,
            dtype=None,
            **_,
    ):
        """
        Args:
            channels: Number of input channels.
            use_conv: Whether to apply depthwise convolutions for learned spatial refinement.
            kernel_size: Kernel size for 1D depthwise convolutions when use_conv is True.
            bias: Whether to use bias in convolution layers.
            gate_layer: Gate activation, either 'sigmoid', 'hardsigmoid', or a module class.
            has_skip: Whether to add residual skip connection to output.
            device: Device to place tensors on.
            dtype: Data type for tensors.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.has_skip = has_skip
        self.use_conv = use_conv

        if use_conv:
            self.conv_h = nn.Conv2d(
                channels, channels,
                kernel_size=(kernel_size, 1),
                stride=1,
                padding=(kernel_size // 2, 0),
                groups=channels,
                bias=bias,
                **dd
            )
            self.conv_w = nn.Conv2d(
                channels, channels,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
                groups=channels,
                bias=bias,
                **dd
            )
        else:
            self.conv_h = nn.Identity()
            self.conv_w = nn.Identity()

        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        identity = x

        # Strip pooling
        x_h = x.mean(dim=3, keepdim=True)  # (N, C, H, 1)
        x_w = x.mean(dim=2, keepdim=True)  # (N, C, 1, W)

        # Optional learned refinement
        x_h = self.conv_h(x_h)
        x_w = self.conv_w(x_w)

        # Combine and gate
        a_hw = self.gate(x_h + x_w)  # broadcasts to (N, C, H, W)

        out = identity * a_hw
        if self.has_skip:
            out = out + identity

        return out

