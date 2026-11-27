"""PyTorch 3D ResNet -- volumetric ResNet backbones for timm

A first official set of 3D ResNet backbones in timm style, for volumetric
inputs of shape [B, C, D, H, W] (medical volumes, CT/MRI/OCT, and video
clips treated as volumes). The models follow the 2D ``ResNet`` class in
``timm/models/resnet.py`` as closely as possible: the same ``create_model``
entry points, feature / head API (``forward_features``, ``forward_head``,
``forward_intermediates``, ``reset_classifier``), and the same stage layout
(stem, four stages, global pool, linear classifier).

The one deliberate difference from a naive 3D port is the anisotropic
treatment of the depth axis. Real medical volumes are heavily anisotropic:
in-plane resolution (H, W) is much finer than the slice thickness (D), so
downsampling depth at the same rate as the in-plane axes throws away the
coarsest axis first and is usually the wrong trade. By default
(``aniso=True``) the stem, the stem pooling, and every stage boundary
downsample the in-plane axes only and preserve the depth resolution, i.e.
the stem uses a (3, 7, 7) kernel with stride (1, 2, 2) instead of the
isotropic (7, 7, 7) / (2, 2, 2). Pass ``aniso=False`` for isotropic
volumes (or video, where the time axis is closer in scale to space).

Example:
    >>> import timm
    >>> model = timm.create_model('resnet18_3d', pretrained=False, in_chans=1, num_classes=2)
    >>> import torch
    >>> out = model(torch.randn(1, 1, 32, 64, 64))  # [B, C, D, H, W]
    >>> out.shape
    torch.Size([1, 2])

Papers:
* `Deep Residual Learning for Image Recognition` - https://arxiv.org/abs/1512.03385
* `Bag of Tricks for Image Classification with Convolutional Neural Networks` - https://arxiv.org/abs/1812.01187

This implementation was written with AI assistance, disclosed per
CONTRIBUTING.md. A human submitter has read the diff.
"""
import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, get_act_layer, get_norm_layer, to_ntuple, get_device_dtype
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['ResNet3d', 'BasicBlock3d', 'Bottleneck3d']


def _get_padding3d(kernel_size: Tuple[int, int, int], stride: Tuple[int, int, int], dilation: Tuple[int, int, int] = (1, 1, 1)) -> Tuple[int, int, int]:
    """Per-axis 'same' padding for a 3D convolution."""
    return tuple(((s - 1) + d * (k - 1)) // 2 for k, s, d in zip(kernel_size, stride, dilation))


def _ntuple3(x: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Normalize an int or 3-tuple to a (depth, height, width) tuple."""
    return to_ntuple(3)(x)


def adaptive_avgmax_pool3d(x, output_size=1):
    x_avg = F.adaptive_avg_pool3d(x, output_size)
    x_max = F.adaptive_max_pool3d(x, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_catavgmax_pool3d(x, output_size=1):
    x_avg = F.adaptive_avg_pool3d(x, output_size)
    x_max = F.adaptive_max_pool3d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


class AdaptiveAvgMaxPool3d(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool3d(x, self.output_size)


class AdaptiveCatAvgMaxPool3d(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool3d(x, self.output_size)


class SelectAdaptivePool3d(nn.Module):
    """Selectable global pooling for 3D feature maps, mirroring SelectAdaptivePool2d."""

    def __init__(
            self,
            pool_type: str = 'avg',
            flatten: bool = True,
    ) -> None:
        """
        Args:
            pool_type: Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax', '' (no pooling).
            flatten: Flatten the pooled output to (B, C) when pooling.
        """
        super().__init__()
        assert pool_type in ('avg', 'max', 'avgmax', 'catavgmax', ''), f'Unknown pool type {pool_type}'
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.flatten = flatten
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool3d(1)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool3d(1)
        elif pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool3d(1)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool3d(1)
        else:
            self.pool = nn.Identity()

    def feat_mult(self) -> int:
        """Feature multiplier for the pooled output (2 when avg and max are concatenated)."""
        return 2 if self.pool_type == 'catavgmax' else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.pool(x)
        if self.flatten:
            x = x.flatten(1)
        return x


def _create_head3d(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        device=None,
        dtype=None,
) -> Tuple[nn.Module, nn.Module]:
    """Create the 3D classifier head (global pool + linear classifier)."""
    global_pool = SelectAdaptivePool3d(pool_type=pool_type, flatten=pool_type != '')
    if num_classes <= 0:
        fc = nn.Identity()
    else:
        fc = nn.Linear(num_features * global_pool.feat_mult(), num_classes, bias=True, device=device, dtype=dtype)
    return global_pool, fc


class BasicBlock3d(nn.Module):
    """Basic residual block for 3D ResNet.

    This is the standard residual block used in ResNet3D-18 and ResNet3D-34.
    All convolutions are 3D; strides and dilations are (depth, height, width)
    tuples so the depth axis can be treated differently from the in-plane axes.
    """
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: Union[int, Tuple[int, int, int]] = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: Union[int, Tuple[int, int, int]] = 1,
            first_dilation: Optional[Union[int, Tuple[int, int, int]]] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm3d,
            drop_path: Optional[nn.Module] = None,
            device=None,
            dtype=None,
    ) -> None:
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers, as (depth, height, width).
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers, as (depth, height, width).
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer class.
            norm_layer: Normalization layer class.
            drop_path: Optional DropPath layer instance.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()

        assert cardinality == 1, 'BasicBlock3d only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock3d does not support changing base width'
        stride = _ntuple3(stride)
        dilation = _ntuple3(dilation)
        first_dilation = _ntuple3(first_dilation) if first_dilation is not None else dilation
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion

        self.conv1 = nn.Conv3d(
            inplanes,
            first_planes,
            kernel_size=3,
            stride=stride,
            padding=first_dilation,
            dilation=first_dilation,
            groups=cardinality,
            bias=False,
            **dd,
        )
        self.bn1 = norm_layer(first_planes, **dd)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv3d(
            first_planes,
            outplanes,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
            **dd,
        )
        self.bn2 = norm_layer(outplanes, **dd)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self) -> None:
        """Initialize the last batch norm layer weights to zero for better convergence."""
        if getattr(self.bn2, 'weight', None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck3d(nn.Module):
    """Bottleneck residual block for 3D ResNet.

    This is the bottleneck block used in ResNet3D-50, ResNet3D-101, and ResNet3D-152.
    """
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: Union[int, Tuple[int, int, int]] = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: Union[int, Tuple[int, int, int]] = 1,
            first_dilation: Optional[Union[int, Tuple[int, int, int]]] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm3d,
            drop_path: Optional[nn.Module] = None,
            device=None,
            dtype=None,
    ) -> None:
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers, as (depth, height, width).
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups for 3x3x3 conv in Bottleneck.
            base_width: Bottleneck channels factor. `planes * base_width / 64 * cardinality`.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers, as (depth, height, width).
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer class.
            norm_layer: Normalization layer class.
            drop_path: Optional DropPath layer instance.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        stride = _ntuple3(stride)
        dilation = _ntuple3(dilation)
        first_dilation = _ntuple3(first_dilation) if first_dilation is not None else dilation
        first_planes = width // reduce_first
        outplanes = planes * self.expansion

        self.conv1 = nn.Conv3d(inplanes, first_planes, kernel_size=1, bias=False, **dd)
        self.bn1 = norm_layer(first_planes, **dd)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv3d(
            first_planes,
            width,
            kernel_size=3,
            stride=stride,
            padding=first_dilation,
            dilation=first_dilation,
            groups=cardinality,
            bias=False,
            **dd,
        )
        self.bn2 = norm_layer(width, **dd)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv3d(width, outplanes, kernel_size=1, bias=False, **dd)
        self.bn3 = norm_layer(outplanes, **dd)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self) -> None:
        """Initialize the last batch norm layer weights to zero for better convergence."""
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x

def downsample_conv3d(
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 1,
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        first_dilation: Optional[Union[int, Tuple[int, int, int]]] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
        device=None,
        dtype=None,
) -> nn.Module:
    """Convolutional downsample for the residual path (3D)."""
    dd = {'device': device, 'dtype': dtype}
    norm_layer = norm_layer or nn.BatchNorm3d
    kernel_size = _ntuple3(kernel_size)
    stride = _ntuple3(stride)
    dilation = _ntuple3(dilation)
    kernel_size = tuple(1 if s == 1 and d == 1 else k for k, s, d in zip(kernel_size, stride, dilation))
    first_dilation = _ntuple3(first_dilation) if first_dilation is not None else dilation
    first_dilation = tuple(fd if k > 1 else 1 for k, fd in zip(kernel_size, first_dilation))
    padding = _get_padding3d(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=first_dilation,
            bias=False,
            **dd
        ),
        norm_layer(out_channels, **dd)
    ])


def downsample_avg3d(
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 1,
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        first_dilation: Optional[Union[int, Tuple[int, int, int]]] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
        device=None,
        dtype=None,
) -> nn.Module:
    """Average-pool downsample for the residual path (3D)."""
    dd = {'device': device, 'dtype': dtype}
    norm_layer = norm_layer or nn.BatchNorm3d
    stride = _ntuple3(stride)
    dilation = _ntuple3(dilation)
    avg_stride = tuple(s if d == 1 else 1 for s, d in zip(stride, dilation))
    if stride == (1, 1, 1) and dilation == (1, 1, 1):
        pool = nn.Identity()
    else:
        # pool kernel matches the stride per axis: a 2-kernel where the axis is
        # downsampled, 1 where it is not, so anisotropic strides keep depth intact
        pool_kernel = tuple(2 if s > 1 else 1 for s in avg_stride)
        pool = nn.AvgPool3d(pool_kernel, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv3d(in_channels, out_channels, 1, stride=1, padding=0, bias=False, **dd),
        norm_layer(out_channels, **dd)
    ])


def make_blocks3d(
        block_fns: Tuple[Union[Type[BasicBlock3d], Type[Bottleneck3d]], ...],
        channels: Tuple[int, ...],
        block_repeats: Tuple[int, ...],
        inplanes: int,
        reduce_first: int = 1,
        output_stride: int = 32,
        down_kernel_size: int = 1,
        avg_down: bool = False,
        drop_path_rate: float = 0.,
        aniso: bool = True,
        device=None,
        dtype=None,
        **kwargs,
) -> Tuple[List[Tuple[str, nn.Module]], List[Dict[str, Any]]]:
    """Create 3D ResNet stages with specified block configurations.

    Strides and dilations are (depth, height, width) tuples. With ``aniso=True``
    (the default) only the in-plane axes are downsampled at stage boundaries,
    so the depth resolution of the volume is preserved through the network;
    with ``aniso=False`` the depth axis is downsampled like the in-plane axes.
    ``output_stride`` applies to the in-plane axes only.

    Args:
        block_fns: Block class to use for each stage.
        channels: Number of channels for each stage.
        block_repeats: Number of blocks to repeat for each stage.
        inplanes: Number of input channels.
        reduce_first: Reduction factor for first convolution in each stage.
        output_stride: Target in-plane output stride of network, 32, 16, or 8.
        down_kernel_size: Kernel size for downsample layers.
        avg_down: Use average pooling for downsample.
        drop_path_rate: Stochastic depth drop rate.
        aniso: Preserve depth resolution (True) or downsample it like the in-plane axes (False).

    Returns:
        Tuple of stage modules list and feature info list.
    """
    dd = {'device': device, 'dtype': dtype}
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4  # in-plane stride after the stem conv (2) and stem maxpool (2)
    dilation = prev_dilation = (1, 1, 1)
    for stage_idx, (block_fn, planes, num_blocks) in enumerate(zip(block_fns, channels, block_repeats)):
        stage_name = f'layer{stage_idx + 1}'
        # stage 0 keeps the resolution, later stages downsample in-plane by 2
        stride = (1, 1, 1) if stage_idx == 0 else (1, 2, 2) if aniso else (2, 2, 2)
        if net_stride >= output_stride:
            # in-plane dilation grows instead of striding, depth dilation stays 1
            dilation = (1, dilation[1] * stride[1], dilation[2] * stride[2])
            stride = (1, 1, 1)
        else:
            net_stride *= stride[1]

        downsample = None
        if stride != (1, 1, 1) or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get('norm_layer'),
                **dd,
            )
            downsample = downsample_avg3d(**down_kwargs) if avg_down else downsample_conv3d(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else (1, 1, 1)
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1) if net_num_blocks > 1 else 0.
            blocks.append(block_fn(
                inplanes,
                planes,
                stride,
                downsample,
                first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None,
                **block_kwargs,
                **dd,
            ))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class ResNet3d(nn.Module):
    """3D ResNet / ResNeXt

    This class implements the 3D ResNet variants for volumetric inputs of shape
    [B, C, D, H, W]. It mirrors the 2D ``ResNet`` class in ``timm/models/resnet.py``:
    the same stem options (single 7x7x7-style conv, or a deep 3-layer stem), the
    same four stages, feature info, and classifier head API.

    The anisotropic handling is the one deliberate difference from a naive 3D
    port. Medical volumes are usually anisotropic (in-plane resolution much
    finer than slice thickness), so with ``aniso=True`` (the default) the stem,
    the stem pooling, and the stage boundaries downsample the in-plane axes
    only and keep the depth resolution intact: the stem is a (3, 7, 7) kernel
    with stride (1, 2, 2) instead of (7, 7, 7) / (2, 2, 2), and stage strides
    are (1, 2, 2) instead of (2, 2, 2). The ``feature_info`` reductions are
    in-plane reductions; the depth reduction is 1 in anisotropic mode.
    Pass ``aniso=False`` for isotropic volumes (or video, where the time axis
    is closer in scale to the spatial axes).

    Stem variants (``stem_type``):
      * '', default - a single (3, 7, 7) conv with a width of stem_width
      * 'deep' - three (1, 3, 3) convolution layers of widths stem_width, stem_width, stem_width * 2
      * 'deep_tiered' - three (1, 3, 3) conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
    """

    def __init__(
            self,
            block: Union[BasicBlock3d, Bottleneck3d],
            layers: Tuple[int, ...],
            num_classes: int = 1000,
            in_chans: int = 1,
            output_stride: int = 32,
            global_pool: str = 'avg',
            cardinality: int = 1,
            base_width: int = 64,
            stem_width: int = 64,
            stem_type: str = '',
            replace_stem_pool: bool = False,
            block_reduce_first: int = 1,
            down_kernel_size: int = 1,
            avg_down: bool = False,
            channels: Optional[Tuple[int, ...]] = (64, 128, 256, 512),
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm3d,
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.,
            zero_init_last: bool = True,
            block_args: Optional[Dict[str, Any]] = None,
            aniso: bool = True,
            device=None,
            dtype=None,
    ):
        """
        Args:
            block: Block class for the residual blocks. Options are BasicBlock3d, Bottleneck3d.
            layers: Number of blocks in each of the four stages.
            num_classes: Number of classification classes (default 1000).
            in_chans: Number of input channels (default 1, single-channel volumes).
            output_stride: In-plane output stride of the network, 32, 16, or 8 (default 32).
            global_pool: Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg').
            cardinality: Number of convolution groups for 3x3x3 conv in Bottleneck (default 1).
            base_width: Bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64).
            stem_width: Number of channels in stem convolutions (default 64).
            stem_type: The type of stem (default ''):
                * '', default - a single (3, 7, 7) conv with a width of stem_width
                * 'deep' - three (1, 3, 3) convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three (1, 3, 3) conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool: Replace stem max-pooling layer with a (1, 3, 3) stride-2 convolution.
            block_reduce_first: Reduction factor for first convolution output width of residual blocks (default 1).
            down_kernel_size: Kernel size of residual block downsample path (default 1).
            avg_down: Use avg pooling for projection skip connection between stages/downsample (default False).
            act_layer: Activation layer class.
            norm_layer: Normalization layer class.
            drop_rate: Dropout probability before classifier, for training (default 0.).
            drop_path_rate: Stochastic depth drop-path rate (default 0.).
            zero_init_last: Zero-initialize the last weight in residual path (usually last BN affine weight).
            block_args: Extra kwargs to pass through to block module.
            aniso: Preserve depth resolution (True, default) or downsample depth like the in-plane axes (False).
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.output_fmt = 'NCDHW'

        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # Anisotropic geometry: (depth, height, width) triples. In anisotropic
        # mode the depth axis is never strided; in isotropic mode it is strided
        # exactly like the in-plane axes.
        stem_kernel = (3, 7, 7) if aniso else (7, 7, 7)
        stem_stride = (1, 2, 2) if aniso else (2, 2, 2)
        stem_padding = (1, 3, 3) if aniso else (3, 3, 3)
        small_kernel = (1, 3, 3) if aniso else (3, 3, 3)
        small_stride = (1, 2, 2) if aniso else (2, 2, 2)
        small_padding = (0, 1, 1) if aniso else (1, 1, 1)

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv3d(in_chans, stem_chs[0], small_kernel, stride=small_stride, padding=small_padding, bias=False, **dd),
                norm_layer(stem_chs[0], **dd),
                act_layer(inplace=True),
                nn.Conv3d(stem_chs[0], stem_chs[1], small_kernel, stride=1, padding=small_padding, bias=False, **dd),
                norm_layer(stem_chs[1], **dd),
                act_layer(inplace=True),
                nn.Conv3d(stem_chs[1], inplanes, small_kernel, stride=1, padding=small_padding, bias=False, **dd)])
        else:
            self.conv1 = nn.Conv3d(in_chans, inplanes, kernel_size=stem_kernel, stride=stem_stride, padding=stem_padding, bias=False, **dd)
        self.bn1 = norm_layer(inplanes, **dd)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*[
                nn.Conv3d(inplanes, inplanes, small_kernel, stride=small_stride, padding=small_padding, bias=False, **dd),
                norm_layer(inplanes, **dd),
                act_layer(inplace=True),
            ])
        else:
            self.maxpool = nn.MaxPool3d(kernel_size=small_kernel, stride=small_stride, padding=small_padding)

        # Feature Blocks
        block_fns = to_ntuple(len(channels))(block)
        stage_modules, stage_feature_info = make_blocks3d(
            block_fns,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            drop_path_rate=drop_path_rate,
            aniso=aniso,
            act_layer=act_layer,
            norm_layer=norm_layer,
            **block_args,
            **dd,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = channels[-1] * block_fns[-1].expansion
        self.global_pool, self.fc = _create_head3d(self.num_features, self.num_classes, pool_type=global_pool, **dd)
        self.head_hidden_size = self.num_features * self.global_pool.feat_mult()

        self.init_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
    def init_weights(self, zero_init_last: bool = True) -> None:
        """Initialize model weights.

        Args:
            zero_init_last: Zero-initialize the last BN in each residual branch.
        """
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, str]:
        """Create regex patterns for parameter grouping.

        Args:
            coarse: Use coarse (stage-level) or fine (block-level) grouping.

        Returns:
            Dictionary mapping group names to regex patterns.
        """
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\\d+)' if coarse else r'^layer(\\d+)\\.(\\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing.

        Args:
            enable: Whether to enable gradient checkpointing.
        """
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only: bool = False) -> Union[str, nn.Module]:
        """Get the classifier module.

        Args:
            name_only: Return classifier module name instead of module.

        Returns:
            Classifier module or name.
        """
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Reset the classifier head.

        Args:
            num_classes: Number of classes for new classifier.
            global_pool: Global pooling type, current pooling is kept if None.
        """
        dd = get_device_dtype(self)
        self.num_classes = num_classes
        if global_pool is None:
            global_pool = self.global_pool.pool_type
        self.global_pool, self.fc = _create_head3d(self.num_features, self.num_classes, pool_type=global_pool, **dd)
        self.head_hidden_size = self.num_features * self.global_pool.feat_mult()
        self.global_pool.train(self.training)
        self.fc.train(self.training)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCDHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            x: Input volume tensor.
            indices: Take last n blocks if int, all if None, select matching indices if sequence.
            norm: Apply norm layer to compatible intermediates.
            stop_early: Stop iterating over blocks when last desired intermediate hit.
            output_fmt: Shape of intermediate feature outputs.
            intermediates_only: Only return intermediate features.

        Returns:
            Features and list of intermediate features or just intermediate features.
        """
        assert output_fmt in ('NCDHW',), 'Output shape must be NCDHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(5, indices)

        # forward pass
        feat_idx = 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if feat_idx in take_indices:
            intermediates.append(x)
        x = self.maxpool(x)

        layer_names = ('layer1', 'layer2', 'layer3', 'layer4')
        if stop_early:
            layer_names = layer_names[:max_index]
        for n in layer_names:
            feat_idx += 1
            layer = getattr(self, n)  # won't work with torchscript, but keeps code reasonable, FML
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(layer, x)  # per block, as forward_features
            else:
                x = layer(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

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
            prune_norm: Whether to prune normalization layers.
            prune_head: Whether to prune the classifier head.

        Returns:
            List of indices that were kept.
        """
        take_indices, max_index = feature_take_indices(5, indices)
        prune_end = max_index < 4
        layer_names = ('layer1', 'layer2', 'layer3', 'layer4')
        layer_names = layer_names[max_index:]
        for n in layer_names:
            setattr(self, n, nn.Sequential())  # empty, not Identity, so (flattened) checkpoint_seq can iterate it
        if max_index == 0:
            self.maxpool = nn.Identity()  # follows the stem feature, may have params (conv pool)
        if prune_end:
            # features now end at the last kept tap, the head is sized for the final features so it goes too
            self.num_features = self.head_hidden_size = self.feature_info[max_index]['num_chs']
        if prune_head or prune_end:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction layers."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass through classifier head.

        Args:
            x: Feature tensor.
            pre_logits: Return features before final classifier layer.

        Returns:
            Output tensor.
        """
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_resnet3d(variant: str, pretrained: bool = False, **kwargs) -> ResNet3d:
    """Create a 3D ResNet model.

    Args:
        variant: Model variant name.
        pretrained: Load pretrained weights.
        **kwargs: Additional model arguments.

    Returns:
        ResNet3d model instance.
    """
    return build_model_with_cfg(ResNet3d, variant, pretrained, **kwargs)


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    """Create a default configuration for 3D ResNet models."""
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (1, 32, 64, 64), 'pool_size': (32, 2, 2),
        'crop_pct': 0.875, 'interpolation': 'trilinear',
        'mean': (0.5,), 'std': (0.5,),
        'first_conv': 'conv1', 'classifier': 'fc',
        'license': 'apache-2.0',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'resnet18_3d': _cfg(),
    'resnet34_3d': _cfg(),
    'resnet50_3d': _cfg(),
})


@register_model
def resnet18_3d(pretrained: bool = False, **kwargs) -> ResNet3d:
    """Constructs a 3D ResNet-18 model."""
    model_args = dict(block=BasicBlock3d, layers=(2, 2, 2, 2))
    return _create_resnet3d('resnet18_3d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet34_3d(pretrained: bool = False, **kwargs) -> ResNet3d:
    """Constructs a 3D ResNet-34 model."""
    model_args = dict(block=BasicBlock3d, layers=(3, 4, 6, 3))
    return _create_resnet3d('resnet34_3d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50_3d(pretrained: bool = False, **kwargs) -> ResNet3d:
    """Constructs a 3D ResNet-50 model."""
    model_args = dict(block=Bottleneck3d, layers=(3, 4, 6, 3))
    return _create_resnet3d('resnet50_3d', pretrained, **dict(model_args, **kwargs))
