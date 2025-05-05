""" EfficientNet, MobileNetV3, etc Blocks

Hacked together by / Copyright 2019, Ross Wightman
"""
from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.layers import create_conv2d, DropPath, make_divisible, create_act_layer, create_aa, to_2tuple, LayerType,\
    ConvNormAct, get_norm_act_layer, MultiQueryAttention2d, Attention2d

__all__ = [
    'SqueezeExcite', 'ConvBnAct', 'DepthwiseSeparableConv', 'InvertedResidual', 'CondConvResidual', 'EdgeResidual',
    'UniversalInvertedResidual', 'MobileAttention'
]

ModuleType = Type[nn.Module]


def num_groups(group_size: Optional[int], channels: int):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class SqueezeExcite(nn.Module):
    """ Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
            self,
            in_chs: int,
            rd_ratio: float = 0.25,
            rd_channels: Optional[int] = None,
            act_layer: LayerType = nn.ReLU,
            gate_layer: LayerType = nn.Sigmoid,
            force_act_layer: Optional[LayerType] = None,
            rd_round_fn: Optional[Callable] = None,
    ):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = create_act_layer(act_layer, inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class ConvBnAct(nn.Module):
    """ Conv + Norm Layer + Activation w/ optional skip connection
    """
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 0,
            pad_type: str = '',
            skip: bool = False,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[LayerType] = None,
            drop_path_rate: float = 0.,
    ):
        super(ConvBnAct, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        groups = num_groups(group_size, in_chs)
        self.has_skip = skip and stride == 1 and in_chs == out_chs
        use_aa = aa_layer is not None and stride > 1  # FIXME handle dilation

        self.conv = create_conv2d(
            in_chs, out_chs, kernel_size,
            stride=1 if use_aa else stride,
            dilation=dilation, groups=groups, padding=pad_type)
        self.bn1 = norm_act_layer(out_chs, inplace=True)
        self.aa = create_aa(aa_layer, channels=out_chs, stride=stride, enable=use_aa)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # output of conv after act, same as block coutput
            return dict(module='bn1', hook_type='forward', num_chs=self.conv.out_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        x = self.aa(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class DepthwiseSeparableConv(nn.Module):
    """ Depthwise-separable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            dw_kernel_size: int = 3,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 1,
            pad_type: str = '',
            noskip: bool = False,
            pw_kernel_size: int = 1,
            pw_act: bool = False,
            s2d: int = 0,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[ModuleType] = None,
            drop_path_rate: float = 0.,
    ):
        super(DepthwiseSeparableConv, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        use_aa = aa_layer is not None and stride > 1  # FIXME handle dilation

        # Space to depth
        if s2d == 1:
            sd_chs = int(in_chs * 4)
            self.conv_s2d = create_conv2d(in_chs, sd_chs, kernel_size=2, stride=2, padding='same')
            self.bn_s2d = norm_act_layer(sd_chs, sd_chs)
            dw_kernel_size = (dw_kernel_size + 1) // 2
            dw_pad_type = 'same' if dw_kernel_size == 2 else pad_type
            in_chs = sd_chs
            use_aa = False  # disable AA
        else:
            self.conv_s2d = None
            self.bn_s2d = None
            dw_pad_type = pad_type

        groups = num_groups(group_size, in_chs)

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size,
            stride=1 if use_aa else stride,
            dilation=dilation, padding=dw_pad_type, groups=groups)
        self.bn1 = norm_act_layer(in_chs, inplace=True)
        self.aa = create_aa(aa_layer, channels=out_chs, stride=stride, enable=use_aa)

        # Squeeze-and-excitation
        self.se = se_layer(in_chs, act_layer=act_layer) if se_layer else nn.Identity()

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_act_layer(out_chs, inplace=True, apply_act=self.has_pw_act)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pw.out_channels)

    def forward(self, x):
        shortcut = x
        if self.conv_s2d is not None:
            x = self.conv_s2d(x)
            x = self.bn_s2d(x)
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.aa(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            dw_kernel_size: int = 3,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 1,
            pad_type: str = '',
            noskip: bool = False,
            exp_ratio: float = 1.0,
            exp_kernel_size: int = 1,
            pw_kernel_size: int = 1,
            s2d: int = 0,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[ModuleType] = None,
            conv_kwargs: Optional[Dict] = None,
            drop_path_rate: float = 0.,
    ):
        super(InvertedResidual, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        conv_kwargs = conv_kwargs or {}
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        use_aa = aa_layer is not None and stride > 1  # FIXME handle dilation

        # Space to depth
        if s2d == 1:
            sd_chs = int(in_chs * 4)
            self.conv_s2d = create_conv2d(in_chs, sd_chs, kernel_size=2, stride=2, padding='same')
            self.bn_s2d = norm_act_layer(sd_chs, sd_chs)
            dw_kernel_size = (dw_kernel_size + 1) // 2
            dw_pad_type = 'same' if dw_kernel_size == 2 else pad_type
            in_chs = sd_chs
            use_aa = False  # disable AA
        else:
            self.conv_s2d = None
            self.bn_s2d = None
            dw_pad_type = pad_type

        mid_chs = make_divisible(in_chs * exp_ratio)
        groups = num_groups(group_size, mid_chs)

        # Point-wise expansion
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_act_layer(mid_chs, inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size,
            stride=1 if use_aa else stride,
            dilation=dilation, groups=groups, padding=dw_pad_type, **conv_kwargs)
        self.bn2 = norm_act_layer(mid_chs, inplace=True)
        self.aa = create_aa(aa_layer, channels=mid_chs, stride=stride, enable=use_aa)

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_act_layer(out_chs, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            return dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pwl.out_channels)

    def forward(self, x):
        shortcut = x
        if self.conv_s2d is not None:
            x = self.conv_s2d(x)
            x = self.bn_s2d(x)
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.aa(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class LayerScale2d(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


class UniversalInvertedResidual(nn.Module):
    """ Universal Inverted Residual Block (aka Universal Inverted Bottleneck, UIB)

    For MobileNetV4 - https://arxiv.org/abs/, referenced from
    https://github.com/tensorflow/models/blob/d93c7e932de27522b2fa3b115f58d06d6f640537/official/vision/modeling/layers/nn_blocks.py#L778
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            dw_kernel_size_start: int = 0,
            dw_kernel_size_mid: int = 3,
            dw_kernel_size_end: int = 0,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 1,
            pad_type: str = '',
            noskip: bool = False,
            exp_ratio: float = 1.0,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[ModuleType] = None,
            conv_kwargs: Optional[Dict] = None,
            drop_path_rate: float = 0.,
            layer_scale_init_value: Optional[float] = 1e-5,
    ):
        super(UniversalInvertedResidual, self).__init__()
        conv_kwargs = conv_kwargs or {}
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        if stride > 1:
            assert dw_kernel_size_start or dw_kernel_size_mid or dw_kernel_size_end

        # FIXME dilation isn't right w/ extra ks > 1 convs
        if dw_kernel_size_start:
            dw_start_stride = stride if not dw_kernel_size_mid else 1
            dw_start_groups = num_groups(group_size, in_chs)
            self.dw_start = ConvNormAct(
                in_chs, in_chs, dw_kernel_size_start,
                stride=dw_start_stride,
                dilation=dilation,  # FIXME
                groups=dw_start_groups,
                padding=pad_type,
                apply_act=False,
                act_layer=act_layer,
                norm_layer=norm_layer,
                aa_layer=aa_layer,
                **conv_kwargs,
            )
        else:
            self.dw_start = nn.Identity()

        # Point-wise expansion
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.pw_exp = ConvNormAct(
            in_chs, mid_chs, 1,
            padding=pad_type,
            act_layer=act_layer,
            norm_layer=norm_layer,
            **conv_kwargs,
        )

        # Middle depth-wise convolution
        if dw_kernel_size_mid:
            groups = num_groups(group_size, mid_chs)
            self.dw_mid = ConvNormAct(
                mid_chs, mid_chs, dw_kernel_size_mid,
                stride=stride,
                dilation=dilation,  # FIXME
                groups=groups,
                padding=pad_type,
                act_layer=act_layer,
                norm_layer=norm_layer,
                aa_layer=aa_layer,
                **conv_kwargs,
            )
        else:
            # keeping mid as identity so it can be hooked more easily for features
            self.dw_mid = nn.Identity()

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.pw_proj = ConvNormAct(
            mid_chs, out_chs, 1,
            padding=pad_type,
            apply_act=False,
            act_layer=act_layer,
            norm_layer=norm_layer,
            **conv_kwargs,
        )

        if dw_kernel_size_end:
            dw_end_stride = stride if not dw_kernel_size_start and not dw_kernel_size_mid else 1
            dw_end_groups = num_groups(group_size, out_chs)
            if dw_end_stride > 1:
                assert not aa_layer
            self.dw_end = ConvNormAct(
                out_chs, out_chs, dw_kernel_size_end,
                stride=dw_end_stride,
                dilation=dilation,
                groups=dw_end_groups,
                padding=pad_type,
                apply_act=False,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **conv_kwargs,
            )
        else:
            self.dw_end = nn.Identity()

        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(out_chs, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            return dict(module='pw_proj.conv', hook_type='forward_pre', num_chs=self.pw_proj.conv.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.pw_proj.conv.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.dw_start(x)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.dw_end(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class MobileAttention(nn.Module):
    """ Mobile Attention Block

    For MobileNetV4 - https://arxiv.org/abs/, referenced from
    https://github.com/tensorflow/models/blob/d93c7e932de27522b2fa3b115f58d06d6f640537/official/vision/modeling/layers/nn_blocks.py#L1504
    """
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dw_kernel_size: int = 3,
            dilation: int = 1,
            group_size: int = 1,
            pad_type: str = '',
            num_heads: int = 8,
            key_dim: int = 64,
            value_dim: int = 64,
            use_multi_query: bool = False,
            query_strides: int = (1, 1),
            kv_stride: int = 1,
            cpe_dw_kernel_size: int = 3,
            noskip: bool = False,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[LayerType] = None,
            drop_path_rate: float = 0.,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            layer_scale_init_value: Optional[float] = 1e-5,
            use_bias: bool = False,
            use_cpe: bool = False,
    ):
        super(MobileAttention, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.query_strides = to_2tuple(query_strides)
        self.kv_stride = kv_stride
        self.has_query_stride = any([s > 1 for s in self.query_strides])

        # This CPE is different than the one suggested in the original paper.
        # https://arxiv.org/abs/2102.10882
        # 1. Rather than adding one CPE before the attention blocks, we add a CPE
        #    into every attention block.
        # 2. We replace the expensive Conv2D by a Separable DW Conv.
        if use_cpe:
            self.conv_cpe_dw = create_conv2d(
                in_chs, in_chs,
                kernel_size=cpe_dw_kernel_size,
                dilation=dilation,
                depthwise=True,
                bias=True,
            )
        else:
            self.conv_cpe_dw = None

        self.norm = norm_act_layer(in_chs, apply_act=False)

        if num_heads is None:
            assert in_chs % key_dim == 0
            num_heads = in_chs // key_dim

        if use_multi_query:
            self.attn = MultiQueryAttention2d(
                in_chs,
                dim_out=out_chs,
                num_heads=num_heads,
                key_dim=key_dim,
                value_dim=value_dim,
                query_strides=query_strides,
                kv_stride=kv_stride,
                dw_kernel_size=dw_kernel_size,
                dilation=dilation,
                padding=pad_type,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                # use_bias=use_bias, # why not here if used w/ mhsa?
            )
        else:
            self.attn = Attention2d(
                in_chs,
                dim_out=out_chs,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                bias=use_bias,
            )

        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(out_chs, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pw.out_channels)

    def forward(self, x):
        if self.conv_cpe_dw is not None:
            x_cpe = self.conv_cpe_dw(x)
            x = x + x_cpe

        shortcut = x
        x = self.norm(x)
        x = self.attn(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut

        return x


class CondConvResidual(InvertedResidual):
    """ Inverted residual block w/ CondConv routing"""

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            dw_kernel_size: int = 3,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 1,
            pad_type: str = '',
            noskip: bool = False,
            exp_ratio: float = 1.0,
            exp_kernel_size: int = 1,
            pw_kernel_size: int = 1,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[ModuleType] = None,
            num_experts: int = 0,
            drop_path_rate: float = 0.,
    ):

        self.num_experts = num_experts
        conv_kwargs = dict(num_experts=self.num_experts)
        super(CondConvResidual, self).__init__(
            in_chs,
            out_chs,
            dw_kernel_size=dw_kernel_size,
            stride=stride,
            dilation=dilation,
            group_size=group_size,
            pad_type=pad_type,
            noskip=noskip,
            exp_ratio=exp_ratio,
            exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            se_layer=se_layer,
            conv_kwargs=conv_kwargs,
            drop_path_rate=drop_path_rate,
        )
        self.routing_fn = nn.Linear(in_chs, self.num_experts)

    def forward(self, x):
        shortcut = x
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))
        x = self.conv_pw(x, routing_weights)
        x = self.bn1(x)
        x = self.conv_dw(x, routing_weights)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x, routing_weights)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class EdgeResidual(nn.Module):
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride

    Originally introduced in `EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML`
        - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

    This layer is also called FusedMBConv in the MobileDet, EfficientNet-X, and EfficientNet-V2 papers
      * MobileDet - https://arxiv.org/abs/2004.14525
      * EfficientNet-X - https://arxiv.org/abs/2102.05610
      * EfficientNet-V2 - https://arxiv.org/abs/2104.00298
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            exp_kernel_size: int = 3,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 0,
            pad_type: str = '',
            force_in_chs: int = 0,
            noskip: bool = False,
            exp_ratio: float = 1.0,
            pw_kernel_size:  int = 1,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[ModuleType] = None,
            drop_path_rate: float = 0.,
    ):
        super(EdgeResidual, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        if force_in_chs > 0:
            mid_chs = make_divisible(force_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        groups = num_groups(group_size, mid_chs)  # NOTE: Using out_chs of conv_exp for groups calc
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        use_aa = aa_layer is not None and stride > 1  # FIXME handle dilation

        # Expansion convolution
        self.conv_exp = create_conv2d(
            in_chs, mid_chs, exp_kernel_size,
            stride=1 if use_aa else stride,
            dilation=dilation, groups=groups, padding=pad_type)
        self.bn1 = norm_act_layer(mid_chs, inplace=True)

        self.aa = create_aa(aa_layer, channels=mid_chs, stride=stride, enable=use_aa)

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_act_layer(out_chs, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, before PWL
            return dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pwl.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.aa(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x
