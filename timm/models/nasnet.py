""" NasNet-A (Large)
 nasnetalarge implementation grabbed from Cadene's pretrained models
 https://github.com/Cadene/pretrained-models.pytorch

Papers:
* `Learning Transferable Architectures for Scalable Image Recognition` - https://arxiv.org/abs/1707.07012
"""
from functools import partial
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import ConvNormAct, create_conv2d, create_pool2d, create_classifier, get_device_dtype
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._registry import register_model, generate_default_cfgs

__all__ = ['NASNetALarge']



class ActConvBn(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: str = '',
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.act = nn.ReLU()
        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, **dd)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, **dd)

    def forward(self, x):
        x = self.act(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class SeparableConv2d(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: str = '',
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.depthwise_conv2d = create_conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            **dd,
        )
        self.pointwise_conv2d = create_conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            **dd,
        )

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            pad_type: str = '',
            stem_cell: bool = False,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        middle_channels = out_channels if stem_cell else in_channels
        self.act_1 = nn.ReLU()
        self.separable_1 = SeparableConv2d(
            in_channels, middle_channels, kernel_size, stride=stride, padding=pad_type, **dd)
        self.bn_sep_1 = nn.BatchNorm2d(middle_channels, eps=0.001, momentum=0.1, **dd)
        self.act_2 = nn.ReLU(inplace=True)
        self.separable_2 = SeparableConv2d(
            middle_channels, out_channels, kernel_size, stride=1, padding=pad_type, **dd)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, **dd)

    def forward(self, x):
        x = self.act_1(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.act_2(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):
    def __init__(
            self,
            stem_size: int,
            num_channels: int = 42,
            pad_type: str = '',
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_channels = num_channels
        self.stem_size = stem_size
        self.conv_1x1 = ActConvBn(self.stem_size, self.num_channels, 1, stride=1, **dd)

        self.comb_iter_0_left = BranchSeparables(
            self.num_channels, self.num_channels, 5, 2, pad_type, **dd)
        self.comb_iter_0_right = BranchSeparables(
            self.stem_size, self.num_channels, 7, 2, pad_type, stem_cell=True, **dd)

        self.comb_iter_1_left = create_pool2d('max', 3, 2, padding=pad_type)
        self.comb_iter_1_right = BranchSeparables(
            self.stem_size, self.num_channels, 7, 2, pad_type, stem_cell=True, **dd)

        self.comb_iter_2_left = create_pool2d('avg', 3, 2, count_include_pad=False, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(
            self.stem_size, self.num_channels, 5, 2, pad_type, stem_cell=True, **dd)

        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(
            self.num_channels, self.num_channels, 3, 1, pad_type, **dd)
        self.comb_iter_4_right = create_pool2d('max', 3, 2, padding=pad_type)

    def forward(self, x):
        x1 = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):

    def __init__(
            self,
            stem_size: int,
            num_channels: int,
            pad_type: str = '',
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_channels = num_channels
        self.stem_size = stem_size
        self.conv_1x1 = ActConvBn(2 * self.num_channels, self.num_channels, 1, stride=1, **dd)

        self.act = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_size, self.num_channels // 2, 1, stride=1, bias=False, **dd))

        self.path_2 = nn.Sequential()
        self.path_2.add_module('pad', nn.ZeroPad2d((-1, 1, -1, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_size, self.num_channels // 2, 1, stride=1, bias=False, **dd))

        self.final_path_bn = nn.BatchNorm2d(self.num_channels, eps=0.001, momentum=0.1, **dd)

        self.comb_iter_0_left = BranchSeparables(self.num_channels, self.num_channels, 5, 2, pad_type, **dd)
        self.comb_iter_0_right = BranchSeparables(self.num_channels, self.num_channels, 7, 2, pad_type, **dd)

        self.comb_iter_1_left = create_pool2d('max', 3, 2, padding=pad_type)
        self.comb_iter_1_right = BranchSeparables(self.num_channels, self.num_channels, 7, 2, pad_type, **dd)

        self.comb_iter_2_left = create_pool2d('avg', 3, 2, count_include_pad=False, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(self.num_channels, self.num_channels, 5, 2, pad_type, **dd)

        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(self.num_channels, self.num_channels, 3, 1, pad_type, **dd)
        self.comb_iter_4_right = create_pool2d('max', 3, 2, padding=pad_type)

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)

        x_relu = self.act(x_conv0)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2(x_relu)
        # final path
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))

        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(
            self,
            in_chs_left: int,
            out_chs_left: int,
            in_chs_right: int,
            out_chs_right: int,
            pad_type: str = '',
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1, **dd)

        self.act = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_chs_left, out_chs_left, 1, stride=1, bias=False, **dd))

        self.path_2 = nn.Sequential()
        self.path_2.add_module('pad', nn.ZeroPad2d((-1, 1, -1, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_chs_left, out_chs_left, 1, stride=1, bias=False, **dd))

        self.final_path_bn = nn.BatchNorm2d(out_chs_left * 2, eps=0.001, momentum=0.1, **dd)

        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 1, pad_type, **dd)
        self.comb_iter_0_right = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type, **dd)

        self.comb_iter_1_left = BranchSeparables(out_chs_right, out_chs_right, 5, 1, pad_type, **dd)
        self.comb_iter_1_right = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type, **dd)

        self.comb_iter_2_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_3_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type, **dd)

    def forward(self, x, x_prev):
        x_relu = self.act(x_prev)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2(x_relu)
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(
            self,
            in_chs_left: int,
            out_chs_left: int,
            in_chs_right: int,
            out_chs_right: int,
            pad_type: str = '',
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, 1, stride=1, padding=pad_type, **dd)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1, padding=pad_type, **dd)

        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 1, pad_type, **dd)
        self.comb_iter_0_right = BranchSeparables(out_chs_left, out_chs_left, 3, 1, pad_type, **dd)

        self.comb_iter_1_left = BranchSeparables(out_chs_left, out_chs_left, 5, 1, pad_type, **dd)
        self.comb_iter_1_right = BranchSeparables(out_chs_left, out_chs_left, 3, 1, pad_type, **dd)

        self.comb_iter_2_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_3_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type, **dd)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell0(nn.Module):

    def __init__(
            self,
            in_chs_left: int,
            out_chs_left: int,
            in_chs_right: int,
            out_chs_right: int,
            pad_type: str = '',
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, 1, stride=1, padding=pad_type, **dd)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1, padding=pad_type, **dd)

        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type, **dd)
        self.comb_iter_0_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type, **dd)

        self.comb_iter_1_left = create_pool2d('max', 3, 2, padding=pad_type)
        self.comb_iter_1_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type, **dd)

        self.comb_iter_2_left = create_pool2d('avg', 3, 2, count_include_pad=False, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type, **dd)

        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type, **dd)
        self.comb_iter_4_right = create_pool2d('max', 3, 2, padding=pad_type)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(
            self,
            in_chs_left: int,
            out_chs_left: int,
            in_chs_right: int,
            out_chs_right: int,
            pad_type: str = '',
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, 1, stride=1, padding=pad_type, **dd)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1, padding=pad_type, **dd)

        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type, **dd)
        self.comb_iter_0_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type, **dd)

        self.comb_iter_1_left = create_pool2d('max', 3, 2, padding=pad_type)
        self.comb_iter_1_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type, **dd)

        self.comb_iter_2_left = create_pool2d('avg', 3, 2, count_include_pad=False, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type, **dd)

        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type, **dd)
        self.comb_iter_4_right = create_pool2d('max', 3, 2, padding=pad_type)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NASNetALarge(nn.Module):
    """NASNetALarge (6 @ 4032) """

    def __init__(
            self,
            num_classes: int = 1000,
            in_chans: int = 3,
            stem_size: int = 96,
            channel_multiplier: int = 2,
            num_features: int = 4032,
            output_stride: int = 32,
            drop_rate: float = 0.,
            global_pool: str = 'avg',
            pad_type: str = 'same',
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.stem_size = stem_size
        self.num_features = num_features
        self.channel_multiplier = channel_multiplier
        assert output_stride == 32

        channels = self.num_features // 24
        # 24 is default value for the architecture

        self.conv0 = ConvNormAct(
            in_channels=in_chans,
            out_channels=self.stem_size,
            kernel_size=3,
            padding=0,
            stride=2,
            norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.1),
            apply_act=False,
            **dd,
        )

        self.cell_stem_0 = CellStem0(
            self.stem_size, num_channels=channels // (channel_multiplier ** 2), pad_type=pad_type, **dd)
        self.cell_stem_1 = CellStem1(
            self.stem_size, num_channels=channels // channel_multiplier, pad_type=pad_type, **dd)

        self.cell_0 = FirstCell(
            in_chs_left=channels, out_chs_left=channels // 2,
            in_chs_right=2 * channels, out_chs_right=channels, pad_type=pad_type, **dd)
        self.cell_1 = NormalCell(
            in_chs_left=2 * channels, out_chs_left=channels,
            in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type, **dd)
        self.cell_2 = NormalCell(
            in_chs_left=6 * channels, out_chs_left=channels,
            in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type, **dd)
        self.cell_3 = NormalCell(
            in_chs_left=6 * channels, out_chs_left=channels,
            in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type, **dd)
        self.cell_4 = NormalCell(
            in_chs_left=6 * channels, out_chs_left=channels,
            in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type, **dd)
        self.cell_5 = NormalCell(
            in_chs_left=6 * channels, out_chs_left=channels,
            in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type, **dd)

        self.reduction_cell_0 = ReductionCell0(
            in_chs_left=6 * channels, out_chs_left=2 * channels,
            in_chs_right=6 * channels, out_chs_right=2 * channels, pad_type=pad_type, **dd)
        self.cell_6 = FirstCell(
            in_chs_left=6 * channels, out_chs_left=channels,
            in_chs_right=8 * channels, out_chs_right=2 * channels, pad_type=pad_type, **dd)
        self.cell_7 = NormalCell(
            in_chs_left=8 * channels, out_chs_left=2 * channels,
            in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type, **dd)
        self.cell_8 = NormalCell(
            in_chs_left=12 * channels, out_chs_left=2 * channels,
            in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type, **dd)
        self.cell_9 = NormalCell(
            in_chs_left=12 * channels, out_chs_left=2 * channels,
            in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type, **dd)
        self.cell_10 = NormalCell(
            in_chs_left=12 * channels, out_chs_left=2 * channels,
            in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type, **dd)
        self.cell_11 = NormalCell(
            in_chs_left=12 * channels, out_chs_left=2 * channels,
            in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type, **dd)

        self.reduction_cell_1 = ReductionCell1(
            in_chs_left=12 * channels, out_chs_left=4 * channels,
            in_chs_right=12 * channels, out_chs_right=4 * channels, pad_type=pad_type, **dd)
        self.cell_12 = FirstCell(
            in_chs_left=12 * channels, out_chs_left=2 * channels,
            in_chs_right=16 * channels, out_chs_right=4 * channels, pad_type=pad_type, **dd)
        self.cell_13 = NormalCell(
            in_chs_left=16 * channels, out_chs_left=4 * channels,
            in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type, **dd)
        self.cell_14 = NormalCell(
            in_chs_left=24 * channels, out_chs_left=4 * channels,
            in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type, **dd)
        self.cell_15 = NormalCell(
            in_chs_left=24 * channels, out_chs_left=4 * channels,
            in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type, **dd)
        self.cell_16 = NormalCell(
            in_chs_left=24 * channels, out_chs_left=4 * channels,
            in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type, **dd)
        self.cell_17 = NormalCell(
            in_chs_left=24 * channels, out_chs_left=4 * channels,
            in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type, **dd)
        self.act = nn.ReLU(inplace=True)
        self.feature_info = [
            dict(num_chs=96, reduction=2, module='conv0'),
            dict(num_chs=168, reduction=4, module='cell_stem_1.conv_1x1.act'),
            dict(num_chs=1008, reduction=8, module='reduction_cell_0.conv_1x1.act'),
            dict(num_chs=2016, reduction=16, module='reduction_cell_1.conv_1x1.act'),
            dict(num_chs=4032, reduction=32, module='act'),
        ]
        # stem / cells in forward order, and the layer index of the cell output used for each feature_info tap
        self.layer_names = (
            ['conv0', 'cell_stem_0', 'cell_stem_1']
            + [f'cell_{i}' for i in range(6)] + ['reduction_cell_0']
            + [f'cell_{i}' for i in range(6, 12)] + ['reduction_cell_1']
            + [f'cell_{i}' for i in range(12, 18)]
        )
        tap_layers = ('conv0', 'cell_stem_0', 'cell_5', 'cell_11', 'cell_17')
        self.feature_ends = [self.layer_names.index(n) for n in tap_layers]

        self.global_pool, self.head_drop, self.last_linear = create_classifier(
            self.num_features,
            self.num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            **dd,
        )
        self.head_hidden_size = self.num_features * self.global_pool.feat_mult()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^conv0|cell_stem_[01]',
            blocks=[
                (r'^cell_(\d+)', None),
                (r'^reduction_cell_0', (6,)),
                (r'^reduction_cell_1', (12,)),
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.last_linear

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        dd = get_device_dtype(self)
        self.num_classes = num_classes
        if global_pool is None:
            global_pool = self.global_pool.pool_type
        self.global_pool, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool, **dd)
        self.head_hidden_size = self.num_features * self.global_pool.feat_mult()
        self.global_pool.train(self.training)
        self.last_linear.train(self.training)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor.
            indices: Take last n features if int, all if None, select matching indices if sequence.
                Index 0 is the conv0 stem output, indices 1-3 the ReLU activated outputs of cell_stem_0, cell_5
                and cell_11 (as consumed by the next cell), index 4 the final activated cell_17 output.
            norm: Apply norm layer to compatible intermediates (unused, no final norm).
            stop_early: Stop iterating over blocks when last desired intermediate hit.
            output_fmt: Shape of intermediate feature outputs.
            intermediates_only: Only return intermediate features.

        Returns:
            Features and list of intermediate features or just intermediate features.
        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.feature_ends), indices)
        last_idx = len(self.feature_ends) - 1

        # forward pass, each cell takes the outputs of the previous two layers
        layer_names = self.layer_names
        if stop_early:
            layer_names = layer_names[:self.feature_ends[max_index] + 1]
        x_prev = x
        feat_idx = 0
        for i, n in enumerate(layer_names):
            layer = getattr(self, n)
            if i < 2:
                x_prev, x = x, layer(x)  # conv0 and cell_stem_0 take a single input
            elif i == 2:
                x_prev, x = x, layer(x_prev, x)  # cell_stem_1 takes (x_conv0, x_stem_0)
            elif isinstance(layer, (ReductionCell0, ReductionCell1)):
                x = layer(x, x_prev)  # the cell after a reduction cell takes the same x_prev
            else:
                x_prev, x = x, layer(x, x_prev)
            if i == self.feature_ends[feat_idx]:
                if feat_idx == last_idx:
                    x = self.act(x)
                if feat_idx in take_indices:
                    if 0 < feat_idx < last_idx:
                        # match the feature_info hook on the next cell's conv_1x1.act
                        intermediates.append(F.relu(x))
                    else:
                        intermediates.append(x)
                feat_idx += 1

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
            prune_norm: Whether to prune normalization layers (unused, no final norm).
            prune_head: Whether to prune the classifier head.

        Returns:
            List of indices that were kept.
        """
        take_indices, max_index = feature_take_indices(len(self.feature_ends), indices)
        num_layers = self.feature_ends[max_index] + 1
        prune_end = num_layers < len(self.layer_names)
        for n in self.layer_names[num_layers:]:
            setattr(self, n, None)  # cells take two inputs, can't be run through Identity (see forward_features)
        self.layer_names = self.layer_names[:num_layers]
        if prune_end:
            # features now end at the last kept tap, the head is sized for the final features so it goes too
            self.num_features = self.head_hidden_size = self.feature_info[max_index]['num_chs']
        if prune_head or prune_end:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        # the layer after each feature tap is checked for None to support pruning (prune_intermediate_layers)
        x_conv0 = self.conv0(x)
        if self.cell_stem_0 is None:
            return x_conv0

        x_stem_0 = self.cell_stem_0(x_conv0)
        if self.cell_stem_1 is None:
            return x_stem_0
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)

        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.cell_5(x_cell_4, x_cell_3)
        if self.reduction_cell_0 is None:
            return x_cell_5

        x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.cell_11(x_cell_10, x_cell_9)
        if self.reduction_cell_1 is None:
            return x_cell_11

        x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_10)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.cell_17(x_cell_16, x_cell_15)
        x = self.act(x_cell_17)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.head_drop(x)
        return x if pre_logits else self.last_linear(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_nasnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        NASNetALarge,
        variant,
        pretrained,
        feature_cfg=dict(feature_cls='hook', no_rewrite=True),  # not possible to re-write this model
        **kwargs,
    )


default_cfgs = generate_default_cfgs({
    'nasnetalarge.tf_in1k': {
        'hf_hub_id': 'timm/',
        'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/nasnetalarge-dc4a7b8b.pth',
        'input_size': (3, 331, 331),
        'pool_size': (11, 11),
        'crop_pct': 0.911,
        'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'num_classes': 1000,
        'first_conv': 'conv0.conv',
        'classifier': 'last_linear',
        'license': 'apache-2.0',
    },
})


@register_model
def nasnetalarge(pretrained=False, **kwargs) -> NASNetALarge:
    """NASNet-A large model architecture.
    """
    model_kwargs = dict(pad_type='same', **kwargs)
    return _create_nasnet('nasnetalarge', pretrained, **model_kwargs)
