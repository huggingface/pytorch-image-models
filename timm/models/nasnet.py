""" NasNet-A (Large)
 nasnetalarge implementation grabbed from Cadene's pretrained models
 https://github.com/Cadene/pretrained-models.pytorch
"""
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import ConvNormAct, create_conv2d, create_pool2d, create_classifier
from ._builder import build_model_with_cfg
from ._registry import register_model, generate_default_cfgs

__all__ = ['NASNetALarge']



class ActConvBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=''):
        super(ActConvBn, self).__init__()
        self.act = nn.ReLU()
        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)

    def forward(self, x):
        x = self.act(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=''):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = create_conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels)
        self.pointwise_conv2d = create_conv2d(
            in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_type='', stem_cell=False):
        super(BranchSeparables, self).__init__()
        middle_channels = out_channels if stem_cell else in_channels
        self.act_1 = nn.ReLU()
        self.separable_1 = SeparableConv2d(
            in_channels, middle_channels, kernel_size, stride=stride, padding=pad_type)
        self.bn_sep_1 = nn.BatchNorm2d(middle_channels, eps=0.001, momentum=0.1)
        self.act_2 = nn.ReLU(inplace=True)
        self.separable_2 = SeparableConv2d(
            middle_channels, out_channels, kernel_size, stride=1, padding=pad_type)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)

    def forward(self, x):
        x = self.act_1(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.act_2(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):
    def __init__(self, stem_size, num_channels=42, pad_type=''):
        super(CellStem0, self).__init__()
        self.num_channels = num_channels
        self.stem_size = stem_size
        self.conv_1x1 = ActConvBn(self.stem_size, self.num_channels, 1, stride=1)

        self.comb_iter_0_left = BranchSeparables(self.num_channels, self.num_channels, 5, 2, pad_type)
        self.comb_iter_0_right = BranchSeparables(self.stem_size, self.num_channels, 7, 2, pad_type, stem_cell=True)

        self.comb_iter_1_left = create_pool2d('max', 3, 2, padding=pad_type)
        self.comb_iter_1_right = BranchSeparables(self.stem_size, self.num_channels, 7, 2, pad_type, stem_cell=True)

        self.comb_iter_2_left = create_pool2d('avg', 3, 2, count_include_pad=False, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(self.stem_size, self.num_channels, 5, 2, pad_type, stem_cell=True)

        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(self.num_channels, self.num_channels, 3, 1, pad_type)
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

    def __init__(self, stem_size, num_channels, pad_type=''):
        super(CellStem1, self).__init__()
        self.num_channels = num_channels
        self.stem_size = stem_size
        self.conv_1x1 = ActConvBn(2 * self.num_channels, self.num_channels, 1, stride=1)

        self.act = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_size, self.num_channels // 2, 1, stride=1, bias=False))
       
        self.path_2 = nn.Sequential()
        self.path_2.add_module('pad', nn.ZeroPad2d((-1, 1, -1, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_size, self.num_channels // 2, 1, stride=1, bias=False))

        self.final_path_bn = nn.BatchNorm2d(self.num_channels, eps=0.001, momentum=0.1)

        self.comb_iter_0_left = BranchSeparables(self.num_channels, self.num_channels, 5, 2, pad_type)
        self.comb_iter_0_right = BranchSeparables(self.num_channels, self.num_channels, 7, 2, pad_type)

        self.comb_iter_1_left = create_pool2d('max', 3, 2, padding=pad_type)
        self.comb_iter_1_right = BranchSeparables(self.num_channels, self.num_channels, 7, 2, pad_type)

        self.comb_iter_2_left = create_pool2d('avg', 3, 2, count_include_pad=False, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(self.num_channels, self.num_channels, 5, 2, pad_type)

        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(self.num_channels, self.num_channels, 3, 1, pad_type)
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

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(FirstCell, self).__init__()
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1)

        self.act = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_chs_left, out_chs_left, 1, stride=1, bias=False))

        self.path_2 = nn.Sequential()
        self.path_2.add_module('pad', nn.ZeroPad2d((-1, 1, -1, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_chs_left, out_chs_left, 1, stride=1, bias=False))

        self.final_path_bn = nn.BatchNorm2d(out_chs_left * 2, eps=0.001, momentum=0.1)

        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 1, pad_type)
        self.comb_iter_0_right = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)

        self.comb_iter_1_left = BranchSeparables(out_chs_right, out_chs_right, 5, 1, pad_type)
        self.comb_iter_1_right = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)

        self.comb_iter_2_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_3_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)

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

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, 1, stride=1, padding=pad_type)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1, padding=pad_type)

        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 1, pad_type)
        self.comb_iter_0_right = BranchSeparables(out_chs_left, out_chs_left, 3, 1, pad_type)

        self.comb_iter_1_left = BranchSeparables(out_chs_left, out_chs_left, 5, 1, pad_type)
        self.comb_iter_1_right = BranchSeparables(out_chs_left, out_chs_left, 3, 1, pad_type)

        self.comb_iter_2_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_3_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)

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

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, 1, stride=1, padding=pad_type)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1, padding=pad_type)

        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type)
        self.comb_iter_0_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type)

        self.comb_iter_1_left = create_pool2d('max', 3, 2, padding=pad_type)
        self.comb_iter_1_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type)

        self.comb_iter_2_left = create_pool2d('avg', 3, 2, count_include_pad=False, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type)

        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)
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

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, 1, stride=1, padding=pad_type)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1, padding=pad_type)

        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type)
        self.comb_iter_0_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type)

        self.comb_iter_1_left = create_pool2d('max', 3, 2, padding=pad_type)
        self.comb_iter_1_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type)

        self.comb_iter_2_left = create_pool2d('avg', 3, 2, count_include_pad=False, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type)

        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)
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
            num_classes=1000,
            in_chans=3,
            stem_size=96,
            channel_multiplier=2,
            num_features=4032,
            output_stride=32,
            drop_rate=0.,
            global_pool='avg',
            pad_type='same',
    ):
        super(NASNetALarge, self).__init__()
        self.num_classes = num_classes
        self.stem_size = stem_size
        self.num_features = self.head_hidden_size = num_features
        self.channel_multiplier = channel_multiplier
        assert output_stride == 32

        channels = self.num_features // 24
        # 24 is default value for the architecture

        self.conv0 = ConvNormAct(
            in_channels=in_chans, out_channels=self.stem_size, kernel_size=3, padding=0, stride=2,
            norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.1), apply_act=False)

        self.cell_stem_0 = CellStem0(
            self.stem_size, num_channels=channels // (channel_multiplier ** 2), pad_type=pad_type)
        self.cell_stem_1 = CellStem1(
            self.stem_size, num_channels=channels // channel_multiplier, pad_type=pad_type)

        self.cell_0 = FirstCell(
            in_chs_left=channels, out_chs_left=channels // 2,
            in_chs_right=2 * channels, out_chs_right=channels, pad_type=pad_type)
        self.cell_1 = NormalCell(
            in_chs_left=2 * channels, out_chs_left=channels,
            in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type)
        self.cell_2 = NormalCell(
            in_chs_left=6 * channels, out_chs_left=channels,
            in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type)
        self.cell_3 = NormalCell(
            in_chs_left=6 * channels, out_chs_left=channels,
            in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type)
        self.cell_4 = NormalCell(
            in_chs_left=6 * channels, out_chs_left=channels,
            in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type)
        self.cell_5 = NormalCell(
            in_chs_left=6 * channels, out_chs_left=channels,
            in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type)

        self.reduction_cell_0 = ReductionCell0(
            in_chs_left=6 * channels, out_chs_left=2 * channels,
            in_chs_right=6 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_6 = FirstCell(
            in_chs_left=6 * channels, out_chs_left=channels,
            in_chs_right=8 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_7 = NormalCell(
            in_chs_left=8 * channels, out_chs_left=2 * channels,
            in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_8 = NormalCell(
            in_chs_left=12 * channels, out_chs_left=2 * channels,
            in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_9 = NormalCell(
            in_chs_left=12 * channels, out_chs_left=2 * channels,
            in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_10 = NormalCell(
            in_chs_left=12 * channels, out_chs_left=2 * channels,
            in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_11 = NormalCell(
            in_chs_left=12 * channels, out_chs_left=2 * channels,
            in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type)

        self.reduction_cell_1 = ReductionCell1(
            in_chs_left=12 * channels, out_chs_left=4 * channels,
            in_chs_right=12 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_12 = FirstCell(
            in_chs_left=12 * channels, out_chs_left=2 * channels,
            in_chs_right=16 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_13 = NormalCell(
            in_chs_left=16 * channels, out_chs_left=4 * channels,
            in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_14 = NormalCell(
            in_chs_left=24 * channels, out_chs_left=4 * channels,
            in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_15 = NormalCell(
            in_chs_left=24 * channels, out_chs_left=4 * channels,
            in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_16 = NormalCell(
            in_chs_left=24 * channels, out_chs_left=4 * channels,
            in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_17 = NormalCell(
            in_chs_left=24 * channels, out_chs_left=4 * channels,
            in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.act = nn.ReLU(inplace=True)
        self.feature_info = [
            dict(num_chs=96, reduction=2, module='conv0'),
            dict(num_chs=168, reduction=4, module='cell_stem_1.conv_1x1.act'),
            dict(num_chs=1008, reduction=8, module='reduction_cell_0.conv_1x1.act'),
            dict(num_chs=2016, reduction=16, module='reduction_cell_1.conv_1x1.act'),
            dict(num_chs=4032, reduction=32, module='act'),
        ]

        self.global_pool, self.head_drop, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool, drop_rate=drop_rate)

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

    def reset_classifier(self, num_classes: int, global_pool: str = 'avg'):
        self.num_classes = num_classes
        self.global_pool, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x_conv0 = self.conv0(x)

        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)

        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.cell_5(x_cell_4, x_cell_3)

        x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.cell_11(x_cell_10, x_cell_9)

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
    },
})


@register_model
def nasnetalarge(pretrained=False, **kwargs) -> NASNetALarge:
    """NASNet-A large model architecture.
    """
    model_kwargs = dict(pad_type='same', **kwargs)
    return _create_nasnet('nasnetalarge', pretrained, **model_kwargs)
