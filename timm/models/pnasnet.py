"""
 pnasnet5large implementation grabbed from Cadene's pretrained models
 Additional credit to https://github.com/creafz

 https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/pnasnet.py

"""
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import ConvNormAct, create_conv2d, create_pool2d, create_classifier
from ._builder import build_model_with_cfg
from ._registry import register_model

__all__ = ['PNASNet5Large']

default_cfgs = {
    'pnasnet5large': {
        'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/pnasnet5large-bf079911.pth',
        'input_size': (3, 331, 331),
        'pool_size': (11, 11),
        'crop_pct': 0.911,
        'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'num_classes': 1000,
        'first_conv': 'conv_0.conv',
        'classifier': 'last_linear',
        'label_offset': 1,  # 1001 classes in pretrained weights
    },
}


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=''):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = create_conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels)
        self.pointwise_conv2d = create_conv2d(
            in_channels, out_channels, kernel_size=1, padding=padding)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, stem_cell=False, padding=''):
        super(BranchSeparables, self).__init__()
        middle_channels = out_channels if stem_cell else in_channels
        self.act_1 = nn.ReLU()
        self.separable_1 = SeparableConv2d(
            in_channels, middle_channels, kernel_size, stride=stride, padding=padding)
        self.bn_sep_1 = nn.BatchNorm2d(middle_channels, eps=0.001)
        self.act_2 = nn.ReLU()
        self.separable_2 = SeparableConv2d(
            middle_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.act_1(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.act_2(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class ActConvBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=''):
        super(ActConvBn, self).__init__()
        self.act = nn.ReLU()
        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.act(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class FactorizedReduction(nn.Module):

    def __init__(self, in_channels, out_channels, padding=''):
        super(FactorizedReduction, self).__init__()
        self.act = nn.ReLU()
        self.path_1 = nn.Sequential(OrderedDict([
            ('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)),
            ('conv', create_conv2d(in_channels, out_channels // 2, kernel_size=1, padding=padding)),
        ]))
        self.path_2 = nn.Sequential(OrderedDict([
            ('pad', nn.ZeroPad2d((-1, 1, -1, 1))),  # shift
            ('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)),
            ('conv', create_conv2d(in_channels, out_channels // 2, kernel_size=1, padding=padding)),
        ]))
        self.final_path_bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.act(x)
        x_path1 = self.path_1(x)
        x_path2 = self.path_2(x)
        out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        return out


class CellBase(nn.Module):

    def cell_forward(self, x_left, x_right):
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_left = self.comb_iter_3_left(x_comb_iter_2)
        x_comb_iter_3_right = self.comb_iter_3_right(x_right)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_left)
        if self.comb_iter_4_right is not None:
            x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        else:
            x_comb_iter_4_right = x_right
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class CellStem0(CellBase):

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(CellStem0, self).__init__()
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, kernel_size=1, padding=pad_type)

        self.comb_iter_0_left = BranchSeparables(
            in_chs_left, out_chs_left, kernel_size=5, stride=2, stem_cell=True, padding=pad_type)
        self.comb_iter_0_right = nn.Sequential(OrderedDict([
            ('max_pool', create_pool2d('max', 3, stride=2, padding=pad_type)),
            ('conv', create_conv2d(in_chs_left, out_chs_left, kernel_size=1, padding=pad_type)),
            ('bn', nn.BatchNorm2d(out_chs_left, eps=0.001)),
        ]))

        self.comb_iter_1_left = BranchSeparables(
            out_chs_right, out_chs_right, kernel_size=7, stride=2, padding=pad_type)
        self.comb_iter_1_right = create_pool2d('max', 3, stride=2, padding=pad_type)

        self.comb_iter_2_left = BranchSeparables(
            out_chs_right, out_chs_right, kernel_size=5, stride=2, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(
            out_chs_right, out_chs_right, kernel_size=3, stride=2, padding=pad_type)

        self.comb_iter_3_left = BranchSeparables(
            out_chs_right, out_chs_right, kernel_size=3, padding=pad_type)
        self.comb_iter_3_right = create_pool2d('max', 3, stride=2, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(
            in_chs_right, out_chs_right, kernel_size=3, stride=2, stem_cell=True, padding=pad_type)
        self.comb_iter_4_right = ActConvBn(
            out_chs_right, out_chs_right, kernel_size=1, stride=2, padding=pad_type)

    def forward(self, x_left):
        x_right = self.conv_1x1(x_left)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class Cell(CellBase):

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type='',
                 is_reduction=False, match_prev_layer_dims=False):
        super(Cell, self).__init__()

        # If `is_reduction` is set to `True` stride 2 is used for
        # convolution and pooling layers to reduce the spatial size of
        # the output of a cell approximately by a factor of 2.
        stride = 2 if is_reduction else 1

        # If `match_prev_layer_dimensions` is set to `True`
        # `FactorizedReduction` is used to reduce the spatial size
        # of the left input of a cell approximately by a factor of 2.
        self.match_prev_layer_dimensions = match_prev_layer_dims
        if match_prev_layer_dims:
            self.conv_prev_1x1 = FactorizedReduction(in_chs_left, out_chs_left, padding=pad_type)
        else:
            self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, kernel_size=1, padding=pad_type)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, kernel_size=1, padding=pad_type)

        self.comb_iter_0_left = BranchSeparables(
            out_chs_left, out_chs_left, kernel_size=5, stride=stride, padding=pad_type)
        self.comb_iter_0_right = create_pool2d('max', 3, stride=stride, padding=pad_type)

        self.comb_iter_1_left = BranchSeparables(
            out_chs_right, out_chs_right, kernel_size=7, stride=stride, padding=pad_type)
        self.comb_iter_1_right = create_pool2d('max', 3, stride=stride, padding=pad_type)

        self.comb_iter_2_left = BranchSeparables(
            out_chs_right, out_chs_right, kernel_size=5, stride=stride, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(
            out_chs_right, out_chs_right, kernel_size=3, stride=stride, padding=pad_type)

        self.comb_iter_3_left = BranchSeparables(out_chs_right, out_chs_right, kernel_size=3)
        self.comb_iter_3_right = create_pool2d('max', 3, stride=stride, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(
            out_chs_left, out_chs_left, kernel_size=3, stride=stride, padding=pad_type)
        if is_reduction:
            self.comb_iter_4_right = ActConvBn(
                out_chs_right, out_chs_right, kernel_size=1, stride=stride, padding=pad_type)
        else:
            self.comb_iter_4_right = None

    def forward(self, x_left, x_right):
        x_left = self.conv_prev_1x1(x_left)
        x_right = self.conv_1x1(x_right)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class PNASNet5Large(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3, output_stride=32, drop_rate=0., global_pool='avg', pad_type=''):
        super(PNASNet5Large, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.num_features = 4320
        assert output_stride == 32

        self.conv_0 = ConvNormAct(
            in_chans, 96, kernel_size=3, stride=2, padding=0,
            norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.1), apply_act=False)

        self.cell_stem_0 = CellStem0(
            in_chs_left=96, out_chs_left=54, in_chs_right=96, out_chs_right=54, pad_type=pad_type)

        self.cell_stem_1 = Cell(
            in_chs_left=96, out_chs_left=108, in_chs_right=270, out_chs_right=108, pad_type=pad_type,
            match_prev_layer_dims=True, is_reduction=True)
        self.cell_0 = Cell(
            in_chs_left=270, out_chs_left=216, in_chs_right=540, out_chs_right=216, pad_type=pad_type,
            match_prev_layer_dims=True)
        self.cell_1 = Cell(
            in_chs_left=540, out_chs_left=216, in_chs_right=1080, out_chs_right=216, pad_type=pad_type)
        self.cell_2 = Cell(
            in_chs_left=1080, out_chs_left=216, in_chs_right=1080, out_chs_right=216, pad_type=pad_type)
        self.cell_3 = Cell(
            in_chs_left=1080, out_chs_left=216, in_chs_right=1080, out_chs_right=216, pad_type=pad_type)

        self.cell_4 = Cell(
            in_chs_left=1080, out_chs_left=432, in_chs_right=1080, out_chs_right=432, pad_type=pad_type,
            is_reduction=True)
        self.cell_5 = Cell(
            in_chs_left=1080, out_chs_left=432, in_chs_right=2160, out_chs_right=432, pad_type=pad_type,
            match_prev_layer_dims=True)
        self.cell_6 = Cell(
            in_chs_left=2160, out_chs_left=432, in_chs_right=2160, out_chs_right=432, pad_type=pad_type)
        self.cell_7 = Cell(
            in_chs_left=2160, out_chs_left=432, in_chs_right=2160, out_chs_right=432, pad_type=pad_type)

        self.cell_8 = Cell(
            in_chs_left=2160, out_chs_left=864, in_chs_right=2160, out_chs_right=864, pad_type=pad_type,
            is_reduction=True)
        self.cell_9 = Cell(
            in_chs_left=2160, out_chs_left=864, in_chs_right=4320, out_chs_right=864, pad_type=pad_type,
            match_prev_layer_dims=True)
        self.cell_10 = Cell(
            in_chs_left=4320, out_chs_left=864, in_chs_right=4320, out_chs_right=864, pad_type=pad_type)
        self.cell_11 = Cell(
            in_chs_left=4320, out_chs_left=864, in_chs_right=4320, out_chs_right=864, pad_type=pad_type)
        self.act = nn.ReLU()
        self.feature_info = [
            dict(num_chs=96, reduction=2, module='conv_0'),
            dict(num_chs=270, reduction=4, module='cell_stem_1.conv_1x1.act'),
            dict(num_chs=1080, reduction=8, module='cell_4.conv_1x1.act'),
            dict(num_chs=2160, reduction=16, module='cell_8.conv_1x1.act'),
            dict(num_chs=4320, reduction=32, module='act'),
        ]

        self.global_pool, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem=r'^conv_0|cell_stem_[01]', blocks=r'^cell_(\d+)')

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x_conv_0 = self.conv_0(x)
        x_stem_0 = self.cell_stem_0(x_conv_0)
        x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
        x = self.act(x_cell_11)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, self.drop_rate, training=self.training)
        return x if pre_logits else self.last_linear(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_pnasnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        PNASNet5Large, variant, pretrained,
        feature_cfg=dict(feature_cls='hook', no_rewrite=True),  # not possible to re-write this model
        **kwargs)


@register_model
def pnasnet5large(pretrained=False, **kwargs):
    r"""PNASNet-5 model architecture from the
    `"Progressive Neural Architecture Search"
    <https://arxiv.org/abs/1712.00559>`_ paper.
    """
    model_kwargs = dict(pad_type='same', **kwargs)
    return _create_pnasnet('pnasnet5large', pretrained, **model_kwargs)
