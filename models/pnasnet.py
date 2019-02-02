from __future__ import print_function, division, absolute_import
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


pretrained_settings = {
    'pnasnet5large': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/pnasnet5large-bf079911.pth',
            'input_space': 'RGB',
            'input_size': [3, 331, 331],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/pnasnet5large-bf079911.pth',
            'input_space': 'RGB',
            'input_size': [3, 331, 331],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}


class MaxPool(nn.Module):

    def __init__(self, kernel_size, stride=1, padding=1, zero_pad=False):
        super(MaxPool, self).__init__()
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0)) if zero_pad else None
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        if self.zero_pad:
            x = self.zero_pad(x)
        x = self.pool(x)
        if self.zero_pad:
            x = x[:, :, 1:, 1:]
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel_size, dw_stride,
                 dw_padding):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels,
                                          kernel_size=dw_kernel_size,
                                          stride=dw_stride, padding=dw_padding,
                                          groups=in_channels, bias=False)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 stem_cell=False, zero_pad=False):
        super(BranchSeparables, self).__init__()
        padding = kernel_size // 2
        middle_channels = out_channels if stem_cell else in_channels
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0)) if zero_pad else None
        self.relu_1 = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, middle_channels,
                                           kernel_size, dw_stride=stride,
                                           dw_padding=padding)
        self.bn_sep_1 = nn.BatchNorm2d(middle_channels, eps=0.001)
        self.relu_2 = nn.ReLU()
        self.separable_2 = SeparableConv2d(middle_channels, out_channels,
                                           kernel_size, dw_stride=1,
                                           dw_padding=padding)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.relu_1(x)
        if self.zero_pad:
            x = self.zero_pad(x)
        x = self.separable_1(x)
        if self.zero_pad:
            x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu_2(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class ReluConvBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ReluConvBn, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class FactorizedReduction(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FactorizedReduction, self).__init__()
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential(OrderedDict([
            ('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)),
            ('conv', nn.Conv2d(in_channels, out_channels // 2,
                               kernel_size=1, bias=False)),
        ]))
        self.path_2 = nn.Sequential(OrderedDict([
            ('pad', nn.ZeroPad2d((0, 1, 0, 1))),
            ('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)),
            ('conv', nn.Conv2d(in_channels, out_channels // 2,
                               kernel_size=1, bias=False)),
        ]))
        self.final_path_bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.relu(x)

        x_path1 = self.path_1(x)

        x_path2 = self.path_2.pad(x)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)

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
        if self.comb_iter_4_right:
            x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        else:
            x_comb_iter_4_right = x_right
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat(
            [x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
             x_comb_iter_4], 1)
        return x_out


class CellStem0(CellBase):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right,
                 out_channels_right):
        super(CellStem0, self).__init__()
        self.conv_1x1 = ReluConvBn(in_channels_right, out_channels_right,
                                   kernel_size=1)
        self.comb_iter_0_left = BranchSeparables(in_channels_left,
                                                 out_channels_left,
                                                 kernel_size=5, stride=2,
                                                 stem_cell=True)
        self.comb_iter_0_right = nn.Sequential(OrderedDict([
            ('max_pool', MaxPool(3, stride=2)),
            ('conv', nn.Conv2d(in_channels_left, out_channels_left,
                               kernel_size=1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels_left, eps=0.001)),
        ]))
        self.comb_iter_1_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=7, stride=2)
        self.comb_iter_1_right = MaxPool(3, stride=2)
        self.comb_iter_2_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=5, stride=2)
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
                                                  out_channels_right,
                                                  kernel_size=3, stride=2)
        self.comb_iter_3_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=3)
        self.comb_iter_3_right = MaxPool(3, stride=2)
        self.comb_iter_4_left = BranchSeparables(in_channels_right,
                                                 out_channels_right,
                                                 kernel_size=3, stride=2,
                                                 stem_cell=True)
        self.comb_iter_4_right = ReluConvBn(out_channels_right,
                                            out_channels_right,
                                            kernel_size=1, stride=2)

    def forward(self, x_left):
        x_right = self.conv_1x1(x_left)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class Cell(CellBase):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right,
                 out_channels_right, is_reduction=False, zero_pad=False,
                 match_prev_layer_dimensions=False):
        super(Cell, self).__init__()

        # If `is_reduction` is set to `True` stride 2 is used for
        # convolutional and pooling layers to reduce the spatial size of
        # the output of a cell approximately by a factor of 2.
        stride = 2 if is_reduction else 1

        # If `match_prev_layer_dimensions` is set to `True`
        # `FactorizedReduction` is used to reduce the spatial size
        # of the left input of a cell approximately by a factor of 2.
        self.match_prev_layer_dimensions = match_prev_layer_dimensions
        if match_prev_layer_dimensions:
            self.conv_prev_1x1 = FactorizedReduction(in_channels_left,
                                                     out_channels_left)
        else:
            self.conv_prev_1x1 = ReluConvBn(in_channels_left,
                                            out_channels_left, kernel_size=1)

        self.conv_1x1 = ReluConvBn(in_channels_right, out_channels_right,
                                   kernel_size=1)
        self.comb_iter_0_left = BranchSeparables(out_channels_left,
                                                 out_channels_left,
                                                 kernel_size=5, stride=stride,
                                                 zero_pad=zero_pad)
        self.comb_iter_0_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_1_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=7, stride=stride,
                                                 zero_pad=zero_pad)
        self.comb_iter_1_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_2_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=5, stride=stride,
                                                 zero_pad=zero_pad)
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
                                                  out_channels_right,
                                                  kernel_size=3, stride=stride,
                                                  zero_pad=zero_pad)
        self.comb_iter_3_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=3)
        self.comb_iter_3_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_4_left = BranchSeparables(out_channels_left,
                                                 out_channels_left,
                                                 kernel_size=3, stride=stride,
                                                 zero_pad=zero_pad)
        if is_reduction:
            self.comb_iter_4_right = ReluConvBn(out_channels_right,
                                                out_channels_right,
                                                kernel_size=1, stride=stride)
        else:
            self.comb_iter_4_right = None

    def forward(self, x_left, x_right):
        x_left = self.conv_prev_1x1(x_left)
        x_right = self.conv_1x1(x_right)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class PNASNet5Large(nn.Module):
    def __init__(self, num_classes=1001):
        super(PNASNet5Large, self).__init__()
        self.num_classes = num_classes
        self.conv_0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 96, kernel_size=3, stride=2, bias=False)),
            ('bn', nn.BatchNorm2d(96, eps=0.001))
        ]))
        self.cell_stem_0 = CellStem0(in_channels_left=96, out_channels_left=54,
                                     in_channels_right=96,
                                     out_channels_right=54)
        self.cell_stem_1 = Cell(in_channels_left=96, out_channels_left=108,
                                in_channels_right=270, out_channels_right=108,
                                match_prev_layer_dimensions=True,
                                is_reduction=True)
        self.cell_0 = Cell(in_channels_left=270, out_channels_left=216,
                           in_channels_right=540, out_channels_right=216,
                           match_prev_layer_dimensions=True)
        self.cell_1 = Cell(in_channels_left=540, out_channels_left=216,
                           in_channels_right=1080, out_channels_right=216)
        self.cell_2 = Cell(in_channels_left=1080, out_channels_left=216,
                           in_channels_right=1080, out_channels_right=216)
        self.cell_3 = Cell(in_channels_left=1080, out_channels_left=216,
                           in_channels_right=1080, out_channels_right=216)
        self.cell_4 = Cell(in_channels_left=1080, out_channels_left=432,
                           in_channels_right=1080, out_channels_right=432,
                           is_reduction=True, zero_pad=True)
        self.cell_5 = Cell(in_channels_left=1080, out_channels_left=432,
                           in_channels_right=2160, out_channels_right=432,
                           match_prev_layer_dimensions=True)
        self.cell_6 = Cell(in_channels_left=2160, out_channels_left=432,
                           in_channels_right=2160, out_channels_right=432)
        self.cell_7 = Cell(in_channels_left=2160, out_channels_left=432,
                           in_channels_right=2160, out_channels_right=432)
        self.cell_8 = Cell(in_channels_left=2160, out_channels_left=864,
                           in_channels_right=2160, out_channels_right=864,
                           is_reduction=True)
        self.cell_9 = Cell(in_channels_left=2160, out_channels_left=864,
                           in_channels_right=4320, out_channels_right=864,
                           match_prev_layer_dimensions=True)
        self.cell_10 = Cell(in_channels_left=4320, out_channels_left=864,
                            in_channels_right=4320, out_channels_right=864)
        self.cell_11 = Cell(in_channels_left=4320, out_channels_left=864,
                            in_channels_right=4320, out_channels_right=864)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.last_linear = nn.Linear(4320, num_classes)

    def features(self, x):
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
        return x_cell_11

    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def pnasnet5large(num_classes=1001, pretrained='imagenet'):
    r"""PNASNet-5 model architecture from the
    `"Progressive Neural Architecture Search"
    <https://arxiv.org/abs/1712.00559>`_ paper.
    """
    if pretrained:
        settings = pretrained_settings['pnasnet5large'][pretrained]
        assert num_classes == settings[
            'num_classes'], 'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = PNASNet5Large(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(model.last_linear.in_features, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']

        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = PNASNet5Large(num_classes=num_classes)
    return model
