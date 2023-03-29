"""Pytorch impl of Gluon Xception
This is a port of the Gluon Xception code and weights, itself ported from a PyTorch DeepLab impl.

Gluon model: (https://gluon-cv.mxnet.io/_modules/gluoncv/model_zoo/xception.html)
Original PyTorch DeepLab impl: https://github.com/jfzhang95/pytorch-deeplab-xception

Hacked together by / Copyright 2020 Ross Wightman
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import create_classifier, get_padding
from ._builder import build_model_with_cfg
from ._registry import register_model

__all__ = ['Xception65']

default_cfgs = {
    'gluon_xception65': {
        'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_xception-7015a15c.pth',
        'input_size': (3, 299, 299),
        'crop_pct': 0.903,
        'pool_size': (10, 10),
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'num_classes': 1000,
        'first_conv': 'conv1',
        'classifier': 'fc'
        # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
    },
}

""" PADDING NOTES
The original PyTorch and Gluon impl of these models dutifully reproduced the 
aligned padding added to Tensorflow models for Deeplab. This padding was compensating
for  Tensorflow 'SAME' padding. PyTorch symmetric padding behaves the way we'd want it to. 
"""


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        # depthwise convolution
        padding = get_padding(kernel_size, stride, dilation)
        self.conv_dw = nn.Conv2d(
            inplanes, inplanes, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(num_features=inplanes)
        # pointwise convolution
        self.conv_pw = nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn(x)
        x = self.conv_pw(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, start_with_relu=True, norm_layer=None):
        super(Block, self).__init__()
        if isinstance(planes, (list, tuple)):
            assert len(planes) == 3
        else:
            planes = (planes,) * 3
        outplanes = planes[-1]

        if outplanes != inplanes or stride != 1:
            self.skip = nn.Sequential()
            self.skip.add_module('conv1', nn.Conv2d(
                inplanes, outplanes, 1, stride=stride, bias=False)),
            self.skip.add_module('bn1', norm_layer(num_features=outplanes))
        else:
            self.skip = None

        rep = OrderedDict()
        for i in range(3):
            rep['act%d' % (i + 1)] = nn.ReLU(inplace=True)
            rep['conv%d' % (i + 1)] = SeparableConv2d(
                inplanes, planes[i], 3, stride=stride if i == 2 else 1, dilation=dilation, norm_layer=norm_layer)
            rep['bn%d' % (i + 1)] = norm_layer(planes[i])
            inplanes = planes[i]

        if not start_with_relu:
            del rep['act1']
        else:
            rep['act1'] = nn.ReLU(inplace=False)
        self.rep = nn.Sequential(rep)

    def forward(self, x):
        skip = x
        if self.skip is not None:
            skip = self.skip(skip)
        x = self.rep(x) + skip
        return x


class Xception65(nn.Module):
    """Modified Aligned Xception.

    NOTE: only the 65 layer version is included here, the 71 layer variant
    was not correct and had no pretrained weights
    """

    def __init__(self, num_classes=1000, in_chans=3, output_stride=32, norm_layer=nn.BatchNorm2d,
                 drop_rate=0., global_pool='avg'):
        super(Xception65, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_dilation = 1
            exit_dilation = (1, 1)
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_dilation = 1
            exit_dilation = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_dilation = 2
            exit_dilation = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(num_features=32)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(num_features=64)
        self.act2 = nn.ReLU(inplace=True)

        self.block1 = Block(64, 128, stride=2, start_with_relu=False, norm_layer=norm_layer)
        self.block1_act = nn.ReLU(inplace=True)
        self.block2 = Block(128, 256, stride=2, start_with_relu=False, norm_layer=norm_layer)
        self.block3 = Block(256, 728, stride=entry_block3_stride, norm_layer=norm_layer)

        # Middle flow
        self.mid = nn.Sequential(OrderedDict([('block%d' % i, Block(
            728, 728, stride=1, dilation=middle_dilation, norm_layer=norm_layer)) for i in range(4, 20)]))

        # Exit flow
        self.block20 = Block(
            728, (728, 1024, 1024), stride=exit_block20_stride, dilation=exit_dilation[0], norm_layer=norm_layer)
        self.block20_act = nn.ReLU(inplace=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_dilation[1], norm_layer=norm_layer)
        self.bn3 = norm_layer(num_features=1536)
        self.act3 = nn.ReLU(inplace=True)

        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_dilation[1], norm_layer=norm_layer)
        self.bn4 = norm_layer(num_features=1536)
        self.act4 = nn.ReLU(inplace=True)

        self.num_features = 2048
        self.conv5 = SeparableConv2d(
            1536, self.num_features, 3, stride=1, dilation=exit_dilation[1], norm_layer=norm_layer)
        self.bn5 = norm_layer(num_features=self.num_features)
        self.act5 = nn.ReLU(inplace=True)
        self.feature_info = [
            dict(num_chs=64, reduction=2, module='act2'),
            dict(num_chs=128, reduction=4, module='block1_act'),
            dict(num_chs=256, reduction=8, module='block3.rep.act1'),
            dict(num_chs=728, reduction=16, module='block20.rep.act1'),
            dict(num_chs=2048, reduction=32, module='act5'),
        ]

        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^conv[12]|bn[12]',
            blocks=[
                (r'^mid\.block(\d+)', None),
                (r'^block(\d+)', None),
                (r'^conv[345]|bn[345]', (99,)),
            ],
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, "gradient checkpointing not supported"

    @torch.jit.ignore
    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.block1(x)
        x = self.block1_act(x)
        # c1 = x
        x = self.block2(x)
        # c2 = x
        x = self.block3(x)

        # Middle flow
        x = self.mid(x)
        # c3 = x

        # Exit flow
        x = self.block20(x)
        x = self.block20_act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)
        return x

    def forward_head(self, x):
        x = self.global_pool(x)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_gluon_xception(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        Xception65, variant, pretrained,
        feature_cfg=dict(feature_cls='hook'),
        **kwargs)


@register_model
def gluon_xception65(pretrained=False, **kwargs):
    """ Modified Aligned Xception-65
    """
    return _create_gluon_xception('gluon_xception65', pretrained, **kwargs)
