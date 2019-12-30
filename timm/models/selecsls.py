"""PyTorch SelecSLS Net example for ImageNet Classification
License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode)
Author: Dushyant Mehta (@mehtadushy)

SelecSLS (core) Network Architecture as proposed in "XNect: Real-time Multi-person 3D
Human Pose Estimation with a Single RGB Camera, Mehta et al."
https://arxiv.org/abs/1907.00837

Based on ResNet implementation in https://github.com/rwightman/pytorch-image-models
and SelecSLS Net implementation in https://github.com/mehtadushy/SelecSLS-Pytorch
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .helpers import load_pretrained
from .adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

__all__ = ['SelecSLS']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (3, 3),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'selecsls42': _cfg(
        url='',
        interpolation='bicubic'),
    'selecsls42b': _cfg(
        url='http://gvv.mpi-inf.mpg.de/projects/XNect/assets/models/SelecSLS42_B.pth',
        interpolation='bicubic'),
    'selecsls60': _cfg(
        url='',
        interpolation='bicubic'),
    'selecsls60b': _cfg(
        url='http://gvv.mpi-inf.mpg.de/projects/XNect/assets/models/SelecSLS60_B.pth',
        interpolation='bicubic'),
    'selecsls84': _cfg(
        url='',
        interpolation='bicubic'),
}


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class SelecSLSBlock(nn.Module):
    def __init__(self, inp, skip, k, oup, is_first, stride):
        super(SelecSLSBlock, self).__init__()
        self.stride = stride
        self.is_first = is_first
        assert stride in [1, 2]

        # Process input with 4 conv blocks with the same number of input and output channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, k, 3, stride, 1, groups=1, bias=False, dilation=1),
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(k, k, 1, 1, 0, groups=1, bias=False, dilation=1),
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(k, k // 2, 3, 1, 1, groups=1, bias=False, dilation=1),
            nn.BatchNorm2d(k // 2),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(k // 2, k, 1, 1, 0, groups=1, bias=False, dilation=1),
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(k, k // 2, 3, 1, 1, groups=1, bias=False, dilation=1),
            nn.BatchNorm2d(k // 2),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(2 * k + (0 if is_first else skip), oup, 1, 1, 0, groups=1, bias=False, dilation=1),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        assert isinstance(x, list)
        assert len(x) in [1, 2]

        d1 = self.conv1(x[0])
        d2 = self.conv3(self.conv2(d1))
        d3 = self.conv5(self.conv4(d2))
        if self.is_first:
            out = self.conv6(torch.cat([d1, d2, d3], 1))
            return [out, out]
        else:
            return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]


class SelecSLS(nn.Module):
    """SelecSLS42 / SelecSLS60 / SelecSLS84

    Parameters
    ----------
    cfg : network config
       String indicating the network config
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    """

    def __init__(self, cfg='selecsls60', num_classes=1000, in_chans=3,
                 drop_rate=0.0, global_pool='avg'):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(SelecSLS, self).__init__()

        self.stem = conv_bn(in_chans, 32, 2)
        # Core Network
        self.features = []
        if cfg == 'selecsls42':
            self.block = SelecSLSBlock
            # Define configuration of the network after the initial neck
            self.selecsls_config = [
                # inp,skip, k, oup, is_first, stride
                [32, 0, 64, 64, True, 2],
                [64, 64, 64, 128, False, 1],
                [128, 0, 144, 144, True, 2],
                [144, 144, 144, 288, False, 1],
                [288, 0, 304, 304, True, 2],
                [304, 304, 304, 480, False, 1],
            ]
            # Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                conv_bn(480, 960, 2),
                conv_bn(960, 1024, 1),
                conv_bn(1024, 1024, 2),
                conv_1x1_bn(1024, 1280),
            )
            self.num_features = 1280
        elif cfg == 'selecsls42b':
            self.block = SelecSLSBlock
            # Define configuration of the network after the initial neck
            self.selecsls_config = [
                # inp,skip, k, oup, is_first, stride
                [32, 0, 64, 64, True, 2],
                [64, 64, 64, 128, False, 1],
                [128, 0, 144, 144, True, 2],
                [144, 144, 144, 288, False, 1],
                [288, 0, 304, 304, True, 2],
                [304, 304, 304, 480, False, 1],
            ]
            # Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                conv_bn(480, 960, 2),
                conv_bn(960, 1024, 1),
                conv_bn(1024, 1280, 2),
                conv_1x1_bn(1280, 1024),
            )
            self.num_features = 1024
        elif cfg == 'selecsls60':
            self.block = SelecSLSBlock
            # Define configuration of the network after the initial neck
            self.selecsls_config = [
                # inp,skip, k, oup, is_first, stride
                [32, 0, 64, 64, True, 2],
                [64, 64, 64, 128, False, 1],
                [128, 0, 128, 128, True, 2],
                [128, 128, 128, 128, False, 1],
                [128, 128, 128, 288, False, 1],
                [288, 0, 288, 288, True, 2],
                [288, 288, 288, 288, False, 1],
                [288, 288, 288, 288, False, 1],
                [288, 288, 288, 416, False, 1],
            ]
            # Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                conv_bn(416, 756, 2),
                conv_bn(756, 1024, 1),
                conv_bn(1024, 1024, 2),
                conv_1x1_bn(1024, 1280),
            )
            self.num_features = 1280
        elif cfg == 'selecsls60b':
            self.block = SelecSLSBlock
            # Define configuration of the network after the initial neck
            self.selecsls_config = [
                # inp,skip, k, oup, is_first, stride
                [32, 0, 64, 64, True, 2],
                [64, 64, 64, 128, False, 1],
                [128, 0, 128, 128, True, 2],
                [128, 128, 128, 128, False, 1],
                [128, 128, 128, 288, False, 1],
                [288, 0, 288, 288, True, 2],
                [288, 288, 288, 288, False, 1],
                [288, 288, 288, 288, False, 1],
                [288, 288, 288, 416, False, 1],
            ]
            # Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                conv_bn(416, 756, 2),
                conv_bn(756, 1024, 1),
                conv_bn(1024, 1280, 2),
                conv_1x1_bn(1280, 1024),
            )
            self.num_features = 1024
        elif cfg == 'selecsls84':
            self.block = SelecSLSBlock
            # Define configuration of the network after the initial neck
            self.selecsls_config = [
                # inp,skip, k, oup, is_first, stride
                [32, 0, 64, 64, True, 2],
                [64, 64, 64, 144, False, 1],
                [144, 0, 144, 144, True, 2],
                [144, 144, 144, 144, False, 1],
                [144, 144, 144, 144, False, 1],
                [144, 144, 144, 144, False, 1],
                [144, 144, 144, 304, False, 1],
                [304, 0, 304, 304, True, 2],
                [304, 304, 304, 304, False, 1],
                [304, 304, 304, 304, False, 1],
                [304, 304, 304, 304, False, 1],
                [304, 304, 304, 304, False, 1],
                [304, 304, 304, 512, False, 1],
            ]
            # Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                conv_bn(512, 960, 2),
                conv_bn(960, 1024, 1),
                conv_bn(1024, 1024, 2),
                conv_1x1_bn(1024, 1280),
            )
            self.num_features = 1280
        else:
            raise ValueError('Invalid net configuration ' + cfg + ' !!!')

        for inp, skip, k, oup, is_first, stride in self.selecsls_config:
            self.features.append(self.block(inp, skip, k, oup, is_first, stride))
        self.features = nn.Sequential(*self.features)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        del self.fc
        if num_classes:
            self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)
        else:
            self.fc = None

    def forward_features(self, x, pool=True):
        x = self.stem(x)
        x = self.features([x])
        x = self.head(x[0])

        if pool:
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


@register_model
def selecsls42(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a SelecSLS42 model.
    """
    default_cfg = default_cfgs['selecsls42']
    model = SelecSLS(
        cfg='selecsls42', num_classes=1000, in_chans=3, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def selecsls42b(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a SelecSLS42_B model.
    """
    default_cfg = default_cfgs['selecsls42b']
    model = SelecSLS(
        cfg='selecsls42b', num_classes=1000, in_chans=3, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def selecsls60(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a SelecSLS60 model.
    """
    default_cfg = default_cfgs['selecsls60']
    model = SelecSLS(
        cfg='selecsls60', num_classes=1000, in_chans=3, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def selecsls60b(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a SelecSLS60_B model.
    """
    default_cfg = default_cfgs['selecsls60b']
    model = SelecSLS(
        cfg='selecsls60b', num_classes=1000, in_chans=3, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def selecsls84(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a SelecSLS84 model.
    """
    default_cfg = default_cfgs['selecsls84']
    model = SelecSLS(
        cfg='selecsls84', num_classes=1000, in_chans=3, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model
