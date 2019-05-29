"""Pytorch ResNet implementation w/ tweaks
This file is a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNext additions added by Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.helpers import load_pretrained
from models.adaptive_avgmax_pool import SelectAdaptivePool2d
from data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_64x4d', 'resnext152_32x4d']
__all__ = ['ResNet'] + _models


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'resnet18': _cfg(url='https://download.pytorch.org/models/resnet18-5c106cde.pth'),
    'resnet34': _cfg(url='https://download.pytorch.org/models/resnet34-333f7ec4.pth'),
    'resnet50': _cfg(url='https://download.pytorch.org/models/resnet50-19c8e357.pth'),
    'resnet101': _cfg(url='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'),
    'resnet152': _cfg(url='https://download.pytorch.org/models/resnet152-b121ed2d.pth'),
    'resnext50_32x4d': _cfg(url='https://www.dropbox.com/s/yxci33lfew51p6a/resnext50_32x4d-068914d1.pth?dl=1',
                            interpolation='bicubic'),
    'resnext101_32x4d': _cfg(url=''),
    'resnext101_64x4d': _cfg(url=''),
    'resnext152_32x4d': _cfg(url=''),
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, base_width=64, drop_rate=0.0):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.drop_rate > 0.:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, base_width=64, drop_rate=0.0):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.drop_rate > 0.:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, in_chans=3,
                 cardinality=1, base_width=64,
                 drop_rate=0.0, block_drop_rate=0.0,
                 global_pool='avg'):
        self.num_classes = num_classes
        self.inplanes = 64
        self.cardinality = cardinality
        self.base_width = base_width
        self.drop_rate = drop_rate
        self.expansion = block.expansion
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], drop_rate=block_drop_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, drop_rate=block_drop_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, drop_rate=block_drop_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, drop_rate=block_drop_rate)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_features = 512 * block.expansion
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def _make_layer(self, block, planes, blocks, stride=1, drop_rate=0.):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.cardinality, self.base_width, drop_rate)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality=self.cardinality, base_width=self.base_width))

        return nn.Sequential(*layers)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

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


def resnet18(num_classes=1000, in_chans=3, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    default_cfg = default_cfgs['resnet18']
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def resnet34(num_classes=1000, in_chans=3, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    default_cfg = default_cfgs['resnet34']
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def resnet50(num_classes=1000, in_chans=3, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    default_cfg = default_cfgs['resnet50']
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def resnet101(num_classes=1000, in_chans=3, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    default_cfg = default_cfgs['resnet101']
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def resnet152(num_classes=1000, in_chans=3, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    """
    default_cfg = default_cfgs['resnet152']
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def resnext50_32x4d(num_classes=1000, in_chans=3, pretrained=False, **kwargs):
    """Constructs a ResNeXt50-32x4d model.
    """
    default_cfg = default_cfgs['resnext50_32x4d']
    model = ResNet(
        Bottleneck, [3, 4, 6, 3], cardinality=32, base_width=4,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def resnext101_32x4d(num_classes=1000, in_chans=3, pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 model.
    """
    default_cfg = default_cfgs['resnext101_32x4d']
    model = ResNet(
        Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=4,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def resnext101_64x4d(num_classes=1000, in_chans=3, pretrained=False, **kwargs):
    """Constructs a ResNeXt101-64x4d model.
    """
    default_cfg = default_cfgs['resnext101_32x4d']
    model = ResNet(
        Bottleneck, [3, 4, 23, 3], cardinality=64, base_width=4,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def resnext152_32x4d(num_classes=1000, in_chans=3, pretrained=False, **kwargs):
    """Constructs a ResNeXt152-32x4d model.
    """
    default_cfg = default_cfgs['resnext152_32x4d']
    model = ResNet(
        Bottleneck, [3, 8, 36, 3], cardinality=32, base_width=4,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model
