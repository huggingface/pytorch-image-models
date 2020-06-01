"""
TResNet: High Performance GPU-Dedicated Architecture
https://arxiv.org/pdf/2003.13630.pdf

Original model: https://github.com/mrT23/TResNet

"""
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import load_pretrained
from .layers import SpaceToDepthModule, AntiAliasDownsampleLayer, SelectAdaptivePool2d, InplaceAbn
from .registry import register_model

__all__ = ['tresnet_m', 'tresnet_l', 'tresnet_xl']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': (0, 0, 0), 'std': (1, 1, 1),
        'first_conv': 'body.conv1', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = {
    'tresnet_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/tresnet_m_80_8-dbc13962.pth'),
    'tresnet_l': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/tresnet_l_81_5-235b486c.pth'),
    'tresnet_xl': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/tresnet_xl_82_0-a2d51b00.pth'),
    'tresnet_m_448': _cfg(
        input_size=(3, 448, 448), pool_size=(14, 14),
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/tresnet_m_448-bc359d10.pth'),
    'tresnet_l_448': _cfg(
        input_size=(3, 448, 448), pool_size=(14, 14),
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/tresnet_l_448-940d0cd1.pth'),
    'tresnet_xl_448': _cfg(
        input_size=(3, 448, 448), pool_size=(14, 14),
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/tresnet_xl_448-8c1815de.pth')
}


class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)

    def feat_mult(self):
        return 1


class FastSEModule(nn.Module):

    def __init__(self, channels, reduction_channels, inplace=True):
        super(FastSEModule, self).__init__()
        self.avg_pool = FastGlobalAvgPool2d()
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=inplace)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, padding=0, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se2 = self.fc1(x_se)
        x_se2 = self.relu(x_se2)
        x_se = self.fc2(x_se2)
        x_se = self.activation(x_se)
        return x * x_se


def IABN2Float(module: nn.Module) -> nn.Module:
    """If `module` is IABN don't use half precision."""
    if isinstance(module, InplaceAbn):
        module.float()
    for child in module.children():
        IABN2Float(child)
    return module


def conv2d_iabn(ni, nf, stride, kernel_size=3, groups=1, act_layer="leaky_relu", act_param=1e-2):
    return nn.Sequential(
        nn.Conv2d(
            ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=False),
        InplaceAbn(nf, act_layer=act_layer, act_param=act_param)
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, aa_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_iabn(inplanes, planes, stride=1, act_param=1e-3)
        else:
            if aa_layer is None:
                self.conv1 = conv2d_iabn(inplanes, planes, stride=2, act_param=1e-3)
            else:
                self.conv1 = nn.Sequential(
                    conv2d_iabn(inplanes, planes, stride=1, act_param=1e-3),
                    aa_layer(channels=planes, filt_size=3, stride=2))

        self.conv2 = conv2d_iabn(planes, planes, stride=1, act_layer="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = FastSEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None:
            out = self.se(out)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True,
                 act_layer="leaky_relu", aa_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_iabn(
            inplanes, planes, kernel_size=1, stride=1, act_layer=act_layer, act_param=1e-3)
        if stride == 1:
            self.conv2 = conv2d_iabn(
                planes, planes, kernel_size=3, stride=1, act_layer=act_layer, act_param=1e-3)
        else:
            if aa_layer is None:
                self.conv2 = conv2d_iabn(
                    planes, planes, kernel_size=3, stride=2, act_layer=act_layer, act_param=1e-3)
            else:
                self.conv2 = nn.Sequential(
                    conv2d_iabn(planes, planes, kernel_size=3, stride=1, act_layer=act_layer, act_param=1e-3),
                    aa_layer(channels=planes, filt_size=3, stride=2))

        self.conv3 = conv2d_iabn(
            planes, planes * self.expansion, kernel_size=1, stride=1, act_layer="identity")

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = FastSEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)

        out = self.conv3(out)
        out = out + residual  # no inplace
        out = self.relu(out)

        return out


class TResNet(nn.Module):
    def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0, no_aa_jit=False,
                 global_pool='avg', drop_rate=0.):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(TResNet, self).__init__()

        # JIT layers
        space_to_depth = SpaceToDepthModule()
        aa_layer = partial(AntiAliasDownsampleLayer, no_jit=no_aa_jit)

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_iabn(in_chans * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(
            BasicBlock, self.planes, layers[0], stride=1, use_se=True, aa_layer=aa_layer)  # 56x56
        layer2 = self._make_layer(
            BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True, aa_layer=aa_layer)  # 28x28
        layer3 = self._make_layer(
            Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True, aa_layer=aa_layer)  # 14x14
        layer4 = self._make_layer(
            Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False, aa_layer=aa_layer)  # 7x7

        # body
        self.body = nn.Sequential(OrderedDict([
            ('SpaceToDepth', space_to_depth),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4)]))

        # head
        self.num_features = (self.planes * 8) * Bottleneck.expansion
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        self.head = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes))]))

        # model initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InplaceAbn):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
            if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, aa_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [conv2d_iabn(
                self.inplanes, planes * block.expansion, kernel_size=1, stride=1, act_layer="identity")]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, use_se=use_se, aa_layer=aa_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, use_se=use_se, aa_layer=aa_layer))
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        self.num_classes = num_classes
        self.head = None
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.head = nn.Sequential(OrderedDict([('fc', nn.Linear(num_features, num_classes))]))
        else:
            self.head = nn.Sequential(OrderedDict([('fc', nn.Identity())]))

    def forward_features(self, x):
        return self.body(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.head(x)
        return x


@register_model
def tresnet_m(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['tresnet_m']
    model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tresnet_l(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['tresnet_l']
    model = TResNet(
        layers=[4, 5, 18, 3], num_classes=num_classes, in_chans=in_chans, width_factor=1.2, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tresnet_xl(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['tresnet_xl']
    model = TResNet(
        layers=[4, 5, 24, 3], num_classes=num_classes, in_chans=in_chans, width_factor=1.3, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tresnet_m_448(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['tresnet_m_448']
    model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tresnet_l_448(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['tresnet_l_448']
    model = TResNet(
        layers=[4, 5, 18, 3], num_classes=num_classes, in_chans=in_chans, width_factor=1.2, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tresnet_xl_448(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['tresnet_xl_448']
    model = TResNet(
        layers=[4, 5, 24, 3], num_classes=num_classes, in_chans=in_chans, width_factor=1.3, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model
