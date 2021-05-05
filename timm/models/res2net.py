""" Res2Net and Res2NeXt
Adapted from Official Pytorch impl at: https://github.com/gasvn/Res2Net/
Paper: `Res2Net: A New Multi-scale Backbone Architecture` - https://arxiv.org/abs/1904.01169
"""
import math

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .registry import register_model
from .resnet import ResNet

__all__ = []


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
    'res2net50_26w_4s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_26w_4s-06e79181.pth'),
    'res2net50_48w_2s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_48w_2s-afed724a.pth'),
    'res2net50_14w_8s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_14w_8s-6527dddc.pth'),
    'res2net50_26w_6s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_26w_6s-19041792.pth'),
    'res2net50_26w_8s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_26w_8s-2c7c9f12.pth'),
    'res2net101_26w_4s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net101_26w_4s-02a759a1.pth'),
    'res2next50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2next50_4s-6ef7e7bf.pth'),
}


class Bottle2neck(nn.Module):
    """ Res2Net/Res2NeXT Bottleneck
    Adapted from https://github.com/gasvn/Res2Net/blob/master/res2net.py
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, base_width=26, scale=4, dilation=1, first_dilation=None,
                 act_layer=nn.ReLU, norm_layer=None, attn_layer=None, **_):
        super(Bottle2neck, self).__init__()
        self.scale = scale
        self.is_first = stride > 1 or downsample is not None
        self.num_scales = max(1, scale - 1)
        width = int(math.floor(planes * (base_width / 64.0))) * cardinality
        self.width = width
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width * scale)

        convs = []
        bns = []
        for i in range(self.num_scales):
            convs.append(nn.Conv2d(
                width, width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, bias=False))
            bns.append(norm_layer(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        if self.is_first:
            # FIXME this should probably have count_include_pad=False, but hurts original weights
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        else:
            self.pool = None

        self.conv3 = nn.Conv2d(width * scale, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.se = attn_layer(outplanes) if attn_layer is not None else None

        self.relu = act_layer(inplace=True)
        self.downsample = downsample

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        spo = []
        sp = spx[0]  # redundant, for torchscript
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 0 or self.is_first:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(sp)
            sp = self.relu(sp)
            spo.append(sp)
        if self.scale > 1:
            if self.pool is not None:
                # self.is_first == True, None check for torchscript
                spo.append(self.pool(spx[-1]))
            else:
                spo.append(spx[-1])
        out = torch.cat(spo, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.relu(out)

        return out


def _create_res2net(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ResNet, variant, pretrained,
        default_cfg=default_cfgs[variant],
        **kwargs)


@register_model
def res2net50_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 26w4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], base_width=26, block_args=dict(scale=4), **kwargs)
    return _create_res2net('res2net50_26w_4s', pretrained, **model_args)


@register_model
def res2net101_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-101 26w4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 23, 3], base_width=26, block_args=dict(scale=4), **kwargs)
    return _create_res2net('res2net101_26w_4s', pretrained, **model_args)


@register_model
def res2net50_26w_6s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 26w6s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], base_width=26, block_args=dict(scale=6), **kwargs)
    return _create_res2net('res2net50_26w_6s', pretrained, **model_args)


@register_model
def res2net50_26w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 26w8s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], base_width=26, block_args=dict(scale=8), **kwargs)
    return _create_res2net('res2net50_26w_8s', pretrained, **model_args)


@register_model
def res2net50_48w_2s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 48w2s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], base_width=48, block_args=dict(scale=2), **kwargs)
    return _create_res2net('res2net50_48w_2s', pretrained, **model_args)


@register_model
def res2net50_14w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 14w8s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], base_width=14, block_args=dict(scale=8), **kwargs)
    return _create_res2net('res2net50_14w_8s', pretrained, **model_args)


@register_model
def res2next50(pretrained=False, **kwargs):
    """Construct Res2NeXt-50 4s
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], base_width=4, cardinality=8, block_args=dict(scale=4), **kwargs)
    return _create_res2net('res2next50', pretrained, **model_args)
