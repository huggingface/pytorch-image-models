"""Pytorch impl of Aligned Xception 41, 65, 71

This is a correct, from scratch impl of Aligned Xception (Deeplab) models compatible with TF weights at
https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md

Hacked together by / Copyright 2020 Ross Wightman
"""
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .helpers import build_model_with_cfg
from .layers import ClassifierHead, ConvBnAct, create_conv2d
from .layers.helpers import to_3tuple
from .registry import register_model

__all__ = ['XceptionAligned']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 299, 299), 'pool_size': (10, 10),
        'crop_pct': 0.903, 'interpolation': 'bicubic',
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'stem.0.conv', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = dict(
    xception41=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_xception_41-e6439c97.pth'),
    xception65=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_xception_65-c9ae96e8.pth'),
    xception71=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_xception_71-8eec7df1.pth'),
)


class SeparableConv2d(nn.Module):
    def __init__(
            self, inplanes, planes, kernel_size=3, stride=1, dilation=1, padding='',
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(SeparableConv2d, self).__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self.kernel_size = kernel_size
        self.dilation = dilation

        # depthwise convolution
        self.conv_dw = create_conv2d(
            inplanes, inplanes, kernel_size, stride=stride,
            padding=padding, dilation=dilation, depthwise=True)
        self.bn_dw = norm_layer(inplanes, **norm_kwargs)
        if act_layer is not None:
            self.act_dw = act_layer(inplace=True)
        else:
            self.act_dw = None

        # pointwise convolution
        self.conv_pw = create_conv2d(inplanes, planes, kernel_size=1)
        self.bn_pw = norm_layer(planes, **norm_kwargs)
        if act_layer is not None:
            self.act_pw = act_layer(inplace=True)
        else:
            self.act_pw = None

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        if self.act_dw is not None:
            x = self.act_dw(x)
        x = self.conv_pw(x)
        x = self.bn_pw(x)
        if self.act_pw is not None:
            x = self.act_pw(x)
        return x


class XceptionModule(nn.Module):
    def __init__(
            self, in_chs, out_chs, stride=1, dilation=1, pad_type='',
            start_with_relu=True, no_skip=False, act_layer=nn.ReLU, norm_layer=None, norm_kwargs=None):
        super(XceptionModule, self).__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        out_chs = to_3tuple(out_chs)
        self.in_channels = in_chs
        self.out_channels = out_chs[-1]
        self.no_skip = no_skip
        if not no_skip and (self.out_channels != self.in_channels or stride != 1):
            self.shortcut = ConvBnAct(
                in_chs, self.out_channels, 1, stride=stride,
                norm_layer=norm_layer, norm_kwargs=norm_kwargs, act_layer=None)
        else:
            self.shortcut = None

        separable_act_layer = None if start_with_relu else act_layer
        self.stack = nn.Sequential()
        for i in range(3):
            if start_with_relu:
                self.stack.add_module(f'act{i + 1}', nn.ReLU(inplace=i > 0))
            self.stack.add_module(f'conv{i + 1}', SeparableConv2d(
                in_chs, out_chs[i], 3, stride=stride if i == 2 else 1, dilation=dilation, padding=pad_type,
                act_layer=separable_act_layer, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            in_chs = out_chs[i]

    def forward(self, x):
        skip = x
        x = self.stack(x)
        if self.shortcut is not None:
            skip = self.shortcut(skip)
        if not self.no_skip:
            x = x + skip
        return x


class XceptionAligned(nn.Module):
    """Modified Aligned Xception
    """

    def __init__(self, block_cfg, num_classes=1000, in_chans=3, output_stride=32,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_rate=0., global_pool='avg'):
        super(XceptionAligned, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}

        layer_args = dict(act_layer=act_layer, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.stem = nn.Sequential(*[
            ConvBnAct(in_chans, 32, kernel_size=3, stride=2, **layer_args),
            ConvBnAct(32, 64, kernel_size=3, stride=1, **layer_args)
        ])

        curr_dilation = 1
        curr_stride = 2
        self.feature_info = []
        self.blocks = nn.Sequential()
        for i, b in enumerate(block_cfg):
            b['dilation'] = curr_dilation
            if b['stride'] > 1:
                self.feature_info += [dict(
                    num_chs=to_3tuple(b['out_chs'])[-2], reduction=curr_stride, module=f'blocks.{i}.stack.act3')]
                next_stride = curr_stride * b['stride']
                if next_stride > output_stride:
                    curr_dilation *= b['stride']
                    b['stride'] = 1
                else:
                    curr_stride = next_stride
            self.blocks.add_module(str(i), XceptionModule(**b, **layer_args))
            self.num_features = self.blocks[-1].out_channels

        self.feature_info += [dict(
            num_chs=self.num_features, reduction=curr_stride, module='blocks.' + str(len(self.blocks) - 1))]

        self.head = ClassifierHead(
            in_chs=self.num_features, num_classes=num_classes, pool_type=global_pool, drop_rate=drop_rate)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _xception(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        XceptionAligned, variant, pretrained, default_cfg=default_cfgs[variant],
        feature_cfg=dict(flatten_sequential=True, feature_cls='hook'), **kwargs)


@register_model
def xception41(pretrained=False, **kwargs):
    """ Modified Aligned Xception-41
    """
    block_cfg = [
        # entry flow
        dict(in_chs=64, out_chs=128, stride=2),
        dict(in_chs=128, out_chs=256, stride=2),
        dict(in_chs=256, out_chs=728, stride=2),
        # middle flow
        *([dict(in_chs=728, out_chs=728, stride=1)] * 8),
        # exit flow
        dict(in_chs=728, out_chs=(728, 1024, 1024), stride=2),
        dict(in_chs=1024, out_chs=(1536, 1536, 2048), stride=1, no_skip=True, start_with_relu=False),
    ]
    model_args = dict(block_cfg=block_cfg, norm_kwargs=dict(eps=.001, momentum=.1), **kwargs)
    return _xception('xception41', pretrained=pretrained, **model_args)


@register_model
def xception65(pretrained=False, **kwargs):
    """ Modified Aligned Xception-65
    """
    block_cfg = [
        # entry flow
        dict(in_chs=64, out_chs=128, stride=2),
        dict(in_chs=128, out_chs=256, stride=2),
        dict(in_chs=256, out_chs=728, stride=2),
        # middle flow
        *([dict(in_chs=728, out_chs=728, stride=1)] * 16),
        # exit flow
        dict(in_chs=728, out_chs=(728, 1024, 1024), stride=2),
        dict(in_chs=1024, out_chs=(1536, 1536, 2048), stride=1, no_skip=True, start_with_relu=False),
    ]
    model_args = dict(block_cfg=block_cfg, norm_kwargs=dict(eps=.001, momentum=.1), **kwargs)
    return _xception('xception65', pretrained=pretrained, **model_args)


@register_model
def xception71(pretrained=False, **kwargs):
    """ Modified Aligned Xception-71
    """
    block_cfg = [
        # entry flow
        dict(in_chs=64, out_chs=128, stride=2),
        dict(in_chs=128, out_chs=256, stride=1),
        dict(in_chs=256, out_chs=256, stride=2),
        dict(in_chs=256, out_chs=728, stride=1),
        dict(in_chs=728, out_chs=728, stride=2),
        # middle flow
        *([dict(in_chs=728, out_chs=728, stride=1)] * 16),
        # exit flow
        dict(in_chs=728, out_chs=(728, 1024, 1024), stride=2),
        dict(in_chs=1024, out_chs=(1536, 1536, 2048), stride=1, no_skip=True, start_with_relu=False),
    ]
    model_args = dict(block_cfg=block_cfg, norm_kwargs=dict(eps=.001, momentum=.1), **kwargs)
    return _xception('xception71', pretrained=pretrained, **model_args)
