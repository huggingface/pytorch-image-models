""" Selective Kernel Networks (ResNet base)

Paper: Selective Kernel Networks (https://arxiv.org/abs/1903.06586)

This was inspired by reading 'Compounding the Performance Improvements...' (https://arxiv.org/abs/2001.06268)
and a streamlined impl at https://github.com/clovaai/assembled-cnn but I ended up building something closer
to the original paper with some modifications of my own to better balance param count vs accuracy.

Hacked together by / Copyright 2020 Ross Wightman
"""
import math

from torch import nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectiveKernel, ConvNormAct, create_attn
from ._builder import build_model_with_cfg
from ._registry import register_model
from .resnet import ResNet


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'skresnet18': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/skresnet18_ra-4eec2804.pth'),
    'skresnet34': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/skresnet34_ra-bdc0ccde.pth'),
    'skresnet50': _cfg(),
    'skresnet50d': _cfg(
        first_conv='conv1.0'),
    'skresnext50_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/skresnext50_ra-f40e40bf.pth'),
}


class SelectiveKernelBasic(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            cardinality=1,
            base_width=64,
            sk_kwargs=None,
            reduce_first=1,
            dilation=1,
            first_dilation=None,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            attn_layer=None,
            aa_layer=None,
            drop_block=None,
            drop_path=None,
    ):
        super(SelectiveKernelBasic, self).__init__()

        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = SelectiveKernel(
            inplanes, first_planes, stride=stride, dilation=first_dilation,
            aa_layer=aa_layer, drop_layer=drop_block, **conv_kwargs, **sk_kwargs)
        self.conv2 = ConvNormAct(
            first_planes, outplanes, kernel_size=3, dilation=dilation, apply_act=False, **conv_kwargs)
        self.se = create_attn(attn_layer, outplanes)
        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.conv2.bn, 'weight', None) is not None:
            nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act(x)
        return x


class SelectiveKernelBottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            cardinality=1,
            base_width=64,
            sk_kwargs=None,
            reduce_first=1,
            dilation=1,
            first_dilation=None,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            attn_layer=None,
            aa_layer=None,
            drop_block=None,
            drop_path=None,
    ):
        super(SelectiveKernelBottleneck, self).__init__()

        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = ConvNormAct(inplanes, first_planes, kernel_size=1, **conv_kwargs)
        self.conv2 = SelectiveKernel(
            first_planes, width, stride=stride, dilation=first_dilation, groups=cardinality,
            aa_layer=aa_layer, drop_layer=drop_block, **conv_kwargs, **sk_kwargs)
        self.conv3 = ConvNormAct(width, outplanes, kernel_size=1, apply_act=False, **conv_kwargs)
        self.se = create_attn(attn_layer, outplanes)
        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.conv3.bn, 'weight', None) is not None:
            nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act(x)
        return x


def _create_skresnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


@register_model
def skresnet18(pretrained=False, **kwargs):
    """Constructs a Selective Kernel ResNet-18 model.

    Different from configs in Select Kernel paper or "Compounding the Performance Improvements..." this
    variation splits the input channels to the selective convolutions to keep param count down.
    """
    sk_kwargs = dict(rd_ratio=1 / 8, rd_divisor=16, split_input=True)
    model_args = dict(
        block=SelectiveKernelBasic, layers=[2, 2, 2, 2], block_args=dict(sk_kwargs=sk_kwargs),
        zero_init_last=False, **kwargs)
    return _create_skresnet('skresnet18', pretrained, **model_args)


@register_model
def skresnet34(pretrained=False, **kwargs):
    """Constructs a Selective Kernel ResNet-34 model.

    Different from configs in Select Kernel paper or "Compounding the Performance Improvements..." this
    variation splits the input channels to the selective convolutions to keep param count down.
    """
    sk_kwargs = dict(rd_ratio=1 / 8, rd_divisor=16, split_input=True)
    model_args = dict(
        block=SelectiveKernelBasic, layers=[3, 4, 6, 3], block_args=dict(sk_kwargs=sk_kwargs),
        zero_init_last=False, **kwargs)
    return _create_skresnet('skresnet34', pretrained, **model_args)


@register_model
def skresnet50(pretrained=False, **kwargs):
    """Constructs a Select Kernel ResNet-50 model.

    Different from configs in Select Kernel paper or "Compounding the Performance Improvements..." this
    variation splits the input channels to the selective convolutions to keep param count down.
    """
    sk_kwargs = dict(split_input=True)
    model_args = dict(
        block=SelectiveKernelBottleneck, layers=[3, 4, 6, 3], block_args=dict(sk_kwargs=sk_kwargs),
        zero_init_last=False, **kwargs)
    return _create_skresnet('skresnet50', pretrained, **model_args)


@register_model
def skresnet50d(pretrained=False, **kwargs):
    """Constructs a Select Kernel ResNet-50-D model.

    Different from configs in Select Kernel paper or "Compounding the Performance Improvements..." this
    variation splits the input channels to the selective convolutions to keep param count down.
    """
    sk_kwargs = dict(split_input=True)
    model_args = dict(
        block=SelectiveKernelBottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(sk_kwargs=sk_kwargs), zero_init_last=False, **kwargs)
    return _create_skresnet('skresnet50d', pretrained, **model_args)


@register_model
def skresnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a Select Kernel ResNeXt50-32x4d model. This should be equivalent to
    the SKNet-50 model in the Select Kernel Paper
    """
    sk_kwargs = dict(rd_ratio=1/16, rd_divisor=32, split_input=False)
    model_args = dict(
        block=SelectiveKernelBottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4,
        block_args=dict(sk_kwargs=sk_kwargs), zero_init_last=False, **kwargs)
    return _create_skresnet('skresnext50_32x4d', pretrained, **model_args)

