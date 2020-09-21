""" ReXNet

A PyTorch impl of `ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network` -
https://arxiv.org/abs/2007.00992

Adapted from original impl at https://github.com/clovaai/rexnet
Copyright (c) 2020-present NAVER Corp. MIT license

Changes for timm, feature extraction, and rounded channel variant hacked together by Ross Wightman
Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
from math import ceil

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from .layers import ClassifierHead, create_act_layer, create_conv2d
from timm.models.registry import register_model
from .layers.activations import sigmoid, Swish, HardSwish, HardSigmoid

def _cfg(url=''):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
    }


default_cfgs = dict(
    rexnet_100=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rexnet/rexnetv1_100-1b4dddf4.pth'),
    rexnet_130=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rexnet/rexnetv1_130-590d768e.pth'),
    rexnet_150=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rexnet/rexnetv1_150-bd1a6aa8.pth'),
    rexnet_200=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rexnet/rexnetv1_200-8c0b7f2d.pth'),
    rexnetr_100=_cfg(
        url=''),
    rexnetr_130=_cfg(
        url=''),
    rexnetr_150=_cfg(
        url=''),
    rexnetr_200=_cfg(
        url=''),
)


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    return new_v

class ConvBn(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, dilation=1, pad_type='', 
                 norm_layer=nn.BatchNorm2d, groups = 1,norm_kwargs=None):
        super(ConvBn, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, groups=groups,padding=pad_type)
        self.bn1 = norm_layer(out_chs, **norm_kwargs)

    def feature_info(self, location):
        if location == 'expansion':  # output of conv after act, same as block coutput
            info = dict(module='act1', hook_type='forward', num_chs=self.conv.out_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv.out_channels)
        return info

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        return x
    
    def fuse_module(self):
        modules_to_fuse = ['conv','bn1']         
        torch.quantization.fuse_modules(self, modules_to_fuse, inplace=True)
        
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(ConvBnAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(out_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self.out_channels = out_chs
    def feature_info(self, location):
        if location == 'expansion':  # output of conv after act, same as block coutput
            info = dict(module='act1', hook_type='forward', num_chs=self.conv.out_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv.out_channels)
        return info

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x
    def fuse_module(self):
        modules_to_fuse = ['conv','bn1']   
        if type(self.act1) == nn.ReLU:
            modules_to_fuse.append('act1')        
        torch.quantization.fuse_modules(self, modules_to_fuse, inplace=True)        
        
class SEWithNorm(nn.Module):

    def __init__(self, channels, reduction=16, act_layer=nn.ReLU, divisor=1, reduction_channels=None,
                 gate_layer='sigmoid'):
        super(SEWithNorm, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduction_channels = reduction_channels or make_divisible(channels // reduction, divisor=divisor)
        self.fc1 = nn.Conv2d(
            channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(reduction_channels)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(
            reduction_channels, channels, kernel_size=1, padding=0, bias=True)
        self.gate = create_act_layer(gate_layer)
        self.quant_mul = nn.quantized.FloatFunctional()
    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.fc1(x_se)
        x_se = self.bn(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return self.quant_mul.mul(x, self.gate(x_se))
    
    def fuse_module(self):        
        modules_to_fuse = ['fc1','bn','act']       
        torch.quantization.fuse_modules(self, modules_to_fuse, inplace=True)
        

class LinearBottleneck(nn.Module):
    def __init__(self, in_chs, out_chs, stride, exp_ratio=1.0, use_se=True, se_rd=12, ch_div=1):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_chs <= out_chs
        self.in_channels = in_chs
        self.out_channels = out_chs

        if exp_ratio != 1.:
            dw_chs = make_divisible(round(in_chs * exp_ratio), divisor=ch_div)
            self.conv_exp = ConvBnAct(in_chs, dw_chs,1, act_layer=Swish)
        else:
            dw_chs = in_chs
            self.conv_exp = None

        self.conv_dw = ConvBn(dw_chs, dw_chs, 3, stride=stride, groups=dw_chs)
        self.se = SEWithNorm(dw_chs, reduction=se_rd, divisor=ch_div) if use_se else None
        self.act_dw = nn.ReLU6()

        self.conv_pwl = ConvBn(dw_chs, out_chs, 1)
        if self.use_shortcut:        
            self.skip_add = nn.quantized.FloatFunctional()
        
    def feat_channels(self, exp=False):
        return self.conv_dw.out_channels if exp else self.out_channels

    def forward(self, x):
        shortcut = x
        if self.conv_exp is not None:
            x = self.conv_exp(x)
        x = self.conv_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.act_dw(x)
        x = self.conv_pwl(x)
        if self.use_shortcut:
            x[:, 0:self.in_channels]= self.skip_add.add(x[:, 0:self.in_channels], shortcut)
        return x

def _block_cfg(width_mult=1.0, depth_mult=1.0, initial_chs=16, final_chs=180, use_se=True, ch_div=1):
    layers = [1, 2, 2, 3, 3, 5]
    strides = [1, 2, 2, 2, 1, 2]
    layers = [ceil(element * depth_mult) for element in layers]
    strides = sum([[element] + [1] * (layers[idx] - 1) for idx, element in enumerate(strides)], [])
    exp_ratios = [1] * layers[0] + [6] * sum(layers[1:])
    depth = sum(layers[:]) * 3
    base_chs = initial_chs / width_mult if width_mult < 1.0 else initial_chs

    # The following channel configuration is a simple instance to make each layer become an expand layer.
    out_chs_list = []
    for i in range(depth // 3):
        out_chs_list.append(make_divisible(round(base_chs * width_mult), divisor=ch_div))
        base_chs += final_chs / (depth // 3 * 1.0)

    if use_se:
        use_ses = [False] * (layers[0] + layers[1]) + [True] * sum(layers[2:])
    else:
        use_ses = [False] * sum(layers[:])

    return zip(out_chs_list, exp_ratios, strides, use_ses)


def _build_blocks(block_cfg, prev_chs, width_mult, se_rd=12, ch_div=1, feature_location='bottleneck'):
    feat_exp = feature_location == 'expansion'
    feat_chs = [prev_chs]
    feature_info = []
    curr_stride = 2
    features = []
    for block_idx, (chs, exp_ratio, stride, se) in enumerate(block_cfg):
        if stride > 1:
            fname = 'stem' if block_idx == 0 else f'features.{block_idx - 1}'
            if block_idx > 0 and feat_exp:
                fname += '.act_dw'
            feature_info += [dict(num_chs=feat_chs[-1], reduction=curr_stride, module=fname)]
            curr_stride *= stride
        features.append(LinearBottleneck(
            in_chs=prev_chs, out_chs=chs, exp_ratio=exp_ratio, stride=stride, use_se=se, se_rd=se_rd, ch_div=ch_div))
        prev_chs = chs
        feat_chs += [features[-1].feat_channels(feat_exp)]
    pen_chs = make_divisible(1280 * width_mult, divisor=ch_div)
    feature_info += [dict(
        num_chs=pen_chs if feat_exp else feat_chs[-1], reduction=curr_stride,
        module=f'features.{len(features) - int(not feat_exp)}')]
    features.append(ConvBnAct(prev_chs, pen_chs,1, act_layer=Swish))
    return features, feature_info


class ReXNetV1(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, global_pool='avg', output_stride=32,
                 initial_chs=16, final_chs=180, width_mult=1.0, depth_mult=1.0, use_se=True,
                 se_rd=12, ch_div=1, drop_rate=0.2, feature_location='bottleneck'):
        super(ReXNetV1, self).__init__()
        self.drop_rate = drop_rate

        assert output_stride == 32  # FIXME support dilation
        stem_base_chs = 32 / width_mult if width_mult < 1.0 else 32
        stem_chs = make_divisible(round(stem_base_chs * width_mult), divisor=ch_div)
        self.stem = ConvBnAct(in_chans, stem_chs, 3, stride=2, act_layer=Swish)

        block_cfg = _block_cfg(width_mult, depth_mult, initial_chs, final_chs, use_se, ch_div)
        features, self.feature_info = _build_blocks(
            block_cfg, stem_chs, width_mult, se_rd, ch_div, feature_location)
        self.num_features = features[-1].out_channels
        self.features = nn.Sequential(*features)

        self.head = ClassifierHead(self.num_features, num_classes, global_pool, drop_rate)
        
        # Quantization Stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub() 
        # FIXME weight init, the original appears to use PyTorch defaults

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.features(x)
        return x

    def forward(self, x):
        x = self.quant(x)
        x = self.forward_features(x)
        x = self.head(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):           
        for m in self.modules():
            if type(m) in [ConvBnAct, ConvBn, SEWithNorm]:
                m.fuse_module()
def _create_rexnet(variant, pretrained, **kwargs):
    feature_cfg = dict(flatten_sequential=True)
    if kwargs.get('feature_location', '') == 'expansion':
        feature_cfg['feature_cls'] = 'hook'
    return build_model_with_cfg(
        ReXNetV1, variant, pretrained, default_cfg=default_cfgs[variant], feature_cfg=feature_cfg, **kwargs)


@register_model
def quant_rexnet_100(pretrained=False, **kwargs):
    """ReXNet V1 1.0x"""
    return _create_rexnet('rexnet_100', pretrained, **kwargs)


@register_model
def quant_rexnet_130(pretrained=False, **kwargs):
    """ReXNet V1 1.3x"""
    return _create_rexnet('rexnet_130', pretrained, width_mult=1.3, **kwargs)


@register_model
def quant_rexnet_150(pretrained=False, **kwargs):
    """ReXNet V1 1.5x"""
    return _create_rexnet('rexnet_150', pretrained, width_mult=1.5, **kwargs)


@register_model
def quant_rexnet_200(pretrained=False, **kwargs):
    """ReXNet V1 2.0x"""
    return _create_rexnet('rexnet_200', pretrained, width_mult=2.0, **kwargs)


@register_model
def quant_rexnetr_100(pretrained=False, **kwargs):
    """ReXNet V1 1.0x w/ rounded (mod 8) channels"""
    return _create_rexnet('rexnetr_100', pretrained, ch_div=8, **kwargs)


@register_model
def quant_rexnetr_130(pretrained=False, **kwargs):
    """ReXNet V1 1.3x w/ rounded (mod 8) channels"""
    return _create_rexnet('rexnetr_130', pretrained, width_mult=1.3, ch_div=8, **kwargs)


@register_model
def quant_rexnetr_150(pretrained=False, **kwargs):
    """ReXNet V1 1.5x w/ rounded (mod 8) channels"""
    return _create_rexnet('rexnetr_150', pretrained, width_mult=1.5, ch_div=8, **kwargs)


@register_model
def quant_rexnetr_200(pretrained=False, **kwargs):
    """ReXNet V1 2.0x w/ rounded (mod 8) channels"""
    return _create_rexnet('rexnetr_200', pretrained, width_mult=2.0, ch_div=8, **kwargs)
