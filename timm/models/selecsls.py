"""PyTorch SelecSLS Net example for ImageNet Classification
License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode)
Author: Dushyant Mehta (@mehtadushy)

SelecSLS (core) Network Architecture as proposed in "XNect: Real-time Multi-person 3D
Human Pose Estimation with a Single RGB Camera, Mehta et al."
https://arxiv.org/abs/1907.00837

Based on ResNet implementation in https://github.com/rwightman/pytorch-image-models
and SelecSLS Net implementation in https://github.com/mehtadushy/SelecSLS-Pytorch
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import create_classifier
from .registry import register_model

__all__ = ['SelecSLS']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (4, 4),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'selecsls42': _cfg(
        url='',
        interpolation='bicubic'),
    'selecsls42b': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-selecsls/selecsls42b-8af30141.pth',
        interpolation='bicubic'),
    'selecsls60': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-selecsls/selecsls60-bbf87526.pth',
        interpolation='bicubic'),
    'selecsls60b': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-selecsls/selecsls60b-94e619b5.pth',
        interpolation='bicubic'),
    'selecsls84': _cfg(
        url='',
        interpolation='bicubic'),
}


class SequentialList(nn.Sequential):

    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (List[torch.Tensor]) -> (List[torch.Tensor])
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (torch.Tensor) -> (List[torch.Tensor])
        pass

    def forward(self, x) -> List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class SelectSeq(nn.Module):
    def __init__(self, mode='index', index=0):
        super(SelectSeq, self).__init__()
        self.mode = mode
        self.index = index

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (List[torch.Tensor]) -> (torch.Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (Tuple[torch.Tensor]) -> (torch.Tensor)
        pass

    def forward(self, x) -> torch.Tensor:
        if self.mode == 'index':
            return x[self.index]
        else:
            return torch.cat(x, dim=1)


def conv_bn(in_chs, out_chs, k=3, stride=1, padding=None, dilation=1):
    if padding is None:
        padding = ((stride - 1) + dilation * (k - 1)) // 2
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs, k, stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(inplace=True)
    )


class SelecSLSBlock(nn.Module):
    def __init__(self, in_chs, skip_chs, mid_chs, out_chs, is_first, stride, dilation=1):
        super(SelecSLSBlock, self).__init__()
        self.stride = stride
        self.is_first = is_first
        assert stride in [1, 2]

        # Process input with 4 conv blocks with the same number of input and output channels
        self.conv1 = conv_bn(in_chs, mid_chs, 3, stride, dilation=dilation)
        self.conv2 = conv_bn(mid_chs, mid_chs, 1)
        self.conv3 = conv_bn(mid_chs, mid_chs // 2, 3)
        self.conv4 = conv_bn(mid_chs // 2, mid_chs, 1)
        self.conv5 = conv_bn(mid_chs, mid_chs // 2, 3)
        self.conv6 = conv_bn(2 * mid_chs + (0 if is_first else skip_chs), out_chs, 1)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        if not isinstance(x, list):
            x = [x]
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
    cfg : network config dictionary specifying block type, feature, and head args
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    """

    def __init__(self, cfg, num_classes=1000, in_chans=3, drop_rate=0.0, global_pool='avg'):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(SelecSLS, self).__init__()

        self.stem = conv_bn(in_chans, 32, stride=2)
        self.features = SequentialList(*[cfg['block'](*block_args) for block_args in cfg['features']])
        self.from_seq = SelectSeq()  # from List[tensor] -> Tensor in module compatible way
        self.head = nn.Sequential(*[conv_bn(*conv_args) for conv_args in cfg['head']])
        self.num_features = cfg['num_features']
        self.feature_info = cfg['feature_info']

        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(self.from_seq(x))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


def _create_selecsls(variant, pretrained, **kwargs):
    cfg = {}
    feature_info = [dict(num_chs=32, reduction=2, module='stem.2')]
    if variant.startswith('selecsls42'):
        cfg['block'] = SelecSLSBlock
        # Define configuration of the network after the initial neck
        cfg['features'] = [
            # in_chs, skip_chs, mid_chs, out_chs, is_first, stride
            (32, 0, 64, 64, True, 2),
            (64, 64, 64, 128, False, 1),
            (128, 0, 144, 144, True, 2),
            (144, 144, 144, 288, False, 1),
            (288, 0, 304, 304, True, 2),
            (304, 304, 304, 480, False, 1),
        ]
        feature_info.extend([
            dict(num_chs=128, reduction=4, module='features.1'),
            dict(num_chs=288, reduction=8, module='features.3'),
            dict(num_chs=480, reduction=16, module='features.5'),
        ])
        # Head can be replaced with alternative configurations depending on the problem
        feature_info.append(dict(num_chs=1024, reduction=32, module='head.1'))
        if variant == 'selecsls42b':
            cfg['head'] = [
                (480, 960, 3, 2),
                (960, 1024, 3, 1),
                (1024, 1280, 3, 2),
                (1280, 1024, 1, 1),
            ]
            feature_info.append(dict(num_chs=1024, reduction=64, module='head.3'))
            cfg['num_features'] = 1024
        else:
            cfg['head'] = [
                (480, 960, 3, 2),
                (960, 1024, 3, 1),
                (1024, 1024, 3, 2),
                (1024, 1280, 1, 1),
            ]
            feature_info.append(dict(num_chs=1280, reduction=64, module='head.3'))
            cfg['num_features'] = 1280

    elif variant.startswith('selecsls60'):
        cfg['block'] = SelecSLSBlock
        # Define configuration of the network after the initial neck
        cfg['features'] = [
            # in_chs, skip_chs, mid_chs, out_chs, is_first, stride
            (32, 0, 64, 64, True, 2),
            (64, 64, 64, 128, False, 1),
            (128, 0, 128, 128, True, 2),
            (128, 128, 128, 128, False, 1),
            (128, 128, 128, 288, False, 1),
            (288, 0, 288, 288, True, 2),
            (288, 288, 288, 288, False, 1),
            (288, 288, 288, 288, False, 1),
            (288, 288, 288, 416, False, 1),
        ]
        feature_info.extend([
            dict(num_chs=128, reduction=4, module='features.1'),
            dict(num_chs=288, reduction=8, module='features.4'),
            dict(num_chs=416, reduction=16, module='features.8'),
        ])
        # Head can be replaced with alternative configurations depending on the problem
        feature_info.append(dict(num_chs=1024, reduction=32, module='head.1'))
        if variant == 'selecsls60b':
            cfg['head'] = [
                (416, 756, 3, 2),
                (756, 1024, 3, 1),
                (1024, 1280, 3, 2),
                (1280, 1024, 1, 1),
            ]
            feature_info.append(dict(num_chs=1024, reduction=64, module='head.3'))
            cfg['num_features'] = 1024
        else:
            cfg['head'] = [
                (416, 756, 3, 2),
                (756, 1024, 3, 1),
                (1024, 1024, 3, 2),
                (1024, 1280, 1, 1),
            ]
            feature_info.append(dict(num_chs=1280, reduction=64, module='head.3'))
            cfg['num_features'] = 1280

    elif variant == 'selecsls84':
        cfg['block'] = SelecSLSBlock
        # Define configuration of the network after the initial neck
        cfg['features'] = [
            # in_chs, skip_chs, mid_chs, out_chs, is_first, stride
            (32, 0, 64, 64, True, 2),
            (64, 64, 64, 144, False, 1),
            (144, 0, 144, 144, True, 2),
            (144, 144, 144, 144, False, 1),
            (144, 144, 144, 144, False, 1),
            (144, 144, 144, 144, False, 1),
            (144, 144, 144, 304, False, 1),
            (304, 0, 304, 304, True, 2),
            (304, 304, 304, 304, False, 1),
            (304, 304, 304, 304, False, 1),
            (304, 304, 304, 304, False, 1),
            (304, 304, 304, 304, False, 1),
            (304, 304, 304, 512, False, 1),
        ]
        feature_info.extend([
            dict(num_chs=144, reduction=4, module='features.1'),
            dict(num_chs=304, reduction=8, module='features.6'),
            dict(num_chs=512, reduction=16, module='features.12'),
        ])
        # Head can be replaced with alternative configurations depending on the problem
        cfg['head'] = [
            (512, 960, 3, 2),
            (960, 1024, 3, 1),
            (1024, 1024, 3, 2),
            (1024, 1280, 3, 1),
        ]
        cfg['num_features'] = 1280
        feature_info.extend([
            dict(num_chs=1024, reduction=32, module='head.1'),
            dict(num_chs=1280, reduction=64, module='head.3')
        ])
    else:
        raise ValueError('Invalid net configuration ' + variant + ' !!!')
    cfg['feature_info'] = feature_info

    # this model can do 6 feature levels by default, unlike most others, leave as 0-4 to avoid surprises?
    return build_model_with_cfg(
        SelecSLS, variant, pretrained,
        default_cfg=default_cfgs[variant],
        model_cfg=cfg,
        feature_cfg=dict(out_indices=(0, 1, 2, 3, 4), flatten_sequential=True),
        **kwargs)


@register_model
def selecsls42(pretrained=False, **kwargs):
    """Constructs a SelecSLS42 model.
    """
    return _create_selecsls('selecsls42', pretrained, **kwargs)


@register_model
def selecsls42b(pretrained=False, **kwargs):
    """Constructs a SelecSLS42_B model.
    """
    return _create_selecsls('selecsls42b', pretrained, **kwargs)


@register_model
def selecsls60(pretrained=False, **kwargs):
    """Constructs a SelecSLS60 model.
    """
    return _create_selecsls('selecsls60', pretrained, **kwargs)


@register_model
def selecsls60b(pretrained=False, **kwargs):
    """Constructs a SelecSLS60_B model.
    """
    return _create_selecsls('selecsls60b', pretrained, **kwargs)


@register_model
def selecsls84(pretrained=False, **kwargs):
    """Constructs a SelecSLS84 model.
    """
    return _create_selecsls('selecsls84', pretrained, **kwargs)
