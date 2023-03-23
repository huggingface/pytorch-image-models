""" Deep Layer Aggregation and DLA w/ Res2Net
DLA original adapted from Official Pytorch impl at:
DLA Paper: `Deep Layer Aggregation` - https://arxiv.org/abs/1707.06484

Res2Net additions from: https://github.com/gasvn/Res2Net/
Res2Net Paper: `Res2Net: A New Multi-scale Backbone Architecture` - https://arxiv.org/abs/1904.01169
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import create_classifier
from ._builder import build_model_with_cfg
from ._registry import register_model

__all__ = ['DLA']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'base_layer.0', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'dla34': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dla34-2b83ff04.pth'),
    'dla46_c': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dla46_c-9b68d685.pth'),
    'dla46x_c': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dla46x_c-6bc5b5c8.pth'),
    'dla60x_c': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dla60x_c-a38e054a.pth'),
    'dla60': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dla60-9e91bd4d.pth'),
    'dla60x': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dla60x-6818f6bb.pth'),
    'dla102': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dla102-21f57b54.pth'),
    'dla102x': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dla102x-7ec0aa2a.pth'),
    'dla102x2': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dla102x2-ac4239c4.pth'),
    'dla169': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dla169-7c767967.pth'),
    'dla60_res2net': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net_dla60_4s-d88db7f9.pth'),
    'dla60_res2next': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2next_dla60_4s-d327927b.pth'),
}


class DlaBasic(nn.Module):
    """DLA Basic"""

    def __init__(self, inplanes, planes, stride=1, dilation=1, **_):
        super(DlaBasic, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, shortcut=None, children: Optional[List[torch.Tensor]] = None):
        if shortcut is None:
            shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut
        out = self.relu(out)

        return out


class DlaBottleneck(nn.Module):
    """DLA/DLA-X Bottleneck"""
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1, dilation=1, cardinality=1, base_width=64):
        super(DlaBottleneck, self).__init__()
        self.stride = stride
        mid_planes = int(math.floor(outplanes * (base_width / 64)) * cardinality)
        mid_planes = mid_planes // self.expansion

        self.conv1 = nn.Conv2d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(
            mid_planes, mid_planes, kernel_size=3, stride=stride, padding=dilation,
            bias=False, dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, shortcut: Optional[torch.Tensor] = None, children: Optional[List[torch.Tensor]] = None):
        if shortcut is None:
            shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)

        return out


class DlaBottle2neck(nn.Module):
    """ Res2Net/Res2NeXT DLA Bottleneck
    Adapted from https://github.com/gasvn/Res2Net/blob/master/dla.py
    """
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1, dilation=1, scale=4, cardinality=8, base_width=4):
        super(DlaBottle2neck, self).__init__()
        self.is_first = stride > 1
        self.scale = scale
        mid_planes = int(math.floor(outplanes * (base_width / 64)) * cardinality)
        mid_planes = mid_planes // self.expansion
        self.width = mid_planes

        self.conv1 = nn.Conv2d(inplanes, mid_planes * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes * scale)

        num_scale_convs = max(1, scale - 1)
        convs = []
        bns = []
        for _ in range(num_scale_convs):
            convs.append(nn.Conv2d(
                mid_planes, mid_planes, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation, groups=cardinality, bias=False))
            bns.append(nn.BatchNorm2d(mid_planes))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) if self.is_first else None

        self.conv3 = nn.Conv2d(mid_planes * scale, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, shortcut: Optional[torch.Tensor] = None, children: Optional[List[torch.Tensor]] = None):
        if shortcut is None:
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
            if self.pool is not None:  # self.is_first == True, None check for torchscript
                spo.append(self.pool(spx[-1]))
            else:
                spo.append(spx[-1])
        out = torch.cat(spo, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)

        return out


class DlaRoot(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut):
        super(DlaRoot, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut

    def forward(self, x_children: List[torch.Tensor]):
        x = self.conv(torch.cat(x_children, 1))
        x = self.bn(x)
        if self.shortcut:
            x += x_children[0]
        x = self.relu(x)

        return x


class DlaTree(nn.Module):
    def __init__(
            self, levels, block, in_channels, out_channels, stride=1, dilation=1, cardinality=1,
            base_width=64, level_root=False, root_dim=0, root_kernel_size=1, root_shortcut=False):
        super(DlaTree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else nn.Identity()
        self.project = nn.Identity()
        cargs = dict(dilation=dilation, cardinality=cardinality, base_width=base_width)
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, **cargs)
            self.tree2 = block(out_channels, out_channels, 1, **cargs)
            if in_channels != out_channels:
                # NOTE the official impl/weights have  project layers in levels > 1 case that are never
                # used, I've moved the project layer here to avoid wasted params but old checkpoints will
                # need strict=False while loading.
                self.project = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_channels))
            self.root = DlaRoot(root_dim, out_channels, root_kernel_size, root_shortcut)
        else:
            cargs.update(dict(root_kernel_size=root_kernel_size, root_shortcut=root_shortcut))
            self.tree1 = DlaTree(
                levels - 1, block, in_channels, out_channels, stride, root_dim=0, **cargs)
            self.tree2 = DlaTree(
                levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels, **cargs)
            self.root = None
        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels

    def forward(self, x, shortcut: Optional[torch.Tensor] = None, children: Optional[List[torch.Tensor]] = None):
        if children is None:
            children = []
        bottom = self.downsample(x)
        shortcut = self.project(bottom)
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, shortcut)
        if self.root is not None:  # levels == 1
            x2 = self.tree2(x1)
            x = self.root([x2, x1] + children)
        else:
            children.append(x1)
            x = self.tree2(x1, None, children)
        return x


class DLA(nn.Module):
    def __init__(
            self, levels, channels, output_stride=32, num_classes=1000, in_chans=3, global_pool='avg',
            cardinality=1, base_width=64, block=DlaBottle2neck, shortcut_root=False, drop_rate=0.0):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.cardinality = cardinality
        self.base_width = base_width
        self.drop_rate = drop_rate
        assert output_stride == 32  # FIXME support dilation

        self.base_layer = nn.Sequential(
            nn.Conv2d(in_chans, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        cargs = dict(cardinality=cardinality, base_width=base_width, root_shortcut=shortcut_root)
        self.level2 = DlaTree(levels[2], block, channels[1], channels[2], 2, level_root=False, **cargs)
        self.level3 = DlaTree(levels[3], block, channels[2], channels[3], 2, level_root=True, **cargs)
        self.level4 = DlaTree(levels[4], block, channels[3], channels[4], 2, level_root=True, **cargs)
        self.level5 = DlaTree(levels[5], block, channels[4], channels[5], 2, level_root=True, **cargs)
        self.feature_info = [
            dict(num_chs=channels[0], reduction=1, module='level0'),  # rare to have a meaningful stride 1 level
            dict(num_chs=channels[1], reduction=2, module='level1'),
            dict(num_chs=channels[2], reduction=4, module='level2'),
            dict(num_chs=channels[3], reduction=8, module='level3'),
            dict(num_chs=channels[4], reduction=16, module='level4'),
            dict(num_chs=channels[5], reduction=32, module='level5'),
        ]

        self.num_features = channels[-1]
        self.global_pool, self.fc = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool, use_conv=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(
                    inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1,
                    padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^base_layer',
            blocks=r'^level(\d+)' if coarse else [
                # an unusual arch, this achieves somewhat more granularity without getting super messy
                (r'^level(\d+)\.tree(\d+)', None),
                (r'^level(\d+)\.root', (2,)),
                (r'^level(\d+)', (1,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool, use_conv=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()

    def forward_features(self, x):
        x = self.base_layer(x)
        x = self.level0(x)
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)
        x = self.level5(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        if pre_logits:
            return self.flatten(x)
        x = self.fc(x)
        return self.flatten(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_dla(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        DLA, variant, pretrained,
        pretrained_strict=False,
        feature_cfg=dict(out_indices=(1, 2, 3, 4, 5)),
        **kwargs)


@register_model
def dla60_res2net(pretrained=False, **kwargs):
    model_kwargs = dict(
        levels=(1, 1, 1, 2, 3, 1), channels=(16, 32, 128, 256, 512, 1024),
        block=DlaBottle2neck, cardinality=1, base_width=28, **kwargs)
    return _create_dla('dla60_res2net', pretrained, **model_kwargs)


@register_model
def dla60_res2next(pretrained=False,**kwargs):
    model_kwargs = dict(
        levels=(1, 1, 1, 2, 3, 1), channels=(16, 32, 128, 256, 512, 1024),
        block=DlaBottle2neck, cardinality=8, base_width=4, **kwargs)
    return _create_dla('dla60_res2next', pretrained, **model_kwargs)


@register_model
def dla34(pretrained=False, **kwargs):  # DLA-34
    model_kwargs = dict(
        levels=[1, 1, 1, 2, 2, 1], channels=[16, 32, 64, 128, 256, 512],
        block=DlaBasic, **kwargs)
    return _create_dla('dla34', pretrained, **model_kwargs)


@register_model
def dla46_c(pretrained=False, **kwargs):  # DLA-46-C
    model_kwargs = dict(
        levels=[1, 1, 1, 2, 2, 1], channels=[16, 32, 64, 64, 128, 256],
        block=DlaBottleneck, **kwargs)
    return _create_dla('dla46_c', pretrained, **model_kwargs)


@register_model
def dla46x_c(pretrained=False, **kwargs):  # DLA-X-46-C
    model_kwargs = dict(
        levels=[1, 1, 1, 2, 2, 1], channels=[16, 32, 64, 64, 128, 256],
        block=DlaBottleneck, cardinality=32, base_width=4, **kwargs)
    return _create_dla('dla46x_c', pretrained, **model_kwargs)


@register_model
def dla60x_c(pretrained=False, **kwargs):  # DLA-X-60-C
    model_kwargs = dict(
        levels=[1, 1, 1, 2, 3, 1], channels=[16, 32, 64, 64, 128, 256],
        block=DlaBottleneck, cardinality=32, base_width=4, **kwargs)
    return _create_dla('dla60x_c', pretrained, **model_kwargs)


@register_model
def dla60(pretrained=False, **kwargs):  # DLA-60
    model_kwargs = dict(
        levels=[1, 1, 1, 2, 3, 1], channels=[16, 32, 128, 256, 512, 1024],
        block=DlaBottleneck, **kwargs)
    return _create_dla('dla60', pretrained, **model_kwargs)


@register_model
def dla60x(pretrained=False, **kwargs):  # DLA-X-60
    model_kwargs = dict(
        levels=[1, 1, 1, 2, 3, 1], channels=[16, 32, 128, 256, 512, 1024],
        block=DlaBottleneck, cardinality=32, base_width=4, **kwargs)
    return _create_dla('dla60x', pretrained, **model_kwargs)


@register_model
def dla102(pretrained=False, **kwargs):  # DLA-102
    model_kwargs = dict(
        levels=[1, 1, 1, 3, 4, 1], channels=[16, 32, 128, 256, 512, 1024],
        block=DlaBottleneck, shortcut_root=True, **kwargs)
    return _create_dla('dla102', pretrained, **model_kwargs)


@register_model
def dla102x(pretrained=False, **kwargs):  # DLA-X-102
    model_kwargs = dict(
        levels=[1, 1, 1, 3, 4, 1], channels=[16, 32, 128, 256, 512, 1024],
        block=DlaBottleneck, cardinality=32, base_width=4, shortcut_root=True, **kwargs)
    return _create_dla('dla102x', pretrained, **model_kwargs)


@register_model
def dla102x2(pretrained=False, **kwargs):  # DLA-X-102 64
    model_kwargs = dict(
        levels=[1, 1, 1, 3, 4, 1], channels=[16, 32, 128, 256, 512, 1024],
        block=DlaBottleneck, cardinality=64, base_width=4, shortcut_root=True, **kwargs)
    return _create_dla('dla102x2', pretrained, **model_kwargs)


@register_model
def dla169(pretrained=False, **kwargs):  # DLA-169
    model_kwargs = dict(
        levels=[1, 1, 2, 3, 5, 1], channels=[16, 32, 128, 256, 512, 1024],
        block=DlaBottleneck, shortcut_root=True, **kwargs)
    return _create_dla('dla169', pretrained, **model_kwargs)
