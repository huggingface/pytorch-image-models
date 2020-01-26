import math
from collections import OrderedDict

import torch
from torch import nn as nn

from timm.models.registry import register_model
from timm.models.helpers import load_pretrained
from timm.models.resnet import ResNet, get_padding, SEModule
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


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
    'skresnet18': _cfg(url=''),
    'skresnet26d': _cfg()
}


class SelectiveKernelAttn(nn.Module):
    def __init__(self, channels, num_paths=2, attn_channels=32,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(SelectiveKernelAttn, self).__init__()
        self.num_paths = num_paths
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = norm_layer(attn_channels)
        self.act = act_layer(inplace=True)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x):
        assert x.shape[1] == self.num_paths
        x = torch.sum(x, dim=1)
        #print('attn sum', x.shape)
        x = self.pool(x)
        #print('attn pool', x.shape)
        x = self.fc_reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_select(x)
        #print('attn sel', x.shape)
        B, C, H, W = x.shape
        x = x.view(B, self.num_paths, C // self.num_paths, H, W)
        #print('attn spl', x.shape)
        x = torch.softmax(x, dim=1)
        return x


def _kernel_valid(k):
    if isinstance(k, (list, tuple)):
        for ki in k:
            return _kernel_valid(ki)
    assert k >= 3 and k % 2


class SelectiveKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=[3, 5], stride=1, dilation=1, groups=1,
                 attn_reduction=16, min_attn_channels=32, keep_3x3=True, use_attn=True,
                 split_input=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(SelectiveKernelConv, self).__init__()
        _kernel_valid(kernel_size)
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2
        if keep_3x3:
            dilation = [dilation * (k - 1) // 2 for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [dilation] * len(kernel_size)
        num_paths = len(kernel_size)
        self.num_paths = num_paths
        self.split_input = split_input
        self.in_channels = in_channels
        self.out_channels = out_channels
        if split_input:
            assert in_channels % num_paths == 0 and out_channels % num_paths == 0
            in_channels = in_channels // num_paths
            out_channels = out_channels // num_paths
        groups = min(out_channels, groups)

        self.paths = nn.ModuleList()
        for k, d in zip(kernel_size, dilation):
            p = get_padding(k, stride, d)
            self.paths.append(nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    in_channels, out_channels, kernel_size=k, stride=stride, padding=p, dilation=d, groups=groups)),
                ('bn', norm_layer(out_channels)),
                ('act', act_layer(inplace=True))
            ])))

        if use_attn:
            attn_channels = max(int(out_channels / attn_reduction), min_attn_channels)
            self.attn = SelectiveKernelAttn(out_channels, num_paths, attn_channels)
        else:
            self.attn = None

    def forward(self, x):
        if self.split_input:
            x_split = torch.split(x, self.in_channels // self.num_paths, 1)
            x_paths = [op(x_split[i]) for i, op in enumerate(self.paths)]
        else:
            x_paths = [op(x) for op in self.paths]

        if self.attn is not None:
            x = torch.stack(x_paths, dim=1)
            # print('paths', x_paths.shape)
            x_attn = self.attn(x)
            #print('attn', x_attn.shape)
            x = x * x_attn
            #print('amul', x.shape)

        if self.split_input:
            B, N, C, H, W = x.shape
            x = x.reshape(B, N * C, H, W)
        else:
            x = torch.sum(x, dim=1)
        #print('aout', x.shape)
        return x


class SelectiveKernelBasic(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, base_width=64, use_se=False, sk_kwargs=None,
                 reduce_first=1, dilation=1, previous_dilation=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(SelectiveKernelBasic, self).__init__()

        sk_kwargs = sk_kwargs or {}
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion

        _selective_first = True  # FIXME temporary, for experiments
        if _selective_first:
            self.conv1 = SelectiveKernelConv(
                inplanes, first_planes, stride=stride, dilation=dilation, **sk_kwargs)
        else:
            self.conv1 = nn.Conv2d(
                inplanes, first_planes, kernel_size=3, stride=stride, padding=dilation,
                dilation=dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        if _selective_first:
            self.conv2 = nn.Conv2d(
                first_planes, outplanes, kernel_size=3, padding=previous_dilation,
                dilation=previous_dilation, bias=False)
        else:
            self.conv2 = SelectiveKernelConv(
                first_planes, outplanes, dilation=previous_dilation, **sk_kwargs)
        self.bn2 = norm_layer(outplanes)
        self.se = SEModule(outplanes, planes // 4) if use_se else None
        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act2(out)

        return out


class SelectiveKernelBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, base_width=64, use_se=False, sk_kwargs=None,
                 reduce_first=1, dilation=1, previous_dilation=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(SelectiveKernelBottleneck, self).__init__()

        sk_kwargs = sk_kwargs or {}
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = SelectiveKernelConv(
            first_planes, width, stride=stride, dilation=dilation, groups=cardinality, **sk_kwargs)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.se = SEModule(outplanes, planes // 4) if use_se else None
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act3(out)

        return out


@register_model
def skresnet26d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-26 model.
    """
    default_cfg = default_cfgs['skresnet26d']
    sk_kwargs = dict(
        keep_3x3=False,
    )
    model = ResNet(
        SelectiveKernelBottleneck, [2, 2, 2, 2],  stem_width=32, stem_type='deep', avg_down=True,
        num_classes=num_classes, in_chans=in_chans, block_args=dict(sk_kwargs=sk_kwargs),
        **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def skresnet18(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-18 model.
    """
    default_cfg = default_cfgs['skresnet18']
    sk_kwargs = dict(
        min_attn_channels=16,
    )
    model = ResNet(
        SelectiveKernelBasic, [2, 2, 2, 2], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sk_kwargs=sk_kwargs), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def sksresnet18(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-18 model.
    """
    default_cfg = default_cfgs['skresnet18']
    sk_kwargs = dict(
        min_attn_channels=16,
        split_input=True
    )
    model = ResNet(
        SelectiveKernelBasic, [2, 2, 2, 2], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sk_kwargs=sk_kwargs), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model