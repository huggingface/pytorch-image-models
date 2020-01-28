import math

from torch import nn as nn

from timm.models.registry import register_model
from timm.models.helpers import load_pretrained
from timm.models.conv2d_layers import SelectiveKernelConv
from timm.models.resnet import ResNet, SEModule
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


class SelectiveKernelBasic(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, base_width=64, use_se=False, sk_kwargs=None,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(SelectiveKernelBasic, self).__init__()

        sk_kwargs = sk_kwargs or {}
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        _selective_first = True  # FIXME temporary, for experiments
        if _selective_first:
            self.conv1 = SelectiveKernelConv(
                inplanes, first_planes, stride=stride, dilation=first_dilation, **sk_kwargs)
        else:
            self.conv1 = nn.Conv2d(
                inplanes, first_planes, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        if _selective_first:
            self.conv2 = nn.Conv2d(
                first_planes, outplanes, kernel_size=3, padding=dilation,
                dilation=dilation, bias=False)
        else:
            self.conv2 = SelectiveKernelConv(
                first_planes, outplanes, dilation=dilation, **sk_kwargs)
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
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(SelectiveKernelBottleneck, self).__init__()

        sk_kwargs = sk_kwargs or {}
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = SelectiveKernelConv(
            first_planes, width, stride=stride, dilation=first_dilation, groups=cardinality, **sk_kwargs)
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