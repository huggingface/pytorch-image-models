"""
An implementation of LCNet Model as defined in:
PP-LCNet: A Lightweight CPU Convolutional Neural Network. https://arxiv.org/abs/2109.15099
The train script of the model is similar to that of MobileNetV3
Code:https://github.com/murufeng/awesome_lightweight_networks/blob/main/light_cnns/mobile_real_time_network/lcnet.py
"""

import torch.nn as nn

from .layers import make_divisible
from .registry import register_model

__all__ = ['LCNet']

class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, num_groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1)//2,
            groups=num_groups,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        out = identity * x
        return out

class DepthSepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dw_kernel_size, use_se=False, pw_kernel_size=1):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = StemConv(
            in_channels, in_channels, kernel_size=dw_kernel_size,
            stride=stride, num_groups=in_channels)
        if self.use_se:
            self.se = SEModule(in_channels)
        self.pw_conv = StemConv(in_channels, out_channels, kernel_size=pw_kernel_size, stride=1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)

        return x


class LCNet(nn.Module):
    def __init__(self, cfgs, block, num_classes=1000, dropout=0.2, scale=1.0, class_expand=1280):
        super(LCNet, self).__init__()
        self.cfgs = cfgs
        self.class_expand = class_expand
        self.block = block

        self.conv1 = StemConv(
            in_channels=3,
            kernel_size=3,
            out_channels=make_divisible(16 * scale),
            stride=2)
        stages = []
        for cfg in self.cfgs:
            layers = []
            for k, inplanes, planes, stride, use_se in cfg:
                in_channel = make_divisible(inplanes * scale)
                out_channel = make_divisible(planes * scale)
                layers.append(block(in_channel, out_channel, stride=stride, dw_kernel_size=k, use_se=use_se))

            stages.append(nn.Sequential(*layers))

        self.blocks = nn.Sequential(*stages)

        out_channel = make_divisible(out_channel * scale)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.last_conv = nn.Conv2d(
            in_channels=out_channel,
            out_channels=self.class_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=dropout)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.f = nn.Linear(self.class_expand, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.f(x)
        return x

def _create_lcnet(scale=1.0, pretrained=False, **kwargs):

    cfgs = [
        # kernel, in_channels, out_channels, stride, use_se
        # stage1
        [[3, 16, 32, 1, False]],
        # stage2
        [[3, 32, 64, 2, False]],
        [[3, 64, 64, 1, False]],
        # stage3
        [[3, 64, 128, 2, False]],
        [[3, 128, 128, 1, False]],
        # stage4
        [[3, 128, 256, 2, False]],
        [[5, 256, 256, 1, False]],
        # stage5
        [[5, 256, 256, 1, False]],
        [[5, 256, 256, 1, False]],
        [[5, 256, 256, 1, False]],
        [[5, 256, 256, 1, False]],
        [[5, 256, 512, 2, True]],
        [[5, 512, 512, 1, True]]
    ]

    return LCNet(cfgs, DepthSepConvBlock, scale=scale, **kwargs)


@register_model
def lcnet_025(pretrained=False, **kwargs):
    """ LCNet-0.25x """
    model = _create_lcnet(scale=0.25, pretrained=pretrained, **kwargs)
    return model

@register_model
def lcnet_050(pretrained=False, **kwargs):
    """ LCNet-0.5x """
    model = _create_lcnet(scale=0.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_100(pretrained=False, **kwargs):
    """ LCNet-1.0x """
    model = _create_lcnet(scale=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_250(pretrained=False, **kwargs):
    """ LCNet-2.5x """
    model = _create_lcnet(scale=2.5, pretrained=pretrained, **kwargs)
    return model



