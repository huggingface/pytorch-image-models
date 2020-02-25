import torch
from torch import nn as nn

from .helpers import make_divisible


class SqueezeExcite(nn.Module):
    """ Squeeze-and-Excitation module as used in Pytorch SENet, SE-ResNeXt implementations

    Args:
        channels (int): number of input and output channels
        reduction (int, float): divisor for attention (squeezed) channels
        act_layer (nn.Module): override the default ReLU activation
    """

    def __init__(self, channels, reduction=16, act_layer=nn.ReLU, divisible_by=1):
        super(SqueezeExcite, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduction_channels = make_divisible(channels // reduction, divisible_by)
        self.fc1 = nn.Conv2d(
            channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(
            reduction_channels, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()


class SqueezeExciteV2(nn.Module):
    """ Squeeze-and-Excitation module as used in EfficientNet, MobileNetV3, related models

    Differs from the original SqueezeExcite impl in that:
      * reduction is specified as a float multiplier instead of divisor (se_ratio)
      * gate function is changeable from sigmoid to alternate (ie hard_sigmoid)
      * layer names match those in weights for the EfficientNet/MobileNetV3 families

    Args:
        channels (int): number of input and output channels
        se_ratio (float): multiplier for attention (squeezed) channels
        reduced_base_chs (int): specify alternate channel count to base the reduction channels on
        act_layer (nn.Module): override the default ReLU activation
        gate_fn (callable): override the default gate function
    """

    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=torch.sigmoid, divisible_by=1, **_):
        super(SqueezeExciteV2, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisible_by)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x
