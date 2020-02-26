"""
ECA module from ECAnet

paper: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
https://arxiv.org/abs/1910.03151

Original ECA model borrowed from https://github.com/BangguWu/ECANet

Modified circular ECA implementation and adaption for use in timm package
by Chris Ha https://github.com/VRandme

Original License:

MIT License

Copyright (c) 2019 BangguWu, Qilong Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class EfficientChannelAttn(nn.Module):
    """Constructs an ECA module.

    Args:
        channels: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        kernel_size: Adaptive selection of kernel size (default=3)
    """
    def __init__(self, channels=None, kernel_size=3, gamma=2, beta=1, gate_fn=None):
        super(EfficientChannelAttn, self).__init__()
        assert kernel_size % 2 == 1

        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.gate_fn = gate_fn

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.shape[0], 1, -1)  # Reshape 4d -> 3d for 1d convolution
        y = self.conv(y)
        y = y.view(x.shape[0], -1, 1, 1)  # Back to 4d
        y = y.sigmoid() if self.gate_fn is None else self.gate_fn(y)
        return x * y.expand_as(x)


def padding1d_circular(input, pad):
    r"""input: torch.tensor([[[0., 1., 2.],
                              [3., 4., 5.]]])
        pad: (1, 2)
        output: tensor([[[2., 0., 1., 2., 0., 1.],
                         [5., 3., 4., 5., 3., 4.]]])

        from: https://github.com/pytorch/pytorch/issues/24504
    """
    input = torch.cat([input, input[:, :, 0:pad[-1]]], dim=2)
    if pad[-1] == 0 and pad[-2] != 0:
        return torch.cat([input[:, :, -(pad[-1] + pad[-2]):], input], dim=2)
    else:
        return torch.cat([input[:, :, -(pad[-1] + pad[-2]):-pad[-1]], input], dim=2)


class CircularEfficientChannelAttn(nn.Module):
    """Constructs a circular ECA module.

    ECA module where the conv uses circular padding rather than zero padding.
    Unlike the spatial dimension, the channels do not have inherent ordering nor
    locality. Although this module in essence, applies such an assumption, it is unnecessary
    to limit the channels on either "edge" from being circularly adapted to each other.
    This will fundamentally increase connectivity and possibly increase performance metrics
    (accuracy, robustness), without signficantly impacting resource metrics
    (parameter size, throughput,latency, etc)

    Args:
        channels: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        kernel_size: Adaptive selection of kernel size (default=3)
    """

    def __init__(self, channels=None, kernel_size=3, gamma=2, beta=1, gate_fn=None):
        super(CircularEfficientChannelAttn, self).__init__()
        assert kernel_size % 2 == 1

        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)

        #  pytorch conv circular padding mode is buggy as of pytorch 1.4, will implement manually
        #  see https://github.com/pytorch/pytorch/pull/17240
        #  https://github.com/pytorch/pytorch/issues/24504
        p = (kernel_size - 1) // 2
        self.padding = (p, p)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=0, bias=False)
        self.gate_fn = gate_fn

    def forward(self, x):
        y = self.avg_pool(x)
        y = padding1d_circular(y.view(x.shape[0], 1, -1), self.padding)  # manual circular padding
        y = self.conv(y)
        y = y.view(x.shape[0], -1, 1, 1)
        y = y.sigmoid() if self.gate_fn is None else self.gate_fn(y)
        return x * y.expand_as(x)
