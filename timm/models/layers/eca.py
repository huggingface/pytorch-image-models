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
from torch import nn
import torch.nn.functional as F


from .create_act import create_act_layer
from .helpers import make_divisible


class EcaModule(nn.Module):
    """Constructs an ECA module.

    Args:
        channels: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        kernel_size: Adaptive selection of kernel size (default=3)
        gamm: used in kernel_size calc, see above
        beta: used in kernel_size calc, see above
        act_layer: optional non-linearity after conv, enables conv bias, this is an experiment
        gate_layer: gating non-linearity to use
    """
    def __init__(
            self, channels=None, kernel_size=3, gamma=2, beta=1, act_layer=None, gate_layer='sigmoid',
            rd_ratio=1/8, rd_channels=None, rd_divisor=8, use_mlp=False):
        super(EcaModule, self).__init__()
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        if use_mlp:
            # NOTE 'mlp' mode is a timm experiment, not in paper
            assert channels is not None
            if rd_channels is None:
                rd_channels = make_divisible(channels * rd_ratio, divisor=rd_divisor)
            act_layer = act_layer or nn.ReLU
            self.conv = nn.Conv1d(1, rd_channels, kernel_size=1, padding=0, bias=True)
            self.act = create_act_layer(act_layer)
            self.conv2 = nn.Conv1d(rd_channels, 1, kernel_size=kernel_size, padding=padding, bias=True)
        else:
            self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.act = None
            self.conv2 = None
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        y = self.conv(y)
        if self.conv2 is not None:
            y = self.act(y)
            y = self.conv2(y)
        y = self.gate(y).view(x.shape[0], -1, 1, 1)
        return x * y.expand_as(x)


EfficientChannelAttn = EcaModule  # alias


class CecaModule(nn.Module):
    """Constructs a circular ECA module.

    ECA module where the conv uses circular padding rather than zero padding.
    Unlike the spatial dimension, the channels do not have inherent ordering nor
    locality. Although this module in essence, applies such an assumption, it is unnecessary
    to limit the channels on either "edge" from being circularly adapted to each other.
    This will fundamentally increase connectivity and possibly increase performance metrics
    (accuracy, robustness), without significantly impacting resource metrics
    (parameter size, throughput,latency, etc)

    Args:
        channels: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        kernel_size: Adaptive selection of kernel size (default=3)
        gamm: used in kernel_size calc, see above
        beta: used in kernel_size calc, see above
        act_layer: optional non-linearity after conv, enables conv bias, this is an experiment
        gate_layer: gating non-linearity to use
    """

    def __init__(self, channels=None, kernel_size=3, gamma=2, beta=1, act_layer=None, gate_layer='sigmoid'):
        super(CecaModule, self).__init__()
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        has_act = act_layer is not None
        assert kernel_size % 2 == 1

        # PyTorch circular padding mode is buggy as of pytorch 1.4
        # see https://github.com/pytorch/pytorch/pull/17240
        # implement manual circular padding
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=0, bias=has_act)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        y = x.mean((2, 3)).view(x.shape[0], 1, -1)
        # Manually implement circular padding, F.pad does not seemed to be bugged
        y = F.pad(y, (self.padding, self.padding), mode='circular')
        y = self.conv(y)
        y = self.gate(y).view(x.shape[0], -1, 1, 1)
        return x * y.expand_as(x)


CircularEfficientChannelAttn = CecaModule
