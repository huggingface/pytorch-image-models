'''
ECA module from ECAnet
original paper: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
https://arxiv.org/abs/1910.03151

https://github.com/BangguWu/ECANet
original ECA model borrowed from original github
modified circular ECA implementation and
adoptation for use in pytorch image models package
by Chris Ha https://github.com/VRandme

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
'''
import math
import torch
from torch import nn
import torch.nn.functional as F

class EcaModule(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        k_size: Adaptive selection of kernel size (default=3)
    """
    def __init__(self, channel=None, k_size=3, gamma=2, beta=1):
        super(EcaModule, self).__init__()
        assert k_size % 2 == 1

        if channel is not None:
            t = int(abs(math.log(channel, 2)+beta) / gamma)
            k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # reshape for convolution
        y = y.view(x.shape[0], 1, -1)
        # Two different branches of ECA module
        y = self.conv(y)
        # Multi-scale information fusion
        y = self.sigmoid(y.view(x.shape[0], -1, 1, 1))
        return x * y.expand_as(x)

class CecaModule(nn.Module):
    """Constructs a circular ECA module.
    the primary difference is that the conv uses a circular padding rather than zero padding.
    This is because unlike images, the channels themselves do not have inherent ordering nor
    locality. Although this module in essence, applies such an assumption, it is unnecessary
    to limit the channels on either "edge" from being circularly adapted to each other.
    This will fundamentally increase connectivity and possibly increase performance metrics
    (accuracy, robustness), without signficantly impacting resource metrics
    (parameter size, throughput,latency, etc)

    Args:
        channel: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        k_size: Adaptive selection of kernel size (default=3)
    """

    def __init__(self, channel=None, k_size=3, gamma=2, beta=1):
        super(CecaModule, self).__init__()
        assert k_size % 2 == 1

        if channel is not None:
            t = int(abs(math.log(channel, 2)+beta) / gamma)
            k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #pytorch circular padding mode is bugged as of pytorch 1.4
        #see https://github.com/pytorch/pytorch/pull/17240

        #implement manual circular padding
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=0, bias=False)
        self.padding = (k_size - 1) // 2
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        #manually implement circular padding, F.pad does not seemed to be bugged
        y = F.pad(y.view(x.shape[0], 1, -1), (self.padding, self.padding), mode='circular')

        # Two different branches of ECA module
        y = self.conv(y)

        # Multi-scale information fusion
        y = self.sigmoid(y.view(x.shape[0], -1, 1, 1))

        return x * y.expand_as(x)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, bias=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


class CecaBam(nn.Module):
    def __init__(self, gate_channels, no_spatial=False):
            super(CecaBam, self).__init__()
            self.CecaModule = CecaModule(gate_channels)
            self.no_spatial=no_spatial
            if not no_spatial:
                self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.CecaModule(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
