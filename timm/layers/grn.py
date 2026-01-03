""" Global Response Normalization Module

Based on the GRN layer presented in
`ConvNeXt-V2 - Co-designing and Scaling ConvNets with Masked Autoencoders` - https://arxiv.org/abs/2301.00808

This implementation
* works for both NCHW and NHWC tensor layouts
* uses affine param names matching existing torch norm layers
* slightly improves eager mode performance via fused addcmul

Hacked together by / Copyright 2023 Ross Wightman
"""

import torch
from torch import nn as nn


class GlobalResponseNorm(nn.Module):
    """ Global Response Normalization layer
    """
    def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            channels_last: bool = True,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        if channels_last:
            self.spatial_dim = (1, 2)
            self.channel_dim = -1
            wb_shape = (1, 1, 1, dim)
        else:
            self.spatial_dim = (2, 3)
            self.channel_dim = 1
            wb_shape = (1, dim, 1, 1)

        self.weight = nn.Parameter(torch.zeros(wb_shape, **dd))
        self.bias = nn.Parameter(torch.zeros(wb_shape, **dd))

    def forward(self, x):
        x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        return torch.addcmul(self.bias, self.weight * x_n + 1, x)
