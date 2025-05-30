"""
BlurPool layer inspired by
 - Kornia's Max_BlurPool2d
 - Making Convolutional Networks Shift-Invariant Again :cite:`zhang2019shiftinvar`

Hacked together by Chris Ha and Ross Wightman
"""
from functools import partial
from math import comb  # Python 3.8
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .padding import get_padding
from .typing import LayerType


class BlurPool2d(nn.Module):
    r"""Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details.
    Corresponds to the Downsample class, which does blurring and subsampling

    Args:
        channels = Number of input channels
        filt_size (int): binomial filter size for blurring. currently supports 3 (default) and 5.
        stride (int): downsampling filter stride

    Returns:
        torch.Tensor: the transformed tensor.
    """
    def __init__(
            self,
            channels: Optional[int] = None,
            filt_size: int = 3,
            stride: int = 2,
            pad_mode: str = 'reflect',
    ) -> None:
        super(BlurPool2d, self).__init__()
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.padding = [get_padding(filt_size, stride, dilation=1)] * 4

        # (0.5 + 0.5 x)^N => coefficients = C(N,k) / 2^N,  k = 0..N
        coeffs = torch.tensor(
            [comb(filt_size - 1, k) for k in range(filt_size)],
            dtype=torch.float32,
        ) / (2 ** (filt_size - 1))  # normalise so coefficients sum to 1
        blur_filter = (coeffs[:, None] * coeffs[None, :])[None, None, :, :]
        if channels is not None:
            blur_filter = blur_filter.repeat(self.channels, 1, 1, 1)
        self.register_buffer('filt', blur_filter, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding, mode=self.pad_mode)
        if self.channels is None:
            channels = x.shape[1]
            weight = self.filt.expand(channels, 1, self.filt_size, self.filt_size)
        else:
            channels = self.channels
            weight = self.filt
        return F.conv2d(x, weight, stride=self.stride, groups=channels)


def create_aa(
        aa_layer: LayerType,
        channels: Optional[int] = None,
        stride: int = 2,
        enable: bool = True,
        noop: Optional[Type[nn.Module]] = nn.Identity
) -> nn.Module:
    """ Anti-aliasing """
    if not aa_layer or not enable:
        return noop() if noop is not None else None

    if isinstance(aa_layer, str):
        aa_layer = aa_layer.lower().replace('_', '').replace('-', '')
        if aa_layer == 'avg' or aa_layer == 'avgpool':
            aa_layer = nn.AvgPool2d
        elif aa_layer == 'blur' or aa_layer == 'blurpool':
            aa_layer = BlurPool2d
        elif aa_layer == 'blurpc':
            aa_layer = partial(BlurPool2d, pad_mode='constant')

        else:
            assert False, f"Unknown anti-aliasing layer ({aa_layer})."

    try:
        return aa_layer(channels=channels, stride=stride)
    except TypeError as e:
        return aa_layer(stride)
