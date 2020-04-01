"""
BlurPool layer inspired by
 - Kornia's Max_BlurPool2d
 - Making Convolutional Networks Shift-Invariant Again :cite:`zhang2019shiftinvar`

Hacked together by Chris Ha and Ross Wightman
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .padding import get_padding


class BlurPool2d(nn.Module):
    r"""Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details.
    Corresponds to the Downsample class, which does blurring and subsampling
    Args:
        channels = Number of input channels
        blur_filter_size (int): binomial filter size for blurring. currently supports 3 (default) and 5.
        stride (int): downsampling filter stride
    Shape:
    Returns:
        torch.Tensor: the transformed tensor.
    Examples:
    """

    def __init__(self, channels, blur_filter_size=3, stride=2) -> None:
        super(BlurPool2d, self).__init__()
        assert blur_filter_size > 1
        self.channels = channels
        self.blur_filter_size = blur_filter_size
        self.stride = stride

        pad_size = [get_padding(blur_filter_size, stride, dilation=1)] * 4
        self.padding = nn.ReflectionPad2d(pad_size)

        blur_matrix = (np.poly1d((0.5, 0.5)) ** (blur_filter_size - 1)).coeffs
        blur_filter = torch.Tensor(blur_matrix[:, None] * blur_matrix[None, :])
        self.blur_filter = blur_filter[None, None, :, :]

    def _apply(self, fn):
        # override nn.Module _apply to prevent need for blur_filter to be registered as a buffer,
        # this keeps it out of state dict, but allows .cuda(), .type(), etc to work as expected
        super(BlurPool2d, self)._apply(fn)
        self.blur_filter = fn(self.blur_filter)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # type: ignore
        return F.conv2d(
            self.padding(input_tensor),
            self.blur_filter.type(input_tensor.dtype).expand(C, -1, -1, -1),
            stride=self.stride, groups=input_tensor.shape[1])
