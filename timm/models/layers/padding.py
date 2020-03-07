""" Padding Helpers

Hacked together by Ross Wightman
"""
import math
from typing import List
from .helpers import ntuple

import torch.nn.functional as F


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    if isinstance(kernel_size, (list, tuple)):
        stride = ntuple(len(kernel_size))(stride)
        dilation = ntuple(len(kernel_size))(dilation)
        return [get_padding(k, s, d) for k, s, d in zip(kernel_size, stride, dilation)]
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), shift: int = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        if shift == 0:
            pl = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]  # ul
        elif shift == 1:
            pl = [pad_w - pad_w // 2, pad_w // 2, pad_h - pad_h // 2, pad_h // 2]  # lr
        elif shift == 2:
            pl = [pad_w - pad_w // 2, pad_w // 2, pad_h // 2, pad_h - pad_h // 2]  # ur
        else:
            pl = [pad_w // 2, pad_w - pad_w // 2, pad_h - pad_h // 2, pad_h // 2]  # ll
        x = F.pad(x, pl)
    return x
