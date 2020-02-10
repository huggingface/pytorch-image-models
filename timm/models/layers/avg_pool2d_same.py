""" AvgPool2d w/ Same Padding

Hacked together by Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math

from .helpers import tup_pair
from .padding import pad_same


def avg_pool2d_same(x, kernel_size: List[int], stride: List[int], padding: List[int] = (0, 0),
                    ceil_mode: bool = False, count_include_pad: bool = True):
    x = pad_same(x, kernel_size, stride)
    return F.avg_pool2d(x, kernel_size, stride, (0, 0), ceil_mode, count_include_pad)


class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """
    def __init__(self, kernel_size: int, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        kernel_size = tup_pair(kernel_size)
        stride = tup_pair(stride)
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0), ceil_mode, count_include_pad)

    def forward(self, x):
        return avg_pool2d_same(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
