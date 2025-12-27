""" Padding Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

from .helpers import to_2tuple


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> Union[int, List[int]]:
    """Calculate symmetric padding for a convolution.

    Recursively handles tuples/lists by computing padding per dimension.

    Args:
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        dilation: Convolution dilation.

    Returns:
        Padding value or list of padding values per dimension.
    """
    if any([isinstance(v, (tuple, list)) for v in [kernel_size, stride, dilation]]):
        kernel_size, stride, dilation = to_2tuple(kernel_size), to_2tuple(stride), to_2tuple(dilation)
        return [get_padding(*a) for a in zip(kernel_size, stride, dilation)]
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def get_same_padding(x: int, kernel_size: int, stride: int, dilation: int):
    """Calculate asymmetric TensorFlow-like 'SAME' padding for a single dimension.

    Args:
        x: Input size for this dimension.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        dilation: Convolution dilation.

    Returns:
        Total padding needed for this dimension.
    """
    if isinstance(x, torch.Tensor):
        return torch.clamp(((x / stride).ceil() - 1) * stride + (kernel_size - 1) * dilation + 1 - x, min=0)
    else:
        return max((math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0)


def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    """Check if SAME padding can be applied statically without runtime overhead.

    Static padding is possible when stride is 1 and kernel expansion is even.

    Args:
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        dilation: Convolution dilation.

    Returns:
        True if static padding is applicable.
    """
    if any([isinstance(v, (tuple, list)) for v in [kernel_size, stride, dilation]]):
        kernel_size, stride, dilation = to_2tuple(kernel_size), to_2tuple(stride), to_2tuple(dilation)
        return all([is_static_pad(*a) for a in zip(kernel_size, stride, dilation)])
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def pad_same_arg(
        input_size: List[int],
        kernel_size: List[int],
        stride: List[int],
        dilation: List[int] = (1, 1),
) -> List[int]:
    """Compute padding arguments for F.pad to achieve SAME padding.

    Args:
        input_size: Input spatial dimensions [height, width].
        kernel_size: Convolution kernel size [height, width].
        stride: Convolution stride [height, width].
        dilation: Convolution dilation [height, width].

    Returns:
        Padding in F.pad format [left, right, top, bottom].
    """
    ih, iw = input_size
    kh, kw = kernel_size
    pad_h = get_same_padding(ih, kh, stride[0], dilation[0])
    pad_w = get_same_padding(iw, kw, stride[1], dilation[1])
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]


def pad_same(
        x,
        kernel_size: List[int],
        stride: List[int],
        dilation: List[int] = (1, 1),
        value: float = 0,
):
    """Apply dynamic SAME padding to input tensor.

    Args:
        x: Input tensor.
        kernel_size: Convolution kernel size [height, width].
        stride: Convolution stride [height, width].
        dilation: Convolution dilation [height, width].
        value: Padding value.

    Returns:
        Padded tensor.
    """
    ih, iw = x.size()[-2:]
    pad_h = get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w = get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
    x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    return x


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    """Resolve padding configuration to numeric value and dynamic flag.

    Supports string padding modes: 'same' (TF-style), 'valid' (no padding),
    or defaults to symmetric padding. For 'same', chooses between static
    (no overhead) or dynamic (runtime overhead) implementation.

    Args:
        padding: Padding specification (int, str, or tuple).
        kernel_size: Convolution kernel size.
        **kwargs: Additional convolution parameters (stride, dilation).

    Returns:
        Tuple of (padding_value, is_dynamic).
    """
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic
