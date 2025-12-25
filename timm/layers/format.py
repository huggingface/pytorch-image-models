from enum import Enum
from typing import Union

import torch


class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'


FormatT = Union[str, Format]


def get_spatial_dim(fmt: FormatT):
    """Return spatial dimension indices for a given tensor format.

    Args:
        fmt: Tensor format (NCHW, NHWC, NCL, or NLC).

    Returns:
        Tuple of spatial dimension indices.
    """
    fmt = Format(fmt)
    if fmt is Format.NLC:
        dim = (1,)
    elif fmt is Format.NCL:
        dim = (2,)
    elif fmt is Format.NHWC:
        dim = (1, 2)
    else:
        dim = (2, 3)
    return dim


def get_channel_dim(fmt: FormatT):
    """Return channel dimension index for a given tensor format.

    Args:
        fmt: Tensor format (NCHW, NHWC, NCL, or NLC).

    Returns:
        Channel dimension index.
    """
    fmt = Format(fmt)
    if fmt is Format.NHWC:
        dim = 3
    elif fmt is Format.NLC:
        dim = 2
    else:
        dim = 1
    return dim


def nchw_to(x: torch.Tensor, fmt: Format):
    """Convert tensor from NCHW format to specified format.

    Args:
        x: Input tensor in NCHW format.
        fmt: Target format.

    Returns:
        Tensor in target format.
    """
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


def nhwc_to(x: torch.Tensor, fmt: Format):
    """Convert tensor from NHWC format to specified format.

    Args:
        x: Input tensor in NHWC format.
        fmt: Target format.

    Returns:
        Tensor in target format.
    """
    if fmt == Format.NCHW:
        x = x.permute(0, 3, 1, 2)
    elif fmt == Format.NLC:
        x = x.flatten(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(1, 2).transpose(1, 2)
    return x
