"""
BlurPool layer inspired by
 - Kornia's Max_BlurPool2d
 - Making Convolutional Networks Shift-Invariant Again :cite:`zhang2019shiftinvar`

Hacked together by Chris Ha and Ross Wightman
"""
from functools import partial
from math import comb  # Python 3.8
from typing import Callable, Optional, Type, Union

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
            device=None,
            dtype=None
    ) -> None:
        super().__init__()
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.padding = [get_padding(filt_size, stride, dilation=1)] * 4

        # (0.5 + 0.5 x)^N => coefficients = C(N,k) / 2^N,  k = 0..N
        coeffs = torch.tensor(
            [comb(filt_size - 1, k) for k in range(filt_size)],
            device='cpu',
            dtype=torch.float32,
        ) / (2 ** (filt_size - 1))  # normalise so coefficients sum to 1
        blur_filter = (coeffs[:, None] * coeffs[None, :])[None, None, :, :]
        if channels is not None:
            blur_filter = blur_filter.repeat(self.channels, 1, 1, 1)

        self.register_buffer(
            'filt',
            blur_filter.to(device=device, dtype=dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding, mode=self.pad_mode)
        if self.channels is None:
            channels = x.shape[1]
            weight = self.filt.expand(channels, 1, self.filt_size, self.filt_size)
        else:
            channels = self.channels
            weight = self.filt
        return F.conv2d(x, weight, stride=self.stride, groups=channels)


def _normalize_aa_layer(aa_layer: LayerType) -> Callable[..., nn.Module]:
    """Map string shorthands to callables (class or partial)."""
    if isinstance(aa_layer, str):
        key = aa_layer.lower().replace('_', '').replace('-', '')
        if key in ('avg', 'avgpool'):
            return nn.AvgPool2d
        if key in ('blur', 'blurpool'):
            return BlurPool2d
        if key == 'blurpc':
            # preconfigure a constant-pad BlurPool2d
            return partial(BlurPool2d, pad_mode='constant')
        raise AssertionError(f"Unknown anti-aliasing layer ({aa_layer}).")
    return aa_layer


def _underlying_cls(layer_callable: Callable[..., nn.Module]):
    """Return the class behind a callable (unwrap partial), else None."""
    if isinstance(layer_callable, partial):
        return layer_callable.func
    return layer_callable if isinstance(layer_callable, type) else None


def _is_blurpool(layer_callable: Callable[..., nn.Module]) -> bool:
    """True if callable is BlurPool2d or a partial of it."""
    cls = _underlying_cls(layer_callable)
    try:
        return issubclass(cls, BlurPool2d)  # cls may be None, protect below
    except TypeError:
        return False
    except Exception:
        return False


def create_aa(
        aa_layer: LayerType,
        channels: Optional[int] = None,
        stride: int = 2,
        enable: bool = True,
        noop: Optional[Type[nn.Module]] = nn.Identity,
        device=None,
        dtype=None,
) -> Optional[nn.Module]:
    """ Anti-aliasing factory that supports strings, classes, and partials. """
    if not aa_layer or not enable:
        return noop() if noop is not None else None

    # Resolve strings to callables
    aa_layer = _normalize_aa_layer(aa_layer)

    # Build kwargs we *intend* to pass
    call_kwargs = {"channels": channels, "stride": stride}

    # Only add device/dtype for BlurPool2d (or partial of it) and don't override if already provided in the partial.
    if _is_blurpool(aa_layer):
        # Check if aa_layer is a partial and already has device/dtype set
        existing_kw = aa_layer.keywords if isinstance(aa_layer, partial) and aa_layer.keywords else {}
        if "device" not in existing_kw and device is not None:
            call_kwargs["device"] = device
        if "dtype" not in existing_kw and dtype is not None:
            call_kwargs["dtype"] = dtype

    # Try (channels, stride, [device, dtype]) first; fall back to (stride) only
    try:
        return aa_layer(**call_kwargs)
    except TypeError:
        # Some layers (e.g., AvgPool2d) may not accept 'channels' and need stride passed as kernel
        return aa_layer(stride)
