"""Small optimizer helpers shared by timm optimizer implementations."""

from typing import List, Optional, Sequence, Union

import torch
from torch import Tensor


def _get_scalar_dtype() -> torch.dtype:
    return torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32


def _init_scalar(
        value: Union[float, Tensor] = 0.0,
        device=None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    if isinstance(value, Tensor):
        dtype = dtype or value.dtype
        device = value.device if device is None else torch.device(device)
        if value.device == device and value.dtype == dtype:
            return value
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, dtype=dtype or _get_scalar_dtype(), device=device)


def _zeros_scalar(device=None, dtype: Optional[torch.dtype] = None) -> Tensor:
    return torch.zeros((), dtype=dtype or _get_scalar_dtype(), device=device)


def _is_compiling() -> bool:
    if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'is_compiling'):
        return torch.compiler.is_compiling()
    return False


def _get_value(x):
    # item() is faster for eager CPU scalar-tensor state, but causes specialization under torch.compile.
    if not torch.jit.is_scripting() and _is_compiling():
        return x
    return x.item() if isinstance(x, torch.Tensor) else x


def _get_capturable_supported_devices(supports_xla: bool = True) -> List[str]:
    capturable_supported_devices = ["cuda", "xpu", "hpu"]
    if not torch.jit.is_scripting():
        try:
            capturable_supported_devices.append(torch._C._get_privateuse1_backend_name())
        except AttributeError:
            pass
    if supports_xla:
        capturable_supported_devices.append("xla")
    return capturable_supported_devices


def _check_capturable_devices(
        params: Sequence[Tensor],
        state_steps: Sequence[Tensor],
        supports_xla: bool = True,
) -> None:
    capturable_supported_devices = _get_capturable_supported_devices(supports_xla=supports_xla)
    assert all(
        p.device.type == step.device.type and p.device.type in capturable_supported_devices
        for p, step in zip(params, state_steps)
    ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."


def _validate_scalar(name: str, value, min_value: float = 0.0, max_value: Optional[float] = None) -> None:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar or scalar tensor.")
        value_float = float(value.detach().cpu())
    else:
        value_float = float(value)
    if value_float < min_value or (max_value is not None and value_float >= max_value):
        raise ValueError(f"Invalid {name}: {value}")


def _add_scaled_(param: Tensor, update: Tensor, scale) -> None:
    if torch.is_tensor(scale):
        param.add_(update * scale)
    else:
        param.add_(update, alpha=scale)


def _addcdiv_scaled_(param: Tensor, tensor1: Tensor, tensor2: Tensor, scale) -> None:
    if torch.is_tensor(scale):
        param.add_(tensor1 / tensor2 * scale)
    else:
        param.addcdiv_(tensor1, tensor2, value=scale)
