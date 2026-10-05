"""Small optimizer helpers shared by timm optimizer implementations."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor


def _load_state_dict_preserving_dtypes(
        optimizer: torch.optim.Optimizer,
        state_dict: Dict[str, Any],
        load_state_dict: Callable[[Dict[str, Any]], None],
        state_dtypes: Callable[[Dict[str, Any], Tensor], Dict[str, Any]],
) -> None:
    """Load optimizer state, restoring selected tensors from the uncast checkpoint.

    The dtype callback maps state keys to their target dtypes, or (dtype, device) for state that does not live on
    the param device; None (or the param dtype) leaves a key to the normal loader.
    Tensor lists (e.g. Kron's Q) are restored elementwise. Copies avoid aliasing the saved tensors. Non-tensor
    values (e.g. Python scalars in older checkpoints) are left to the normal loader / __setstate__.
    """
    source = state_dict

    def is_tensor_value(value: Any) -> bool:
        if isinstance(value, Tensor):
            return True
        return isinstance(value, (list, tuple)) and len(value) > 0 and all(isinstance(v, Tensor) for v in value)

    def copy_value(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
        if isinstance(value, Tensor):
            return value.to(device=device, dtype=dtype, copy=True)
        return type(value)(copy_value(v, device, dtype) for v in value)

    def capture_source(_optimizer: torch.optim.Optimizer, saved: Dict[str, Any]) -> None:
        nonlocal source
        source = saved

    def restore(loaded: torch.optim.Optimizer) -> None:
        for group, saved_group in zip(loaded.param_groups, source['param_groups']):
            for param, param_id in zip(group['params'], saved_group['params']):
                saved = source['state'].get(param_id, {})
                for key, target in state_dtypes(group, param).items():
                    dtype, device = target if isinstance(target, tuple) else (target, param.device)
                    # a target matching the param dtype is already handled (aliased if possible) by the normal loader
                    if dtype is not None and dtype != param.dtype and is_tensor_value(saved.get(key)):
                        loaded.state[param][key] = copy_value(saved[key], device, dtype)

    if (
        hasattr(optimizer, 'register_load_state_dict_pre_hook')
        and hasattr(optimizer, 'register_load_state_dict_post_hook')
    ):
        # Capture after user pre-hooks (which can remap params), and restore before user post-hooks.
        pre_handle = optimizer.register_load_state_dict_pre_hook(capture_source)
        post_handle = optimizer.register_load_state_dict_post_hook(restore, prepend=True)
        try:
            load_state_dict(state_dict)
        finally:
            pre_handle.remove()
            post_handle.remove()
    else:
        # Older PyTorch has no optimizer load hooks.
        load_state_dict(state_dict)
        restore(optimizer)


def _get_scalar_dtype() -> torch.dtype:
    return torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32


def _max_lr_snapshot(lr: Union[float, Tensor], corrected_weight_decay: bool = False) -> float:
    """Snapshot the constructor lr as the max lr for corrected weight decay (lr ** 2 / max_lr).

    A float copy, so a tensor lr updated in place (e.g. for CUDA graphs) does not alias it.
    """
    max_lr = float(lr.detach()) if isinstance(lr, Tensor) else float(lr)
    if corrected_weight_decay and max_lr <= 0:
        raise ValueError(
            f'Corrected weight decay requires a positive max lr, got {max_lr}. Construct the optimizer with the '
            f'peak lr.')
    return max_lr


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
    if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'is_compiling'):
        return torch._dynamo.is_compiling()
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
