"""Small optimizer helpers shared by timm optimizer implementations."""

import functools
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor
try:
    from torch.optim.optimizer import _default_to_fused_or_foreach
except ImportError:
    _default_to_fused_or_foreach = None


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
                    # a target matching the param dtype and device is already handled (aliased if possible) by the
                    # normal loader
                    if dtype is None or (dtype == param.dtype and torch.device(device) == param.device):
                        continue
                    if is_tensor_value(saved.get(key)):
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


# _foreach_maximum_(TensorList, Scalar) overload used by cautious foreach impls, checked once (not traceable by dynamo)
try:
    _HAS_FOREACH_MAXIMUM_SCALAR = 'Scalar' in torch.ops.aten._foreach_maximum_.overloads()
except Exception:
    _HAS_FOREACH_MAXIMUM_SCALAR = False


def _resolve_foreach(
        foreach: Optional[bool],
        caution: bool = False,
        lr: Union[float, Tensor, None] = None,
        params: Optional[List[Tensor]] = None,
) -> bool:
    """Resolve foreach=None (default) for timm optimizers w/ single and multi-tensor impls."""
    if foreach is not None:
        return foreach
    if caution and not _HAS_FOREACH_MAXIMUM_SCALAR:
        # cannot do foreach if this overload doesn't exist when caution enabled
        return False
    if torch.is_tensor(lr):
        # a tensor lr is supported by the single tensor path
        return False
    if params is None:
        return True
    # match PyTorch, default to foreach for devices w/ foreach (multi-tensor) kernels (e.g. CUDA, not CPU or XLA)
    if _default_to_fused_or_foreach is not None:
        return _default_to_fused_or_foreach(params, differentiable=False, use_fused=False)[1]
    return all(type(p) in (torch.Tensor, torch.nn.Parameter) and p.is_cuda for p in params)


def _foreach_trust_ratios(w_norms: List[Tensor], g_norms: List[Tensor], fn) -> List[Tensor]:
    """Compute per param (layer-wise) trust ratios fn(w_norm, g_norm) w/ one vectorized op per (device, dtype)."""
    ratios: List[Optional[Tensor]] = [None] * len(w_norms)
    groups = {}
    for i, w in enumerate(w_norms):
        groups.setdefault((w.device, w.dtype), []).append(i)
    for idx in groups.values():
        r = fn(torch.stack([w_norms[i] for i in idx]), torch.stack([g_norms[i] for i in idx]))
        for i, ri in zip(idx, r.unbind(0)):
            ratios[i] = ri
    return ratios


# foreach (multi-tensor) step fns are run in chunks of params sized to keep a chunk's working set in the L2 cache,
# processing all params per op would stream every tensor from memory for each op. Max elements per chunk override,
# None = derive from the device L2 cache size, 0 = don't chunk.
_FOREACH_CHUNK_OVERRIDE: Optional[int] = (
    int(os.environ['TIMM_FOREACH_CHUNK_SIZE']) if 'TIMM_FOREACH_CHUNK_SIZE' in os.environ else None)
# the chunk size is derived from the device L2 cache size, available in PyTorch >= 2.5
_HAS_L2_CACHE_SIZE = hasattr(getattr(torch._C, '_CudaDeviceProperties', None), 'L2_cache_size')
_L2_CACHE_SIZE = {}


def _foreach_chunk_elements(device: torch.device, element_size: int) -> Optional[int]:
    """Max elements per foreach chunk so a chunk's ~4 tensor streams stay in the L2 cache, None = don't chunk."""
    if _FOREACH_CHUNK_OVERRIDE is not None:
        return _FOREACH_CHUNK_OVERRIDE or None
    if device.type != 'cuda' or not _HAS_L2_CACHE_SIZE:
        return None
    if device not in _L2_CACHE_SIZE:
        _L2_CACHE_SIZE[device] = torch.cuda.get_device_properties(device).L2_cache_size
    budget = _L2_CACHE_SIZE[device] // (4 * element_size)
    # small chunks (small L2 caches) cost more in per chunk launch overhead than the cache reuse saves,
    # chunks < ~4M elements measured slower than no chunking for large models
    return budget if budget >= 2 ** 22 else None


def _foreach_chunks(params: List[Tensor]) -> List[List[int]]:
    """Consecutive param index chunks w/ total numel <= chunk size, a larger tensor forms its own chunk."""
    budget = _foreach_chunk_elements(params[0].device, params[0].element_size()) if params else None
    if budget is None:
        return [list(range(len(params)))]
    chunks, cur, total = [], [], 0
    for i, p in enumerate(params):
        if cur and total + p.numel() > budget:
            chunks.append(cur)
            cur, total = [], 0
        cur.append(i)
        total += p.numel()
    return chunks + [cur] if cur else chunks


def _foreach_chunked(num_per_param_args: int, writeback: Sequence[int] = ()):
    """Decorator, run a foreach (multi-tensor) step fn per chunk of params, see _foreach_chunks.

    Args:
        num_per_param_args: the first num_per_param_args positional args are per param lists (sliced per chunk).
        writeback: indices of per param lists the step fn assigns elements of (e.g. new momentum buffers), those
            elements are written back to the caller's list.
    """
    def decorator(fn):
        if not _HAS_L2_CACHE_SIZE and not _FOREACH_CHUNK_OVERRIDE:
            return fn  # no L2 cache size in this PyTorch to derive chunks from (and no override), use fn as is

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            per_param, rest = args[:num_per_param_args], args[num_per_param_args:]
            chunks = _foreach_chunks(per_param[0])
            if len(chunks) <= 1:
                return fn(*args, **kwargs)
            for idx in chunks:
                chunk_args = [[v[i] for i in idx] if v else v for v in per_param]
                fn(*chunk_args, *rest, **kwargs)
                for a in writeback:
                    for j, i in enumerate(idx):
                        per_param[a][i] = chunk_args[a][j]
        return wrapper
    return decorator


def _foreach_lerp_(tensors: List[Tensor], ends: List[Tensor], weight: float) -> None:
    """In-place foreach lerp (EMA update), w/ a mul + add fallback for PyTorch < 2.0."""
    if hasattr(torch, '_foreach_lerp_'):
        torch._foreach_lerp_(tensors, ends, weight)
    else:
        torch._foreach_mul_(tensors, 1. - weight)
        torch._foreach_add_(tensors, ends, alpha=weight)


def _foreach_lerp(tensors: List[Tensor], ends: List[Tensor], weight: float) -> List[Tensor]:
    """Out-of-place foreach lerp (EMA update), w/ a mul + add fallback for PyTorch < 2.0."""
    if hasattr(torch, '_foreach_lerp'):
        return torch._foreach_lerp(tensors, ends, weight)
    out = torch._foreach_mul(tensors, 1. - weight)
    torch._foreach_add_(out, ends, alpha=weight)
    return out


# _foreach_add_(TensorList, Tensor, alpha) overload is available in PyTorch >= 2.2
try:
    _HAS_FOREACH_ADD_TENSOR = 'Tensor' in torch.ops.aten._foreach_add_.overloads()
except Exception:
    _HAS_FOREACH_ADD_TENSOR = False


def _foreach_increment_steps(state_steps: List[Tensor]) -> None:
    """Increment step counts. CPU steps use the slow path of foreach, wrapping a 1 for each step tensor, a pre-wrapped
    scalar (w/ alpha to select the right overload) avoids that.
    """
    if not _is_compiling() and state_steps and state_steps[0].is_cpu and _HAS_FOREACH_ADD_TENSOR:
        torch._foreach_add_(state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
    else:
        torch._foreach_add_(state_steps, 1)


def _rms_clip_denom_(denom: Tensor, grad: Tensor, clipping_threshold: float) -> None:
    """Scale an update denominator in place so the update (numerator / denom) is RMS clipped.

    denom *= max(1, rms / clipping_threshold) w/ rms the RMS of the normalized gradient grad / denom, as per
    StableAdamW (https://arxiv.org/abs/2304.13013) and Adafactor update clipping. No host sync.
    """
    rms = grad.div(denom).norm() / (grad.numel() ** 0.5)
    denom.mul_((rms / clipping_threshold).clamp_(min=1.0))


def _foreach_rms_clip_denom_(denoms: List[Tensor], grads: List[Tensor], clipping_threshold: float) -> None:
    """Multi-tensor _rms_clip_denom_, the per tensor clip factors are computed w/ one vectorized op per (device, dtype).
    """
    norms = torch._foreach_norm(torch._foreach_div(grads, denoms))
    # rms / clipping_threshold, per tensor scales as Python scalars (no host to device copy, CUDA graph capture safe)
    torch._foreach_mul_(norms, [1. / (g.numel() ** 0.5 * clipping_threshold) for g in grads])
    clips: List[Optional[Tensor]] = [None] * len(norms)
    groups = {}
    for i, n in enumerate(norms):
        groups.setdefault((n.device, n.dtype), []).append(i)
    for idx in groups.values():
        clip = torch.stack([norms[i] for i in idx]).clamp_(min=1.0)
        for i, c in zip(idx, clip.unbind(0)):
            clips[i] = c
    torch._foreach_mul_(denoms, clips)


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
