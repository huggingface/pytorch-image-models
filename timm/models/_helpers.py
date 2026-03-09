""" Model creation / weight loading / state_dict helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
import argparse
import logging
import os
import pickle
from typing import Any, Callable, Dict, Optional, Union

import torch

try:
    import safetensors.torch

    _has_safetensors = True
except ImportError:
    _has_safetensors = False

_logger = logging.getLogger(__name__)

__all__ = [
    'clean_state_dict',
    'load_checkpoint',
    'load_state_dict',
    'remap_state_dict',
    'resume_checkpoint',
]


def _checkpoint_unsafe_globals(checkpoint_path: str) -> str:
    if not hasattr(torch.serialization, 'get_unsafe_globals_in_checkpoint'):
        return ''
    try:
        unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(str(checkpoint_path))
    except Exception:
        unsafe_globals = []
    return f" Unsupported globals: {', '.join(unsafe_globals)}." if unsafe_globals else ''


def _torch_load(
        checkpoint_path: str,
        map_location: Union[str, torch.device] = 'cpu',
        weights_only: bool = True,
):
    use_safe_globals = weights_only and hasattr(torch.serialization, 'safe_globals')
    try:
        if use_safe_globals:
            # Compatibility: timm training checkpoints often include argparse.Namespace in `args`.
            with torch.serialization.safe_globals([argparse.Namespace]):
                return torch.load(checkpoint_path, map_location=map_location, weights_only=weights_only)
        return torch.load(checkpoint_path, map_location=map_location, weights_only=weights_only)
    except TypeError as e:
        if not weights_only:
            return torch.load(checkpoint_path, map_location=map_location)
        raise RuntimeError(
            f"weights_only=True is not supported by this PyTorch build (torch=={torch.__version__}). "
            "No automatic unsafe pickle fallback is performed. "
            "Upgrade PyTorch, or explicitly set weights_only=False only for trusted local checkpoints."
        ) from e
    except pickle.UnpicklingError as e:
        if not weights_only:
            raise
        raise RuntimeError(
            "weights_only=True blocked loading this checkpoint because it requires non-allowlisted pickle globals."
            f"{_checkpoint_unsafe_globals(checkpoint_path)} "
            "No automatic unsafe pickle fallback is performed. "
            "If this checkpoint is trusted, retry with weights_only=False."
        ) from e


def _remove_prefix(text: str, prefix: str) -> str:
    # FIXME replace with 3.9 stdlib fn when min at 3.9
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = {}
    to_remove = (
        'module.',  # DDP wrapper
        '_orig_mod.',  # torchcompile dynamo wrapper
    )
    for k, v in state_dict.items():
        for r in to_remove:
            k = _remove_prefix(k, r)
        cleaned_state_dict[k] = v
    return cleaned_state_dict


def load_state_dict(
        checkpoint_path: str,
        use_ema: bool = True,
        device: Union[str, torch.device] = 'cpu',
        weights_only: bool = True,
) -> Dict[str, Any]:
    """Load state dictionary from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file.
        use_ema: Whether to use EMA weights if available.
        device: Device to load checkpoint to.
        weights_only: Whether to load only weights (torch.load parameter).

    Returns:
        State dictionary loaded from checkpoint.
    """
    if checkpoint_path and os.path.isfile(checkpoint_path):
        # Check if safetensors or not and load weights accordingly
        if str(checkpoint_path).endswith(".safetensors"):
            assert _has_safetensors, "`pip install safetensors` to use .safetensors"
            checkpoint = safetensors.torch.load_file(checkpoint_path, device=device)
        else:
            checkpoint = _torch_load(checkpoint_path, map_location=device, weights_only=weights_only)

        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        state_dict = clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(
        model: torch.nn.Module,
        checkpoint_path: str,
        use_ema: bool = True,
        device: Union[str, torch.device] = 'cpu',
        strict: bool = True,
        remap: bool = False,
        filter_fn: Optional[Callable] = None,
        weights_only: bool = True,
) -> Any:
    """Load checkpoint into model.

    Args:
        model: Model to load checkpoint into.
        checkpoint_path: Path to checkpoint file.
        use_ema: Whether to use EMA weights if available.
        device: Device to load checkpoint to.
        strict: Whether to strictly enforce state_dict keys match.
        remap: Whether to remap state dict keys by order.
        filter_fn: Optional function to filter state dict.
        weights_only: Whether to load only weights (torch.load parameter).

    Returns:
        Incompatible keys from model.load_state_dict().
    """
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return

    state_dict = load_state_dict(checkpoint_path, use_ema, device=device, weights_only=weights_only)
    if remap:
        state_dict = remap_state_dict(state_dict, model)
    elif filter_fn:
        state_dict = filter_fn(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def remap_state_dict(
        state_dict: Dict[str, Any],
        model: torch.nn.Module,
        allow_reshape: bool = True
) -> Dict[str, Any]:
    """Remap checkpoint by iterating over state dicts in order (ignoring original keys).

    This assumes models (and originating state dict) were created with params registered in same order.

    Args:
        state_dict: State dict to remap.
        model: Model whose state dict keys to use.
        allow_reshape: Whether to allow reshaping tensors to match.

    Returns:
        Remapped state dictionary.
    """
    out_dict = {}
    for (ka, va), (kb, vb) in zip(model.state_dict().items(), state_dict.items()):
        assert va.numel() == vb.numel(), f'Tensor size mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed.'
        if va.shape != vb.shape:
            if allow_reshape:
                vb = vb.reshape(va.shape)
            else:
                assert False, f'Tensor shape mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed.'
        out_dict[ka] = vb
    return out_dict


def resume_checkpoint(
        model: torch.nn.Module,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_scaler: Optional[Any] = None,
        log_info: bool = True,
        weights_only: bool = True,
) -> Optional[int]:
    """Resume training from checkpoint.

    Args:
        model: Model to load checkpoint into.
        checkpoint_path: Path to checkpoint file.
        optimizer: Optional optimizer to restore state.
        loss_scaler: Optional AMP loss scaler to restore state.
        log_info: Whether to log loading info.
        weights_only: Whether to load only weights via torch.load.

    Returns:
        Resume epoch number if available, else None.
    """
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = _torch_load(checkpoint_path, map_location='cpu', weights_only=weights_only)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            state_dict = clean_state_dict(checkpoint['state_dict'])
            model.load_state_dict(state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

                if log_info:
                    _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()
