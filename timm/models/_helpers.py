""" Model creation / weight loading / state_dict helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import os
from typing import Any, Callable, Dict, Optional, Union

import torch

try:
    import safetensors.torch

    _has_safetensors = True
except ImportError:
    _has_safetensors = False

_logger = logging.getLogger(__name__)

__all__ = ['clean_state_dict', 'load_state_dict', 'load_checkpoint', 'remap_state_dict', 'resume_checkpoint']


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
        weights_only: bool = False,
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
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location=device)

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
        weights_only: bool = False,
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
) -> Optional[int]:
    """Resume training from checkpoint.

    Args:
        model: Model to load checkpoint into.
        checkpoint_path: Path to checkpoint file.
        optimizer: Optional optimizer to restore state.
        loss_scaler: Optional AMP loss scaler to restore state.
        log_info: Whether to log loading info.

    Returns:
        Resume epoch number if available, else None.
    """
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
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
