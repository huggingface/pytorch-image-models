"""Input resolution for built-in scripts and exported models.

The legacy config resolvers retain their existing behavior for external callers.
"""

import logging
from typing import Any, Dict, Optional

from timm.models._input import _requested_input, get_model_input_config
from .config import resolve_data_config
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


_logger = logging.getLogger(__name__)


def resolve_input_data_config(
        model,
        args: Optional[Dict[str, Any]] = None,
        pretrained_cfg: Optional[Dict[str, Any]] = None,
        use_test_size: bool = False,
        verbose: bool = False,
) -> Dict[str, Any]:
    """Resolve preprocessing against the current model's input constraints.

    Fixed dimensions and channels must agree with explicit requests. Flexible
    models retain train/test preprocessing recommendations unless overridden.
    Normalization uses the prefetch loaders' existing channel adaptation policy.
    """
    args = args or {}
    cfg = pretrained_cfg if pretrained_cfg is not None else getattr(model, 'pretrained_cfg', {})
    current = get_model_input_config(model)
    requested_kwargs = dict(args.get('model_kwargs') or {})
    if not current['fixed_input_size'] and (args.get('img_size') is not None or args.get('input_size') is not None):
        requested_kwargs.pop('img_size', None)
    channels, size = _requested_input(args, requested_kwargs)
    model_channels, *model_size = current['input_size']
    model_size = tuple(model_size)
    if channels is not None and channels != model_channels:
        raise ValueError(f'Requested {channels} input channels, but the model accepts {model_channels}.')
    if current['fixed_input_size']:
        if size is not None and size != model_size:
            raise ValueError(f'Requested image size {size}, but the model requires {model_size}.')
        size = model_size
    elif size is None:
        recommended = cfg.get('test_input_size') if use_test_size else None
        size = tuple((recommended or cfg.get('input_size', current['input_size']))[-2:])

    resolved_args = dict(args, input_size=(model_channels, *size), in_chans=model_channels)
    # Apply the same explicit/default mean/std handling with and without prefetching.
    for key, default in (('mean', IMAGENET_DEFAULT_MEAN), ('std', IMAGENET_DEFAULT_STD)):
        value = args.get(key)
        if value is None:
            value = cfg.get(key)
            if value is None:
                value = default
            if isinstance(value, (int, float)):
                value = (value,)
            if len(value) not in (1, model_channels):
                value = (sum(value) / len(value),)
                _logger.warning('Pretrained %s does not match %s channels; using its mean value.', key, model_channels)
        if value is not None:
            if isinstance(value, (int, float)):
                value = (value,)
            if len(value) not in (1, model_channels):
                raise ValueError(f'{key} must have one value or {model_channels} values, got {len(value)}.')
            resolved_args[key] = tuple(value)
    return resolve_data_config(
        resolved_args,
        pretrained_cfg=cfg,
        model=model,
        use_test_size=use_test_size,
        verbose=verbose,
    )
