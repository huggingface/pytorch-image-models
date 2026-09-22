"""Current model traits and construction arguments, separate from source checkpoint metadata."""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

from torch import Tensor, nn


def _spatial_size(size: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(size, int):
        size = (size, size)
    size = tuple(size)
    if len(size) != 2 or any(s <= 0 for s in size):
        raise ValueError(f'Expected a positive image size (H, W), got {size}.')
    return size


def _requested_input(
        args: Optional[Dict[str, Any]],
        model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
    args = args or {}
    model_kwargs = model_kwargs or {}
    channels = {}
    sizes = {}
    if args.get('input_size') is not None:
        size = tuple(args['input_size'])
        if len(size) != 3:
            raise ValueError('input_size must contain (C, H, W).')
        channels['input_size'] = size[0]
        sizes['input_size'] = _spatial_size(size[1:])
    for key in ('in_chans', 'chans'):
        if args.get(key) is not None:
            channels[key] = args[key]
    if args.get('img_size') is not None:
        sizes['img_size'] = _spatial_size(args['img_size'])
    if model_kwargs.get('in_chans') is not None:
        channels['model_kwargs.in_chans'] = model_kwargs['in_chans']
    if model_kwargs.get('img_size') is not None:
        sizes['model_kwargs.img_size'] = _spatial_size(model_kwargs['img_size'])
    if len(set(channels.values())) > 1:
        raise ValueError(f'Conflicting input channel counts: {channels}.')
    if len(set(sizes.values())) > 1:
        raise ValueError(f'Conflicting input image sizes: {sizes}.')
    channel_count = next(iter(channels.values()), None)
    if channel_count is not None and channel_count < 1:
        raise ValueError('Input channel count must be positive.')
    return channel_count, next(iter(sizes.values()), None)


def _is_fixed_input_size(model: nn.Module, kwargs: Dict[str, Any], pretrained_cfg: Dict[str, Any]) -> bool:
    traits = getattr(model, '_traits', {})
    if 'fixed_input_size' in traits:
        fixed = traits['fixed_input_size']
    else:
        patch_embed = getattr(model, 'patch_embed', None)
        fixed = getattr(patch_embed, 'strict_img_size', pretrained_cfg.get('fixed_input_size', False))
    return bool(
        fixed
        and not kwargs.get('dynamic_img_size', getattr(model, 'dynamic_img_size', False))
        and kwargs.get('strict_img_size', True)
    )


def _init_model_traits(model: nn.Module, kwargs: Dict[str, Any], pretrained_cfg: Dict[str, Any]) -> None:
    """Initialize instance-owned traits, preserving declarations made by the model.

    An explicit ``first_conv=None`` disables automatic input-stem adaptation.
    """
    traits = dict(getattr(model, '_traits', {}))
    size = kwargs.get('img_size', traits.get('img_size'))
    traits['img_size'] = _spatial_size(size) if size is not None else None
    traits['fixed_input_size'] = _is_fixed_input_size(model, kwargs, pretrained_cfg)
    traits.setdefault('first_conv', pretrained_cfg.get('first_conv'))
    if isinstance(traits['first_conv'], list):
        traits['first_conv'] = tuple(traits['first_conv'])
    model._traits = traits


def update_model_input_size(
        model: nn.Module,
        img_size: Optional[Union[int, Sequence[int]]],
        **model_args: Any,
) -> None:
    """Record a supported model resize without changing the source checkpoint metadata."""
    size = _spatial_size(img_size) if img_size is not None else None
    model._traits = dict(getattr(model, '_traits', {}), img_size=size)
    args = dict(getattr(model, '_model_args', {}))
    args.update({k: v for k, v in model_args.items() if v is not None})
    if size is not None:
        args['img_size'] = size
    else:
        args.pop('img_size', None)
    model._model_args = args


def get_model_traits(model: nn.Module) -> Dict[str, Any]:
    """Return a copy of the model's traits, augmented with current channel and class counts.

    Existing ``in_chans`` and ``num_classes`` attributes remain authoritative;
    they are read here instead of being duplicated in the stored traits.
    Attributes absent from a feature wrapper are omitted.
    """
    traits = dict(getattr(model, '_traits', {}))
    traits.update({k: getattr(model, k) for k in ('in_chans', 'num_classes') if hasattr(model, k)})
    return traits


def get_model_input_config(model: nn.Module) -> Dict[str, Any]:
    """Return the current image channels, nominal size, and fixed-size constraint.

    The nominal size is a recommendation for flexible models. Source checkpoint
    metadata remains in ``model.pretrained_cfg`` and is never modified here.
    """
    cfg = getattr(model, 'pretrained_cfg', {})
    traits = get_model_traits(model)
    source_size = cfg.get('input_size', (3, 224, 224))
    size = traits.get('img_size') or source_size[-2:]
    channels = traits.get('in_chans', source_size[0])
    fixed = traits.get('fixed_input_size', cfg.get('fixed_input_size', False))
    return dict(
        input_size=(channels, *size),
        fixed_input_size=fixed,
        first_conv=traits.get('first_conv', cfg.get('first_conv')),
    )


def get_pretrained_grid_size(
        model: nn.Module,
        patch_weight: Optional[Tensor],
        num_tokens: int,
) -> Optional[Tuple[int, int]]:
    """Recover a source positional grid from checkpoint construction metadata and its patch kernel."""
    if patch_weight is None or patch_weight.ndim != 4:
        return None
    source_args = getattr(model, 'pretrained_model_args', {})
    size = source_args.get('img_size')
    if size is None:
        size = getattr(model, 'pretrained_cfg', {}).get('input_size', (3, 224, 224))[-2:]
    size = _spatial_size(size)
    grid = tuple(s // p for s, p in zip(size, patch_weight.shape[-2:]))
    # Older/custom checkpoints may lack accurate size metadata. Preserve the
    # caller's existing fallback unless the source grid accounts for every token.
    return grid if grid[0] * grid[1] == num_tokens else None


def get_model_args(model: nn.Module, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return reconstruction arguments, including supported changes made after construction.

    Runtime arguments are excluded by the factory. Drop rates are omitted unless
    explicitly supplied in overrides. Non-serializable architecture arguments
    require an explicit serializable replacement when exporting.
    """
    args = {
        k: v
        for k, v in getattr(model, '_model_args', {}).items()
        if k not in ('drop_rate', 'drop_path_rate', 'drop_block_rate') and not k.endswith('_drop_rate')
    }
    args.update(overrides or {})
    traits = get_model_traits(model)
    current = {k: traits[k] for k in ('in_chans', 'num_classes') if k in traits}
    pool = getattr(model, 'global_pool', getattr(getattr(model, 'head', None), 'global_pool', None))
    pool = getattr(pool, 'pool_type', pool)
    if isinstance(pool, str):
        current['global_pool'] = pool
    if traits.get('img_size') is not None:
        current['img_size'] = traits['img_size']
    for key, value in current.items():
        if overrides and key in overrides:
            supplied = _spatial_size(overrides[key]) if key == 'img_size' else overrides[key]
            if supplied != value:
                raise ValueError(f'Export argument {key}={supplied} disagrees with the current model ({value}).')
        args[key] = value
    return args
