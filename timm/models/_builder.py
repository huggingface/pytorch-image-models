import dataclasses
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from torch import nn as nn
from torch.hub import load_state_dict_from_url

from timm.models._features import FeatureListNet, FeatureDictNet, FeatureHookNet, FeatureGetterNet
from timm.models._features_fx import FeatureGraphNet
from timm.models._helpers import load_state_dict
from timm.models._hub import has_hf_hub, download_cached_file, check_cached_file, load_state_dict_from_hf, \
    load_state_dict_from_path, load_custom_from_hf
from timm.models._manipulate import adapt_input_conv
from timm.models._pretrained import PretrainedCfg
from timm.models._prune import adapt_model_from_file
from timm.models._registry import get_pretrained_cfg

_logger = logging.getLogger(__name__)

# Global variables for rarely used pretrained checkpoint download progress and hash check.
# Use set_pretrained_download_progress / set_pretrained_check_hash functions to toggle.
_DOWNLOAD_PROGRESS = False
_CHECK_HASH = False
_USE_OLD_CACHE = int(os.environ.get('TIMM_USE_OLD_CACHE', 0)) > 0

__all__ = [
    'set_pretrained_download_progress',
    'set_pretrained_check_hash',
    'load_custom_pretrained',
    'load_pretrained',
    'pretrained_cfg_for_features',
    'resolve_pretrained_cfg',
    'build_model_with_cfg',
]


ModelT = TypeVar("ModelT", bound=nn.Module)              # any subclass of nn.Module


def _resolve_pretrained_source(pretrained_cfg: Dict[str, Any]) -> Tuple[str, str]:
    cfg_source = pretrained_cfg.get('source', '')
    pretrained_url = pretrained_cfg.get('url', None)
    pretrained_file = pretrained_cfg.get('file', None)
    pretrained_sd = pretrained_cfg.get('state_dict', None)
    hf_hub_id = pretrained_cfg.get('hf_hub_id', None)

    # resolve where to load pretrained weights from
    load_from = ''
    pretrained_loc = ''
    if cfg_source == 'hf-hub' and has_hf_hub(necessary=True):
        # hf-hub specified as source via model identifier
        load_from = 'hf-hub'
        assert hf_hub_id
        pretrained_loc = hf_hub_id
    elif cfg_source == 'local-dir':
        load_from = 'local-dir'
        pretrained_loc = pretrained_file
    else:
        # default source == timm or unspecified
        if pretrained_sd:
            # direct state_dict pass through is the highest priority
            load_from = 'state_dict'
            pretrained_loc = pretrained_sd
            assert isinstance(pretrained_loc, dict)
        elif pretrained_file:
            # file load override is the second-highest priority if set
            load_from = 'file'
            pretrained_loc = pretrained_file
        else:
            old_cache_valid = False
            if _USE_OLD_CACHE:
                # prioritized old cached weights if exists and env var enabled
                old_cache_valid = check_cached_file(pretrained_url) if pretrained_url else False
            if not old_cache_valid and hf_hub_id and has_hf_hub(necessary=True):
                # hf-hub available as alternate weight source in default_cfg
                load_from = 'hf-hub'
                pretrained_loc = hf_hub_id
            elif pretrained_url:
                load_from = 'url'
                pretrained_loc = pretrained_url

    if load_from == 'hf-hub' and pretrained_cfg.get('hf_hub_filename', None):
        # if a filename override is set, return tuple for location w/ (hub_id, filename)
        pretrained_loc = pretrained_loc, pretrained_cfg['hf_hub_filename']
    return load_from, pretrained_loc


def set_pretrained_download_progress(enable: bool = True) -> None:
    """ Set download progress for pretrained weights on/off (globally). """
    global _DOWNLOAD_PROGRESS
    _DOWNLOAD_PROGRESS = enable


def set_pretrained_check_hash(enable: bool = True) -> None:
    """ Set hash checking for pretrained weights on/off (globally). """
    global _CHECK_HASH
    _CHECK_HASH = enable


def load_custom_pretrained(
        model: nn.Module,
        pretrained_cfg: Optional[Dict[str, Any]] = None,
        load_fn: Optional[Callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Loads a custom (read non .pth) weight file

    Downloads checkpoint file into cache-dir like torch.hub based loaders, but calls
    a passed in custom load fun, or the `load_pretrained` model member fn.

    If the object is already present in `model_dir`, it's deserialized and returned.
    The default value of `model_dir` is ``<hub_dir>/checkpoints`` where
    `hub_dir` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        model: The instantiated model to load weights into
        pretrained_cfg: Default pretrained model cfg
        load_fn: An external standalone fn that loads weights into provided model, otherwise a fn named
            'load_pretrained' on the model will be called if it exists
        cache_dir: Override model checkpoint cache dir for this load
    """
    pretrained_cfg = pretrained_cfg or getattr(model, 'pretrained_cfg', None)
    if not pretrained_cfg:
        _logger.warning("Invalid pretrained config, cannot load weights.")
        return

    load_from, pretrained_loc = _resolve_pretrained_source(pretrained_cfg)
    if not load_from:
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return
    if load_from == 'hf-hub':
        _logger.warning("Hugging Face hub not currently supported for custom load pretrained models.")
    elif load_from == 'url':
        pretrained_loc = download_cached_file(
            pretrained_loc,
            check_hash=_CHECK_HASH,
            progress=_DOWNLOAD_PROGRESS,
            cache_dir=cache_dir,
        )

    if load_fn is not None:
        load_fn(model, pretrained_loc)
    elif hasattr(model, 'load_pretrained'):
        model.load_pretrained(pretrained_loc)
    else:
        _logger.warning("Valid function to load pretrained weights is not available, using random initialization.")


def load_pretrained(
        model: nn.Module,
        pretrained_cfg: Optional[Dict[str, Any]] = None,
        num_classes: int = 1000,
        in_chans: int = 3,
        filter_fn: Optional[Callable] = None,
        strict: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
) -> None:
    """ Load pretrained checkpoint

    Args:
        model: PyTorch module
        pretrained_cfg: Configuration for pretrained weights / target dataset
        num_classes: Number of classes for target model. Will adapt pretrained if different.
        in_chans: Number of input chans for target model. Will adapt pretrained if different.
        filter_fn: state_dict filter fn for load (takes state_dict, model as args)
        strict: Strict load of checkpoint
        cache_dir: Override model checkpoint cache dir for this load
    """
    pretrained_cfg = pretrained_cfg or getattr(model, 'pretrained_cfg', None)
    if not pretrained_cfg:
        raise RuntimeError("Invalid pretrained config, cannot load weights. Use `pretrained=False` for random init.")

    load_from, pretrained_loc = _resolve_pretrained_source(pretrained_cfg)
    if load_from == 'state_dict':
        _logger.info(f'Loading pretrained weights from state dict')
        state_dict = pretrained_loc  # pretrained_loc is the actual state dict for this override
    elif load_from == 'file':
        _logger.info(f'Loading pretrained weights from file ({pretrained_loc})')
        if pretrained_cfg.get('custom_load', False):
            model.load_pretrained(pretrained_loc)
            return
        else:
            state_dict = load_state_dict(pretrained_loc)
    elif load_from == 'url':
        _logger.info(f'Loading pretrained weights from url ({pretrained_loc})')
        if pretrained_cfg.get('custom_load', False):
            pretrained_loc = download_cached_file(
                pretrained_loc,
                progress=_DOWNLOAD_PROGRESS,
                check_hash=_CHECK_HASH,
                cache_dir=cache_dir,
            )
            model.load_pretrained(pretrained_loc)
            return
        else:
            try:
                state_dict = load_state_dict_from_url(
                    pretrained_loc,
                    map_location='cpu',
                    progress=_DOWNLOAD_PROGRESS,
                    check_hash=_CHECK_HASH,
                    weights_only=True,
                    model_dir=cache_dir,
                )
            except TypeError:
                state_dict = load_state_dict_from_url(
                    pretrained_loc,
                    map_location='cpu',
                    progress=_DOWNLOAD_PROGRESS,
                    check_hash=_CHECK_HASH,
                    model_dir=cache_dir,
                )
    elif load_from == 'hf-hub':
        _logger.info(f'Loading pretrained weights from Hugging Face hub ({pretrained_loc})')
        if isinstance(pretrained_loc, (list, tuple)):
            custom_load = pretrained_cfg.get('custom_load', False)
            if isinstance(custom_load, str) and custom_load == 'hf':
                load_custom_from_hf(*pretrained_loc, model, cache_dir=cache_dir)
                return
            else:
                state_dict = load_state_dict_from_hf(*pretrained_loc, cache_dir=cache_dir)
        else:
            state_dict = load_state_dict_from_hf(pretrained_loc, weights_only=True, cache_dir=cache_dir)
    elif load_from == 'local-dir':
        _logger.info(f'Loading pretrained weights from local directory ({pretrained_loc})')
        pretrained_path = Path(pretrained_loc)
        if pretrained_path.is_dir():
            state_dict = load_state_dict_from_path(pretrained_path)
        else:
            raise RuntimeError(f"Specified path is not a directory: {pretrained_loc}")
    else:
        model_name = pretrained_cfg.get('architecture', 'this model')
        raise RuntimeError(f"No pretrained weights exist for {model_name}. Use `pretrained=False` for random init.")

    if filter_fn is not None:
        try:
            state_dict = filter_fn(state_dict, model)
        except TypeError as e:
            # for backwards compat with filter fn that take one arg
            state_dict = filter_fn(state_dict)

    input_convs = pretrained_cfg.get('first_conv', None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                _logger.info(
                    f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifiers = pretrained_cfg.get('classifier', None)
    label_offset = pretrained_cfg.get('label_offset', 0)
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != pretrained_cfg['num_classes']:
            for classifier_name in classifiers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                state_dict.pop(classifier_name + '.weight', None)
                state_dict.pop(classifier_name + '.bias', None)
            strict = False
        elif label_offset > 0:
            for classifier_name in classifiers:
                # special case for pretrained weights with an extra background class in pretrained weights
                classifier_weight = state_dict[classifier_name + '.weight']
                state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
                classifier_bias = state_dict[classifier_name + '.bias']
                state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]

    load_result = model.load_state_dict(state_dict, strict=strict)
    if load_result.missing_keys:
        _logger.info(
            f'Missing keys ({", ".join(load_result.missing_keys)}) discovered while loading pretrained weights.'
            f' This is expected if model is being adapted.')
    if load_result.unexpected_keys:
        _logger.warning(
            f'Unexpected keys ({", ".join(load_result.unexpected_keys)}) found while loading pretrained weights.'
            f' This may be expected if model is being adapted.')


def pretrained_cfg_for_features(pretrained_cfg: Dict[str, Any]) -> Dict[str, Any]:
    pretrained_cfg = deepcopy(pretrained_cfg)
    # remove default pretrained cfg fields that don't have much relevance for feature backbone
    to_remove = ('num_classes', 'classifier', 'global_pool')  # add default final pool size?
    for tr in to_remove:
        pretrained_cfg.pop(tr, None)
    return pretrained_cfg


def _filter_kwargs(kwargs: Dict[str, Any], names: List[str]) -> None:
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)


def _update_default_model_kwargs(pretrained_cfg, kwargs, kwargs_filter) -> None:
    """ Update the default_cfg and kwargs before passing to model

    Args:
        pretrained_cfg: input pretrained cfg (updated in-place)
        kwargs: keyword args passed to model build fn (updated in-place)
        kwargs_filter: keyword arg keys that must be removed before model __init__
    """
    # Set model __init__ args that can be determined by default_cfg (if not already passed as kwargs)
    default_kwarg_names = ('num_classes', 'global_pool', 'in_chans')
    if pretrained_cfg.get('fixed_input_size', False):
        # if fixed_input_size exists and is True, model takes an img_size arg that fixes its input size
        default_kwarg_names += ('img_size',)

    for n in default_kwarg_names:
        # for legacy reasons, model __init__args uses img_size + in_chans as separate args while
        # pretrained_cfg has one input_size=(C, H ,W) entry
        if n == 'img_size':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        elif n == 'num_classes':
            default_val = pretrained_cfg.get(n, None)
            # if default is < 0, don't pass through to model
            if default_val is not None and default_val >= 0:
                kwargs.setdefault(n, pretrained_cfg[n])
        else:
            default_val = pretrained_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, pretrained_cfg[n])

    # Filter keyword args for task specific model variants (some 'features only' models, etc.)
    _filter_kwargs(kwargs, names=kwargs_filter)


def resolve_pretrained_cfg(
        variant: str,
        pretrained_cfg: Optional[Union[str, Dict[str, Any]]] = None,
        pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
) -> PretrainedCfg:
    """Resolve pretrained configuration from various sources."""
    model_with_tag = variant
    pretrained_tag = None
    if pretrained_cfg:
        if isinstance(pretrained_cfg, dict):
            # pretrained_cfg dict passed as arg, validate by converting to PretrainedCfg
            pretrained_cfg = PretrainedCfg(**pretrained_cfg)
        elif isinstance(pretrained_cfg, str):
            pretrained_tag = pretrained_cfg
            pretrained_cfg = None

    # fallback to looking up pretrained cfg in model registry by variant identifier
    if not pretrained_cfg:
        if pretrained_tag:
            model_with_tag = '.'.join([variant, pretrained_tag])
        pretrained_cfg = get_pretrained_cfg(model_with_tag)

    if not pretrained_cfg:
        _logger.warning(
            f"No pretrained configuration specified for {model_with_tag} model. Using a default."
            f" Please add a config to the model pretrained_cfg registry or pass explicitly.")
        pretrained_cfg = PretrainedCfg()  # instance with defaults

    pretrained_cfg_overlay = pretrained_cfg_overlay or {}
    if not pretrained_cfg.architecture:
        pretrained_cfg_overlay.setdefault('architecture', variant)
    pretrained_cfg = dataclasses.replace(pretrained_cfg, **pretrained_cfg_overlay)

    return pretrained_cfg


def build_model_with_cfg(
        model_cls: Union[Type[ModelT], Callable[..., ModelT]],
        variant: str,
        pretrained: bool,
        pretrained_cfg: Optional[Dict] = None,
        pretrained_cfg_overlay: Optional[Dict] = None,
        model_cfg: Optional[Any] = None,
        feature_cfg: Optional[Dict] = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Optional[Callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        kwargs_filter: Optional[Tuple[str]] = None,
        **kwargs,
) -> ModelT:
    """ Build model with specified default_cfg and optional model_cfg

    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretrained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation
      * pruning config / model adaptation

    Args:
        model_cls: Model class
        variant: Model variant name
        pretrained: Load the pretrained weights
        pretrained_cfg: Model's pretrained weight/task config
        pretrained_cfg_overlay: Entries that will override those in pretrained_cfg
        model_cfg: Model's architecture config
        feature_cfg: Feature extraction adapter config
        pretrained_strict: Load pretrained weights strictly
        pretrained_filter_fn: Filter callable for pretrained weights
        cache_dir: Override model cache dir for Hugging Face Hub and Torch checkpoints
        kwargs_filter: Kwargs keys to filter (remove) before passing to model
        **kwargs: Model args passed through to model __init__
    """
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}

    # resolve and update model pretrained config and model kwargs
    pretrained_cfg = resolve_pretrained_cfg(
        variant,
        pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay=pretrained_cfg_overlay
    )
    pretrained_cfg = pretrained_cfg.to_dict()

    _update_default_model_kwargs(pretrained_cfg, kwargs, kwargs_filter)

    # Setup for feature extraction wrapper done at end of this fn
    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')
        if 'feature_cls' in kwargs:
            feature_cfg['feature_cls'] = kwargs.pop('feature_cls')

    # Instantiate the model
    if model_cfg is None:
        model = model_cls(**kwargs)
    else:
        model = model_cls(cfg=model_cfg, **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat

    if pruned:
        model = adapt_model_from_file(model, variant)

    # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        load_pretrained(
            model,
            pretrained_cfg=pretrained_cfg,
            num_classes=num_classes_pretrained,
            in_chans=kwargs.get('in_chans', 3),
            filter_fn=pretrained_filter_fn,
            strict=pretrained_strict,
            cache_dir=cache_dir,
        )

    # Wrap the model in a feature extraction module if enabled
    if features:
        use_getter = False
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()

                # flatten_sequential only valid for some feature extractors
                if feature_cls not in ('dict', 'list', 'hook'):
                    feature_cfg.pop('flatten_sequential', None)

                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                elif feature_cls == 'list':
                    feature_cls = FeatureListNet
                elif feature_cls == 'dict':
                    feature_cls = FeatureDictNet
                elif feature_cls == 'fx':
                    feature_cls = FeatureGraphNet
                elif feature_cls == 'getter':
                    use_getter = True
                    feature_cls = FeatureGetterNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        else:
            feature_cls = FeatureListNet

        output_fmt = getattr(model, 'output_fmt', None)
        if output_fmt is not None and not use_getter:  # don't set default for intermediate feat getter
            feature_cfg.setdefault('output_fmt', output_fmt)

        model = feature_cls(model, **feature_cfg)
        model.pretrained_cfg = pretrained_cfg_for_features(pretrained_cfg)  # add back pretrained cfg
        model.default_cfg = model.pretrained_cfg  # alias for rename backwards compat (default_cfg -> pretrained_cfg)

    return model
