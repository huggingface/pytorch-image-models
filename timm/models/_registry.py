""" Model Registry
Hacked together by / Copyright 2020 Ross Wightman
"""

import fnmatch
import re
import sys
import warnings
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union, Tuple

from ._pretrained import PretrainedCfg, DefaultCfg

__all__ = [
    'split_model_name_tag', 'get_arch_name', 'register_model', 'generate_default_cfgs',
    'list_models', 'list_pretrained', 'is_model', 'model_entrypoint', 'list_modules', 'is_model_in_modules',
    'get_pretrained_cfg_value', 'is_model_pretrained', 'get_arch_pretrained_cfgs'
]

_module_to_models: Dict[str, Set[str]] = defaultdict(set)  # dict of sets to check membership of model in module
_model_to_module: Dict[str, str] = {}  # mapping of model names to module names
_model_entrypoints: Dict[str, Callable[..., Any]] = {}  # mapping of model names to architecture entrypoint fns
_model_has_pretrained: Set[str] = set()  # set of model names that have pretrained weight url present
_model_default_cfgs: Dict[str, PretrainedCfg] = {}  # central repo for model arch -> default cfg objects
_model_pretrained_cfgs: Dict[str, PretrainedCfg] = {}  # central repo for model arch.tag -> pretrained cfgs
_model_with_tags: Dict[str, List[str]] = defaultdict(list)  # shortcut to map each model arch to all model + tag names
_module_to_deprecated_models: Dict[str, Dict[str, Optional[str]]] = defaultdict(dict)
_deprecated_models: Dict[str, Optional[str]] = {}


def split_model_name_tag(model_name: str, no_tag: str = '') -> Tuple[str, str]:
    """ Split a model name into architecture and pretrained tag (ie 'resnet50.a1_in1k' -> 'resnet50', 'a1_in1k').

    Args:
        model_name: Model name, with or without a pretrained tag.
        no_tag: Value to return for the tag if the model name has none.

    Returns:
        Tuple of (architecture name, pretrained tag).
    """
    model_name, *tag_list = model_name.split('.', 1)
    tag = tag_list[0] if tag_list else no_tag
    return model_name, tag


def get_arch_name(model_name: str) -> str:
    """ Get the architecture name from a model name, stripping any pretrained tag."""
    return split_model_name_tag(model_name)[0]


def generate_default_cfgs(cfgs: Dict[str, Union[Dict[str, Any], PretrainedCfg]]):
    """ Group 'arch.tag' keyed pretrained cfgs into a DefaultCfg per architecture.

    The first tag w/ weights becomes the default for its architecture, unless an untagged key w/ weights or
    a tag ending in '*' explicitly marks the default.

    Args:
        cfgs: Pretrained cfgs (as dicts or PretrainedCfg) keyed by 'arch' or 'arch.tag'.

    Returns:
        Dict of DefaultCfg keyed by architecture name.
    """
    out = defaultdict(DefaultCfg)
    default_set = set()  # no tag and tags ending with * are prioritized as default

    for k, v in cfgs.items():
        if isinstance(v, dict):
            v = PretrainedCfg(**v)
        has_weights = v.has_weights

        model, tag = split_model_name_tag(k)
        is_default_set = model in default_set
        priority = (has_weights and not tag) or (tag.endswith('*') and not is_default_set)
        tag = tag.strip('*')

        default_cfg = out[model]

        if priority:
            default_cfg.tags.appendleft(tag)
            default_set.add(model)
        elif has_weights and not default_cfg.is_pretrained:
            default_cfg.tags.appendleft(tag)
        else:
            default_cfg.tags.append(tag)

        if has_weights:
            default_cfg.is_pretrained = True

        default_cfg.cfgs[tag] = v

    return out


def _clear_model_entries(model_name: str):
    """ Remove all registry entries for a model name, used when an existing registration is overwritten."""
    old_module = _model_to_module.pop(model_name, None)
    if old_module is not None:
        old_models = _module_to_models.get(old_module)
        if old_models is not None:
            old_models.discard(model_name)
            if not old_models:
                del _module_to_models[old_module]  # don't list modules left w/o models
        if old_module in _module_to_deprecated_models:
            _module_to_deprecated_models[old_module].pop(model_name, None)
    for name in _model_with_tags.pop(model_name, []) + [model_name]:
        _model_pretrained_cfgs.pop(name, None)
        _model_has_pretrained.discard(name)
    _model_default_cfgs.pop(model_name, None)
    _deprecated_models.pop(model_name, None)


def register_model(fn: Callable[..., Any]) -> Callable[..., Any]:
    """ Decorator that registers a model entrypoint fn and any pretrained cfgs in its module's default_cfgs.

    Registering a name that already exists overwrites the previous registration and its pretrained cfgs.

    Args:
        fn: Model entrypoint fn, registered under its __name__.

    Returns:
        The unmodified entrypoint fn.
    """
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        if model_name not in mod.__all__:
            mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]  # type: ignore

    # add entries to registry dict/sets
    if model_name in _model_entrypoints:
        warnings.warn(
            f'Overwriting {model_name} in registry with {fn.__module__}.{model_name}. This is because the name being '
            'registered conflicts with an existing name. Please check if this is not expected.',
            stacklevel=2,
        )
        # remove stale module, tag, and pretrained cfg entries of the previous registration
        _clear_model_entries(model_name)
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
    if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
        # this will catch all models that have entrypoint matching cfg key, but miss any aliasing
        # entrypoints or non-matching combos
        default_cfg = mod.default_cfgs[model_name]
        if not isinstance(default_cfg, DefaultCfg):
            # new style default cfg dataclass w/ multiple entries per model-arch
            assert isinstance(default_cfg, dict)
            # old style cfg dict per model-arch
            pretrained_cfg = PretrainedCfg(**default_cfg)
            default_cfg = DefaultCfg(tags=deque(['']), cfgs={'': pretrained_cfg})

        for tag_idx, tag in enumerate(default_cfg.tags):
            is_default = tag_idx == 0
            pretrained_cfg = default_cfg.cfgs[tag]
            model_name_tag = '.'.join([model_name, tag]) if tag else model_name
            replace_items = dict(architecture=model_name, tag=tag if tag else None)
            if pretrained_cfg.hf_hub_id and pretrained_cfg.hf_hub_id == 'timm/':
                # auto-complete hub name w/ architecture.tag
                replace_items['hf_hub_id'] = pretrained_cfg.hf_hub_id + model_name_tag
            pretrained_cfg = replace(pretrained_cfg, **replace_items)

            if is_default:
                _model_pretrained_cfgs[model_name] = pretrained_cfg
                if pretrained_cfg.has_weights:
                    # add tagless entry if it's default and has weights
                    _model_has_pretrained.add(model_name)

            if tag:
                _model_pretrained_cfgs[model_name_tag] = pretrained_cfg
                if pretrained_cfg.has_weights:
                    # add model w/ tag if tag is valid
                    _model_has_pretrained.add(model_name_tag)
                _model_with_tags[model_name].append(model_name_tag)
            else:
                _model_with_tags[model_name].append(model_name)  # has empty tag (to slowly remove these instances)

        _model_default_cfgs[model_name] = default_cfg

    return fn


def _deprecated_model_shim(deprecated_name: str, current_fn: Callable = None, current_tag: str = ''):
    def _fn(pretrained=False, **kwargs):
        assert current_fn is not None,  f'Model {deprecated_name} has been removed with no replacement.'
        current_name = '.'.join([current_fn.__name__, current_tag]) if current_tag else current_fn.__name__
        warnings.warn(f'Mapping deprecated model name {deprecated_name} to current {current_name}.', stacklevel=2)
        pretrained_cfg = kwargs.pop('pretrained_cfg', None)
        return current_fn(pretrained=pretrained, pretrained_cfg=pretrained_cfg or current_tag, **kwargs)
    return _fn


def register_model_deprecations(module_name: str, deprecation_map: Dict[str, Optional[str]]):
    """ Register deprecated model names that map to current models (w/ a warning) or error if removed.

    Args:
        module_name: Name of the module (ie __name__) containing the current model entrypoints.
        deprecation_map: Mapping of deprecated name to current 'arch' or 'arch.tag' name, None if removed.
    """
    mod = sys.modules[module_name]
    module_name_split = module_name.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    for deprecated, current in deprecation_map.items():
        if hasattr(mod, '__all__') and deprecated not in mod.__all__:
            mod.__all__.append(deprecated)
        current_fn = None
        current_tag = ''
        if current:
            current_name, current_tag = split_model_name_tag(current)
            current_fn = getattr(mod, current_name)
        deprecated_entrypoint_fn = _deprecated_model_shim(deprecated, current_fn, current_tag)
        setattr(mod, deprecated, deprecated_entrypoint_fn)
        if deprecated in _model_entrypoints:
            _clear_model_entries(deprecated)
        _model_entrypoints[deprecated] = deprecated_entrypoint_fn
        _model_to_module[deprecated] = module_name
        _module_to_models[module_name].add(deprecated)
        _deprecated_models[deprecated] = current
        _module_to_deprecated_models[module_name][deprecated] = current


def _natural_key(string_: str) -> List[Union[int, str]]:
    """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _expand_filter(filter: str):
    """ Expand a 'base_filter' to ['base_filter.*', 'base_filter'] if it has no tag portion."""
    filter_base, filter_tag = split_model_name_tag(filter)
    if not filter_tag:
        return ['.'.join([filter_base, '*']), filter]
    else:
        return [filter]


def _to_str_list(value: Union[str, Iterable[str], None]) -> List[str]:
    """ Normalize a single str or an iterable of str to a list (empty for '' or None)."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def list_models(
        filter: Union[str, Iterable[str]] = '',
        module: Union[str, Iterable[str]] = '',
        pretrained: bool = False,
        exclude_filters: Union[str, Iterable[str]] = '',
        name_matches_cfg: bool = False,
        include_tags: Optional[bool] = None,
) -> List[str]:
    """ Return list of available model names, sorted naturally.

    Args:
        filter: Wildcard filter(s) that work with fnmatch.
        module: Limit model selection to specific submodule(s) (ie 'vision_transformer').
        pretrained: Include only models with valid pretrained weights if True.
        exclude_filters: Wildcard filter(s) to exclude models after including them with filter.
        name_matches_cfg: Include only models w/ model_name matching default_cfg name (excludes some aliases).
        include_tags: Include pretrained tags in model names (model.tag). If None, set to the value of pretrained.

    Returns:
        The sorted list of model names.

    Example:
        list_models('gluon_resnet*') -- returns all models starting with 'gluon_resnet'
        list_models('*resnext*', 'resnet') -- returns all models with 'resnext' in 'resnet' module
    """
    include_filters = _to_str_list(filter)
    exclude_filters = _to_str_list(exclude_filters)
    modules = _to_str_list(module)

    if include_tags is None:
        # FIXME should this be default behaviour? or default to include_tags=True?
        include_tags = pretrained

    if not modules:
        all_models: Set[str] = set(_model_entrypoints.keys())
    else:
        all_models: Set[str] = set()
        for m in modules:
            all_models.update(_module_to_models.get(m, ()))
    all_models = all_models - _deprecated_models.keys()  # remove deprecated models from listings

    if include_tags:
        # expand model names to include names w/ pretrained tags
        models_with_tags: Set[str] = set()
        for m in all_models:
            models_with_tags.update(_model_with_tags.get(m) or (m,))  # no cfgs registered, keep untagged name
        all_models = models_with_tags
        # expand include and exclude filters to include a '.*' for proper match if no tags in filter
        include_filters = [ef for f in include_filters for ef in _expand_filter(f)]
        exclude_filters = [ef for f in exclude_filters for ef in _expand_filter(f)]

    if include_filters:
        models: Set[str] = set()
        for f in include_filters:
            include_models = fnmatch.filter(all_models, f)  # include these models
            if len(include_models):
                models = models.union(include_models)
    else:
        models = all_models

    if exclude_filters:
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)  # exclude these models
            if len(exclude_models):
                models = models.difference(exclude_models)

    if pretrained:
        models = _model_has_pretrained.intersection(models)

    if name_matches_cfg:
        models = set(_model_pretrained_cfgs).intersection(models)

    return sorted(models, key=_natural_key)


def list_pretrained(
        filter: Union[str, Iterable[str]] = '',
        exclude_filters: Union[str, Iterable[str]] = '',
) -> List[str]:
    """ Return list of pretrained model names (w/ tags), sorted naturally.

    Args:
        filter: Wildcard filter(s) that work with fnmatch.
        exclude_filters: Wildcard filter(s) to exclude models after including them with filter.

    Returns:
        The sorted list of model names.
    """
    return list_models(
        filter=filter,
        pretrained=True,
        exclude_filters=exclude_filters,
        include_tags=True,
    )


def get_deprecated_models(module: str = '') -> Dict[str, Optional[str]]:
    """ Get deprecated model names mapped to their current names (None if removed).

    Args:
        module: Limit to deprecations registered in a specific submodule, all if empty.

    Returns:
        Dict of deprecated name to current name.
    """
    all_deprecated = _module_to_deprecated_models.get(module, {}) if module else _deprecated_models
    return deepcopy(all_deprecated)


def is_model(model_name: str) -> bool:
    """ Check if a model architecture is registered, ignoring any pretrained tag."""
    arch_name = get_arch_name(model_name)
    return arch_name in _model_entrypoints


def model_entrypoint(model_name: str, module_filter: Optional[str] = None) -> Callable[..., Any]:
    """ Fetch the model entrypoint fn for a model name, ignoring any pretrained tag.

    Args:
        model_name: Model name, with or without a pretrained tag.
        module_filter: If set, raise if the model isn't in this submodule.

    Returns:
        The model entrypoint fn.
    """
    arch_name = get_arch_name(model_name)
    if module_filter and arch_name not in _module_to_models.get(module_filter, {}):
        raise RuntimeError(f'Model ({model_name} not found in module {module_filter}.')
    return _model_entrypoints[arch_name]


def list_modules() -> List[str]:
    """ Return sorted list of module names that contain model entrypoints."""
    modules = _module_to_models.keys()
    return sorted(modules)


def is_model_in_modules(
        model_name: str, module_names: Union[str, Iterable[str]]
) -> bool:
    """ Check if a model exists within a subset of modules.

    Args:
        model_name: Model name to check, with or without a pretrained tag.
        module_names: Name(s) of modules to search in.
    """
    arch_name = get_arch_name(model_name)
    return any(arch_name in _module_to_models.get(n, ()) for n in _to_str_list(module_names))


def is_model_pretrained(model_name: str) -> bool:
    """ Check if a model name (w/ or w/o tag) has pretrained weights registered."""
    return model_name in _model_has_pretrained


def get_pretrained_cfg(model_name: str, allow_unregistered: bool = True) -> Optional[PretrainedCfg]:
    """ Get a copy of the pretrained cfg for a model name, the default tag is used if none specified.

    Args:
        model_name: Model name, with or without a pretrained tag.
        allow_unregistered: Return None instead of raising if the architecture has no pretrained cfgs.

    Returns:
        The pretrained cfg, or None if unregistered and allowed.

    Raises:
        RuntimeError: If the tag is invalid for the architecture, or it has no cfgs and allow_unregistered is False.
    """
    if model_name in _model_pretrained_cfgs:
        return deepcopy(_model_pretrained_cfgs[model_name])
    arch_name, tag = split_model_name_tag(model_name)
    if arch_name in _model_default_cfgs:
        # if model arch exists, but the tag is wrong, error out
        raise RuntimeError(f'Invalid pretrained tag ({tag}) for {arch_name}.')
    if allow_unregistered:
        # if model arch doesn't exist, it has no pretrained_cfg registered, allow a default to be created
        return None
    raise RuntimeError(f'Model architecture ({arch_name}) has no pretrained cfg registered.')


def get_pretrained_cfg_value(model_name: str, cfg_key: str) -> Optional[Any]:
    """ Get a specific pretrained cfg value by key for a model name, None if the key doesn't exist.

    Raises:
        RuntimeError: If the model has no pretrained cfg registered or the tag is invalid.
    """
    cfg = get_pretrained_cfg(model_name, allow_unregistered=False)
    return getattr(cfg, cfg_key, None)


def get_arch_pretrained_cfgs(model_name: str) -> Dict[str, PretrainedCfg]:
    """ Get copies of all pretrained cfgs for a model architecture.

    Args:
        model_name: Model name, any pretrained tag is ignored.

    Returns:
        Dict of pretrained cfgs keyed by model name w/ tag, empty if the architecture is unknown.
    """
    arch_name, _ = split_model_name_tag(model_name)
    model_names = _model_with_tags.get(arch_name, [])
    cfgs = {m: deepcopy(_model_pretrained_cfgs[m]) for m in model_names}
    return cfgs
