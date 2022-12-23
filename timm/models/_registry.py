""" Model Registry
Hacked together by / Copyright 2020 Ross Wightman
"""

import fnmatch
import re
import sys
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import replace
from typing import List, Optional, Union, Tuple

from ._pretrained import PretrainedCfg, DefaultCfg, split_model_name_tag

__all__ = [
    'list_models', 'list_pretrained', 'is_model', 'model_entrypoint', 'list_modules', 'is_model_in_modules',
    'get_pretrained_cfg_value', 'is_model_pretrained', 'get_arch_name']

_module_to_models = defaultdict(set)  # dict of sets to check membership of model in module
_model_to_module = {}  # mapping of model names to module names
_model_entrypoints = {}  # mapping of model names to architecture entrypoint fns
_model_has_pretrained = set()  # set of model names that have pretrained weight url present
_model_default_cfgs = dict()  # central repo for model arch -> default cfg objects
_model_pretrained_cfgs = dict()  # central repo for model arch.tag -> pretrained cfgs
_model_with_tags = defaultdict(list)  # shortcut to map each model arch to all model + tag names


def get_arch_name(model_name: str) -> Tuple[str, Optional[str]]:
    return split_model_name_tag(model_name)[0]


def register_model(fn):
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # add entries to registry dict/sets
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


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def list_models(
        filter: Union[str, List[str]] = '',
        module: str = '',
        pretrained=False,
        exclude_filters: str = '',
        name_matches_cfg: bool = False,
        include_tags: Optional[bool] = None,
):
    """ Return list of available model names, sorted alphabetically

    Args:
        filter (str) - Wildcard filter string that works with fnmatch
        module (str) - Limit model selection to a specific submodule (ie 'vision_transformer')
        pretrained (bool) - Include only models with valid pretrained weights if True
        exclude_filters (str or list[str]) - Wildcard filters to exclude models after including them with filter
        name_matches_cfg (bool) - Include only models w/ model_name matching default_cfg name (excludes some aliases)
        include_tags (Optional[boo]) - Include pretrained tags in model names (model.tag). If None, defaults
            set to True when pretrained=True else False (default: None)
    Example:
        model_list('gluon_resnet*') -- returns all models starting with 'gluon_resnet'
        model_list('*resnext*, 'resnet') -- returns all models with 'resnext' in 'resnet' module
    """
    if include_tags is None:
        # FIXME should this be default behaviour? or default to include_tags=True?
        include_tags = pretrained

    if module:
        all_models = list(_module_to_models[module])
    else:
        all_models = _model_entrypoints.keys()

    if include_tags:
        # expand model names to include names w/ pretrained tags
        models_with_tags = []
        for m in all_models:
            models_with_tags.extend(_model_with_tags[m])
        all_models = models_with_tags

    if filter:
        models = []
        include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
        for f in include_filters:
            include_models = fnmatch.filter(all_models, f)  # include these models
            if len(include_models):
                models = set(models).union(include_models)
    else:
        models = all_models

    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)  # exclude these models
            if len(exclude_models):
                models = set(models).difference(exclude_models)

    if pretrained:
        models = _model_has_pretrained.intersection(models)

    if name_matches_cfg:
        models = set(_model_pretrained_cfgs).intersection(models)

    return list(sorted(models, key=_natural_key))


def list_pretrained(
        filter: Union[str, List[str]] = '',
        exclude_filters: str = '',
):
    return list_models(
        filter=filter,
        pretrained=True,
        exclude_filters=exclude_filters,
        include_tags=True,
    )


def is_model(model_name):
    """ Check if a model name exists
    """
    arch_name = get_arch_name(model_name)
    return arch_name in _model_entrypoints


def model_entrypoint(model_name, module_filter: Optional[str] = None):
    """Fetch a model entrypoint for specified model name
    """
    arch_name = get_arch_name(model_name)
    if module_filter and arch_name not in _module_to_models.get(module_filter, {}):
        raise RuntimeError(f'Model ({model_name} not found in module {module_filter}.')
    return _model_entrypoints[arch_name]


def list_modules():
    """ Return list of module names that contain models / model entrypoints
    """
    modules = _module_to_models.keys()
    return list(sorted(modules))


def is_model_in_modules(model_name, module_names):
    """Check if a model exists within a subset of modules
    Args:
        model_name (str) - name of model to check
        module_names (tuple, list, set) - names of modules to search in
    """
    arch_name = get_arch_name(model_name)
    assert isinstance(module_names, (tuple, list, set))
    return any(arch_name in _module_to_models[n] for n in module_names)


def is_model_pretrained(model_name):
    return model_name in _model_has_pretrained


def get_pretrained_cfg(model_name, allow_unregistered=True):
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


def get_pretrained_cfg_value(model_name, cfg_key):
    """ Get a specific model default_cfg value by key. None if key doesn't exist.
    """
    cfg = get_pretrained_cfg(model_name, allow_unregistered=False)
    return getattr(cfg, cfg_key, None)
