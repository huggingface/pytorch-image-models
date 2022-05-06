""" Model Registry
Hacked together by / Copyright 2020 Ross Wightman
"""

import sys
import re
import fnmatch
from collections import defaultdict
from copy import deepcopy

__all__ = ['list_models', 'is_model', 'model_entrypoint', 'list_modules', 'is_model_in_modules',
           'is_pretrained_cfg_key', 'has_pretrained_cfg_key', 'get_pretrained_cfg_value', 'is_model_pretrained']

_module_to_models = defaultdict(set)  # dict of sets to check membership of model in module
_model_to_module = {}  # mapping of model names to module names
_model_entrypoints = {}  # mapping of model names to entrypoint fns
_model_has_pretrained = set()  # set of model names that have pretrained weight url present
_model_pretrained_cfgs = dict()  # central repo for model default_cfgs


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
    has_valid_pretrained = False  # check if model has a pretrained url to allow filtering on this
    if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
        # this will catch all models that have entrypoint matching cfg key, but miss any aliasing
        # entrypoints or non-matching combos
        cfg = mod.default_cfgs[model_name]
        has_valid_pretrained = (
            ('url' in cfg and 'http' in cfg['url']) or
            ('file' in cfg and cfg['file']) or
            ('hf_hub_id' in cfg and cfg['hf_hub_id'])
        )
        _model_pretrained_cfgs[model_name] = mod.default_cfgs[model_name]
    if has_valid_pretrained:
        _model_has_pretrained.add(model_name)
    return fn


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def list_models(filter='', module='', pretrained=False, exclude_filters='', name_matches_cfg=False):
    """ Return list of available model names, sorted alphabetically

    Args:
        filter (str) - Wildcard filter string that works with fnmatch
        module (str) - Limit model selection to a specific sub-module (ie 'gen_efficientnet')
        pretrained (bool) - Include only models with pretrained weights if True
        exclude_filters (str or list[str]) - Wildcard filters to exclude models after including them with filter
        name_matches_cfg (bool) - Include only models w/ model_name matching default_cfg name (excludes some aliases)

    Example:
        list_models('gluon_resnet*') -- returns all models starting with 'gluon_resnet'
        list_models('*resnext*, 'resnet') -- returns all models with 'resnext' in 'resnet' module
    """
    if module:
        all_models = list(_module_to_models[module])
    else:
        all_models = _model_entrypoints.keys()
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


def list_benchmarks(filter='', module='', pretrained=False, exclude_filters='', name_matches_cfg=False):
    """ Return list of available benchmarks on imagenet, sorted alphabetically

    Args:
        filter (str) - Wildcard filter string that works with fnmatch
        module (str) - Limit model selection to a specific sub-module (ie 'gen_efficientnet')
        pretrained (bool) - Include only models with pretrained weights if True
        exclude_filters (str or list[str]) - Wildcard filters to exclude models after including them with filter
        name_matches_cfg (bool) - Include only models w/ model_name matching default_cfg name (excludes some aliases)

    Example:
        list_benchmarks('gluon_resnet*') -- returns a pandas dataframe with all the benchmarks starting with 'gluon_resnet'
    """
    models = list_models(filter=filter, module=module, pretrained=pretrained, exclude_filters=exclude_filters, name_matches_cfg=name_matches_cfg)
    df = pd.read_csv("https://raw.githubusercontent.com/rwightman/pytorch-image-models/master/results/results-imagenet.csv")    
    
    return df[df["model"].isin(models)]


def is_model(model_name):
    """ Check if a model name exists
    """
    return model_name in _model_entrypoints


def model_entrypoint(model_name):
    """Fetch a model entrypoint for specified model name
    """
    return _model_entrypoints[model_name]


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
    assert isinstance(module_names, (tuple, list, set))
    return any(model_name in _module_to_models[n] for n in module_names)


def is_model_pretrained(model_name):
    return model_name in _model_has_pretrained


def get_pretrained_cfg(model_name):
    if model_name in _model_pretrained_cfgs:
        return deepcopy(_model_pretrained_cfgs[model_name])
    return {}


def has_pretrained_cfg_key(model_name, cfg_key):
    """ Query model default_cfgs for existence of a specific key.
    """
    if model_name in _model_pretrained_cfgs and cfg_key in _model_pretrained_cfgs[model_name]:
        return True
    return False


def is_pretrained_cfg_key(model_name, cfg_key):
    """ Return truthy value for specified model default_cfg key, False if does not exist.
    """
    if model_name in _model_pretrained_cfgs and _model_pretrained_cfgs[model_name].get(cfg_key, False):
        return True
    return False


def get_pretrained_cfg_value(model_name, cfg_key):
    """ Get a specific model default_cfg value by key. None if it doesn't exist.
    """
    if model_name in _model_pretrained_cfgs:
        return _model_pretrained_cfgs[model_name].get(cfg_key, None)
    return None
