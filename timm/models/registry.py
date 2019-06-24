import sys
import re
import fnmatch
from collections import defaultdict

__all__ = ['list_models', 'is_model', 'model_entrypoint', 'list_modules', 'is_model_in_modules']

_module_to_models = defaultdict(set)
_model_to_module = {}
_model_entrypoints = {}


def register_model(fn):
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    _model_entrypoints[fn.__name__] = fn
    _model_to_module[fn.__name__] = module_name
    _module_to_models[module_name].add(fn.__name__)
    return fn


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def list_models(filter='', module=''):
    """ Return list of available model names, sorted alphabetically

    Args:
        filter (str) - Wildcard filter string that works with fnmatch
        module (str) - Limit model selection to a specific sub-module (ie 'gen_efficientnet')

    Example:
        model_list('gluon_resnet*') -- returns all models starting with 'gluon_resnet'
        model_list('*resnext*, 'resnet') -- returns all models with 'resnext' in 'resnet' module
    """
    if module:
        models = list(_module_to_models[module])
    else:
        models = _model_entrypoints.keys()
    if filter:
        models = fnmatch.filter(models, filter)
    return list(sorted(models, key=_natural_key))


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

