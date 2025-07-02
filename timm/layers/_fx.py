from typing import Callable, Dict, List, Optional, Union, Tuple, Type

import torch
from torch import nn

try:
    # NOTE we wrap torchvision fns to use timm leaf / no trace definitions
    from torchvision.models.feature_extraction import create_feature_extractor as _create_feature_extractor
    from torchvision.models.feature_extraction import get_graph_node_names as _get_graph_node_names
    has_fx_feature_extraction = True
except ImportError:
    has_fx_feature_extraction = False


__all__ = [
    'register_notrace_module',
    'is_notrace_module',
    'get_notrace_modules',
    'register_notrace_function',
    'is_notrace_function',
    'get_notrace_functions',
    'create_feature_extractor',
    'get_graph_node_names',
]

# modules to treat as leafs when tracing
_leaf_modules = set()


def register_notrace_module(module: Type[nn.Module]):
    """
    Any module not under timm.models.layers should get this decorator if we don't want to trace through it.
    """
    _leaf_modules.add(module)
    return module


def is_notrace_module(module: Type[nn.Module]):
    return module in _leaf_modules


def get_notrace_modules():
    return list(_leaf_modules)


# Functions we want to autowrap (treat them as leaves)
_autowrap_functions = set()


def register_notrace_function(name_or_fn):
    _autowrap_functions.add(name_or_fn)
    return name_or_fn


def is_notrace_function(func: Callable):
    return func in _autowrap_functions


def get_notrace_functions():
    return list(_autowrap_functions)


def get_graph_node_names(model: nn.Module) -> Tuple[List[str], List[str]]:
    return _get_graph_node_names(
        model,
        tracer_kwargs={
            'leaf_modules': list(_leaf_modules),
            'autowrap_functions': list(_autowrap_functions)
        }
    )


def create_feature_extractor(model: nn.Module, return_nodes: Union[Dict[str, str], List[str]]):
    assert has_fx_feature_extraction, 'Please update to PyTorch 1.10+, torchvision 0.11+ for FX feature extraction'
    return _create_feature_extractor(
        model, return_nodes,
        tracer_kwargs={
            'leaf_modules': list(_leaf_modules),
            'autowrap_functions': list(_autowrap_functions)
        }
    )