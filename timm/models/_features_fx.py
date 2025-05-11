""" PyTorch FX Based Feature Extraction Helpers
Using https://pytorch.org/vision/stable/feature_extraction.html
"""
from typing import Callable, Dict, List, Optional, Union, Tuple, Type

import torch
from torch import nn

from ._features import _get_feature_info, _get_return_layers

try:
    # NOTE we wrap torchvision fns to use timm leaf / no trace definitions
    from torchvision.models.feature_extraction import create_feature_extractor as _create_feature_extractor
    from torchvision.models.feature_extraction import get_graph_node_names as _get_graph_node_names
    has_fx_feature_extraction = True
except ImportError:
    has_fx_feature_extraction = False

# Layers we went to treat as leaf modules
from timm.layers import Conv2dSame, ScaledStdConv2dSame, CondConv2d, StdConv2dSame, Format
from timm.layers import resample_abs_pos_embed, resample_abs_pos_embed_nhwc
from timm.layers.non_local_attn import BilinearAttnTransform
from timm.layers.pool2d_same import MaxPool2dSame, AvgPool2dSame
from timm.layers.norm_act import (
    BatchNormAct2d,
    SyncBatchNormAct,
    FrozenBatchNormAct2d,
    GroupNormAct,
    GroupNorm1Act,
    LayerNormAct,
    LayerNormAct2d
)

__all__ = ['register_notrace_module', 'is_notrace_module', 'get_notrace_modules',
           'register_notrace_function', 'is_notrace_function', 'get_notrace_functions',
           'create_feature_extractor', 'get_graph_node_names', 'FeatureGraphNet', 'GraphExtractNet']


# NOTE: By default, any modules from timm.models.layers that we want to treat as leaf modules go here
# BUT modules from timm.models should use the registration mechanism below
_leaf_modules = {
    BilinearAttnTransform,  # reason: flow control t <= 1
    # Reason: get_same_padding has a max which raises a control flow error
    Conv2dSame, MaxPool2dSame, ScaledStdConv2dSame, StdConv2dSame, AvgPool2dSame,
    CondConv2d,  # reason: TypeError: F.conv2d received Proxy in groups=self.groups * B (because B = x.shape[0]),
    BatchNormAct2d,
    SyncBatchNormAct,
    FrozenBatchNormAct2d,
    GroupNormAct,
    GroupNorm1Act,
    LayerNormAct,
    LayerNormAct2d,
}

try:
    from timm.layers import InplaceAbn
    _leaf_modules.add(InplaceAbn)
except ImportError:
    pass


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
_autowrap_functions = {
    resample_abs_pos_embed,
    resample_abs_pos_embed_nhwc,
}


def register_notrace_function(func: Callable):
    """
    Decorator for functions which ought not to be traced through
    """
    _autowrap_functions.add(func)
    return func


def is_notrace_function(func: Callable):
    return func in _autowrap_functions


def get_notrace_functions():
    return list(_autowrap_functions)


def get_graph_node_names(model: nn.Module) -> Tuple[List[str], List[str]]:
    return _get_graph_node_names(
        model,
        tracer_kwargs={'leaf_modules': list(_leaf_modules), 'autowrap_functions': list(_autowrap_functions)}
    )


def create_feature_extractor(model: nn.Module, return_nodes: Union[Dict[str, str], List[str]]):
    assert has_fx_feature_extraction, 'Please update to PyTorch 1.10+, torchvision 0.11+ for FX feature extraction'
    return _create_feature_extractor(
        model, return_nodes,
        tracer_kwargs={'leaf_modules': list(_leaf_modules), 'autowrap_functions': list(_autowrap_functions)}
    )


class FeatureGraphNet(nn.Module):
    """ A FX Graph based feature extractor that works with the model feature_info metadata
    """
    return_dict: torch.jit.Final[bool]

    def __init__(
            self,
            model: nn.Module,
            out_indices: Tuple[int, ...],
            out_map: Optional[Dict] = None,
            output_fmt: str = 'NCHW',
            return_dict: bool = False,
    ):
        super().__init__()
        assert has_fx_feature_extraction, 'Please update to PyTorch 1.10+, torchvision 0.11+ for FX feature extraction'
        self.feature_info = _get_feature_info(model, out_indices)
        if out_map is not None:
            assert len(out_map) == len(out_indices)
        self.output_fmt = Format(output_fmt)
        return_nodes = _get_return_layers(self.feature_info, out_map)
        self.graph_module = create_feature_extractor(model, return_nodes)
        self.return_dict = return_dict

    def forward(self, x):
        out = self.graph_module(x)
        if self.return_dict:
            return out
        return list(out.values())


class GraphExtractNet(nn.Module):
    """ A standalone feature extraction wrapper that maps dict -> list or single tensor
    NOTE:
      * one can use feature_extractor directly if dictionary output is desired
      * unlike FeatureGraphNet, this is intended to be used standalone and not with model feature_info
      metadata for builtin feature extraction mode
      * create_feature_extractor can be used directly if dictionary output is acceptable

    Args:
        model: model to extract features from
        return_nodes: node names to return features from (dict or list)
        squeeze_out: if only one output, and output in list format, flatten to single tensor
        return_dict: return as dictionary from extractor with node names as keys, ignores squeeze_out arg
    """
    return_dict: torch.jit.Final[bool]

    def __init__(
            self,
            model: nn.Module,
            return_nodes: Union[Dict[str, str], List[str]],
            squeeze_out: bool = True,
            return_dict: bool = False,
    ):
        super().__init__()
        self.squeeze_out = squeeze_out
        self.graph_module = create_feature_extractor(model, return_nodes)
        self.return_dict = return_dict

    def forward(self, x) -> Union[List[torch.Tensor], torch.Tensor]:
        out = self.graph_module(x)
        if self.return_dict:
            return out
        out = list(out.values())
        return out[0] if self.squeeze_out and len(out) == 1 else out
