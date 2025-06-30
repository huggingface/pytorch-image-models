""" PyTorch FX Based Feature Extraction Helpers
Using https://pytorch.org/vision/stable/feature_extraction.html
"""
from typing import Callable, Dict, List, Optional, Union, Tuple, Type

import torch
from torch import nn

from timm.layers import (
    create_feature_extractor,
    get_graph_node_names,
    register_notrace_module,
    register_notrace_function,
    is_notrace_module,
    is_notrace_function,
    get_notrace_functions,
    get_notrace_modules,
    Format,
 )
from ._features import _get_feature_info, _get_return_layers



__all__ = [
    'register_notrace_module',
    'is_notrace_module',
    'get_notrace_modules',
    'register_notrace_function',
    'is_notrace_function',
    'get_notrace_functions',
    'create_feature_extractor',
    'get_graph_node_names',
    'FeatureGraphNet',
    'GraphExtractNet',
]


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
