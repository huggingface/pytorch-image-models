""" PyTorch FX Based Feature Extraction Helpers
An extension/alternative to timm.models.features making use of PyTorch FX. Here, the idea is to:
    1. Symbolically trace a model producing a graph based intermediate representation (PyTorch FX functionality with
        some custom tweaks)
    2. Identify desired feature extraction nodes and reconfigure them as output nodes while deleting all unecessary
        nodes. (custom - inspired by https://github.com/pytorch/vision/pull/3597)
    3. Write the resulting graph into a GraphModule (PyTorch FX functionality)
Copyright 2021 Alexander Soare
"""
from typing import Callable, Dict
import math
from collections import OrderedDict
from pprint import pprint
from inspect import ismethod
import re
import warnings

import torch
from torch import nn
from torch import fx
import torch.nn.functional as F

from .features import _get_feature_info
from .fx_helpers import fx_and, fx_float_to_int

# Layers we went to treat as leaf modules for FeatureGraphNet
from .layers import Conv2dSame, ScaledStdConv2dSame, BatchNormAct2d, BlurPool2d, CondConv2d, StdConv2dSame
from .layers import GatherExcite, DropPath
from .layers.non_local_attn import BilinearAttnTransform
from .layers.pool2d_same import MaxPool2dSame, AvgPool2dSame


# These modules will not be traced through.
_leaf_modules = {
    Conv2dSame, ScaledStdConv2dSame, BatchNormAct2d, BlurPool2d, CondConv2d, StdConv2dSame, GatherExcite, DropPath,
    BilinearAttnTransform, MaxPool2dSame, AvgPool2dSame
}

try:
    from .layers import InplaceAbn
    _leaf_modules.add(InplaceAbn)
except ImportError:
    pass


def register_leaf_module(module: nn.Module):
    """
    Any module not under timm.models.layers should get this decorator if we don't want to trace through it.
    """
    _leaf_modules.add(module)
    return module


# These functions will not be traced through
_autowrap_functions=(fx_float_to_int, fx_and)


class TimmTracer(fx.Tracer):
    """
    Temporary bridge from torch.fx.Tracer to include any general workarounds required to make FX work for us
    """
    def __init__(self, autowrap_modules=(math, ), autowrap_functions=(), enable_cpatching=False):
        super().__init__(autowrap_modules=autowrap_modules, enable_cpatching=enable_cpatching)
        # FIXME: This is a workaround pending on a PyTorch PR https://github.com/pytorch/pytorch/pull/62106
        self._autowrap_function_ids.update(set([id(f) for f in autowrap_functions]))

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
        # FIXME: This is a workaround pending on a PyTorch PR https://github.com/pytorch/pytorch/pull/62095
        if target == F.pad:
            kwargs['value'] = float(kwargs['value'])
        return super().create_node(kind, target, args, kwargs, name=name, type_expr=type_expr)


class LeafNodeTracer(TimmTracer):
    """
    Account for desired leaf nodes according to _leaf_modules and _autowrap functions
    """
    def __init__(self):
        super().__init__(autowrap_functions=_autowrap_functions)

    def is_leaf_module(self, m: nn.Module, module_qualname: str) -> bool:
        if isinstance(m, tuple(_leaf_modules)):
            return True
        return super().is_leaf_module(m, module_qualname)


# Taken from https://github.com/pytorch/examples/blob/master/fx/module_tracer.py with modifications for storing
# qualified names for all Nodes, not just top-level Modules
class NodePathTracer(LeafNodeTracer):
    """
    NodePathTracer is an FX tracer that, for each operation, also records the qualified name of the Node from which the
    operation originated. A qualified name here is a `.` seperated path walking the hierarchy from top level module
    down to leaf operation or leaf module. The name of the top level module is not included as part of the qualified
    name. For example, if we trace a module who's forward method applies a ReLU module, the qualified name for that
    node will simply be 'relu'.
    """
    def __init__(self):
        super().__init__()
        # Track the qualified name of the Node being traced
        self.current_module_qualname : str = ''
        # A map from FX Node to the qualified name
        self.node_to_qualname = OrderedDict()

    def call_module(self, m: torch.nn.Module, forward: Callable, args, kwargs):
        """
        Override of Tracer.call_module (see https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.call_module).
        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Installs the qualified name of the caller in `current_module_qualname` for retrieval by `create_proxy`
        3) Once a leaf module is reached, calls `create_proxy`
        4) Restores the caller's qualified name into current_module_qualname
        """
        old_qualname = self.current_module_qualname
        try:
            module_qualname = self.path_of_module(m)
            self.current_module_qualname = module_qualname
            if not self.is_leaf_module(m, module_qualname):
                out = forward(*args, **kwargs)
                return out
            return self.create_proxy('call_module', module_qualname, args, kwargs)
        finally:
            self.current_module_qualname = old_qualname

    def create_proxy(self, kind: str, target: fx.node.Target, args, kwargs, name=None, type_expr=None):
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_qualname`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_qualname[proxy.node] = self._get_node_qualname(
            self.current_module_qualname, proxy.node)
        return proxy

    def _get_node_qualname(self, module_qualname: str, node: fx.node.Node):
        node_qualname = module_qualname
        if node.op == 'call_module':
            # Node terminates in a leaf module so the module_qualname is a complete description of the node
            # Just need to check if this module has appeared before. If so add postfix counter starting from _1 for the
            # first reappearance (this follows the way that repeated leaf ops are enumerated by PyTorch FX)
            for existing_qualname in reversed(self.node_to_qualname.values()):
                # Check to see if existing_qualname is of the form {node_qualname} or {node_qualname}_{int}
                if re.match(rf'{node_qualname}(_[0-9]+)?$', existing_qualname) is not None:
                    postfix = existing_qualname.replace(node_qualname, '')
                    if len(postfix):
                        # existing_qualname is of the form {node_qualname}_{int}
                        next_index = int(postfix[1:]) + 1
                    else:
                        # existing_qualname is of the form {node_qualname}
                        next_index = 1
                    node_qualname += f'_{next_index}'
                    break    
        else:
            # Node terminates in non- leaf module so the node name needs to be appended
            if len(node_qualname) > 0:  # only append '.' if we are deeper than the top level module 
                node_qualname += '.'
            node_qualname += str(node)
        return node_qualname


def print_graph_node_qualified_names(model: nn.Module):
    """
    Dev utility to prints nodes in order of execution. Useful for choosing `nodes` for a FeatureGraphNet design.
    This is useful for two reasons:
        1. Not all submodules are traced through. Some are treated as leaf modules. See `LeafNodeTracer`
        2. Leaf ops that occur more than once in the graph get a `_{counter}` postfix.
        
    WARNING: Changes to the operations in the original module might not change the module's overall behaviour, but they
    may result in changes to the postfixes for the names of repeated ops, thereby breaking feature extraction.
    """
    tracer = NodePathTracer()
    tracer.trace(model)
    pprint(list(tracer.node_to_qualname.values()))


def get_intermediate_nodes(model: nn.Module, return_nodes: Dict[str, str]) -> nn.Module:
    """
    Creates a new FX-based module that returns intermediate nodes from a given model. This is achieved by re-writing
    the computation graph of the model via FX to return the desired nodes as outputs. All unused nodes are removed,
    together with their corresponding parameters.
    Args:
        model (nn.Module): model on which we will extract the features
        return_nodes (Dict[name, new_name]): a dict containing the names (or partial names - see note below) of the
            nodes for which the activations will be returned as the keys. The values of the dict are the names
            of the returned activations (which the user can specify).
            A note on node specification: A node is specified as a `.` seperated path walking the hierarchy from top
            level module down to leaf operation or leaf module. For instance `blocks.5.3.bn1`. Nevertheless, the keys
            in this dict need not be fully specified. One could provide `blocks.5` as a key, and the last node with
            that prefix will be selected.
            While designing a feature extractor one can use the `print_graph_node_qualified_names` utility as a guide
            to which nodes are available.
    Acknowledgement: Starter code from https://github.com/pytorch/vision/pull/3597
    """
    return_nodes = {str(k): str(v) for k, v in return_nodes.items()}

    # Instantiate our NodePathTracer and use that to trace the model
    tracer = NodePathTracer()
    graph = tracer.trace(model)

    name = model.__class__.__name__ if isinstance(model, nn.Module) else model.__name__
    graph_module = fx.GraphModule(tracer.root, graph, name)
    
    available_nodes = [f'{v}.{k}' for k, v in tracer.node_to_qualname.items()]
    # FIXME We don't know if we should expect this to happen
    assert len(set(available_nodes)) == len(available_nodes), \
        "There are duplicate nodes! Please raise an issue https://github.com/rwightman/pytorch-image-models/issues"
    # Check that all outputs in return_nodes are present in the model
    for query in return_nodes.keys():
        if not any([m.startswith(query) for m in available_nodes]):
            raise ValueError(f"return_node: {query} is not present in model")

    # Remove existing output nodes
    orig_output_node = None
    for n in reversed(graph_module.graph.nodes):
        if n.op == "output":
            orig_output_node = n
    assert orig_output_node
    # And remove it
    graph_module.graph.erase_node(orig_output_node)
    # Find nodes corresponding to return_nodes and make them into output_nodes
    nodes = [n for n in graph_module.graph.nodes]
    output_nodes = OrderedDict()
    for n in reversed(nodes):
        if 'tensor_constant' in str(n):
            # NOTE Without this control flow we would get a None value for
            # `module_qualname = tracer.node_to_qualname.get(n)`. On the other hand, we can safely assume that we'll
            # never need to get this as an interesting intermediate node.
            continue
        module_qualname = tracer.node_to_qualname.get(n)
        for query in return_nodes:
            depth = query.count('.')
            if '.'.join(module_qualname.split('.')[:depth+1]) == query:
                output_nodes[return_nodes[query]] = n
                return_nodes.pop(query)
                break
    output_nodes = OrderedDict(reversed(list(output_nodes.items())))

    # And add them in the end of the graph
    with graph_module.graph.inserting_after(nodes[-1]):
        graph_module.graph.output(output_nodes)

    # Remove unused modules / parameters
    graph_module.graph.eliminate_dead_code()
    graph_module.recompile()
    graph_module = fx.GraphModule(graph_module, graph_module.graph, name)
    return graph_module


class FeatureGraphNet(nn.Module):
    """
    Take the provided model and transform it into a graph module. This class wraps the resulting graph module while
    also keeping the original model's non-parameter properties for reference. The original model is discarded.

    WARNING: Changes to the operations in the original module might not change the module's overall behaviour, but they
    may result in changes to the postfixes for the names of repeated ops, thereby breaking feature extraction.

    TODO: FIX THIS
    WARNING: This puts the input model into eval mode prior to tracing. This means that any control flow dependent on
    the model being in train mode will be lost.
    """
    def __init__(self, model, out_indices, out_map=None):
        super().__init__()
        model.eval()
        self.feature_info = _get_feature_info(model, out_indices)
        if out_map is not None:
            assert len(out_map) == len(out_indices)
        # NOTE the feature_info key is innapropriately named 'module' because prior to FX only modules could be
        # provided. Recall that here, we may also provide nodes referring to individual ops
        return_nodes = {info['module']: out_map[i] if out_map is not None else info['module']
                        for i, info in enumerate(self.feature_info) if i in out_indices}
        self.graph_module = get_intermediate_nodes(model, return_nodes)
        # Keep non-parameter model properties for reference
        for attr_str in model.__dir__():
            attr = getattr(model, attr_str)
            if (not attr_str.startswith('_') and attr_str not in self.__dir__() and not ismethod(attr)
                    and not isinstance(attr, (nn.Module, nn.Parameter))):
                setattr(self, attr_str, attr)

    def forward(self, x):
        return list(self.graph_module(x).values())

    def train(self, mode=True):
        """
        NOTE: This also covers `self.eval()` as that just does self.train(False)
        """
        if mode:
            warnings.warn(
                "Setting a FeatureGraphNet to training mode won't necessarily have the desired effect. Control "
                "flow depending on `self.training` will follow the `False` path. See FeatureGraphNet doc-string "
                "for more details.")
        super().train(mode)