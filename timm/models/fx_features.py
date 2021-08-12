""" PyTorch FX Based Feature Extraction Helpers
An extension/alternative to timm.models.features making use of PyTorch FX. Here, the idea is to:
    1. Symbolically trace a model producing a graph based intermediate representation (PyTorch FX functionality with
        some custom tweaks)
    2. Identify desired feature extraction nodes and reconfigure them as output nodes while deleting all unecessary
        nodes. (custom - inspired by https://github.com/pytorch/vision/pull/3597)
    3. Write the resulting graph into a GraphModule (PyTorch FX functionality)
Copyright 2021 Alexander Soare
"""
from typing import Callable, Dict, Union, List, Optional
import math
from collections import OrderedDict
from pprint import pprint
from inspect import ismethod
import re
import warnings
from copy import deepcopy
from itertools import chain

import torch
from torch import nn
from torch import fx
import torch.nn.functional as F
from torch.fx.graph_module import _copy_attr

from .features import _get_feature_info
from .fx_helpers import fx_float_to_int

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
_autowrap_functions=(fx_float_to_int,)


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


def _is_subseq(x, y):
    """Check if y is a subseqence of x
    https://stackoverflow.com/a/24017747/4391249
    """
    iter_x = iter(x)
    return all(any(x_item == y_item for x_item in iter_x) for y_item in y)


# Taken from https://github.com/pytorch/examples/blob/master/fx/module_tracer.py with modifications for storing
# qualified names for all Nodes, not just top-level Modules
class NodePathTracer(LeafNodeTracer):
    """
    NodePathTracer is an FX tracer that, for each operation, also records the
    qualified name of the Node from which the operation originated. A
    qualified name here is a `.` seperated path walking the hierarchy from top
    level module down to leaf operation or leaf module. The name of the top
    level module is not included as part of the qualified name. For example,
    if we trace a module who's forward method applies a ReLU module, the
    qualified name for that node will simply be 'relu'.

    Some notes on the specifics:
        - Nodes are recorded to `self.node_to_qualname` which is a dictionary
          mapping a given Node object to its qualified name.
        - Nodes are recorded in the order which they are executed during
          tracing.
        - When a duplicate qualified name is encountered, a suffix of the form
          _{int} is added. The counter starts from 1.
    """
    def __init__(self, *args, **kwargs):
        super(NodePathTracer, self).__init__(*args, **kwargs)
        # Track the qualified name of the Node being traced
        self.current_module_qualname = ''
        # A map from FX Node to the qualified name
        self.node_to_qualname = OrderedDict()

    def call_module(self, m: torch.nn.Module, forward: Callable, args, kwargs):
        """
        Override of `fx.Tracer.call_module`
        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Adds the qualified name of the caller to
           `current_module_qualname` for retrieval by `create_proxy`
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

    def create_proxy(self, kind: str, target: fx.node.Target, args, kwargs,
                     name=None, type_expr=None) -> fx.proxy.Proxy:
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_qualname`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_qualname[proxy.node] = self._get_node_qualname(
            self.current_module_qualname, proxy.node)
        return proxy

    def _get_node_qualname(
            self, module_qualname: str, node: fx.node.Node) -> str:
        node_qualname = module_qualname
        if node.op == 'call_module':
            # Node terminates in a leaf module so the module_qualname is a
            # complete description of the node
            for existing_qualname in reversed(self.node_to_qualname.values()):
                # Check to see if existing_qualname is of the form
                # {node_qualname} or {node_qualname}_{int}
                if re.match(rf'{node_qualname}(_[0-9]+)?$',
                            existing_qualname) is not None:
                    postfix = existing_qualname.replace(node_qualname, '')
                    if len(postfix):
                        # Existing_qualname is of the form {node_qualname}_{int}
                        next_index = int(postfix[1:]) + 1
                    else:
                        # existing_qualname is of the form {node_qualname}
                        next_index = 1
                    node_qualname += f'_{next_index}'
                    break
        else:
            # Node terminates in non- leaf module so the node name needs to be
            # appended
            if len(node_qualname) > 0:
                # Only append '.' if we are deeper than the top level module
                node_qualname += '.'
            node_qualname += str(node)
        return node_qualname


def _warn_graph_differences(
        train_tracer: NodePathTracer, eval_tracer: NodePathTracer):
    """
    Utility function for warning the user if there are differences between
    the train graph and the eval graph.
    """
    train_nodes = list(train_tracer.node_to_qualname.values())
    eval_nodes = list(eval_tracer.node_to_qualname.values())

    if len(train_nodes) == len(eval_nodes) and [
            t == e for t, e in zip(train_nodes, eval_nodes)]:
        return

    suggestion_msg = (
        "When choosing nodes for feature extraction, you may need to specify "
        "output nodes for train and eval mode separately")

    if _is_subseq(train_nodes, eval_nodes):
        msg = ("NOTE: The nodes obtained by tracing the model in eval mode "
               "are a subsequence of those obtained in train mode. ")
    elif _is_subseq(eval_nodes, train_nodes):
        msg = ("NOTE: The nodes obtained by tracing the model in train mode "
               "are a subsequence of those obtained in eval mode. ")
    else:
        msg = ("The nodes obtained by tracing the model in train mode "
               "are different to those obtained in eval mode. ")
    warnings.warn(msg + suggestion_msg)


def print_graph_node_qualified_names(
        model: nn.Module, tracer_kwargs: Dict = {}):
    """
    Dev utility to prints nodes in order of execution. Useful for choosing
    nodes for a FeatureGraphNet design. There are two reasons that qualified
    node names can't easily be read directly from the code for a model:
        1. Not all submodules are traced through. Modules from `torch.nn` all
           fall within this category.
        2. Node qualified names that occur more than once in the graph get a
           `_{counter}` postfix.
    The model will be traced twice: once in train mode, and once in eval mode.
    If there are discrepancies between the graphs produced, both sets will
    be printed and the user will be warned.

    Args:
        model (nn.Module): model on which we will extract the features
        tracer_kwargs (Dict): a dictionary of keywork arguments for
            `NodePathTracer` (which passes them onto it's parent class
            `torch.fx.Tracer`).
    """
    train_tracer = NodePathTracer(**tracer_kwargs)
    train_tracer.trace(model.train())
    eval_tracer = NodePathTracer(**tracer_kwargs)
    eval_tracer.trace(model.eval())
    train_nodes = list(train_tracer.node_to_qualname.values())
    eval_nodes = list(eval_tracer.node_to_qualname.values())
    if len(train_nodes) == len(eval_nodes) and [
            t == e for t, e in zip(train_nodes, eval_nodes)]:
        # Nodes are aligned in train vs eval mode
        pprint(list(train_tracer.node_to_qualname.values()))
        return
    print("Nodes from train mode:")
    pprint(list(train_tracer.node_to_qualname.values()))
    print()
    print("Nodes from eval mode:")
    pprint(list(eval_tracer.node_to_qualname.values()))
    print()
    _warn_graph_differences(train_tracer, eval_tracer)
            

class DualGraphModule(fx.GraphModule):
    """
    A derivative of `fx.GraphModule`. Differs in the following ways:
    - Requires a train and eval version of the underlying graph
    - Copies submodules according to the nodes of both train and eval graphs.
    - Calling train(mode) switches between train graph and eval graph. 
    """
    def __init__(self,
                 root: torch.nn.Module,
                 train_graph: fx.Graph,
                 eval_graph: fx.Graph,
                 class_name: str = 'GraphModule'):
        """
        Args:
            root (torch.nn.Module): module from which the copied module
                hierarchy is built
            train_graph (Graph): the graph that should be used in train mode
            eval_graph (Graph): the graph that should be used in eval mode
        """
        super(fx.GraphModule, self).__init__()
        
        self.__class__.__name__ = class_name

        self.train_graph = train_graph
        self.eval_graph = eval_graph

        # Copy all get_attr and call_module ops (indicated by BOTH train and
        # eval graphs)
        for node in chain(iter(train_graph.nodes), iter(eval_graph.nodes)):
            if node.op in ['get_attr', 'call_module']:
                assert isinstance(node.target, str)
                _copy_attr(root, self, node.target)

        # eval mode by default
        self.eval()
        self.graph = eval_graph

        # (borrowed from fx.GraphModule):
        # Store the Tracer class responsible for creating a Graph separately as part of the
        # GraphModule state, except when the Tracer is defined in a local namespace.
        # Locally defined Tracers are not pickleable. This is needed because torch.package will
        # serialize a GraphModule without retaining the Graph, and needs to use the correct Tracer
        # to re-create the Graph during deserialization.
        # TODO uncomment this when https://github.com/pytorch/pytorch/pull/63121 is available
        # assert self.eval_graph._tracer_cls == self.train_graph._tracer_cls, \
        #     "Train mode and eval mode should use the same tracer class"
        # self._tracer_cls = None
        # if self.graph._tracer_cls and '<locals>' not in self.graph._tracer_cls.__qualname__:
        #     self._tracer_cls = self.graph._tracer_cls
        
    def train(self, mode=True):
        """
        Swap out the graph depending on the training mode.
        NOTE this should be safe when calling model.eval() because that just
        calls this with mode == False.
        """
        if mode:
            self.graph = self.train_graph
        else:
            self.graph = self.eval_graph
        return super().train(mode=mode)


def build_feature_graph_net(
        model: nn.Module,
        return_nodes: Union[List[str], Dict[str, str]],
        train_return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
        eval_return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
        tracer_kwargs: Dict = {}) -> fx.GraphModule:
    """
    Creates a new graph module that returns intermediate nodes from a given
    model as dictionary with user specified keys as strings, and the requested
    outputs as values. This is achieved by re-writing the computation graph of
    the model via FX to return the desired nodes as outputs. All unused nodes
    are removed, together with their corresponding parameters.

    A note on node specification: A node qualified name is specified as a `.`
    seperated path walking the hierarchy from top level module down to leaf
    operation or leaf module. For instance `blocks.5.3.bn1`. The keys of the
    `return_nodes` argument should point to either a node's qualified name,
    or some truncated version of it. For example, one could provide `blocks.5`
    as a key, and the last node with that prefix will be selected.
    `print_graph_node_qualified_names` is a useful helper function for getting
    a list of qualified names of a model.

    An attempt is made to keep all non-parametric properties of the original
    model, but existing properties of the constructed `GraphModule` are not
    overwritten.

    Args:
        model (nn.Module): model on which we will extract the features
        return_nodes (Union[List[name], Dict[name, new_name]])): either a list
            or a dict containing the names (or partial names - see note above)
            of the nodes for which the activations will be returned. If it is
            a `Dict`, the keys are the qualified node names, and the values
            are the user-specified keys for the graph module's returned
            dictionary. If it is a `List`, it is treated as a `Dict` mapping
            node specification strings directly to output names.
        tracer_kwargs (Dict): a dictionary of keywork arguments for
            `NodePathTracer` (which passes them onto it's parent class
            `torch.fx.Tracer`).

    Examples::

        >>> model = torchvision.models.resnet18()
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> graph_module = torchvision.models._utils.build_feature_graph_net(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = graph_module(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]

    """
    is_training = model.training

    if isinstance(return_nodes, list):
        return_nodes = {n: n for n in return_nodes}
    return_nodes = {str(k): str(v) for k, v in return_nodes.items()}

    assert not ((train_return_nodes is None) ^ (eval_return_nodes is None)), \
        ("If any of `train_return_nodes` and `eval_return_nodes` are "
         "specified, then both should be specified")

    if train_return_nodes is None:
        train_return_nodes = deepcopy(return_nodes)
        eval_return_nodes = deepcopy(return_nodes)

    # Repeat the tracing and graph rewriting for train and eval mode
    tracers = {}
    graphs = {}
    return_nodes = {
        'train': train_return_nodes,
        'eval': eval_return_nodes
    }
    for mode in ['train', 'eval']:
        if mode == 'train':
            model.train()
        elif mode == 'eval':
            model.eval()

        # Instantiate our NodePathTracer and use that to trace the model
        tracer = NodePathTracer(**tracer_kwargs)
        graph = tracer.trace(model)

        name = model.__class__.__name__ if isinstance(
            model, nn.Module) else model.__name__
        graph_module = fx.GraphModule(tracer.root, graph, name)

        available_nodes = [f'{v}.{k}' for k, v in tracer.node_to_qualname.items()]
        # FIXME We don't know if we should expect this to happen
        assert len(set(available_nodes)) == len(available_nodes), \
            "There are duplicate nodes! Please raise an issue https://github.com/pytorch/vision/issues"
        # Check that all outputs in return_nodes are present in the model
        for query in return_nodes[mode].keys():
            if not any([m.startswith(query) for m in available_nodes]):
                raise ValueError(f"return_node: {query} is not present in model")

        # Remove existing output nodes (train mode)
        orig_output_nodes = []
        for n in reversed(graph_module.graph.nodes):
            if n.op == "output":
                orig_output_nodes.append(n)
        assert len(orig_output_nodes)
        for n in orig_output_nodes:
            graph_module.graph.erase_node(n)

        # Find nodes corresponding to return_nodes and make them into output_nodes
        nodes = [n for n in graph_module.graph.nodes]
        output_nodes = OrderedDict()
        for n in reversed(nodes):
            if 'tensor_constant' in str(n):
                # NOTE Without this control flow we would get a None value for
                # `module_qualname = tracer.node_to_qualname.get(n)`.
                # On the other hand, we can safely assume that we'll never need to
                # get this as an interesting intermediate node.
                continue
            module_qualname = tracer.node_to_qualname.get(n)
            for query in return_nodes[mode]:
                depth = query.count('.')
                if '.'.join(module_qualname.split('.')[:depth + 1]) == query:
                    output_nodes[return_nodes[mode][query]] = n
                    return_nodes[mode].pop(query)
                    break
        output_nodes = OrderedDict(reversed(list(output_nodes.items())))

        # And add them in the end of the graph
        with graph_module.graph.inserting_after(nodes[-1]):
            graph_module.graph.output(output_nodes)

        # Remove unused modules / parameters
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        # Keep track of the tracer and graph so we can choose the main one
        tracers[mode] = tracer
        graphs[mode] = graph

    # Warn user if there are any discrepancies between the graphs of the 
    # train and eval modes
    _warn_graph_differences(tracers['train'], tracers['eval'])

    # Build the final graph module
    graph_module = DualGraphModule(
        model, graphs['train'], graphs['eval'], class_name=name)

    # Keep non-parameter model properties for reference
    for attr_str in model.__dir__():
        attr = getattr(model, attr_str)
        if (not attr_str.startswith('_')
                and attr_str not in graph_module.__dir__()
                and not ismethod(attr)
                and not isinstance(attr, (nn.Module, nn.Parameter))):
            setattr(graph_module, attr_str, attr)

    # Restore original training mode
    graph_module.train(is_training)

    return graph_module


class FeatureGraphNet(nn.Module):
    def __init__(self, model, out_indices, out_map=None):
        super().__init__()
        self.feature_info = _get_feature_info(model, out_indices)
        if out_map is not None:
            assert len(out_map) == len(out_indices)
        return_nodes = {info['module']: out_map[i] if out_map is not None else info['module']
                        for i, info in enumerate(self.feature_info) if i in out_indices}
        self.graph_module = build_feature_graph_net(model, return_nodes)
        
    def forward(self, x):
        return list(self.graph_module(x).values())
