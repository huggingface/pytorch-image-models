""" PyTorch Feature Extraction Helpers

A collection of classes, functions, modules to help extract features from models
and provide a common interface for describing them.

The return_layers, module re-writing idea inspired by torchvision IntermediateLayerGetter
https://github.com/pytorch/vision/blob/d88d8961ae51507d0cb680329d985b1488b1b76b/torchvision/models/_utils.py

Hacked together by / Copyright 2020 Ross Wightman
"""
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from timm.layers import Format, _assert
from ._manipulate import checkpoint

__all__ = [
    'FeatureInfo', 'FeatureHooks', 'FeatureDictNet', 'FeatureListNet', 'FeatureHookNet', 'FeatureGetterNet',
    'feature_take_indices'
]


def feature_take_indices(
        num_features: int,
        indices: Optional[Union[int, List[int]]] = None,
        as_set: bool = False,
) -> Tuple[List[int], int]:
    """ Determine the absolute feature indices to 'take' from.

    Note: This function can be called in forward() so must be torchscript compatible,
    which requires some incomplete typing and workaround hacks.

    Args:
        num_features: total number of features to select from
        indices: indices to select,
          None -> select all
          int -> select last n
          list/tuple of int -> return specified (-ve indices specify from end)
        as_set: return as a set

    Returns:
        List (or set) of absolute (from beginning) indices, Maximum index
    """
    if indices is None:
        indices = num_features  # all features if None

    if isinstance(indices, int):
        # convert int -> last n indices
        _assert(0 < indices <= num_features, f'last-n ({indices}) is out of range (1 to {num_features})')
        take_indices = [num_features - indices + i for i in range(indices)]
    else:
        take_indices: List[int] = []
        for i in indices:
            idx = num_features + i if i < 0 else i
            _assert(0 <= idx < num_features, f'feature index {idx} is out of range (0 to {num_features - 1})')
            take_indices.append(idx)

    if not torch.jit.is_scripting() and as_set:
        return set(take_indices), max(take_indices)

    return take_indices, max(take_indices)


def _out_indices_as_tuple(x: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    if isinstance(x, int):
        # if indices is an int, take last N features
        return tuple(range(-x, 0))
    return tuple(x)


OutIndicesT = Union[int, Tuple[int, ...]]


class FeatureInfo:

    def __init__(
            self,
            feature_info: List[Dict],
            out_indices: OutIndicesT,
    ):
        out_indices = _out_indices_as_tuple(out_indices)
        prev_reduction = 1
        for i, fi in enumerate(feature_info):
            # sanity check the mandatory fields, there may be additional fields depending on the model
            assert 'num_chs' in fi and fi['num_chs'] > 0
            assert 'reduction' in fi and fi['reduction'] >= prev_reduction
            prev_reduction = fi['reduction']
            assert 'module' in fi
            fi.setdefault('index', i)
        self.out_indices = out_indices
        self.info = feature_info

    def from_other(self, out_indices: OutIndicesT):
        out_indices = _out_indices_as_tuple(out_indices)
        return FeatureInfo(deepcopy(self.info), out_indices)

    def get(self, key: str, idx: Optional[Union[int, List[int]]] = None):
        """ Get value by key at specified index (indices)
        if idx == None, returns value for key at each output index
        if idx is an integer, return value for that feature module index (ignoring output indices)
        if idx is a list/tuple, return value for each module index (ignoring output indices)
        """
        if idx is None:
            return [self.info[i][key] for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [self.info[i][key] for i in idx]
        else:
            return self.info[idx][key]

    def get_dicts(self, keys: Optional[List[str]] = None, idx: Optional[Union[int, List[int]]] = None):
        """ return info dicts for specified keys (or all if None) at specified indices (or out_indices if None)
        """
        if idx is None:
            if keys is None:
                return [self.info[i] for i in self.out_indices]
            else:
                return [{k: self.info[i][k] for k in keys} for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [self.info[i] if keys is None else {k: self.info[i][k] for k in keys} for i in idx]
        else:
            return self.info[idx] if keys is None else {k: self.info[idx][k] for k in keys}

    def channels(self, idx: Optional[Union[int, List[int]]] = None):
        """ feature channels accessor
        """
        return self.get('num_chs', idx)

    def reduction(self, idx: Optional[Union[int, List[int]]] = None):
        """ feature reduction (output stride) accessor
        """
        return self.get('reduction', idx)

    def module_name(self, idx: Optional[Union[int, List[int]]] = None):
        """ feature module name accessor
        """
        return self.get('module', idx)

    def __getitem__(self, item):
        return self.info[item]

    def __len__(self):
        return len(self.info)


class FeatureHooks:
    """ Feature Hook Helper

    This module helps with the setup and extraction of hooks for extracting features from
    internal nodes in a model by node name.

    FIXME This works well in eager Python but needs redesign for torchscript.
    """

    def __init__(
            self,
            hooks: Sequence[Union[str, Dict]],
            named_modules: dict,
            out_map: Sequence[Union[int, str]] = None,
            default_hook_type: str = 'forward',
    ):
        # setup feature hooks
        self._feature_outputs = defaultdict(OrderedDict)
        self._handles = []
        modules = {k: v for k, v in named_modules}
        for i, h in enumerate(hooks):
            hook_name = h if isinstance(h, str) else h['module']
            m = modules[hook_name]
            hook_id = out_map[i] if out_map else hook_name
            hook_fn = partial(self._collect_output_hook, hook_id)
            hook_type = default_hook_type
            if isinstance(h, dict):
                hook_type = h.get('hook_type', default_hook_type)
            if hook_type == 'forward_pre':
                handle = m.register_forward_pre_hook(hook_fn)
            elif hook_type == 'forward':
                handle = m.register_forward_hook(hook_fn)
            else:
                assert False, "Unsupported hook type"
            self._handles.append(handle)

    def _collect_output_hook(self, hook_id, *args):
        x = args[-1]  # tensor we want is last argument, output for fwd, input for fwd_pre
        if isinstance(x, tuple):
            x = x[0]  # unwrap input tuple
        self._feature_outputs[x.device][hook_id] = x

    def get_output(self, device) -> Dict[str, torch.tensor]:
        output = self._feature_outputs[device]
        self._feature_outputs[device] = OrderedDict()  # clear after reading
        return output


def _module_list(module, flatten_sequential=False):
    # a yield/iter would be better for this but wouldn't be compatible with torchscript
    ml = []
    for name, module in module.named_children():
        if flatten_sequential and isinstance(module, nn.Sequential):
            # first level of Sequential containers is flattened into containing model
            for child_name, child_module in module.named_children():
                combined = [name, child_name]
                ml.append(('_'.join(combined), '.'.join(combined), child_module))
        else:
            ml.append((name, name, module))
    return ml


def _get_feature_info(net, out_indices: OutIndicesT):
    feature_info = getattr(net, 'feature_info')
    if isinstance(feature_info, FeatureInfo):
        return feature_info.from_other(out_indices)
    elif isinstance(feature_info, (list, tuple)):
        return FeatureInfo(net.feature_info, out_indices)
    else:
        assert False, "Provided feature_info is not valid"


def _get_return_layers(feature_info, out_map):
    module_names = feature_info.module_name()
    return_layers = {}
    for i, name in enumerate(module_names):
        return_layers[name] = out_map[i] if out_map is not None else feature_info.out_indices[i]
    return return_layers


class FeatureDictNet(nn.ModuleDict):
    """ Feature extractor with OrderedDict return

    Wrap a model and extract features as specified by the out indices, the network is
    partially re-built from contained modules.

    There is a strong assumption that the modules have been registered into the model in the same
    order as they are used. There should be no reuse of the same nn.Module more than once, including
    trivial modules like `self.relu = nn.ReLU`.

    Only submodules that are directly assigned to the model class (`model.feature1`) or at most
    one Sequential container deep (`model.features.1`, with flatten_sequent=True) can be captured.
    All Sequential containers that are directly assigned to the original model will have their
    modules assigned to this module with the name `model.features.1` being changed to `model.features_1`
    """
    def __init__(
            self,
            model: nn.Module,
            out_indices: OutIndicesT = (0, 1, 2, 3, 4),
            out_map: Sequence[Union[int, str]] = None,
            output_fmt: str = 'NCHW',
            feature_concat: bool = False,
            flatten_sequential: bool = False,
    ):
        """
        Args:
            model: Model from which to extract features.
            out_indices: Output indices of the model features to extract.
            out_map: Return id mapping for each output index, otherwise str(index) is used.
            feature_concat: Concatenate intermediate features that are lists or tuples instead of selecting
                first element e.g. `x[0]`
            flatten_sequential: Flatten first two-levels of sequential modules in model (re-writes model modules)
        """
        super(FeatureDictNet, self).__init__()
        self.feature_info = _get_feature_info(model, out_indices)
        self.output_fmt = Format(output_fmt)
        self.concat = feature_concat
        self.grad_checkpointing = False
        self.return_layers = {}

        return_layers = _get_return_layers(self.feature_info, out_map)
        modules = _module_list(model, flatten_sequential=flatten_sequential)
        remaining = set(return_layers.keys())
        layers = OrderedDict()
        for new_name, old_name, module in modules:
            layers[new_name] = module
            if old_name in remaining:
                # return id has to be consistently str type for torchscript
                self.return_layers[new_name] = str(return_layers[old_name])
                remaining.remove(old_name)
            if not remaining:
                break
        assert not remaining and len(self.return_layers) == len(return_layers), \
            f'Return layers ({remaining}) are not present in model'
        self.update(layers)

    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    def _collect(self, x) -> (Dict[str, torch.Tensor]):
        out = OrderedDict()
        for i, (name, module) in enumerate(self.items()):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # Skipping checkpoint of first module because need a gradient at input
                # Skipping last because networks with in-place ops might fail w/ checkpointing enabled
                # NOTE: first_or_last module could be static, but recalc in is_scripting guard to avoid jit issues
                first_or_last_module = i == 0 or i == max(len(self) - 1, 0)
                x = module(x) if first_or_last_module else checkpoint(module, x)
            else:
                x = module(x)

            if name in self.return_layers:
                out_id = self.return_layers[name]
                if isinstance(x, (tuple, list)):
                    # If model tap is a tuple or list, concat or select first element
                    # FIXME this may need to be more generic / flexible for some nets
                    out[out_id] = torch.cat(x, 1) if self.concat else x[0]
                else:
                    out[out_id] = x
        return out

    def forward(self, x) -> Dict[str, torch.Tensor]:
        return self._collect(x)


class FeatureListNet(FeatureDictNet):
    """ Feature extractor with list return

    A specialization of FeatureDictNet that always returns features as a list (values() of dict).
    """
    def __init__(
            self,
            model: nn.Module,
            out_indices: OutIndicesT = (0, 1, 2, 3, 4),
            output_fmt: str = 'NCHW',
            feature_concat: bool = False,
            flatten_sequential: bool = False,
    ):
        """
        Args:
            model: Model from which to extract features.
            out_indices: Output indices of the model features to extract.
            feature_concat: Concatenate intermediate features that are lists or tuples instead of selecting
                first element e.g. `x[0]`
            flatten_sequential: Flatten first two-levels of sequential modules in model (re-writes model modules)
        """
        super().__init__(
            model,
            out_indices=out_indices,
            output_fmt=output_fmt,
            feature_concat=feature_concat,
            flatten_sequential=flatten_sequential,
        )

    def forward(self, x) -> (List[torch.Tensor]):
        return list(self._collect(x).values())


class FeatureHookNet(nn.ModuleDict):
    """ FeatureHookNet

    Wrap a model and extract features specified by the out indices using forward/forward-pre hooks.

    If `no_rewrite` is True, features are extracted via hooks without modifying the underlying
    network in any way.

    If `no_rewrite` is False, the model will be re-written as in the
    FeatureList/FeatureDict case by folding first to second (Sequential only) level modules into this one.

    FIXME this does not currently work with Torchscript, see FeatureHooks class
    """
    def __init__(
            self,
            model: nn.Module,
            out_indices: OutIndicesT = (0, 1, 2, 3, 4),
            out_map: Optional[Sequence[Union[int, str]]] = None,
            return_dict: bool = False,
            output_fmt: str = 'NCHW',
            no_rewrite: Optional[bool] = None,
            flatten_sequential: bool = False,
            default_hook_type: str = 'forward',
    ):
        """

        Args:
            model: Model from which to extract features.
            out_indices: Output indices of the model features to extract.
            out_map: Return id mapping for each output index, otherwise str(index) is used.
            return_dict: Output features as a dict.
            no_rewrite: Enforce that model is not re-written if True, ie no modules are removed / changed.
                flatten_sequential arg must also be False if this is set True.
            flatten_sequential: Re-write modules by flattening first two levels of nn.Sequential containers.
            default_hook_type: The default hook type to use if not specified in model.feature_info.
        """
        super().__init__()
        assert not torch.jit.is_scripting()
        self.feature_info = _get_feature_info(model, out_indices)
        self.return_dict = return_dict
        self.output_fmt = Format(output_fmt)
        self.grad_checkpointing = False
        if no_rewrite is None:
            no_rewrite = not flatten_sequential
        layers = OrderedDict()
        hooks = []
        if no_rewrite:
            assert not flatten_sequential
            if hasattr(model, 'reset_classifier'):  # make sure classifier is removed?
                model.reset_classifier(0)
            layers['body'] = model
            hooks.extend(self.feature_info.get_dicts())
        else:
            modules = _module_list(model, flatten_sequential=flatten_sequential)
            remaining = {
                f['module']: f['hook_type'] if 'hook_type' in f else default_hook_type
                for f in self.feature_info.get_dicts()
            }
            for new_name, old_name, module in modules:
                layers[new_name] = module
                for fn, fm in module.named_modules(prefix=old_name):
                    if fn in remaining:
                        hooks.append(dict(module=fn, hook_type=remaining[fn]))
                        del remaining[fn]
                if not remaining:
                    break
            assert not remaining, f'Return layers ({remaining}) are not present in model'
        self.update(layers)
        self.hooks = FeatureHooks(hooks, model.named_modules(), out_map=out_map)

    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    def forward(self, x):
        for i, (name, module) in enumerate(self.items()):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # Skipping checkpoint of first module because need a gradient at input
                # Skipping last because networks with in-place ops might fail w/ checkpointing enabled
                # NOTE: first_or_last module could be static, but recalc in is_scripting guard to avoid jit issues
                first_or_last_module = i == 0 or i == max(len(self) - 1, 0)
                x = module(x) if first_or_last_module else checkpoint(module, x)
            else:
                x = module(x)
        out = self.hooks.get_output(x.device)
        return out if self.return_dict else list(out.values())


class FeatureGetterNet(nn.ModuleDict):
    """ FeatureGetterNet

    Wrap models with a feature getter method, like 'get_intermediate_layers'

    """
    def __init__(
            self,
            model: nn.Module,
            out_indices: OutIndicesT = 4,
            out_map: Optional[Sequence[Union[int, str]]] = None,
            return_dict: bool = False,
            output_fmt: str = 'NCHW',
            norm: bool = False,
            prune: bool = True,
    ):
        """

        Args:
            model: Model to wrap.
            out_indices: Indices of features to extract.
            out_map: Remap feature names for dict output (WIP, not supported).
            return_dict: Return features as dictionary instead of list (WIP, not supported).
            norm: Apply final model norm to all output features (if possible).
        """
        super().__init__()
        if prune and hasattr(model, 'prune_intermediate_layers'):
            # replace out_indices after they've been normalized, -ve indices will be invalid after prune
            out_indices = model.prune_intermediate_layers(
                out_indices,
                prune_norm=not norm,
            )
        self.feature_info = _get_feature_info(model, out_indices)
        self.model = model
        self.out_indices = out_indices
        self.out_map = out_map
        self.return_dict = return_dict
        self.output_fmt = Format(output_fmt)
        self.norm = norm

    def forward(self, x):
        features = self.model.forward_intermediates(
            x,
            indices=self.out_indices,
            norm=self.norm,
            output_fmt=self.output_fmt,
            intermediates_only=True,
        )
        return features
