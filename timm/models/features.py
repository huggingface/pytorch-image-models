""" PyTorch Feature Extraction Helpers

A collection of classes, functions, modules to help extract features from models
and provide a common interface for describing them.

Hacked together by Ross Wightman
"""
from collections import OrderedDict
from typing import Dict, List, Tuple, Any
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureInfo:

    def __init__(self, feature_info: List[Dict], out_indices: Tuple[int]):
        prev_reduction = 1
        for fi in feature_info:
            # sanity check the mandatory fields, there may be additional fields depending on the model
            assert 'num_chs' in fi and fi['num_chs'] > 0
            assert 'reduction' in fi and fi['reduction'] >= prev_reduction
            prev_reduction = fi['reduction']
            assert 'module' in fi
        self._out_indices = out_indices
        self._info = feature_info

    def from_other(self, out_indices: Tuple[int]):
        return FeatureInfo(deepcopy(self._info), out_indices)

    def channels(self, idx=None):
        """ feature channels accessor
        if idx == None, returns feature channel count at each output index
        if idx is an integer, return feature channel count for that feature module index
        """
        if isinstance(idx, int):
            return self._info[idx]['num_chs']
        return [self._info[i]['num_chs'] for i in self._out_indices]

    def reduction(self, idx=None):
        """ feature reduction (output stride) accessor
        if idx == None, returns feature reduction factor at each output index
        if idx is an integer, return feature channel count at that feature module index
        """
        if isinstance(idx, int):
            return self._info[idx]['reduction']
        return [self._info[i]['reduction'] for i in self._out_indices]

    def module_name(self, idx=None):
        """ feature module name accessor
        if idx == None, returns feature module name at each output index
        if idx is an integer, return feature module name at that feature module index
        """
        if isinstance(idx, int):
            return self._info[idx]['module']
        return [self._info[i]['module'] for i in self._out_indices]

    def get_by_key(self, idx=None, keys=None):
        """ return info dicts for specified keys (or all if None) at specified idx (or out_indices if None)
        """
        if isinstance(idx, int):
            return self._info[idx] if keys is None else {k: self._info[idx][k] for k in keys}
        if keys is None:
            return [self._info[i] for i in self._out_indices]
        else:
            return [{k: self._info[i][k] for k in keys} for i in self._out_indices]

    def __getitem__(self, item):
        return self._info[item]

    def __len__(self):
        return len(self._info)


def _module_list(module, flatten_sequential=False):
    # a yield/iter would be better for this but wouldn't be compatible with torchscript
    ml = []
    for name, module in module.named_children():
        if flatten_sequential and isinstance(module, nn.Sequential):
            # first level of Sequential containers is flattened into containing model
            for child_name, child_module in module.named_children():
                ml.append(('_'.join([name, child_name]), child_module))
        else:
            ml.append((name, module))
    return ml


def _check_return_layers(input_return_layers, modules):
    return_layers = {}
    for k, v in input_return_layers.items():
        ks = k.split('.')
        assert 0 < len(ks) <= 2
        return_layers['_'.join(ks)] = v
    return_set = set(return_layers.keys())
    sdiff = return_set - {name for name, _ in modules}
    if sdiff:
        raise ValueError(f'return_layers {sdiff} are not present in model')
    return return_layers, return_set


class LayerGetterDict(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model as a dictionary

    Originally based on IntermediateLayerGetter at
    https://github.com/pytorch/vision/blob/d88d8961ae51507d0cb680329d985b1488b1b76b/torchvision/models/_utils.py

    It has a strong assumption that the modules have been registered into the model in the same
    order as they are used. This means that one should **not** reuse the same nn.Module twice
    in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly assigned to the model
    class (`model.feature1`) or at most one Sequential container deep (`model.features.1`, so
    long as `features` is a sequential container assigned to the model).

    All Sequential containers that are directly assigned to the original model will have their
    modules assigned to this module with the name `model.features.1` being changed to `model.features_1`

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        concat (bool): whether to concatenate intermediate features that are lists or tuples
            vs select element [0]
        flatten_sequential (bool): whether to flatten sequential modules assigned to model

    """

    def __init__(self, model, return_layers, concat=False, flatten_sequential=False):
        modules = _module_list(model, flatten_sequential=flatten_sequential)
        self.return_layers, remaining = _check_return_layers(return_layers, modules)
        layers = OrderedDict()
        self.concat = concat
        for name, module in modules:
            layers[name] = module
            if name in remaining:
                remaining.remove(name)
            if not remaining:
                break
        super(LayerGetterDict, self).__init__(layers)

    def forward(self, x) -> Dict[Any, torch.Tensor]:
        out = OrderedDict()
        for name, module in self.items():
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


class LayerGetterList(nn.Sequential):
    """
    Module wrapper that returns intermediate layers from a model as a list

    Originally based on IntermediateLayerGetter at
    https://github.com/pytorch/vision/blob/d88d8961ae51507d0cb680329d985b1488b1b76b/torchvision/models/_utils.py

    It has a strong assumption that the modules have been registered into the model in the same
    order as they are used. This means that one should **not** reuse the same nn.Module twice
    in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly assigned to the model
    class (`model.feature1`) or at most one Sequential container deep (`model.features.1`) so
    long as `features` is a sequential container assigned to the model and flatten_sequent=True.

    All Sequential containers that are directly assigned to the original model will have their
    modules assigned to this module with the name `model.features.1` being changed to `model.features_1`

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        concat (bool): whether to concatenate intermediate features that are lists or tuples
            vs select element [0]
        flatten_sequential (bool): whether to flatten sequential modules assigned to model

    """

    def __init__(self, model, return_layers, concat=False, flatten_sequential=False):
        super(LayerGetterList, self).__init__()
        modules = _module_list(model, flatten_sequential=flatten_sequential)
        self.return_layers, remaining = _check_return_layers(return_layers, modules)
        self.concat = concat
        for name, module in modules:
            self.add_module(name, module)
            if name in remaining:
                remaining.remove(name)
            if not remaining:
                break

    def forward(self, x) -> List[torch.Tensor]:
        out = []
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                if isinstance(x, (tuple, list)):
                    # If model tap is a tuple or list, concat or select first element
                    # FIXME this may need to be more generic / flexible for some nets
                    out.append(torch.cat(x, 1) if self.concat else x[0])
                else:
                    out.append(x)
        return out


def _resolve_feature_info(net, out_indices, feature_info=None):
    if feature_info is None:
        feature_info = getattr(net, 'feature_info')
    if isinstance(feature_info, FeatureInfo):
        return feature_info.from_other(out_indices)
    elif isinstance(feature_info, (list, tuple)):
        return FeatureInfo(net.feature_info, out_indices)
    else:
        assert False, "Provided feature_info is not valid"


class FeatureNet(nn.Module):
    """ FeatureNet

    Wrap a model and extract features as specified by the out indices, the network
    is partially re-built from contained modules using the LayerGetters.

    Please read the docstrings of the LayerGetter classes, they will not work on all models.
    """
    def __init__(
            self, net,
            out_indices=(0, 1, 2, 3, 4), out_map=None, out_as_dict=False,
            feature_info=None, feature_concat=False, flatten_sequential=False):
        super(FeatureNet, self).__init__()
        self.feature_info = _resolve_feature_info(net, out_indices, feature_info)
        module_names = self.feature_info.module_name()
        return_layers = {}
        for i in range(len(out_indices)):
            return_layers[module_names[i]] = out_map[i] if out_map is not None else out_indices[i]
        lg_args = dict(return_layers=return_layers, concat=feature_concat, flatten_sequential=flatten_sequential)
        self.body = LayerGetterDict(net, **lg_args) if out_as_dict else LayerGetterList(net, **lg_args)

    def forward(self, x):
        output = self.body(x)
        return output
