""" PyTorch Feature Extraction Helpers

A collection of classes, functions, modules to help extract features from models
and provide a common interface for describing them.

Hacked together by Ross Wightman
"""
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import partial
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn


class FeatureInfo:

    def __init__(self, feature_info: List[Dict], out_indices: Tuple[int]):
        prev_reduction = 1
        for fi in feature_info:
            # sanity check the mandatory fields, there may be additional fields depending on the model
            assert 'num_chs' in fi and fi['num_chs'] > 0
            assert 'reduction' in fi and fi['reduction'] >= prev_reduction
            prev_reduction = fi['reduction']
            assert 'module' in fi
        self.out_indices = out_indices
        self.info = feature_info

    def from_other(self, out_indices: Tuple[int]):
        return FeatureInfo(deepcopy(self.info), out_indices)

    def channels(self, idx=None):
        """ feature channels accessor
        if idx == None, returns feature channel count at each output index
        if idx is an integer, return feature channel count for that feature module index
        """
        if isinstance(idx, int):
            return self.info[idx]['num_chs']
        return [self.info[i]['num_chs'] for i in self.out_indices]

    def reduction(self, idx=None):
        """ feature reduction (output stride) accessor
        if idx == None, returns feature reduction factor at each output index
        if idx is an integer, return feature channel count at that feature module index
        """
        if isinstance(idx, int):
            return self.info[idx]['reduction']
        return [self.info[i]['reduction'] for i in self.out_indices]

    def module_name(self, idx=None):
        """ feature module name accessor
        if idx == None, returns feature module name at each output index
        if idx is an integer, return feature module name at that feature module index
        """
        if isinstance(idx, int):
            return self.info[idx]['module']
        return [self.info[i]['module'] for i in self.out_indices]

    def get_by_key(self, idx=None, keys=None):
        """ return info dicts for specified keys (or all if None) at specified idx (or out_indices if None)
        """
        if isinstance(idx, int):
            return self.info[idx] if keys is None else {k: self.info[idx][k] for k in keys}
        if keys is None:
            return [self.info[i] for i in self.out_indices]
        else:
            return [{k: self.info[i][k] for k in keys} for i in self.out_indices]

    def __getitem__(self, item):
        return self.info[item]

    def __len__(self):
        return len(self.info)


class FeatureHooks:

    def __init__(self, hooks, named_modules, out_as_dict=False, out_map=None, default_hook_type='forward'):
        # setup feature hooks
        modules = {k: v for k, v in named_modules}
        for i, h in enumerate(hooks):
            hook_name = h['module']
            m = modules[hook_name]
            hook_id = out_map[i] if out_map else hook_name
            hook_fn = partial(self._collect_output_hook, hook_id)
            hook_type = h['hook_type'] if 'hook_type' in h else default_hook_type
            if hook_type == 'forward_pre':
                m.register_forward_pre_hook(hook_fn)
            elif hook_type == 'forward':
                m.register_forward_hook(hook_fn)
            else:
                assert False, "Unsupported hook type"
        self._feature_outputs = defaultdict(OrderedDict)
        self.out_as_dict = out_as_dict

    def _collect_output_hook(self, hook_id, *args):
        x = args[-1]  # tensor we want is last argument, output for fwd, input for fwd_pre
        if isinstance(x, tuple):
            x = x[0]  # unwrap input tuple
        self._feature_outputs[x.device][hook_id] = x

    def get_output(self, device) -> List[torch.tensor]:   # FIXME deal with diff return types for torchscript?
        if self.out_as_dict:
            output = self._feature_outputs[device]
        else:
            output = list(self._feature_outputs[device].values())
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


class LayerGetterHooks(nn.ModuleDict):
    """ LayerGetterHooks
    TODO
    """

    def __init__(self, model, feature_info, flatten_sequential=False, out_as_dict=False, out_map=None,
                 default_hook_type='forward'):
        modules = _module_list(model, flatten_sequential=flatten_sequential)
        remaining = {f['module']: f['hook_type'] if 'hook_type' in f else default_hook_type for f in feature_info}
        layers = OrderedDict()
        hooks = []
        for new_name, old_name, module in modules:
            layers[new_name] = module
            for fn, fm in module.named_modules(prefix=old_name):
                if fn in remaining:
                    hooks.append(dict(module=fn, hook_type=remaining[fn]))
                    del remaining[fn]
            if not remaining:
                break
        assert not remaining, f'Return layers ({remaining}) are not present in model'
        super(LayerGetterHooks, self).__init__(layers)
        self.hooks = FeatureHooks(hooks, model.named_modules(), out_as_dict=out_as_dict, out_map=out_map)

    def forward(self, x) -> Dict[Any, torch.Tensor]:
        for name, module in self.items():
            x = module(x)
        return self.hooks.get_output(x.device)


class LayerGetterDict(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model as a dictionary

    Originally based on concepts from IntermediateLayerGetter at
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
        self.return_layers = {}
        self.concat = concat
        modules = _module_list(model, flatten_sequential=flatten_sequential)
        remaining = set(return_layers.keys())
        layers = OrderedDict()
        for new_name, old_name, module in modules:
            layers[new_name] = module
            if old_name in remaining:
                self.return_layers[new_name] = return_layers[old_name]
                remaining.remove(old_name)
            if not remaining:
                break
        assert not remaining and len(self.return_layers) == len(return_layers), \
            f'Return layers ({remaining}) are not present in model'
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

    Originally based on concepts from IntermediateLayerGetter at
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
        self.return_layers = {}
        self.concat = concat
        modules = _module_list(model, flatten_sequential=flatten_sequential)
        remaining = set(return_layers.keys())
        for new_name, orig_name, module in modules:
            self.add_module(new_name, module)
            if orig_name in remaining:
                self.return_layers[new_name] = return_layers[orig_name]
                remaining.remove(orig_name)
            if not remaining:
                break
        assert not remaining and len(self.return_layers) == len(return_layers), \
            f'Return layers ({remaining}) are not present in model'

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


def _get_return_layers(feature_info, out_map):
    module_names = feature_info.module_name()
    return_layers = {}
    for i, name in enumerate(module_names):
        return_layers[name] = out_map[i] if out_map is not None else feature_info.out_indices[i]
    return return_layers


class FeatureNet(nn.Module):
    """ FeatureNet

    Wrap a model and extract features as specified by the out indices, the network
    is partially re-built from contained modules using the LayerGetters.

    Please read the docstrings of the LayerGetter classes, they will not work on all models.
    """
    def __init__(
            self, net,
            out_indices=(0, 1, 2, 3, 4), out_map=None, out_as_dict=False, use_hooks=False,
            feature_info=None, feature_concat=False, flatten_sequential=False):
        super(FeatureNet, self).__init__()
        self.feature_info = _resolve_feature_info(net, out_indices, feature_info)
        if use_hooks:
            self.body = LayerGetterHooks(net, self.feature_info, out_as_dict=out_as_dict, out_map=out_map)
        else:
            return_layers = _get_return_layers(self.feature_info, out_map)
            lg_args = dict(return_layers=return_layers, concat=feature_concat, flatten_sequential=flatten_sequential)
            self.body = LayerGetterDict(net, **lg_args) if out_as_dict else LayerGetterList(net, **lg_args)

    def forward(self, x):
        output = self.body(x)
        return output


class FeatureHookNet(nn.Module):
    """ FeatureHookNet

    Wrap a model and extract features specified by the out indices.

    Features are extracted via hooks without modifying the underlying network in any way. If only
    part of the model is used it is up to the caller to remove unneeded layers as this wrapper
    does not rewrite and remove unused top-level modules like FeatureNet with LayerGetter.
    """
    def __init__(
            self, net,
            out_indices=(0, 1, 2, 3, 4), out_as_dict=False, out_map=None,
            feature_info=None, feature_concat=False):
        super(FeatureHookNet, self).__init__()
        self.feature_info = _resolve_feature_info(net, out_indices, feature_info)
        self.body = net
        self.hooks = FeatureHooks(
            self.feature_info, self.body.named_modules(), out_as_dict=out_as_dict, out_map=out_map)

    def forward(self, x):
        self.body(x)
        return self.hooks.get_output(x.device)
