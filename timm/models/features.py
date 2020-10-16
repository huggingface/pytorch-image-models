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
from typing import Dict, List, Tuple

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

    def get(self, key, idx=None):
        """ Get value by key at specified index (indices)
        if idx == None, returns value for key at each output index
        if idx is an integer, return value for that feature module index (ignoring output indices)
        if idx is a list/tupple, return value for each module index (ignoring output indices)
        """
        if idx is None:
            return [self.info[i][key] for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [self.info[i][key] for i in idx]
        else:
            return self.info[idx][key]

    def get_dicts(self, keys=None, idx=None):
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

    def channels(self, idx=None):
        """ feature channels accessor
        """
        return self.get('num_chs', idx)

    def reduction(self, idx=None):
        """ feature reduction (output stride) accessor
        """
        return self.get('reduction', idx)

    def module_name(self, idx=None):
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
    internal nodes in a model by node name. This works quite well in eager Python but needs
    redesign for torcscript.
    """

    def __init__(self, hooks, named_modules, out_map=None, default_hook_type='forward'):
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


def _get_feature_info(net, out_indices):
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

    Arguments:
        model (nn.Module): model from which we will extract the features
        out_indices (tuple[int]): model output indices to extract features for
        out_map (sequence): list or tuple specifying desired return id for each out index,
            otherwise str(index) is used
        feature_concat (bool): whether to concatenate intermediate features that are lists or tuples
            vs select element [0]
        flatten_sequential (bool): whether to flatten sequential modules assigned to model
    """
    def __init__(
            self, model,
            out_indices=(0, 1, 2, 3, 4), out_map=None, feature_concat=False, flatten_sequential=False):
        super(FeatureDictNet, self).__init__()
        self.feature_info = _get_feature_info(model, out_indices)
        self.concat = feature_concat
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

    def _collect(self, x) -> (Dict[str, torch.Tensor]):
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

    def forward(self, x) -> Dict[str, torch.Tensor]:
        return self._collect(x)


class FeatureListNet(FeatureDictNet):
    """ Feature extractor with list return

    See docstring for FeatureDictNet above, this class exists only to appease Torchscript typing constraints.
    In eager Python we could have returned List[Tensor] vs Dict[id, Tensor] based on a member bool.
    """
    def __init__(
            self, model,
            out_indices=(0, 1, 2, 3, 4), out_map=None, feature_concat=False, flatten_sequential=False):
        super(FeatureListNet, self).__init__(
            model, out_indices=out_indices, out_map=out_map, feature_concat=feature_concat,
            flatten_sequential=flatten_sequential)

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
            self, model,
            out_indices=(0, 1, 2, 3, 4), out_map=None, out_as_dict=False, no_rewrite=False,
            feature_concat=False, flatten_sequential=False, default_hook_type='forward'):
        super(FeatureHookNet, self).__init__()
        assert not torch.jit.is_scripting()
        self.feature_info = _get_feature_info(model, out_indices)
        self.out_as_dict = out_as_dict
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
            remaining = {f['module']: f['hook_type'] if 'hook_type' in f else default_hook_type
                         for f in self.feature_info.get_dicts()}
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

    def forward(self, x):
        for name, module in self.items():
            x = module(x)
        out = self.hooks.get_output(x.device)
        return out if self.out_as_dict else list(out.values())
