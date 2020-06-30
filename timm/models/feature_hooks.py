""" PyTorch Feature Hook Helper

This class helps gather features from a network via hooks specified on the module name.

Hacked together by Ross Wightman
"""
import torch

from collections import defaultdict, OrderedDict
from functools import partial
from typing import List


class FeatureHooks:

    def __init__(self, hooks, named_modules, output_as_dict=False):
        # setup feature hooks
        modules = {k: v for k, v in named_modules}
        for h in hooks:
            hook_name = h['module']
            m = modules[hook_name]
            hook_fn = partial(self._collect_output_hook, hook_name)
            if h['hook_type'] == 'forward_pre':
                m.register_forward_pre_hook(hook_fn)
            elif h['hook_type'] == 'forward':
                m.register_forward_hook(hook_fn)
            else:
                assert False, "Unsupported hook type"
        self._feature_outputs = defaultdict(OrderedDict)
        self.output_as_dict = output_as_dict

    def _collect_output_hook(self, name, *args):
        x = args[-1]  # tensor we want is last argument, output for fwd, input for fwd_pre
        if isinstance(x, tuple):
            x = x[0]  # unwrap input tuple
        self._feature_outputs[x.device][name] = x

    def get_output(self, device) -> List[torch.tensor]:
        if self.output_as_dict:
            output = self._feature_outputs[device]
        else:
            output = list(self._feature_outputs[device].values())
        self._feature_outputs[device] = OrderedDict()  # clear after reading
        return output
