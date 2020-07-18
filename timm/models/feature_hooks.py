""" PyTorch Feature Hook Helper

This class helps gather features from a network via hooks specified on the module name.

Hacked together by Ross Wightman
"""
import torch

from collections import defaultdict, OrderedDict
from functools import partial, partialmethod
from typing import List


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

    def get_output(self, device) -> List[torch.tensor]:
        if self.out_as_dict:
            output = self._feature_outputs[device]
        else:
            output = list(self._feature_outputs[device].values())
        self._feature_outputs[device] = OrderedDict()  # clear after reading
        return output
