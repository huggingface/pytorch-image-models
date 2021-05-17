from functools import partial

import torch

from timm.utils.agc import adaptive_clip_grad


def get_clip_grad_fn(mode: str = 'norm', norm_type: float = 2.0):
    if mode == 'norm':
        return partial(torch.nn.utils.clip_grad_norm_, norm_type=norm_type)
    elif mode == 'value':
        return torch.nn.utils.clip_grad_value_
    elif mode == 'agc':
        return partial(adaptive_clip_grad, norm_type=norm_type)
    else:
        assert False, f"Unknown clip mode ({mode})."


def get_clip_parameters(model, skip_last=0):
    if hasattr(model, 'get_clip_parameters'):
        return model.get_clip_parameters()
    else:
        if skip_last:
            return list(model.parameters())[::-skip_last]
        else:
            return model.parameters()
