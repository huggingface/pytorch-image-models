""" Model / state_dict utils

Hacked together by / Copyright 2020 Ross Wightman
"""
from .model_ema import ModelEma
import torch 
import fnmatch

def unwrap_model(model):
    if isinstance(model, ModelEma):
        return unwrap_model(model.ema)
    else:
        return model.module if hasattr(model, 'module') else model


def get_state_dict(model, unwrap_fn=unwrap_model):
    return unwrap_fn(model).state_dict()


def avg_sq_ch_mean(model, input, output): 
    "calculate average channel square mean of output activations"
    return torch.mean(output.mean(axis=[0,2,3])**2).item()


def avg_ch_var(model, input, output): 
    "calculate average channel variance of output activations"
    return torch.mean(output.var(axis=[0,2,3])).item()\


def avg_ch_var_residual(model, input, output): 
    "calculate average channel variance of output activations"
    return torch.mean(output.var(axis=[0,2,3])).item()


class ActivationStatsHook:
    """Iterates through each of `model`'s modules and matches modules using unix pattern 
    matching based on `hook_fn_locs` and registers `hook_fn` to the module if there is 
    a match. 

    Arguments:
        model (nn.Module): model from which we will extract the activation stats
        hook_fn_locs (List[str]): List of `hook_fn` locations based on Unix type string 
            matching with the name of model's modules. 
        hook_fns (List[Callable]): List of hook functions to be registered at every
            module in `layer_names`.
    
    Inspiration from https://docs.fast.ai/callback.hook.html.

    Refer to https://gist.github.com/amaarora/6e56942fcb46e67ba203f3009b30d950 for an example 
    on how to plot Signal Propogation Plots using `ActivationStatsHook`.
    """

    def __init__(self, model, hook_fn_locs, hook_fns):
        self.model = model
        self.hook_fn_locs = hook_fn_locs
        self.hook_fns = hook_fns
        if len(hook_fn_locs) != len(hook_fns):
            raise ValueError("Please provide `hook_fns` for each `hook_fn_locs`, \
                their lengths are different.")
        self.stats = dict((hook_fn.__name__, []) for hook_fn in hook_fns)
        for hook_fn_loc, hook_fn in zip(hook_fn_locs, hook_fns): 
            self.register_hook(hook_fn_loc, hook_fn)

    def _create_hook(self, hook_fn):
        def append_activation_stats(module, input, output):
            out = hook_fn(module, input, output)
            self.stats[hook_fn.__name__].append(out)
        return append_activation_stats
        
    def register_hook(self, hook_fn_loc, hook_fn):
        for name, module in self.model.named_modules():
            if not fnmatch.fnmatch(name, hook_fn_loc):
                continue
            module.register_forward_hook(self._create_hook(hook_fn))


def extract_spp_stats(model, 
                      hook_fn_locs,
                      hook_fns, 
                      input_shape=[8, 3, 224, 224]):
    """Extract average square channel mean and variance of activations during 
    forward pass to plot Signal Propogation Plots (SPP).
    
    Paper: https://arxiv.org/abs/2101.08692

    Example Usage: https://gist.github.com/amaarora/6e56942fcb46e67ba203f3009b30d950
    """ 
    x = torch.normal(0., 1., input_shape)
    hook = ActivationStatsHook(model, hook_fn_locs=hook_fn_locs, hook_fns=hook_fns)
    _ = model(x)
    return hook.stats
    