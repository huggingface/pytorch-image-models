""" Model / state_dict utils

Hacked together by / Copyright 2020 Ross Wightman
"""
from .model_ema import ModelEma
import torch 


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
    return torch.mean(output.var(axis=[0,2,3])).item()


class ActivationStatsHook:
    """Iterates through each of `model`'s modules and if module's class name 
    is present in `layer_names` then registers `hook_fns` inside that module
    and stores activation stats inside `self.stats`.

    Arguments:
        model (nn.Module): model from which we will extract the activation stats
        layer_names (List[str]): The layer name to look for to register forward 
            hook. Example, `BasicBlock`, `Bottleneck`
        hook_fns (List[Callable]): List of hook functions to be registered at every
            module in `layer_names`.
    
    Inspiration from https://docs.fast.ai/callback.hook.html.
    """

    def __init__(self, model, layer_names, hook_fns=[avg_sq_ch_mean, avg_ch_var]):
        self.model = model
        self.layer_names = layer_names 
        self.hook_fns = hook_fns
        self.stats = dict((hook_fn.__name__, []) for hook_fn in hook_fns)
        for hook_fn in hook_fns: 
            self.register_hook(layer_names, hook_fn)

    def _create_hook(self, hook_fn):
        def append_activation_stats(module, input, output):
            out = hook_fn(module, input, output)
            self.stats[hook_fn.__name__].append(out)
        return append_activation_stats
        
    def register_hook(self, layer_names, hook_fn):
        for layer in self.model.modules():
            layer_name = layer.__class__.__name__
            if layer_name not in layer_names: 
                continue
            layer.register_forward_hook(self._create_hook(hook_fn))


def extract_spp_stats(model, 
                      layer_names, 
                      hook_fns=[avg_sq_ch_mean, avg_ch_var], 
                      input_shape=[8, 3, 224, 224]):
    """Extract average square channel mean and variance of activations during 
    forward pass to plot Signal Propogation Plots (SPP).
    
    Paper: https://arxiv.org/abs/2101.08692
    """ 
    x = torch.normal(0., 1., input_shape)
    hook = ActivationStatsHook(model, layer_names, hook_fns)
    _ = model(x)
    return hook.stats
    