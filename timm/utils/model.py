""" Model / state_dict utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import fnmatch

import torch
from torchvision.ops.misc import FrozenBatchNorm2d


_SUB_MODULE_ATTR = ('module', 'model')


def unwrap_model(model, recursive=True):
    for attr in _SUB_MODULE_ATTR:
        sub_module = getattr(model, attr, None)
        if sub_module is not None:
            return unwrap_model(sub_module) if recursive else sub_module
    return model


def get_state_dict(model, unwrap_fn=unwrap_model):
    return unwrap_fn(model).state_dict()


def avg_sq_ch_mean(model, input, output):
    """ calculate average channel square mean of output activations
    """
    return torch.mean(output.mean(axis=[0, 2, 3]) ** 2).item()


def avg_ch_var(model, input, output):
    """ calculate average channel variance of output activations
    """
    return torch.mean(output.var(axis=[0, 2, 3])).item()


def avg_ch_var_residual(model, input, output):
    """ calculate average channel variance of output activations
    """
    return torch.mean(output.var(axis=[0, 2, 3])).item()


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


def extract_spp_stats(
        model,
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


def freeze_batch_norm_2d(module):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    if isinstance(module, (torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for name, child in module.named_children():
            new_child = freeze_batch_norm_2d(child)
            if new_child is not child:
                res.add_module(name, new_child)
    return res


def unfreeze_batch_norm_2d(module):
    """
    Converts all `FrozenBatchNorm2d` layers of provided module into `BatchNorm2d`. If `module` is itself and instance
    of `FrozenBatchNorm2d`, it is converted into `BatchNorm2d` and returned. Otherwise, the module is walked
    recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    if isinstance(module, FrozenBatchNorm2d):
        res = torch.nn.BatchNorm2d(module.num_features)
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for name, child in module.named_children():
            new_child = unfreeze_batch_norm_2d(child)
            if new_child is not child:
                res.add_module(name, new_child)
    return res


def _freeze_unfreeze(root_module, submodules=[], include_bn_running_stats=True, mode='freeze'):
    """
    Freeze or unfreeze parameters of the specified modules and those of all their hierarchical descendants. This is
    done in place.
    Args:
        root_module (nn.Module, optional): Root module relative to which the `submodules` are referenced.
        submodules (list[str]): List of modules for which the parameters will be (un)frozen. They are to be provided as
            named modules relative to the root module (accessible via `root_module.named_modules()`). An empty list
            means that the whole root module will be (un)frozen. Defaults to []
        include_bn_running_stats (bool): Whether to also (un)freeze the running statistics of batch norm 2d layers.
            Defaults to `True`.
        mode (bool): Whether to freeze ("freeze") or unfreeze ("unfreeze"). Defaults to `"freeze"`.
    """
    assert mode in ["freeze", "unfreeze"], '`mode` must be one of "freeze" or "unfreeze"'

    if isinstance(root_module, (torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.batchnorm.SyncBatchNorm)):
        # Raise assertion here because we can't convert it in place
        raise AssertionError(
            "You have provided a batch norm layer as the `root module`. Please use "
            "`timm.utils.model.freeze_batch_norm_2d` or `timm.utils.model.unfreeze_batch_norm_2d` instead.")

    if isinstance(submodules, str):
        submodules = [submodules]

    named_modules = submodules
    submodules = [root_module.get_submodule(m) for m in submodules]

    if not len(submodules):
        named_modules, submodules = list(zip(*root_module.named_children()))

    for n, m in zip(named_modules, submodules):
        # (Un)freeze parameters
        for p in m.parameters():
            p.requires_grad = False if mode == 'freeze' else True
        if include_bn_running_stats:
            # Helper to add submodule specified as a named_module
            def _add_submodule(module, name, submodule):
                split = name.rsplit('.', 1)
                if len(split) > 1:
                    module.get_submodule(split[0]).add_module(split[1], submodule)
                else:
                    module.add_module(name, submodule)

            # Freeze batch norm
            if mode == 'freeze':
                res = freeze_batch_norm_2d(m)
                # It's possible that `m` is a type of BatchNorm in itself, in which case `unfreeze_batch_norm_2d` won't
                # convert it in place, but will return the converted result. In this case `res` holds the converted
                # result and we may try to re-assign the named module
                if isinstance(m, (torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.batchnorm.SyncBatchNorm)):
                    _add_submodule(root_module, n, res)
            # Unfreeze batch norm
            else:
                res = unfreeze_batch_norm_2d(m)
                # Ditto. See note above in mode == 'freeze' branch
                if isinstance(m, FrozenBatchNorm2d):
                    _add_submodule(root_module, n, res)


def freeze(root_module, submodules=[], include_bn_running_stats=True):
    """
    Freeze parameters of the specified modules and those of all their hierarchical descendants. This is done in place.
    Args:
        root_module (nn.Module): Root module relative to which `submodules` are referenced.
        submodules (list[str]): List of modules for which the parameters will be frozen. They are to be provided as
            named modules relative to the root module (accessible via `root_module.named_modules()`). An empty list
            means that the whole root module will be frozen. Defaults to `[]`.
        include_bn_running_stats (bool): Whether to also freeze the running statistics of `BatchNorm2d` and
            `SyncBatchNorm` layers. These will be converted to `FrozenBatchNorm2d` in place. Hint: During fine tuning,
            it's good practice to freeze batch norm stats. And note that these are different to the affine parameters
            which are just normal PyTorch parameters. Defaults to `True`.

    Hint: If you want to freeze batch norm ONLY, use `timm.utils.model.freeze_batch_norm_2d`.

    Examples::

        >>> model = timm.create_model('resnet18')
        >>> # Freeze up to and including layer2
        >>> submodules = [n for n, _ in model.named_children()]
        >>> print(submodules)
        ['conv1', 'bn1', 'act1', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'global_pool', 'fc']
        >>> freeze(model, submodules[:submodules.index('layer2') + 1])
        >>> # Check for yourself that it works as expected
        >>> print(model.layer2[0].conv1.weight.requires_grad)
        False
        >>> print(model.layer3[0].conv1.weight.requires_grad)
        True
        >>> # Unfreeze
        >>> unfreeze(model)
    """
    _freeze_unfreeze(root_module, submodules, include_bn_running_stats=include_bn_running_stats, mode="freeze")


def unfreeze(root_module, submodules=[], include_bn_running_stats=True):
    """
    Unfreeze parameters of the specified modules and those of all their hierarchical descendants. This is done in place.
    Args:
        root_module (nn.Module): Root module relative to which `submodules` are referenced.
        submodules (list[str]): List of submodules for which the parameters will be (un)frozen. They are to be provided
            as named modules relative to the root module (accessible via `root_module.named_modules()`). An empty
            list means that the whole root module will be unfrozen. Defaults to `[]`.
        include_bn_running_stats (bool): Whether to also unfreeze the running statistics of `FrozenBatchNorm2d` layers.
            These will be converted to `BatchNorm2d` in place. Defaults to `True`.

    See example in docstring for `freeze`.
    """
    _freeze_unfreeze(root_module, submodules, include_bn_running_stats=include_bn_running_stats, mode="unfreeze")
