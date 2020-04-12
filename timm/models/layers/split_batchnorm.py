""" Split BatchNorm

A PyTorch BatchNorm layer that splits input batch into N equal parts and passes each through
a separate BN layer. The first split is passed through the parent BN layers with weight/bias
keys the same as the original BN. All other splits pass through BN sub-layers under the '.aux_bn'
namespace.

This allows easily removing the auxiliary BN layers after training to efficiently
achieve the 'Auxiliary BatchNorm' as described in the AdvProp Paper, section 4.2,
'Disentangled Learning via An Auxiliary BN'

Hacked together by Ross Wightman
"""
import torch
import torch.nn as nn


class SplitBatchNorm2d(torch.nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, num_splits=2):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        assert num_splits > 1, 'Should have at least one aux BN layer (num_splits at least 2)'
        self.num_splits = num_splits
        self.aux_bn = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_splits - 1)])

    def forward(self, input: torch.Tensor):
        if self.training:  # aux BN only relevant while training
            split_size = input.shape[0] // self.num_splits
            assert input.shape[0] == split_size * self.num_splits, "batch size must be evenly divisible by num_splits"
            split_input = input.split(split_size)
            x = [super().forward(split_input[0])]
            for i, a in enumerate(self.aux_bn):
                x.append(a(split_input[i + 1]))
            return torch.cat(x, dim=0)
        else:
            return super().forward(input)


def convert_splitbn_model(module, num_splits=2):
    """
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with `SplitBatchnorm2d`.
    Args:
        module (torch.nn.Module): input module
        num_splits: number of separate batchnorm layers to split input across
    Example::
        >>> # model is an instance of torch.nn.Module
        >>> model = timm.models.convert_splitbn_model(model, num_splits=2)
    """
    mod = module
    if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
        return module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = SplitBatchNorm2d(
            module.num_features, module.eps, module.momentum, module.affine,
            module.track_running_stats, num_splits=num_splits)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        mod.num_batches_tracked = module.num_batches_tracked
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
        for aux in mod.aux_bn:
            aux.running_mean = module.running_mean.clone()
            aux.running_var = module.running_var.clone()
            aux.num_batches_tracked = module.num_batches_tracked.clone()
            if module.affine:
                aux.weight.data = module.weight.data.clone().detach()
                aux.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_splitbn_model(child, num_splits=num_splits))
    del module
    return mod
