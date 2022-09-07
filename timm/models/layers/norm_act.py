""" Normalization + Activation Layers

Provides Norm+Act fns for standard PyTorch norm layers such as
* BatchNorm
* GroupNorm
* LayerNorm

This allows swapping with alternative layers that are natively both norm + act such as
* EvoNorm (evo_norm.py)
* FilterResponseNorm (filter_response_norm.py)
* InplaceABN (inplace_abn.py)

Hacked together by / Copyright 2022 Ross Wightman
"""
from typing import Union, List, Optional, Any

import torch
from torch import nn as nn
from torch.nn import functional as F

from .create_act import get_act_layer
from .fast_norm import is_fast_norm, fast_group_norm, fast_layer_norm
from .trace_utils import _assert


class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            apply_act=True,
            act_layer=nn.ReLU,
            inplace=True,
            drop_layer=None,
            device=None,
            dtype=None
    ):
        try:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super(BatchNormAct2d, self).__init__(
                num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats,
                **factory_kwargs
            )
        except TypeError:
            # NOTE for backwards compat with old PyTorch w/o factory device/dtype support
            super(BatchNormAct2d, self).__init__(
                num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        act_layer = get_act_layer(act_layer)  # string -> nn.Module
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        # cut & paste of torch.nn.BatchNorm2d.forward impl to avoid issues with torchscript and tracing
        _assert(x.ndim == 4, f'expected 4D input (got {x.ndim}D input)')

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        x = self.drop(x)
        x = self.act(x)
        return x


class SyncBatchNormAct(nn.SyncBatchNorm):
    # Thanks to Selim Seferbekov (https://github.com/rwightman/pytorch-image-models/issues/1254)
    # This is a quick workaround to support SyncBatchNorm for timm BatchNormAct2d layers
    # but ONLY when used in conjunction with the timm conversion function below.
    # Do not create this module directly or use the PyTorch conversion function.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)  # SyncBN doesn't work with torchscript anyways, so this is fine
        if hasattr(self, "drop"):
            x = self.drop(x)
        if hasattr(self, "act"):
            x = self.act(x)
        return x


def convert_sync_batchnorm(module, process_group=None):
    # convert both BatchNorm and BatchNormAct layers to Synchronized variants
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if isinstance(module, BatchNormAct2d):
            # convert timm norm + act layer
            module_output = SyncBatchNormAct(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group=process_group,
            )
            # set act and drop attr from the original module
            module_output.act = module.act
            module_output.drop = module.drop
        else:
            # convert standard BatchNorm layers
            module_output = torch.nn.SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, convert_sync_batchnorm(child, process_group))
    del module
    return module_output


def group_norm_tpu(x, w, b, groups: int = 32, eps: float = 1e-5, diff_sqm: bool = False, flatten: bool = False):
    # This is a workaround for some odd behaviour running on PyTorch XLA w/ TPUs.
    x_shape = x.shape
    x_dtype = x.dtype
    if flatten:
        norm_shape = (x_shape[0], groups, -1)
        reduce_dim = -1
    else:
        norm_shape = (x_shape[0], groups, x_shape[1] // groups) + x_shape[2:]
        reduce_dim = tuple(range(2, x.ndim + 1))
    affine_shape = (1, -1) + (1,) * (x.ndim - 2)
    x = x.reshape(norm_shape)
    # x = x.to(torch.float32)  # for testing w/ AMP
    xm = x.mean(dim=reduce_dim, keepdim=True)
    if diff_sqm:
        # difference of squared mean and mean squared, faster on TPU
        var = (x.square().mean(dim=reduce_dim, keepdim=True) - xm.square()).clamp(0)
    else:
        var = (x - xm).square().mean(dim=reduce_dim, keepdim=True)
    x = (x - xm.expand(norm_shape)) / var.add(eps).sqrt().expand(norm_shape)
    x = x.reshape(x_shape) * w.view(affine_shape) + b.view(affine_shape)
    # x = x.to(x_dtype)  # for testing w/ AMP
    return x


def _num_groups(num_channels, num_groups, group_size):
    if group_size:
        assert num_channels % group_size == 0
        return num_channels // group_size
    return num_groups


class GroupNormAct(nn.GroupNorm):
    # NOTE num_channel and num_groups order flipped for easier layer swaps / binding of fixed args
    def __init__(
            self, num_channels, num_groups=32, eps=1e-5, affine=True, group_size=None,
            apply_act=True, act_layer=nn.ReLU, inplace=True, drop_layer=None):
        super(GroupNormAct, self).__init__(
            _num_groups(num_channels, num_groups, group_size), num_channels, eps=eps, affine=affine)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        act_layer = get_act_layer(act_layer)  # string -> nn.Module
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        if False:  # FIXME TPU temporary while resolving some performance issues
            x = group_norm_tpu(x, self.weight, self.bias, self.num_groups, self.eps)
        else:
            if self._fast_norm:
                x = fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
            else:
                x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x


class LayerNormAct(nn.LayerNorm):
    def __init__(
            self, normalization_shape: Union[int, List[int], torch.Size], eps=1e-5, affine=True,
            apply_act=True, act_layer=nn.ReLU, inplace=True, drop_layer=None):
        super(LayerNormAct, self).__init__(normalization_shape, eps=eps, elementwise_affine=affine)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        act_layer = get_act_layer(act_layer)  # string -> nn.Module
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x


class LayerNormAct2d(nn.LayerNorm):
    def __init__(
            self, num_channels, eps=1e-5, affine=True,
            apply_act=True, act_layer=nn.ReLU, inplace=True, drop_layer=None):
        super(LayerNormAct2d, self).__init__(num_channels, eps=eps, elementwise_affine=affine)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        act_layer = get_act_layer(act_layer)  # string -> nn.Module
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        x = self.drop(x)
        x = self.act(x)
        return x
