""" Normalization + Activation Layers
"""
from typing import Union, List

import torch
from torch import nn as nn
from torch.nn import functional as F

from .trace_utils import _assert
from .create_act import get_act_layer


class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """
    def __init__(
            self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
            apply_act=True, act_layer=nn.ReLU, inplace=True, drop_layer=None):
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

    def forward(self, x):
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

    def forward(self, x):
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

    def forward(self, x):
        x = F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        x = self.drop(x)
        x = self.act(x)
        return x
