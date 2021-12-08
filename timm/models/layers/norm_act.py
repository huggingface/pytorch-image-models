""" Normalization + Activation Layers
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from .create_act import get_act_layer


class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 apply_act=True, act_layer=nn.ReLU, inplace=True, drop_block=None):
        super(BatchNormAct2d, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        if isinstance(act_layer, str):
            act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()

    def _forward_jit(self, x):
        """ A cut & paste of the contents of the PyTorch BatchNorm2d forward function
        """
        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        x = F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        return x

    @torch.jit.ignore
    def _forward_python(self, x):
        return super(BatchNormAct2d, self).forward(x)

    def forward(self, x):
        # FIXME cannot call parent forward() and maintain jit.script compatibility?
        if torch.jit.is_scripting():
            x = self._forward_jit(x)
        else:
            x = self._forward_python(x)
        x = self.act(x)
        return x


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


class GroupNormAct(nn.GroupNorm):
    # NOTE num_channel and num_groups order flipped for easier layer swaps / binding of fixed args
    def __init__(self, num_channels, num_groups=32, eps=1e-5, affine=True,
                 apply_act=True, act_layer=nn.ReLU, inplace=True, drop_block=None):
        super(GroupNormAct, self).__init__(num_groups, num_channels, eps=eps, affine=affine)
        if isinstance(act_layer, str):
            act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        if False:  # FIXME TPU temporary while resolving some performance issues
            x = group_norm_tpu(x, self.weight, self.bias, self.num_groups, self.eps)
        else:
            x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        x = self.act(x)
        return x
