""" Normalization + Activation Layers
"""
from torch import nn as nn
from torch.nn import functional as F


class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm + Activation

    This module performs BatchNorm + Actibation in s manner that will remain bavkwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, act_layer=nn.ReLU, inplace=True):
        super(BatchNormAct2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.act = act_layer(inplace=inplace)

    def forward(self, x):
        # FIXME cannot call parent forward() and maintain jit.script compatibility?
        # x = super(BatchNormAct2d, self).forward(x)

        # BEGIN nn.BatchNorm2d forward() cut & paste
        # self._check_input_dim(x)

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
        # END BatchNorm2d forward()

        x = self.act(x)
        return x
