""" Normalization layers and wrappers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True):
        # NOTE num_channels is swapped to first arg for consistency in swapping norm layers with BN
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)

    def forward(self, x):
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial BCHW tensors """
    def __init__(self, num_channels):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class FrozenBatchNorm2d(torchvision.ops.misc.FrozenBatchNorm2d):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Inherits from torchvision while adding the `convert_frozen_batchnorm` from
    https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/layers/batch_norm.py
    """

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Converts all BatchNorm layers of provided module into FrozenBatchNorm. If `module` is a type of BatchNorm, it
        converts it into FrozenBatchNorm. Otherwise, the module is walked recursively and BatchNorm type layers are
        converted in place.

        Args:
            module (torch.nn.Module): Any PyTorch module. It doesn't have to be a BatchNorm variant in itself.

        Returns:
            torch.nn.Module: Resulting module
        """
        res = module
        if isinstance(module,  (nn.modules.batchnorm.BatchNorm2d,  nn.modules.batchnorm.SyncBatchNorm)):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

