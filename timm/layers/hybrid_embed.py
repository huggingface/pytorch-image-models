""" Image to Patch Hybird Embedding Layer

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn as nn
import torch.nn.functional as F

from .format import Format, nchw_to
from .helpers import to_2tuple
from .patch_embed import resample_patch_embed


_logger = logging.getLogger(__name__)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            backbone: nn.Module,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 1,
            feature_size: Optional[Union[int, Tuple[int, int]]] = None,
            feature_ratio: Optional[Union[int, Tuple[int, int]]] = None,
            in_chans: int = 3,
            embed_dim: int = 768,
            bias: bool = True,
            proj: bool = True,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        self.backbone = backbone
        self.in_chans = in_chans
        (
            self.img_size,
            self.patch_size,
            self.feature_size,
            self.feature_ratio,
            self.feature_dim,
            self.grid_size,
            self.num_patches,
        ) = self._init_backbone(
            img_size=img_size,
            patch_size=patch_size,
            feature_size=feature_size,
            feature_ratio=feature_ratio,
        )

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        if not dynamic_img_pad:
            assert self.feature_size[0] % self.patch_size[0] == 0 and self.feature_size[1] % self.patch_size[1] == 0

        if proj:
            self.proj = nn.Conv2d(
                self.feature_dim,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=bias,
            )
        else:
            assert self.feature_dim == embed_dim, \
                f'The feature dim ({self.feature_dim} must match embed dim ({embed_dim}) when projection disabled.'
            self.proj = nn.Identity()

    def _init_backbone(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 1,
            feature_size: Optional[Union[int, Tuple[int, int]]] = None,
            feature_ratio: Optional[Union[int, Tuple[int, int]]] = None,
            feature_dim: Optional[int] = None,
    ):
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = self.backbone.training
                if training:
                    self.backbone.eval()
                o = self.backbone(torch.zeros(1, self.in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                self.backbone.train(training)
            feature_ratio = tuple([s // f for s, f in zip(img_size, feature_size)])
        else:
            feature_size = to_2tuple(feature_size)
            feature_ratio = to_2tuple(feature_ratio or 16)
            if feature_dim is None:
                if hasattr(self.backbone, 'feature_info'):
                    feature_dim = self.backbone.feature_info.channels()[-1]
                else:
                    feature_dim = self.backbone.num_features
        grid_size = tuple([f // p for f, p in zip(feature_size, patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, patch_size, feature_size, feature_ratio, feature_dim, grid_size, num_patches

    def set_input_size(
            self,
            img_size: Optional[Union[int, Tuple[int, int]]] = None,
            patch_size: Optional[Union[int, Tuple[int, int]]] = None,
            feature_size: Optional[Union[int, Tuple[int, int]]] = None,
            feature_ratio: Optional[Union[int, Tuple[int, int]]] = None,
            feature_dim: Optional[int] = None,
    ):
        assert img_size is not None or patch_size is not None
        img_size = img_size or self.img_size
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = to_2tuple(patch_size)
        if new_patch_size is not None and new_patch_size != self.patch_size:
            assert isinstance(self.proj, nn.Conv2d), 'HybridEmbed must have a projection layer to change patch size.'
            with torch.no_grad():
                new_proj = nn.Conv2d(
                    self.proj.in_channels,
                    self.proj.out_channels,
                    kernel_size=new_patch_size,
                    stride=new_patch_size,
                    bias=self.proj.bias is not None,
                )
                new_proj.weight.copy_(resample_patch_embed(self.proj.weight, new_patch_size, verbose=True))
                if self.proj.bias is not None:
                    new_proj.bias.copy_(self.proj.bias)
                self.proj = new_proj
            patch_size = new_patch_size
        patch_size = patch_size or self.patch_size

        if img_size != self.img_size or patch_size != self.patch_size:
            (
                self.img_size,
                self.patch_size,
                self.feature_size,
                self.feature_ratio,
                self.feature_dim,
                self.grid_size,
                self.num_patches,
            ) = self._init_backbone(
                img_size=img_size,
                patch_size=patch_size,
                feature_size=feature_size,
                feature_ratio=feature_ratio,
                feature_dim=feature_dim,
            )

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        total_reduction = (
            self.feature_ratio[0] * self.patch_size[0],
            self.feature_ratio[1] * self.patch_size[1]
        )
        if as_scalar:
            return max(total_reduction)
        else:
            return total_reduction

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """ Get feature grid size taking account dynamic padding and backbone network feat reduction
        """
        feat_size = (img_size[0] // self.feature_ratio[0], img_size[1] // self.feature_ratio[1])
        if self.dynamic_img_pad:
            return math.ceil(feat_size[0] / self.patch_size[0]), math.ceil(feat_size[1] / self.patch_size[1])
        else:
            return feat_size[0] // self.patch_size[0], feat_size[1] // self.patch_size[1]

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable=enable)
        elif hasattr(self.backbone, 'grad_checkpointing'):
            self.backbone.grad_checkpointing = enable

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        _, _, H, W = x.shape
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        return x


class HybridEmbedWithSize(HybridEmbed):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(
            self,
            backbone: nn.Module,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 1,
            feature_size: Optional[Union[int, Tuple[int, int]]] = None,
            feature_ratio: Optional[Union[int, Tuple[int, int]]] = None,
            in_chans: int = 3,
            embed_dim: int = 768,
            bias=True,
            proj=True,
    ):
        super().__init__(
            backbone=backbone,
            img_size=img_size,
            patch_size=patch_size,
            feature_size=feature_size,
            feature_ratio=feature_ratio,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=bias,
            proj=proj,
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable=enable)
        elif hasattr(self.backbone, 'grad_checkpointing'):
            self.backbone.grad_checkpointing = enable

    def forward(self, x) -> Tuple[torch.Tensor, List[int]]:
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2), x.shape[-2:]