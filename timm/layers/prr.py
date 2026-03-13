""" Patch Representation Refinement (PRR)

Parameter-free multi-head self-attention applied before the classification head to ensure diverse gradient flow across
patch positions, improving representations for dense prediction tasks.

Paper: 'Locality-Attending Vision Transformer' - https://arxiv.org/abs/2603.04892

Reference impl: https://github.com/sinahmr/LocAtViT
"""
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from .config import use_fused_attn


class PRR(nn.Module):
    """ Patch Representation Refinement (PRR) module

    Parameter-free multi-head self-attention that refines patch representations before the classification head.
    In standard ViT, only the [CLS] token receives direct supervision from the classification loss, leaving patch
    representations at the final layer under-optimized for dense prediction. PRR addresses this issue by aggregating
    information from all positions non-uniformly, ensuring better representations at spatial positions. This
    module's output at the [CLS] position can then be passed to the head (other pooling methods can be used as well).

    Supports both fused (scaled_dot_product_attention) and manual implementations.
    """
    fused_attn: Final[bool]

    def __init__(
            self, dim: int, num_heads: int, nchw: bool = False,
            pre_norm: bool = False, post_norm: bool = False, norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        """
        Initialize the PRR module.

        Args:
            dim: Input dimension of the token embeddings.
            num_heads: Number of attention heads.
            nchw: Whether the input's shape is NCHW. NLC will be assumed otherwise.
            pre_norm: Whether to apply normalization before PRR module.
            post_norm: Whether to apply normalization after PRR module.
            norm_layer: Normalization layer constructor for pre_norm and post_norm if enabled.
        """
        super().__init__()
        self.fused_attn = use_fused_attn()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.nchw = nchw
        self.pre_norm = norm_layer(dim) if pre_norm else nn.Identity()
        self.post_norm = norm_layer(dim) if post_norm else nn.Identity()

    def forward(self, x: torch.Tensor):
        shape = x.shape
        if self.nchw:
            x = x.movedim(1, -1)
        x = self.pre_norm(x)
        x = x.flatten(1, -2)
        x = x.view(x.shape[0], x.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(x, x, x)
        else:
            attn = (x * self.scale) @ x.transpose(-2, -1)
            attn = torch.softmax(attn, dim=-1)
            x = attn @ x
        x = x.permute(0, 2, 1, 3).flatten(2, 3)
        x = self.post_norm(x)
        if self.nchw:
            x = x.movedim(-1, 1)
        return x.reshape(shape)
