""" Halo Self Attention

Paper: `Scaling Local Self-Attention for Parameter Efficient Visual Backbones`
    - https://arxiv.org/abs/2103.12731

@misc{2103.12731,
Author = {Ashish Vaswani and Prajit Ramachandran and Aravind Srinivas and Niki Parmar and Blake Hechtman and
    Jonathon Shlens},
Title = {Scaling Local Self-Attention for Parameter Efficient Visual Backbones},
Year = {2021},
}

Status:
This impl is a WIP, there is no official ref impl and some details in paper weren't clear to me.

Trying to match the 'H1' variant in the paper, my parameter counts are 2M less and the model
is extremely slow. Something isn't right. However, the models do appear to train and experimental
variants with attn in C4 and/or C5 stages are tolerable speed.

Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Tuple, List

import torch
from torch import nn
import torch.nn.functional as F


def rel_logits_1d(q, rel_k, permute_mask: List[int]):
    """ Compute relative logits along one dimension

    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925

    Args:
        q: (batch, height, width, dim)
        rel_k: (2 * window - 1, dim)
        permute_mask: permute output dim according to this
    """
    B, H, W, dim = q.shape
    rel_size = rel_k.shape[0]
    win_size = (rel_size + 1) // 2

    x = (q @ rel_k.transpose(-1, -2))
    x = x.reshape(-1, W, rel_size)

    # pad to shift from relative to absolute indexing
    x_pad = F.pad(x, [0, 1]).flatten(1)
    x_pad = F.pad(x_pad, [0, rel_size - W])

    # reshape and slice out the padded elements
    x_pad = x_pad.reshape(-1, W + 1, rel_size)
    x = x_pad[:, :W, win_size - 1:]

    # reshape and tile
    x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    return x.permute(permute_mask)


class PosEmbedRel(nn.Module):
    """ Relative Position Embedding
    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925

    """
    def __init__(self, block_size, win_size, dim_head, scale):
        """
        Args:
            block_size (int): block size
            win_size (int): neighbourhood window size
            dim_head (int): attention head dim
            scale (float): scale factor (for init)
        """
        super().__init__()
        self.block_size = block_size
        self.dim_head = dim_head
        self.scale = scale
        self.height_rel = nn.Parameter(torch.randn(win_size * 2 - 1, dim_head) * self.scale)
        self.width_rel = nn.Parameter(torch.randn(win_size * 2 - 1, dim_head) * self.scale)

    def forward(self, q):
        B, BB, HW, _ = q.shape

        # relative logits in width dimension.
        q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
        rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))

        # relative logits in height dimension.
        q = q.transpose(1, 2)
        rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))

        rel_logits = rel_logits_h + rel_logits_w
        rel_logits = rel_logits.reshape(B, BB, HW, -1)
        return rel_logits


class HaloAttn(nn.Module):
    """ Halo Attention

    Paper: `Scaling Local Self-Attention for Parameter Efficient Visual Backbones`
        - https://arxiv.org/abs/2103.12731
    """
    def __init__(
            self, dim, dim_out=None, stride=1, num_heads=8, dim_head=16, block_size=8, halo_size=3, qkv_bias=False):
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        self.stride = stride
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_qk = num_heads * dim_head
        self.dim_v = dim_out
        self.block_size = block_size
        self.halo_size = halo_size
        self.win_size = block_size + halo_size * 2  # neighbourhood window size
        self.scale = self.dim_head ** -0.5

        # FIXME not clear if this stride behaviour is what the paper intended, not really clear
        # Also, the paper mentions using a 3D conv for dealing with the blocking/gather, and leaving
        # data in unfolded block form. I haven't wrapped my head around how that'd look.
        self.q = nn.Conv2d(dim, self.dim_qk, 1, stride=self.stride, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, self.dim_qk + self.dim_v, 1, bias=qkv_bias)

        self.pos_embed = PosEmbedRel(
            block_size=block_size // self.stride, win_size=self.win_size, dim_head=self.dim_head, scale=self.scale)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.block_size == 0 and W % self.block_size == 0
        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        q = self.q(x)
        q = F.unfold(q, kernel_size=self.block_size // self.stride, stride=self.block_size // self.stride)
        # B, num_heads * dim_head * block_size ** 2, num_blocks
        q = q.reshape(B * self.num_heads, self.dim_head, -1, num_blocks).transpose(1, 3)
        # B * num_heads, num_blocks, block_size ** 2, dim_head

        kv = self.kv(x)
        # FIXME I 'think' this unfold does what I want it to, but I should investigate
        k = F.unfold(kv, kernel_size=self.win_size, stride=self.block_size, padding=self.halo_size)
        k = k.reshape(
            B * self.num_heads, self.dim_head + (self.dim_v // self.num_heads), -1, num_blocks).transpose(1, 3)
        k, v = torch.split(k, [self.dim_head, self.dim_v // self.num_heads], dim=-1)

        attn_logits = (q @ k.transpose(-1, -2)) * self.scale  # FIXME should usual attn scale be applied?
        attn_logits = attn_logits + self.pos_embed(q)  # B * num_heads, block_size ** 2, win_size ** 2

        attn_out = attn_logits.softmax(dim=-1)
        attn_out = (attn_out @ v).transpose(1, 3)  # B * num_heads, dim_v // num_heads, block_size ** 2, num_blocks
        attn_out = F.fold(
            attn_out.reshape(B, -1, num_blocks),
            (H // self.stride, W // self.stride),
            kernel_size=self.block_size // self.stride, stride=self.block_size // self.stride)
        # B, dim_out, H // stride, W // stride
        return attn_out
