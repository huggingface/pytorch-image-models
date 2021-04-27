""" Bottleneck Self Attention (Bottleneck Transformers)

Paper: `Bottleneck Transformers for Visual Recognition` - https://arxiv.org/abs/2101.11605

@misc{2101.11605,
Author = {Aravind Srinivas and Tsung-Yi Lin and Niki Parmar and Jonathon Shlens and Pieter Abbeel and Ashish Vaswani},
Title = {Bottleneck Transformers for Visual Recognition},
Year = {2021},
}

Based on ref gist at: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2

This impl is a WIP but given that it is based on the ref gist likely not too far off.

Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import to_2tuple


def rel_logits_1d(q, rel_k, permute_mask: List[int]):
    """ Compute relative logits along one dimension

    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925

    Args:
        q: (batch, heads, height, width, dim)
        rel_k: (2 * width - 1, dim)
        permute_mask: permute output dim according to this
    """
    B, H, W, dim = q.shape
    x = (q @ rel_k.transpose(-1, -2))
    x = x.reshape(-1, W, 2 * W -1)

    # pad to shift from relative to absolute indexing
    x_pad = F.pad(x, [0, 1]).flatten(1)
    x_pad = F.pad(x_pad, [0, W - 1])

    # reshape and slice out the padded elements
    x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    x = x_pad[:, :W, W - 1:]

    # reshape and tile
    x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    return x.permute(permute_mask)


class PosEmbedRel(nn.Module):
    """ Relative Position Embedding
    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925
    """
    def __init__(self, feat_size, dim_head, scale):
        super().__init__()
        self.height, self.width = to_2tuple(feat_size)
        self.dim_head = dim_head
        self.scale = scale
        self.height_rel = nn.Parameter(torch.randn(self.height * 2 - 1, dim_head) * self.scale)
        self.width_rel = nn.Parameter(torch.randn(self.width * 2 - 1, dim_head) * self.scale)

    def forward(self, q):
        B, num_heads, HW, _ = q.shape

        # relative logits in width dimension.
        q = q.reshape(B * num_heads, self.height, self.width, -1)
        rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))

        # relative logits in height dimension.
        q = q.transpose(1, 2)
        rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))

        rel_logits = rel_logits_h + rel_logits_w
        rel_logits = rel_logits.reshape(B, num_heads, HW, HW)
        return rel_logits


class BottleneckAttn(nn.Module):
    """ Bottleneck Attention
    Paper: `Bottleneck Transformers for Visual Recognition` - https://arxiv.org/abs/2101.11605
    """
    def __init__(self, dim, dim_out=None, feat_size=None, stride=1, num_heads=4, qkv_bias=False):
        super().__init__()
        assert feat_size is not None, 'A concrete feature size matching expected input (H, W) is required'
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.dim_head = dim_out // num_heads
        self.scale = self.dim_head ** -0.5

        self.qkv = nn.Conv2d(dim, self.dim_out * 3, 1, bias=qkv_bias)

        # NOTE I'm only supporting relative pos embedding for now
        self.pos_embed = PosEmbedRel(feat_size, dim_head=self.dim_head, scale=self.scale)

        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.pos_embed.height and W == self.pos_embed.width

        x = self.qkv(x)  # B, 3 * num_heads * dim_head, H, W
        x = x.reshape(B, -1, self.dim_head, H * W).transpose(-1, -2)
        q, k, v = torch.split(x, self.num_heads, dim=1)

        attn_logits = (q @ k.transpose(-1, -2)) * self.scale
        attn_logits = attn_logits + self.pos_embed(q)  # B, num_heads, H * W, H * W

        attn_out = attn_logits.softmax(dim = -1)
        attn_out = (attn_out @ v).transpose(1, 2).reshape(B, self.dim_out, H, W) # B, dim_out, H, W
        attn_out = self.pool(attn_out)
        return attn_out


