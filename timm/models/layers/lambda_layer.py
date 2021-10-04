""" Lambda Layer

Paper: `LambdaNetworks: Modeling Long-Range Interactions Without Attention`
    - https://arxiv.org/abs/2102.08602

@misc{2102.08602,
Author = {Irwan Bello},
Title = {LambdaNetworks: Modeling Long-Range Interactions Without Attention},
Year = {2021},
}

Status:
This impl is a WIP. Code snippets in the paper were used as reference but
good chance some details are missing/wrong.

I've only implemented local lambda conv based pos embeddings.

For a PyTorch impl that includes other embedding options checkout
https://github.com/lucidrains/lambda-networks

Hacked together by / Copyright 2021 Ross Wightman
"""
import torch
from torch import nn
import torch.nn.functional as F

from .helpers import to_2tuple
from .weight_init import trunc_normal_


def rel_pos_indices(size):
    size = to_2tuple(size)
    pos = torch.stack(torch.meshgrid(torch.arange(size[0]), torch.arange(size[1]))).flatten(1)
    rel_pos = pos[:, None, :] - pos[:, :, None]
    rel_pos[0] += size[0] - 1
    rel_pos[1] += size[1] - 1
    return rel_pos  # 2, H * W, H * W


class LambdaLayer(nn.Module):
    """Lambda Layer

    Paper: `LambdaNetworks: Modeling Long-Range Interactions Without Attention`
        - https://arxiv.org/abs/2102.08602

    NOTE: intra-depth parameter 'u' is fixed at 1. It did not appear worth the complexity to add.
    """
    def __init__(
            self,
            dim, dim_out=None, feat_size=None, stride=1, num_heads=4, dim_head=16, r=7, qkv_bias=False):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.dim_k = dim_head  # query depth 'k'
        self.num_heads = num_heads
        assert self.dim_out % num_heads == 0, ' should be divided by num_heads'
        self.dim_v = self.dim_out // num_heads  # value depth 'v'

        self.qkv = nn.Conv2d(
            dim,
            num_heads * dim_head + dim_head + self.dim_v,
            kernel_size=1, bias=qkv_bias)
        self.norm_q = nn.BatchNorm2d(num_heads * dim_head)
        self.norm_v = nn.BatchNorm2d(self.dim_v)

        if r is not None:
            # local lambda convolution for pos
            self.conv_lambda = nn.Conv3d(1, dim_head, (r, r, 1), padding=(r // 2, r // 2, 0))
            self.pos_emb = None
            self.rel_pos_indices = None
        else:
            # relative pos embedding
            assert feat_size is not None
            feat_size = to_2tuple(feat_size)
            rel_size = [2 * s - 1 for s in feat_size]
            self.conv_lambda = None
            self.pos_emb = nn.Parameter(torch.zeros(rel_size[0], rel_size[1], self.dim_k))
            self.register_buffer('rel_pos_indices', rel_pos_indices(feat_size), persistent=False)

        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.qkv.weight, std=self.dim ** -0.5)
        if self.conv_lambda is not None:
            trunc_normal_(self.conv_lambda.weight, std=self.dim_k ** -0.5)
        if self.pos_emb is not None:
            trunc_normal_(self.pos_emb, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        M = H * W
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [
            self.num_heads * self.dim_k, self.dim_k, self.dim_v], dim=1)
        q = self.norm_q(q).reshape(B, self.num_heads, self.dim_k, M).transpose(-1, -2)  # B, num_heads, M, K
        v = self.norm_v(v).reshape(B, self.dim_v, M).transpose(-1, -2)  # B, M, V
        k = F.softmax(k.reshape(B, self.dim_k, M), dim=-1)  # B, K, M

        content_lam = k @ v  # B, K, V
        content_out = q @ content_lam.unsqueeze(1)  # B, num_heads, M, V

        if self.pos_emb is None:
            position_lam = self.conv_lambda(v.reshape(B, 1, H, W, self.dim_v))  # B, H, W, V, K
            position_lam = position_lam.reshape(B, 1, self.dim_k, H * W, self.dim_v).transpose(2, 3)  # B, 1, M, K, V
        else:
            # FIXME relative pos embedding path not fully verified
            pos_emb = self.pos_emb[self.rel_pos_indices[0], self.rel_pos_indices[1]].expand(B, -1, -1, -1)
            position_lam = (pos_emb.transpose(-1, -2) @ v.unsqueeze(1)).unsqueeze(1)  # B, 1, M, K, V
        position_out = (q.unsqueeze(-2) @ position_lam).squeeze(-2)  # B, num_heads, M, V

        out = (content_out + position_out).transpose(-1, -2).reshape(B, C, H, W)  # B, C (num_heads * V), H, W
        out = self.pool(out)
        return out
