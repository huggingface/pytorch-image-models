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

from .weight_init import trunc_normal_


class LambdaLayer(nn.Module):
    """Lambda Layer w/ lambda conv position embedding

    Paper: `LambdaNetworks: Modeling Long-Range Interactions Without Attention`
        - https://arxiv.org/abs/2102.08602
    """
    def __init__(
            self,
            dim, dim_out=None, stride=1, num_heads=4, dim_head=16, r=7, qkv_bias=False):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.dim_k = dim_head  # query depth 'k'
        self.num_heads = num_heads
        assert self.dim_out % num_heads == 0, ' should be divided by num_heads'
        self.dim_v = self.dim_out // num_heads  # value depth 'v'
        self.r = r  # relative position neighbourhood (lambda conv kernel size)

        self.qkv = nn.Conv2d(
            dim,
            num_heads * dim_head + dim_head + self.dim_v,
            kernel_size=1, bias=qkv_bias)
        self.norm_q = nn.BatchNorm2d(num_heads * dim_head)
        self.norm_v = nn.BatchNorm2d(self.dim_v)

        # NOTE currently only supporting the local lambda convolutions for positional
        self.conv_lambda = nn.Conv3d(1, dim_head, (r, r, 1), padding=(r // 2, r // 2, 0))

        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()

    def reset_parameters(self):
        trunc_normal_(self.qkv.weight, std=self.dim ** -0.5)
        trunc_normal_(self.conv_lambda.weight, std=self.dim_k ** -0.5)

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

        position_lam = self.conv_lambda(v.reshape(B, 1, H, W, self.dim_v))  # B, H, W, V, K
        position_lam = position_lam.reshape(B, 1, self.dim_k, H * W, self.dim_v).transpose(2, 3)  # B, 1, M, K, V
        position_out = (q.unsqueeze(-2) @ position_lam).squeeze(-2)  # B, num_heads, M, V

        out = (content_out + position_out).transpose(3, 1).reshape(B, C, H, W)  # B, C (num_heads * V), H, W
        out = self.pool(out)
        return out
