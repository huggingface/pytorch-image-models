""" Multi-Scale Vision Transformer v2

@inproceedings{li2021improved,
  title={MViTv2: Improved multiscale vision transformers for classification and detection},
  author={Li, Yanghao and Wu, Chao-Yuan and Fan, Haoqi and Mangalam, Karttikeya and Xiong, Bo and Malik, Jitendra and Feichtenhofer, Christoph},
  booktitle={CVPR},
  year={2022}
}

Code adapted from original Apache 2.0 licensed impl at https://github.com/facebookresearch/mvit
Original copyright below.

Modifications and timm support by / Copyright 2022, Ross Wightman
"""
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.
import operator
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial, reduce
from typing import Union, List, Tuple, Optional

import torch
import torch.utils.checkpoint as checkpoint
from torch import nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import Mlp, DropPath, trunc_normal_tf_, get_norm_layer, to_2tuple
from ._builder import build_model_with_cfg
from ._features_fx import register_notrace_function
from ._registry import register_model, register_model_deprecations, generate_default_cfgs

__all__ = ['MultiScaleVit', 'MultiScaleVitCfg']  # model_registry will add each entrypoint fn to this


@dataclass
class MultiScaleVitCfg:
    depths: Tuple[int, ...] = (2, 3, 16, 3)
    embed_dim: Union[int, Tuple[int, ...]] = 96
    num_heads: Union[int, Tuple[int, ...]] = 1
    mlp_ratio: float = 4.
    pool_first: bool = False
    expand_attn: bool = True
    qkv_bias: bool = True
    use_cls_token: bool = False
    use_abs_pos: bool = False
    residual_pooling: bool = True
    mode: str = 'conv'
    kernel_qkv: Tuple[int, int] = (3, 3)
    stride_q: Optional[Tuple[Tuple[int, int]]] = ((1, 1), (2, 2), (2, 2), (2, 2))
    stride_kv: Optional[Tuple[Tuple[int, int]]] = None
    stride_kv_adaptive: Optional[Tuple[int, int]] = (4, 4)
    patch_kernel: Tuple[int, int] = (7, 7)
    patch_stride: Tuple[int, int] = (4, 4)
    patch_padding: Tuple[int, int] = (3, 3)
    pool_type: str = 'max'
    rel_pos_type: str = 'spatial'
    act_layer: Union[str, Tuple[str, str]] = 'gelu'
    norm_layer: Union[str, Tuple[str, str]] = 'layernorm'
    norm_eps: float = 1e-6

    def __post_init__(self):
        num_stages = len(self.depths)
        if not isinstance(self.embed_dim, (tuple, list)):
            self.embed_dim = tuple(self.embed_dim * 2 ** i for i in range(num_stages))
        assert len(self.embed_dim) == num_stages

        if not isinstance(self.num_heads, (tuple, list)):
            self.num_heads = tuple(self.num_heads * 2 ** i for i in range(num_stages))
        assert len(self.num_heads) == num_stages

        if self.stride_kv_adaptive is not None and self.stride_kv is None:
            _stride_kv = self.stride_kv_adaptive
            pool_kv_stride = []
            for i in range(num_stages):
                if min(self.stride_q[i]) > 1:
                    _stride_kv = [
                        max(_stride_kv[d] // self.stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                pool_kv_stride.append(tuple(_stride_kv))
            self.stride_kv = tuple(pool_kv_stride)


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
            self,
            dim_in=3,
            dim_out=768,
            kernel=(7, 7),
            stride=(4, 4),
            padding=(3, 3),
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x) -> Tuple[torch.Tensor, List[int]]:
        x = self.proj(x)
        # B C H W -> B HW C
        return x.flatten(2).transpose(1, 2), x.shape[-2:]


@register_notrace_function
def reshape_pre_pool(
        x,
        feat_size: List[int],
        has_cls_token: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    H, W = feat_size
    if has_cls_token:
        cls_tok, x = x[:, :, :1, :], x[:, :, 1:, :]
    else:
        cls_tok = None
    x = x.reshape(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2).contiguous()
    return x, cls_tok


@register_notrace_function
def reshape_post_pool(
        x,
        num_heads: int,
        cls_tok: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, List[int]]:
    feat_size = [x.shape[2], x.shape[3]]
    L_pooled = x.shape[2] * x.shape[3]
    x = x.reshape(-1, num_heads, x.shape[1], L_pooled).transpose(2, 3)
    if cls_tok is not None:
        x = torch.cat((cls_tok, x), dim=2)
    return x, feat_size


@register_notrace_function
def cal_rel_pos_type(
        attn: torch.Tensor,
        q: torch.Tensor,
        has_cls_token: bool,
        q_size: List[int],
        k_size: List[int],
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_token else 0
    q_h, q_w = q_size
    k_h, k_w = k_size

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
            torch.arange(q_h, device=q.device).unsqueeze(-1) * q_h_ratio -
            torch.arange(k_h, device=q.device).unsqueeze(0) * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
            torch.arange(q_w, device=q.device).unsqueeze(-1) * q_w_ratio -
            torch.arange(k_w, device=q.device).unsqueeze(0) * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    rel_h = rel_pos_h[dist_h.long()]
    rel_w = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, rel_h)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, rel_w)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h.unsqueeze(-1)
        + rel_w.unsqueeze(-2)
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class MultiScaleAttentionPoolFirst(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            feat_size,
            num_heads=8,
            qkv_bias=True,
            mode="conv",
            kernel_q=(1, 1),
            kernel_kv=(1, 1),
            stride_q=(1, 1),
            stride_kv=(1, 1),
            has_cls_token=True,
            rel_pos_type='spatial',
            residual_pooling=True,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5
        self.has_cls_token = has_cls_token
        padding_q = tuple([int(q // 2) for q in kernel_q])
        padding_kv = tuple([int(kv // 2) for kv in kernel_kv])

        self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if prod(kernel_q) == 1 and prod(stride_q) == 1:
            kernel_q = None
        if prod(kernel_kv) == 1 and prod(stride_kv) == 1:
            kernel_kv = None
        self.mode = mode
        self.unshared = mode == 'conv_unshared'
        self.pool_q, self.pool_k, self.pool_v = None, None, None
        self.norm_q, self.norm_k, self.norm_v = None, None, None
        if mode in ("avg", "max"):
            pool_op = nn.MaxPool2d if mode == "max" else nn.AvgPool2d
            if kernel_q:
                self.pool_q = pool_op(kernel_q, stride_q, padding_q)
            if kernel_kv:
                self.pool_k = pool_op(kernel_kv, stride_kv, padding_kv)
                self.pool_v = pool_op(kernel_kv, stride_kv, padding_kv)
        elif mode == "conv" or mode == "conv_unshared":
            dim_conv = dim // num_heads if mode == "conv" else dim
            if kernel_q:
                self.pool_q = nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                )
                self.norm_q = norm_layer(dim_conv)
            if kernel_kv:
                self.pool_k = nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                self.norm_k = norm_layer(dim_conv)
                self.pool_v = nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                self.norm_v = norm_layer(dim_conv)
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        # relative pos embedding
        self.rel_pos_type = rel_pos_type
        if self.rel_pos_type == 'spatial':
            assert feat_size[0] == feat_size[1]
            size = feat_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            trunc_normal_tf_(self.rel_pos_h, std=0.02)
            trunc_normal_tf_(self.rel_pos_w, std=0.02)

        self.residual_pooling = residual_pooling

    def forward(self, x, feat_size: List[int]):
        B, N, _ = x.shape

        fold_dim = 1 if self.unshared else self.num_heads
        x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
        q = k = v = x

        if self.pool_q is not None:
            q, q_tok = reshape_pre_pool(q, feat_size, self.has_cls_token)
            q = self.pool_q(q)
            q, q_size = reshape_post_pool(q, self.num_heads, q_tok)
        else:
            q_size = feat_size
        if self.norm_q is not None:
            q = self.norm_q(q)

        if self.pool_k is not None:
            k, k_tok = reshape_pre_pool(k, feat_size, self.has_cls_token)
            k = self.pool_k(k)
            k, k_size = reshape_post_pool(k, self.num_heads, k_tok)
        else:
            k_size = feat_size
        if self.norm_k is not None:
            k = self.norm_k(k)

        if self.pool_v is not None:
            v, v_tok = reshape_pre_pool(v, feat_size, self.has_cls_token)
            v = self.pool_v(v)
            v, v_size = reshape_post_pool(v, self.num_heads, v_tok)
        else:
            v_size = feat_size
        if self.norm_v is not None:
            v = self.norm_v(v)

        q_N = q_size[0] * q_size[1] + int(self.has_cls_token)
        q = q.transpose(1, 2).reshape(B, q_N, -1)
        q = self.q(q).reshape(B, q_N, self.num_heads, -1).transpose(1, 2)

        k_N = k_size[0] * k_size[1] + int(self.has_cls_token)
        k = k.transpose(1, 2).reshape(B, k_N, -1)
        k = self.k(k).reshape(B, k_N, self.num_heads, -1)

        v_N = v_size[0] * v_size[1] + int(self.has_cls_token)
        v = v.transpose(1, 2).reshape(B, v_N, -1)
        v = self.v(v).reshape(B, v_N, self.num_heads, -1).transpose(1, 2)

        attn = (q * self.scale) @ k
        if self.rel_pos_type == 'spatial':
            attn = cal_rel_pos_type(
                attn,
                q,
                self.has_cls_token,
                q_size,
                k_size,
                self.rel_pos_h,
                self.rel_pos_w,
            )
        attn = attn.softmax(dim=-1)
        x = attn @ v

        if self.residual_pooling:
            x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        return x, q_size


class MultiScaleAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            feat_size,
            num_heads=8,
            qkv_bias=True,
            mode="conv",
            kernel_q=(1, 1),
            kernel_kv=(1, 1),
            stride_q=(1, 1),
            stride_kv=(1, 1),
            has_cls_token=True,
            rel_pos_type='spatial',
            residual_pooling=True,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5
        self.has_cls_token = has_cls_token
        padding_q = tuple([int(q // 2) for q in kernel_q])
        padding_kv = tuple([int(kv // 2) for kv in kernel_kv])

        self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if prod(kernel_q) == 1 and prod(stride_q) == 1:
            kernel_q = None
        if prod(kernel_kv) == 1 and prod(stride_kv) == 1:
            kernel_kv = None
        self.mode = mode
        self.unshared = mode == 'conv_unshared'
        self.norm_q, self.norm_k, self.norm_v = None, None, None
        self.pool_q, self.pool_k, self.pool_v = None, None, None
        if mode in ("avg", "max"):
            pool_op = nn.MaxPool2d if mode == "max" else nn.AvgPool2d
            if kernel_q:
                self.pool_q = pool_op(kernel_q, stride_q, padding_q)
            if kernel_kv:
                self.pool_k = pool_op(kernel_kv, stride_kv, padding_kv)
                self.pool_v = pool_op(kernel_kv, stride_kv, padding_kv)
        elif mode == "conv" or mode == "conv_unshared":
            dim_conv = dim_out // num_heads if mode == "conv" else dim_out
            if kernel_q:
                self.pool_q = nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                )
                self.norm_q = norm_layer(dim_conv)
            if kernel_kv:
                self.pool_k = nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                self.norm_k = norm_layer(dim_conv)
                self.pool_v = nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                self.norm_v = norm_layer(dim_conv)
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        # relative pos embedding
        self.rel_pos_type = rel_pos_type
        if self.rel_pos_type == 'spatial':
            assert feat_size[0] == feat_size[1]
            size = feat_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            trunc_normal_tf_(self.rel_pos_h, std=0.02)
            trunc_normal_tf_(self.rel_pos_w, std=0.02)

        self.residual_pooling = residual_pooling

    def forward(self, x, feat_size: List[int]):
        B, N, _ = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        if self.pool_q is not None:
            q, q_tok = reshape_pre_pool(q, feat_size, self.has_cls_token)
            q = self.pool_q(q)
            q, q_size = reshape_post_pool(q, self.num_heads, q_tok)
        else:
            q_size = feat_size
        if self.norm_q is not None:
            q = self.norm_q(q)

        if self.pool_k is not None:
            k, k_tok = reshape_pre_pool(k, feat_size, self.has_cls_token)
            k = self.pool_k(k)
            k, k_size = reshape_post_pool(k, self.num_heads, k_tok)
        else:
            k_size = feat_size
        if self.norm_k is not None:
            k = self.norm_k(k)

        if self.pool_v is not None:
            v, v_tok = reshape_pre_pool(v, feat_size, self.has_cls_token)
            v = self.pool_v(v)
            v, _ = reshape_post_pool(v, self.num_heads, v_tok)
        if self.norm_v is not None:
            v = self.norm_v(v)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel_pos_type == 'spatial':
            attn = cal_rel_pos_type(
                attn,
                q,
                self.has_cls_token,
                q_size,
                k_size,
                self.rel_pos_h,
                self.rel_pos_w,
            )
        attn = attn.softmax(dim=-1)
        x = attn @ v

        if self.residual_pooling:
            x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        return x, q_size


class MultiScaleBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            num_heads,
            feat_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            kernel_q=(1, 1),
            kernel_kv=(1, 1),
            stride_q=(1, 1),
            stride_kv=(1, 1),
            mode="conv",
            has_cls_token=True,
            expand_attn=False,
            pool_first=False,
            rel_pos_type='spatial',
            residual_pooling=True,
    ):
        super().__init__()
        proj_needed = dim != dim_out
        self.dim = dim
        self.dim_out = dim_out
        self.has_cls_token = has_cls_token

        self.norm1 = norm_layer(dim)

        self.shortcut_proj_attn = nn.Linear(dim, dim_out) if proj_needed and expand_attn else None
        if stride_q and prod(stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
            stride_skip = stride_q
            padding_skip = [int(skip // 2) for skip in kernel_skip]
            self.shortcut_pool_attn = nn.MaxPool2d(kernel_skip, stride_skip, padding_skip)
        else:
            self.shortcut_pool_attn = None

        att_dim = dim_out if expand_attn else dim
        attn_layer = MultiScaleAttentionPoolFirst if pool_first else MultiScaleAttention
        self.attn = attn_layer(
            dim,
            att_dim,
            num_heads=num_heads,
            feat_size=feat_size,
            qkv_bias=qkv_bias,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            has_cls_token=has_cls_token,
            mode=mode,
            rel_pos_type=rel_pos_type,
            residual_pooling=residual_pooling,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(att_dim)
        mlp_dim_out = dim_out
        self.shortcut_proj_mlp = nn.Linear(dim, dim_out) if proj_needed and not expand_attn else None
        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=int(att_dim * mlp_ratio),
            out_features=mlp_dim_out,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _shortcut_pool(self, x, feat_size: List[int]):
        if self.shortcut_pool_attn is None:
            return x
        if self.has_cls_token:
            cls_tok, x = x[:, :1, :], x[:, 1:, :]
        else:
            cls_tok = None
        B, L, C = x.shape
        H, W = feat_size
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.shortcut_pool_attn(x)
        x = x.reshape(B, C, -1).transpose(1, 2)
        if cls_tok is not None:
            x = torch.cat((cls_tok, x), dim=1)
        return x

    def forward(self, x, feat_size: List[int]):
        x_norm = self.norm1(x)
        # NOTE as per the original impl, this seems odd, but shortcut uses un-normalized input if no proj
        x_shortcut = x if self.shortcut_proj_attn is None else self.shortcut_proj_attn(x_norm)
        x_shortcut = self._shortcut_pool(x_shortcut, feat_size)
        x, feat_size_new = self.attn(x_norm, feat_size)
        x = x_shortcut + self.drop_path1(x)

        x_norm = self.norm2(x)
        x_shortcut = x if self.shortcut_proj_mlp is None else self.shortcut_proj_mlp(x_norm)
        x = x_shortcut + self.drop_path2(self.mlp(x_norm))
        return x, feat_size_new


class MultiScaleVitStage(nn.Module):

    def __init__(
            self,
            dim,
            dim_out,
            depth,
            num_heads,
            feat_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            mode="conv",
            kernel_q=(1, 1),
            kernel_kv=(1, 1),
            stride_q=(1, 1),
            stride_kv=(1, 1),
            has_cls_token=True,
            expand_attn=False,
            pool_first=False,
            rel_pos_type='spatial',
            residual_pooling=True,
            norm_layer=nn.LayerNorm,
            drop_path=0.0,
    ):
        super().__init__()
        self.grad_checkpointing = False

        self.blocks = nn.ModuleList()
        if expand_attn:
            out_dims = (dim_out,) * depth
        else:
            out_dims = (dim,) * (depth - 1) + (dim_out,)

        for i in range(depth):
            attention_block = MultiScaleBlock(
                dim=dim,
                dim_out=out_dims[i],
                num_heads=num_heads,
                feat_size=feat_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                kernel_q=kernel_q,
                kernel_kv=kernel_kv,
                stride_q=stride_q if i == 0 else (1, 1),
                stride_kv=stride_kv,
                mode=mode,
                has_cls_token=has_cls_token,
                pool_first=pool_first,
                rel_pos_type=rel_pos_type,
                residual_pooling=residual_pooling,
                expand_attn=expand_attn,
                norm_layer=norm_layer,
                drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
            )
            dim = out_dims[i]
            self.blocks.append(attention_block)
            if i == 0:
                feat_size = tuple([size // stride for size, stride in zip(feat_size, stride_q)])

        self.feat_size = feat_size

    def forward(self, x, feat_size: List[int]):
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x, feat_size = checkpoint.checkpoint(blk, x, feat_size)
            else:
                x, feat_size = blk(x, feat_size)
        return x, feat_size


class MultiScaleVit(nn.Module):
    """
    Improved Multiscale Vision Transformers for Classification and Detection
    Yanghao Li*, Chao-Yuan Wu*, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2112.01526

    Multiscale Vision Transformers
    Haoqi Fan*, Bo Xiong*, Karttikeya Mangalam*, Yanghao Li*, Zhicheng Yan, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2104.11227
    """

    def __init__(
            self,
            cfg: MultiScaleVitCfg,
            img_size: Tuple[int, int] = (224, 224),
            in_chans: int = 3,
            global_pool: Optional[str] = None,
            num_classes: int = 1000,
            drop_path_rate: float = 0.,
            drop_rate: float = 0.,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        norm_layer = partial(get_norm_layer(cfg.norm_layer), eps=cfg.norm_eps)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        if global_pool is None:
            global_pool = 'token' if cfg.use_cls_token else 'avg'
        self.global_pool = global_pool
        self.depths = tuple(cfg.depths)
        self.expand_attn = cfg.expand_attn

        embed_dim = cfg.embed_dim[0]
        self.patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.patch_kernel,
            stride=cfg.patch_stride,
            padding=cfg.patch_padding,
        )
        patch_dims = (img_size[0] // cfg.patch_stride[0], img_size[1] // cfg.patch_stride[1])
        num_patches = prod(patch_dims)

        if cfg.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_prefix_tokens = 1
            pos_embed_dim = num_patches + 1
        else:
            self.num_prefix_tokens = 0
            self.cls_token = None
            pos_embed_dim = num_patches

        if cfg.use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_dim, embed_dim))
        else:
            self.pos_embed = None

        num_stages = len(cfg.embed_dim)
        feat_size = patch_dims
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            if cfg.expand_attn:
                dim_out = cfg.embed_dim[i]
            else:
                dim_out = cfg.embed_dim[min(i + 1, num_stages - 1)]
            stage = MultiScaleVitStage(
                dim=embed_dim,
                dim_out=dim_out,
                depth=cfg.depths[i],
                num_heads=cfg.num_heads[i],
                feat_size=feat_size,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                mode=cfg.mode,
                pool_first=cfg.pool_first,
                expand_attn=cfg.expand_attn,
                kernel_q=cfg.kernel_qkv,
                kernel_kv=cfg.kernel_qkv,
                stride_q=cfg.stride_q[i],
                stride_kv=cfg.stride_kv[i],
                has_cls_token=cfg.use_cls_token,
                rel_pos_type=cfg.rel_pos_type,
                residual_pooling=cfg.residual_pooling,
                norm_layer=norm_layer,
                drop_path=dpr[i],
            )
            embed_dim = dim_out
            feat_size = stage.feat_size
            self.stages.append(stage)

        self.num_features = embed_dim
        self.norm = norm_layer(embed_dim)
        self.head = nn.Sequential(OrderedDict([
            ('drop', nn.Dropout(self.drop_rate)),
            ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        ]))

        if self.pos_embed is not None:
            trunc_normal_tf_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            trunc_normal_tf_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_tf_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {k for k, _ in self.named_parameters()
                if any(n in k for n in ["pos_embed", "rel_pos_h", "rel_pos_w", "cls_token"])}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^patch_embed',  # stem and embed
            blocks=[(r'^stages\.(\d+)', None), (r'^norm', (99999,))]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Sequential(OrderedDict([
            ('drop', nn.Dropout(self.drop_rate)),
            ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        ]))

    def forward_features(self, x):
        x, feat_size = self.patch_embed(x)
        B, N, C = x.shape

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        for stage in self.stages:
            x, feat_size = stage(x, feat_size)

        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            if self.global_pool == 'avg':
                x = x[:, self.num_prefix_tokens:].mean(1)
            else:
                x = x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    if 'stages.0.blocks.0.norm1.weight' in state_dict:
        return state_dict

    import re
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']

    depths = getattr(model, 'depths', None)
    expand_attn = getattr(model, 'expand_attn', True)
    assert depths is not None, 'model requires depth attribute to remap checkpoints'
    depth_map = {}
    block_idx = 0
    for stage_idx, d in enumerate(depths):
        depth_map.update({i: (stage_idx, i - block_idx) for i in range(block_idx, block_idx + d)})
        block_idx += d

    out_dict = {}
    for k, v in state_dict.items():
        k = re.sub(
            r'blocks\.(\d+)',
            lambda x: f'stages.{depth_map[int(x.group(1))][0]}.blocks.{depth_map[int(x.group(1))][1]}',
            k)

        if expand_attn:
            k = re.sub(r'stages\.(\d+).blocks\.(\d+).proj', f'stages.\\1.blocks.\\2.shortcut_proj_attn', k)
        else:
            k = re.sub(r'stages\.(\d+).blocks\.(\d+).proj', f'stages.\\1.blocks.\\2.shortcut_proj_mlp', k)
        if 'head' in k:
            k = k.replace('head.projection', 'head.fc')
        out_dict[k] = v

    # for k, v in state_dict.items():
    #     if model.pos_embed is not None and k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
    #         # To resize pos embedding when using model at different size from pretrained weights
    #         v = resize_pos_embed(
    #             v,
    #             model.pos_embed,
    #             0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
    #             model.patch_embed.grid_size
    #         )

    return out_dict


model_cfgs = dict(
    mvitv2_tiny=MultiScaleVitCfg(
        depths=(1, 2, 5, 2),
    ),
    mvitv2_small=MultiScaleVitCfg(
        depths=(1, 2, 11, 2),
    ),
    mvitv2_base=MultiScaleVitCfg(
        depths=(2, 3, 16, 3),
    ),
    mvitv2_large=MultiScaleVitCfg(
        depths=(2, 6, 36, 4),
        embed_dim=144,
        num_heads=2,
        expand_attn=False,
    ),

    mvitv2_small_cls=MultiScaleVitCfg(
        depths=(1, 2, 11, 2),
        use_cls_token=True,
    ),
    mvitv2_base_cls=MultiScaleVitCfg(
        depths=(2, 3, 16, 3),
        use_cls_token=True,
    ),
    mvitv2_large_cls=MultiScaleVitCfg(
        depths=(2, 6, 36, 4),
        embed_dim=144,
        num_heads=2,
        use_cls_token=True,
        expand_attn=True,
    ),
    mvitv2_huge_cls=MultiScaleVitCfg(
        depths=(4, 8, 60, 8),
        embed_dim=192,
        num_heads=3,
        use_cls_token=True,
        expand_attn=True,
    ),
)


def _create_mvitv2(variant, cfg_variant=None, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Multiscale Vision Transformer models.')

    return build_model_with_cfg(
        MultiScaleVit,
        variant,
        pretrained,
        model_cfg=model_cfgs[variant] if not cfg_variant else model_cfgs[cfg_variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True),
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head.fc',
        'fixed_input_size': True,
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'mvitv2_tiny.fb_in1k': _cfg(url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_T_in1k.pyth'),
    'mvitv2_small.fb_in1k': _cfg(url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_S_in1k.pyth'),
    'mvitv2_base.fb_in1k': _cfg(url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_B_in1k.pyth'),
    'mvitv2_large.fb_in1k': _cfg(url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_L_in1k.pyth'),

    'mvitv2_small_cls': _cfg(url=''),
    'mvitv2_base_cls.fb_inw21k': _cfg(
        url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_B_in21k.pyth',
        num_classes=19168),
    'mvitv2_large_cls.fb_inw21k': _cfg(
        url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_L_in21k.pyth',
        num_classes=19168),
    'mvitv2_huge_cls.fb_inw21k': _cfg(
        url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_H_in21k.pyth',
        num_classes=19168),
})


@register_model
def mvitv2_tiny(pretrained=False, **kwargs) -> MultiScaleVit:
    return _create_mvitv2('mvitv2_tiny', pretrained=pretrained, **kwargs)


@register_model
def mvitv2_small(pretrained=False, **kwargs) -> MultiScaleVit:
    return _create_mvitv2('mvitv2_small', pretrained=pretrained, **kwargs)


@register_model
def mvitv2_base(pretrained=False, **kwargs) -> MultiScaleVit:
    return _create_mvitv2('mvitv2_base', pretrained=pretrained, **kwargs)


@register_model
def mvitv2_large(pretrained=False, **kwargs) -> MultiScaleVit:
    return _create_mvitv2('mvitv2_large', pretrained=pretrained, **kwargs)


@register_model
def mvitv2_small_cls(pretrained=False, **kwargs) -> MultiScaleVit:
    return _create_mvitv2('mvitv2_small_cls', pretrained=pretrained, **kwargs)


@register_model
def mvitv2_base_cls(pretrained=False, **kwargs) -> MultiScaleVit:
    return _create_mvitv2('mvitv2_base_cls', pretrained=pretrained, **kwargs)


@register_model
def mvitv2_large_cls(pretrained=False, **kwargs) -> MultiScaleVit:
    return _create_mvitv2('mvitv2_large_cls', pretrained=pretrained, **kwargs)


@register_model
def mvitv2_huge_cls(pretrained=False, **kwargs) -> MultiScaleVit:
    return _create_mvitv2('mvitv2_huge_cls', pretrained=pretrained, **kwargs)
