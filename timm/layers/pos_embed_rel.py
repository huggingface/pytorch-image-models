""" Relative position embedding modules and functions

Hacked together by / Copyright 2022 Ross Wightman
"""
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grid import ndgrid
from .interpolate import RegularGridInterpolator
from .mlp import Mlp
from .weight_init import trunc_normal_

_USE_SCIPY = int(os.environ.get('TIMM_USE_SCIPY_INTERP', 0)) > 0


def gen_relative_position_index(
        q_size: Tuple[int, int],
        k_size: Optional[Tuple[int, int]] = None,
        class_token: bool = False,
) -> torch.Tensor:
    # Adapted with significant modifications from Swin / BeiT codebases
    # get pair-wise relative position index for each token inside the window
    assert k_size is None, 'Different q & k sizes not currently supported'  # FIXME

    coords = torch.stack(ndgrid(torch.arange(q_size[0]), torch.arange(q_size[1]))).flatten(1)  # 2, Wh, Ww
    relative_coords = coords[:, :, None] - coords[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0)  # Qh*Qw, Kh*Kw, 2
    relative_coords[:, :, 0] += q_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += q_size[1] - 1
    relative_coords[:, :, 0] *= 2 * q_size[1] - 1
    num_relative_distance = (2 * q_size[0] - 1) * (2 * q_size[1] - 1)

    # else:
    #     # FIXME different q vs k sizes is a WIP, need to better offset the two grids?
    #     q_coords = torch.stack(
    #         ndgrid(
    #             torch.arange(q_size[0]),
    #             torch.arange(q_size[1])
    #         )
    #     ).flatten(1)  # 2, Wh, Ww
    #     k_coords = torch.stack(
    #         ndgrid(
    #             torch.arange(k_size[0]),
    #             torch.arange(k_size[1])
    #         )
    #     ).flatten(1)
    #     relative_coords = q_coords[:, :, None] - k_coords[:, None, :]  # 2, Wh*Ww, Wh*Ww
    #     relative_coords = relative_coords.permute(1, 2, 0)  # Qh*Qw, Kh*Kw, 2
    #     relative_coords[:, :, 0] += max(q_size[0], k_size[0]) - 1  # shift to start from 0
    #     relative_coords[:, :, 1] += max(q_size[1], k_size[1]) - 1
    #     relative_coords[:, :, 0] *= k_size[1] + q_size[1] - 1
    #     relative_position_index = relative_coords.sum(-1)  # Qh*Qw, Kh*Kw
    #     num_relative_distance = (q_size[0] + k_size[0] - 1) * (q_size[1] + k_size[1] - 1) + 3

    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

    if class_token:
        # handle cls to token & token 2 cls & cls to cls as per beit for rel pos bias
        # NOTE not intended or tested with MLP log-coords
        relative_position_index = F.pad(relative_position_index, [1, 0, 1, 0])
        relative_position_index[0, 0:] = num_relative_distance
        relative_position_index[0:, 0] = num_relative_distance + 1
        relative_position_index[0, 0] = num_relative_distance + 2

    return relative_position_index.contiguous()


def resize_rel_pos_bias_table_simple(
        rel_pos_bias,
        new_window_size: Tuple[int, int],
        new_bias_shape: Tuple[int, ...],
):
    dst_size = (new_window_size[0] * 2 - 1, new_window_size[1] * 2 - 1)
    if rel_pos_bias.ndim == 3:
        # TF maxvit style (num_heads, H, W) bias shape, no extra tokens currently supported
        _, dst_h, dst_w = new_bias_shape
        num_attn_heads, src_h, src_w = rel_pos_bias.shape
        assert dst_h == dst_size[0] and dst_w == dst_size[1]
        if src_h != dst_h or src_w != dst_w:
            rel_pos_bias = torch.nn.functional.interpolate(
                rel_pos_bias.unsqueeze(0),
                size=dst_size,
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
    else:
        assert rel_pos_bias.ndim == 2
        # (num_pos, num_heads) (aka flat) bias shape
        dst_num_pos, _ = new_bias_shape
        src_num_pos, num_attn_heads = rel_pos_bias.shape
        num_extra_tokens = dst_num_pos - (dst_size[0] * dst_size[1])
        src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
        src_size = (src_size, src_size)  # FIXME could support non-equal src if argument passed

        if src_size[0] != dst_size[0] or src_size[1] != dst_size[1]:
            if num_extra_tokens:
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]
            else:
                extra_tokens = None

            rel_pos_bias = torch.nn.functional.interpolate(
                rel_pos_bias.transpose(1, 0).reshape((1, -1, src_size[0], src_size[1])),
                size=dst_size,
                mode="bicubic",
                align_corners=False,
            ).view(-1, dst_num_pos - num_extra_tokens).transpose(0, 1)

            if extra_tokens is not None:
                rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)

    return rel_pos_bias


def resize_rel_pos_bias_table_levit(
        position_bias_table,
        new_size,
        interpolation: str = 'bicubic',
        antialias: bool = True,
):
    """
    Resample relative position bias table suggested in LeVit
    Adapted from: https://github.com/microsoft/Cream/blob/main/TinyViT/utils.py
    """
    L1, nH1 = position_bias_table.size()
    L2, nH2 = new_size
    assert nH1 == nH2
    if L1 != L2:
        orig_dtype = position_bias_table.dtype
        position_bias_table = position_bias_table.float()
        # bicubic interpolate relative_position_bias_table if not match
        S1 = int(L1 ** 0.5)
        S2 = int(L2 ** 0.5)
        relative_position_bias_table_resized = F.interpolate(
            position_bias_table.permute(1, 0).view(1, nH1, S1, S1),
            size=(S2, S2),
            mode=interpolation,
            antialias=antialias)
        relative_position_bias_table_resized = \
            relative_position_bias_table_resized.view(nH2, L2).permute(1, 0)
        relative_position_bias_table_resized.to(orig_dtype)
        return relative_position_bias_table_resized
    else:
        return position_bias_table


def resize_rel_pos_bias_table(
        rel_pos_bias,
        new_window_size: Tuple[int, int],
        new_bias_shape: Tuple[int, ...],
):
    """ Resize relative position bias table using more advanced interpolation.

    Modified from code in Microsoft Unilm (https://github.com/microsoft/unilm) repo (BeiT, BeiT-v2, etc).

    https://github.com/microsoft/unilm/blob/5255d52de86dad642810f5849dd357769346c1d7/beit/run_class_finetuning.py#L351

    Args:
        rel_pos_bias:
        new_window_size:
        new_bias_shape:

    Returns:

    """
    if _USE_SCIPY:
        from scipy import interpolate

    dst_size = (new_window_size[0] * 2 - 1, new_window_size[1] * 2 - 1)
    if rel_pos_bias.ndim == 3:
        # TF maxvit style (num_heads, H, W) bias shape, no extra tokens currently supported
        num_extra_tokens = 0
        _, dst_h, dst_w = new_bias_shape
        assert dst_h == dst_size[0] and dst_w == dst_size[1]
        num_attn_heads, src_h, src_w = rel_pos_bias.shape
        src_size = (src_h, src_w)
        has_flat_shape = False
    else:
        assert rel_pos_bias.ndim == 2
        # (num_pos, num_heads) (aka flat) bias shape
        dst_num_pos, _ = new_bias_shape
        src_num_pos, num_attn_heads = rel_pos_bias.shape
        num_extra_tokens = dst_num_pos - (dst_size[0] * dst_size[1])
        src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
        src_size = (src_size, src_size)
        has_flat_shape = True

    if src_size[0] != dst_size[0] or src_size[1] != dst_size[1]:
        # print("Interpolating position from %dx%d to %dx%d" % (src_size[0], src_size[1], dst_size[0], dst_size[1]))
        if num_extra_tokens:
            extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
            rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]
        else:
            extra_tokens = None

        def geometric_progression(a, r, n):
            return a * (1.0 - r ** n) / (1.0 - r)

        def _calc(src, dst):
            left, right = 1.01, 1.5
            while right - left > 1e-6:
                q = (left + right) / 2.0
                gp = geometric_progression(1, q, src // 2)
                if gp > dst // 2:
                    right = q
                else:
                    left = q

            dis = []
            cur = 1
            for i in range(src // 2):
                dis.append(cur)
                cur += q ** (i + 1)
            r_ids = [-_ for _ in reversed(dis)]
            return r_ids + [0] + dis

        y = _calc(src_size[0], dst_size[0])
        x = _calc(src_size[1], dst_size[1])
        yx = [torch.tensor(y), torch.tensor(x)]
        # print("Original positions = %s" % str(x))

        ty = dst_size[0] // 2.0
        tx = dst_size[1] // 2.0
        dy = torch.arange(-ty, ty + 0.1, 1.0)
        dx = torch.arange(-tx, tx + 0.1, 1.0)
        dyx = ndgrid(dy, dx)
        # print("Target positions = %s" % str(dx))

        all_rel_pos_bias = []
        for i in range(num_attn_heads):
            if has_flat_shape:
                z = rel_pos_bias[:, i].view(src_size[0], src_size[1]).float()
            else:
                z = rel_pos_bias[i, :, :].float()

            if _USE_SCIPY:
                # Original beit code uses scipy w/ cubic interpolation
                f = interpolate.interp2d(x, y, z.numpy(), kind='cubic')
                r = torch.Tensor(f(dx, dy)).contiguous().to(rel_pos_bias.device)
            else:
                # Without scipy dependency, I've found a reasonably simple impl
                # that supports uneven spaced interpolation pts with 'linear' interp.
                # Results are comparable to scipy for model accuracy in most cases.
                f = RegularGridInterpolator(yx, z)
                r = f(dyx).contiguous().to(rel_pos_bias.device)

            if has_flat_shape:
                r = r.view(-1, 1)
            all_rel_pos_bias.append(r)

        if has_flat_shape:
            rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
        else:
            rel_pos_bias = torch.cat(all_rel_pos_bias, dim=0)

        if extra_tokens is not None:
            assert has_flat_shape
            rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)

    return rel_pos_bias


class RelPosBias(nn.Module):
    """ Relative Position Bias
    Adapted from Swin-V1 relative position bias impl, modularized.
    """

    def __init__(self, window_size, num_heads, prefix_tokens=0):
        super().__init__()
        assert prefix_tokens <= 1
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.bias_shape = (self.window_area + prefix_tokens,) * 2 + (num_heads,)

        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3 * prefix_tokens
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_distance, num_heads))
        self.register_buffer(
            "relative_position_index",
            gen_relative_position_index(self.window_size, class_token=prefix_tokens > 0).view(-1),
            persistent=False,
        )

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def get_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]
        # win_h * win_w, win_h * win_w, num_heads
        relative_position_bias = relative_position_bias.view(self.bias_shape).permute(2, 0, 1)
        return relative_position_bias.unsqueeze(0).contiguous()

    def forward(self, attn, shared_rel_pos: Optional[torch.Tensor] = None):
        return attn + self.get_bias()


def gen_relative_log_coords(
        win_size: Tuple[int, int],
        pretrained_win_size: Tuple[int, int] = (0, 0),
        mode='swin',
):
    assert mode in ('swin', 'cr')
    # as per official swin-v2 impl, supporting timm specific 'cr' log coords as well
    relative_coords_h = torch.arange(-(win_size[0] - 1), win_size[0]).to(torch.float32)
    relative_coords_w = torch.arange(-(win_size[1] - 1), win_size[1]).to(torch.float32)
    relative_coords_table = torch.stack(ndgrid(relative_coords_h, relative_coords_w))
    relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous()  # 2*Wh-1, 2*Ww-1, 2
    if mode == 'swin':
        if pretrained_win_size[0] > 0:
            relative_coords_table[:, :, 0] /= (pretrained_win_size[0] - 1)
            relative_coords_table[:, :, 1] /= (pretrained_win_size[1] - 1)
        else:
            relative_coords_table[:, :, 0] /= (win_size[0] - 1)
            relative_coords_table[:, :, 1] /= (win_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            1.0 + relative_coords_table.abs()) / math.log2(8)
    else:
        # mode == 'cr'
        relative_coords_table = torch.sign(relative_coords_table) * torch.log(
            1.0 + relative_coords_table.abs())

    return relative_coords_table


class RelPosMlp(nn.Module):
    """ Log-Coordinate Relative Position MLP
    Based on ideas presented in Swin-V2 paper (https://arxiv.org/abs/2111.09883)

    This impl covers the 'swin' implementation as well as two timm specific modes ('cr', and 'rw')
    """
    def __init__(
            self,
            window_size,
            num_heads=8,
            hidden_dim=128,
            prefix_tokens=0,
            mode='cr',
            pretrained_window_size=(0, 0)
    ):
        super().__init__()
        self.window_size = window_size
        self.window_area = self.window_size[0] * self.window_size[1]
        self.prefix_tokens = prefix_tokens
        self.num_heads = num_heads
        self.bias_shape = (self.window_area,) * 2 + (num_heads,)
        if mode == 'swin':
            self.bias_act = nn.Sigmoid()
            self.bias_gain = 16
            mlp_bias = (True, False)
        else:
            self.bias_act = nn.Identity()
            self.bias_gain = None
            mlp_bias = True

        self.mlp = Mlp(
            2,  # x, y
            hidden_features=hidden_dim,
            out_features=num_heads,
            act_layer=nn.ReLU,
            bias=mlp_bias,
            drop=(0.125, 0.)
        )

        self.register_buffer(
            "relative_position_index",
            gen_relative_position_index(window_size).view(-1),
            persistent=False)

        # get relative_coords_table
        self.register_buffer(
            "rel_coords_log",
            gen_relative_log_coords(window_size, pretrained_window_size, mode=mode),
            persistent=False)

    def get_bias(self) -> torch.Tensor:
        relative_position_bias = self.mlp(self.rel_coords_log)
        if self.relative_position_index is not None:
            relative_position_bias = relative_position_bias.view(-1, self.num_heads)[self.relative_position_index]
            relative_position_bias = relative_position_bias.view(self.bias_shape)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        relative_position_bias = self.bias_act(relative_position_bias)
        if self.bias_gain is not None:
            relative_position_bias = self.bias_gain * relative_position_bias
        if self.prefix_tokens:
            relative_position_bias = F.pad(relative_position_bias, [self.prefix_tokens, 0, self.prefix_tokens, 0])
        return relative_position_bias.unsqueeze(0).contiguous()

    def forward(self, attn, shared_rel_pos: Optional[torch.Tensor] = None):
        return attn + self.get_bias()


def generate_lookup_tensor(
        length: int,
        max_relative_position: Optional[int] = None,
):
    """Generate a one_hot lookup tensor to reindex embeddings along one dimension.

    Args:
        length: the length to reindex to.
        max_relative_position: the maximum relative position to consider.
            Relative position embeddings for distances above this threshold
            are zeroed out.
    Returns:
        a lookup Tensor of size [length, length, vocab_size] that satisfies
            ret[n,m,v] = 1{m - n + max_relative_position = v}.
    """
    if max_relative_position is None:
        max_relative_position = length - 1
    # Return the cached lookup tensor, otherwise compute it and cache it.
    vocab_size = 2 * max_relative_position + 1
    ret = torch.zeros(length, length, vocab_size)
    for i in range(length):
        for x in range(length):
            v = x - i + max_relative_position
            if abs(x - i) > max_relative_position:
                continue
            ret[i, x, v] = 1
    return ret


def reindex_2d_einsum_lookup(
        relative_position_tensor,
        height: int,
        width: int,
        height_lookup: torch.Tensor,
        width_lookup: torch.Tensor,
) -> torch.Tensor:
    """Reindex 2d relative position bias with 2 independent einsum lookups.

    Adapted from:
     https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/attention_utils.py

    Args:
        relative_position_tensor: tensor of shape
            [..., vocab_height, vocab_width, ...].
        height: height to reindex to.
        width: width to reindex to.
        height_lookup: one-hot height lookup
        width_lookup: one-hot width lookup
    Returns:
        reindexed_tensor: a Tensor of shape
            [..., height * width, height * width, ...]
    """
    reindexed_tensor = torch.einsum('nhw,ixh->nixw', relative_position_tensor, height_lookup)
    reindexed_tensor = torch.einsum('nixw,jyw->nijxy', reindexed_tensor, width_lookup)
    area = height * width
    return reindexed_tensor.reshape(relative_position_tensor.shape[0], area, area)


class RelPosBiasTf(nn.Module):
    """ Relative Position Bias Impl (Compatible with Tensorflow MaxViT models)
    Adapted from:
     https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/attention_utils.py
    """
    def __init__(self, window_size, num_heads, prefix_tokens=0):
        super().__init__()
        assert prefix_tokens <= 1
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.num_heads = num_heads

        vocab_height = 2 * window_size[0] - 1
        vocab_width = 2 * window_size[1] - 1
        self.bias_shape = (self.num_heads, vocab_height, vocab_width)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.bias_shape))
        self.register_buffer('height_lookup', generate_lookup_tensor(window_size[0]), persistent=False)
        self.register_buffer('width_lookup', generate_lookup_tensor(window_size[1]), persistent=False)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.relative_position_bias_table, std=.02)

    def get_bias(self) -> torch.Tensor:
        # FIXME change to not use one-hot/einsum?
        return reindex_2d_einsum_lookup(
            self.relative_position_bias_table,
            self.window_size[0],
            self.window_size[1],
            self.height_lookup,
            self.width_lookup
        )

    def forward(self, attn, shared_rel_pos: Optional[torch.Tensor] = None):
        return attn + self.get_bias()
