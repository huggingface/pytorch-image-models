""" Shifted Window Attn

This is a WIP experiment to apply windowed attention from the Swin Transformer
to a stand-alone module for use as an attn block in conv nets.

Based on original swin window code at https://github.com/microsoft/Swin-Transformer
Swin Transformer paper: https://arxiv.org/pdf/2103.14030.pdf
"""
from typing import Optional

import torch
import torch.nn as nn

from .drop import DropPath
from .helpers import to_2tuple
from .weight_init import trunc_normal_


def window_partition(x, win_size: int):
    """
    Args:
        x: (B, H, W, C)
        win_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)
    return windows


def window_reverse(windows, win_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        win_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        win_size (int): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    """

    def __init__(
            self, dim, dim_out=None, feat_size=None, stride=1, win_size=8, shift_size=None, num_heads=8,
            qkv_bias=True, attn_drop=0.):

        super().__init__()
        self.dim_out = dim_out or dim
        self.feat_size = to_2tuple(feat_size)
        self.win_size = win_size
        self.shift_size = shift_size or win_size // 2
        if min(self.feat_size) <= win_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.win_size = min(self.feat_size)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-window_size"
        self.num_heads = num_heads
        head_dim = self.dim_out // num_heads
        self.scale = head_dim ** -0.5

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.feat_size
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.win_size),
                slice(-self.win_size, -self.shift_size),
                slice(-self.shift_size, None))
            w_slices = (
                slice(0, -self.win_size),
                slice(-self.win_size, -self.shift_size),
                slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.win_size)  # num_win, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.win_size * self.win_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            # 2 * Wh - 1 * 2 * Ww - 1, nH
            torch.zeros((2 * self.win_size - 1) * (2 * self.win_size - 1), num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size)
        coords_w = torch.arange(self.win_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size - 1
        relative_coords[:, :, 0] *= 2 * self.win_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, self.dim_out * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()

    def reset_parameters(self):
        trunc_normal_(self.qkv.weight, std=self.qkv.weight.shape[1] ** -0.5)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        win_size_sq = self.win_size * self.win_size
        x_windows = window_partition(shifted_x, self.win_size)  # num_win * B, window_size, window_size, C
        x_windows = x_windows.view(-1, win_size_sq, C)  # num_win * B, window_size*window_size, C
        BW, N, _ = x_windows.shape

        qkv = self.qkv(x_windows)
        qkv = qkv.reshape(BW, N, 3, self.num_heads, self.dim_out // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(win_size_sq, win_size_sq, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh * Ww, Wh * Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        if self.attn_mask is not None:
            num_win = self.attn_mask.shape[0]
            attn = attn.view(B, num_win, self.num_heads, N, N) + self.attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(BW, N, self.dim_out)

        # merge windows
        x = x.view(-1, self.win_size, self.win_size, self.dim_out)
        shifted_x = window_reverse(x, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H, W, self.dim_out).permute(0, 3, 1, 2)
        x = self.pool(x)
        return x


