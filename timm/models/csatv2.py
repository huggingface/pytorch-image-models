"""CSATv2

A frequency-domain vision model using DCT transforms with spatial attention.

Paper: TBD

This model created by members of MLPA Lab. Welcome feedback and suggestion, questions.
gusdlf93@naver.com
juno.demie.oh@gmail.com

Refined for timm by Ross Wightman
"""
import math
import warnings
from functools import partial, reduce
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import trunc_normal_, DropPath, Mlp, LayerNorm2d, Attention, NormMlpClassifierHead
from timm.layers.grn import GlobalResponseNorm
from timm.models._builder import build_model_with_cfg
from timm.models._features import feature_take_indices
from timm.models._manipulate import checkpoint, checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['CSATv2', 'csatv2']

# DCT frequency normalization statistics (Y, Cb, Cr channels x 64 coefficients)
_DCT_MEAN = (
    (932.42657, -0.00260, 0.33415, -0.02840, 0.00003, -0.02792, -0.00183, 0.00006,
     0.00032, 0.03402, -0.00571, 0.00020, 0.00006, -0.00038, -0.00558, -0.00116,
     -0.00000, -0.00047, -0.00008, -0.00030, 0.00942, 0.00161, -0.00009, -0.00006,
     -0.00014, -0.00035, 0.00001, -0.00220, 0.00033, -0.00002, -0.00003, -0.00020,
     0.00007, -0.00000, 0.00005, 0.00293, -0.00004, 0.00006, 0.00019, 0.00004,
     0.00006, -0.00015, -0.00002, 0.00007, 0.00010, -0.00004, 0.00008, 0.00000,
     0.00008, -0.00001, 0.00015, 0.00002, 0.00007, 0.00003, 0.00004, -0.00001,
     0.00004, -0.00000, 0.00002, -0.00000, -0.00008, -0.00000, -0.00003, 0.00003),
    (962.34735, -0.00428, 0.09835, 0.00152, -0.00009, 0.00312, -0.00141, -0.00001,
     -0.00013, 0.01050, 0.00065, 0.00006, -0.00000, 0.00003, 0.00264, 0.00000,
     0.00001, 0.00007, -0.00006, 0.00003, 0.00341, 0.00163, 0.00004, 0.00003,
     -0.00001, 0.00008, -0.00000, 0.00090, 0.00018, -0.00006, -0.00001, 0.00007,
     -0.00003, -0.00001, 0.00006, 0.00084, -0.00000, -0.00001, 0.00000, 0.00004,
     -0.00001, -0.00002, 0.00000, 0.00001, 0.00002, 0.00001, 0.00004, 0.00011,
     0.00000, -0.00003, 0.00011, -0.00002, 0.00001, 0.00001, 0.00001, 0.00001,
     -0.00007, -0.00003, 0.00001, 0.00000, 0.00001, 0.00002, 0.00001, 0.00000),
    (1053.16101, -0.00213, -0.09207, 0.00186, 0.00013, 0.00034, -0.00119, 0.00002,
     0.00011, -0.00984, 0.00046, -0.00007, -0.00001, -0.00005, 0.00180, 0.00042,
     0.00002, -0.00010, 0.00004, 0.00003, -0.00301, 0.00125, -0.00002, -0.00003,
     -0.00001, -0.00001, -0.00001, 0.00056, 0.00021, 0.00001, -0.00001, 0.00002,
     -0.00001, -0.00001, 0.00005, -0.00070, -0.00002, -0.00002, 0.00005, -0.00004,
     -0.00000, 0.00002, -0.00002, 0.00001, 0.00000, -0.00003, 0.00004, 0.00007,
     0.00001, 0.00000, 0.00013, -0.00000, 0.00000, 0.00002, -0.00000, -0.00001,
     -0.00004, -0.00003, 0.00000, 0.00001, -0.00001, 0.00001, -0.00000, 0.00000),
)

_DCT_VAR = (
    (270372.37500, 6287.10645, 5974.94043, 1653.10889, 1463.91748, 1832.58997, 755.92468, 692.41528,
     648.57184, 641.46881, 285.79288, 301.62100, 380.43405, 349.84027, 374.15891, 190.30960,
     190.76746, 221.64578, 200.82646, 145.87979, 126.92046, 62.14622, 67.75562, 102.42001,
     129.74922, 130.04631, 103.12189, 97.76417, 53.17402, 54.81048, 73.48712, 81.04342,
     69.35100, 49.06024, 33.96053, 37.03279, 20.48858, 24.94830, 33.90822, 44.54912,
     47.56363, 40.03160, 30.43313, 22.63899, 26.53739, 26.57114, 21.84404, 17.41557,
     15.18253, 10.69678, 11.24111, 12.97229, 15.08971, 15.31646, 8.90409, 7.44213,
     6.66096, 6.97719, 4.17834, 3.83882, 4.51073, 2.36646, 2.41363, 1.48266),
    (18839.21094, 321.70932, 300.15259, 77.47830, 76.02293, 89.04748, 33.99642, 34.74807,
     32.12333, 28.19588, 12.04675, 14.26871, 18.45779, 16.59588, 15.67892, 7.37718,
     8.56312, 10.28946, 9.41013, 6.69090, 5.16453, 2.55186, 3.03073, 4.66765,
     5.85418, 5.74644, 4.33702, 3.66948, 1.95107, 2.26034, 3.06380, 3.50705,
     3.06359, 2.19284, 1.54454, 1.57860, 0.97078, 1.13941, 1.48653, 1.89996,
     1.95544, 1.64950, 1.24754, 0.93677, 1.09267, 1.09516, 0.94163, 0.78966,
     0.72489, 0.50841, 0.50909, 0.55664, 0.63111, 0.64125, 0.38847, 0.33378,
     0.30918, 0.33463, 0.20875, 0.19298, 0.21903, 0.13380, 0.13444, 0.09554),
    (17127.39844, 292.81421, 271.45209, 66.64056, 63.60253, 76.35437, 28.06587, 27.84831,
     25.96656, 23.60370, 9.99173, 11.34992, 14.46955, 12.92553, 12.69353, 5.91537,
     6.60187, 7.90891, 7.32825, 5.32785, 4.29660, 2.13459, 2.44135, 3.66021,
     4.50335, 4.38959, 3.34888, 2.97181, 1.60633, 1.77010, 2.35118, 2.69018,
     2.38189, 1.74596, 1.26014, 1.31684, 0.79327, 0.92046, 1.17670, 1.47609,
     1.50914, 1.28725, 0.99898, 0.74832, 0.85736, 0.85800, 0.74663, 0.63508,
     0.58748, 0.41098, 0.41121, 0.44663, 0.50277, 0.51519, 0.31729, 0.27336,
     0.25399, 0.27241, 0.17353, 0.16255, 0.18440, 0.11602, 0.11511, 0.08450),
)


def _zigzag_permutation(rows: int, cols: int) -> List[int]:
    """Generate zigzag scan order for DCT coefficients."""
    idx_matrix = np.arange(0, rows * cols, 1).reshape(rows, cols).tolist()
    dia = [[] for _ in range(rows + cols - 1)]
    zigzag = []
    for i in range(rows):
        for j in range(cols):
            s = i + j
            if s % 2 == 0:
                dia[s].insert(0, idx_matrix[i][j])
            else:
                dia[s].append(idx_matrix[i][j])
    for d in dia:
        zigzag.extend(d)
    return zigzag


def _dct_kernel_type_2(
        kernel_size: int,
        orthonormal: bool,
        device=None,
        dtype=None,
) -> torch.Tensor:
    """Generate Type-II DCT kernel matrix."""
    dd = dict(device=device, dtype=dtype)
    x = torch.eye(kernel_size, **dd)
    v = x.clone().contiguous().view(-1, kernel_size)
    v = torch.cat([v, v.flip([1])], dim=-1)
    v = torch.fft.fft(v, dim=-1)[:, :kernel_size]
    k = (
        torch.tensor(-1j, device=device, dtype=torch.complex64) * torch.pi
        * torch.arange(kernel_size, device=device, dtype=torch.long)[None, :]
    )
    k = torch.exp(k / (kernel_size * 2))
    v = v * k
    v = v.real
    if orthonormal:
        v[:, 0] = v[:, 0] * torch.sqrt(torch.tensor(1 / (kernel_size * 4), **dd))
        v[:, 1:] = v[:, 1:] * torch.sqrt(torch.tensor(1 / (kernel_size * 2), **dd))
    v = v.contiguous().view(*x.shape)
    return v


def _dct_kernel_type_3(
        kernel_size: int,
        orthonormal: bool,
        device=None,
        dtype=None,
) -> torch.Tensor:
    """Generate Type-III DCT kernel matrix (inverse of Type-II)."""
    return torch.linalg.inv(_dct_kernel_type_2(kernel_size, orthonormal, device, dtype))


class Dct1d(nn.Module):
    """1D Discrete Cosine Transform layer."""

    def __init__(
            self,
            kernel_size: int,
            kernel_type: int = 2,
            orthonormal: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        dd = dict(device=device, dtype=dtype)
        super().__init__()
        kernel = {'2': _dct_kernel_type_2, '3': _dct_kernel_type_3}
        dct_weights = kernel[f'{kernel_type}'](kernel_size, orthonormal, **dd).T
        self.register_buffer('weights', dct_weights.contiguous())
        self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weights, self.bias)


class Dct2d(nn.Module):
    """2D Discrete Cosine Transform layer."""

    def __init__(
            self,
            kernel_size: int,
            kernel_type: int = 2,
            orthonormal: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        dd = dict(device=device, dtype=dtype)
        super().__init__()
        self.transform = Dct1d(kernel_size, kernel_type, orthonormal, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(self.transform(x).transpose(-1, -2)).transpose(-1, -2)


def _split_out_chs(out_chs: int, ratio=(24, 4, 4)):
    # reduce ratio to smallest integers (24,4,4) -> (6,1,1)
    g = reduce(math.gcd, ratio)
    r = tuple(x // g for x in ratio)
    denom = sum(r)

    assert out_chs % denom == 0 and out_chs >= denom, (
        f"out_chs={out_chs} can't be split into Y/Cb/Cr with ratio {ratio} "
        f"(reduced {r}); out_chs must be a multiple of {denom}."
    )

    unit = out_chs // denom
    y, cb, cr = (ri * unit for ri in r)
    assert y + cb + cr == out_chs and min(y, cb, cr) > 0
    return y, cb, cr


class LearnableDct2d(nn.Module):
    """Learnable 2D DCT stem with RGB to YCbCr conversion and frequency selection."""

    def __init__(
            self,
            kernel_size: int,
            kernel_type: int = 2,
            orthonormal: bool = True,
            out_chs: int = 32,
            device=None,
            dtype=None,
    ) -> None:
        dd = dict(device=device, dtype=dtype)
        super().__init__()
        self.k = kernel_size
        self.transform = Dct2d(kernel_size, kernel_type, orthonormal, **dd)
        self.permutation = _zigzag_permutation(kernel_size, kernel_size)

        y_ch, cb_ch, cr_ch = _split_out_chs(out_chs, ratio=(24, 4, 4))
        self.conv_y  = nn.Conv2d(kernel_size ** 2, y_ch,  kernel_size=1, padding=0, **dd)
        self.conv_cb = nn.Conv2d(kernel_size ** 2, cb_ch, kernel_size=1, padding=0, **dd)
        self.conv_cr = nn.Conv2d(kernel_size ** 2, cr_ch, kernel_size=1, padding=0, **dd)

        # Register empty buffers for DCT normalization statistics
        self.register_buffer('mean', torch.empty(3, 64, device=device, dtype=dtype), persistent=False)
        self.register_buffer('var', torch.empty(3, 64, device=device, dtype=dtype), persistent=False)
        # Shape (3, 1, 1) for BCHW broadcasting
        self.register_buffer('imagenet_mean', torch.empty(3, 1, 1, device=device, dtype=dtype), persistent=False)
        self.register_buffer('imagenet_std', torch.empty(3, 1, 1, device=device, dtype=dtype), persistent=False)

        if not self.mean.is_meta:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize buffers."""
        self._init_buffers()

    def _init_buffers(self) -> None:
        """Compute and fill non-persistent buffer values."""
        self.mean.copy_(torch.tensor(_DCT_MEAN))
        self.var.copy_(torch.tensor(_DCT_VAR))
        self.imagenet_mean.copy_(torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        self.imagenet_std.copy_(torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))

    def init_non_persistent_buffers(self) -> None:
        """Initialize non-persistent buffers."""
        self._init_buffers()

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from ImageNet normalized to [0, 255] range."""
        return x.mul(self.imagenet_std).add_(self.imagenet_mean) * 255

    def _rgb_to_ycbcr(self, x: torch.Tensor) -> torch.Tensor:
        """Convert RGB to YCbCr color space (BCHW input/output)."""
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        y = r * 0.299 + g * 0.587 + b * 0.114
        cb = 0.564 * (b - y) + 128
        cr = 0.713 * (r - y) + 128
        return torch.stack([y, cb, cr], dim=1)

    def _frequency_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize DCT coefficients using precomputed statistics."""
        std = self.var ** 0.5 + 1e-8
        return (x - self.mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self._denormalize(x)
        x = self._rgb_to_ycbcr(x)
        # Extract non-overlapping k x k patches
        x = x.reshape(b, c, h // self.k, self.k, w // self.k, self.k)  # (B, C, H//k, k, W//k, k)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, H//k, W//k, C, k, k)
        x = self.transform(x)
        x = x.reshape(-1, c, self.k * self.k)
        x = x[:, :, self.permutation]
        x = self._frequency_normalize(x)
        x = x.reshape(b, h // self.k, w // self.k, c, -1)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        x_y = self.conv_y(x[:, 0])
        x_cb = self.conv_cb(x[:, 1])
        x_cr = self.conv_cr(x[:, 2])
        return torch.cat([x_y, x_cb, x_cr], dim=1)


class Dct2dStats(nn.Module):
    """Utility module to compute DCT coefficient statistics."""

    def __init__(
            self,
            kernel_size: int,
            kernel_type: int = 2,
            orthonormal: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        dd = dict(device=device, dtype=dtype)
        super().__init__()
        self.k = kernel_size
        self.transform = Dct2d(kernel_size, kernel_type, orthonormal, **dd)
        self.permutation = _zigzag_permutation(kernel_size, kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        # Extract non-overlapping k x k patches
        x = x.reshape(b, c, h // self.k, self.k, w // self.k, self.k)  # (B, C, H//k, k, W//k, k)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, H//k, W//k, C, k, k)
        x = self.transform(x)
        x = x.reshape(-1, c, self.k * self.k)
        x = x[:, :, self.permutation]
        x = x.reshape(b * (h // self.k) * (w // self.k), c, -1)

        mean_list = torch.zeros([3, 64])
        var_list = torch.zeros([3, 64])
        for i in range(3):
            mean_list[i] = torch.mean(x[:, i], dim=0)
            var_list[i] = torch.var(x[:, i], dim=0)
        return mean_list, var_list


class Block(nn.Module):
    """ConvNeXt-style block with spatial attention."""

    def __init__(
            self,
            dim: int,
            drop_path: float = 0.,
            device=None,
            dtype=None,
    ) -> None:
        dd = dict(device=device, dtype=dtype)
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, **dd)
        self.norm = nn.LayerNorm(dim, eps=1e-6, **dd)
        self.pwconv1 = nn.Linear(dim, 4 * dim, **dd)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(4 * dim, channels_last=True, **dd)
        self.pwconv2 = nn.Linear(4 * dim, dim, **dd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SpatialAttention(**dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        attn = self.attn(x)
        attn = F.interpolate(attn, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = x * attn

        return shortcut + self.drop_path(x)


class SpatialTransformerBlock(nn.Module):
    """Lightweight transformer block for spatial attention (1-channel, 7x7 grid).

    This is a simplified transformer with single-head, 1-dim attention over spatial
    positions. Used inside SpatialAttention where input is 1 channel at 7x7 resolution.
    """

    def __init__(
            self,
            device=None,
            dtype=None,
    ) -> None:
        dd = dict(device=device, dtype=dtype)
        super().__init__()
        # Single-head attention with 1-dim q/k/v (no output projection needed)
        self.pos_embed = PosConv(in_chans=1, **dd)
        self.norm1 = nn.LayerNorm(1, **dd)
        self.qkv = nn.Linear(1, 3, bias=False, **dd)

        # Feedforward: 1 -> 4 -> 1
        self.norm2 = nn.LayerNorm(1, **dd)
        self.mlp = Mlp(1, 4, 1, act_layer=nn.GELU, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Attention block
        shortcut = x
        x_t = x.flatten(2).transpose(1, 2)  # (B, N, 1)
        x_t = self.norm1(x_t)
        x_t = self.pos_embed(x_t, (H, W))

        # Simple single-head attention with scalar q/k/v
        qkv = self.qkv(x_t)  # (B, N, 3)
        q, k, v = qkv.unbind(-1)  # each (B, N)
        attn = (q @ k.transpose(-1, -2)).softmax(dim=-1)  # (B, N, N)
        x_t = (attn @ v).unsqueeze(-1)  # (B, N, 1)

        x_t = x_t.transpose(1, 2).reshape(B, C, H, W)
        x = shortcut + x_t

        # Feedforward block
        shortcut = x
        x_t = x.flatten(2).transpose(1, 2)
        x_t = self.mlp(self.norm2(x_t))
        x_t = x_t.transpose(1, 2).reshape(B, C, H, W)
        x = shortcut + x_t

        return x


class SpatialAttention(nn.Module):
    """Spatial attention module using channel statistics and transformer."""

    def __init__(
            self,
            device=None,
            dtype=None,
    ) -> None:
        dd = dict(device=device, dtype=dtype)
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, **dd)
        self.attn = SpatialTransformerBlock(**dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_avg = x.mean(dim=1, keepdim=True)
        x_max = x.amax(dim=1, keepdim=True)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.attn(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with optional downsampling and convolutional position encoding."""

    def __init__(
            self,
            inp: int,
            oup: int,
            num_heads: int = 8,
            attn_head_dim: int = 32,
            downsample: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            drop_path: float = 0.,
            device=None,
            dtype=None,
    ) -> None:
        dd = dict(device=device, dtype=dtype)
        super().__init__()
        hidden_dim = int(inp * 4)
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False, **dd)
        else:
            self.pool1 = nn.Identity()
            self.pool2 = nn.Identity()
            self.proj = nn.Identity()

        self.pos_embed = PosConv(in_chans=inp, **dd)
        self.norm1 = nn.LayerNorm(inp, **dd)
        self.attn = Attention(
            dim=inp,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            dim_out=oup,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **dd,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(oup, **dd)
        self.mlp = Mlp(oup, hidden_dim, oup, act_layer=nn.GELU, drop=proj_drop, **dd)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample:
            shortcut = self.proj(self.pool1(x))
            x_t = self.pool2(x)
            B, C, H, W = x_t.shape
            x_t = x_t.flatten(2).transpose(1, 2)
            x_t = self.norm1(x_t)
            x_t = self.pos_embed(x_t, (H, W))
            x_t = self.attn(x_t)
            x_t = x_t.transpose(1, 2).reshape(B, -1, H, W)
            x = shortcut + self.drop_path1(x_t)
        else:
            B, C, H, W = x.shape
            shortcut = x
            x_t = x.flatten(2).transpose(1, 2)
            x_t = self.norm1(x_t)
            x_t = self.pos_embed(x_t, (H, W))
            x_t = self.attn(x_t)
            x_t = x_t.transpose(1, 2).reshape(B, -1, H, W)
            x = shortcut + self.drop_path1(x_t)

        # MLP block
        B, C, H, W = x.shape
        shortcut = x
        x_t = x.flatten(2).transpose(1, 2)
        x_t = self.mlp(self.norm2(x_t))
        x_t = x_t.transpose(1, 2).reshape(B, C, H, W)
        x = shortcut + self.drop_path2(x_t)

        return x


class PosConv(nn.Module):
    """Convolutional position encoding."""

    def __init__(
            self,
            in_chans: int,
            device=None,
            dtype=None,
    ) -> None:
        dd = dict(device=device, dtype=dtype)
        super().__init__()
        self.proj = nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, padding=1, bias=True, groups=in_chans, **dd)

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        B, N, C = x.shape
        H, W = size
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        return x.flatten(2).transpose(1, 2)


class CSATv2(nn.Module):
    """CSATv2: Frequency-domain vision model with spatial attention.

    A hybrid architecture that processes images in the DCT frequency domain
    with ConvNeXt-style blocks and transformer attention.
    """

    def __init__(
            self,
            num_classes: int = 1000,
            in_chans: int = 3,
            dims: Tuple[int, ...] = (32, 72, 168, 386),
            depths: Tuple[int, ...] = (2, 2, 8, 6),
            transformer_depths: Tuple[int, ...] = (0, 0, 2, 2),
            drop_path_rate: float = 0.0,
            transformer_drop_path: bool = False,
            global_pool: str = 'avg',
            device=None,
            dtype=None,
            **kwargs,
    ) -> None:
        dd = dict(device=device, dtype=dtype)
        super().__init__()
        if in_chans != 3:
            warnings.warn(
                f'CSATv2 is designed for 3-channel RGB input. '
                f'in_chans={in_chans} may not work correctly with the DCT stem.'
            )
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.grad_checkpointing = False

        self.num_features = dims[-1]
        self.head_hidden_size = self.num_features

        # Build feature_info dynamically
        self.feature_info = [dict(num_chs=dims[0], reduction=8, module='stem_dct')]
        reduction = 8
        for i, dim in enumerate(dims):
            if i > 0:
                reduction *= 2
            self.feature_info.append(dict(num_chs=dim, reduction=reduction, module=f'stages.{i}'))

        # Build drop path rates for all blocks (0 for transformer blocks when transformer_drop_path=False)
        total_blocks = sum(depths) if transformer_drop_path else sum(d - t for d, t in zip(depths, transformer_depths))
        dp_iter = iter(torch.linspace(0, drop_path_rate, total_blocks).tolist())
        dp_rates = []
        for depth, t_depth in zip(depths, transformer_depths):
            dp_rates += [next(dp_iter) for _ in range(depth - t_depth)]
            dp_rates += [next(dp_iter) if transformer_drop_path else 0. for _ in range(t_depth)]

        self.stem_dct = LearnableDct2d(8, out_chs=dims[0], **dd)

        # Build stages dynamically
        dp_iter = iter(dp_rates)
        stages = []
        for i, (dim, depth, t_depth) in enumerate(zip(dims, depths, transformer_depths)):
            layers = (
                # Downsample at start of stage (except first stage)
                ([nn.Conv2d(dims[i - 1], dim, kernel_size=2, stride=2, **dd)] if i > 0 else []) +
                # Conv blocks
                [Block(dim=dim, drop_path=next(dp_iter), **dd) for _ in range(depth - t_depth)] +
                # Transformer blocks at end of stage
                [TransformerBlock(inp=dim, oup=dim, drop_path=next(dp_iter), **dd) for _ in range(t_depth)] +
                # Trailing LayerNorm (except last stage)
                ([LayerNorm2d(dim, eps=1e-6, **dd)] if i < len(depths) - 1 else [])
            )
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)

        self.head = NormMlpClassifierHead(dims[-1], num_classes, pool_type=global_pool, **dd)

        if not self.stem_dct.conv_y.weight.is_meta:
            self.init_weights(needs_reset=False)

    def init_weights(self, needs_reset: bool = True):
        self.apply(partial(self._init_weights, needs_reset=needs_reset))

    def _init_weights(self, m: nn.Module, needs_reset: bool = True) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif needs_reset and hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head.reset(num_classes, pool_type=global_pool)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_dct(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        return x

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass returning intermediate features.

        Args:
            x: Input image tensor.
            indices: Indices of features to return (0=stem_dct, 1-4=stages). None returns all.
            norm: Apply norm layer to final intermediate (unused, for API compat).
            stop_early: Stop iterating when last desired intermediate is reached.
            output_fmt: Output format, must be 'NCHW'.
            intermediates_only: Only return intermediate features.

        Returns:
            List of intermediate features or tuple of (final features, intermediates).
        """
        assert output_fmt == 'NCHW', 'Output format must be NCHW.'
        intermediates = []
        # 5 feature levels: stem_dct (0) + stages 0-3 (1-4)
        take_indices, max_index = feature_take_indices(len(self.stages) + 1, indices)

        x = self.stem_dct(x)
        if 0 in take_indices:
            intermediates.append(x)

        if torch.jit.is_scripting() or not stop_early:
            stages = self.stages
        else:
            # max_index is 0-4, stages are 1-4, so we need max_index stages
            stages = self.stages[:max_index] if max_index > 0 else []

        for feat_idx, stage in enumerate(stages):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(stage, x)
            else:
                x = stage(x)
            if feat_idx + 1 in take_indices:  # +1 because stem is index 0
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ) -> List[int]:
        """Prune layers not required for specified intermediates.

        Args:
            indices: Indices of intermediate layers to keep (0=stem_dct, 1-4=stages).
            prune_norm: Whether to prune the final norm layer.
            prune_head: Whether to prune the classifier head.

        Returns:
            List of indices that were kept.
        """
        # 5 feature levels: stem_dct (0) + stages 0-3 (1-4)
        take_indices, max_index = feature_take_indices(len(self.stages) + 1, indices)
        # max_index is 0-4, stages are 1-4, so we keep max_index stages
        self.stages = self.stages[:max_index] if max_index > 0 else nn.Sequential()

        if prune_norm:
            self.head.norm = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')

        return take_indices

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.forward_head(x)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 512, 512),
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'interpolation': 'bilinear', 'crop_pct': 1.0,
        'classifier': 'head.fc', 'first_conv': [],
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'csatv2': _cfg(
        hf_hub_id='timm/',
    ),
    'csatv2_21m': _cfg(),
})


def checkpoint_filter_fn(state_dict: dict, model: nn.Module) -> dict:
    """Remap original CSATv2 checkpoint to timm format.

    Handles two key structural changes:
    1) Stage naming: stages1/2/3/4 -> stages.0/1/2/3
    2) Downsample position: moved from end of stage N to start of stage N+1
    """
    if "stages.0.0.grn.weight" in state_dict:
        return state_dict  # already in timm format

    import re

    # FIXME this downsample idx is wired to the original 'csatv2' model size
    downsample_idx = {1: 3, 2: 3, 3: 9}  # original stage -> downsample index

    dct_re   = re.compile(r"^dct\.")
    stage_re = re.compile(r"^stages([1-4])\.(\d+)\.(.*)$")
    head_re  = re.compile(r"^head\.")
    norm_re  = re.compile(r"^norm\.")

    def remap_stage(m: re.Match) -> str:
        stage, idx, rest = int(m.group(1)), int(m.group(2)), m.group(3)
        if stage in downsample_idx and idx == downsample_idx[stage]:
            return f"stages.{stage}.0.{rest}"                 # move downsample to next stage @0
        if stage == 1:
            return f"stages.0.{idx}.{rest}"                  # stage1 -> stages.0
        return f"stages.{stage - 1}.{idx + 1}.{rest}"        # stage2-4 -> stages.1-3, shift +1

    out = {}
    for k, v in state_dict.items():
        # dct -> stem_dct, and Y/Cb/Cr conv names
        k = dct_re.sub("stem_dct.", k)
        k = (k.replace(".Y_Conv.",  ".conv_y.")
               .replace(".Cb_Conv.", ".conv_cb.")
               .replace(".Cr_Conv.", ".conv_cr."))

        # stage remap + downsample relocation
        k = stage_re.sub(remap_stage, k)

        # GRN: gamma/beta -> weight/bias (reshape)
        if "grn.gamma" in k:
            k, v = k.replace("grn.gamma", "grn.weight"), v.reshape(-1)
        elif "grn.beta" in k:
            k, v = k.replace("grn.beta", "grn.bias"), v.reshape(-1)

        # FeedForward(nn.Sequential) -> Mlp + norm renames
        if ".ff.net.0." in k:
            k = k.replace(".ff.net.0.", ".mlp.fc1.")
        elif ".ff.net.3." in k:
            k = k.replace(".ff.net.3.", ".mlp.fc2.")
        elif ".ff_norm." in k:
            k = k.replace(".ff_norm.", ".norm2.")
        elif ".attn_norm." in k:
            k = k.replace(".attn_norm.", ".norm1.")

        # attention -> attn (handle nested first)
        if ".attention.attention." in k:
            k = (k.replace(".attention.attention.attn.to_qkv.", ".attn.attn.qkv.")
                   .replace(".attention.attention.attn.",        ".attn.attn.")
                   .replace(".attention.attention.",             ".attn.attn."))
        elif ".attention." in k:
            k = k.replace(".attention.", ".attn.")

        # TransformerBlock attention name remaps
        if ".attn.to_qkv." in k:
            k = k.replace(".attn.to_qkv.", ".attn.qkv.")
        elif ".attn.to_out.0." in k:
            k = k.replace(".attn.to_out.0.", ".attn.proj.")

        # .attn.pos_embed -> .pos_embed (but not SpatialTransformerBlock's .attn.attn.pos_embed)
        if ".attn.pos_embed." in k and ".attn.attn." not in k:
            k = k.replace(".attn.pos_embed.", ".pos_embed.")

        # head -> head.fc, norm -> head.norm (order matters)
        k = head_re.sub("head.fc.", k)
        k = norm_re.sub("head.norm.", k)

        out[k] = v

    return out


def _create_csatv2(variant: str, pretrained: bool = False, **kwargs) -> CSATv2:
    out_indices = kwargs.pop('out_indices', (1, 2, 3, 4))
    return build_model_with_cfg(
        CSATv2,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, flatten_sequential=True),
        default_cfg=default_cfgs[variant],
        **kwargs,
    )


@register_model
def csatv2(pretrained: bool = False, **kwargs) -> CSATv2:
    return _create_csatv2('csatv2', pretrained, **kwargs)


@register_model
def csatv2_21m(pretrained: bool = False, **kwargs) -> CSATv2:
    # experimental ~20-21M param larger model to validate flexible arch spec
    model_args = dict(
        dims = (48, 96, 224, 448),
        depths = (3, 3, 10, 8),
        transformer_depths = (0, 0, 4, 3)

    )
    return _create_csatv2('csatv2_21m', pretrained, **dict(model_args, **kwargs))