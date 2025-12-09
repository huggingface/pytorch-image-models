"""CSATv2

A frequency-domain vision model using DCT transforms with spatial attention.

Paper: TBD
"""
import math
import warnings
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
_DCT_MEAN = [
    [932.42657, -0.00260, 0.33415, -0.02840, 0.00003, -0.02792, -0.00183, 0.00006,
     0.00032, 0.03402, -0.00571, 0.00020, 0.00006, -0.00038, -0.00558, -0.00116,
     -0.00000, -0.00047, -0.00008, -0.00030, 0.00942, 0.00161, -0.00009, -0.00006,
     -0.00014, -0.00035, 0.00001, -0.00220, 0.00033, -0.00002, -0.00003, -0.00020,
     0.00007, -0.00000, 0.00005, 0.00293, -0.00004, 0.00006, 0.00019, 0.00004,
     0.00006, -0.00015, -0.00002, 0.00007, 0.00010, -0.00004, 0.00008, 0.00000,
     0.00008, -0.00001, 0.00015, 0.00002, 0.00007, 0.00003, 0.00004, -0.00001,
     0.00004, -0.00000, 0.00002, -0.00000, -0.00008, -0.00000, -0.00003, 0.00003],
    [962.34735, -0.00428, 0.09835, 0.00152, -0.00009, 0.00312, -0.00141, -0.00001,
     -0.00013, 0.01050, 0.00065, 0.00006, -0.00000, 0.00003, 0.00264, 0.00000,
     0.00001, 0.00007, -0.00006, 0.00003, 0.00341, 0.00163, 0.00004, 0.00003,
     -0.00001, 0.00008, -0.00000, 0.00090, 0.00018, -0.00006, -0.00001, 0.00007,
     -0.00003, -0.00001, 0.00006, 0.00084, -0.00000, -0.00001, 0.00000, 0.00004,
     -0.00001, -0.00002, 0.00000, 0.00001, 0.00002, 0.00001, 0.00004, 0.00011,
     0.00000, -0.00003, 0.00011, -0.00002, 0.00001, 0.00001, 0.00001, 0.00001,
     -0.00007, -0.00003, 0.00001, 0.00000, 0.00001, 0.00002, 0.00001, 0.00000],
    [1053.16101, -0.00213, -0.09207, 0.00186, 0.00013, 0.00034, -0.00119, 0.00002,
     0.00011, -0.00984, 0.00046, -0.00007, -0.00001, -0.00005, 0.00180, 0.00042,
     0.00002, -0.00010, 0.00004, 0.00003, -0.00301, 0.00125, -0.00002, -0.00003,
     -0.00001, -0.00001, -0.00001, 0.00056, 0.00021, 0.00001, -0.00001, 0.00002,
     -0.00001, -0.00001, 0.00005, -0.00070, -0.00002, -0.00002, 0.00005, -0.00004,
     -0.00000, 0.00002, -0.00002, 0.00001, 0.00000, -0.00003, 0.00004, 0.00007,
     0.00001, 0.00000, 0.00013, -0.00000, 0.00000, 0.00002, -0.00000, -0.00001,
     -0.00004, -0.00003, 0.00000, 0.00001, -0.00001, 0.00001, -0.00000, 0.00000],
]

_DCT_VAR = [
    [270372.37500, 6287.10645, 5974.94043, 1653.10889, 1463.91748, 1832.58997, 755.92468, 692.41528,
     648.57184, 641.46881, 285.79288, 301.62100, 380.43405, 349.84027, 374.15891, 190.30960,
     190.76746, 221.64578, 200.82646, 145.87979, 126.92046, 62.14622, 67.75562, 102.42001,
     129.74922, 130.04631, 103.12189, 97.76417, 53.17402, 54.81048, 73.48712, 81.04342,
     69.35100, 49.06024, 33.96053, 37.03279, 20.48858, 24.94830, 33.90822, 44.54912,
     47.56363, 40.03160, 30.43313, 22.63899, 26.53739, 26.57114, 21.84404, 17.41557,
     15.18253, 10.69678, 11.24111, 12.97229, 15.08971, 15.31646, 8.90409, 7.44213,
     6.66096, 6.97719, 4.17834, 3.83882, 4.51073, 2.36646, 2.41363, 1.48266],
    [18839.21094, 321.70932, 300.15259, 77.47830, 76.02293, 89.04748, 33.99642, 34.74807,
     32.12333, 28.19588, 12.04675, 14.26871, 18.45779, 16.59588, 15.67892, 7.37718,
     8.56312, 10.28946, 9.41013, 6.69090, 5.16453, 2.55186, 3.03073, 4.66765,
     5.85418, 5.74644, 4.33702, 3.66948, 1.95107, 2.26034, 3.06380, 3.50705,
     3.06359, 2.19284, 1.54454, 1.57860, 0.97078, 1.13941, 1.48653, 1.89996,
     1.95544, 1.64950, 1.24754, 0.93677, 1.09267, 1.09516, 0.94163, 0.78966,
     0.72489, 0.50841, 0.50909, 0.55664, 0.63111, 0.64125, 0.38847, 0.33378,
     0.30918, 0.33463, 0.20875, 0.19298, 0.21903, 0.13380, 0.13444, 0.09554],
    [17127.39844, 292.81421, 271.45209, 66.64056, 63.60253, 76.35437, 28.06587, 27.84831,
     25.96656, 23.60370, 9.99173, 11.34992, 14.46955, 12.92553, 12.69353, 5.91537,
     6.60187, 7.90891, 7.32825, 5.32785, 4.29660, 2.13459, 2.44135, 3.66021,
     4.50335, 4.38959, 3.34888, 2.97181, 1.60633, 1.77010, 2.35118, 2.69018,
     2.38189, 1.74596, 1.26014, 1.31684, 0.79327, 0.92046, 1.17670, 1.47609,
     1.50914, 1.28725, 0.99898, 0.74832, 0.85736, 0.85800, 0.74663, 0.63508,
     0.58748, 0.41098, 0.41121, 0.44663, 0.50277, 0.51519, 0.31729, 0.27336,
     0.25399, 0.27241, 0.17353, 0.16255, 0.18440, 0.11602, 0.11511, 0.08450],
]


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
    factory_kwargs = dict(device=device, dtype=dtype)
    x = torch.eye(kernel_size, **factory_kwargs)
    v = x.clone().contiguous().view(-1, kernel_size)
    v = torch.cat([v, v.flip([1])], dim=-1)
    v = torch.fft.fft(v, dim=-1)[:, :kernel_size]
    try:
        k = torch.tensor(-1j, **factory_kwargs) * torch.pi * torch.arange(kernel_size, **factory_kwargs)[None, :]
    except AttributeError:
        k = torch.tensor(-1j, **factory_kwargs) * math.pi * torch.arange(kernel_size, **factory_kwargs)[None, :]
    k = torch.exp(k / (kernel_size * 2))
    v = v * k
    v = v.real
    if orthonormal:
        v[:, 0] = v[:, 0] * torch.sqrt(torch.tensor(1 / (kernel_size * 4), **factory_kwargs))
        v[:, 1:] = v[:, 1:] * torch.sqrt(torch.tensor(1 / (kernel_size * 2), **factory_kwargs))
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
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        kernel = {'2': _dct_kernel_type_2, '3': _dct_kernel_type_3}
        dct_weights = kernel[f'{kernel_type}'](kernel_size, orthonormal, **factory_kwargs).T
        self.register_buffer('weights', dct_weights)
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
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        self.transform = Dct1d(kernel_size, kernel_type, orthonormal, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(self.transform(x).transpose(-1, -2)).transpose(-1, -2)


class LearnableDct2d(nn.Module):
    """Learnable 2D DCT stem with RGB to YCbCr conversion and frequency selection."""

    def __init__(
            self,
            kernel_size: int,
            kernel_type: int = 2,
            orthonormal: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        self.k = kernel_size
        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), stride=(kernel_size, kernel_size))
        self.transform = Dct2d(kernel_size, kernel_type, orthonormal, **factory_kwargs)
        self.permutation = _zigzag_permutation(kernel_size, kernel_size)
        self.Y_Conv = nn.Conv2d(kernel_size ** 2, 24, kernel_size=1, padding=0)
        self.Cb_Conv = nn.Conv2d(kernel_size ** 2, 4, kernel_size=1, padding=0)
        self.Cr_Conv = nn.Conv2d(kernel_size ** 2, 4, kernel_size=1, padding=0)

        self.register_buffer('mean', torch.tensor(_DCT_MEAN), persistent=False)
        self.register_buffer('var', torch.tensor(_DCT_VAR), persistent=False)
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]), persistent=False)
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]), persistent=False)

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from ImageNet normalized to [0, 255] range."""
        return x.mul(self.imagenet_std).add_(self.imagenet_mean) * 255

    def _rgb_to_ycbcr(self, x: torch.Tensor) -> torch.Tensor:
        """Convert RGB to YCbCr color space."""
        y = (x[:, :, :, 0] * 0.299) + (x[:, :, :, 1] * 0.587) + (x[:, :, :, 2] * 0.114)
        cb = 0.564 * (x[:, :, :, 2] - y) + 128
        cr = 0.713 * (x[:, :, :, 0] - y) + 128
        return torch.stack([y, cb, cr], dim=-1)

    def _frequency_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize DCT coefficients using precomputed statistics."""
        std = self.var ** 0.5 + 1e-8
        return (x - self.mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self._denormalize(x)
        x = self._rgb_to_ycbcr(x)
        x = x.permute(0, 3, 1, 2)
        x = self.unfold(x).transpose(-1, -2)
        x = x.reshape(b, h // self.k, w // self.k, c, self.k, self.k)
        x = self.transform(x)
        x = x.reshape(-1, c, self.k * self.k)
        x = x[:, :, self.permutation]
        x = self._frequency_normalize(x)
        x = x.reshape(b, h // self.k, w // self.k, c, -1)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        x_y = self.Y_Conv(x[:, 0])
        x_cb = self.Cb_Conv(x[:, 1])
        x_cr = self.Cr_Conv(x[:, 2])
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
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        self.k = kernel_size
        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), stride=(kernel_size, kernel_size))
        self.transform = Dct2d(kernel_size, kernel_type, orthonormal, **factory_kwargs)
        self.permutation = _zigzag_permutation(kernel_size, kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = self.unfold(x).transpose(-1, -2)
        x = x.reshape(b, h // self.k, w // self.k, c, self.k, self.k)
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
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(4 * dim, channels_last=True)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attention = SpatialAttention()

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

        attention = self.attention(x)
        up_attn = F.interpolate(attention, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = x * up_attn

        return shortcut + self.drop_path(x)


class SpatialTransformerBlock(nn.Module):
    """Lightweight transformer block for spatial attention (1-channel, 7x7 grid).

    This is a simplified transformer with single-head, 1-dim attention over spatial
    positions. Used inside SpatialAttention where input is 1 channel at 7x7 resolution.
    """

    def __init__(self) -> None:
        super().__init__()
        # Single-head attention with 1-dim q/k/v (no output projection needed)
        self.pos_embed = PosConv(in_chans=1)
        self.norm1 = nn.LayerNorm(1)
        self.qkv = nn.Linear(1, 3, bias=False)

        # Feedforward: 1 -> 4 -> 1
        self.norm2 = nn.LayerNorm(1)
        self.mlp = Mlp(1, 4, 1, act_layer=nn.GELU)

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

    def __init__(self) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.attention = SpatialTransformerBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_avg = x.mean(dim=1, keepdim=True)
        x_max = x.amax(dim=1, keepdim=True)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.attention(x)
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
    ) -> None:
        super().__init__()
        hidden_dim = int(inp * 4)
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        else:
            self.pool1 = nn.Identity()
            self.pool2 = nn.Identity()
            self.proj = nn.Identity()

        self.pos_embed = PosConv(in_chans=inp)
        self.norm1 = nn.LayerNorm(inp)
        self.attn = Attention(
            dim=inp,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            dim_out=oup,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm2 = nn.LayerNorm(oup)
        self.mlp = Mlp(oup, hidden_dim, oup, act_layer=nn.GELU, drop=proj_drop)

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
            x = shortcut + x_t
        else:
            B, C, H, W = x.shape
            shortcut = x
            x_t = x.flatten(2).transpose(1, 2)
            x_t = self.norm1(x_t)
            x_t = self.pos_embed(x_t, (H, W))
            x_t = self.attn(x_t)
            x_t = x_t.transpose(1, 2).reshape(B, -1, H, W)
            x = shortcut + x_t

        # MLP block
        B, C, H, W = x.shape
        shortcut = x
        x_t = x.flatten(2).transpose(1, 2)
        x_t = self.mlp(self.norm2(x_t))
        x_t = x_t.transpose(1, 2).reshape(B, C, H, W)
        x = shortcut + x_t

        return x


class PosConv(nn.Module):
    """Convolutional position encoding."""

    def __init__(
            self,
            in_chans: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, padding=1, bias=True, groups=in_chans)

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
            drop_path_rate: float = 0.0,
            global_pool: str = 'avg',
            **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.grad_checkpointing = False

        dims = [32, 72, 168, 386]
        self.num_features = dims[-1]
        self.head_hidden_size = self.num_features

        self.feature_info = [
            dict(num_chs=dims[0], reduction=8, module='stem_dct'),
            dict(num_chs=dims[0], reduction=8, module='stages.0'),
            dict(num_chs=dims[1], reduction=16, module='stages.1'),
            dict(num_chs=dims[2], reduction=32, module='stages.2'),
            dict(num_chs=dims[3], reduction=64, module='stages.3'),
        ]

        depths = [2, 2, 6, 4]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stem_dct = LearnableDct2d(8)

        self.stages = nn.Sequential(
            nn.Sequential(
                Block(dim=dims[0], drop_path=dp_rates[0]),
                Block(dim=dims[0], drop_path=dp_rates[1]),
                LayerNorm2d(dims[0], eps=1e-6),
            ),
            nn.Sequential(
                nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),
                Block(dim=dims[1], drop_path=dp_rates[2]),
                Block(dim=dims[1], drop_path=dp_rates[3]),
                LayerNorm2d(dims[1], eps=1e-6),
            ),
            nn.Sequential(
                nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2),
                Block(dim=dims[2], drop_path=dp_rates[4]),
                Block(dim=dims[2], drop_path=dp_rates[5]),
                Block(dim=dims[2], drop_path=dp_rates[6]),
                Block(dim=dims[2], drop_path=dp_rates[7]),
                Block(dim=dims[2], drop_path=dp_rates[8]),
                Block(dim=dims[2], drop_path=dp_rates[9]),
                TransformerBlock(inp=dims[2], oup=dims[2]),
                TransformerBlock(inp=dims[2], oup=dims[2]),
                LayerNorm2d(dims[2], eps=1e-6),
            ),
            nn.Sequential(
                nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2),
                Block(dim=dims[3], drop_path=dp_rates[10]),
                Block(dim=dims[3], drop_path=dp_rates[11]),
                Block(dim=dims[3], drop_path=dp_rates[12]),
                Block(dim=dims[3], drop_path=dp_rates[13]),
                TransformerBlock(inp=dims[3], oup=dims[3]),
                TransformerBlock(inp=dims[3], oup=dims[3]),
            ),
        )

        self.head = NormMlpClassifierHead(dims[-1], num_classes, pool_type=global_pool)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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


default_cfgs = generate_default_cfgs({
    'csatv2': {
        'url': 'https://huggingface.co/Hyunil/CSATv2/resolve/main/CSATv2_ImageNet_timm.pth',
        'num_classes': 1000,
        'input_size': (3, 512, 512),
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'interpolation': 'bilinear',
        'crop_pct': 1.0,
        'classifier': 'head.fc',
        'first_conv': [],
    },
})


def checkpoint_filter_fn(state_dict: dict, model: nn.Module) -> dict:
    """Remap original CSATv2 checkpoint to timm format.

    Handles two key structural changes:
    1. Stage naming: stages1/2/3/4 -> stages.0/1/2/3
    2. Downsample position: moved from end of stage N to start of stage N+1
    """
    if 'stages.0.0.grn.weight' in state_dict:
        return state_dict  # Already in timm format

    import re

    # Downsample indices in original checkpoint (Conv2d at end of each stage)
    # These move to index 0 of the next stage
    downsample_idx = {1: 3, 2: 3, 3: 9}  # stage -> downsample index

    def remap_stage(m):
        stage = int(m.group(1))
        idx = int(m.group(2))
        rest = m.group(3)
        if stage in downsample_idx and idx == downsample_idx[stage]:
            # Downsample moves to start of next stage
            return f'stages.{stage}.0.{rest}'
        elif stage == 1:
            # Stage 1 -> stages.0, indices unchanged
            return f'stages.0.{idx}.{rest}'
        else:
            # Stages 2-4 -> stages.1-3, indices shift +1 (after downsample)
            return f'stages.{stage - 1}.{idx + 1}.{rest}'

    out_dict = {}
    for k, v in state_dict.items():
        # Remap dct -> stem_dct
        k = re.sub(r'^dct\.', 'stem_dct.', k)

        # Remap stage names with index adjustments for downsample relocation
        k = re.sub(r'^stages([1-4])\.(\d+)\.(.*)$', remap_stage, k)

        # Remap GRN: gamma/beta -> weight/bias with reshape
        if 'grn.gamma' in k:
            k = k.replace('grn.gamma', 'grn.weight')
            v = v.reshape(-1)
        elif 'grn.beta' in k:
            k = k.replace('grn.beta', 'grn.bias')
            v = v.reshape(-1)

        # Remap FeedForward (nn.Sequential) to Mlp: net.0 -> fc1, net.3 -> fc2
        # Also rename ff -> mlp, ff_norm -> norm2, attn_norm -> norm1
        if '.ff.net.0.' in k:
            k = k.replace('.ff.net.0.', '.mlp.fc1.')
        elif '.ff.net.3.' in k:
            k = k.replace('.ff.net.3.', '.mlp.fc2.')
        elif '.ff_norm.' in k:
            k = k.replace('.ff_norm.', '.norm2.')
        elif '.attn_norm.' in k:
            k = k.replace('.attn_norm.', '.norm1.')

        # SpatialTransformerBlock: flatten .attention.attention.attn. -> .attention.attention.
        # and remap to_qkv -> qkv
        if '.attention.attention.attn.' in k:
            k = k.replace('.attention.attention.attn.to_qkv.', '.attention.attention.qkv.')
            k = k.replace('.attention.attention.attn.', '.attention.attention.')

        # TransformerBlock: remap attention layer names
        # to_qkv -> qkv, to_out.0 -> proj, attn.pos_embed -> pos_embed
        if '.attn.to_qkv.' in k:
            k = k.replace('.attn.to_qkv.', '.attn.qkv.')
        elif '.attn.to_out.0.' in k:
            k = k.replace('.attn.to_out.0.', '.attn.proj.')

        if '.attn.pos_embed.' in k:
            k = k.replace('.attn.pos_embed.', '.pos_embed.')

        # Remap head -> head.fc, norm -> head.norm (order matters)
        k = re.sub(r'^head\.', 'head.fc.', k)
        k = re.sub(r'^norm\.', 'head.norm.', k)

        out_dict[k] = v

    return out_dict


def _create_csatv2(variant: str, pretrained: bool = False, **kwargs) -> CSATv2:
    return build_model_with_cfg(
        CSATv2,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3, 4), flatten_sequential=True),
        default_cfg=default_cfgs[variant],
        **kwargs,
    )


@register_model
def csatv2(pretrained: bool = False, **kwargs) -> CSATv2:
    return _create_csatv2('csatv2', pretrained, **kwargs)
