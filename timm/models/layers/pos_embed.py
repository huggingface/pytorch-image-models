import math
from typing import List, Tuple, Optional, Union

import torch
from torch import nn as nn


def pixel_freq_bands(
        num_bands: int,
        max_freq: float = 224.,
        linear_bands: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
):
    if linear_bands:
        bands = torch.linspace(1.0, max_freq / 2, num_bands, dtype=dtype, device=device)
    else:
        bands = 2 ** torch.linspace(0, math.log(max_freq, 2) - 1, num_bands, dtype=dtype, device=device)
    return bands * torch.pi


def inv_freq_bands(
        num_bands: int,
        temperature: float = 100000.,
        step: int = 2,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    inv_freq = 1. / (temperature ** (torch.arange(0, num_bands, step, dtype=dtype, device=device) / num_bands))
    return inv_freq


def build_sincos2d_pos_embed(
        feat_shape: List[int],
        dim: int = 64,
        temperature: float = 10000.,
        reverse_coord: bool = False,
        interleave_sin_cos: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None
) -> torch.Tensor:
    """

    Args:
        feat_shape:
        dim:
        temperature:
        reverse_coord: stack grid order W, H instead of H, W
        interleave_sin_cos: sin, cos, sin, cos stack instead of sin, sin, cos, cos
        dtype:
        device:

    Returns:

    """
    assert dim % 4 == 0, 'Embed dimension must be divisible by 4 for sin-cos 2D position embedding'
    pos_dim = dim // 4
    bands = inv_freq_bands(pos_dim, temperature=temperature, step=1, dtype=dtype, device=device)

    if reverse_coord:
        feat_shape = feat_shape[::-1]  # stack W, H instead of H, W
    grid = torch.stack(
        torch.meshgrid([torch.arange(s, device=device, dtype=dtype) for s in feat_shape])).flatten(1).transpose(0, 1)
    pos2 = grid.unsqueeze(-1) * bands.unsqueeze(0)
    # FIXME add support for unflattened spatial dim?

    stack_dim = 2 if interleave_sin_cos else 1  # stack sin, cos, sin, cos  instead of sin sin cos cos
    pos_emb = torch.stack([torch.sin(pos2), torch.cos(pos2)], dim=stack_dim).flatten(1)
    return pos_emb


def build_fourier_pos_embed(
        feat_shape: List[int],
        bands: Optional[torch.Tensor] = None,
        num_bands: int = 64,
        max_res: int = 224,
        linear_bands: bool = False,
        include_grid: bool = False,
        concat_out: bool = True,
        in_pixels: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(num_bands, float(max_res), linear_bands=linear_bands, dtype=dtype, device=device)
        else:
            bands = inv_freq_bands(num_bands, step=1, dtype=dtype, device=device)
    else:
        if device is None:
            device = bands.device
        if dtype is None:
            dtype = bands.dtype

    if in_pixels:
        grid = torch.stack(torch.meshgrid(
            [torch.linspace(-1., 1., steps=s, device=device, dtype=dtype) for s in feat_shape]), dim=-1)
    else:
        grid = torch.stack(torch.meshgrid(
            [torch.arange(s, device=device, dtype=dtype) for s in feat_shape]), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands

    pos_sin, pos_cos = pos.sin(), pos.cos()
    out = (grid, pos_sin, pos_cos) if include_grid else (pos_sin, pos_cos)
    # FIXME torchscript doesn't like multiple return types, probably need to always cat?
    if concat_out:
        out = torch.cat(out, dim=-1)
    return out


class FourierEmbed(nn.Module):

    def __init__(self, max_res: int = 224, num_bands: int = 64, concat_grid=True, keep_spatial=False):
        super().__init__()
        self.max_res = max_res
        self.num_bands = num_bands
        self.concat_grid = concat_grid
        self.keep_spatial = keep_spatial
        self.register_buffer('bands', pixel_freq_bands(max_res, num_bands), persistent=False)

    def forward(self, x):
        B, C = x.shape[:2]
        feat_shape = x.shape[2:]
        emb = build_fourier_pos_embed(
            feat_shape,
            self.bands,
            include_grid=self.concat_grid,
            dtype=x.dtype,
            device=x.device)
        emb = emb.transpose(-1, -2).flatten(len(feat_shape))
        batch_expand = (B,) + (-1,) * (x.ndim - 1)

        # FIXME support nD
        if self.keep_spatial:
            x = torch.cat([x, emb.unsqueeze(0).expand(batch_expand).permute(0, 3, 1, 2)], dim=1)
        else:
            x = torch.cat([x.permute(0, 2, 3, 1), emb.unsqueeze(0).expand(batch_expand)], dim=-1)
            x = x.reshape(B, feat_shape.numel(), -1)

        return x


def rot(x):
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rot_embed(x: torch.Tensor, sin_emb, cos_emb):
    return x * cos_emb + rot(x) * sin_emb


def apply_rot_embed_list(x: List[torch.Tensor], sin_emb, cos_emb):
    if isinstance(x, torch.Tensor):
        x = [x]
    return [t * cos_emb + rot(t) * sin_emb for t in x]


def apply_rot_embed_split(x: torch.Tensor, emb):
    split = emb.shape[-1] // 2
    return x * emb[:, :split] + rot(x) * emb[:, split:]


def build_rotary_pos_embed(
        feat_shape: List[int],
        bands: Optional[torch.Tensor] = None,
        dim: int = 64,
        max_freq: float = 224,
        linear_bands: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
):
    """
    NOTE: shape arg should include spatial dim only
    """
    feat_shape = torch.Size(feat_shape)
    
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape, bands=bands, num_bands=dim // 4, max_res=max_freq, linear_bands=linear_bands,
        concat_out=False, device=device, dtype=dtype)
    N = feat_shape.numel()
    sin_emb = sin_emb.reshape(N, -1).repeat_interleave(2, -1)
    cos_emb = cos_emb.reshape(N, -1).repeat_interleave(2, -1)
    return sin_emb, cos_emb


class RotaryEmbedding(nn.Module):
    """ Rotary position embedding

    NOTE: This is my initial attempt at impl rotary embedding for spatial use, it has not
    been well tested, and will likely change. It will be moved to its own file.

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """
    def __init__(self, dim, max_res=224, linear_bands: bool = False):
        super().__init__()
        self.dim = dim
        self.register_buffer('bands', pixel_freq_bands(dim // 4, max_res, linear_bands=linear_bands), persistent=False)

    def get_embed(self, shape: List[int]):
        return build_rotary_pos_embed(shape, self.bands)

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        sin_emb, cos_emb = self.get_embed(x.shape[2:])
        return apply_rot_embed(x, sin_emb, cos_emb)
