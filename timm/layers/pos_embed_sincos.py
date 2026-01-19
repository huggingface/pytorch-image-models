""" Sin-cos, fourier, rotary position embedding modules and functions

Hacked together by / Copyright 2022 Ross Wightman
"""
import math
from typing import List, Tuple, Optional, Union

import torch
from torch import nn as nn

from ._fx import register_notrace_function
from .grid import ndgrid
from .trace_utils import _assert
from .weight_init import is_meta_device


def pixel_freq_bands(
        num_bands: int,
        max_freq: float = 224.,
        linear_bands: bool = True,
        device: Optional[torch.device] = None,
):
    if linear_bands:
        bands = torch.linspace(1.0, max_freq / 2, num_bands, dtype=torch.float32, device=device)
    else:
        bands = 2 ** torch.linspace(0, math.log(max_freq, 2) - 1, num_bands, dtype=torch.float32, device=device)
    return bands * torch.pi


def freq_bands(
        num_bands: int,
        temperature: float = 10000.,
        step: int = 2,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    exp = torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(torch.float32) / num_bands
    bands = 1. / (temperature ** exp)
    return bands


def build_sincos2d_pos_embed(
        feat_shape: List[int],
        dim: int = 64,
        temperature: float = 10000.,
        reverse_coord: bool = False,
        interleave_sin_cos: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
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
    bands = freq_bands(pos_dim, temperature=temperature, step=1, device=device)

    if reverse_coord:
        feat_shape = feat_shape[::-1]  # stack W, H instead of H, W
    grid = torch.stack(ndgrid([
        torch.arange(s, device=device, dtype=torch.int64).to(torch.float32)
        for s in feat_shape
    ])).flatten(1).transpose(0, 1)
    pos2 = grid.unsqueeze(-1) * bands.unsqueeze(0)
    # FIXME add support for unflattened spatial dim?

    stack_dim = 2 if interleave_sin_cos else 1  # stack sin, cos, sin, cos  instead of sin sin cos cos
    pos_emb = torch.stack([torch.sin(pos2), torch.cos(pos2)], dim=stack_dim).flatten(1)
    return pos_emb.to(dtype=dtype)


def swap_shape_xy(seq: List[int]) -> List[int]:
    if len(seq) < 2:
        return seq
    return [seq[1], seq[0]] + list(seq[2:])


def build_fourier_pos_embed(
        feat_shape: List[int],
        bands: torch.Tensor,
        include_grid: bool = False,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        grid_offset: float = 0.,
        grid_indexing: str = 'ij',
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """

    Args:
        feat_shape: Feature shape for embedding.
        bands: Pre-calculated frequency bands.
        include_grid: Include the spatial grid in output.
        in_pixels: Output in pixel freq.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        grid_offset: Constant offset to add to grid for non-pixel freq.
        grid_indexing: Indexing mode for meshgrid ('ij' or 'xy')
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    device = device or bands.device
    dtype = dtype or bands.dtype

    if grid_indexing == 'xy':
        feat_shape = swap_shape_xy(feat_shape)
        if ref_feat_shape is not None:
            ref_feat_shape = swap_shape_xy(ref_feat_shape)

    if in_pixels:
        t = [
            torch.linspace(-1., 1., steps=s, device=device, dtype=torch.float32)
            for s in feat_shape
        ]
    else:
        t = [
            torch.arange(s, device=device, dtype=torch.int64).to(torch.float32) + grid_offset
            for s in feat_shape
        ]

    if ref_feat_shape is not None:
        # eva's scheme for resizing rope embeddings (ref shape = pretrain)
        t = [x / f * r for x, f, r in zip(t, feat_shape, ref_feat_shape)]

    grid = torch.stack(torch.meshgrid(t, indexing=grid_indexing), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands

    pos_sin, pos_cos = pos.sin().to(dtype=dtype), pos.cos().to(dtype=dtype)
    out = [grid, pos_sin, pos_cos] if include_grid else [pos_sin, pos_cos]
    return out


class FourierEmbed(nn.Module):

    def __init__(
            self,
            max_res: int = 224,
            num_bands: int = 64,
            concat_grid=True,
            keep_spatial=False,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.max_res = max_res
        self.num_bands = num_bands
        self.concat_grid = concat_grid
        self.keep_spatial = keep_spatial
        self.register_buffer('bands', torch.empty(num_bands, device=device, dtype=dtype), persistent=False)

        if not is_meta_device(device):
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters and buffers."""
        self._init_buffers()

    def _init_buffers(self) -> None:
        """Compute and fill non-persistent buffer values."""
        self.bands.copy_(pixel_freq_bands(self.num_bands, self.max_res))

    def init_non_persistent_buffers(self) -> None:
        """Initialize non-persistent buffers."""
        self._init_buffers()

    def forward(self, x):
        B, C = x.shape[:2]
        feat_shape = x.shape[2:]
        emb = build_fourier_pos_embed(
            feat_shape,
            self.bands,
            include_grid=self.concat_grid,
            dtype=x.dtype,
            device=x.device,
        )
        emb = torch.cat(emb, dim=-1)
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
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x1  x0 -x3  x2 -x5  x4]
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rot_embed(
        x: torch.Tensor,
        sin_emb: torch.Tensor,
        cos_emb: torch.Tensor,
        half: bool = False,
) -> torch.Tensor:
    # x: [..., D], eg [x0, x1, x2, x3, x4, x5]
    if half:
        # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
        # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2
        # rope_rotate_half(x): eg [-x3, -x4, -x5, x0, x1, x2]
        return x * cos_emb + rope_rotate_half(x) * sin_emb
    else:
        # sin: [..., D], eg [sin0, sin0, sin1, sin1, sin2, sin2]
        # cos: [..., D], eg [cos0, cos0, cos1, cos1, cos2, cos2]
        # rot(x): eg [-x1, x0, -x3, x2, -x5, x4]
        return x * cos_emb + rot(x) * sin_emb


def apply_rot_embed_list(
        x: List[torch.Tensor],
        sin_emb: torch.Tensor,
        cos_emb: torch.Tensor,
        half: bool = False
) -> List[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        x = [x]
    # x: [..., D], eg [x0, x1, x2, x3, x4, x5]
    if half:
        # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
        # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2
        # rope_rotate_half(x): eg [-x3, -x4, -x5, x0, x1, x2]
        return [t * cos_emb + rope_rotate_half(t) * sin_emb for t in x]
    else:
        # sin: [..., D], eg [sin0, sin0, sin1, sin1, sin2, sin2]
        # cos: [..., D], eg [cos0, cos0, cos1, cos1, cos2, cos2]
        # rot(x): eg [-x1, x0, -x3, x2, -x5, x4]
        return [t * cos_emb + rot(t) * sin_emb for t in x]


def apply_rot_embed_cat(
        x: torch.Tensor,
        emb: torch.Tensor,
        half: bool = False
) -> torch.Tensor:
    sin_emb, cos_emb = emb.chunk(2, -1)
    # x: [..., D], eg [x0, x1, x2, x3, x4, x5]
    if half:
        # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
        # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2
        # rope_rotate_half(x), eg [-x3, -x4, -x5, x0, x1, x2]
        return x * cos_emb + rope_rotate_half(x) * sin_emb
    else:
        # sin: [..., D], eg [sin0, sin0, sin1, sin1, sin2, sin2]
        # cos: [..., D], eg [cos0, cos0, cos1, cos1, cos2, cos2]
        # rot(x), eg [-x1, x0, -x3, x2, -x5, x4]
        return x * cos_emb + rot(x) * sin_emb


def apply_keep_indices_nlc(
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        keep_indices: torch.Tensor,
        pos_embed_has_batch: bool = False,
) -> torch.Tensor:
    """ Apply keep indices to different ROPE shapes

    Expected pos_embed shapes:
    * [seq_len, pos_embed_dim] --> output [batch_size, seq_len, pos_embed_dim]
    * [num_heads, seq_len, pos_embed_dim] --> output [batch_size, num_heads, seq_len, pos_embed_dim]
    * [depth, num_heads, seq_len, pos_embed_dim] --> output [batch_size, depth, num_heads, seq_len, pos_embed_dim]

    And all of the above with leading batch dimension already present if `pos_embed_has_batch == True`

    """
    if pos_embed_has_batch:
        # Pos embed already includes batch dim
        _assert(pos_embed.ndim >= 3, 'Incorrect number of dimensions')  # At least [batch, seq_len, pos_embed_dim]
    else:
        # Add batch dimension and expand to batch size
        _assert(pos_embed.ndim >= 2, 'Incorrect number of dimensions')  # At least [seq_len, pos_embed_dim]
        expand_shape = (x.shape[0],) + (-1,) * pos_embed.ndim
        pos_embed = pos_embed.unsqueeze(0).expand(expand_shape)

    # Reshape keep_indices to add singleton dims
    keep_shape = (keep_indices.shape[0],) + (1,) * (pos_embed.ndim - 3) + (keep_indices.shape[1], 1)
    keep_indices = keep_indices.view(keep_shape)

    # Expand all dims to match position embedding except the gather dim (second-last)
    keep_expand = list(pos_embed.shape)
    keep_expand[-2] = -1
    keep_indices = keep_indices.expand(keep_expand)

    return pos_embed.gather(-2, keep_indices)


def build_rotary_pos_embed(
        feat_shape: List[int],
        bands: torch.Tensor,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        grid_offset: float = 0.,
        grid_indexing: str = 'ij',
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
):
    """

    Args:
        feat_shape: Spatial shape of the target tensor for embedding.
        bands: Optional pre-generated frequency bands
        in_pixels: Pixel vs language (inv freq) mode.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        grid_offset: Constant offset to add to grid for non-pixel freq.
        grid_indexing: Indexing mode for meshgrid ('ij' or 'xy')
        device: Output device.
        dtype: Output dtype.

    Returns:

    """
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        grid_offset=grid_offset,
        grid_indexing=grid_indexing,
        device=device,
        dtype=dtype,
    )
    num_spatial_dim = 1
    # this would be much nicer as a .numel() call to torch.Size(), but torchscript sucks
    for x in feat_shape:
        num_spatial_dim *= x
    sin_emb = sin_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    return sin_emb, cos_emb


class RotaryEmbedding(nn.Module):
    """ Rotary position embedding

    NOTE: This is my initial attempt at impl rotary embedding for spatial use, it has not
    been well tested, and will likely change. It will be moved to its own file.

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
            self,
            dim,
            max_res=224,
            temperature=10000,
            in_pixels=True,
            linear_bands: bool = False,
            feat_shape: Optional[List[int]] = None,
            ref_feat_shape: Optional[List[int]] = None,
            grid_offset: float = 0.,
            grid_indexing: str = 'ij',
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.linear_bands = linear_bands
        self.in_pixels = in_pixels
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape
        self.grid_offset = grid_offset
        self.grid_indexing = grid_indexing

        # Track which mode we're in
        self._use_cached_embed = feat_shape is not None

        if feat_shape is None:
            # bands mode: cache bands, rebuild embeddings on each get_embed call
            bands_shape = (dim // 4,)
            self.register_buffer('bands', torch.empty(bands_shape, device=device, dtype=dtype), persistent=False)
            self.pos_embed_sin = None
            self.pos_embed_cos = None
        else:
            # embed mode: cache full sin/cos embeddings
            self.bands = None
            num_pos = 1
            for s in feat_shape:
                num_pos *= s
            emb_shape = (num_pos, dim)
            self.register_buffer('pos_embed_sin', torch.empty(emb_shape, device=device, dtype=dtype), persistent=False)
            self.register_buffer('pos_embed_cos', torch.empty(emb_shape, device=device, dtype=dtype), persistent=False)

        if not is_meta_device(device):
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters and buffers."""
        self._init_buffers()

    def _init_buffers(self) -> None:
        """Compute and fill non-persistent buffer values."""
        if not self._use_cached_embed:
            self.bands.copy_(self._compute_bands())
        else:
            emb_sin, emb_cos = self._get_pos_embed_values(self.feat_shape)
            self.pos_embed_sin.copy_(emb_sin)
            self.pos_embed_cos.copy_(emb_cos)

    def _compute_bands(self, device=None, dtype=None):
        """Compute frequency bands."""
        if self.in_pixels:
            bands = pixel_freq_bands(
                self.dim // 4,
                float(self.max_res),
                linear_bands=self.linear_bands,
            )
        else:
            bands = freq_bands(
                self.dim // 4,
                temperature=self.temperature,
                step=1,
            )
        return bands.to(device=device, dtype=dtype)

    def _get_pos_embed_values(self, feat_shape: List[int], device=None, dtype=torch.float32):
        bands = self._compute_bands(device, dtype)
        emb_sin, emb_cos = build_rotary_pos_embed(
            feat_shape=feat_shape,
            bands=bands,
            in_pixels=self.in_pixels,
            ref_feat_shape=self.ref_feat_shape,
            grid_offset=self.grid_offset,
            grid_indexing=self.grid_indexing,
            device=device,
            dtype=dtype,
        )
        return emb_sin, emb_cos

    def init_non_persistent_buffers(self) -> None:
        """Initialize non-persistent buffers."""
        self._init_buffers()

    def update_feat_shape(self, feat_shape: List[int]):
        if self.feat_shape is not None and feat_shape != self.feat_shape:
            # only update if feat_shape was set and different from previous value
            assert self.pos_embed_sin is not None
            assert self.pos_embed_cos is not None
            self.pos_embed_sin, self.pos_embed_cos = self._get_pos_embed_values(
                feat_shape,
                device=self.pos_embed_sin.device,
                dtype=self.pos_embed_sin.dtype,
            )
            self.feat_shape = feat_shape

    def get_embed(self, shape: Optional[List[int]] = None):
        if shape is not None and self.bands is not None:
            # rebuild embeddings every call, use if target shape changes
            return build_rotary_pos_embed(
                shape,
                self.bands,
                in_pixels=self.in_pixels,
                ref_feat_shape=self.ref_feat_shape,
                grid_offset=self.grid_offset,
                grid_indexing=self.grid_indexing,
            )
        elif self.pos_embed_sin is not None and self.pos_embed_cos is not None:
            return self.pos_embed_sin, self.pos_embed_cos
        else:
            assert False, "get_embed() requires pre-computed pos embeds or valid shape w/ pre-computed bands"

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        sin_emb, cos_emb = self.get_embed(x.shape[2:])
        return apply_rot_embed(x, sin_emb, cos_emb)


class RotaryEmbeddingCat(nn.Module):
    """ Rotary position embedding w/ concatenatd sin & cos

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
            self,
            dim: int,
            max_res: int = 224,
            temperature: float = 10000,
            in_pixels: bool = True,
            linear_bands: bool = False,
            feat_shape: Optional[List[int]] = None,
            ref_feat_shape: Optional[List[int]] = None,
            grid_offset: float = 0.,
            grid_indexing: str = 'ij',
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.linear_bands = linear_bands
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape
        self.grid_offset = grid_offset
        self.grid_indexing = grid_indexing

        # Track which mode we're in
        self._use_cached_embed = feat_shape is not None

        if feat_shape is None:
            # bands mode: cache bands, rebuild embeddings on each get_embed call
            bands_shape = (dim // 4,)
            self.register_buffer('bands', torch.empty(bands_shape, device=device, dtype=dtype), persistent=False)
            self.pos_embed = None
        else:
            # embed mode: cache full embeddings
            self.bands = None
            num_pos = 1
            for s in feat_shape:
                num_pos *= s
            emb_shape = (num_pos, dim * 2)  # concatenated sin & cos
            self.register_buffer('pos_embed', torch.empty(emb_shape, device=device, dtype=dtype), persistent=False)

        if not is_meta_device(device):
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters and buffers."""
        self._init_buffers()

    def _init_buffers(self) -> None:
        """Compute and fill non-persistent buffer values."""
        if not self._use_cached_embed:
            self.bands.copy_(self._compute_bands())
        else:
            self.pos_embed.copy_(self._get_pos_embed_values(self.feat_shape))

    def _compute_bands(self, device=None, dtype=None):
        """Compute frequency bands."""
        if self.in_pixels:
            bands = pixel_freq_bands(
                self.dim // 4,
                float(self.max_res),
                linear_bands=self.linear_bands,
            )
        else:
            bands = freq_bands(
                self.dim // 4,
                temperature=self.temperature,
                step=1,
            )
        return bands.to(device=device, dtype=dtype)

    def _get_pos_embed_values(self, feat_shape: List[int], device=None, dtype=torch.float32):
        bands = self._compute_bands(device, dtype)
        embeds = build_rotary_pos_embed(
            feat_shape=feat_shape,
            bands=bands,
            in_pixels=self.in_pixels,
            ref_feat_shape=self.ref_feat_shape,
            grid_offset=self.grid_offset,
            grid_indexing=self.grid_indexing,
            device=device,
            dtype=dtype,
        )
        return torch.cat(embeds, -1)

    def init_non_persistent_buffers(self) -> None:
        """Initialize non-persistent buffers."""
        self._init_buffers()

    def update_feat_shape(self, feat_shape: List[int]):
        if self.feat_shape is not None and feat_shape != self.feat_shape:
            # only update if feat_shape was set and different from previous value
            assert self.pos_embed is not None
            self.pos_embed = self._get_pos_embed_values(
                feat_shape,
                device=self.pos_embed.device,
                dtype=self.pos_embed.dtype,
            )
            self.feat_shape = feat_shape

    def get_embed(self, shape: Optional[List[int]] = None):
        if shape is not None and self.bands is not None:
            # rebuild embeddings from cached bands every call, use if target shape changes
            embeds = build_rotary_pos_embed(
                shape,
                self.bands,
                in_pixels=self.in_pixels,
                ref_feat_shape=self.ref_feat_shape,
                grid_offset=self.grid_offset,
                grid_indexing=self.grid_indexing,
            )
            return torch.cat(embeds, -1)
        elif self.pos_embed is not None:
            return self.pos_embed
        else:
            assert False, "get_embed() requires pre-computed pos embed or valid shape w/ pre-computed bands"

    def get_batch_embeds(
            self,
            shapes: List[Tuple[int, int]],
            seq_len: Optional[int] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate ROPE embeddings for multiple grid shapes efficiently.

        Computes embeddings for the maximum grid size once, then extracts
        and flattens the relevant portions for each requested shape.

        Args:
            shapes: List of (H, W) tuples representing different grid sizes

        Returns:
            List of concatenated sin/cos embeddings for each shape,
            where each tensor has shape (H*W, dim)
        """
        if not shapes:
            return []

        # Check if we have pre-computed bands
        if self.bands is None:
            # If we have pre-computed pos_embed for a fixed shape, we can't do batch generation
            raise RuntimeError("Batch embedding generation requires cached bands, not pre-computed embeddings")

        # Find max dimensions across all shapes
        max_h = max(h for h, w in shapes)
        max_w = max(w for h, w in shapes)

        # Generate embeddings for max size ONCE
        sin_emb, cos_emb = build_rotary_pos_embed(
            feat_shape=(max_h, max_w),
            bands=self.bands,
            in_pixels=self.in_pixels,
            ref_feat_shape=self.ref_feat_shape,
            grid_offset=self.grid_offset,
            grid_indexing=self.grid_indexing,
        )

        # sin_emb and cos_emb are (max_h * max_w, dim//2)
        # concat and reshape to 2D for slicing
        rope_embed_2d = torch.cat([sin_emb, cos_emb], dim=-1).view(max_h, max_w, -1)

        if seq_len is not None:
            flat_embeds = torch.zeros(len(shapes), seq_len, rope_embed_2d.shape[-1]).type_as(sin_emb)
            for i, (h, w) in enumerate(shapes):
                src_len = h * w
                flat_embeds[i, :src_len] = rope_embed_2d[:h, :w].reshape(src_len, -1)
            return flat_embeds
        else:
            flat_embeds_list = [rope_embed_2d[:h, :w].reshape(h * w, -1) for h, w in shapes]
            return flat_embeds_list

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        pos_embed = self.get_embed(x.shape[2:])
        return apply_rot_embed_cat(x, pos_embed)


def init_random_2d_freqs(
        head_dim: int,
        depth: int,
        num_heads: int,
        temperature: float = 10.0,
        rotate: bool = True,
        *,
        device=None,
        dtype=torch.float32,
) -> torch.Tensor:
    """ Vectorised 2D ROPE frequencies with random rotation for mixed mode ROPE.
    Returns:
         Tensor (2, depth, num_heads, head_dim//2)
    """
    # base magnitudes, shape: (head_dim//4,)
    mag = 1.0 / (temperature ** (torch.arange(0, head_dim, 4, device=device, dtype=dtype) / head_dim))

    # (1,1,L) so it broadcasts over both depth and heads
    mag = mag.unsqueeze(0).unsqueeze(0)  # (1,1,L)

    # random (or zero) rotation per head *and* per block
    if rotate:
        angles = torch.rand(depth, num_heads, 1, device=device, dtype=dtype) * 2 * torch.pi
    else:
        angles = torch.zeros(depth, num_heads, 1, device=device, dtype=dtype)

    # build (depth, num_heads, 2·L) == head_dim//2 on the last axis
    fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(angles + torch.pi / 2)], dim=-1)
    fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(angles + torch.pi / 2)], dim=-1)

    # (2, depth, num_heads, head_dim//2)
    return torch.stack([fx, fy], dim=0)


@torch.fx.wrap
@register_notrace_function
def get_mixed_grid(
        shape: List[int],
        grid_indexing: str = 'ij',
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if grid_indexing == 'xy':
        shape = swap_shape_xy(shape)
    x_pos, y_pos = torch.meshgrid(
        torch.arange(shape[0], device=device, dtype=torch.float32),
        torch.arange(shape[1], device=device, dtype=torch.float32),
        indexing=grid_indexing,
    )
    t_x = x_pos.to(dtype).flatten()
    t_y = y_pos.to(dtype).flatten()
    return t_x, t_y


def get_mixed_freqs(
        freqs: torch.Tensor,
        t_x: torch.Tensor,
        t_y: torch.Tensor,
) -> torch.Tensor:
    """Compute mixed (learnable) frequencies."""
    # Create position indices
    dtype = freqs.dtype
    freqs = freqs.float()
    freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
    freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
    combined = freqs_x + freqs_y  # shape: (num_heads, N, dim//4)
    sin_emb = torch.sin(combined).repeat_interleave(2, -1)  # (N, dim//2)
    cos_emb = torch.cos(combined).repeat_interleave(2, -1)  # (N, dim//2)
    rope_embeds = torch.cat([sin_emb, cos_emb], dim=-1)  # (num_heads, H*W, head_dim)
    return rope_embeds.to(dtype)


class RotaryEmbeddingMixed(nn.Module):
    """Rotary position embedding with depth-dependent learnable frequencies.

    This implementation supports mixed (learnable) ROPE. In mixed mode,
    each transformer block has its own set of learnable frequency parameters.

    Based on 'Rotary Position Embedding for Vision: https://arxiv.org/abs/2403.13298)'
    Compatible with original at https://github.com/naver-ai/rope-vit
    """
    def __init__(
            self,
            dim: int,
            depth: int,
            num_heads: int,
            temperature: float = 10.0,
            feat_shape: Optional[List[int]] = None,
            grid_indexing: str = 'xy',
            device=None,
            dtype=None,
    ):
        """Initialize rotary embeddings.

        Args:
            dim: Embedding dimension (should be divisible by 4)
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            temperature: Base for frequency computation
            feat_shape: Spatial dimensions [H, W] if known in advance
            grid_indexing: How to index grid positions ('xy' or 'ij')
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.temperature = temperature
        self.feat_shape = feat_shape
        self.grid_indexing = grid_indexing

        head_dim = dim // num_heads
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"

        freqs = init_random_2d_freqs(
            head_dim,
            depth,
            num_heads,
            temperature=temperature,
            rotate=True,
            device=device,
            dtype=dtype,
        )  # (2, depth, num_heads, head_dim//2)
        self.freqs = nn.Parameter(freqs)

        if feat_shape is not None:
            # cache pre-computed grid
            num_pos = 1
            for s in feat_shape:
                num_pos *= s
            self.register_buffer('t_x', torch.empty(num_pos, device=device, dtype=dtype), persistent=False)
            self.register_buffer('t_y', torch.empty(num_pos, device=device, dtype=dtype), persistent=False)
            if not is_meta_device(device):
                self._init_buffers()
        else:
            self.t_x = self.t_y = None

    def _init_buffers(self) -> None:
        """Compute and fill non-persistent buffer values."""
        if self.feat_shape is not None:
            t_x, t_y = self._get_grid_values(self.feat_shape)
            self.t_x.copy_(t_x)
            self.t_y.copy_(t_y)

    def reset_parameters(self) -> None:
        """Initialize parameters and buffers."""
        self._init_buffers()

    def _get_grid_values(self, feat_shape: Optional[List[int]]):
        t_x, t_y = get_mixed_grid(
            feat_shape,
            grid_indexing=self.grid_indexing,
            device=self.freqs.device,
        )
        return t_x, t_y

    def update_feat_shape(self, feat_shape: Optional[List[int]]):
        if self.feat_shape is not None and feat_shape != self.feat_shape:
            assert self.t_x is not None
            assert self.t_y is not None
            t_x, t_y = self._get_grid_values(feat_shape)
            self.t_x = t_x.to(self.t_x.device, self.t_x.dtype)
            self.t_y = t_y.to(self.t_y.device, self.t_y.dtype)
            self.feat_shape = feat_shape

    def init_non_persistent_buffers(self) -> None:
        """Initialize non-persistent buffers."""
        self._init_buffers()

    def get_embed(self, shape: Optional[List[int]] = None) -> torch.Tensor:
        """Generate rotary embeddings for the given spatial shape.

        Args:
            shape: Spatial dimensions [H, W]

        Returns:
            Tensor of shape (depth, H*W, dim) containing concatenated sin/cos embeddings
        """
        if shape is not None:
            t_x, t_y = get_mixed_grid(
                shape,
                grid_indexing=self.grid_indexing,
                device=self.freqs.device
            )
        elif self.t_x is not None and self.t_y is not None:
            t_x, t_y = self.t_x, self.t_y
        else:
            assert False, "get_embed() requires pre-computed t_x/t_y or valid shape"

        return get_mixed_freqs(self.freqs, t_x, t_y)

    def get_batch_embeds(
            self,
            shapes: List[Tuple[int, int]],
            seq_len: Optional[int] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate ROPE embeddings for multiple grid shapes efficiently.

        Computes embeddings for the maximum grid size once, then extracts
        and flattens the relevant portions for each requested shape.

        Args:
            shapes: List of (H, W) tuples representing different grid sizes
            seq_len: If provided, return padded tensor of this length. Otherwise return list.

        Returns:
            If seq_len is provided: Padded tensor of shape (len(shapes), depth, num_heads, seq_len, dim)
            Otherwise: List of tensors with shape (depth, num_heads, H*W, dim) for each shape
        """
        if not shapes:
            return []

        # Find max dimensions
        max_h = max(h for h, w in shapes)
        max_w = max(w for h, w in shapes)

        # Generate embeddings for max size ONCE
        t_x, t_y = get_mixed_grid(
            [max_h, max_w],
            grid_indexing=self.grid_indexing,
            device=self.freqs.device
        )
        max_embed = get_mixed_freqs(self.freqs, t_x, t_y)  # (depth, num_heads, max_h*max_w, dim)

        # Reshape to 2D grid for easy slicing
        depth, num_heads, _, dim = max_embed.shape
        max_embed_2d = max_embed.view(depth, num_heads, max_h, max_w, dim)

        if seq_len is not None:
            # Return padded tensor
            B = len(shapes)
            padded = torch.zeros(B, depth, num_heads, seq_len, dim, device=self.freqs.device, dtype=self.freqs.dtype)
            for i, (h, w) in enumerate(shapes):
                # Slice and flatten
                embed_slice = max_embed_2d[:, :, :h, :w].reshape(depth, num_heads, h * w, dim)
                actual_len = h * w
                padded[i, :, :, :actual_len] = embed_slice
            return padded
        else:
            # Return list
            results = []
            for h, w in shapes:
                # Slice and flatten
                embed_slice = max_embed_2d[:, :, :h, :w].reshape(depth, num_heads, h * w, dim)
                results.append(embed_slice)
            return results

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        pos_embed = self.get_embed(x.shape[2:])
        return apply_rot_embed_cat(x, pos_embed)

    def no_weight_decay(self):
        """Exclude frequency parameters from weight decay."""
        return {'freqs'}


@torch.fx.wrap
@register_notrace_function
def make_coords_dinov3(
        height: int,
        width: int,
        normalize_coords: str = 'separate',
        grid_indexing: str = 'ij',
        grid_offset: float = 0.,
        device: torch.device = 'cpu',
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Make coordinate grid matching offset and normalization of original.
    Returns: coords with shape (HW, 2) in [-1, 1].
    """
    # 0.5-centered indices with optional offset
    coords_h = torch.arange(0.5, height, device=device, dtype=torch.float32) + grid_offset
    coords_w = torch.arange(0.5, width, device=device, dtype=torch.float32) + grid_offset

    # Normalization denominators
    if normalize_coords == "max":
        denom = float(max(height, width))
        h_denom = denom
        w_denom = denom
    elif normalize_coords == "min":
        denom = float(min(height, width))
        h_denom = denom
        w_denom = denom
    elif normalize_coords == "separate":
        h_denom = float(height)
        w_denom = float(width)
    else:
        raise ValueError(f"Unknown normalize_coords: {normalize_coords}")

    # Normalize to [0, 1]
    coords_h = coords_h / h_denom
    coords_w = coords_w / w_denom
    coords_h = coords_h.to(dtype)
    coords_w = coords_w.to(dtype)

    # Create grid then map to [-1, 1]
    if grid_indexing == "xy":
        grid_w, grid_h = torch.meshgrid(coords_w, coords_h, indexing="xy")
        coords = torch.stack([grid_h, grid_w], dim=-1)  # (H, W, 2) -> (h, w order)
    else:
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # (H, W, 2)
    coords = coords.flatten(0, 1)  # (HW, 2)
    coords = 2.0 * coords - 1.0  # (H, W, 2) in [-1, 1]
    return coords


class RotaryEmbeddingDinoV3(nn.Module):
    """RoPE for timm DinoV3 port, numerically matching original.

    Math is aligned to original DinoV3 RopePositionEmbedding at https://github.com/facebookresearch/dinov3:
      - 0.5-centered coords normalized by H/W (or min/max), mapped to [-1,1]
      - training-time augmentations (shift/jitter/rescale)
      - periods schedule equals Rope's temperature (base) or min/max period
    """

    def __init__(
            self,
            dim: int,
            temperature: Optional[float] = 100.0,
            min_period: Optional[float] = None,
            max_period: Optional[float] = None,
            feat_shape: Optional[List[int]] = None,
            normalize_coords: str = "separate",  # 'min', 'max', 'separate'
            grid_offset: float = 0.0,
            grid_indexing: str = "ij",
            rotate_half: bool = True,
            shift_coords: Optional[float] = None,
            jitter_coords: Optional[float] = None,  # interpreted as factor J >= 1
            rescale_coords: Optional[float] = None,  # interpreted as factor R >= 1
            device=None,
            dtype=None,
    ):
        super().__init__()

        # Dimensions / output format
        self.dim = dim  # equal to head_dim for most vit applications
        self.rotate_half = rotate_half

        # Period schedule parameters
        self.temperature = float(temperature)
        self.min_period = min_period
        self.max_period = max_period

        # Coord processing + augs
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.aug_active = any([a is not None for a in [self.shift_coords, self.jitter_coords, self.rescale_coords]])

        # Grid config
        self.feat_shape = feat_shape
        self.grid_offset = grid_offset
        self.grid_indexing = grid_indexing

        # Register empty buffer for periods
        periods_shape = (dim // 4,)
        self.register_buffer("periods", torch.empty(periods_shape, device=device, dtype=dtype), persistent=False)

        if feat_shape is not None:
            # Register empty buffer for cached embeddings
            num_pos = feat_shape[0] * feat_shape[1]
            emb_shape = (num_pos, dim * 2)  # concatenated sin & cos
            self.register_buffer("pos_embed_cached", torch.empty(emb_shape, device=device, dtype=dtype), persistent=False)
        else:
            self.pos_embed_cached = None

        if not is_meta_device(device):
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters and buffers."""
        self._init_buffers()

    def _init_buffers(self) -> None:
        """Compute and fill non-persistent buffer values."""
        self.periods.copy_(self._compute_periods())
        if self.feat_shape is not None and self.pos_embed_cached is not None:
            rope_embed = self._create_embed(self.feat_shape, no_aug=True)
            self.pos_embed_cached.copy_(rope_embed)

    def _compute_periods(self, device: torch.device = 'cpu', dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Construct periods from either min/max or temperature."""
        dim = self.dim // 4

        if self.min_period is not None and self.max_period is not None:
            exponents = torch.linspace(0, 1, dim, device='cpu', dtype=torch.float32)
            periods = self.min_period * ((self.max_period / self.min_period) ** exponents)
        else:
            if self.temperature is None:
                raise ValueError("Provide either min/max periods or `temperature`.")
            exponents = 2.0 * torch.arange(dim, device='cpu', dtype=torch.float32) / (self.dim // 2)
            periods = self.temperature ** exponents

        # NOTE: The original dinv3 model weights have periods downcast to bfloat16 in persistent buffers,
        # loaded models will differ a bit vs timm as periods is not persistent and generated in float32 by default
        return periods.to(device=device, dtype=dtype)

    def _apply_coord_augs(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply shift/jitter/rescale train time augmentations."""
        if not self.training or not self.aug_active:
            return coords

        device = coords.device
        dtype = coords.dtype

        # Shift per-axis in [-s, +s]
        if self.shift_coords is not None:
            shift = float(self.shift_coords)
            shift_hw = torch.empty(2, device=device, dtype=dtype).uniform_(-shift, shift)
            coords = coords + shift_hw[None, :]

        # Jitter: per-axis log-uniform factor in [1/J, J]
        if self.jitter_coords is not None:
            jitter_factor = float(self.jitter_coords)
            if jitter_factor <= 0:
                raise ValueError("jitter_coords must be > 0 (interpreted as multiplicative factor).")
            jitter_max = math.log(jitter_factor)
            jitter_hw = torch.empty(2, device=device, dtype=dtype).uniform_(-jitter_max, jitter_max).exp()
            coords = coords * jitter_hw[None, :]

        # Rescale: shared scalar log-uniform factor in [1/R, R]
        if self.rescale_coords is not None:
            rescale_factor = float(self.rescale_coords)
            if rescale_factor <= 0:
                raise ValueError("rescale_coords must be > 0 (interpreted as multiplicative factor).")
            rescale_max = math.log(rescale_factor)
            rescale = torch.empty(1, device=device, dtype=dtype).uniform_(-rescale_max, rescale_max).exp()
            coords = coords * rescale

        return coords

    def _get_pos_embed_from_coords(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return sin/cos embeddings with either 'half' or 'interleaved' layout."""
        # coords: (HW, 2); periods: (dim)
        dim = self.dim // 4
        device = self.periods.device
        dtype = self.periods.dtype
        assert self.periods.numel() == dim

        # NOTE this is a slightly later device/dtype switch than original
        coords = coords[:, :, None].to(device=device, dtype=dtype)
        angles = 2 * math.pi * coords / self.periods[None, None, :]
        angles = angles.flatten(1)  # (HW, dim // 2)

        if self.rotate_half:
            # Tile (half layout) (HW, dim // 2) -> (HW, dim)
            angles = angles.tile(2)
        else:
            # Interleaved layout (HW, dim // 2) -> (HW, dim)
            angles = angles.repeat_interleave(2, dim=-1)

        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return sin, cos

    def _create_embed(
            self,
            feat_shape: List[int],
            no_aug: bool = False,
    ) -> torch.Tensor:
        H, W = feat_shape
        coords = make_coords_dinov3(
            H, W,
            normalize_coords=self.normalize_coords,
            grid_indexing=self.grid_indexing,
            grid_offset=self.grid_offset,
        )  # (HW, 2)
        if not no_aug:
            coords = self._apply_coord_augs(coords)
        sin, cos = self._get_pos_embed_from_coords(coords)  # 2 * (HW, dim)
        rope_embed = torch.cat([sin, cos], dim=-1)  # (HW, 2*dim)
        return rope_embed

    def _cache_embed(self, feat_shape: List[int]):
        # create non-augmented embeds for cache
        rope_embed = self._create_embed(feat_shape, no_aug=True)
        self.register_buffer("pos_embed_cached", rope_embed, persistent=False)
        self.feat_shape = feat_shape

    def update_feat_shape(self, feat_shape: List[int]):
        if self.feat_shape is not None and feat_shape != self.feat_shape:
            # only update if feat_shape was set (valid cache) and different from previous value
            self._cache_embed(feat_shape)

    def init_non_persistent_buffers(self) -> None:
        """Initialize non-persistent buffers."""
        self._init_buffers()

    def get_embed(self, shape: Optional[List[int]] = None) -> torch.Tensor:
        """Generate rope_embed matching DINOv3 RopePositionEmbedding numerics.

        Returns: (HW, num_heads, 2 * head_dim) with last dim = [sin, cos] cat.
        """
        if shape is not None:
            rope_embed = self._create_embed(shape)
        else:
            need_create = self.pos_embed_cached is None or (self.training and self.aug_active)
            if need_create:
                assert self.feat_shape is not None, 'feature shape must be cached on create'
                rope_embed = self._create_embed(self.feat_shape)
            else:
                assert self.pos_embed_cached is not None
                rope_embed = self.pos_embed_cached

        return rope_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get and apply rotary embeddings to x"""
        # assuming channel-first tensor where spatial dim are >= 2
        pos_embed = self.get_embed(x.shape[2:])
        return apply_rot_embed_cat(x, pos_embed, half=self.rotate_half)


def create_rope_embed(
        rope_type: str = 'cat',
        dim: int = 768,
        num_heads: int = 12,
        **kwargs
) -> nn.Module:
    """Factory function for creating rotary position embeddings.

    Args:
        rope_type: Type of RoPE to create. Options:
            - 'base': Basic RotaryEmbedding
            - 'cat': RotaryEmbeddingCat (concatenated sin/cos)
            - 'mixed': RotaryEmbeddingMixed (learnable per-depth frequencies)
            - 'dinov3': RotaryEmbeddingDinoV3 (with coordinate transforms)
        dim: Total embedding dimension
        num_heads: Number of attention heads
        **kwargs: Additional arguments passed to the specific RoPE class

    Returns:
        Rotary embedding module
    """
    if rope_type == 'base':
        kwargs.pop('rotate_half', None)  # doesn't support
        return RotaryEmbedding(dim=dim // num_heads, **kwargs)
    elif rope_type == 'cat':
        kwargs.pop('rotate_half', None)  # doesn't support
        return RotaryEmbeddingCat(dim=dim // num_heads, **kwargs)
    elif rope_type == 'mixed':
        # Mixed requires depth parameter, generates differing embeddings per layer and head
        kwargs.pop('in_pixels', None)  # doesn't support
        kwargs.pop('ref_feat_shape', None)  # doesn't support
        return RotaryEmbeddingMixed(dim=dim, num_heads=num_heads, **kwargs)
    elif rope_type == 'dinov3':
        kwargs.pop('in_pixels', None)  # doesn't support
        kwargs.pop('ref_feat_shape', None)  # doesn't support
        return RotaryEmbeddingDinoV3(dim=dim // num_heads, **kwargs)
    else:
        raise ValueError(f"Unknown RoPE type: {rope_type}")
