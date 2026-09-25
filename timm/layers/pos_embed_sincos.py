""" Sin-cos, fourier, rotary position embedding modules and functions

Hacked together by / Copyright 2022 Ross Wightman
"""
import math
from numbers import Real
from typing import List, Tuple, Optional, Union

import torch
from torch import nn as nn

from ._fx import register_notrace_function
from .grid import ndgrid
from .trace_utils import _assert


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


def _build_bands(
        num_bands: int,
        in_pixels: bool = True,
        max_res: int = 224,
        temperature: float = 10000.,
        linear_bands: bool = False,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build frequency bands from config.

    Args:
        num_bands: Number of frequency bands.
        in_pixels: Pixel based bands (max_res, linear_bands) if True, else temperature based inv-freq bands.
        max_res: Maximum resolution for pixel based bands.
        temperature: Temperature for inv-freq bands.
        linear_bands: Linear (instead of log) band spacing for pixel based bands.
        device: Output device.

    Returns:
        Bands with shape (num_bands,), float32.
    """
    if in_pixels:
        return pixel_freq_bands(num_bands, float(max_res), linear_bands=linear_bands, device=device)
    return freq_bands(num_bands, temperature=temperature, step=1, device=device)


@torch.fx.wrap
@register_notrace_function
def _build_grid(
        feat_shape: List[int],
        grid_type: str = 'index',
        ref_feat_shape: Optional[List[int]] = None,
        grid_offset: float = 0.,
        grid_indexing: str = 'ij',
        normalize_coords: str = 'separate',
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        ndim: Optional[int] = None,
) -> torch.Tensor:
    """Build the coordinate grid that sin-cos / rotary embeddings are computed over.

    Args:
        feat_shape: Spatial shape of the feature map, e.g. (H, W).
        grid_type: Coordinate type, one of
            'index': integer positions in [0, s) (+ grid_offset),
            'pixel': positions evenly spaced in [-1, 1],
            'centered': 0.5-centered positions (+ grid_offset) normalized to [0, 1] then mapped to [-1, 1] (DINOv3).
        ref_feat_shape: Rescale positions so that feat_shape maps onto this reference (pretrain) shape (EVA style).
        grid_offset: Constant offset added to 'index' and 'centered' positions.
        grid_indexing: 'ij' keeps coordinate channels in feat_shape order, 'xy' swaps the first two axes.
        normalize_coords: Denominator for 'centered' grids, 'separate' (each axis by its own size), 'min' or 'max'.
        device: Output device.
        dtype: Grid dtype; normalized coordinates are cast before the final mapping to [-1, 1].
        ndim: Required spatial rank, if the caller only supports a fixed number of axes.

    Returns:
        Grid with shape (*feat_shape, len(feat_shape)), float32 by default. Spatial layout always follows feat_shape,
        only the coordinate channel order changes with grid_indexing.
    """
    if ndim is not None and len(feat_shape) != ndim:
        raise ValueError(f'Expected {ndim} spatial dimensions, got {len(feat_shape)}')
    if grid_indexing == 'xy':
        feat_shape = swap_shape_xy(feat_shape)
        if ref_feat_shape is not None:
            ref_feat_shape = swap_shape_xy(ref_feat_shape)

    if grid_type == 'pixel':
        t = [
            torch.linspace(-1., 1., steps=s, device=device, dtype=torch.float32)
            for s in feat_shape
        ]
    elif grid_type == 'centered':
        if normalize_coords == 'max':
            denom = [float(max(feat_shape)) for _ in feat_shape]
        elif normalize_coords == 'min':
            denom = [float(min(feat_shape)) for _ in feat_shape]
        elif normalize_coords == 'separate':
            denom = [float(s) for s in feat_shape]
        else:
            raise ValueError(f"Unknown normalize_coords: {normalize_coords}")
        t = [
            (torch.arange(0.5, s, device=device, dtype=torch.float32) + grid_offset) / d
            for s, d in zip(feat_shape, denom)
        ]
    elif grid_type == 'index':
        t = [
            torch.arange(s, device=device, dtype=torch.int64).to(torch.float32) + grid_offset
            for s in feat_shape
        ]
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")

    if ref_feat_shape is not None:
        # eva's scheme for resizing rope embeddings (ref shape = pretrain)
        t = [x / f * r for x, f, r in zip(t, feat_shape, ref_feat_shape)]

    grid = torch.stack(torch.meshgrid(t, indexing=grid_indexing), dim=-1).to(dtype=dtype)
    if grid_type == 'centered':
        grid = 2.0 * grid - 1.0  # [0, 1] -> [-1, 1]
    return grid


def _rope_layout(x: torch.Tensor, rotate_half: bool = False) -> torch.Tensor:
    """Expand per-frequency values (..., D/2) to the (..., D) layout expected by apply_rot_embed*().

    rotate_half=False gives the interleaved layout [f0, f0, f1, f1, ...] (apply_rot_embed(half=False)),
    rotate_half=True gives the half layout [f0, f1, .., f0, f1, ..] (apply_rot_embed(half=True)).
    Pure data movement, so it can be applied to angles or to sin / cos values interchangeably.
    """
    if rotate_half:
        return torch.cat([x, x], dim=-1)
    return x.repeat_interleave(2, -1)


def _coord_aug_config(
        shift_coords: Optional[float],
        jitter_coords: Optional[float],
        rescale_coords: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
    """Normalize and validate coordinate augmentation parameters without consuming RNG."""
    if not torch.jit.is_scripting():
        for name, value in (
                ('shift_coords', shift_coords), ('jitter_coords', jitter_coords), ('rescale_coords', rescale_coords),
        ):
            if value is None:
                continue
            is_scalar_tensor = isinstance(value, torch.Tensor) and value.numel() == 1 and value.dtype != torch.bool
            if isinstance(value, bool) or not (isinstance(value, Real) or is_scalar_tensor):
                raise TypeError(f'{name} must be a real number or single-element tensor, not a boolean or string.')
    shift = float(shift_coords) if shift_coords is not None else None
    jitter = float(jitter_coords) if jitter_coords is not None else None
    rescale = float(rescale_coords) if rescale_coords is not None else None
    if shift is not None and (not math.isfinite(shift) or shift < 0):
        raise ValueError('shift_coords must be finite and >= 0.')
    if jitter is not None and (not math.isfinite(jitter) or jitter < 1):
        raise ValueError('jitter_coords must be finite and >= 1 (multiplicative factor).')
    if rescale is not None and (not math.isfinite(rescale) or rescale < 1):
        raise ValueError('rescale_coords must be finite and >= 1 (multiplicative factor).')
    return shift, jitter, rescale, shift is not None or jitter is not None or rescale is not None


@torch.fx.wrap
@register_notrace_function
def _apply_coord_augs(
        coords: torch.Tensor,
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
) -> torch.Tensor:
    """Apply one spatial transform to all positions, without modifying the input grid.

    Shift is measured in the grid's existing units. Jitter (independent per-axis scale) and rescale
    (shared scale) are dimensionless log-uniform factors. Callers gate augmentation on training mode.
    Preserve DINOv3's shift, jitter, rescale operation and RNG order, sampling on the grid's device/dtype.

    FX tracing with a constant grid shape can freeze augmentation draws. Trace in eval mode.
    """
    shift_coords, jitter_coords, rescale_coords, _ = _coord_aug_config(shift_coords, jitter_coords, rescale_coords)
    device = coords.device
    dtype = coords.dtype
    num_axes = coords.shape[-1]
    if shift_coords is not None:
        shift = float(shift_coords)
        shift_hw = torch.empty(num_axes, device=device, dtype=dtype).uniform_(-shift, shift)
        coords = coords + shift_hw[None, :]
    if jitter_coords is not None:
        jitter_factor = float(jitter_coords)
        jitter_max = math.log(jitter_factor)
        jitter_hw = torch.empty(num_axes, device=device, dtype=dtype).uniform_(-jitter_max, jitter_max).exp()
        coords = coords * jitter_hw[None, :]
    if rescale_coords is not None:
        rescale_factor = float(rescale_coords)
        rescale_max = math.log(rescale_factor)
        rescale = torch.empty(1, device=device, dtype=dtype).uniform_(-rescale_max, rescale_max).exp()
        coords = coords * rescale
    return coords


@torch.fx.wrap
@register_notrace_function
def _freq_buffer_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    """Frequency (bands / periods) and grid position buffers are kept at >= float32 precision.

    Frequencies are multiplied by (possibly large) positions before sin / cos, so quantizing either to a low precision
    dtype compounds with position. Output embeddings are cast once, at the end, instead.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if dtype.is_floating_point and torch.finfo(dtype).bits < 32:
        return torch.float32
    return dtype


def _pad_rope_embeds(embeds: List[torch.Tensor], seq_len: int) -> torch.Tensor:
    """Pad embeddings (..., N, D) and stack them along a new batch dimension."""
    shape = (len(embeds),) + embeds[0].shape[:-2] + (seq_len, embeds[0].shape[-1])
    padded = embeds[0].new_zeros(shape)
    for i, embed in enumerate(embeds):
        if embed.shape[-2] > seq_len:
            raise ValueError('seq_len must be at least the number of positions in each embedding')
        padded[i, ..., :embed.shape[-2], :] = embed
    return padded


def build_fourier_pos_embed(
        feat_shape: List[int],
        bands: Optional[torch.Tensor] = None,
        num_bands: int = 64,
        max_res: int = 224,
        temperature: float = 10000.,
        linear_bands: bool = False,
        include_grid: bool = False,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        grid_offset: float = 0.,
        grid_indexing: str = 'ij',
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        grid_type: Optional[str] = None,
        normalize_coords: str = 'separate',
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
) -> List[torch.Tensor]:
    """

    Args:
        feat_shape: Feature shape for embedding.
        bands: Pre-calculated frequency bands.
        num_bands: Number of frequency bands (determines output dim).
        max_res: Maximum resolution for pixel based freq.
        temperature: Temperature for non-pixel freq.
        linear_bands: Linear band spacing for pixel based freq.
        include_grid: Include the spatial grid in output.
        in_pixels: Output in pixel freq.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        grid_offset: Constant offset to add to grid for non-pixel freq.
        grid_indexing: Indexing mode for meshgrid ('ij' or 'xy')
        dtype: Output dtype.
        device: Output device.
        grid_type: Grid coordinate type ('index', 'pixel', 'centered'), overrides the in_pixels based grid selection
            when set (in_pixels still selects the band type if bands are not given). See _build_grid().
        normalize_coords: Normalization for 'centered' grids ('separate', 'min', 'max').
        shift_coords: Optional random per-axis shift in the grid's units, applied when provided.
        jitter_coords: Optional log-uniform per-axis scale in [1/J, J], applied when provided.
        rescale_coords: Optional log-uniform shared scale in [1/R, R], applied when provided.
            These functional arguments are explicit; modules only pass them during augmented training.

    Returns:

    """
    if bands is None:
        bands = _build_bands(
            num_bands,
            in_pixels=in_pixels,
            max_res=max_res,
            temperature=temperature,
            linear_bands=linear_bands,
            device=device,
        )
    else:
        if device is None:
            device = bands.device
        if dtype is None:
            dtype = bands.dtype

    grid_type_: str = 'pixel' if in_pixels else 'index'
    if grid_type is not None:
        grid_type_ = grid_type
    grid = _build_grid(
        feat_shape,
        grid_type=grid_type_,
        ref_feat_shape=ref_feat_shape,
        grid_offset=grid_offset,
        grid_indexing=grid_indexing,
        normalize_coords=normalize_coords,
        device=device,
    )
    if shift_coords is not None or jitter_coords is not None or rescale_coords is not None:
        grid = _apply_coord_augs(grid, shift_coords, jitter_coords, rescale_coords)
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

        # TODO: skip init when on meta device when safe to do so
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
    """Apply rotary embeddings and return the result in x's dtype.

    Multiplication and addition use normal dtype promotion, preserving higher-precision sin/cos until
    the completed rotation is cast back to x's dtype. Already low-precision sin/cos are not upcast.
    """
    # x: [..., D], eg [x0, x1, x2, x3, x4, x5]
    if half:
        # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
        # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
        # rope_rotate_half(x): eg [-x3, -x4, -x5, x0, x1, x2]
        out = x * cos_emb + rope_rotate_half(x) * sin_emb
    else:
        # sin: [..., D], eg [sin0, sin0, sin1, sin1, sin2, sin2]
        # cos: [..., D], eg [cos0, cos0, cos1, cos1, cos2, cos2]
        # rot(x): eg [-x1, x0, -x3, x2, -x5, x4]
        out = x * cos_emb + rot(x) * sin_emb
    return out.to(dtype=x.dtype)


def apply_rot_embed_list(
        x: List[torch.Tensor],
        sin_emb: torch.Tensor,
        cos_emb: torch.Tensor,
        half: bool = False
) -> List[torch.Tensor]:
    """Apply rotary embeddings to each input, preserving each input's dtype after the rotation."""
    if isinstance(x, torch.Tensor):
        x = [x]
    return [apply_rot_embed(t, sin_emb, cos_emb, half=half) for t in x]


def apply_rot_embed_cat(
        x: torch.Tensor,
        emb: torch.Tensor,
        half: bool = False
) -> torch.Tensor:
    """Apply concatenated sin/cos embeddings, casting only the completed rotation to x's dtype."""
    sin_emb, cos_emb = emb.chunk(2, -1)
    return apply_rot_embed(x, sin_emb, cos_emb, half=half)


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
        bands: Optional[torch.Tensor] = None,
        dim: int = 64,
        max_res: int = 224,
        temperature: float = 10000.,
        linear_bands: bool = False,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        grid_offset: float = 0.,
        grid_indexing: str = 'ij',
        rotate_half: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        grid_type: Optional[str] = None,
        normalize_coords: str = 'separate',
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
):
    """

    Args:
        feat_shape: Spatial shape of the target tensor for embedding.
        bands: Optional pre-generated frequency bands
        dim: Output dimension of embedding tensor.
        max_res: Maximum resolution for pixel mode.
        temperature: Temperature (inv freq) for non-pixel mode
        linear_bands: Linearly (instead of log) spaced bands for pixel mode
        in_pixels: Pixel vs language (inv freq) mode.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        grid_offset: Constant offset to add to grid for non-pixel freq.
        grid_indexing: Indexing mode for meshgrid ('ij' or 'xy')
        rotate_half: Emit the 'half' layout ([f0, f1, .., f0, f1, ..], for apply_rot_embed(half=True)) instead
            of the default interleaved layout ([f0, f0, f1, f1, ..], for apply_rot_embed(half=False)).
        device: Output device.
        dtype: Output dtype.
        grid_type: Grid coordinate type ('index', 'pixel', 'centered'), overrides the in_pixels based grid selection
            when set (in_pixels still selects the band type if bands are not given). See _build_grid().
        normalize_coords: Normalization for 'centered' grids ('separate', 'min', 'max').
        shift_coords: Optional random per-axis shift in the grid's units, applied when provided.
        jitter_coords: Optional log-uniform per-axis scale in [1/J, J], applied when provided.
        rescale_coords: Optional log-uniform shared scale in [1/R, R], applied when provided.

    Returns:

    """
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // 4,
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        grid_offset=grid_offset,
        grid_indexing=grid_indexing,
        device=device,
        dtype=dtype,
        grid_type=grid_type,
        normalize_coords=normalize_coords,
        shift_coords=shift_coords,
        jitter_coords=jitter_coords,
        rescale_coords=rescale_coords,
    )
    # (*feat_shape, ndim, num_bands) -> (N, ndim * num_bands) via view ops (no python loop over feat_shape, fx friendly)
    sin_emb = _rope_layout(sin_emb.flatten(-2).flatten(0, -2), rotate_half)
    cos_emb = _rope_layout(cos_emb.flatten(-2).flatten(0, -2), rotate_half)
    return sin_emb, cos_emb


class _RotaryEmbeddingBase(nn.Module):
    """Shared configuration, frequency bands, and cache lifecycle for axial RoPE.

    Subclasses own cache allocation, storage, and the tuple or concatenated embedding API.
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
            rotate_half: bool = False,
            grid_type: Optional[str] = None,
            normalize_coords: str = 'separate',
            device=None,
            dtype=None,
            shift_coords: Optional[float] = None,
            jitter_coords: Optional[float] = None,
            rescale_coords: Optional[float] = None,
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
        self.rotate_half = rotate_half
        # Coordinate grid selection is independent of the band type selected by in_pixels.
        self.grid_type = grid_type if grid_type is not None else ('pixel' if in_pixels else 'index')
        self.normalize_coords = normalize_coords
        self.shift_coords, self.jitter_coords, self.rescale_coords, self.aug_active = _coord_aug_config(
            shift_coords, jitter_coords, rescale_coords,
        )
        self._use_cached_embed = feat_shape is not None
        # Keep bands in both modes, including when augmentation is enabled after construction.
        self.register_buffer(
            'bands', torch.empty(dim // 4, device=device, dtype=_freq_buffer_dtype(dtype)), persistent=False,
        )

        self._create_cache(math.prod(feat_shape) if feat_shape is not None else None, device, dtype)
        # TODO: skip init when on meta device when safe to do so
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters and buffers."""
        self._init_buffers()

    def _init_buffers(self) -> None:
        # Initial embeddings use CPU bands before copying, preserving the existing cache numerics.
        bands = self._compute_bands()
        self.bands.copy_(bands)
        if self._use_cached_embed:
            self._init_cached_embed(bands)

    def init_non_persistent_buffers(self) -> None:
        """Initialize non-persistent buffers."""
        self._init_buffers()

    def update_feat_shape(self, feat_shape: List[int]) -> None:
        if self.feat_shape is not None and feat_shape != self.feat_shape:
            bands = self._compute_bands(device=self.bands.device)
            self.bands.copy_(bands)
            self._update_cached_embed(feat_shape, bands)
            self.feat_shape = feat_shape

    def _apply(self, fn, *args, **kwargs):
        module = super()._apply(fn, *args, **kwargs)
        # .to(dtype) / .half() / .bfloat16() cast all floating buffers, rebuild bands if they were downcast
        if _freq_buffer_dtype(self.bands.dtype) != self.bands.dtype:
            self.register_buffer('bands', self._compute_bands(device=self.bands.device), persistent=False)
        return module

    def _compute_bands(
            self,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Compute bands on CPU before copying, preserving existing init / resize / dtype-conversion numerics."""
        bands = _build_bands(
            self.dim // 4,
            in_pixels=self.in_pixels,
            max_res=self.max_res,
            temperature=float(self.temperature),
            linear_bands=self.linear_bands,
        )
        return bands.to(device=device, dtype=dtype)

    def _build_embed(
            self,
            feat_shape: List[int],
            bands: torch.Tensor,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            augment: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build sin & cos embeddings (N, dim), casting once after sin/cos."""
        if device is not None:
            bands = bands.to(device=device)
        if dtype is None and augment:
            dtype = bands.dtype
        return build_rotary_pos_embed(
            feat_shape,
            bands=bands,
            in_pixels=self.in_pixels,
            ref_feat_shape=self.ref_feat_shape,
            grid_offset=self.grid_offset,
            grid_indexing=self.grid_indexing,
            rotate_half=self.rotate_half,
            device=device,
            dtype=torch.float32 if dtype is None else dtype,
            grid_type=self.grid_type,
            normalize_coords=self.normalize_coords,
            shift_coords=self.shift_coords if augment else None,
            jitter_coords=self.jitter_coords if augment else None,
            rescale_coords=self.rescale_coords if augment else None,
        )


class RotaryEmbedding(_RotaryEmbeddingBase):
    """Axial rotary position embeddings with separate sine and cosine outputs.

    Set rotate_half=True for the half layout instead of interleaved pairs. When applying get_embed()
    outputs directly, pass the same setting as half to apply_rot_embed().

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/

    Optional shift_coords, jitter_coords and rescale_coords augment coordinates only during training.
    Shift uses the existing grid's units; jitter and rescale are multiplicative factors. One transform
    is shared by all positions per get_embed() call. Cached embeddings remain unaugmented.
    FX tracing with a constant grid shape can freeze augmentation draws. Trace in eval mode.
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
            grid_type: Optional[str] = None,
            normalize_coords: str = 'separate',
            device=None,
            dtype=None,
            shift_coords: Optional[float] = None,
            jitter_coords: Optional[float] = None,
            rescale_coords: Optional[float] = None,
            rotate_half: bool = False,
    ):
        super().__init__(
            dim=dim, max_res=max_res, temperature=temperature, in_pixels=in_pixels, linear_bands=linear_bands,
            feat_shape=feat_shape, ref_feat_shape=ref_feat_shape, grid_offset=grid_offset, grid_indexing=grid_indexing,
            rotate_half=rotate_half, grid_type=grid_type, normalize_coords=normalize_coords, device=device, dtype=dtype,
            shift_coords=shift_coords, jitter_coords=jitter_coords, rescale_coords=rescale_coords,
        )

    def _create_cache(self, num_pos: Optional[int], device, dtype) -> None:
        if num_pos is None:
            self.pos_embed_sin = None
            self.pos_embed_cos = None
        else:
            emb_shape = (num_pos, self.dim)
            self.register_buffer('pos_embed_sin', torch.empty(emb_shape, device=device, dtype=dtype), persistent=False)
            self.register_buffer('pos_embed_cos', torch.empty(emb_shape, device=device, dtype=dtype), persistent=False)

    def _init_cached_embed(self, bands: torch.Tensor) -> None:
        emb_sin, emb_cos = self._build_embed(self.feat_shape, bands)
        self.pos_embed_sin.copy_(emb_sin)
        self.pos_embed_cos.copy_(emb_cos)

    def _update_cached_embed(self, feat_shape: List[int], bands: torch.Tensor) -> None:
        assert self.pos_embed_sin is not None
        assert self.pos_embed_cos is not None
        self.pos_embed_sin, self.pos_embed_cos = self._build_embed(feat_shape, bands, dtype=self.pos_embed_sin.dtype)

    def get_embed(
            self,
            shape: Optional[List[int]] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get (sin, cos) embeddings, each with shape (N, dim).

        Args:
            shape: Spatial shape to build embeddings for (bands mode), ignored in cached embed mode.
            dtype: Output dtype; unaugmented generation defaults to float32, augmented generation to the
                frequency buffer dtype (at least float32), and unaugmented cached output to the cache dtype.
        """
        if not self._use_cached_embed and shape is not None:
            # rebuild embeddings every call, use if target shape changes
            return self._build_embed(shape, self.bands, dtype=dtype, augment=self.training and self.aug_active)
        elif self.pos_embed_sin is not None and self.pos_embed_cos is not None:
            if self.training and self.aug_active:
                assert self.feat_shape is not None
                return self._build_embed(
                    self.feat_shape,
                    self.bands,
                    dtype=dtype,
                    augment=True,
                )
            if dtype is not None:
                return self.pos_embed_sin.to(dtype=dtype), self.pos_embed_cos.to(dtype=dtype)
            return self.pos_embed_sin, self.pos_embed_cos
        else:
            assert False, "get_embed() requires pre-computed pos embeds or valid shape w/ pre-computed bands"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to a channels-first tensor (B, C, *spatial) with C == dim."""
        shape = x.shape
        sin_emb, cos_emb = self.get_embed(shape[2:])
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x_flat = apply_rot_embed(x_flat, sin_emb, cos_emb, half=self.rotate_half)
        return x_flat.transpose(1, 2).reshape(shape)


class RotaryEmbeddingCat(_RotaryEmbeddingBase):
    """Axial rotary position embeddings with concatenated sine and cosine outputs.

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/

    Coordinate augmentation follows RotaryEmbedding: opt-in, training-only, in the grid's existing units.
    FX tracing with a constant grid shape can freeze augmentation draws. Trace in eval mode.
    """

    def _create_cache(self, num_pos: Optional[int], device, dtype) -> None:
        if num_pos is None:
            self.pos_embed = None
        else:
            emb_shape = (num_pos, self.dim * 2)
            self.register_buffer('pos_embed', torch.empty(emb_shape, device=device, dtype=dtype), persistent=False)

    def _init_cached_embed(self, bands: torch.Tensor) -> None:
        self.pos_embed.copy_(self._build_embed_cat(self.feat_shape, bands))

    def _update_cached_embed(self, feat_shape: List[int], bands: torch.Tensor) -> None:
        assert self.pos_embed is not None
        self.pos_embed = self._build_embed_cat(feat_shape, bands, dtype=self.pos_embed.dtype)

    def _build_embed_cat(
            self,
            feat_shape: List[int],
            bands: torch.Tensor,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            augment: bool = False,
    ) -> torch.Tensor:
        sin_emb, cos_emb = self._build_embed(
            feat_shape, bands, device=device, dtype=dtype, augment=augment,
        )
        return torch.cat([sin_emb, cos_emb], -1)

    def get_embed(
            self,
            shape: Optional[List[int]] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Get concatenated sin & cos embedding with shape (N, 2 * dim).

        Args:
            shape: Spatial shape to build embeddings for (bands mode), ignored in cached embed mode.
            dtype: Output dtype; unaugmented generation defaults to float32, augmented generation to the
                frequency buffer dtype (at least float32), and unaugmented cached output to the cache dtype.
        """
        if not self._use_cached_embed and shape is not None:
            # rebuild embeddings from cached bands every call, use if target shape changes
            return self._build_embed_cat(shape, self.bands, dtype=dtype, augment=self.training and self.aug_active)
        elif self.pos_embed is not None:
            if self.training and self.aug_active:
                assert self.feat_shape is not None
                return self._build_embed_cat(
                    self.feat_shape,
                    self.bands,
                    dtype=dtype,
                    augment=True,
                )
            if dtype is not None:
                return self.pos_embed.to(dtype=dtype)
            return self.pos_embed
        else:
            assert False, "get_embed() requires pre-computed pos embed or valid shape w/ pre-computed bands"

    def get_batch_embeds(
            self,
            shapes: List[Tuple[int, int]],
            seq_len: Optional[int] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate ROPE embeddings for multiple grid shapes efficiently.

        For shape independent grids ('index' grid w/o ref_feat_shape rescale) the embedding is computed once for the
        max grid size and sliced for each requested shape. Otherwise ('pixel' / 'centered' grids or ref_feat_shape
        set, where coordinates depend on the grid size) each shape is computed separately.
        Augmented training samples one transform per shapes entry, matching DINOv3 batch generation.

        Args:
            shapes: List of (H, W) tuples representing different grid sizes
            seq_len: If provided, return padded tensor of this length. Otherwise return list.
            dtype: Output dtype; None uses float32 for unaugmented generation and the frequency buffer
                dtype (at least float32) during augmented training.

        Returns:
            If seq_len is provided: Padded tensor of shape (len(shapes), seq_len, 2 * dim)
            Otherwise: List of concatenated sin/cos embeddings, each with shape (H * W, 2 * dim)
        """
        if not shapes:
            return []

        if self._use_cached_embed:
            # If we have pre-computed pos_embed for a fixed shape, we can't do batch generation
            raise RuntimeError("Batch embedding generation requires cached bands, not pre-computed embeddings")

        if self.grid_type == 'index' and self.ref_feat_shape is None and not (self.training and self.aug_active):
            # Generate embeddings for max size ONCE, reshape to 2D and slice each shape
            max_h = max(h for h, w in shapes)
            max_w = max(w for h, w in shapes)
            rope_embed_2d = self._build_embed_cat((max_h, max_w), self.bands, dtype=dtype).view(max_h, max_w, -1)
            embeds = [rope_embed_2d[:h, :w].reshape(h * w, -1) for h, w in shapes]
        else:
            embeds = [self.get_embed((h, w), dtype=dtype) for h, w in shapes]

        if seq_len is not None:
            return _pad_rope_embeds(embeds, seq_len)
        return embeds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to a channels-first tensor (B, C, *spatial) with C == dim."""
        shape = x.shape
        pos_embed = self.get_embed(shape[2:])
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x_flat = apply_rot_embed_cat(x_flat, pos_embed, half=self.rotate_half)
        return x_flat.transpose(1, 2).reshape(shape)


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
    """Return independent, contiguous coordinate vectors for a two-dimensional (H, W) grid."""
    grid = _build_grid(shape, grid_indexing=grid_indexing, device=device, ndim=2).reshape(-1, 2).to(dtype=dtype)
    return (
        grid[:, 0].clone(memory_format=torch.contiguous_format),
        grid[:, 1].clone(memory_format=torch.contiguous_format),
    )


def get_mixed_freqs(
        freqs: torch.Tensor,
        t_x: torch.Tensor,
        t_y: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Build (..., N, 2 * head_dim) embeddings from frequencies (2, ..., head_dim // 2).

    Compute angles and sin/cos in float32, including under autocast. By default the output uses the freqs dtype.
    """
    if dtype is None:
        dtype = freqs.dtype
    freqs = freqs.float()
    t_x = t_x.float()
    t_y = t_y.float()
    # These are outer products; elementwise multiplication avoids autocast's matmul downcast.
    freqs_x = t_x.unsqueeze(-1) * freqs[0].unsqueeze(-2)
    freqs_y = t_y.unsqueeze(-1) * freqs[1].unsqueeze(-2)
    combined = freqs_x + freqs_y  # (..., N, head_dim // 2)
    sin_emb = _rope_layout(combined.sin())
    cos_emb = _rope_layout(combined.cos())
    rope_embeds = torch.cat([sin_emb, cos_emb], dim=-1)
    return rope_embeds.to(dtype)


class RotaryEmbeddingMRope(nn.Module):
    """Interleaved multimodal RoPE (Qwen2-VL style) for vision, matching the reference GenLIP layout.

    Drop-in sibling of ``RotaryEmbeddingCat``: ``get_embed(shape) -> [N, 2*dim]``, consumed by
    ``apply_rot_embed_cat(..., half=True)`` (no new apply path / no separate sin/cos tensors). The ``dim // 2``
    frequency channels are assigned to height/width/temporal axes in a strided ``T,H,W,T,H,W,...`` interleave
    (the reference ``apply_interleaved_mrope``): channels ``1,4,7,...`` -> height, ``2,5,8,...`` -> width, and
    the remainder -> temporal. ``mrope_section`` sets the strided extent per axis; the actual per-axis channel
    *counts* equal ``mrope_section`` only for the standard equal-section configs that tile exactly
    (``3*section == dim // 2``, e.g. ``(12,12,12)`` -> 36 channels = 12/12/12), and are otherwise the clamped
    interleave (e.g. ``dim=64, (8,12,12)`` -> 11/11/10) -- this matches the reference, which also clamps.

    For an image encoder there is no text, so the temporal channels sit at position 0 (inert) and this reduces
    to a 2-axis ``(h, w)`` rope -- numerically identical to a checkpoint trained with the reference MRoPE.

    Only ``grid_indexing='ij'`` is supported (GenLIP / NaFlex ``(y, x)`` row-major patch order); ``'xy'`` would
    require mirroring the timm axial shape-swap and is intentionally not implemented here.

    Optional training-only coordinate augmentation transforms height/width in patch-index units,
    keeping temporal positions at zero. One transform is sampled per get_embed() call.
    FX tracing with a constant grid shape can freeze augmentation draws. Trace in eval mode.
    """

    def __init__(
            self,
            dim: int,
            mrope_section: Tuple[int, int, int] = (8, 12, 12),
            temperature: float = 10000.,
            grid_indexing: str = 'ij',
            device=None,
            dtype=None,
            shift_coords: Optional[float] = None,
            jitter_coords: Optional[float] = None,
            rescale_coords: Optional[float] = None,
    ):
        super().__init__()
        assert dim % 2 == 0, 'dim (head_dim) must be even'
        # Interleaved layout: H/W sections set the strided extent, temporal is the remainder -- so unlike
        # chunked MRoPE, sum(mrope_section) need not equal dim // 2 (e.g. GenLIP L/16 (8,12,12), head_dim 72).
        assert all(s >= 0 for s in mrope_section), \
            f"mrope_section entries must be non-negative, got {mrope_section}"
        assert grid_indexing == 'ij', \
            "RotaryEmbeddingMRope supports grid_indexing='ij' only (GenLIP/NaFlex (y,x) patch order)."
        self.dim = dim
        self.mrope_section = mrope_section
        self.temperature = temperature
        self.grid_indexing = grid_indexing
        self.shift_coords, self.jitter_coords, self.rescale_coords, self.aug_active = _coord_aug_config(
            shift_coords, jitter_coords, rescale_coords,
        )

        self.register_buffer(
            'inv_freq', torch.empty(dim // 2, device=device, dtype=_freq_buffer_dtype(dtype)), persistent=False,
        )
        self.register_buffer('axis', torch.empty(dim // 2, device=device, dtype=torch.long), persistent=False)
        self.reset_parameters()

    def _compute_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        # Theta-style frequencies, one per channel (the same vector for every axis, as in Qwen2-VL MRoPE).
        return freq_bands(self.dim, temperature=float(self.temperature), step=2, device=device)

    def reset_parameters(self) -> None:
        self.init_non_persistent_buffers()

    def init_non_persistent_buffers(self) -> None:
        """Rebuild frequencies and axis assignments, including after materializing a meta module."""
        self.inv_freq.copy_(self._compute_inv_freq(device=self.inv_freq.device))

        # axis id per channel over dim//2: 0 = temporal (inert for images), 1 = height, 2 = width.
        # Slice assignment clamps the stop to dim//2, matching the reference `slice(offset, section*3, 3)`.
        # The temporal section is the leftover remainder (sec_t is nominal, not used directly).
        _sec_t, sec_h, sec_w = self.mrope_section
        self.axis.zero_()  # default temporal
        self.axis[1:sec_h * 3:3] = 1  # H at channels {1, 4, 7, ...}
        self.axis[2:sec_w * 3:3] = 2  # W at channels {2, 5, 8, ...}

    def _apply(self, fn, *args, **kwargs):
        module = super()._apply(fn, *args, **kwargs)
        if _freq_buffer_dtype(self.inv_freq.dtype) != self.inv_freq.dtype:
            self.register_buffer('inv_freq', self._compute_inv_freq(device=self.inv_freq.device), persistent=False)
        return module

    def get_embed(self, shape: List[int], dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Args:
            shape: ``(H, W)`` patch grid.
            dtype: Output dtype; None uses the frequency buffer dtype (float32 by default).

        Returns:
            Rope tensor of shape ``[H*W, 2*dim]`` for ``apply_rot_embed_cat(..., half=True)``.
        """
        grid = _build_grid(shape, grid_indexing=self.grid_indexing, device=self.inv_freq.device, ndim=2).reshape(-1, 2)
        if self.training and self.aug_active:
            grid = _apply_coord_augs(grid, self.shift_coords, self.jitter_coords, self.rescale_coords)
        pos = torch.cat([torch.zeros_like(grid[:, :1]), grid], dim=-1).index_select(-1, self.axis)
        # temporal channels keep pos=0 -> angle 0 -> cos=1, sin=0 -> identity (inert)

        angles = pos * self.inv_freq  # [N, dim//2]
        sin = _rope_layout(angles.sin(), rotate_half=True)
        cos = _rope_layout(angles.cos(), rotate_half=True)
        return torch.cat([sin, cos], dim=-1).to(dtype=dtype)  # [N, 2*dim]


class RotaryEmbeddingMixed(nn.Module):
    """Rotary position embedding with depth-dependent learnable frequencies.

    This implementation supports mixed (learnable) ROPE. In mixed mode,
    each transformer block has its own set of learnable frequency parameters.

    Optional training-only coordinate augmentation uses patch-index units, with one transform shared
    across all positions, heads and blocks per get_embed() call. Cached grid positions are never modified.
    FX tracing with a constant grid shape can freeze augmentation draws. Trace in eval mode.

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
            shift_coords: Optional[float] = None,
            jitter_coords: Optional[float] = None,
            rescale_coords: Optional[float] = None,
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
        if feat_shape is not None and len(feat_shape) != 2:
            raise ValueError(f'Expected 2 spatial dimensions, got {len(feat_shape)}')
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.temperature = temperature
        self.feat_shape = feat_shape
        self.grid_indexing = grid_indexing
        self.shift_coords, self.jitter_coords, self.rescale_coords, self.aug_active = _coord_aug_config(
            shift_coords, jitter_coords, rescale_coords,
        )

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
            grid_dtype = _freq_buffer_dtype(dtype)
            self.register_buffer('t_x', torch.empty(num_pos, device=device, dtype=grid_dtype), persistent=False)
            self.register_buffer('t_y', torch.empty(num_pos, device=device, dtype=grid_dtype), persistent=False)
            # TODO: skip init when on meta device when safe to do so
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

    def _apply(self, fn, *args, **kwargs):
        module = super()._apply(fn, *args, **kwargs)
        # .to(dtype) / .half() / .bfloat16() cast all floating buffers, rebuild grid positions if they were downcast
        if self.t_x is not None and _freq_buffer_dtype(self.t_x.dtype) != self.t_x.dtype:
            t_x, t_y = self._get_grid_values(self.feat_shape)
            self.register_buffer('t_x', t_x.to(device=self.t_x.device), persistent=False)
            self.register_buffer('t_y', t_y.to(device=self.t_y.device), persistent=False)
        return module

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

    def get_embed(
            self,
            shape: Optional[List[int]] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Generate rotary embeddings for the given spatial shape.

        Args:
            shape: Spatial dimensions [H, W]
            dtype: Output dtype; None uses the frequency parameter dtype, promoted to at least float32
                during augmented training. Angle and sin/cos computation remains float32.

        Returns:
            Tensor of shape (depth, num_heads, H * W, 2 * head_dim), with head_dim = dim // num_heads.
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

        if self.training and self.aug_active:
            coords = _apply_coord_augs(
                torch.stack([t_x, t_y], dim=-1).float(),
                self.shift_coords, self.jitter_coords, self.rescale_coords,
            )
            t_x, t_y = coords[:, 0], coords[:, 1]
            if dtype is None:
                if torch.jit.is_scripting():
                    dtype = torch.float64 if self.freqs.dtype == torch.float64 else torch.float32
                else:
                    dtype = _freq_buffer_dtype(self.freqs.dtype)
        return get_mixed_freqs(self.freqs, t_x, t_y, dtype=dtype)

    def get_batch_embeds(
            self,
            shapes: List[Tuple[int, int]],
            seq_len: Optional[int] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate ROPE embeddings for multiple grid shapes efficiently.

        Computes embeddings for the maximum grid size once, then extracts
        and flattens the relevant portions for each requested shape.
        Augmented training instead samples one transform per shapes entry, shared by all heads/blocks.

        Args:
            shapes: List of (H, W) tuples representing different grid sizes
            seq_len: If provided, return padded tensor of this length. Otherwise return list.
            dtype: Output dtype; None uses the frequency parameter dtype, promoted to at least float32
                during augmented training. Angle and sin/cos computation remains float32.

        Returns:
            If seq_len is provided: Tensor of shape (len(shapes), depth, num_heads, seq_len, 2 * head_dim).
            Otherwise: List of tensors with shape (depth, num_heads, H * W, 2 * head_dim).
        """
        if not shapes:
            return []

        if self.training and self.aug_active:
            embeds = [self.get_embed((h, w), dtype=dtype) for h, w in shapes]
        else:
            # Generate embeddings for the maximum grid once, then slice each requested shape.
            max_h = max(h for h, w in shapes)
            max_w = max(w for h, w in shapes)
            t_x, t_y = get_mixed_grid(
                [max_h, max_w],
                grid_indexing=self.grid_indexing,
                device=self.freqs.device,
            )
            max_embed = get_mixed_freqs(self.freqs, t_x, t_y, dtype=dtype)
            depth, num_heads, _, dim = max_embed.shape
            max_embed_2d = max_embed.view(depth, num_heads, max_h, max_w, dim)
            embeds = [max_embed_2d[:, :, :h, :w].reshape(depth, num_heads, h * w, dim) for h, w in shapes]
        if seq_len is not None:
            return _pad_rope_embeds(embeds, seq_len)
        return embeds

    def forward(self, x: torch.Tensor, block_index: int = 0) -> torch.Tensor:
        """Apply one block's frequencies to (B, dim, H, W), splitting channels into attention heads.

        For attention tensors (B, num_heads, N, head_dim), use
        apply_rot_embed_cat(x, self.get_embed(shape)[block_index]) directly.
        """
        _assert(x.ndim == 4, 'Mixed RoPE forward expects a (B, dim, H, W) tensor')
        _assert(x.shape[1] == self.dim, 'Mixed RoPE forward expects dim channels')
        _assert(block_index >= 0, 'block_index must be in [0, depth)')
        _assert(block_index < self.depth, 'block_index must be in [0, depth)')
        shape = x.shape
        pos_embed = self.get_embed(shape[2:])[block_index]
        x_heads = x.reshape(shape[0], self.num_heads, self.dim // self.num_heads, -1).transpose(-2, -1)
        x_heads = apply_rot_embed_cat(x_heads, pos_embed)
        return x_heads.transpose(-2, -1).reshape(shape)

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
    """Make centered, normalized coordinates (H * W, 2), matching RotaryEmbeddingDinoV3.

    Spatial positions stay in row-major order. Coordinate channels are (height, width) for 'ij',
    (width, height) for 'xy'. Normalize in float32, then cast before mapping [0, 1] to [-1, 1],
    preserving the helper's low-precision rounding order.

    Retained as a compatibility helper for callers needing coordinates. For rotary embeddings, prefer
    RotaryEmbeddingDinoV3 or build_rotary_pos_embed with grid_type='centered'.
    """
    return _build_grid(
        [height, width],
        grid_type='centered',
        grid_offset=grid_offset,
        grid_indexing=grid_indexing,
        normalize_coords=normalize_coords,
        device=device,
        dtype=dtype,
    ).flatten(0, 1)


class RotaryEmbeddingDinoV3(nn.Module):
    """RoPE for timm DinoV3 port, numerically matching original.

    FX tracing with a constant grid shape can freeze augmentation draws. Trace in eval mode.

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
        if feat_shape is not None and len(feat_shape) != 2:
            raise ValueError(f'Expected 2 spatial dimensions, got {len(feat_shape)}')

        # Dimensions / output format
        self.dim = dim  # equal to head_dim for most vit applications
        self.rotate_half = rotate_half

        # Period schedule parameters
        self.temperature = float(temperature) if temperature is not None else None
        self.min_period = min_period
        self.max_period = max_period

        # Coord processing + augs
        self.normalize_coords = normalize_coords
        self.shift_coords, self.jitter_coords, self.rescale_coords, self.aug_active = _coord_aug_config(
            shift_coords, jitter_coords, rescale_coords,
        )

        # Grid config
        self.feat_shape = feat_shape
        self.grid_offset = grid_offset
        self.grid_indexing = grid_indexing

        # Register empty buffer for periods (kept at >= float32, see _apply)
        periods_shape = (dim // 4,)
        self.register_buffer(
            "periods",
            torch.empty(periods_shape, device=device, dtype=_freq_buffer_dtype(dtype)),
            persistent=False,
        )

        if feat_shape is not None:
            # Register empty buffer for cached embeddings
            num_pos = feat_shape[0] * feat_shape[1]
            emb_shape = (num_pos, dim * 2)  # concatenated sin & cos
            self.register_buffer(
                "pos_embed_cached",
                torch.empty(emb_shape, device=device, dtype=dtype),
                persistent=False,
            )
        else:
            self.pos_embed_cached = None

        # TODO: skip init when on meta device when safe to do so
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

    def _apply(self, fn, *args, **kwargs):
        module = super()._apply(fn, *args, **kwargs)
        # .to(dtype) / .half() / .bfloat16() cast all floating buffers, rebuild periods if they were downcast
        if _freq_buffer_dtype(self.periods.dtype) != self.periods.dtype:
            self.register_buffer('periods', self._compute_periods(device=self.periods.device), persistent=False)
        return module

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
        # Evaluate each frequency once, then expand to the requested rotation layout.
        sin = _rope_layout(angles.sin(), self.rotate_half)
        cos = _rope_layout(angles.cos(), self.rotate_half)
        return sin, cos

    def _create_embed(
            self,
            feat_shape: List[int],
            no_aug: bool = False,
    ) -> torch.Tensor:
        coords = _build_grid(
            feat_shape,
            grid_type='centered',
            grid_offset=self.grid_offset,
            grid_indexing=self.grid_indexing,
            normalize_coords=self.normalize_coords,
            ndim=2,
        )  # (H, W, 2), built on CPU to preserve timm's grid numerics and augmentation RNG device
        coords = coords.flatten(0, -2)  # (HW, 2)
        if not no_aug and self.training and self.aug_active:
            coords = _apply_coord_augs(coords, self.shift_coords, self.jitter_coords, self.rescale_coords)
        sin, cos = self._get_pos_embed_from_coords(coords)  # 2 * (HW, dim)
        rope_embed = torch.cat([sin, cos], dim=-1)  # (HW, 2*dim)
        return rope_embed

    def _cache_embed(self, feat_shape: List[int]):
        # create non-augmented embeds for cache
        rope_embed = self._create_embed(feat_shape, no_aug=True)
        if self.pos_embed_cached is not None:
            rope_embed = rope_embed.to(dtype=self.pos_embed_cached.dtype)
        self.register_buffer("pos_embed_cached", rope_embed, persistent=False)
        self.feat_shape = feat_shape

    def update_feat_shape(self, feat_shape: List[int]):
        if self.feat_shape is not None and feat_shape != self.feat_shape:
            # only update if feat_shape was set (valid cache) and different from previous value
            self._cache_embed(feat_shape)

    def init_non_persistent_buffers(self) -> None:
        """Initialize non-persistent buffers."""
        self._init_buffers()

    def get_embed(
            self,
            shape: Optional[List[int]] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Generate rope_embed matching DINOv3 RopePositionEmbedding numerics.

        Args:
            shape: Spatial shape to build embeddings for, uses the cached feat_shape if None.
            dtype: Output dtype; None uses the periods dtype (at least float32) for generated embeddings
                and the cache dtype for unaugmented cached output.

        Returns: (HW, 2 * dim) with last dim = [sin, cos] cat.
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

        if dtype is not None:
            rope_embed = rope_embed.to(dtype=dtype)
        return rope_embed

    def get_batch_embeds(
            self,
            shapes: List[Tuple[int, int]],
            seq_len: Optional[int] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate ROPE embeddings for multiple grid shapes.

        Coordinates are normalized per grid size so each shape is computed separately (no max-shape slicing).

        Args:
            shapes: List of (H, W) tuples representing different grid sizes
            seq_len: If provided, return padded tensor of this length. Otherwise return list.
            dtype: Output dtype, None keeps the default.

        Returns:
            If seq_len is provided: Padded tensor of shape (len(shapes), seq_len, 2 * dim)
            Otherwise: List of concatenated sin/cos embeddings, each with shape (H * W, 2 * dim)
        """
        if not shapes:
            return []

        embeds = [self.get_embed((h, w), dtype=dtype) for h, w in shapes]

        if seq_len is not None:
            return _pad_rope_embeds(embeds, seq_len)
        return embeds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to a channels-first tensor (B, C, *spatial) with C == dim."""
        shape = x.shape
        pos_embed = self.get_embed(shape[2:])
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x_flat = apply_rot_embed_cat(x_flat, pos_embed, half=self.rotate_half)
        return x_flat.transpose(1, 2).reshape(shape)


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
            - 'mrope': RotaryEmbeddingMRope (interleaved multimodal RoPE; `mrope_section` defaults to (8, 12, 12))
        dim: Total embedding dimension
        num_heads: Number of attention heads
        **kwargs: Additional arguments passed to the specific RoPE class

    Returns:
        Rotary embedding module
    """
    if rope_type == 'base':
        return RotaryEmbedding(dim=dim // num_heads, **kwargs)
    elif rope_type == 'cat':
        return RotaryEmbeddingCat(dim=dim // num_heads, **kwargs)
    elif rope_type == 'mixed':
        # Mixed requires depth parameter, generates differing embeddings per layer and head
        if kwargs.pop('rotate_half', False):
            raise ValueError("rope_type='mixed' requires rotate_half=False")
        kwargs.pop('in_pixels', None)  # doesn't support
        kwargs.pop('ref_feat_shape', None)  # doesn't support
        return RotaryEmbeddingMixed(dim=dim, num_heads=num_heads, **kwargs)
    elif rope_type == 'dinov3':
        kwargs.pop('in_pixels', None)  # doesn't support
        kwargs.pop('ref_feat_shape', None)  # doesn't support
        return RotaryEmbeddingDinoV3(dim=dim // num_heads, **kwargs)
    elif rope_type == 'mrope':
        if not kwargs.pop('rotate_half', True):
            raise ValueError("rope_type='mrope' requires rotate_half=True")
        for k in ('in_pixels', 'ref_feat_shape', 'feat_shape'):
            kwargs.pop(k, None)  # mrope builds the half-layout cat tensor itself; these don't apply
        return RotaryEmbeddingMRope(dim=dim // num_heads, **kwargs)
    else:
        raise ValueError(f"Unknown RoPE type: {rope_type}")
