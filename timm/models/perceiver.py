""" Perceiver

Paper: `Perceiver: General Perception with Iterative Attention` - https://arxiv.org/abs/2103.03206

Official Deepmind code: TBD (doesn't exist yet)

Fourier feature position embedding references:
 * Official NeRF impl - https://github.com/bmild/nerf
 * Lucidrain's Perceiver impl - https://github.com/lucidrains/perceiver-pytorch

Status:
* Work in progress, currently running training trials with S and M models (rather slow)

Hacked together by / copyright Ross Wightman, 2021.
"""
import math
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import (
    Mlp,
    DropPath,
    PatchEmbed,
    trunc_normal_,
    lecun_normal_,
    to_ntuple,
    Attention,
    AttentionRope,
    create_rope_embed,
    apply_rot_embed_cat,
    use_fused_attn,
    register_notrace_module
)
from ._builder import build_model_with_cfg
from ._manipulate import named_apply
from ._registry import generate_default_cfgs, register_model

__all__ = ['Perceiver']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': (), 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'perceiver_ss.untrained': _cfg(input_size=(3, 192, 192)),
    'perceiver_s.untrained': _cfg(input_size=(3, 192, 192)),
    'perceiver_m.untrained': _cfg(),
    'perceiver_m_ls.untrained': _cfg(),
    'perceiver_l.untrained': _cfg(),
})


def build_linear_positions(
        index_dims,
        device=None,
        dtype=torch.float32,
        output_range=(-1., 1.)
):
    axes = [
        torch.linspace(output_range[0], output_range[1], steps=s, device=device, dtype=dtype)
        for s in index_dims
    ]
    grid = torch.meshgrid(*axes, indexing='ij')
    return torch.stack(grid, dim=-1)  # (*index_dims, d)


def generate_fourier_features(
        pos,                      # (n, d), values in [-1, 1]
        num_bands,
        max_res,           # tuple/list length d
        concat_pos=True,
):
    device, dtype = pos.device, pos.dtype
    d = pos.shape[-1]
    # (d, K)
    freq_bands = torch.stack([
        torch.linspace(1.0, float(res) / 2.0, steps=num_bands, device=device, dtype=dtype)
        for res in max_res
    ], dim=0)

    # (n, d, K) -> (n, d*K)
    per_pos = (pos[:, :, None] * freq_bands[None, :, :]).reshape(pos.shape[0], d * num_bands)
    enc = torch.cat([torch.sin(torch.pi * per_pos), torch.cos(torch.pi * per_pos)], dim=-1)

    if concat_pos:
        enc = torch.cat([pos, enc], dim=-1)
    return enc


def fourier_grid_official(
        index_dims,
        num_bands: int = 64,
        max_res=None,
        device=None,
        dtype=torch.float32,
        concat_pos=True,
):
    max_res = tuple(max_res) if max_res is not None else tuple(index_dims)
    grid = build_linear_positions(index_dims, device=device, dtype=dtype)     # (*dims, d)
    pos = grid.reshape(-1, len(index_dims))                                   # (n, d)
    return generate_fourier_features(pos, num_bands, max_res, concat_pos)  # (n, pe_dim)


def fourier_encode(x, max_freq_log2: int = 8, num_bands: int = 64):
    """ Fourier feature embedding.
    Referenced official NeRF code and Lucidrain's PyTorch Perceiver impl.
    """
    # FIXME this will likely need to change once official code / weights are available
    x = x.unsqueeze(-1)
    bands = 2 ** torch.linspace(0, max_freq_log2 - 1, num_bands, device=x.device, dtype=x.dtype)
    x_bands = x * math.pi * bands
    x = torch.cat([x, x_bands.sin(), x_bands.cos()], dim=-1)
    return x


def fourier_grid(
        shape: List[int],
        max_freq_log2: int = 8,
        num_bands: int = 64,
        device: torch.device = torch.device('cuda'),
):
    grid = torch.stack(torch.meshgrid([torch.linspace(-1., 1., steps=s, device=device) for s in shape]), dim=-1)
    enc_pos = fourier_encode(grid, max_freq_log2, num_bands)
    return enc_pos.transpose(-1, -2).flatten(len(shape))


def _devkey(device: torch.device) -> Tuple[str, int]:
    return device.type, (-1 if device.index is None else int(device.index))


def _prod(xs: Sequence[int]) -> int:
    p = 1
    for v in xs:
        p *= int(v)
    return p


@register_notrace_module
class InputAdapter(nn.Module):
    """
    Perceiver grid input adapter: Fourier pos enc (+ optional projection) + combine with tokens.

    Position encoding (DeepMind official scheme):
      freq_bands[dim] = linspace(1.0, max_resolution[dim]/2, num_bands)
      per_pos = flatten(pos[:, dim] * freq_bands[dim]) over dims & bands
      enc = [sin(pi*per_pos), cos(pi*per_pos)]
      if concat_pos: prepend raw pos coords

    Combine modes:
      - combine="concat" (Perceiver default):
          data = cat([x_tokens, pos_tokens], dim=-1)
          If pos_proj=True, pos_tokens are projected so that final channel dim == out_dim.
          Output channels:
            * if pos_proj=False: out_dim = in_chans + raw_pos_dim
            * if pos_proj=True : out_dim must be provided, pos_proj_out = out_dim - in_chans

      - combine="add" (alternate):
          x_tokens projected to out_dim (Identity if already matching), pos projected to out_dim:
            data = x_proj(x_tokens) + pos_proj(pos_raw)
          out_dim must be provided.

    Inputs:
      x:
        - (B, N, C) -> requires feat_size
        - (B, C, *feat_size) -> flattened to (B, N, C) if flatten_spatial_input=True
      feat_size:
        - tuple of length pos_ndim, required when x is (B, N, C)

    Caching:
      - pos grid, freq bands, raw Fourier encodings computed & stored in float32
      - cached enc cast to x.dtype on apply before any projection
      - keys:
          * pos grid + raw enc: (feat_size, device_type, device_index)
          * freq bands: (max_resolution, device_type, device_index)
      - caches disabled under torchscript/tracing
    """

    def __init__(
        self,
        *,
        in_chans: int,
        pos_ndim: int = 2,
        num_bands: int = 64,
        concat_pos: bool = True,
        max_resolution: Optional[Sequence[int]] = None,  # if None, uses feat_size each forward (official default)
        combine: str = "concat",                          # "concat" | "add"
        out_dim: Optional[int] = None,                    # required for add; required for concat if pos_proj=True
        pos_proj: bool = False,                           # only meaningful for concat
        flatten_spatial_input: bool = True,
        output_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()

        self.in_chans = int(in_chans)
        self.pos_ndim = int(pos_ndim)
        self.num_bands = int(num_bands)
        self.concat_pos = bool(concat_pos)
        self.flatten_spatial_input = bool(flatten_spatial_input)
        self.output_range = (float(output_range[0]), float(output_range[1]))

        self.max_resolution = tuple(int(x) for x in max_resolution) if max_resolution is not None else None
        if self.max_resolution is not None and len(self.max_resolution) != self.pos_ndim:
            raise ValueError(f"max_resolution ndim {len(self.max_resolution)} != pos_ndim {self.pos_ndim}")

        combine = str(combine).lower()
        if combine not in ("concat", "add"):
            raise ValueError(f"combine must be 'concat' or 'add', got {combine!r}")
        self.combine = combine

        # Raw pos feature width (fixed given pos_ndim/num_bands/concat_pos)
        self.raw_pos_dim = self.pos_ndim * (2 * self.num_bands) + (self.pos_ndim if self.concat_pos else 0)

        # Projections
        self.pos_proj_enabled = bool(pos_proj) if self.combine == "concat" else True  # add always projects pos
        self.out_dim: int

        self.x_proj: Optional[nn.Module]
        self.pos_proj_layer: Optional[nn.Module]

        if self.combine == "concat":
            if self.pos_proj_enabled:
                if out_dim is None:
                    raise ValueError("out_dim is required when combine='concat' and pos_proj=True.")
                out_dim_i = int(out_dim)
                if out_dim_i <= self.in_chans:
                    raise ValueError(f"out_dim ({out_dim_i}) must be > in_chans ({self.in_chans}) for concat+pos_proj.")
                pos_out = out_dim_i - self.in_chans
                self.pos_proj_layer = nn.Linear(self.raw_pos_dim, pos_out)
                self.out_dim = out_dim_i
            else:
                # No projection: output dim is fixed by raw_pos_dim
                expected = self.in_chans + self.raw_pos_dim
                if out_dim is not None and int(out_dim) != expected:
                    raise ValueError(
                        f"out_dim={int(out_dim)} does not match in_chans+raw_pos_dim={expected}. "
                        f"Either set out_dim={expected} or enable pos_proj=True."
                    )
                self.pos_proj_layer = None
                self.out_dim = expected

            self.x_proj = None  # concat mode keeps x as-is

        else:
            # add mode
            if out_dim is None:
                raise ValueError("out_dim is required when combine='add'.")
            out_dim_i = int(out_dim)
            if out_dim_i <= 0:
                raise ValueError("out_dim must be > 0 when combine='add'.")
            self.out_dim = out_dim_i

            self.x_proj = nn.Identity() if self.in_chans == self.out_dim else nn.Linear(self.in_chans, self.out_dim)
            self.pos_proj_layer = nn.Linear(self.raw_pos_dim, self.out_dim)

        # float32 caches (not in state_dict)
        self._pos_grid_cache: Dict[Tuple[Tuple[int, ...], str, int], torch.Tensor] = {}
        self._freq_cache: Dict[Tuple[Tuple[int, ...], str, int], torch.Tensor] = {}
        self._fourier_cache: Dict[Tuple[Tuple[int, ...], str, int], torch.Tensor] = {}

    @torch.jit.ignore
    def clear_cache(self) -> None:
        self._pos_grid_cache.clear()
        self._freq_cache.clear()
        self._fourier_cache.clear()

    def _get_max_resolution(self, feat_size: Tuple[int, ...]) -> Tuple[int, ...]:
        return self.max_resolution if self.max_resolution is not None else feat_size

    def _maybe_flatten_x(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, ...]]]:
        if x.ndim == 3:
            return x, None

        # Expect spatial: (B, C, *feat_size)
        if x.ndim != 2 + self.pos_ndim:
            raise ValueError(
                f"Expected x as (B,N,C) or (B,C,*feat_size) with feat_size ndim={self.pos_ndim}; got x.ndim={x.ndim}."
            )
        if not self.flatten_spatial_input:
            raise ValueError("x is spatial but flatten_spatial_input=False; expected tokens (B,N,C).")

        B = x.shape[0]
        C = x.shape[1]
        feat_size = tuple(int(s) for s in x.shape[2:])
        if len(feat_size) != self.pos_ndim:
            raise ValueError(f"Inferred feat_size ndim {len(feat_size)} != pos_ndim {self.pos_ndim}")

        x_tokens = x.reshape(B, C, -1).transpose(1, 2).contiguous()  # (B, N, C)
        return x_tokens, feat_size

    def _resolve_feat_size(
        self,
        inferred_feat_size: Optional[Tuple[int, ...]],
        feat_size: Optional[Sequence[int]],
        *,
        require: bool,
    ) -> Tuple[int, ...]:
        if inferred_feat_size is not None:
            return inferred_feat_size
        if feat_size is not None:
            fs = tuple(int(s) for s in feat_size)
            if len(fs) != self.pos_ndim:
                raise ValueError(f"feat_size ndim {len(fs)} != pos_ndim {self.pos_ndim}")
            return fs
        if require:
            raise ValueError("feat_size is required when x is (B,N,C).")
        return ()

    @torch.no_grad()
    def _build_linear_positions(self, feat_size: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        lo, hi = self.output_range
        axes = [torch.linspace(lo, hi, steps=s, device=device, dtype=torch.float32) for s in feat_size]
        grid = torch.meshgrid(*axes, indexing="ij")
        return torch.stack(grid, dim=-1).reshape(_prod(feat_size), self.pos_ndim)  # (N,d), f32

    @torch.no_grad()
    def _get_pos_grid(self, feat_size: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        k = (feat_size, *_devkey(device))
        t = self._pos_grid_cache.get(k)
        if t is None:
            t = self._build_linear_positions(feat_size, device=device)
            self._pos_grid_cache[k] = t
        return t

    @torch.no_grad()
    def _get_freq_bands(self, max_resolution: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        k = (max_resolution, *_devkey(device))
        fb = self._freq_cache.get(k)
        if fb is None:
            fb = torch.stack(
                [
                    torch.linspace(
                        1.0, float(res) / 2.0,
                        steps=self.num_bands,
                        device=device,
                        dtype=torch.float32,
                    )
                    for res in max_resolution
                ],
                dim=0,
            )  # (d, K), f32
            self._freq_cache[k] = fb
        return fb

    @torch.no_grad()
    def _fourier_features(self, pos: torch.Tensor, max_resolution: Tuple[int, ...]) -> torch.Tensor:
        """
        (N, raw_pos_dim) float32, DeepMind ordering:
          [pos (optional), sin(all dims/bands), cos(all dims/bands)]
        """
        freq_bands = self._get_freq_bands(max_resolution, device=pos.device)  # (d,K)
        per_pos = (pos[:, :, None] * freq_bands[None, :, :]).reshape(pos.shape[0], self.pos_ndim * self.num_bands)
        enc = torch.cat([torch.sin(math.pi * per_pos), torch.cos(math.pi * per_pos)], dim=-1)
        if self.concat_pos:
            enc = torch.cat([pos, enc], dim=-1)
        return enc  # f32

    def forward(self, x: torch.Tensor, feat_size: Optional[Sequence[int]] = None) -> torch.Tensor:
        x_tokens, inferred_fs = self._maybe_flatten_x(x)
        B, N, C = x_tokens.shape
        if C != self.in_chans:
            raise ValueError(f"x token dim C={C} != in_chans={self.in_chans}")

        fs = self._resolve_feat_size(inferred_fs, feat_size, require=True)
        if _prod(fs) != N:
            raise ValueError(f"feat_size {tuple(fs)} implies N={_prod(fs)}, but x has N={N}.")

        device = x_tokens.device
        out_dtype = x_tokens.dtype
        max_res = self._get_max_resolution(tuple(fs))

        use_cache = (not torch.jit.is_scripting()) and (not torch.jit.is_tracing())

        if use_cache:
            k = (tuple(fs), *_devkey(device))
            enc_f32 = self._fourier_cache.get(k)
            if enc_f32 is None:
                pos = self._get_pos_grid(tuple(fs), device=device)  # f32
                enc_f32 = self._fourier_features(pos, max_res)      # f32
                self._fourier_cache[k] = enc_f32
        else:
            pos = self._build_linear_positions(tuple(fs), device=device)
            enc_f32 = self._fourier_features(pos, max_res)

        # Cast on apply, then optional projection
        pos_enc = enc_f32.to(dtype=out_dtype)  # (N, raw_pos_dim) in model dtype
        if self.pos_proj_layer is not None:
            pos_enc = self.pos_proj_layer(pos_enc)  # (N, P)

        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)  # (B, N, P)

        if self.combine == "concat":
            out = torch.cat([x_tokens, pos_enc], dim=-1)
            # sanity: out last dim should match self.out_dim
            if out.shape[-1] != self.out_dim:
                raise RuntimeError(f"concat output dim {out.shape[-1]} != configured out_dim {self.out_dim}")
            return out

        # add mode: project x and add
        assert self.x_proj is not None
        x_emb = self.x_proj(x_tokens)  # (B, N, out_dim)
        if x_emb.shape[-1] != self.out_dim or pos_enc.shape[-1] != self.out_dim:
            raise RuntimeError("add mode projections did not produce out_dim channels.")
        return x_emb + pos_enc



class CrossAttention(nn.Module):
    """Cross-attention module with SDPA and optional RoPE support.

    Q comes from latents, KV from data (e.g., patches).
    Supports separate RoPE embeddings for Q and K.

    Args:
        latent_dim: Input dimension for latents (queries).
        data_dim: Input dimension for data (keys/values).
        attn_dim: Attention dimension. If None, uses min(latent_dim, data_dim).
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in Q, K, V projections.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for output projection.
        rotate_half: Use 'half' RoPE layout instead of 'interleaved'.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            latent_dim: int,
            data_dim: int,
            attn_dim: Optional[int] = None,
            num_heads: int = 1,
            qkv_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            rotate_half: bool = True,
    ):
        super().__init__()
        assert latent_dim % num_heads == 0, f"dim {latent_dim} should be divided by num_heads {num_heads}."

        self.latent_dim = latent_dim
        self.attn_dim = attn_dim or min(latent_dim, data_dim)
        self.num_heads = num_heads
        self.head_dim = self.attn_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.rotate_half = rotate_half

        self.q = nn.Linear(latent_dim, self.attn_dim, bias=qkv_bias)
        self.kv = nn.Linear(data_dim, self.attn_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attn_dim, latent_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            latent: torch.Tensor,
            data: torch.Tensor,
            rope_q: Optional[torch.Tensor] = None,
            rope_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N_q, _ = latent.shape
        N_kv = data.shape[1]

        q = self.q(latent).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(data).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Apply RoPE to queries and keys
        if rope_q is not None:
            q = apply_rot_embed_cat(q, rope_q, half=self.rotate_half).type_as(v)
        if rope_k is not None:
            k = apply_rot_embed_cat(k, rope_k, half=self.rotate_half).type_as(v)

        if self.fused_attn:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N_q, self.attn_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)


class CrossBlock(nn.Module):
    """Cross-attention block with MLP.

    Latents cross-attend to data with optional RoPE support.
    """

    def __init__(
            self,
            latent_dim: int,
            data_dim: int,
            num_heads: int,
            attn_dim: Optional[int] = None,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1_latent = norm_layer(latent_dim)
        self.norm1_data = norm_layer(data_dim)
        self.attn = CrossAttention(
            latent_dim,
            data_dim,
            num_heads=num_heads,
            attn_dim=attn_dim,
            qkv_bias=qkv_bias,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(latent_dim)
        mlp_hidden_dim = int(latent_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=latent_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
            self,
            latent: torch.Tensor,
            data: torch.Tensor,
            rope_q: Optional[torch.Tensor] = None,
            rope_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent = latent + self.drop_path(self.attn(
            self.norm1_latent(latent),
            self.norm1_data(data),
            rope_q=rope_q,
            rope_k=rope_k,
        ))
        latent = latent + self.drop_path(self.mlp(self.norm2(latent)))
        return latent


class CrossBlockLayerScale(nn.Module):
    """Cross-attention block with LayerScale and MLP."""

    def __init__(
            self,
            latent_dim: int,
            data_dim: int,
            num_heads: int,
            attn_dim: Optional[int] = None,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            init_values: float = 1e-5,
            drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1_latent = norm_layer(latent_dim)
        self.norm1_data = norm_layer(data_dim)
        self.attn = CrossAttention(
            latent_dim,
            data_dim,
            num_heads=num_heads,
            attn_dim=attn_dim,
            qkv_bias=qkv_bias,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(latent_dim)
        mlp_hidden_dim = int(latent_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=latent_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.ls1 = nn.Parameter(init_values * torch.ones(latent_dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(latent_dim))

    def forward(
            self,
            latent: torch.Tensor,
            data: torch.Tensor,
            rope_q: Optional[torch.Tensor] = None,
            rope_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent = latent + self.drop_path(self.ls1 * self.attn(
            self.norm1_latent(latent),
            self.norm1_data(data),
            rope_q=rope_q,
            rope_k=rope_k,
        ))
        latent = latent + self.drop_path(self.ls2 * self.mlp(self.norm2(latent)))
        return latent


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP.

    Uses timm's standard Attention (with SDPA) or AttentionRope for RoPE support.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            use_rope: bool = False,
    ):
        super().__init__()
        self.use_rope = use_rope
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = AttentionRope(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                num_prefix_tokens=0,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_drop=drop,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, rope: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_rope:
            x = x + self.drop_path(self.attn(self.norm1(x), rope=rope))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerBlockLayerScale(nn.Module):
    """Transformer block with LayerScale, self-attention and MLP."""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            init_values: float = 1e-5,
            drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            use_rope: bool = False,
    ):
        super().__init__()
        self.use_rope = use_rope
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = AttentionRope(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                num_prefix_tokens=0,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_drop=drop,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor, rope: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_rope:
            x = x + self.drop_path(self.ls1 * self.attn(self.norm1(x), rope=rope))
        else:
            x = x + self.drop_path(self.ls1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ls2 * self.mlp(self.norm2(x)))
        return x


class TransformerStack(nn.Module):
    """A stack of transformer blocks with optional RoPE support."""

    def __init__(
            self,
            depth: int,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            block: Optional[nn.Module] = None,
            use_rope: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.use_rope = use_rope
        block = block or TransformerBlock
        self.blocks = nn.ModuleList([
            block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, use_rope=use_rope, **kwargs)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, rope: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, rope=rope)
        return x


def get_layer_layout(cross_depths, num_stages=8, share_weights=None):
    if isinstance(cross_depths, (tuple, list)):
        stage_cross_depths = tuple(cross_depths)
        stage_cross_depths = (stage_cross_depths + (0,) * num_stages)[:num_stages]
    else:
        stage_cross_depths = to_ntuple(num_stages)(cross_depths)
    prev_cross_key = ''
    prev_transformer_key = ''
    keys = []
    num_cross = 0
    num_transformer = 0
    for i, cd in enumerate(stage_cross_depths):
        for j in range(cd):
            key = prev_cross_key
            if share_weights is None or num_cross <= share_weights[0]:
                key = f'c{i}_{j}'
            keys += [key]
            prev_cross_key = key
            num_cross += 1
        key = prev_transformer_key
        if share_weights is None or num_transformer <= share_weights[1]:
            key = f't{i}'
        keys += [key]
        prev_transformer_key = key
        num_transformer += 1
    return keys


class Perceiver(nn.Module):
    """Perceiver with optional RoPE support.

    Paper: `Perceiver: General Perception with Iterative Attention` - https://arxiv.org/abs/2103.03206

    Args:
        img_size: Input image size (used for RoPE grid).
        patch_size: Patch size for patch embedding. If None, uses raw pixels (RoPE mode only).
        in_chans: Number of input channels.
        num_classes: Number of classes for classification head.
        num_stages: Number of stages (cross + transformer stack repeats).
        cross_depths: Number of cross-attention blocks per stage.
        transformer_depth: Number of transformer blocks per stage.
        latent_dim: Latent dimension.
        num_latents: Number of latent tokens.
        num_latent_heads: Number of attention heads for latent self-attention.
        latent_mlp_ratio: MLP ratio for latent blocks.
        cross_attn_dim: Attention dimension for cross-attention.
        num_cross_heads: Number of cross-attention heads.
        cross_mlp_ratio: MLP ratio for cross-attention blocks.
        share_weights: Starting index for weight sharing (cross, transformer).
        rope_type: Type of RoPE ('dinov3', 'cat', or None for Fourier pos embed).
        latent_rope: Whether to use RoPE for latents.
        data_bands: Number of Fourier bands for position encoding (when not using RoPE).
        data_ndim: Number of spatial dimensions for position encoding.
        data_combine: How to combine input with position encoding ('concat' or 'add').
        data_out_dim: Output dimension for input adapter. Required for 'add' mode or
            'concat' with data_pos_proj=True. If None with 'concat' mode, uses raw dims.
        data_pos_proj: Project position features before concat (only for 'concat' mode).
        data_concat_pos: Concatenate raw position coordinates to Fourier features.
        data_max_resolution: Max resolution for Fourier frequency bands. If None, uses feat_size.
        qkv_bias: Enable bias for QKV projections.
        cross_block: Cross-attention block class.
        transformer_block: Transformer block class.
        norm_layer: Normalization layer.
        act_layer: Activation layer.
        drop_rate: Dropout rate.
        drop_path_rate: Stochastic depth rate.
        weight_init: Weight initialization scheme.
    """

    def __init__(
            self,
            img_size: int = 224,
            patch_size: Optional[int] = None,
            in_chans: int = 3,
            num_classes: int = 1000,
            num_stages: int = 8,
            cross_depths: Tuple[int, ...] = (1,),
            transformer_depth: int = 6,
            latent_dim: int = 1024,
            num_latents: int = 256,
            num_latent_heads: int = 8,
            latent_mlp_ratio: float = 1.0,
            cross_attn_dim: Optional[int] = None,
            num_cross_heads: int = 1,
            cross_mlp_ratio: float = 1.0,
            share_weights: Optional[Tuple[int, int]] = (1, 0),
            rope_type: Optional[str] = None,
            latent_rope: bool = True,
            data_bands: int = 64,
            data_ndim: int = 2,
            data_combine: str = 'concat',
            data_out_dim: Optional[int] = None,
            data_pos_proj: bool = False,
            data_concat_pos: bool = True,
            data_max_resolution: Optional[Sequence[int]] = None,
            qkv_bias: bool = True,
            cross_block: Optional[nn.Module] = None,
            transformer_block: Optional[nn.Module] = None,
            norm_layer: Optional[nn.Module] = None,
            act_layer: Optional[nn.Module] = None,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.head_hidden_size = self.latent_dim = latent_dim
        self.rope_type = rope_type
        self.use_rope = rope_type is not None

        cross_block = cross_block or CrossBlock
        transformer_block = transformer_block or TransformerBlock
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        if self.use_rope:
            # RoPE mode: optional patch embedding, no Fourier positional embedding
            # Round up num_latents to next perfect square for 2D RoPE grid
            sqrt_latents = int(math.ceil(num_latents ** 0.5))
            num_latents = sqrt_latents * sqrt_latents
            latent_grid = [sqrt_latents, sqrt_latents]
            self.num_latents = num_latents
            self.latents = nn.Parameter(torch.zeros(num_latents, latent_dim))

            # Default cross_attn_dim to latent_dim when using RoPE
            cross_attn_dim = cross_attn_dim or latent_dim
            self.input_adapter = None

            if patch_size is not None and patch_size > 0:
                # Patch embedding mode: reduces sequence length significantly
                self.patch_embed = PatchEmbed(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=cross_attn_dim,
                    bias=True,
                )
                self.data_dim = cross_attn_dim
                grid_size = self.patch_embed.grid_size
                data_feat_shape = list(grid_size)
            else:
                # Raw pixel mode: full H*W sequence
                self.patch_embed = None
                self.data_dim = in_chans
                data_feat_shape = [img_size, img_size]

            # RoPE for cross-attention K (data positions)
            self.rope = create_rope_embed(
                rope_type=rope_type,
                dim=cross_attn_dim,
                num_heads=num_cross_heads,
                feat_shape=data_feat_shape,
                rotate_half=True,
            )

            # RoPE for latents (cross-attention Q and self-attention)
            if latent_rope and num_latents > 1:
                self.cross_latent_rope = create_rope_embed(
                    rope_type=rope_type,
                    dim=cross_attn_dim,
                    num_heads=num_cross_heads,
                    feat_shape=latent_grid,
                    rotate_half=True,
                )
                # RoPE for latent self-attention (latents in latent_dim space)
                self.latent_rope = create_rope_embed(
                    rope_type=rope_type,
                    dim=latent_dim,
                    num_heads=num_latent_heads,
                    feat_shape=latent_grid,
                    rotate_half=True,
                )
            else:
                self.cross_latent_rope = None
                self.latent_rope = None
        else:
            # Fourier positional embedding mode via InputAdapter
            # No constraint on num_latents for non-RoPE mode
            self.num_latents = num_latents
            self.latents = nn.Parameter(torch.zeros(num_latents, latent_dim))

            self.patch_embed = None
            self.input_adapter = InputAdapter(
                in_chans=in_chans,
                pos_ndim=data_ndim,
                num_bands=data_bands,
                concat_pos=data_concat_pos,
                max_resolution=data_max_resolution,
                combine=data_combine,
                out_dim=data_out_dim,
                pos_proj=data_pos_proj,
            )
            self.data_dim = self.input_adapter.out_dim
            self.rope = None
            self.cross_latent_rope = None
            self.latent_rope = None

        self.blocks_cross = nn.ModuleDict()
        self.blocks_trans = nn.ModuleDict()
        self.layer_keys = get_layer_layout(cross_depths, num_stages, share_weights)
        unique_keys = list(dict.fromkeys(self.layer_keys))
        for i, k in enumerate(unique_keys):
            stage_args = dict(
                qkv_bias=qkv_bias,
                drop=drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            if k.startswith('c'):
                self.blocks_cross[k] = cross_block(
                    latent_dim=latent_dim,
                    data_dim=self.data_dim,
                    attn_dim=cross_attn_dim,
                    num_heads=num_cross_heads,
                    mlp_ratio=cross_mlp_ratio,
                    **stage_args,
                )
            else:
                self.blocks_trans[k] = TransformerStack(
                    depth=transformer_depth,
                    dim=latent_dim,
                    num_heads=num_latent_heads,
                    mlp_ratio=latent_mlp_ratio,
                    block=transformer_block,
                    use_rope=self.latent_rope is not None,
                    **stage_args,
                )

        self.norm = norm_layer(latent_dim)
        self.head = nn.Linear(latent_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.latents, std=.02)
        named_apply(partial(_init_weights, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'latents'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.latent_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if pre_logits:
            return x
        return self.head(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        if self.use_rope:
            # RoPE mode: patch embedding or raw pixels
            if self.patch_embed is not None:
                data = self.patch_embed(x)  # (B, num_patches, embed_dim)
            else:
                data = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
            rope_k = self.rope.get_embed()  # for K (data) in cross-attention
            rope_q = self.cross_latent_rope.get_embed() if self.cross_latent_rope is not None else None
            latent_rope = self.latent_rope.get_embed() if self.latent_rope is not None else None
        else:
            # InputAdapter mode: combine input with Fourier position features
            data = self.input_adapter(x)  # (B, H*W, data_dim)
            rope_k = None
            rope_q = None
            latent_rope = None

        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        for k in self.layer_keys:
            if k.startswith('c'):
                cross_block = self.blocks_cross[k]
                latents = cross_block(latents, data, rope_q=rope_q, rope_k=rope_k)
            else:
                transformer = self.blocks_trans[k]
                latents = transformer(latents, rope=latent_rope)

        latents = self.norm(latents)
        return latents.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ weight initialization
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if 'mlp' in name:
                    nn.init.normal_(module.bias, std=1e-6)
                else:
                    nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _create_perceiver(variant: str, pretrained: bool = False, **kwargs) -> Perceiver:
    """Create Perceiver model."""
    return build_model_with_cfg(
        Perceiver,
        variant,
        pretrained,
        **kwargs,
    )


@register_model
def perceiver_ss(pretrained: bool = False, **kwargs) -> Perceiver:
    """Perceiver-Small (Shared).

    One initial cross attn and all transformer stacks shared. ~11M params.
    """
    model_args = dict(
        cross_depths=(1,), latent_dim=512, num_latents=256, cross_attn_dim=128, data_bands=36,
    )
    return _create_perceiver('perceiver_ss', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def perceiver_s(pretrained: bool = False, **kwargs) -> Perceiver:
    """Perceiver-Small.

    One initial cross attn and all but first transformer stacks shared. ~20M params.
    """
    model_args = dict(
        cross_depths=(1,), latent_dim=512, num_latents=256, cross_attn_dim=128, data_bands=36,
        share_weights=(1, 1),
    )
    return _create_perceiver('perceiver_s', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def perceiver_m(pretrained: bool = False, **kwargs) -> Perceiver:
    """Perceiver-Medium.

    Two cross attn (one per each initial transformer stack), all transformers shared. ~25M params.
    """
    model_args = dict(
        cross_depths=(1,) * 2, latent_dim=768, num_latents=384, cross_attn_dim=160, data_bands=40,
    )
    return _create_perceiver('perceiver_m', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def perceiver_m_ls(pretrained: bool = False, **kwargs) -> Perceiver:
    """Perceiver-Medium w/ LayerScale + Affine.

    Two cross attn (one per each initial transformer stack), all transformers shared. ~25M params.
    LayerScale + Affine influenced by CaiT, LeViT, ResMLP from Facebook AI.
    """
    model_args = dict(
        cross_depths=(1,) * 2, latent_dim=768, num_latents=384, cross_attn_dim=160, data_bands=40,
        transformer_block=TransformerBlockLayerScale, cross_block=CrossBlockLayerScale,
        norm_layer=Affine,
    )
    return _create_perceiver('perceiver_m_ls', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def perceiver_l(pretrained: bool = False, **kwargs) -> Perceiver:
    """Perceiver-Large.

    One cross attn per 8 transformer stacks. All but first cross attn shared, all transformer stacks shared.
    This variant is closest to the paper model for reported ImageNet results. ~45M params.
    """
    model_args = dict(cross_depths=1, latent_dim=1024, num_latents=512)
    return _create_perceiver('perceiver_l', pretrained=pretrained, **dict(model_args, **kwargs))