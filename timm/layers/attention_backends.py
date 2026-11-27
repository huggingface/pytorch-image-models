""" Attention backend dispatch for variable length (padded) sequences.

`attn_mask` passed through blocks to attention modules may be:
 * None or a dense Tensor mask -> F.scaled_dot_product_attention
 * VarlenMeta -> packed (unpadded) tokens w/ torch.nn.attention.varlen.varlen_attn (Flash / cuDNN varlen kernels)
 * BlockMask -> torch.nn.attention.flex_attention (compiled)

For the varlen backend the model packs valid tokens of the batch into a single [1, T, C] sequence once (norms, MLPs,
and projections are per-token so blocks are unchanged), runs all blocks, and re-pads at the end.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention.varlen import varlen_attn
    has_varlen = True
except ImportError:
    has_varlen = False

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
    has_flex = True
except ImportError:
    BlockMask = None
    has_flex = False

_compiled_flex = None


def _get_compiled_flex():
    global _compiled_flex
    if _compiled_flex is None:
        # default (automatic) dynamic shapes, avoids a recompile for every distinct NaFlex sequence length
        _compiled_flex = torch.compile(flex_attention)
    return _compiled_flex


@dataclass
class VarlenMeta:
    """ Packing metadata for varlen attention over a padded batch. """
    cu_seqlens: torch.Tensor  # int32 [B + 1]
    max_seqlen: int
    indices: torch.Tensor  # flat indices of valid tokens in the [B * N] padded layout
    batch_size: int
    seq_len: int


def build_varlen_meta(valid: torch.Tensor) -> VarlenMeta:
    """ Build packing metadata from a [B, N] bool validity mask. Valid tokens need not be contiguous. """
    B, N = valid.shape
    lengths = valid.sum(1, dtype=torch.int32)
    cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    indices = torch.nonzero(valid.flatten(), as_tuple=False).flatten()
    return VarlenMeta(
        cu_seqlens=cu_seqlens,
        max_seqlen=int(lengths.max().item()),
        indices=indices,
        batch_size=B,
        seq_len=N,
    )


def pack_tokens(x: torch.Tensor, meta: VarlenMeta) -> torch.Tensor:
    """ [B, N, ...] -> [1, T, ...] (valid tokens only) """
    return x.flatten(0, 1).index_select(0, meta.indices).unsqueeze(0)


def unpack_tokens(x: torch.Tensor, meta: VarlenMeta) -> torch.Tensor:
    """ [1, T, ...] -> [B, N, ...] w/ zeros at padded positions """
    x = x.squeeze(0)
    out = x.new_zeros((meta.batch_size * meta.seq_len, *x.shape[1:]))
    out.index_copy_(0, meta.indices, x)
    return out.reshape(meta.batch_size, meta.seq_len, *x.shape[1:])


def build_flex_padding_mask(valid: torch.Tensor, symmetric: bool = False) -> 'BlockMask':
    """ FlexAttention block mask from a [B, N] bool key validity mask.

    Keys must be valid. If symmetric, queries must be valid too (padded query rows attend to themselves only).
    """
    assert has_flex
    B, N = valid.shape
    valid = valid.contiguous()

    if symmetric:
        def mask_mod(b, h, q_idx, kv_idx):
            return (valid[b, kv_idx] & valid[b, q_idx]) | (q_idx == kv_idx)
    else:
        def mask_mod(b, h, q_idx, kv_idx):
            return valid[b, kv_idx]

    return create_block_mask(mask_mod, B, None, N, N, device=valid.device)


def dispatch_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask=None,
        dropout_p: float = 0.,
        is_causal: bool = False,
        scale: Optional[float] = None,
) -> torch.Tensor:
    """ Attention over [B, H, N, D] q / k / v w/ the backend selected by the type of attn_mask. """
    if isinstance(attn_mask, VarlenMeta):
        # packed: B == 1, [1, H, T, D] -> [T, H, D]
        assert q.shape[0] == 1, 'varlen attention expects packed [1, H, T, D] inputs'
        assert dropout_p == 0., 'attention dropout not supported w/ varlen backend'
        out = varlen_attn(
            q[0].transpose(0, 1),
            k[0].transpose(0, 1),
            v[0].transpose(0, 1),
            attn_mask.cu_seqlens,
            attn_mask.cu_seqlens,
            attn_mask.max_seqlen,
            attn_mask.max_seqlen,
            scale=scale,
            window_size=(-1, 0) if is_causal else (-1, -1),
        )
        return out.transpose(0, 1).unsqueeze(0)
    if BlockMask is not None and isinstance(attn_mask, BlockMask):
        assert dropout_p == 0., 'attention dropout not supported w/ flex backend'
        return _get_compiled_flex()(q, k, v, block_mask=attn_mask, scale=scale)
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
