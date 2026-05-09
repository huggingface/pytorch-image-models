# Copyright 2025 Google LLC (Apache 2.0)
# Adapted for timm by [your name]
"""TIPSv2 vision encoder for timm.

Paper: https://arxiv.org/abs/2604.12012
HF Hub: https://huggingface.co/google/tipsv2-b14
"""
import os
import warnings
import functools
import math
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model
# from timm.models._pretrained import PretrainedCfg


# ── Copy these from image_encoder.py ─────────────────────────────────────────
# PatchEmbed, Mlp, SwiGLUFFN, Attention, LayerScale, DropPath, Block,
# VisionTransformer (rename to TipsV2VisionTransformer to avoid conflict)
# init_weights_vit_timm
# ─────────────────────────────────────────────────────────────────────────────
class Mlp(nn.Module):
  """Transformer MLP, following DINOv2 implementation."""

  def __init__(
      self,
      in_features: int,
      hidden_features: Optional[int] = None,
      out_features: Optional[int] = None,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      drop: float = 0.0,
      bias: bool = True,
  ) -> None:
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
    self.act = act_layer()
    self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
    self.drop = nn.Dropout(drop)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    return x


def make_2tuple(x):
  if isinstance(x, tuple):
    assert len(x) == 2
    return x

  assert isinstance(x, int)
  return (x, x)


class PatchEmbed(nn.Module):
  """2D image to patch embedding: (B,C,H,W) -> (B,N,D)."""

  def __init__(
      self,
      img_size: Union[int, Tuple[int, int]] = 224,
      patch_size: Union[int, Tuple[int, int]] = 16,
      in_chans: int = 3,
      embed_dim: int = 768,
      norm_layer: Optional[Callable] = None,  # pylint: disable=g-bare-generic
      flatten_embedding: bool = True,
  ) -> None:
    super().__init__()

    image_hw = make_2tuple(img_size)
    patch_hw = make_2tuple(patch_size)
    patch_grid_size = (
        image_hw[0] // patch_hw[0],
        image_hw[1] // patch_hw[1],
    )

    self.img_size = image_hw
    self.patch_size = patch_hw
    self.patches_resolution = patch_grid_size
    self.num_patches = patch_grid_size[0] * patch_grid_size[1]

    self.in_chans = in_chans
    self.embed_dim = embed_dim

    self.flatten_embedding = flatten_embedding

    self.proj = nn.Conv2d(
        in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw
    )
    self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.shape
    patch_h, patch_w = self.patch_size

    assert (
        h % patch_h == 0
    ), f"Input image height {h} is not a multiple of patch height {patch_h}"
    assert (
        w % patch_w == 0
    ), f"Input image width {w} is not a multiple of patch width: {patch_w}"

    x = self.proj(x)  # B C H W
    h, w = x.size(2), x.size(3)
    x = x.flatten(2).transpose(1, 2)  # B HW C
    x = self.norm(x)
    if not self.flatten_embedding:
      x = x.reshape(-1, h, w, self.embed_dim)  # B H W C
    return x

  def flops(self) -> float:
    ho, wo = self.patches_resolution
    flops = (
        ho
        * wo
        * self.embed_dim
        * self.in_chans
        * (self.patch_size[0] * self.patch_size[1])
    )
    if self.norm is not None:
      flops += ho * wo * self.embed_dim
    return flops

class SwiGLUFFN(nn.Module):
  """SwiGLU FFN layer, following DINOv2 implementation."""

  def __init__(
      self,
      in_features: int,
      hidden_features: Optional[int] = None,
      out_features: Optional[int] = None,
      act_layer: Callable[..., nn.Module] = None,
      drop: float = 0.0,
      bias: bool = True,
  ) -> None:
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
    self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x12 = self.w12(x)
    x1, x2 = x12.chunk(2, dim=-1)
    hidden = F.silu(x1) * x2
    return self.w3(hidden)
  
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
  if XFORMERS_ENABLED:
    from xformers.ops import SwiGLU, memory_efficient_attention, unbind, fmha, scaled_index_add, index_select_cat  # pylint: disable=g-multiple-import, g-import-not-at-top

    XFORMERS_AVAILABLE = True
    warnings.warn("xFormers is available (SwiGLU)")
  else:
    warnings.warn("xFormers is disabled (SwiGLU)")
    raise ImportError
except ImportError:
  SwiGLU = SwiGLUFFN
  XFORMERS_AVAILABLE = False

  warnings.warn("xFormers is not available (SwiGLU)")
  
class SwiGLUFFNFused(SwiGLU):
  """SwiGLU FFN layer, following DINOv2 implementation."""

  def __init__(
      self,
      in_features: int,
      hidden_features: Optional[int] = None,
      out_features: Optional[int] = None,
      act_layer: Callable[..., nn.Module] = None,  # pylint: disable=unused-argument
      drop: float = 0.0,  # pylint: disable=unused-argument
      bias: bool = True,
  ) -> None:
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
    super().__init__(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        bias=bias,
    )


class Attention(nn.Module):
  """Attention layer, following DINOv2 implementation."""

  def __init__(
      self,
      dim: int,
      num_heads: int = 8,
      qkv_bias: bool = False,
      proj_bias: bool = True,
      attn_drop: float = 0.0,
      proj_drop: float = 0.0,
  ) -> None:
    super().__init__()
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = head_dim**-0.5

    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim, bias=proj_bias)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    b_dim, n_dim, c_dim = x.shape
    qkv = (
        self.qkv(x)
        .reshape(b_dim, n_dim, 3, self.num_heads, c_dim // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )

    q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    attn = q @ k.transpose(-2, -1)

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(b_dim, n_dim, c_dim)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

class LayerScale(nn.Module):
  """Layer scale, following DINOv2 implementation."""

  def __init__(
      self,
      dim: int,
      init_values: Union[float, torch.Tensor] = 1e-5,
      inplace: bool = False,
  ) -> None:
    super().__init__()
    self.inplace = inplace
    self.gamma = nn.Parameter(init_values * torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.gamma) if self.inplace else x * self.gamma

def drop_path_impl(x, drop_prob: float = 0.0, training: bool = False):
  if drop_prob == 0.0 or not training:
    return x
  keep_prob = 1 - drop_prob
  shape = (x.shape[0],) + (1,) * (
      x.ndim - 1
  )  # work with diff dim tensors, not just 2D ConvNets
  random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
  if keep_prob > 0.0:
    random_tensor.div_(keep_prob)
  output = x * random_tensor
  return output

class DropPath(nn.Module):
  """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

  def __init__(self, drop_prob=None):
    super(DropPath, self).__init__()
    self.drop_prob = drop_prob

  def forward(self, x):
    return drop_path_impl(x, self.drop_prob, self.training)

class Block(nn.Module):
  """Transformer Block Implementation, following DINOv2 implementation."""

  def __init__(
      self,
      dim: int,
      num_heads: int,
      mlp_ratio: float = 4.0,
      qkv_bias: bool = False,
      proj_bias: bool = True,
      ffn_bias: bool = True,
      drop: float = 0.0,
      attn_drop: float = 0.0,
      init_values=None,
      drop_path: float = 0.0,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
      attn_class: Callable[..., nn.Module] = Attention,
      ffn_layer: Callable[..., nn.Module] = Mlp,
  ) -> None:
    super().__init__()
    self.norm1 = norm_layer(dim)
    self.attn = attn_class(
        dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        proj_bias=proj_bias,
        attn_drop=attn_drop,
        proj_drop=drop,
    )
    self.ls1 = (
        LayerScale(dim, init_values=init_values)
        if init_values
        else nn.Identity()
    )
    self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    self.norm2 = norm_layer(dim)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = ffn_layer(
        in_features=dim,
        hidden_features=mlp_hidden_dim,
        act_layer=act_layer,
        drop=drop,
        bias=ffn_bias,
    )
    self.ls2 = (
        LayerScale(dim, init_values=init_values)
        if init_values
        else nn.Identity()
    )
    self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    self.sample_drop_ratio = drop_path

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    def attn_residual_func(x: torch.Tensor) -> torch.Tensor:
      return self.ls1(self.attn(self.norm1(x)))

    def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
      return self.ls2(self.mlp(self.norm2(x)))

    if self.training and self.sample_drop_ratio > 0.1:
      # the overhead is compensated only for a drop path rate larger than 0.1
      x = drop_add_residual_stochastic_depth(
          x,
          residual_func=attn_residual_func,
          sample_drop_ratio=self.sample_drop_ratio,
      )
      x = drop_add_residual_stochastic_depth(
          x,
          residual_func=ffn_residual_func,
          sample_drop_ratio=self.sample_drop_ratio,
      )
    elif self.training and self.sample_drop_ratio > 0.0:
      x = x + self.drop_path1(attn_residual_func(x))
      x = x + self.drop_path1(ffn_residual_func(x))
    else:
      x = x + attn_residual_func(x)
      x = x + ffn_residual_func(x)
    return x

def drop_add_residual_stochastic_depth(
    x: torch.Tensor,
    residual_func: Callable[[torch.Tensor], torch.Tensor],
    sample_drop_ratio: float = 0.0,
) -> torch.Tensor:
  """This function is taken from the original implementation in DINOv2 to implement stochastic depth in the image encoder."""
  # 1) extract subset using permutation
  b, _, _ = x.shape
  sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
  brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
  x_subset = x[brange]

  # 2) apply residual_func to get residual
  residual = residual_func(x_subset)

  x_flat = x.flatten(1)
  residual = residual.flatten(1)

  residual_scale_factor = b / sample_subset_size

  # 3) add the residual
  x_plus_residual = torch.index_add(
      x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
  )
  return x_plus_residual.view_as(x)

def named_apply(
    fn: Callable,  # pylint: disable=g-bare-generic
    module: nn.Module,
    name="",
    depth_first=True,
    include_root=False,
) -> nn.Module:
  """Apply a function to a module and its children."""
  if not depth_first and include_root:
    fn(module=module, name=name)
  for child_name, child_module in module.named_children():
    child_name = ".".join((name, child_name)) if name else child_name
    named_apply(
        fn=fn,
        module=child_module,
        name=child_name,
        depth_first=depth_first,
        include_root=True,
    )
  if depth_first and include_root:
    fn(module=module, name=name)
  return module

class BlockChunk(nn.ModuleList):

  def forward(self, x):
    for b in self:
      x = b(x)
    return x

class TipsV2VisionTransformer(nn.Module):
  """TipsV2VisionTransformer implementation."""

  def __init__(
      self,
      img_size=224,
      patch_size=16,
      in_chans=3,
      embed_dim=768,
      depth=12,
      num_heads=12,
      mlp_ratio=4.0,
      qkv_bias=True,
      ffn_bias=True,
      proj_bias=True,
      drop_path_rate=0.0,
      drop_path_uniform=False,
      init_values=None,  # for layerscale: None or 0 => no layerscale
      embed_layer=PatchEmbed,
      act_layer=nn.GELU,
      block_fn=Block,
      ffn_layer="mlp",
      block_chunks=1,
      num_register_tokens=0,
      interpolate_antialias=False,
      interpolate_offset=0.1,
      **kwargs,
  ):
    if kwargs:
        #logging.warning("TipsV2VisionTransformer: unused kwargs: %s", kwargs.keys())
        pass
    
    """Defines the Vision Transformer model.

    Args:
      img_size (int, tuple): input image size
      patch_size (int, tuple): patch size
      in_chans (int): number of input channels
      embed_dim (int): embedding dimension
      depth (int): depth of transformer
      num_heads (int): number of attention heads
      mlp_ratio (int): ratio of mlp hidden dim to embedding dim
      qkv_bias (bool): enable bias for qkv if True
      ffn_bias (bool): enable bias for ffn if True
      proj_bias (bool): enable bias for proj in attn if True
      drop_path_rate (float): stochastic depth rate
      drop_path_uniform (bool): apply uniform drop rate across blocks
      init_values (float): layer-scale init values
      embed_layer (nn.Module): patch embedding layer
      act_layer (nn.Module): MLP activation layer
      block_fn (nn.Module): transformer block class
      ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
      block_chunks: (int) split block sequence into block_chunks units for FSDP
        wrap
      num_register_tokens: (int) number of extra cls tokens (so-called
        "registers")
      interpolate_antialias: (str) flag to apply anti-aliasing when
        interpolating positional embeddings
      interpolate_offset: (float) work-around offset to apply when interpolating
        positional embeddings
    """
    super().__init__()
    norm_layer = functools.partial(nn.LayerNorm, eps=1e-6)

    self.num_features = self.embed_dim = (
        embed_dim  # num_features for consistency with other models
    )
    self.num_tokens = 1
    self.n_blocks = depth
    self.num_heads = num_heads
    self.patch_size = patch_size
    self.num_register_tokens = num_register_tokens
    self.interpolate_antialias = interpolate_antialias
    self.interpolate_offset = interpolate_offset

    self.patch_embed = embed_layer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
    )
    num_patches = self.patch_embed.num_patches

    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(
        torch.zeros(1, num_patches + self.num_tokens, embed_dim)
    )
    assert num_register_tokens >= 0
    self.register_tokens = (
        nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        if num_register_tokens
        else None
    )

    if drop_path_uniform:
      dpr = [drop_path_rate] * depth
    else:
      dpr = [
          drop_path_rate * i / max(depth - 1, 1) for i in range(depth)
      ]  # stochastic depth decay rule

    if ffn_layer == "mlp":
      ffn_layer = Mlp
    elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
      ffn_layer = SwiGLUFFNFused
    else:
      raise NotImplementedError

    blocks_list = [
        block_fn(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop_path=dpr[i],
            norm_layer=norm_layer,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            init_values=init_values,
        )
        for i in range(depth)
    ]
    if block_chunks > 0:
      self.chunked_blocks = True
      chunked_blocks = []
      chunksize = depth // block_chunks
      for i in range(0, depth, chunksize):
        # this is to keep the block index consistent if we chunk the block list
        chunked_blocks.append(
            [nn.Identity()] * i + blocks_list[i : i + chunksize]
        )
      self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
    else:
      self.chunked_blocks = False
      self.blocks = nn.ModuleList(blocks_list)

    self.norm = norm_layer(embed_dim)
    self.head = nn.Identity()

    self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

    self.init_weights()

  def init_weights(self):
    nn.init.trunc_normal_(self.pos_embed, std=0.02)
    nn.init.normal_(self.cls_token, std=1e-6)
    if self.register_tokens is not None:
      nn.init.normal_(self.register_tokens, std=1e-6)
    named_apply(init_weights_vit_timm, self)

  def interpolate_pos_encoding(self, x, w, h):
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    num_patches = self.pos_embed.shape[1] - 1
    if npatch == num_patches and w == h:
      return self.pos_embed
    pos_embed = self.pos_embed.float()
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // self.patch_size
    h0 = h // self.patch_size
    num_patches_dim = int(
        math.sqrt(num_patches)
    )  # Recover the number of patches in each dimension
    assert num_patches == num_patches_dim * num_patches_dim
    kwargs = {}
    if self.interpolate_offset:
      sx = float(w0 + self.interpolate_offset) / num_patches_dim
      sy = float(h0 + self.interpolate_offset) / num_patches_dim
      kwargs["scale_factor"] = (sx, sy)
    else:
      # Simply specify an output size instead of a scale factor
      kwargs["size"] = (w0, h0)
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(
            1, num_patches_dim, num_patches_dim, dim
        ).permute(0, 3, 1, 2),
        mode="bilinear",
        antialias=self.interpolate_antialias,
        **kwargs,
    )
    assert (w0, h0) == patch_pos_embed.shape[-2:]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
        previous_dtype
    )

  def prepare_tokens_with_masks(self, x, masks=None):
    _, _, w, h = x.shape
    x = self.patch_embed(x)
    if masks is not None:
      x = torch.where(
          masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
      )

    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + self.interpolate_pos_encoding(x, w, h)

    if self.register_tokens is not None:
      x = torch.cat(
          (
              x[:, :1],
              self.register_tokens.expand(x.shape[0], -1, -1),
              x[:, 1:],
          ),
          dim=1,
      )

    return x

  def forward_features_list(self, x_list, masks_list):
    x = [
        self.prepare_tokens_with_masks(x, masks)
        for x, masks in zip(x_list, masks_list)
    ]
    for blk in self.blocks:
      x = blk(x)

    all_x = x
    output = []
    for x, masks in zip(all_x, masks_list):
      x_norm = self.norm(x)
      output.append({
          "x_norm_1st_clstoken": x_norm[:, :1],
          "x_norm_2nd_clstoken": x_norm[:, 1 : self.num_register_tokens + 1],
          "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
          "x_prenorm": x,
          "masks": masks,
      })
    return output

  def forward_features(self, x, masks=None):
    if isinstance(x, list):
      return self.forward_features_list(x, masks)

    x = self.prepare_tokens_with_masks(x, masks)

    for blk in self.blocks:
      x = blk(x)

    x_norm = self.norm(x)
    return {
        "x_norm_1st_clstoken": x_norm[:, :1],
        "x_norm_2nd_clstoken": x_norm[:, 1 : self.num_register_tokens + 1],
        "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
        "x_prenorm": x,
        "masks": masks,
    }

  def _get_intermediate_layers_not_chunked(self, x, n=1):
    x = self.prepare_tokens_with_masks(x)
    # If n is an int, take the n last blocks. If it's a list, take them
    output, total_block_len = [], len(self.blocks)
    blocks_to_take = (
        range(total_block_len - n, total_block_len) if isinstance(n, int) else n
    )
    for i, blk in enumerate(self.blocks):
      x = blk(x)
      if i in blocks_to_take:
        output.append(x)
    assert len(output) == len(
        blocks_to_take
    ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
    return output

  def _get_intermediate_layers_chunked(self, x, n=1):
    x = self.prepare_tokens_with_masks(x)
    output, i, total_block_len = [], 0, len(self.blocks[-1])
    # If n is an int, take the n last blocks. If it's a list, take them
    blocks_to_take = (
        range(total_block_len - n, total_block_len) if isinstance(n, int) else n
    )
    for block_chunk in self.blocks:
      for blk in block_chunk[i:]:  # Passing the nn.Identity()
        x = blk(x)
        if i in blocks_to_take:
          output.append(x)
        i += 1
    assert len(output) == len(
        blocks_to_take
    ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
    return output

  def get_intermediate_layers(
      self,
      x: torch.torch.Tensor,
      n: Union[int, Sequence] = 1,  # Layers or n last layers to take  # pylint: disable=g-bare-generic
      reshape: bool = False,
      return_class_token: bool = False,
      norm=True,
  ) -> Tuple[Union[torch.torch.Tensor, Tuple[torch.torch.Tensor]]]:  # pylint: disable=g-one-element-tuple
    if self.chunked_blocks:
      outputs = self._get_intermediate_layers_chunked(x, n)
    else:
      outputs = self._get_intermediate_layers_not_chunked(x, n)
    if norm:
      outputs = [self.norm(out) for out in outputs]
    class_tokens = [out[:, 0] for out in outputs]
    outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
    if reshape:
      batch_size, _, w, h = x.shape
      outputs = [
          out.reshape(
              batch_size, w // self.patch_size, h // self.patch_size, -1
          )
          .permute(0, 3, 1, 2)
          .contiguous()
          for out in outputs
      ]
    if return_class_token:
      return tuple(zip(outputs, class_tokens))
    return tuple(outputs)

  def forward(self, *args, is_training=False, **kwargs):
    ret = self.forward_features(*args, **kwargs)
    if is_training:
      return ret
    else:
      return self.head(ret["x_norm_1st_clstoken"]), self.head(
          ret["x_norm_2nd_clstoken"]
      ), ret["x_norm_patchtokens"]

def init_weights_vit_timm(module: nn.Module, name: str = ""):  # pylint: disable=unused-argument
  """ViT weight initialization, original timm impl (for reproducibility)."""
  if isinstance(module, nn.Linear):
    nn.init.trunc_normal_(module.weight, std=0.02)
    if module.bias is not None:
      nn.init.zeros_(module.bias)

# ── Pretrained configs ────────────────────────────────────────────────────────
default_cfgs = {
    "tipsv2_b14": dict(
        # hf_hub_id="SankethSingh/tipsv2_b14_timm",
        input_size=(3, 448, 448),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    ),
    "tipsv2_l14": dict(
        # hf_hub_id="SankethSingh/tipsv2_l14_timm",
        input_size=(3, 448, 448),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    ),
    # add so400m, giant later
}


# ── Model registration ────────────────────────────────────────────────────────
@register_model
def tipsv2_b14(pretrained=False, **kwargs):
    kwargs.pop("default_cfg", None)

    model_args = dict(
        img_size=448,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_register_tokens=1,
        ffn_layer="swiglu",
        interpolate_antialias=True,
        interpolate_offset=0.0,
        init_values=1e-5,
    )
    model_args.update(kwargs)

    return build_model_with_cfg(
        TipsV2VisionTransformer,
        "tipsv2_b14",
        pretrained=pretrained,
        default_cfg=default_cfgs["tipsv2_b14"],
        **model_args,
    )


@register_model
def tipsv2_l14(pretrained=False, **kwargs):
    kwargs.pop("default_cfg", None)

    model_args = dict(
        img_size=448,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_register_tokens=1,
        ffn_layer="swiglu",
        interpolate_antialias=True,
        interpolate_offset=0.0,
        init_values=1e-5,
    )
    model_args.update(kwargs)

    return build_model_with_cfg(
        TipsV2VisionTransformer,
        "tipsv2_l14",
        pretrained=pretrained,
        default_cfg=default_cfgs["tipsv2_l14"],
        **model_args,
    )