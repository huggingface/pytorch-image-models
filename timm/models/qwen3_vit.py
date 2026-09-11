"""Qwen3 Vision Transformer

Vision encoder of the Qwen3-VL / Qwen3.5 / Qwen3.8 multimodal models from Alibaba Qwen team, and of the
driving-domain Qwen-Drive-1.0 built on that same tower.

A plain pre-norm ViT (SigLIP-2 style widths, GELU-tanh MLP, fused QKV with bias) with two position encodings
applied together: a learned absolute grid (48x48 for the released models, bilinearly resampled to the input grid
with align_corners=True) and axial 2D RoPE (theta 10000, per-axis frequency blocks, 'half' rotation layout).
There is no final norm on the encoder patch tokens. Classifier variants wrap the encoder with average pooling,
normalization, and a linear classification head. The separate `_enc` variants preserve the native VLM projector
('merger'), which pixel-unshuffles 2x2 patch tokens and maps them to the LLM width via `out_features`.
Classifiers can also retain this projector with `encoder_pool='merge'`; `_merge` variants enable it by default.

For example, `qwen3_vit_306m.qwen3_vl_4b` with `num_classes=45` returns image logits (B, 45), while
`qwen3_vit_306m_enc.qwen3_vl_4b` returns projected tokens (B, H*W/4, 2560). Encoder checkpoint weights
load into the classifier under `encoder.*`; its classification head is initialized separately.

Weights are native timm remaps of the source VLM vision tensors. The Conv3d patch embedding over
`temporal_patch_size=2` frames is folded into a Conv2d since the image path feeds the same frame twice
(mathematically identical, verified against the transformers reference).

Reference: https://github.com/QwenLM/Qwen3-VL, https://github.com/QwenLM/Qwen-Drive-1.0,
transformers `Qwen3VLVisionModel` / `Qwen3_5VisionModel`.
Weights are released under Apache-2.0 by the Qwen team, except Qwen3.8-Flash-Next (Qwen Community License 1.0).

Copyright 2026 Yonghye Kwon
"""

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import (
    LayerNorm,
    PatchEmbed,
    RotaryEmbeddingCat,
    calculate_drop_path_rates,
    get_device_dtype,
    register_notrace_function,
    trunc_normal_,
)
from timm.layers.trace_utils import _assert
from ._builder import build_model_with_cfg, resolve_pretrained_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint
from ._registry import generate_default_cfgs, register_model
from .eva import EvaBlock

__all__ = ['Qwen3VitEncoder', 'Qwen3VitClassifier']


@register_notrace_function
def resample_pos_embed_grid(pos_embed: torch.Tensor, grid_size: int, new_size: Tuple[int, int]) -> torch.Tensor:
    """Resample a square learned pos embed (1, G*G, C) to (1, H*W, C) the way the Qwen vision encoder does it:
    bilinear with align_corners=True. Identity when the grid already matches."""
    H, W = new_size
    if H == grid_size and W == grid_size:
        return pos_embed
    C = pos_embed.shape[-1]
    # interpolate in at least float32 (half types lack CPU bilinear support and lose precision)
    calc_dtype = pos_embed.dtype if pos_embed.dtype == torch.float64 else torch.float32
    x = pos_embed.reshape(1, grid_size, grid_size, C).permute(0, 3, 1, 2).to(calc_dtype)
    x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
    return x.permute(0, 2, 3, 1).reshape(1, H * W, C).to(pos_embed.dtype)


class Qwen3VitPatchMerger(nn.Module):
    """Qwen VLM projector: per-token norm, spatial `k x k` pixel-unshuffle, MLP to the LLM width.

    `out_features` is the source LLM hidden size. `pre_logits` returns the `k*k*dim` merged token
    after the first projection + activation, before the final VLM projection.
    """

    def __init__(
            self,
            dim: int,
            merge_size: int = 2,
            out_features: int = 0,
            norm_layer: Callable = partial(LayerNorm, eps=1e-6),
            act_layer: Callable = nn.GELU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.merge_size = merge_size
        self.hidden_size = dim * merge_size**2
        self.out_features = out_features if out_features > 0 else self.hidden_size
        self.norm = norm_layer(dim, **dd)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, **dd)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_size, out_features, **dd) if out_features > 0 else nn.Identity()

    def merge(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) -> (B, H*W / k^2, k*k*C), tokens of each k x k block concatenated in row-major order."""
        B, H, W, C = x.shape
        k = self.merge_size
        # two asserts: `and` on traced shapes is control flow for torch.fx
        _assert(H % k == 0, 'patch grid height must be divisible by merge_size')
        _assert(W % k == 0, 'patch grid width must be divisible by merge_size')
        x = x.reshape(B, H // k, k, W // k, k, C).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(B, (H // k) * (W // k), k * k * C)

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.merge(self.norm(x))
        x = self.act(self.fc1(x))
        return x if pre_logits else self.fc2(x)


class Qwen3VitEncoder(nn.Module):
    """Qwen3-VL / Qwen3.5 / Qwen3.8 vision encoder.

    `forward_features` returns the un-normalized patch tokens as (B, H, W, C) (`output_fmt='NHWC'`, matches the
    reference `last_hidden_state` up to the reference's spatial-merge token ordering). Output of `forward` depends
    on `global_pool`:
      * 'merge' (default): projector output (B, H*W/4, out_features) -- the tokens the LLM consumes
      * 'avg': mean over patch tokens -> (B, embed_dim)
      * '': raw patch features -> (B, H, W, embed_dim), identical to forward_features
    """

    output_fmt: str = 'NHWC'

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 768,
            patch_size: int = 16,
            in_chans: int = 3,
            out_features: int = 0,
            global_pool: str = 'merge',
            embed_dim: int = 1152,
            depth: int = 27,
            num_heads: int = 16,
            mlp_ratio: float = 4304 / 1152,
            merge_size: int = 2,
            pos_embed_grid_size: int = 48,
            rope_temperature: float = 10000.0,
            norm_eps: float = 1e-6,
            act_layer: Optional[Callable] = None,
            proj_drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            device=None,
            dtype=None,
    ):
        """
        Args:
            img_size: Nominal input size, only used for the feature-info reduction / default grid. Any input whose
                patch grid is divisible by `merge_size` works (the pos embed is resampled per input).
            patch_size: Patch size.
            in_chans: Number of input channels.
            out_features: Output width of the VLM projector (source LLM hidden size). Zero disables its last
                projection. This is independent of the number of classes in a downstream classifier.
            global_pool: 'merge' (projector), 'avg' (mean pool) or '' (raw tokens).
            embed_dim: Token width.
            depth: Number of transformer blocks.
            num_heads: Attention heads.
            mlp_ratio: MLP hidden / embed_dim.
            merge_size: Spatial merge factor of the projector.
            pos_embed_grid_size: Side of the learned absolute position embedding grid.
            rope_temperature: RoPE base (theta).
            norm_eps: LayerNorm eps (blocks and projector).
            act_layer: Block MLP activation, default GELU (tanh approximation) as in the reference.
            proj_drop_rate: Dropout on attention / MLP projections.
            attn_drop_rate: Attention dropout.
            drop_path_rate: Stochastic depth rate.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert global_pool in ('', 'avg', 'merge')
        assert embed_dim % num_heads == 0
        self.num_classes = 0
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.out_features = out_features
        self.merge_size = merge_size
        self.pos_embed_grid_size = pos_embed_grid_size
        self.num_prefix_tokens = 0
        self.feature_dim = -1  # channels-last (B, H, W, C) features
        self.grad_checkpointing = False

        norm_layer = partial(LayerNorm, eps=norm_eps)
        act_layer = act_layer or partial(nn.GELU, approximate='tanh')
        head_dim = embed_dim // num_heads

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=True,
            strict_img_size=False,
            output_fmt='NHWC',
            **dd,
        )
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size
        self.pos_embed = nn.Parameter(torch.empty(1, pos_embed_grid_size**2, embed_dim, **dd))
        # Axial 2D RoPE on integer patch coordinates, head_dim // 4 frequencies per axis laid out as
        # [h-block, w-block] and tiled for the 'half' rotation -- the Qwen vision RoPE layout.
        self.rope = RotaryEmbeddingCat(
            dim=head_dim,
            temperature=rope_temperature,
            in_pixels=False,
            feat_shape=None,
            rotate_half=True,
            **dd,
        )

        dpr = calculate_drop_path_rates(drop_path_rate, depth)
        self.blocks = nn.ModuleList([
            EvaBlock(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=True,
                qkv_fused=True,
                mlp_ratio=mlp_ratio,
                num_prefix_tokens=0,
                attn_type='rope',
                rotate_half=True,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                **dd,
            )
            for i in range(depth)
        ])
        self.feature_info = [dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]

        self.merger = (
            Qwen3VitPatchMerger(
                embed_dim,
                merge_size=merge_size,
                out_features=out_features,
                norm_layer=norm_layer,
                **dd,
            )
            if global_pool == 'merge'
            else None
        )

        self.init_weights(needs_reset=False)

    @torch.jit.ignore
    def init_weights(self, needs_reset: bool = True) -> None:
        """Initialize weights and, after ``to_empty()``, restore parameters and RoPE buffers.

        Args:
            needs_reset: Reset modules that self-initialize in their constructors. False during construction.
        """
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(partial(self._init_weights, needs_reset=needs_reset))

    def _init_weights(self, m: nn.Module, needs_reset: bool = True) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif needs_reset and hasattr(m, 'reset_parameters') and m is not self:
            m.reset_parameters()

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return {'pos_embed'}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^patch_embed|pos_embed|rope',
            blocks=[(r'^blocks\.(\d+)', None), (r'^merger', (99999,))],
        )

    def _pos_embed(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add the (resampled) learned pos embed to NHWC patches, return flattened tokens + rope table."""
        B, H, W, C = x.shape
        pos_embed = resample_pos_embed_grid(self.pos_embed, self.pos_embed_grid_size, (H, W))
        x = x.reshape(B, H * W, C) + pos_embed.to(x.dtype)
        rope = self.rope.get_embed(shape=(H, W))
        return x, rope

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            output_dict: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Unused, the model has no prefix tokens.
            norm: Unused, the model has no final norm.
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs ('NCHW' or 'NLC')
            intermediates_only: Only return intermediate features
            output_dict: Return 'image_intermediates' and, unless intermediates_only, 'image_features' in a dict
        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x, rope = self._pos_embed(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rope)
            else:
                x = blk(x, rope=rope)
            if i in take_indices:
                intermediates.append(x)

        if reshape:
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        if intermediates_only:
            return {'image_intermediates': intermediates} if output_dict else intermediates

        x = x.reshape(B, H, W, -1)
        return {'image_features': x, 'image_intermediates': intermediates} if output_dict else (x, intermediates)

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ) -> List[int]:
        """Prune blocks and optionally the native projector for intermediate feature extraction.

        ``prune_norm`` is unused because the encoder has no final backbone norm.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_head:
            self.merger = None
            self.global_pool = ''
            self.out_features = 0
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x, rope = self._pos_embed(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rope)
            else:
                x = blk(x, rope=rope)
        return x.reshape(B, H, W, -1)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        raise NotImplementedError('Qwen3VitEncoder does not support classification use cases.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw patch tokens and apply the configured native pooling/projector."""
        x = self.forward_features(x)
        if self.merger is not None:
            return self.merger(x)
        if self.global_pool == 'avg':
            x = x.mean(dim=(1, 2))
        return x


class Qwen3VitClassifier(nn.Module):
    """Classification wrapper: Qwen encoder patch tokens -> mean pool -> norm -> linear head.

    `encoder_pool='merge'` retains the native VLM projector before classification pooling. `num_classes`
    controls the classifier independently of `out_features`, the native projector's output width.
    """

    output_fmt: str = 'NHWC'

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 768,
            patch_size: int = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            encoder_pool: str = '',
            out_features: int = 0,
            embed_dim: int = 1152,
            depth: int = 27,
            num_heads: int = 16,
            mlp_ratio: float = 4304 / 1152,
            merge_size: int = 2,
            pos_embed_grid_size: int = 48,
            rope_temperature: float = 10000.0,
            norm_eps: float = 1e-6,
            act_layer: Optional[Callable] = None,
            final_norm: bool = True,
            drop_rate: float = 0.0,
            proj_drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert global_pool in ('', 'avg')
        assert encoder_pool in ('', 'merge')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.encoder_pool = encoder_pool
        self.encoder = Qwen3VitEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            global_pool=encoder_pool,
            out_features=out_features,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            merge_size=merge_size,
            pos_embed_grid_size=pos_embed_grid_size,
            rope_temperature=rope_temperature,
            norm_eps=norm_eps,
            act_layer=act_layer,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            **dd,
        )
        self.embed_dim = embed_dim
        self.num_features = self.head_hidden_size = (
            self.encoder.merger.out_features if self.encoder.merger is not None else embed_dim
        )
        self.output_fmt = 'NLC' if encoder_pool == 'merge' else 'NHWC'
        self.norm = LayerNorm(self.num_features, eps=norm_eps, affine=False, **dd) if final_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.num_features, num_classes, **dd) if num_classes > 0 else nn.Identity()
        self.encoder._init_weights(self.head)
        self.num_prefix_tokens = 0
        self.feature_dim = -1
        self.feature_info = [dict(info, module=f'encoder.{info["module"]}') for info in self.encoder.feature_info]

    @torch.jit.ignore
    def init_weights(self, needs_reset: bool = True) -> None:
        """Initialize the encoder and classification head, including after ``to_empty()``."""
        self.encoder.init_weights(needs_reset=needs_reset)
        self.encoder._init_weights(self.head, needs_reset=needs_reset)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return {f'encoder.{k}' for k in self.encoder.no_weight_decay()}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^encoder\.(?:patch_embed|pos_embed|rope)',
            blocks=[(r'^encoder\.blocks\.(\d+)', None), (r'^encoder\.merger|^norm|^head', (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.encoder.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        dd = get_device_dtype(self)
        self.head = nn.Linear(self.num_features, num_classes, **dd) if num_classes > 0 else nn.Identity()
        self.encoder._init_weights(self.head)
        self.head.train(self.training)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-classifier features: raw NHWC patches, or projected NLC tokens when encoder_pool='merge'."""
        return self.encoder(x)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = x.flatten(1, -2)
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        x = self.head_drop(self.norm(x))
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            output_dict: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
        """Return backbone intermediates and the configured pre-classifier features.

        ``norm`` is unused: the classifier's post-pooling normalization is not applied to patch tokens.
        The merger, when enabled, applies only to the final features, not the intermediate maps.
        ``output_dict`` returns 'image_intermediates' and, unless ``intermediates_only``, 'image_features'.
        """
        x, intermediates = self.encoder.forward_intermediates(
            x,
            indices=indices,
            return_prefix_tokens=return_prefix_tokens,
            norm=norm,
            stop_early=stop_early,
            output_fmt=output_fmt,
            intermediates_only=False,
        )
        if intermediates_only:
            return {'image_intermediates': intermediates} if output_dict else intermediates
        if self.encoder.merger is not None:
            x = self.encoder.merger(x)
        return {'image_features': x, 'image_intermediates': intermediates} if output_dict else (x, intermediates)

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ) -> List[int]:
        """Prune encoder blocks and the classifier; ``prune_norm`` has no backbone norm to remove."""
        take_indices = self.encoder.prune_intermediate_layers(indices, prune_norm=prune_norm, prune_head=False)
        if prune_head:
            self.reset_classifier(0)
        return take_indices


def checkpoint_filter_fn_encoder(
        state_dict: Dict[str, torch.Tensor],
        model: Qwen3VitEncoder,
) -> Dict[str, torch.Tensor]:
    """Remap `model.visual.*` tensors of a Qwen3-VL / Qwen3.5 / Qwen3.8 / Qwen-Drive checkpoint to timm keys.

    Every non-vision tensor (the LLM) is dropped, so a raw VLM shard can be passed directly.
    """
    out_dict = {}
    for k, v in state_dict.items():
        # `vlm.model.visual.` is Qwen-Drive, which nests the whole VLM under a `vlm.` attribute
        for prefix in ('vlm.model.visual.', 'model.visual.', 'visual.', 'encoder.'):
            if k.startswith(prefix):
                k = k[len(prefix) :]
                break
        else:
            if not k.startswith(('patch_embed.', 'pos_embed', 'blocks.', 'merger.')):
                continue  # LLM / other tensors

        if k.startswith('rotary_pos_emb') or k.startswith('deepstack_merger_list'):
            # rope table is regenerated, deepstack projectors (Qwen3-VL) are not carried by the backbone
            continue
        if k == 'patch_embed.proj.weight' and v.ndim == 5:
            # Conv3d over temporal_patch_size frames -> Conv2d. Images are fed as the same frame repeated, so the
            # 3d conv equals a 2d conv with the temporal kernel slices summed. Sum in float64 so the (bf16) slices
            # are combined exactly and rounded once to the model dtype: summing in bf16 gave a 4e-3 patch-embed
            # error, summing in fp32 still left a ~1e-10 residual in float64 checks against the reference.
            v = v.double().sum(dim=2)
        if k == 'pos_embed.weight':
            k = 'pos_embed'
            v = v.unsqueeze(0)
        if k.startswith('merger.'):
            if model.merger is None:
                continue
            k = k.replace('merger.linear_fc1.', 'merger.fc1.').replace('merger.linear_fc2.', 'merger.fc2.')
            if k.startswith('merger.fc2.') and isinstance(model.merger.fc2, nn.Identity):
                continue
        k = k.replace('.mlp.linear_fc1.', '.mlp.fc1.').replace('.mlp.linear_fc2.', '.mlp.fc2.')
        out_dict[k] = v
    return out_dict


def checkpoint_filter_fn_classifier(
        state_dict: Dict[str, torch.Tensor],
        model: Qwen3VitClassifier,
) -> Dict[str, torch.Tensor]:
    """Load HF vision / timm encoder weights, or round-trip a fine-tuned classifier checkpoint."""
    local = {k: v for k, v in state_dict.items() if k.startswith(('norm.', 'head.'))}
    encoder = checkpoint_filter_fn_encoder({k: v for k, v in state_dict.items() if k not in local}, model.encoder)
    return {**{f'encoder.{k}': v for k, v in encoder.items()}, **local}


def _create_qwen3_vit_classifier(variant: str, pretrained: bool = False, **kwargs) -> Qwen3VitClassifier:
    pretrained_cfg = resolve_pretrained_cfg(
        variant,
        pretrained_cfg=kwargs.pop('pretrained_cfg', None),
        pretrained_cfg_overlay=kwargs.pop('pretrained_cfg_overlay', None),
    )
    if kwargs.get('encoder_pool') == 'merge' and 'out_features' not in kwargs:
        kwargs['out_features'] = _resolve_qwen3_out_features(variant, pretrained_cfg.tag)
    out_indices = kwargs.pop('out_indices', 3)
    return build_model_with_cfg(
        Qwen3VitClassifier,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn_classifier,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )


def _create_qwen3_vit_encoder(variant: str, pretrained: bool = False, **kwargs) -> Qwen3VitEncoder:
    pretrained_cfg = resolve_pretrained_cfg(
        variant,
        pretrained_cfg=kwargs.pop('pretrained_cfg', None),
        pretrained_cfg_overlay=kwargs.pop('pretrained_cfg_overlay', None),
    )
    if 'out_features' not in kwargs:
        kwargs['out_features'] = _resolve_qwen3_out_features(variant, pretrained_cfg.tag)
    out_indices = kwargs.pop('out_indices', 3)
    return build_model_with_cfg(
        Qwen3VitEncoder,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn_encoder,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        kwargs_filter=('num_classes', 'drop_rate'),
        **kwargs,
    )


def _resolve_qwen3_out_features(variant: str, tag: str) -> int:
    source_arch = variant.rsplit('_', 1)[0] if variant.endswith(('_enc', '_merge')) else variant
    source_name = f'{source_arch}.{tag}'
    if source_name not in _SOURCE_CFGS:
        raise ValueError('Specify out_features for a Qwen projector with a custom pretrained configuration.')
    return _SOURCE_CFGS[source_name]['out_features']


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        # native 48x48 learned pos-embed grid, any size with an even patch grid works (dynamic resample + rope)
        'input_size': (3, 768, 768),
        'pool_size': None,
        'fixed_input_size': False,
        'crop_pct': 1.0,
        'crop_mode': 'squash',
        'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'num_classes': 0,
        'license': 'apache-2.0',
        **kwargs,
    }


_SOURCE_CFGS = {
    # Upstream checkpoints for conversion (the shard holding `model.visual.*`).
    # `out_features` is the LLM hidden size of the encoder projector, independent of classifier labels.
    'qwen3_vit_88m.qwen3_5_0_8b': _cfg(
        hf_hub_id='Qwen/Qwen3.5-0.8B',
        hf_hub_filename='model.safetensors-00001-of-00001.safetensors',
        out_features=1024,
        origin_url='https://huggingface.co/Qwen/Qwen3.5-0.8B',
    ),
    'qwen3_vit_306m.qwen3_5_2b': _cfg(
        hf_hub_id='Qwen/Qwen3.5-2B',
        hf_hub_filename='model.safetensors-00001-of-00001.safetensors',
        out_features=2048,
        origin_url='https://huggingface.co/Qwen/Qwen3.5-2B',
    ),
    'qwen3_vit_306m.qwen3_5_4b': _cfg(
        hf_hub_id='Qwen/Qwen3.5-4B',
        hf_hub_filename='model.safetensors-00002-of-00002.safetensors',
        out_features=2560,
        origin_url='https://huggingface.co/Qwen/Qwen3.5-4B',
    ),
    'qwen3_vit_416m.qwen3_8_27b': _cfg(
        hf_hub_id='Qwen/Qwen3.8-27B',
        hf_hub_filename='model-00001-of-00018.safetensors',
        out_features=5120,
        origin_url='https://huggingface.co/Qwen/Qwen3.8-27B',
    ),
    'qwen3_vit_416m.qwen3_8_flash_next': _cfg(
        hf_hub_id='Qwen/Qwen3.8-Flash-Next',
        hf_hub_filename='model-00001-of-00131.safetensors',
        out_features=2560,
        origin_url='https://huggingface.co/Qwen/Qwen3.8-Flash-Next',
        license='qwen-community-1.0',
    ),
    'qwen3_vit_416m.qwen3_5_27b': _cfg(
        hf_hub_id='Qwen/Qwen3.5-27B',
        hf_hub_filename='model.safetensors-00011-of-00011.safetensors',
        out_features=5120,
        origin_url='https://huggingface.co/Qwen/Qwen3.5-27B',
    ),
    'qwen3_vit_416m.qwen3_5_9b': _cfg(
        hf_hub_id='Qwen/Qwen3.5-9B',
        hf_hub_filename='model.safetensors-00004-of-00004.safetensors',
        out_features=4096,
        origin_url='https://huggingface.co/Qwen/Qwen3.5-9B',
    ),
    # Qwen3-VL towers: same architecture. The three DeepStack projectors these checkpoints carry
    # (`deepstack_merger_list`, mid-block tokens -> LLM width) are not part of the backbone and are dropped,
    # the backbone and main projector load fully; use forward_intermediates() for the mid-block tokens.
    'qwen3_vit_306m.qwen3_vl_4b': _cfg(
        hf_hub_id='Qwen/Qwen3-VL-4B-Instruct',
        hf_hub_filename='model-00002-of-00002.safetensors',
        out_features=2560,
        origin_url='https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct',
    ),
    'qwen3_vit_416m.qwen3_vl_8b': _cfg(
        hf_hub_id='Qwen/Qwen3-VL-8B-Instruct',
        hf_hub_filename='model-00004-of-00004.safetensors',
        out_features=4096,
        origin_url='https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct',
    ),
    'qwen3_vit_416m.qwen3_vl_32b': _cfg(
        hf_hub_id='Qwen/Qwen3-VL-32B-Instruct',
        hf_hub_filename='model-00014-of-00014.safetensors',
        out_features=5120,
        origin_url='https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct',
    ),
    'qwen3_vit_416m.qwen3_vl_30b_a3b': _cfg(
        hf_hub_id='Qwen/Qwen3-VL-30B-A3B-Instruct',
        hf_hub_filename='model-00013-of-00013.safetensors',
        out_features=2048,
        origin_url='https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct',
    ),
    'qwen3_vit_416m.qwen3_vl_235b_a22b': _cfg(
        hf_hub_id='Qwen/Qwen3-VL-235B-A22B-Instruct',
        hf_hub_filename='model-00001-of-00096.safetensors',
        out_features=4096,
        origin_url='https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct',
    ),
    # Qwen-Drive-1.0: the Qwen3.5-4B tower after driving-domain training. Same architecture and projector width,
    # but the weights moved (mean |diff| vs Qwen3.5-4B ~2e-5 in early blocks rising to ~3e-4 at the projector),
    # so it is a distinct tag rather than an alias. Its checkpoint nests the VLM under `vlm.`, and the driving
    # heads (BEV perception, planning expert) live in separate files -- nothing extra to drop here.
    'qwen3_vit_306m.qwen_drive_1_0_4b': _cfg(
        hf_hub_id='Qwen/Qwen-Drive-1.0-4B',
        hf_hub_filename='model.safetensors',
        out_features=2560,
        origin_url='https://huggingface.co/Qwen/Qwen-Drive-1.0-4B',
    ),
}


default_cfgs = generate_default_cfgs({
    f'{name.split(".", 1)[0]}{suffix}.{name.split(".", 1)[1]}': dict(
        ((k, v) for k, v in cfg.items() if k not in ('out_features', 'hf_hub_filename')),
        hf_hub_id='timm/',
        first_conv='patch_embed.proj' if suffix == '_enc' else 'encoder.patch_embed.proj',
        classifier=None if suffix == '_enc' else 'head',
    )
    for name, cfg in _SOURCE_CFGS.items()
    for suffix in ('', '_merge', '_enc')
})


@register_model
def qwen3_vit_88m(pretrained: bool = False, **kwargs) -> Qwen3VitClassifier:
    """Qwen3.5 vision classifier, 12 x 768 (88M w/o projector). Source: Qwen3.5-0.8B."""
    model_args = dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0)
    return _create_qwen3_vit_classifier('qwen3_vit_88m', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def qwen3_vit_88m_merge(pretrained: bool = False, **kwargs) -> Qwen3VitClassifier:
    """Qwen3.5-0.8B vision classifier retaining the native spatial merger and 1024-wide projection."""
    model_args = dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, encoder_pool='merge')
    return _create_qwen3_vit_classifier('qwen3_vit_88m_merge', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def qwen3_vit_306m(pretrained: bool = False, **kwargs) -> Qwen3VitClassifier:
    """Qwen3.5 vision classifier, 24 x 1024 (306M w/o projector). Source: Qwen3.5-2B / 4B, Qwen-Drive-1.0-4B."""
    model_args = dict(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0)
    return _create_qwen3_vit_classifier('qwen3_vit_306m', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def qwen3_vit_306m_merge(pretrained: bool = False, **kwargs) -> Qwen3VitClassifier:
    """Qwen vision classifier retaining the native spatial merger and source LLM projection."""
    model_args = dict(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0, encoder_pool='merge')
    return _create_qwen3_vit_classifier('qwen3_vit_306m_merge', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def qwen3_vit_416m(pretrained: bool = False, **kwargs) -> Qwen3VitClassifier:
    """Qwen3.5 / Qwen3.8 vision classifier, 27 x 1152 (416M w/o projector). Source: Qwen3.5-9B / 27B, Qwen3.8."""
    model_args = dict(embed_dim=1152, depth=27, num_heads=16, mlp_ratio=4304 / 1152)
    return _create_qwen3_vit_classifier('qwen3_vit_416m', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def qwen3_vit_416m_merge(pretrained: bool = False, **kwargs) -> Qwen3VitClassifier:
    """Qwen vision classifier retaining the native spatial merger and source LLM projection."""
    model_args = dict(embed_dim=1152, depth=27, num_heads=16, mlp_ratio=4304 / 1152, encoder_pool='merge')
    return _create_qwen3_vit_classifier('qwen3_vit_416m_merge', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def qwen3_vit_88m_enc(pretrained: bool = False, **kwargs) -> Qwen3VitEncoder:
    """Qwen native vision encoder with spatial merger and source LLM projection."""
    model_args = dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0)
    return _create_qwen3_vit_encoder('qwen3_vit_88m_enc', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def qwen3_vit_306m_enc(pretrained: bool = False, **kwargs) -> Qwen3VitEncoder:
    """Qwen native vision encoder with spatial merger and source LLM projection."""
    model_args = dict(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0)
    return _create_qwen3_vit_encoder('qwen3_vit_306m_enc', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def qwen3_vit_416m_enc(pretrained: bool = False, **kwargs) -> Qwen3VitEncoder:
    """Qwen native vision encoder with spatial merger and source LLM projection."""
    model_args = dict(embed_dim=1152, depth=27, num_heads=16, mlp_ratio=4304 / 1152)
    return _create_qwen3_vit_encoder('qwen3_vit_416m_enc', pretrained=pretrained, **dict(model_args, **kwargs))
