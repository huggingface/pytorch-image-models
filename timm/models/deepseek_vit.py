"""DeepSeek Vision Transformer

Vision encoder of the DeepSeek-V4 / V4.1 multimodal models from DeepSeek-AI.

A plain pre-norm ViT with RMSNorm, fused QKV with bias, a bias-free SwiGLU MLP and axial 2D RoPE
(theta 10000, per-axis frequency blocks, 'half' rotation layout) as the only position encoding --
there is no learned position embedding, so any patch grid works without resampling. Patches are
embedded by a linear projection over flattened `14 x 14` pixel patches (loaded here as the
equivalent Conv2d) and the encoder ends with an RMSNorm over the patch tokens.
Set `dynamic_img_pad=True` to zero-pad arbitrary image sizes to the patch grid on the bottom/right.

Classifier variants wrap the encoder with average pooling, normalization and a linear head. The
separate `_enc` variants preserve the native VLM projector (the DeepSeek 'aligner'), which
pixel-unshuffles `3 x 3` patch tokens and maps them to the LLM width via `out_features`.
Classifiers can also retain this projector with `encoder_pool='align'`; `_align` variants enable it
by default.

For example, `deepseek_vit_412m.deepseek_v4_1_flash` with `num_classes=45` returns image logits
(B, 45), while `deepseek_vit_412m_enc.deepseek_v4_1_flash` returns projected tokens
(B, ceil(H/3)*ceil(W/3), 5120). Encoder checkpoint weights load into the classifier under
`encoder.*`; its classification head is initialized separately.

Weights are remapped on the fly from the DeepSeek checkpoint shard that holds the vision tower, so
`pretrained=True` reads the original `deepseek-ai/*` repos with no re-hosting; every non-vision
(LLM) tensor in that shard is dropped by the checkpoint filter.

Reference: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash (`inference/vision.py`),
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp. Weights are MIT licensed.

Copyright 2026 Yonghye Kwon
"""

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import (
    AttentionRope,
    DropPath,
    GluMlp,
    PatchEmbed,
    RmsNormFp32,
    RotaryEmbeddingCat,
    calculate_drop_path_rates,
    get_device_dtype,
    trunc_normal_,
)
from ._builder import build_model_with_cfg, resolve_pretrained_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint
from ._registry import generate_default_cfgs, register_model

__all__ = ['DeepseekVitEncoder', 'DeepseekVitClassifier']


class DeepseekVitBlock(nn.Module):
    """Pre-norm block: RoPE attention (fused QKV w/ bias) and a bias-free packed SwiGLU MLP.

    The reference keeps bias on the attention projections but not on the MLP, which is why this is a
    local block rather than `EvaBlock` (that one has no MLP bias control).
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 2.75,
            proj_drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            norm_layer: Callable = RmsNormFp32,
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.norm1 = norm_layer(dim, **dd)
        self.attn = AttentionRope(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
            qkv_fused=True,
            num_prefix_tokens=0,
            rotate_half=True,
            proj_bias=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **dd,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim, **dd)
        # packed SwiGLU: one fc1 emitting [gate, up], matching the reference `w1(x).chunk(2, -1)`
        self.mlp = GluMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio) * 2,
            act_layer=nn.SiLU,
            gate_last=False,
            bias=False,
            drop=proj_drop,
            **dd,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, rope: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class DeepseekVitAligner(nn.Module):
    """DeepSeek VLM projector ('aligner'): `r x r` pixel-unshuffle of patch tokens, MLP to the LLM width.

    `out_features` is the source LLM hidden size. `pre_logits` returns the activated first projection,
    before the final one.

    The unshuffle uses channel-major `(C, kh, kw)` order, matching the reference's `F.unfold`.
    Qwen/Gemma's `(kh, kw, C)` order would permute the pretrained projection's inputs.
    Patch grids are zero-padded on the bottom/right to a multiple of `r`.
    """

    def __init__(
            self,
            dim: int,
            downsample_ratio: int = 3,
            out_features: int = 0,
            act_layer: Callable = nn.GELU,
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        assert out_features > 0, 'the DeepSeek aligner projects to the source LLM width, pass out_features'
        assert downsample_ratio > 0
        self.downsample_ratio = downsample_ratio
        self.hidden_size = dim * downsample_ratio**2
        self.out_features = out_features
        self.fc1 = nn.Linear(self.hidden_size, out_features, **dd)
        self.act = act_layer()
        self.fc2 = nn.Linear(out_features, out_features, **dd)

    def unshuffle(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) -> (B, ceil(H/r) * ceil(W/r), C*r*r), channel-major within each `r x r` block."""
        B, H, W, C = x.shape
        r = self.downsample_ratio
        x = x.permute(0, 3, 1, 2)
        # pad unconditionally: `if pad:` would be control flow on traced shapes for torch.fx
        pad_h, pad_w = -H % r, -W % r
        x = F.pad(x, (0, pad_w, 0, pad_h))
        h, w = (H + pad_h) // r, (W + pad_w) // r
        x = x.reshape(B, C, h, r, w, r).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, h * w, C * r * r)

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.act(self.fc1(self.unshuffle(x)))
        return x if pre_logits else self.fc2(x)


class DeepseekVitEncoder(nn.Module):
    """DeepSeek-V4 / V4.1 vision encoder.

    `forward_features` returns the final-norm patch tokens as (B, H, W, C) (`output_fmt='NHWC'`,
    matching the reference `ViT` output up to that flattening). Output of `forward` depends on
    `global_pool`:
      * 'align' (default): projector output (B, ceil(H/3)*ceil(W/3), out_features) -- the tokens the
        LLM consumes
      * 'avg': mean over patch tokens -> (B, embed_dim)
      * '': raw patch features -> (B, H, W, embed_dim), identical to forward_features
    """

    output_fmt: str = 'NHWC'

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 546,
            patch_size: int = 14,
            in_chans: int = 3,
            out_features: int = 0,
            global_pool: str = 'align',
            embed_dim: int = 1024,
            depth: int = 32,
            num_heads: int = 16,
            mlp_ratio: float = 2816 / 1024,
            downsample_ratio: int = 3,
            rope_temperature: float = 10000.0,
            norm_eps: float = 1e-6,
            proj_drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            dynamic_img_pad: bool = False,
            device=None,
            dtype=None,
    ) -> None:
        """
        Args:
            img_size: Nominal input size, used for the patch embedding's default grid. Input sizes
                must be divisible by `patch_size` unless `dynamic_img_pad` is enabled.
            patch_size: Patch size.
            in_chans: Number of input channels.
            out_features: Output width of the VLM projector (source LLM hidden size). Required when
                the projector is enabled, independent of a downstream classifier's number of classes.
            global_pool: 'align' (projector), 'avg' (mean pool) or '' (raw tokens).
            embed_dim: Token width.
            depth: Number of transformer blocks.
            num_heads: Attention heads.
            mlp_ratio: MLP hidden / embed_dim.
            downsample_ratio: Spatial unshuffle factor of the projector.
            rope_temperature: RoPE base (theta).
            norm_eps: RMSNorm eps.
            proj_drop_rate: Dropout on attention / MLP projections.
            attn_drop_rate: Attention dropout.
            drop_path_rate: Stochastic depth rate.
            dynamic_img_pad: Zero-pad the bottom/right image edges to a multiple of the patch size.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert global_pool in ('', 'avg', 'align')
        assert embed_dim % num_heads == 0
        self.num_classes = 0
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.out_features = out_features
        self.downsample_ratio = downsample_ratio
        self.num_prefix_tokens = 0
        self.feature_dim = -1  # channels-last (B, H, W, C) features
        self.grad_checkpointing = False

        norm_layer = partial(RmsNormFp32, eps=norm_eps)
        head_dim = embed_dim // num_heads
        assert head_dim % 4 == 0, 'Axial 2D RoPE requires a head dimension divisible by four.'

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=True,
            strict_img_size=False,
            dynamic_img_pad=dynamic_img_pad,
            output_fmt='NHWC',
            **dd,
        )
        r = self.patch_embed.feat_ratio()
        # Axial 2D RoPE on integer patch coordinates, head_dim // 4 frequencies per axis laid out as
        # [h-block, w-block] and tiled for the 'half' rotation -- the DeepSeek vision RoPE layout.
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
            DeepseekVitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                **dd,
            )
            for i in range(depth)
        ])
        self.feature_info = [dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]
        self.norm = norm_layer(embed_dim, **dd)

        self.aligner = (
            DeepseekVitAligner(
                embed_dim,
                downsample_ratio=downsample_ratio,
                out_features=out_features,
                **dd,
            )
            if global_pool == 'align'
            else None
        )

        self.init_weights(needs_reset=False)

    @torch.jit.ignore
    def init_weights(self, needs_reset: bool = True) -> None:
        """Initialize weights and, after ``to_empty()``, restore parameters and RoPE buffers.

        Args:
            needs_reset: Reset modules that self-initialize in their constructors. False during construction.
        """
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
        return set()

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^patch_embed|rope',
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm|^aligner', (99999,))],
        )

    def _pos_embed(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flatten NHWC patches to tokens and build the RoPE table for this grid (no learned pos embed)."""
        B, H, W, C = x.shape
        return x.reshape(B, H * W, C), self.rope.get_embed(shape=(H, W))

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
            norm: Apply the final encoder norm to the intermediates
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
                intermediates.append(self.norm(x) if norm else x)

        if reshape:
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        if intermediates_only:
            return {'image_intermediates': intermediates} if output_dict else intermediates

        x = self.norm(x).reshape(B, H, W, -1)
        return {'image_features': x, 'image_intermediates': intermediates} if output_dict else (x, intermediates)

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ) -> List[int]:
        """Prune blocks, the final norm and/or the native projector for intermediate feature extraction."""
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        self.feature_info = self.feature_info[:max_index + 1]
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.aligner = None
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
        return self.norm(x).reshape(B, H, W, -1)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        raise NotImplementedError('DeepseekVitEncoder does not support classification use cases.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw patch tokens and apply the configured native pooling/projector."""
        x = self.forward_features(x)
        if self.aligner is not None:
            return self.aligner(x)
        if self.global_pool == 'avg':
            x = x.mean(dim=(1, 2))
        return x


class DeepseekVitClassifier(nn.Module):
    """Classification wrapper: DeepSeek encoder patch tokens -> mean pool -> norm -> linear head.

    `encoder_pool='align'` retains the native VLM projector before classification pooling.
    `num_classes` controls the classifier independently of `out_features`, the projector's output width.
    `dynamic_img_pad=True` enables image padding to accept sizes not divisible by the patch size.
    """

    output_fmt: str = 'NHWC'

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 546,
            patch_size: int = 14,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            encoder_pool: str = '',
            out_features: int = 0,
            embed_dim: int = 1024,
            depth: int = 32,
            num_heads: int = 16,
            mlp_ratio: float = 2816 / 1024,
            downsample_ratio: int = 3,
            rope_temperature: float = 10000.0,
            norm_eps: float = 1e-6,
            final_norm: bool = True,
            drop_rate: float = 0.0,
            proj_drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            dynamic_img_pad: bool = False,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert global_pool in ('', 'avg')
        assert encoder_pool in ('', 'align')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.encoder_pool = encoder_pool
        self.encoder = DeepseekVitEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            global_pool=encoder_pool,
            out_features=out_features,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            downsample_ratio=downsample_ratio,
            rope_temperature=rope_temperature,
            norm_eps=norm_eps,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            dynamic_img_pad=dynamic_img_pad,
            **dd,
        )
        self.embed_dim = embed_dim
        self.num_features = self.head_hidden_size = (
            self.encoder.aligner.out_features if self.encoder.aligner is not None else embed_dim
        )
        self.output_fmt = 'NLC' if encoder_pool == 'align' else 'NHWC'
        # the encoder already ends in an affine RMSNorm, so the classifier norm is affine-free
        self.norm = RmsNormFp32(self.num_features, eps=norm_eps, affine=False, **dd) if final_norm else nn.Identity()
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
            stem=r'^encoder\.(?:patch_embed|rope)',
            blocks=[(r'^encoder\.blocks\.(\d+)', None), (r'^encoder\.(?:norm|aligner)|^norm|^head', (99999,))],
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
        """Normalized NHWC patches, or projected NLC tokens when encoder_pool='align'."""
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

        ``norm`` applies the encoder's final norm to the intermediates; the classifier's own
        post-pooling normalization is never applied to patch tokens. The projector, when enabled,
        applies only to the final features, not the intermediate maps. ``output_dict`` returns
        'image_intermediates' and, unless ``intermediates_only``, 'image_features'.
        """
        output = self.encoder.forward_intermediates(
            x,
            indices=indices,
            return_prefix_tokens=return_prefix_tokens,
            norm=norm,
            stop_early=stop_early,
            output_fmt=output_fmt,
            intermediates_only=False,
        )
        assert isinstance(output, tuple)
        x, intermediates = output
        if intermediates_only:
            return {'image_intermediates': intermediates} if output_dict else intermediates
        if self.encoder.aligner is not None:
            x = self.encoder.aligner(x)
        return {'image_features': x, 'image_intermediates': intermediates} if output_dict else (x, intermediates)

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ) -> List[int]:
        """Prune encoder blocks and optionally the projector and classifier."""
        take_indices = self.encoder.prune_intermediate_layers(indices, prune_norm=prune_norm, prune_head=prune_head)
        self.feature_info = [dict(info, module=f'encoder.{info["module"]}') for info in self.encoder.feature_info]
        if prune_head:
            self.encoder_pool = ''
            self.output_fmt = 'NHWC'
            self.num_features = self.head_hidden_size = self.encoder.num_features
            if isinstance(self.norm, RmsNormFp32):
                self.norm = RmsNormFp32(
                    self.num_features,
                    eps=self.norm.eps,
                    affine=False,
                    **get_device_dtype(self),
                )
                self.norm.train(self.training)
            self.reset_classifier(0)
        return take_indices


def checkpoint_filter_fn_encoder(
        state_dict: Dict[str, torch.Tensor],
        model: DeepseekVitEncoder,
) -> Dict[str, torch.Tensor]:
    """Remap `vision.*` / `aligner.*` tensors of a DeepSeek-V4 / V4.1 checkpoint (or shard) to timm keys.

    Every non-vision tensor (the LLM) is dropped, so a raw checkpoint shard can be passed directly.

    The source checkpoint keeps the vision tower under `vision.` / `aligner.` and the LLM at the top
    level, where `norm.weight` and `head.weight` collide with timm's own names (the shard the
    pretrained cfgs point at happens to hold no such tensor, but the full checkpoint does). So an
    unprefixed key is trusted only when the dict has no `vision.` keys at all, i.e. when it is
    already in timm layout.
    """
    from_source = any(k.startswith('vision.') for k in state_dict)
    out_dict = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.'):  # timm classifier checkpoint
            k = k.replace('encoder.', '', 1)
        elif from_source:
            if k.startswith('vision.'):
                k = k.replace('vision.', '', 1)
            elif not k.startswith('aligner.'):
                continue  # LLM tensors, incl. the image_start/end/newline span embeddings
        elif not k.startswith(('patch_embed.', 'blocks.', 'norm.', 'aligner.')):
            continue

        if k.startswith('aligner.'):
            if model.aligner is None:
                continue
            k = k.replace('aligner.w1.', 'aligner.fc1.', 1).replace('aligner.w2.', 'aligner.fc2.', 1)
            out_dict[k] = v
            continue
        if k == 'patch_embed.proj.weight' and v.ndim == 2:
            # the reference embeds patches with a Linear over the flattened (C, pH, pW) patch; that is
            # exactly a Conv2d kernel of stride=patch_size, so this is a reshape, not an approximation
            v = v.reshape(v.shape[0], -1, *model.patch_embed.patch_size)
        k = k.replace('.attn.wqkv.', '.attn.qkv.').replace('.attn.wo.', '.attn.proj.')
        k = k.replace('.mlp.w1.', '.mlp.fc1.').replace('.mlp.w2.', '.mlp.fc2.')
        out_dict[k] = v
    return out_dict


def checkpoint_filter_fn_classifier(
        state_dict: Dict[str, torch.Tensor],
        model: DeepseekVitClassifier,
) -> Dict[str, torch.Tensor]:
    """Load DeepSeek vision / timm encoder weights, or round-trip a fine-tuned classifier checkpoint.

    Bare `norm.` / `head.` keys belong to the classifier only when the checkpoint has an `encoder.`
    namespace. Otherwise `norm.` is the native encoder norm, or the LLM norm in a source checkpoint.
    """
    from_source = any(k.startswith('vision.') for k in state_dict)
    from_classifier = not from_source and any(k.startswith('encoder.') for k in state_dict)
    local = {k: v for k, v in state_dict.items() if k.startswith(('norm.', 'head.'))} if from_classifier else {}
    encoder = checkpoint_filter_fn_encoder({k: v for k, v in state_dict.items() if k not in local}, model.encoder)
    return {**{f'encoder.{k}': v for k, v in encoder.items()}, **local}


def _resolve_deepseek_out_features(variant: str, tag: str) -> int:
    source_arch = variant.rsplit('_', 1)[0] if variant.endswith(('_enc', '_align')) else variant
    source_name = f'{source_arch}.{tag}'
    if source_name not in _SOURCE_CFGS:
        raise ValueError('Specify out_features for a DeepSeek aligner with a custom pretrained configuration.')
    return _SOURCE_CFGS[source_name]['out_features']


def _create_deepseek_vit_classifier(variant: str, pretrained: bool = False, **kwargs) -> DeepseekVitClassifier:
    pretrained_cfg = resolve_pretrained_cfg(
        variant,
        pretrained_cfg=kwargs.pop('pretrained_cfg', None),
        pretrained_cfg_overlay=kwargs.pop('pretrained_cfg_overlay', None),
    )
    if kwargs.get('encoder_pool') == 'align' and 'out_features' not in kwargs:
        kwargs['out_features'] = _resolve_deepseek_out_features(variant, pretrained_cfg.tag)
    out_indices = kwargs.pop('out_indices', 3)
    return build_model_with_cfg(
        DeepseekVitClassifier,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn_classifier,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )


def _create_deepseek_vit_encoder(variant: str, pretrained: bool = False, **kwargs) -> DeepseekVitEncoder:
    pretrained_cfg = resolve_pretrained_cfg(
        variant,
        pretrained_cfg=kwargs.pop('pretrained_cfg', None),
        pretrained_cfg_overlay=kwargs.pop('pretrained_cfg_overlay', None),
    )
    if 'out_features' not in kwargs:
        kwargs['out_features'] = _resolve_deepseek_out_features(variant, pretrained_cfg.tag)
    out_indices = kwargs.pop('out_indices', 3)
    return build_model_with_cfg(
        DeepseekVitEncoder,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn_encoder,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        kwargs_filter=('num_classes', 'drop_rate'),
        **kwargs,
    )


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        # RoPE-only position encoding: any size divisible by the patch size works, the projector pads
        # a leftover row/column. The default is the square grid the reference preprocessing produces
        # at the checkpoint's own `vision_min_pixels`.
        'input_size': (3, 546, 546),
        # smallest sensible square grid: 9x9 patches, an exact 3x3 aligner grid with no padding.
        # Needed because a plain min(size, target) downscale would not land on a multiple of 14.
        'min_input_size': (3, 126, 126),
        'pool_size': None,
        'fixed_input_size': False,
        'crop_pct': 1.0,
        'crop_mode': 'border',
        'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'num_classes': 0,
        'license': 'mit',
        **kwargs,
    }


_SOURCE_CFGS = {
    # Upstream checkpoints for conversion: shard 1 of each repo holds the whole `vision.*` tower and
    # `aligner.*`. `out_features` is the LLM hidden size the aligner projects to, independent of any
    # classifier labels.
    'deepseek_vit_412m.deepseek_v4_1_flash': _cfg(
        hf_hub_id='deepseek-ai/DeepSeek-V4.1-Flash',
        hf_hub_filename='model-00001-of-00048.safetensors',
        out_features=5120,
        input_size=(3, 546, 546),  # vision_min_pixels 544^2 -> 39x39 patches
        origin_url='https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash',
    ),
    'deepseek_vit_412m.deepseek_v4_flash_vision_exp': _cfg(
        hf_hub_id='deepseek-ai/DeepSeek-V4-Flash-Vision-Exp',
        hf_hub_filename='model-00001-of-00048.safetensors',
        out_features=4096,
        input_size=(3, 392, 392),  # vision_min_pixels 384^2 -> 28x28 patches
        origin_url='https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp',
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
    for suffix in ('', '_align', '_enc')
})


@register_model
def deepseek_vit_412m(pretrained: bool = False, **kwargs) -> DeepseekVitClassifier:
    """DeepSeek vision classifier, 32 x 1024 (412M w/o projector). Source: DeepSeek-V4 / V4.1."""
    return _create_deepseek_vit_classifier('deepseek_vit_412m', pretrained=pretrained, **kwargs)


@register_model
def deepseek_vit_412m_align(pretrained: bool = False, **kwargs) -> DeepseekVitClassifier:
    """DeepSeek vision classifier retaining the native aligner and source LLM projection."""
    model_args = dict(encoder_pool='align')
    return _create_deepseek_vit_classifier(
        'deepseek_vit_412m_align',
        pretrained=pretrained,
        **dict(model_args, **kwargs),
    )


@register_model
def deepseek_vit_412m_enc(pretrained: bool = False, **kwargs) -> DeepseekVitEncoder:
    """DeepSeek native vision encoder with the aligner and source LLM projection."""
    return _create_deepseek_vit_encoder('deepseek_vit_412m_enc', pretrained=pretrained, **kwargs)
