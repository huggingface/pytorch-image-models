"""EfficientViM

Efficient ViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality
- paper: https://arxiv.org/abs/2411.15241 (CVPR 2025)
- code: https://github.com/mlvlab/EfficientViM

Hidden State Mixer based State Space Duality (HSM-SSD) treats the recurrent Mamba-style
state update as a chunked dual (attention-like) form. The mixer here is intentionally
plain PyTorch -- conv, softmax, matmul -- with no mamba_ssm / causal_conv1d / Triton
dependency, so the module remains CPU- and TorchScript-friendly.

The classification head is a 4-way softmax-weighted fusion rather than a single Linear:
three heads consume the pooled per-stage hidden states and the fourth the pooled final
feature map. The distillation variant (`.dist` checkpoints) adds a parallel `heads_dist` /
`weights_dist` path that returns a `(logits, logits_dist)` tuple when training and their
mean in eval, mirroring `LevitDistilled`.

Ported from the official implementation (MIT license, Copyright (c) 2024 MLVlab), fvcore
FLOP-counting helper dropped.

Modifications by / Copyright 2025 Ross Wightman
"""
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import (
    DropPath,
    SelectAdaptivePool2d,
    SqueezeExcite,
    calculate_drop_path_rates,
    trunc_normal_,
    get_device_dtype,
)
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._features_fx import register_notrace_module
from ._manipulate import checkpoint
from ._registry import register_model, generate_default_cfgs

__all__ = ['EfficientViM', 'EfficientViMDistilled']


class LayerNorm1d(nn.Module):
    """LayerNorm across channels of a (B, C, L) tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True, device=None, dtype=None):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class LayerNorm2d(nn.Module):
    """LayerNorm across channels of a (B, C, H, W) tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True, device=None, dtype=None):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class ConvLayer1d(nn.Module):
    """Conv1d + optional norm + optional act, as in the reference implementation."""

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            norm: Optional[Type[nn.Module]] = nn.BatchNorm1d,
            act_layer: Optional[Type[nn.Module]] = nn.ReLU,
            bn_weight_init: float = 1.,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.conv = nn.Conv1d(
            in_dim, out_dim, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=False, **dd)
        self.norm = norm(out_dim, **dd) if norm else None
        self.act = act_layer() if act_layer else None
        if self.norm is not None and isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvLayer2d(nn.Module):
    """Conv2d + optional norm + optional act, as in the reference implementation."""

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            norm: Optional[Type[nn.Module]] = nn.BatchNorm2d,
            act_layer: Optional[Type[nn.Module]] = nn.ReLU,
            bn_weight_init: float = 1.,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.conv = nn.Conv2d(
            in_dim, out_dim, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=False, **dd)
        self.norm = norm(out_dim, **dd) if norm else None
        self.act = act_layer() if act_layer else None
        if self.norm is not None and isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Stem(nn.Module):
    """4-layer stride-2 conv stem, reduction 16."""

    def __init__(
            self,
            in_dim: int,
            dim: int,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.conv = nn.Sequential(
            ConvLayer2d(in_dim, dim // 8, kernel_size=3, stride=2, padding=1, **dd),
            ConvLayer2d(dim // 8, dim // 4, kernel_size=3, stride=2, padding=1, **dd),
            ConvLayer2d(dim // 4, dim // 2, kernel_size=3, stride=2, padding=1, **dd),
            ConvLayer2d(dim // 2, dim, kernel_size=3, stride=2, padding=1, act_layer=None, **dd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FFN(nn.Module):
    def __init__(
            self,
            in_dim: int,
            dim: int,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.fc1 = ConvLayer2d(in_dim, dim, 1, **dd)
        self.fc2 = ConvLayer2d(dim, in_dim, 1, act_layer=None, bn_weight_init=0, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class PatchMerging(nn.Module):
    """MBConv-style downsample with SE, plus the reference implementation's depthwise bypasses."""

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            ratio: float = 4.0,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        hidden_dim = int(out_dim * ratio)
        self.conv = nn.Sequential(
            ConvLayer2d(in_dim, hidden_dim, kernel_size=1, **dd),
            ConvLayer2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, groups=hidden_dim, **dd),
            SqueezeExcite(hidden_dim, .25, **dd),
            ConvLayer2d(hidden_dim, out_dim, kernel_size=1, act_layer=None, **dd),
        )
        self.dwconv1 = ConvLayer2d(in_dim, in_dim, 3, padding=1, groups=in_dim, act_layer=None, **dd)
        self.dwconv2 = ConvLayer2d(out_dim, out_dim, 3, padding=1, groups=out_dim, act_layer=None, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dwconv1(x)
        x = self.conv(x)
        x = x + self.dwconv2(x)
        return x


@register_notrace_module  # reason: FX can't symbolically trace the int(sqrt(L)) seq-len to spatial reshape
class HSMSSD(nn.Module):
    """Hidden State Mixer based State Space Duality.

    The Mamba-style gated recurrence is computed in its dual form: input-dependent
    `A` (via `dt`) and `B` combine into a chunk transition matrix applied to the
    flattened spatial tokens, then the resulting hidden state is read out with `C`.
    Everything is expressed with conv / softmax / matmul, no custom kernels.
    """
    def __init__(
            self,
            dim: int,
            ssd_expand: float = 1.,
            state_dim: int = 64,
            a_init_range: Tuple[float, float] = (1., 16.),
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * dim)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1d(dim, 3 * state_dim, 1, norm=None, act_layer=None, **dd)
        conv_dim = self.state_dim * 3
        self.dw = ConvLayer2d(
            conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0, **dd)
        self.hz_proj = ConvLayer1d(dim, 2 * self.d_inner, 1, norm=None, act_layer=None, **dd)
        self.out_proj = ConvLayer1d(self.d_inner, dim, 1, norm=None, act_layer=None, bn_weight_init=0, **dd)

        a = torch.empty(self.state_dim, dtype=torch.float32, device=device).uniform_(*a_init_range)
        self.A = nn.Parameter(a)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1, device=device, dtype=dtype))
        self.D._no_weight_decay = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, seq_len = x.shape
        hw = int(math.sqrt(seq_len))

        BCdt = self.dw(self.BCdt_proj(x).view(batch, -1, hw, hw)).flatten(2)
        b, c, dt = torch.split(BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)
        a = (dt + self.A.view(1, -1, 1)).softmax(-1)

        ab = a * b
        h = x @ ab.transpose(-2, -1)

        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1)
        h = self.out_proj(h * self.act(z) + h * self.D)
        y = h @ c  # B C N, B C L -> B C L

        y = y.view(batch, -1, hw, hw).contiguous()
        return y, h


class EfficientViMBlock(nn.Module):
    """Depthwise conv -> HSM-SSD mixer -> depthwise conv -> FFN, each with a LayerScale residual."""

    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 4.,
            ssd_expand: float = 1.,
            state_dim: int = 64,
            drop_path: float = 0.,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.mixer = HSMSSD(dim, ssd_expand=ssd_expand, state_dim=state_dim, **dd)
        self.norm = LayerNorm1d(dim, **dd)

        self.dwconv1 = ConvLayer2d(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None, **dd)
        self.dwconv2 = ConvLayer2d(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None, **dd)
        self.ffn = FFN(dim, int(dim * mlp_ratio), **dd)

        # LayerScale, one gate per branch. Applied to the branch delta so it composes with
        # DropPath on the branch output without changing the reference behaviour at
        # drop_path=0 (where the gates multiply the branch, not the residual).
        self.alpha = nn.Parameter(
            1e-4 * torch.ones(4, dim, device=device, dtype=dtype), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)

        # dwconv1
        x = x + self.drop_path(alpha[0] * (self.dwconv1(x) - x))

        # HSM-SSD
        x_prev = x
        x, h = self.mixer(self.norm(x.flatten(2)))
        x = x_prev + self.drop_path(alpha[1] * (x - x_prev))

        # dwconv2
        x = x + self.drop_path(alpha[2] * (self.dwconv2(x) - x))

        # FFN
        x = x + self.drop_path(alpha[3] * (self.ffn(x) - x))
        return x, h


class EfficientViMStage(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            depth: int,
            mlp_ratio: float = 4.,
            downsample: Optional[Type[nn.Module]] = None,
            ssd_expand: float = 1.,
            state_dim: int = 64,
            drop_path: Optional[List[float]] = None,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.depth = depth
        self.grad_checkpointing = False
        drop_path = drop_path or [0.] * depth
        self.blocks = nn.ModuleList([
            EfficientViMBlock(
                dim=in_dim, mlp_ratio=mlp_ratio, ssd_expand=ssd_expand, state_dim=state_dim,
                drop_path=drop_path[i], device=device, dtype=dtype)
            for i in range(depth)])
        self.downsample = downsample(in_dim=in_dim, out_dim=out_dim, device=device, dtype=dtype) \
            if downsample is not None else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (out, feature_map, hidden_state).

        `out` is the input to the next stage (post-downsample when present), `feature_map`
        is this stage's output at its own resolution -- the tensor `feature_info` points at.
        The feature map is first so that feature-extraction wrappers, which take `x[0]` of a
        tuple-returning module, tap the right tensor.
        """
        h = None
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x, h = checkpoint(blk, x)
            else:
                x, h = blk(x)

        x_out = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x_out, x, h


class EfficientViM(nn.Module):
    """EfficientViM backbone with multi-stage hidden-state fused classification head.

    Three HSM-SSD stages operate at reductions 16 / 32 / 64. Each stage yields both a 2-D
    feature map (`x_out`, exposed via `feature_info` / `forward_intermediates`) and the
    final hidden state `h` of its last block, which the classification head fuses together
    with the pooled final feature map under a learned softmax weighting.

    Args:
        in_chans: Number of input image channels.
        num_classes: Number of classifier output classes.
        embed_dim: Channel width of each of the three stages.
        depths: Number of blocks in each of the three stages.
        mlp_ratio: FFN expansion ratio.
        ssd_expand: Expansion of the HSM-SSD hidden state.
        state_dim: Per-stage HSM-SSD state dimension.
        global_pool: Global pooling type for the final feature map head, 'avg' or '' to disable.
        drop_path_rate: Stochastic depth rate, linearly ramped across all blocks.
    """

    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            embed_dim: Tuple[int, ...] = (128, 256, 512),
            depths: Tuple[int, ...] = (2, 2, 2),
            mlp_ratio: float = 4.,
            ssd_expand: float = 1.,
            state_dim: Tuple[int, ...] = (49, 25, 9),
            global_pool: str = 'avg',
            drop_path_rate: float = 0.,
            device=None,
            dtype=None,
            **kwargs,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = tuple(embed_dim)
        self.grad_checkpointing = False
        self.feature_info = []

        self.patch_embed = Stem(in_dim=in_chans, dim=embed_dim[0])
        dpr = calculate_drop_path_rates(drop_path_rate, sum(depths))

        stages = []
        block_idx = 0
        # stem reduces by 16; each downsample that follows a stage doubles it again. The
        # x_out map a stage reports is taken *before* its trailing downsample.
        reduction = 16
        for i_layer in range(self.num_layers):
            stage_dpr = dpr[block_idx:block_idx + depths[i_layer]]
            block_idx += depths[i_layer]
            stages.append(EfficientViMStage(
                in_dim=int(embed_dim[i_layer]),
                out_dim=int(embed_dim[i_layer + 1]) if (i_layer < self.num_layers - 1) else 0,
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                ssd_expand=ssd_expand,
                state_dim=state_dim[i_layer],
                drop_path=stage_dpr,
            ))
            self.feature_info.append(
                dict(num_chs=embed_dim[i_layer], reduction=reduction, module=f'stages.{i_layer}'))
            if i_layer < self.num_layers - 1:
                reduction *= 2
        self.stages = nn.ModuleList(stages)

        self.num_features = self.head_hidden_size = embed_dim[-1]

        # Weights for multi-stage hidden-state fusion
        self.weights = nn.Parameter(torch.ones(4))
        self.norm = nn.ModuleList([
            LayerNorm1d(embed_dim[0]),
            LayerNorm1d(embed_dim[1]),
            LayerNorm1d(embed_dim[2]),
            LayerNorm2d(embed_dim[2]),
        ])
        self.heads = self._build_heads(num_classes, dd)

        # don't flatten when pooling disabled, the feature map is returned as-is
        self.head_pool = SelectAdaptivePool2d(pool_type=global_pool, flatten=bool(global_pool))

        self.apply(self._init_weights)

    def _build_heads(self, num_classes: int, dd: Optional[Dict[str, Any]] = None) -> nn.ModuleList:
        dd = dd or {}
        return nn.ModuleList([
            nn.Linear(self.embed_dim[0], num_classes, **dd) if num_classes > 0 else nn.Identity(),
            nn.Linear(self.embed_dim[1], num_classes, **dd) if num_classes > 0 else nn.Identity(),
            nn.Linear(self.embed_dim[2], num_classes, **dd) if num_classes > 0 else nn.Identity(),
            nn.Linear(self.embed_dim[2], num_classes, **dd) if num_classes > 0 else nn.Identity(),
        ])

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm1d, LayerNorm2d, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return {name for name, module in self.named_modules() if getattr(module, '_no_weight_decay', False)}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        matcher = dict(
            stem=r'^patch_embed',
            blocks=[
                (r'^stages\.(\d+)' if coarse else r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm\.(\d+)', (99999,)),
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable
        for stage in self.stages:
            stage.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.ModuleList:
        return self.heads

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        dd = get_device_dtype(self)
        was_training = self.training
        self.num_classes = num_classes
        if global_pool is not None:
            # don't flatten when pooling disabled, the feature map is returned as-is
            self.head_pool = SelectAdaptivePool2d(pool_type=global_pool, flatten=bool(global_pool))
            self.head_pool.train(was_training)
        self.heads = self._build_heads(num_classes, dd)
        self.heads.train(was_training)

    @torch.jit.ignore
    def set_distilled_training(self, enable: bool = True):
        self.distilled_training = enable

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor.
            indices: Take last n blocks if int, all if None, select matching indices if sequence.
            norm: Apply norm layer to compatible intermediates (applies the per-stage LayerNorm
                to the stage's own output map only).
            stop_early: Stop iterating over blocks when last desired intermediate hit.
            output_fmt: Shape of intermediate feature outputs.
            intermediates_only: Only return intermediate features.

        Returns:
            List of intermediate features or tuple of (final features, intermediates).
        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages), indices)

        x = self.patch_embed(x)
        stages = self.stages if (torch.jit.is_scripting() or not stop_early) else self.stages[:max_index + 1]
        for feat_idx, stage in enumerate(stages):
            # stage returns (feature_map, next_stage_input, hidden_state)
            x_out, x, _ = stage(x)
            if feat_idx in take_indices:
                intermediates.append(self.norm[feat_idx](x_out) if norm else x_out)

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
            indices: Indices of intermediate layers to keep.
            prune_norm: Whether to prune normalization layer.
            prune_head: Whether to prune the classifier head.

        Returns:
            List of indices that were kept.
        """
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        self.stages = self.stages[:max_index + 1]
        if prune_head:
            self.reset_classifier(0)
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for stage in self.stages:
            _, x, _ = stage(x)
        return x

    def _forward_stage_states(
            self,
            x: torch.Tensor,
            want_hidden: bool,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run all stages, returning the final map and the pooled per-stage hidden states.

        The hidden states are what heads 0-2 consume. When `want_hidden` is False the pooled
        final map is repeated as a placeholder, avoiding an Optional that TorchScript cannot
        narrow; callers that pass False ignore the list.
        """
        x = self.patch_embed(x)
        hidden: List[torch.Tensor] = []
        # zip rather than index self.norm, ModuleList indexing needs literals under scripting
        for stage, norm in zip(self.stages, self.norm):
            _, x, h = stage(x)
            if want_hidden:
                h = norm(h)
                h = F.adaptive_avg_pool1d(h, 1).flatten(1)
                hidden.append(h)
        if not hidden:
            hidden = [x]  # scripting / encoder-only: heads are Identity, list unused
        return x, hidden

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Fused multi-stage head over a final feature map.

        NOTE: unlike most timm models, `forward_head` alone cannot reconstruct the per-stage
        hidden states heads 0-2 consume -- they are produced alongside the final map by
        `forward`. Calling it directly with a feature map (as the `FeatureListNet` /
        `FeatureGraphNet` / `FeatureGetterNet` wrappers and `forward_intermediates`
        consumers do) yields the head-3 path over the pooled map.
        """
        x = self.norm[3](x)
        x = self.head_pool(x)
        if pre_logits:
            return x
        return self.heads[3](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        want_hidden = self.num_classes > 0
        x, hidden = self._forward_stage_states(x, want_hidden=want_hidden)
        x = self.norm[3](x)
        x = self.head_pool(x)
        if not want_hidden:
            # encoder-only (num_classes == 0): heads are Identity, only the pooled map is meaningful
            return x
        weights = self.weights.softmax(-1)

        z = weights[0] * self.heads[0](hidden[0])
        z = z + weights[1] * self.heads[1](hidden[1])
        z = z + weights[2] * self.heads[2](hidden[2])
        z = z + weights[3] * self.heads[3](x)
        return z


class EfficientViMDistilled(EfficientViM):
    """EfficientViM with a parallel distilled head.

    Adds `heads_dist` / `weights_dist`, fused the same way as the primary head. Training with
    `distilled_training=True` returns the `(logits, logits_dist)` tuple, otherwise (and always
    in eval) the two predictions are averaged, matching `LevitDistilled`.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        dd = get_device_dtype(self)
        num_classes = self.num_classes
        self.weights_dist = nn.Parameter(torch.ones(4))
        self.heads_dist = self._build_heads(num_classes, dd)
        self.distilled_training = False

    @torch.jit.ignore
    def get_classifier(self) -> Tuple[nn.ModuleList, nn.ModuleList]:
        return self.heads, self.heads_dist

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        super().reset_classifier(num_classes, global_pool=global_pool)
        dd = get_device_dtype(self)
        self.heads_dist = self._build_heads(num_classes, dd)
        self.heads_dist.train(self.training)

    @torch.jit.ignore
    def set_distilled_training(self, enable: bool = True):
        self.distilled_training = enable

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.norm[3](x)
        x = self.head_pool(x)
        if pre_logits:
            return x
        z, z_dist = self.heads[3](x), self.heads_dist[3](x)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return z, z_dist
        # during standard train/finetune, inference average the classifier predictions
        return (z + z_dist) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        want_hidden = self.num_classes > 0
        x, hidden = self._forward_stage_states(x, want_hidden=want_hidden)
        x = self.norm[3](x)
        x = self.head_pool(x)
        if not want_hidden:
            # encoder-only (num_classes == 0): heads are Identity, only the pooled map is meaningful
            return x
        weights = self.weights.softmax(-1)
        weights_dist = self.weights_dist.softmax(-1)

        z = weights[0] * self.heads[0](hidden[0])
        z_dist = weights_dist[0] * self.heads_dist[0](hidden[0])
        z = z + weights[1] * self.heads[1](hidden[1])
        z_dist = z_dist + weights_dist[1] * self.heads_dist[1](hidden[1])
        z = z + weights[2] * self.heads[2](hidden[2])
        z_dist = z_dist + weights_dist[2] * self.heads_dist[2](hidden[2])
        z = z + weights[3] * self.heads[3](x)
        z_dist = z_dist + weights_dist[3] * self.heads_dist[3](x)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return z, z_dist
        # during standard train/finetune, inference average the classifier predictions
        return (z + z_dist) / 2


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """Map upstream EfficientViM checkpoints onto this module's names.

    Module names are kept identical to the reference implementation, so no key remapping is
    needed -- only unwrapping the training-checkpoint envelope and, for the non-distilled
    model, dropping the tensors of the parallel distillation head. The released checkpoints
    nest the weights under `model` (final) and `model_ema` (EMA); the EMA weights match the
    paper's reported top-1, so they are preferred when present.
    """
    if 'patch_embed.conv.0.conv.weight' not in state_dict:
        for envelope_key in ('model_ema', 'model', 'state_dict'):
            if isinstance(state_dict.get(envelope_key), dict):
                state_dict = state_dict[envelope_key]
                break
    if not isinstance(model, EfficientViMDistilled):
        # distilled tensors have no target module, loading would report them unexpected
        state_dict = {
            k: v for k, v in state_dict.items() if not (k.startswith('heads_dist.') or k == 'weights_dist')}
    return state_dict


def _cfg(**kwargs: Any) -> Dict[str, Any]:
    return {
        'url': '',
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (4, 4),
        'crop_pct': 0.95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.conv.0.conv',
        # all four fusion heads are num_classes-sized -- list them so timm resets every one on a
        # num_classes change (a single 'heads.3' leaves heads.0..2 mismatched on transfer).
        'classifier': ('heads.0', 'heads.1', 'heads.2', 'heads.3'),
        'origin_url': 'https://github.com/mlvlab/EfficientViM', 'license': 'mit',
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'efficientvim_m1.e300_in1k': _cfg(
        hf_hub_id='remyxai/efficientvim_m1.e300_in1k',
    ),
    'efficientvim_m1.e450_in1k': _cfg(
        hf_hub_id='remyxai/efficientvim_m1.e450_in1k',
    ),
    'efficientvim_m1_dist.in1k': _cfg(
        hf_hub_id='remyxai/efficientvim_m1_dist.in1k',
        classifier=('heads.0', 'heads.1', 'heads.2', 'heads.3',
                    'heads_dist.0', 'heads_dist.1', 'heads_dist.2', 'heads_dist.3'),
    ),
    'efficientvim_m2.e300_in1k': _cfg(
        hf_hub_id='remyxai/efficientvim_m2.e300_in1k',
    ),
    'efficientvim_m2.e450_in1k': _cfg(
        hf_hub_id='remyxai/efficientvim_m2.e450_in1k',
    ),
    'efficientvim_m2_dist.in1k': _cfg(
        hf_hub_id='remyxai/efficientvim_m2_dist.in1k',
        classifier=('heads.0', 'heads.1', 'heads.2', 'heads.3',
                    'heads_dist.0', 'heads_dist.1', 'heads_dist.2', 'heads_dist.3'),
    ),
    'efficientvim_m3.e300_in1k': _cfg(
        hf_hub_id='remyxai/efficientvim_m3.e300_in1k',
    ),
    'efficientvim_m3.e450_in1k': _cfg(
        hf_hub_id='remyxai/efficientvim_m3.e450_in1k',
    ),
    'efficientvim_m3_dist.in1k': _cfg(
        hf_hub_id='remyxai/efficientvim_m3_dist.in1k',
        classifier=('heads.0', 'heads.1', 'heads.2', 'heads.3',
                    'heads_dist.0', 'heads_dist.1', 'heads_dist.2', 'heads_dist.3'),
    ),
    'efficientvim_m4.e300_in1k': _cfg(
        hf_hub_id='remyxai/efficientvim_m4.e300_in1k',
        input_size=(3, 256, 256), pool_size=(4, 4),
    ),
    'efficientvim_m4.e450_in1k': _cfg(
        hf_hub_id='remyxai/efficientvim_m4.e450_in1k',
        input_size=(3, 256, 256), pool_size=(4, 4),
    ),
    'efficientvim_m4_dist.in1k': _cfg(
        hf_hub_id='remyxai/efficientvim_m4_dist.in1k',
        input_size=(3, 256, 256), pool_size=(4, 4),
        classifier=('heads.0', 'heads.1', 'heads.2', 'heads.3',
                    'heads_dist.0', 'heads_dist.1', 'heads_dist.2', 'heads_dist.3'),
    ),
})


def _create_efficientvim(
        variant: str,
        pretrained: bool = False,
        distilled: bool = False,
        **kwargs: Any,
) -> EfficientViM:
    # stages is an nn.ModuleList (upstream state-dict layout) and each stage returns a
    # (feature_map, next_input, hidden_state) tuple, so use the hook / getter feature
    # wrappers rather than the flattening FeatureListNet (mvitv2 precedent).
    feature_cfg = dict(out_indices=(0, 1, 2), feature_cls='hook')
    return build_model_with_cfg(
        EfficientViMDistilled if distilled else EfficientViM, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=feature_cfg,
        **kwargs,
    )


@register_model
def efficientvim_m1(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M1: 6.7M params, 72.9 top-1 @ 300 epochs."""
    model_args = dict(
        embed_dim=(128, 192, 320), depths=(2, 2, 2), state_dim=(49, 25, 9),
    )
    return _create_efficientvim("efficientvim_m1", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def efficientvim_m2(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M2: 13.9M params, 75.4 top-1 @ 300 epochs."""
    model_args = dict(
        embed_dim=(128, 256, 512), depths=(2, 2, 2), state_dim=(49, 25, 9),
    )
    return _create_efficientvim("efficientvim_m2", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def efficientvim_m3(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M3: 16.6M params, 77.6 top-1 @ 300 epochs."""
    model_args = dict(
        embed_dim=(224, 320, 512), depths=(2, 2, 2), state_dim=(49, 25, 9),
    )
    return _create_efficientvim("efficientvim_m3", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def efficientvim_m4(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M4: 19.6M params, 79.4 top-1 @ 300 epochs, 256x256 input."""
    model_args = dict(
        embed_dim=(224, 320, 512), depths=(3, 4, 2), state_dim=(64, 32, 16),
    )
    return _create_efficientvim("efficientvim_m4", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def efficientvim_m1_dist(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M1, distilled: 6.7M params, 74.6 top-1."""
    model_args = dict(
        embed_dim=(128, 192, 320), depths=(2, 2, 2), state_dim=(49, 25, 9),
    )
    return _create_efficientvim(
        "efficientvim_m1_dist", pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))


@register_model
def efficientvim_m2_dist(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M2, distilled: 13.9M params, 76.7 top-1."""
    model_args = dict(
        embed_dim=(128, 256, 512), depths=(2, 2, 2), state_dim=(49, 25, 9),
    )
    return _create_efficientvim(
        "efficientvim_m2_dist", pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))


@register_model
def efficientvim_m3_dist(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M3, distilled: 16.6M params, 79.1 top-1."""
    model_args = dict(
        embed_dim=(224, 320, 512), depths=(2, 2, 2), state_dim=(49, 25, 9),
    )
    return _create_efficientvim(
        "efficientvim_m3_dist", pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))


@register_model
def efficientvim_m4_dist(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M4, distilled: 19.6M params, 80.7 top-1, 256x256 input."""
    model_args = dict(
        embed_dim=(224, 320, 512), depths=(3, 4, 2), state_dim=(64, 32, 16),
    )
    return _create_efficientvim(
        "efficientvim_m4_dist", pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))
