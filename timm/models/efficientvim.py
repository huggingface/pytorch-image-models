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
feature map. The distillation variant (`_dist` models) adds a parallel `heads_dist` /
`weights_dist` path that returns a `(logits, logits_dist)` tuple when training and their
mean in eval, mirroring `LevitDistilled`.

For deployment, timm.utils.reparameterize_model(model.eval()) folds Conv-BatchNorm
pairs and depthwise residual branches without changing the feature or classifier APIs.

Ported from the official implementation (MIT license, Copyright (c) 2024 MLVlab), fvcore
FLOP-counting helper dropped.

Modifications by / Copyright 2025 Ross Wightman
"""

from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import (
    DropPath,
    LayerNorm,
    LayerNorm2d,
    SqueezeExcite,
    calculate_drop_path_rates,
    trunc_normal_,
    get_device_dtype,
)
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint
from ._registry import register_model, generate_default_cfgs

__all__ = ['EfficientViM', 'EfficientViMDistilled']


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
            bn_weight_init: float = 1.0,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            **dd,
        )
        self.norm = norm(out_dim, **dd) if norm else None
        self.act = act_layer() if act_layer else None
        if self.norm is not None and isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    @torch.no_grad()
    def fuse(self) -> nn.Module:
        """Fold BatchNorm into the convolution for evaluation."""
        if isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            self.conv = fuse_conv_bn_eval(self.conv, self.norm)
            self.norm = None
        return self

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
            bn_weight_init: float = 1.0,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            **dd,
        )
        self.norm = norm(out_dim, **dd) if norm else None
        self.act = act_layer() if act_layer else None
        if self.norm is not None and isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    @torch.no_grad()
    def fuse(self) -> nn.Module:
        """Fold BatchNorm into the convolution for evaluation."""
        if isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            self.conv = fuse_conv_bn_eval(self.conv, self.norm)
            self.norm = None
        return self

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


@torch.no_grad()
def _fuse_depthwise_residual(conv: nn.Conv2d, scale: Optional[torch.Tensor] = None) -> None:
    """Fold a residual identity and optional branch gate into a depthwise convolution."""
    assert conv.groups == conv.in_channels == conv.out_channels
    assert conv.stride == (1, 1) and conv.dilation == (1, 1)
    kh, kw = conv.kernel_size
    if scale is None:
        conv.weight[:, 0, kh // 2, kw // 2] += 1
    else:
        scale = scale.reshape(-1)
        conv.weight.mul_(scale[:, None, None, None])
        if conv.bias is not None:
            conv.bias.mul_(scale)
        conv.weight[:, 0, kh // 2, kw // 2] += 1 - scale


class PatchMerging(nn.Module):
    """MBConv-style downsample with SE, plus the reference implementation's depthwise bypasses."""

    fused: torch.jit.Final[bool]

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
        self.fused = False
        hidden_dim = int(out_dim * ratio)
        self.conv = nn.Sequential(
            ConvLayer2d(in_dim, hidden_dim, kernel_size=1, **dd),
            ConvLayer2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, groups=hidden_dim, **dd),
            SqueezeExcite(hidden_dim, 0.25, **dd),
            ConvLayer2d(hidden_dim, out_dim, kernel_size=1, act_layer=None, **dd),
        )
        self.dwconv1 = ConvLayer2d(in_dim, in_dim, 3, padding=1, groups=in_dim, act_layer=None, **dd)
        self.dwconv2 = ConvLayer2d(out_dim, out_dim, 3, padding=1, groups=out_dim, act_layer=None, **dd)

    @torch.no_grad()
    def fuse(self) -> nn.Module:
        """Fold the depthwise residual branches for evaluation."""
        if not self.fused:
            assert not self.training, 'Fusion requires evaluation mode.'
            self.dwconv1.fuse()
            self.dwconv2.fuse()
            _fuse_depthwise_residual(self.dwconv1.conv)
            _fuse_depthwise_residual(self.dwconv2.conv)
            self.fused = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv1(x) if self.fused else x + self.dwconv1(x)
        x = self.conv(x)
        x = self.dwconv2(x) if self.fused else x + self.dwconv2(x)
        return x


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
            ssd_expand: float = 1.0,
            state_dim: int = 64,
            a_init_range: Tuple[float, float] = (1.0, 16.0),
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
            conv_dim,
            conv_dim,
            3,
            1,
            1,
            groups=conv_dim,
            norm=None,
            act_layer=None,
            bn_weight_init=0,
            **dd,
        )
        self.hz_proj = ConvLayer1d(dim, 2 * self.d_inner, 1, norm=None, act_layer=None, **dd)
        self.out_proj = ConvLayer1d(self.d_inner, dim, 1, norm=None, act_layer=None, bn_weight_init=0, **dd)

        self.a_init_range = a_init_range
        self.A = nn.Parameter(torch.empty(self.state_dim, **dd))
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.A, *self.a_init_range)
        nn.init.ones_(self.D)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = x.shape
        x = x.flatten(2)

        BCdt = self.dw(self.BCdt_proj(x).reshape(batch, -1, height, width)).flatten(2)
        b, c, dt = torch.split(BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)
        a = (dt + self.A.view(1, -1, 1)).softmax(-1)

        ab = a * b
        h = x @ ab.transpose(-2, -1)

        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1)
        h = self.out_proj(h * self.act(z) + h * self.D)
        y = h @ c  # B C N, B C L -> B C L

        y = y.reshape(batch, -1, height, width).contiguous()
        return y, h


class EfficientViMBlock(nn.Module):
    """Depthwise conv -> HSM-SSD mixer -> depthwise conv -> FFN, each with a LayerScale residual."""

    fused: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 4.0,
            ssd_expand: float = 1.0,
            state_dim: int = 64,
            drop_path: float = 0.0,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.fused = False
        self.mixer = HSMSSD(dim, ssd_expand=ssd_expand, state_dim=state_dim, **dd)
        self.norm = LayerNorm2d(dim, eps=1e-5, **dd)

        self.dwconv1 = ConvLayer2d(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None, **dd)
        self.dwconv2 = ConvLayer2d(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None, **dd)
        self.ffn = FFN(dim, int(dim * mlp_ratio), **dd)

        # LayerScale, one gate per branch. Applied to the branch delta so it composes with
        # DropPath on the branch output without changing the reference behaviour at
        # drop_path=0 (where the gates multiply the branch, not the residual).
        self.alpha = nn.Parameter(
            1e-4 * torch.ones(4, dim, device=device, dtype=dtype),
            requires_grad=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.alpha, 1e-4)

    @torch.no_grad()
    def fuse(self) -> nn.Module:
        """Fold the gated depthwise residual branches for evaluation."""
        if not self.fused:
            assert not self.training, 'Fusion requires evaluation mode.'
            alpha = self.alpha.detach().sigmoid()
            self.dwconv1.fuse()
            self.dwconv2.fuse()
            _fuse_depthwise_residual(self.dwconv1.conv, alpha[0])
            _fuse_depthwise_residual(self.dwconv2.conv, alpha[2])
            self.fused = True
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)

        # dwconv1
        x = self.dwconv1(x) if self.fused else x + self.drop_path(alpha[0] * (self.dwconv1(x) - x))

        # HSM-SSD
        x_prev = x
        x, h = self.mixer(self.norm(x))
        x = x_prev + self.drop_path(alpha[1] * (x - x_prev))

        # dwconv2
        x = self.dwconv2(x) if self.fused else x + self.drop_path(alpha[2] * (self.dwconv2(x) - x))

        # FFN
        x = x + self.drop_path(alpha[3] * (self.ffn(x) - x))
        return x, h


class EfficientViMStage(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            depth: int,
            mlp_ratio: float = 4.0,
            downsample: Optional[Type[nn.Module]] = None,
            ssd_expand: float = 1.0,
            state_dim: int = 64,
            drop_path: Optional[List[float]] = None,
            device=None,
            dtype=None,
    ):
        super().__init__()
        assert depth > 0
        self.depth = depth
        self.grad_checkpointing = False
        drop_path = drop_path or [0.0] * depth
        self.blocks = nn.ModuleList([
            EfficientViMBlock(
                dim=in_dim,
                mlp_ratio=mlp_ratio,
                ssd_expand=ssd_expand,
                state_dim=state_dim,
                drop_path=drop_path[i],
                device=device,
                dtype=dtype,
            )
            for i in range(depth)
        ])
        self.downsample = (
            downsample(in_dim=in_dim, out_dim=out_dim, device=device, dtype=dtype) if downsample is not None else None
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (feature_map, next_stage_input, hidden_state).

        `next_stage_input` is the input to the next stage (post-downsample when present), `feature_map`
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
        global_pool: Global pooling type, 'avg' or '' to return unpooled branch features.
        drop_rate: Classifier dropout rate.
        drop_path_rate: Stochastic depth rate, linearly ramped across all blocks.
    """

    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            embed_dim: Tuple[int, ...] = (128, 256, 512),
            depths: Tuple[int, ...] = (2, 2, 2),
            mlp_ratio: float = 4.0,
            ssd_expand: float = 1.0,
            state_dim: Tuple[int, ...] = (49, 25, 9),
            global_pool: str = 'avg',
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert global_pool in ('avg', '')
        assert global_pool or num_classes == 0
        assert len(depths) == len(embed_dim) == len(state_dim)
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = tuple(embed_dim)
        self.grad_checkpointing = False
        self.feature_info = []

        self.patch_embed = Stem(in_dim=in_chans, dim=embed_dim[0], **dd)
        dpr = calculate_drop_path_rates(drop_path_rate, sum(depths))

        stages = []
        block_idx = 0
        # stem reduces by 16; each downsample that follows a stage doubles it again. The
        # x_out map a stage reports is taken *before* its trailing downsample.
        reduction = 16
        for i_layer in range(self.num_layers):
            stage_dpr = dpr[block_idx : block_idx + depths[i_layer]]
            block_idx += depths[i_layer]
            stages.append(
                EfficientViMStage(
                    in_dim=int(embed_dim[i_layer]),
                    out_dim=int(embed_dim[i_layer + 1]) if (i_layer < self.num_layers - 1) else 0,
                    depth=depths[i_layer],
                    mlp_ratio=mlp_ratio,
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    ssd_expand=ssd_expand,
                    state_dim=state_dim[i_layer],
                    drop_path=stage_dpr,
                    **dd,
                )
            )
            self.feature_info.append(
                dict(num_chs=embed_dim[i_layer], reduction=reduction, module=f'stages.{i_layer}')
            )
            if i_layer < self.num_layers - 1:
                reduction *= 2
        self.stages = nn.ModuleList(stages)

        self.num_features = self.head_hidden_size = sum(embed_dim) + embed_dim[-1]
        self.weights = nn.Parameter(torch.ones(self.num_layers + 1, **dd), requires_grad=num_classes > 0)
        self.norm = nn.ModuleList([
            *[LayerNorm(dim, eps=1e-5, **dd) for dim in embed_dim],
            LayerNorm2d(embed_dim[-1], eps=1e-5, **dd),
        ])
        self.head_drop = nn.Dropout(drop_rate)
        self.heads = self._build_heads(num_classes, dd)
        self.init_weights(needs_reset=False)

    def _build_heads(self, num_classes: int, dd: Optional[Dict[str, Any]] = None) -> nn.ModuleList:
        dd = dd or {}
        return nn.ModuleList([
            nn.Linear(dim, num_classes, **dd) if num_classes > 0 else nn.Identity()
            for dim in (*self.embed_dim, self.embed_dim[-1])
        ])

    @torch.jit.ignore
    def init_weights(self, needs_reset: bool = True) -> None:
        """Initialize parameters and restore buffers after ``to_empty()``."""
        nn.init.ones_(self.weights)
        if hasattr(self, 'weights_dist'):
            nn.init.ones_(self.weights_dist)
        self.apply(partial(self._init_weights, needs_reset=needs_reset))

    def _init_weights(self, m: nn.Module, needs_reset: bool = True) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_parameters()
        elif needs_reset and hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return {name for name, _ in self.named_parameters() if name.endswith('.D')}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^patch_embed',
            blocks=r'^stages\.(\d+)'
            if coarse
            else [
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^stages\.(\d+)\.downsample', (99999,)),
                (r'^norm', (99999,)),
            ],
        )

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
            assert global_pool in ('avg', '')
            self.global_pool = global_pool
        assert self.global_pool or num_classes == 0
        self.heads = self._build_heads(num_classes, dd)
        self.heads.train(was_training)
        self.weights.requires_grad_(num_classes > 0)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
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
        features = []
        if torch.jit.is_scripting() or not stop_early:
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]
        for feat_idx, (stage, stage_norm) in enumerate(zip(stages, self.norm)):
            x_out, x, h = stage(x)
            if feat_idx in take_indices:
                intermediates.append(stage_norm(x_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) if norm else x_out)
            if not intermediates_only:
                features.append(stage_norm(h.transpose(1, 2)).transpose(1, 2))

        if intermediates_only:
            return intermediates
        if torch.jit.is_scripting() or not stop_early or max_index == len(self.stages) - 1:
            x = self.norm[-1](x)
        else:
            x = x_out
        features.append(x.flatten(2))
        return features, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ) -> List[int]:
        """Prune stages not required for the specified intermediate features.

        With prune_head, also remove output norms unused by the selected intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        num_layers = max_index + 1
        if num_layers < self.num_layers:
            assert prune_head, 'The multi-stage classifier requires all stages.'
            self.stages = self.stages[:num_layers]
            self.feature_info = self.feature_info[:num_layers]
            self.stages[-1].downsample = None
            self.norm = nn.ModuleList([*self.norm[:num_layers], nn.Identity()])
            self.weights = nn.Parameter(torch.cat([self.weights[:num_layers], self.weights[-1:]]).detach())
            if hasattr(self, 'weights_dist'):
                self.weights_dist = nn.Parameter(
                    torch.cat([self.weights_dist[:num_layers], self.weights_dist[-1:]]).detach()
                )
            self.embed_dim = self.embed_dim[:num_layers]
            self.num_layers = num_layers
            self.num_features = self.head_hidden_size = sum(self.embed_dim) + self.embed_dim[-1]
        if prune_norm:
            self.norm = nn.ModuleList([nn.Identity() for _ in self.norm])
        elif prune_head:
            # Keep only norms used by the selected intermediate maps. The final-map
            # norm and unselected hidden-state norms belong to the removed head paths.
            self.norm = nn.ModuleList([
                norm if i in take_indices else nn.Identity() for i, norm in enumerate(self.norm)
            ])
        if prune_head:
            self.reset_classifier(0)
        return take_indices

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return normalized, unpooled hidden states and the final map as NCL tensors."""
        x = self.patch_embed(x)
        features = []
        for stage, norm in zip(self.stages, self.norm):
            _, x, h = stage(x)
            features.append(norm(h.transpose(1, 2)).transpose(1, 2))
        features.append(self.norm[-1](x).flatten(2))
        return features

    def forward_head(
            self,
            x: List[torch.Tensor],
            pre_logits: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if not self.global_pool:
            return x
        x = [self.head_drop(t.mean(-1)) for t in x]
        if pre_logits or self.num_classes == 0:
            return torch.cat(x, dim=1)
        weights = self.weights.softmax(-1)
        logits = [weights[i] * head(x[i]) for i, head in enumerate(self.heads)]
        return torch.stack(logits).sum(0)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.forward_features(x)
        return self.forward_head(x)


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
        self.weights_dist = nn.Parameter(torch.ones(self.num_layers + 1, **dd), requires_grad=num_classes > 0)
        self.heads_dist = self._build_heads(num_classes, dd)
        self.heads_dist.apply(self._init_weights)
        self.distilled_training = False

    @torch.jit.ignore
    def get_classifier(self) -> Tuple[nn.ModuleList, nn.ModuleList]:
        return self.heads, self.heads_dist

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        super().reset_classifier(num_classes, global_pool=global_pool)
        dd = get_device_dtype(self)
        self.heads_dist = self._build_heads(num_classes, dd)
        self.heads_dist.train(self.training)
        self.weights_dist.requires_grad_(num_classes > 0)

    @torch.jit.ignore
    def set_distilled_training(self, enable: bool = True):
        self.distilled_training = enable

    def forward_head(
            self,
            x: List[torch.Tensor],
            pre_logits: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if not self.global_pool:
            return x
        x = [self.head_drop(t.mean(-1)) for t in x]
        if pre_logits or self.num_classes == 0:
            return torch.cat(x, dim=1)
        weights = self.weights.softmax(-1)
        weights_dist = self.weights_dist.softmax(-1)
        logits = [weights[i] * head(x[i]) for i, head in enumerate(self.heads)]
        logits_dist = [weights_dist[i] * head(x[i]) for i, head in enumerate(self.heads_dist)]
        z, z_dist = torch.stack(logits).sum(0), torch.stack(logits_dist).sum(0)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            return z, z_dist
        return (z + z_dist) / 2


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """Map upstream EfficientViM checkpoints onto this module's names.

    Module names match the reference implementation. Normalization parameters are flattened,
    training-checkpoint envelopes are unwrapped, and the parallel distillation head is
    dropped for non-distilled models. The released checkpoints
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
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith('heads_dist.') or k == 'weights_dist')}
    # Upstream norms store broadcast-shaped affine parameters instead of channel vectors.
    state_dict = dict(state_dict)
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for suffix in ('weight', 'bias'):
                key = f'{name}.{suffix}'
                if key in state_dict:
                    state_dict[key] = state_dict[key].reshape(module.normalized_shape)
    return state_dict


def _cfg(**kwargs: Any) -> Dict[str, Any]:
    return {
        'url': '',
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.875,
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.conv.0.conv',
        # all four fusion heads are num_classes-sized -- list them so timm resets every one on a
        # num_classes change (a single 'heads.3' leaves heads.0..2 mismatched on transfer).
        'classifier': ('heads.0', 'heads.1', 'heads.2', 'heads.3'),
        'paper_ids': 'arXiv:2411.15241',
        'origin_url': 'https://github.com/mlvlab/EfficientViM',
        'license': 'mit',
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'efficientvim_m1.e300_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'efficientvim_m1.e450_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'efficientvim_m1_dist.in1k': _cfg(
        hf_hub_id='timm/',
        classifier=(
            'heads.0',
            'heads.1',
            'heads.2',
            'heads.3',
            'heads_dist.0',
            'heads_dist.1',
            'heads_dist.2',
            'heads_dist.3',
        ),
    ),
    'efficientvim_m2.e300_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'efficientvim_m2.e450_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'efficientvim_m2_dist.in1k': _cfg(
        hf_hub_id='timm/',
        classifier=(
            'heads.0',
            'heads.1',
            'heads.2',
            'heads.3',
            'heads_dist.0',
            'heads_dist.1',
            'heads_dist.2',
            'heads_dist.3',
        ),
    ),
    'efficientvim_m3.e300_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'efficientvim_m3.e450_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'efficientvim_m3_dist.in1k': _cfg(
        hf_hub_id='timm/',
        classifier=(
            'heads.0',
            'heads.1',
            'heads.2',
            'heads.3',
            'heads_dist.0',
            'heads_dist.1',
            'heads_dist.2',
            'heads_dist.3',
        ),
    ),
    'efficientvim_m4.e300_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256),
    ),
    'efficientvim_m4.e450_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256),
    ),
    'efficientvim_m4_dist.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256),
        classifier=(
            'heads.0',
            'heads.1',
            'heads.2',
            'heads.3',
            'heads_dist.0',
            'heads_dist.1',
            'heads_dist.2',
            'heads_dist.3',
        ),
    ),
})


def _create_efficientvim(
        variant: str,
        pretrained: bool = False,
        distilled: bool = False,
        **kwargs: Any,
) -> EfficientViM:
    feature_cfg = dict(out_indices=(0, 1, 2), feature_cls='getter')
    return build_model_with_cfg(
        EfficientViMDistilled if distilled else EfficientViM,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=feature_cfg,
        **kwargs,
    )


@register_model
def efficientvim_m1(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M1: 6.7M params, 72.9 top-1 @ 300 epochs."""
    model_args = dict(
        embed_dim=(128, 192, 320),
        depths=(2, 2, 2),
        state_dim=(49, 25, 9),
    )
    return _create_efficientvim("efficientvim_m1", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def efficientvim_m2(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M2: 13.9M params, 75.4 top-1 @ 300 epochs."""
    model_args = dict(
        embed_dim=(128, 256, 512),
        depths=(2, 2, 2),
        state_dim=(49, 25, 9),
    )
    return _create_efficientvim("efficientvim_m2", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def efficientvim_m3(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M3: 16.6M params, 77.6 top-1 @ 300 epochs."""
    model_args = dict(
        embed_dim=(224, 320, 512),
        depths=(2, 2, 2),
        state_dim=(49, 25, 9),
    )
    return _create_efficientvim("efficientvim_m3", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def efficientvim_m4(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M4: 19.6M params, 79.4 top-1 @ 300 epochs, 256x256 input."""
    model_args = dict(
        embed_dim=(224, 320, 512),
        depths=(3, 4, 2),
        state_dim=(64, 32, 16),
    )
    return _create_efficientvim("efficientvim_m4", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def efficientvim_m1_dist(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M1, distilled: 6.7M params, 74.6 top-1."""
    model_args = dict(
        embed_dim=(128, 192, 320),
        depths=(2, 2, 2),
        state_dim=(49, 25, 9),
    )
    return _create_efficientvim(
        "efficientvim_m1_dist",
        pretrained=pretrained,
        distilled=True,
        **dict(model_args, **kwargs),
    )


@register_model
def efficientvim_m2_dist(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M2, distilled: 13.9M params, 76.7 top-1."""
    model_args = dict(
        embed_dim=(128, 256, 512),
        depths=(2, 2, 2),
        state_dim=(49, 25, 9),
    )
    return _create_efficientvim(
        "efficientvim_m2_dist",
        pretrained=pretrained,
        distilled=True,
        **dict(model_args, **kwargs),
    )


@register_model
def efficientvim_m3_dist(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M3, distilled: 16.6M params, 79.1 top-1."""
    model_args = dict(
        embed_dim=(224, 320, 512),
        depths=(2, 2, 2),
        state_dim=(49, 25, 9),
    )
    return _create_efficientvim(
        "efficientvim_m3_dist",
        pretrained=pretrained,
        distilled=True,
        **dict(model_args, **kwargs),
    )


@register_model
def efficientvim_m4_dist(pretrained: bool = False, **kwargs: Any) -> EfficientViM:
    """EfficientViM-M4, distilled: 19.6M params, 80.7 top-1, 256x256 input."""
    model_args = dict(
        embed_dim=(224, 320, 512),
        depths=(3, 4, 2),
        state_dim=(64, 32, 16),
    )
    return _create_efficientvim(
        "efficientvim_m4_dist",
        pretrained=pretrained,
        distilled=True,
        **dict(model_args, **kwargs),
    )
