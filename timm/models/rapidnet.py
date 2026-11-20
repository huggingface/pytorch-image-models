"""RapidNet
RapidNet: Multi-Level Dilated Convolution Based Mobile Backbone (WACV2025)
- `RapidNet: Multi-Level Dilated Convolution Based Mobile Backbone` - https://arxiv.org/abs/2412.10995
- code: https://github.com/mmunir127/RapidNet-Official

@InProceedings{Munir_2025_WACV,
    author    = {Munir, Mustafa and Rahman, Md Mostafijur and Marculescu, Radu},
    title     = {RapidNet: Multi-Level Dilated Convolution Based Mobile Backbone},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {8291-8301}
}
Modifications by / Copyright 2026 Ryan Hou & Ross Wightman, original copyrights below
"""
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import (
    ConvNormAct,
    DropPath,
    LayerScale2d,
    Linear,
    SelectAdaptivePool2d,
    calculate_drop_path_rates,
    get_device_dtype,
    trunc_normal_,
)
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['RapidNet']


class DepthWiseSeparable(nn.Module):
    """Pointwise-expand -> depthwise -> pointwise-project block."""

    def __init__(
            self,
            in_dim: int,
            kernel: int = 3,
            expansion_ratio: float = 4.,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        hidden_dim = int(in_dim * expansion_ratio)
        self.pw1 = ConvNormAct(in_dim, hidden_dim, 1, bias=True, norm_layer=norm_layer, act_layer=act_layer, **dd)
        self.dw = ConvNormAct(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel,
            groups=hidden_dim,
            bias=True,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **dd
        )
        self.pw2 = ConvNormAct(hidden_dim, in_dim, 1, bias=True, norm_layer=norm_layer, apply_act=False, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw1(x)
        x = self.dw(x)
        x = self.pw2(x)
        return x


class InvertedResidual(nn.Module):
    """Local-stage residual block (DepthWiseSeparable + LayerScale)."""

    def __init__(
            self,
            dim: int,
            kernel: int = 3,
            expansion_ratio: float = 4.,
            drop_path: float = 0.,
            layer_scale_init_value: Optional[float] = 1e-5,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.dws = DepthWiseSeparable(dim, kernel, expansion_ratio, norm_layer, act_layer, **dd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(dim, layer_scale_init_value, **dd)
        else:
            self.layer_scale = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.layer_scale(self.dws(x)))


class DilatedConv(nn.Module):
    """Parallel multi-level dilated convolutions (d=2 and d=3)."""

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.conv_d2 = ConvNormAct(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=3,
            dilation=2,
            bias=True,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **dd
        )
        self.conv_d3 = ConvNormAct(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=3,
            dilation=3,
            bias=True,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **dd,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_d2(x) + self.conv_d3(x)


class RepCPE(nn.Module):
    """Reparameterizable conditional positional encoding (depthwise 7x7 + skip)."""

    def __init__(self, dim: int, device=None, dtype=None):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.pe = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe(x) + x

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        """Fuse residual (identity) into depthwise pe conv, return plain conv."""
        kernel_size = self.pe.kernel_size
        center = (kernel_size[0] // 2, kernel_size[1] // 2)
        fused = nn.Conv2d(
            in_channels=self.pe.in_channels,
            out_channels=self.pe.out_channels,
            kernel_size=kernel_size,
            stride=self.pe.stride,
            padding=self.pe.padding,
            dilation=self.pe.dilation,
            groups=self.pe.groups,
            bias=self.pe.bias is not None,
            device=self.pe.weight.device,
            dtype=self.pe.weight.dtype,
        )
        fused.weight.copy_(self.pe.weight)
        fused.weight[:, :, center[0], center[1]].add_(1.)
        if self.pe.bias is not None:
            fused.bias.copy_(self.pe.bias)
        return fused


class ConvFFN(nn.Module):
    """Large-kernel convolutional FFN (DW7 + 1x1 expand + 1x1 project)."""

    def __init__(
            self,
            in_dim: int,
            hidden_dim: Optional[int] = None,
            out_dim: Optional[int] = None,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.conv = ConvNormAct(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=7,
            groups=in_dim,
            bias=False,
            norm_layer=norm_layer,
            apply_act=False,
            **dd
        )
        self.fc1 = ConvNormAct(in_dim, hidden_dim, 1, bias=True, norm_layer=norm_layer, act_layer=act_layer, **dd)
        self.fc2 = ConvNormAct(hidden_dim, out_dim, 1, bias=True, norm_layer=norm_layer, apply_act=False, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class MLDC(nn.Module):
    """Multi-Level Dilated Convolution mixer (RepCPE + 1x1 + dilated conv + 1x1)."""

    def __init__(
            self,
            dim: int,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.cpe = RepCPE(dim, **dd)
        self.fc1 = ConvNormAct(dim, dim, kernel_size=1, bias=True, norm_layer=norm_layer, apply_act=False, **dd)
        self.graph_conv = DilatedConv(dim, dim, norm_layer=norm_layer, act_layer=act_layer, **dd)
        self.fc2 = ConvNormAct(dim, dim, kernel_size=1, bias=True, norm_layer=norm_layer, apply_act=False, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cpe(x)
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        return x


class MLDC_LKFFN(nn.Module):
    """Global-stage block (MLDC mixer + ConvFFN, dual LayerScale residuals)."""

    def __init__(
            self,
            in_dim: int,
            drop_path: float = 0.,
            layer_scale_init_value: Optional[float] = 1e-5,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.mixer = MLDC(in_dim, norm_layer=norm_layer, act_layer=act_layer, **dd)
        self.ffn = ConvFFN(in_dim, in_dim * 4, in_dim, norm_layer=norm_layer, act_layer=act_layer, **dd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale2d(in_dim, layer_scale_init_value, **dd)
            self.layer_scale_2 = LayerScale2d(in_dim, layer_scale_init_value, **dd)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1(self.mixer(x)))
        x = x + self.drop_path(self.layer_scale_2(self.ffn(x)))
        return x


class Stage(nn.Module):
    """One RapidNet stage: strided downsample (iff in_dim != dim) + local/global blocks."""

    def __init__(
            self,
            in_dim: int,
            dim: int,
            local_depth: int,
            global_depth: int,
            drop_path: Union[float, List[float]] = 0.,
            layer_scale_init_value: Optional[float] = 1e-5,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.grad_checkpointing = False
        self.downsample = ConvNormAct(
            in_channels=in_dim,
            out_channels=dim,
            kernel_size=3,
            stride=2,
            bias=True,
            norm_layer=norm_layer,
            apply_act=False,
            **dd,
        ) if in_dim != dim else nn.Identity()

        if isinstance(drop_path, float):
            drop_path = [drop_path] * (local_depth + global_depth)
        blocks = []
        for i in range(local_depth):
            blocks.append(InvertedResidual(
                dim,
                kernel=3,
                expansion_ratio=4.,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                **dd,
            ))
        for i in range(global_depth):
            blocks.append(MLDC_LKFFN(
                dim,
                drop_path=drop_path[local_depth + i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                **dd,
            ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class RapidNet(nn.Module):
    """RapidNet backbone + classifier head (timm style)."""

    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            dims: Tuple[int, ...] = (32, 64, 112, 224),
            depths_local: Tuple[int, ...] = (2, 2, 6, 2),
            depths_global: Tuple[int, ...] = (0, 0, 2, 2),
            emb_dims: int = 512,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            layer_scale_init_value: Optional[float] = 1e-5,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert len(dims) == len(depths_local) == len(depths_global) == 4
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.feature_info = []

        # stem: /4, 3 -> c0/2 -> c0
        self.stem = nn.Sequential(
            ConvNormAct(in_chans, dims[0] // 2, 3, 2, bias=True, norm_layer=norm_layer, act_layer=act_layer, **dd),
            ConvNormAct(dims[0] // 2, dims[0], 3, 2, bias=True, norm_layer=norm_layer, act_layer=act_layer, **dd),
        )
        prev_dim = dims[0]

        stage_depths = [n_local + n_global for n_local, n_global in zip(depths_local, depths_global)]
        dpr = calculate_drop_path_rates(drop_path_rate, stage_depths, stagewise=True)

        stages = []
        for i in range(len(dims)):
            stages.append(Stage(
                in_dim=prev_dim,
                dim=dims[i],
                local_depth=depths_local[i],
                global_depth=depths_global[i],
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                **dd,
            ))
            prev_dim = dims[i]
            self.feature_info.append(dict(num_chs=dims[i], reduction=2 ** (i + 2), module=f'stages.{i}'))
        self.stages = nn.Sequential(*stages)

        # classifier head: fused 1x1 proj to emb_dims + Linear heads (dual for distillation)
        self.num_features = dims[-1]
        self.head_hidden_size = emb_dims
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        assert self.global_pool.feat_mult() == 1, 'Pooling that changes the pooled width is not supported.'
        self.conv_head = ConvNormAct(dims[-1], emb_dims, 1, bias=True, norm_layer=norm_layer, act_layer=act_layer, **dd)
        self.flatten = nn.Flatten(1) if self.global_pool.pool_type else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = Linear(emb_dims, num_classes, **dd) if num_classes > 0 else nn.Identity()
        self.head_dist = Linear(emb_dims, num_classes, **dd) if num_classes > 0 else nn.Identity()
        self.distilled_training = False  # set True to train w/ separate distillation heads
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return set()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        matcher = dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^conv_head', (99999,)),
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> Tuple[nn.Module, nn.Module]:
        return self.head, self.head_dist

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None, reset_other: bool = False):
        dd = get_device_dtype(self)
        self.num_classes = num_classes
        if global_pool is None:
            global_pool = self.global_pool.pool_type
        # cannot meaningfully change pooling of efficient head after creation    
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        assert self.global_pool.feat_mult() == 1, 'Pooling that changes the pooled width is not supported.'
        self.global_pool.train(self.training)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.flatten.train(self.training)
        if reset_other:
            self.conv_head = nn.Identity().train(self.training)
            self.head_hidden_size = self.num_features
        self.head = Linear(self.head_hidden_size, num_classes, **dd) if num_classes > 0 else nn.Identity()
        self.head_dist = Linear(self.head_hidden_size, num_classes, **dd) if num_classes > 0 else nn.Identity()
        self.head.train(self.training)
        self.head_dist.train(self.training)

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
            norm: Unused; there is no final feature normalization layer.
            stop_early: Stop iterating over blocks when last desired intermediate hit.
            output_fmt: Shape of intermediate feature outputs.
            intermediates_only: Only return intermediate features.
        Returns:
            List of intermediate features or tuple of (final features, intermediates).
        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages), indices)

        x = self.stem(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]

        for feat_idx, stage in enumerate(stages):
            x = stage(x)
            if feat_idx in take_indices:
                intermediates.append(x)

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
        prune_end = max_index < len(self.stages) - 1
        self.stages = self.stages[:max_index + 1]  # truncate blocks w/ stem as idx 0
        if prune_end:
            # features now end at the last kept tap, the head is sized for the final features so it goes too
            self.num_features = self.feature_info[max_index]['num_chs']
        if prune_head or prune_end:
            self.reset_classifier(0, '', reset_other=True)
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.flatten(x)
        x = self.head_drop(x)
        if pre_logits:
            return x
        x, x_dist = self.head(x), self.head_dist(x)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train/finetune, inference average the classifier predictions
            return (x + x_dist) / 2

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @torch.no_grad()
    def fuse(self):
        def fuse_children(net: nn.Module):
            for child_name, child in net.named_children():
                if hasattr(child, "fuse"):
                    fused = child.fuse()
                    if fused is not child:
                        setattr(net, child_name, fused)
                        fuse_children(fused)
                    else:
                        fuse_children(child)
                else:
                    fuse_children(child)

        fuse_children(self)
        return self


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    """Remap official RapidNet-Official keys to timm RapidNet (ConvNormAct) keys.

    Official layout: stem.stem.{0..5} / backbone.{s}.{j} (j==0 & s>0 is Downsample)
    / prediction.{1,2} / head / dist_head, LayerScale as (C,1,1) params,
    heads as (N,emb,1,1) convs. timm layout uses ConvNormAct (.conv/.bn).
    """
    state_dict = state_dict.get('state_dict', state_dict)
    state_dict = state_dict.get('model', state_dict)
    if 'stem.0.conv.weight' in state_dict:
        return state_dict
    # local (InvertedResidual) blocks own `.dws.` keys and use `layer_scale`,
    # global (MLDC_LKFFN) blocks use `layer_scale_1/2`; record (stage, block) idx
    local_idx = set()
    for k in state_dict:
        m = re.match(r'^backbone\.(\d+)\.(\d+)\.dws\.', k)
        if m:
            s, j = int(m.group(1)), int(m.group(2))
            local_idx.add((s, j - 1 if s > 0 else j))
            continue
        m = re.match(r'^stages\.(\d+)\.blocks\.(\d+)\.dws\.', k)
        if m:
            local_idx.add((int(m.group(1)), int(m.group(2))))
    out_dict = {}
    for k, v in state_dict.items():
        k = re.sub(r'^stem\.stem\.0\.', 'stem.0.conv.', k)
        k = re.sub(r'^stem\.stem\.1\.', 'stem.0.bn.', k)
        k = re.sub(r'^stem\.stem\.3\.', 'stem.1.conv.', k)
        k = re.sub(r'^stem\.stem\.4\.', 'stem.1.bn.', k)
        k = re.sub(r'^backbone\.', 'stages.', k)
        m = re.match(r'^stages\.(\d+)\.(\d+)\.(.*)$', k)
        if m:
            s, j, rest = int(m.group(1)), int(m.group(2)), m.group(3)
            if s > 0 and j == 0:
                k = f'stages.{s}.downsample.{rest}'
                k = k.replace('.conv.0.', '.conv.').replace('.conv.1.', '.bn.')
            else:
                bi = j - 1 if s > 0 else j
                k = f'stages.{s}.blocks.{bi}.{rest}'
                if (s, bi) in local_idx and rest == 'layer_scale_1':
                    k = k[:-(len('layer_scale_1'))] + 'layer_scale.gamma'
                    v = v.reshape(v.shape[0])
                k = k.replace('.dws.pw1.', '.dws.pw1.conv.')
                k = k.replace('.dws.norm1.', '.dws.pw1.bn.')
                k = k.replace('.dws.dw.', '.dws.dw.conv.')
                k = k.replace('.dws.norm2.', '.dws.dw.bn.')
                k = k.replace('.dws.pw2.', '.dws.pw2.conv.')
                k = k.replace('.dws.norm3.', '.dws.pw2.bn.')
                k = k.replace('.mixer.fc1.0.', '.mixer.fc1.conv.')
                k = k.replace('.mixer.fc1.1.', '.mixer.fc1.bn.')
                k = k.replace('.mixer.fc2.0.', '.mixer.fc2.conv.')
                k = k.replace('.mixer.fc2.1.', '.mixer.fc2.bn.')
                k = k.replace('.graph_conv.nn.0.', '.graph_conv.conv_d2.conv.')
                k = k.replace('.graph_conv.nn.1.', '.graph_conv.conv_d2.bn.')
                k = k.replace('.graph_conv.nn_2.0.', '.graph_conv.conv_d3.conv.')
                k = k.replace('.graph_conv.nn_2.1.', '.graph_conv.conv_d3.bn.')
                k = k.replace('.ffn.norm1.', '.ffn.conv.bn.')
                k = k.replace('.ffn.fc1.', '.ffn.fc1.conv.')
                k = k.replace('.ffn.norm2.', '.ffn.fc1.bn.')
                k = k.replace('.ffn.fc2.', '.ffn.fc2.conv.')
                k = k.replace('.ffn.norm3.', '.ffn.fc2.bn.')
        k = k.replace('prediction.1.', 'conv_head.conv.')
        k = k.replace('prediction.2.', 'conv_head.bn.')
        k = re.sub(r'^dist_head\.', 'head_dist.', k)
        if re.match(r'layer_scale(_\d+)?$', k.rsplit('.', 1)[-1]) and not k.endswith('.gamma'):
            k = k + '.gamma'
            v = v.reshape(v.shape[0])
        if k in ('head.weight', 'head_dist.weight'):
            v = v.squeeze(-1).squeeze(-1)
        out_dict[k] = v
    return out_dict


def _cfg(url: str = '', **kwargs: Any) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0.conv', 'classifier': ('head', 'head_dist'),
        'license': 'apache-2.0',
        'paper_ids': 'arXiv:2412.10995',
        'paper_name': 'RapidNet: Multi-Level Dilated Convolution Based Mobile Backbone',
        'origin_url': 'https://github.com/mmunir127/RapidNet-Official',
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'rapidnet_t.dist_in1k': _cfg(
        hf_hub_id='SLDGroup/RapidNet',
        hf_hub_filename='RapidNet_Ti.pth',
    ),
    'rapidnet_s.dist_in1k': _cfg(
        hf_hub_id='SLDGroup/RapidNet',
        hf_hub_filename='RapidNet_S.pth',
    ),
    'rapidnet_m.dist_in1k': _cfg(
        hf_hub_id='SLDGroup/RapidNet',
        hf_hub_filename='RapidNet_M.pth',
    ),
    'rapidnet_b.dist_in1k': _cfg(
        hf_hub_id='SLDGroup/RapidNet',
        hf_hub_filename='RapidNet_B.pth',
    ),
})


def _create_rapidnet(variant: str, pretrained: bool = False, **kwargs: Any) -> RapidNet:
    model = build_model_with_cfg(
        RapidNet, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )
    return model


@register_model
def rapidnet_t(pretrained: bool = False, **kwargs: Any) -> RapidNet:
    model_args = dict(
        dims=(32, 64, 112, 224),
        depths_local=(2, 2, 6, 2),
        depths_global=(0, 0, 2, 2),
        emb_dims=512,
        drop_path_rate=0.1,
    )
    return _create_rapidnet('rapidnet_t', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def rapidnet_s(pretrained: bool = False, **kwargs: Any) -> RapidNet:
    model_args = dict(
        dims=(32, 64, 112, 224),
        depths_local=(3, 3, 9, 3),
        depths_global=(0, 0, 3, 3),
        emb_dims=512,
        drop_path_rate=0.1,
    )
    return _create_rapidnet('rapidnet_s', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def rapidnet_m(pretrained: bool = False, **kwargs: Any) -> RapidNet:
    model_args = dict(
        dims=(32, 64, 160, 320),
        depths_local=(3, 3, 9, 3),
        depths_global=(0, 0, 3, 3),
        emb_dims=512,
        drop_path_rate=0.1,
    )
    return _create_rapidnet('rapidnet_m', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def rapidnet_b(pretrained: bool = False, **kwargs: Any) -> RapidNet:
    model_args = dict(
        dims=(64, 128, 224, 416),
        depths_local=(3, 3, 9, 3),
        depths_global=(0, 0, 3, 3),
        emb_dims=768,
        drop_path_rate=0.1,
    )
    return _create_rapidnet('rapidnet_b', pretrained=pretrained, **dict(model_args, **kwargs))
