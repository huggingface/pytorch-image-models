"""CPUBone

CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities
Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni
Conference on Computer Vision and Pattern Recognition (CVPR), 2025

Adapted from the original implementation (see `cpubone_original.py`) to idiomatic timm code.
Remaining cleanup steps are tracked in `TODO_cpubone.md` at the repository root.
"""
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import LayerType, get_act_layer, get_norm_layer
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._features_fx import register_notrace_module
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['CPUBone']


def remap_legacy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap keys from original CPUBone checkpoints to the current model layout."""
    remapped = {}
    for k, v in state_dict.items():
        # conv_proj was nn.Sequential([conv, bn]) → now ConvLayer with .conv / .bn
        k = k.replace(".conv_proj.0.", ".conv_proj.conv.")
        k = k.replace(".conv_proj.1.", ".conv_proj.norm.")
        # head was OpSequential([ConvLayer, AdaptiveAvgPool2d, LinearLayer, LinearLayer]) → named children
        k = k.replace("head.op_list.0.", "head.in_conv.")
        k = k.replace("head.op_list.2.", "head.pre_classifier.")
        k = k.replace("head.op_list.3.", "head.classifier.")
        # backbone was a submodule → stem / stages are now top-level children
        k = k.replace("backbone.input_stem.", "stem.")
        k = k.replace("backbone.stages.", "stages.")
        # input_stem / stages were OpSequential (module list under .op_list) → plain nn.Sequential
        k = k.replace(".op_list.", ".")
        remapped[k] = v
    return remapped


def get_same_padding(kernel_size: int, stride: int = 1) -> int:
    """Padding that keeps 'same' spatial behaviour for the kernel sizes used in CPUBone.

    kernel_size 2 is special: with stride 2 it tiles exactly (no padding), with stride 1 it needs an
    asymmetric left/top pad, signalled by -1 and handled in ConvLayer with an explicit ZeroPad2d.
    """
    if kernel_size == 2:
        return 0 if stride == 2 else -1
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


class LinearLayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            use_bias: bool = True,
            dropout: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
            act_layer: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        # note: covers nn.LayerNorm, whose first arg is normalized_shape rather than num_features
        self.norm = norm_layer(out_features) if norm_layer is not None else None
        self.act = act_layer() if act_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
            self,
            main: nn.Module,
            shortcut: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.main = main
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is None:
            return self.main(x)
        return self.main(x) + self.shortcut(x)


class ConvLayer(nn.Module):
    """Conv + optional norm + optional act.

    NOTE deliberately not timm's ConvNormAct: CPUBone needs activation-without-norm and an
    asymmetric left/top pad for even kernels at stride 1, both of which ConvNormAct can only
    express by subverting its module layout (e.g. an act module in the `bn` slot).
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            use_bias: bool = False,
            norm_layer: Optional[Type[nn.Module]] = nn.BatchNorm2d,
            act_layer: Optional[Type[nn.Module]] = nn.ReLU,
    ):
        super().__init__()
        padding = get_same_padding(kernel_size, stride)
        if padding == -1:
            # even kernel at stride 1: pad asymmetrically (left/top) to keep the spatial size
            self.conv = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                    groups=groups,
                    bias=use_bias,
                ),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=use_bias,
            )
        self.norm = norm_layer(out_channels) if norm_layer is not None else None
        self.act = act_layer() if act_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class MBConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            mid_channels: Optional[int] = None,
            expand_ratio: float = 6,
            grouping: int = 1,
            use_bias: Tuple[bool, bool, bool] = (False, False, False),
            norm_layer: Tuple[Optional[Type[nn.Module]], ...] = (nn.BatchNorm2d, nn.BatchNorm2d, nn.BatchNorm2d),
            act_layer: Tuple[Optional[Type[nn.Module]], ...] = (nn.ReLU6, nn.ReLU6, None),
    ):
        super().__init__()
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        # pointwise expand
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            groups=grouping,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            use_bias=use_bias[0],
        )
        # depthwise
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            use_bias=use_bias[1],
        )
        # pointwise project
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            groups=1,
            norm_layer=norm_layer[2],
            act_layer=act_layer[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            mid_channels: Optional[int] = None,
            expand_ratio: float = 6,
            groups: int = 1,
            use_bias: Tuple[bool, bool] = (False, False),
            norm_layer: Tuple[Optional[Type[nn.Module]], ...] = (nn.BatchNorm2d, nn.BatchNorm2d),
            act_layer: Tuple[Optional[Type[nn.Module]], ...] = (nn.ReLU6, None),
    ):
        super().__init__()
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            groups=1,
            use_bias=use_bias[1],
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


@register_notrace_module  # forward() reshapes based on runtime tensor shapes, not FX-traceable
class ConvAttention(nn.Module):
    def __init__(
            self,
            input_dim: int,
            head_dim_mul: float = 1.0,
            att_stride: int = 4,
            att_kernel: int = 7,
            fuseconv: bool = False,
            smallkernel: bool = False,
            lose_transpose: bool = False,
    ):
        super().__init__()
        self.num_heads = int(max(1, (input_dim * head_dim_mul) // 30))
        self.head_dim = int((input_dim // self.num_heads) * head_dim_mul)
        self.num_keys = 3

        total_dim = int(self.head_dim * self.num_heads * self.num_keys)

        self.conv_proj = ConvLayer(
            input_dim,
            input_dim,
            kernel_size=2 if smallkernel else att_kernel,
            stride=att_stride,
            groups=input_dim,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
        )
        self.pwise = nn.Sequential(nn.Conv2d(input_dim, total_dim, kernel_size=1, stride=1, padding=0, bias=False))

        self.o_proj_inpdim = self.head_dim * self.num_heads
        self.o_proj = nn.Conv2d(self.o_proj_inpdim, input_dim, kernel_size=1, stride=1, padding=0)

        # NOTE the att_stride == 1 case replaces the module created just before; the redundant first init is
        # kept so the RNG stream matches the original implementation's parameter initialization exactly
        self.upsampling = nn.ConvTranspose2d(
            input_dim, input_dim, kernel_size=att_stride * 2, stride=att_stride, padding=att_stride // 2, groups=input_dim)
        if att_stride == 1:
            self.upsampling = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim)

        if fuseconv:
            self.o_proj = nn.Identity()
            if att_stride == 1:
                self.upsampling = nn.ConvTranspose2d(self.o_proj_inpdim, input_dim, kernel_size=3, stride=1, padding=1)
            else:
                self.upsampling = nn.ConvTranspose2d(
                    self.o_proj_inpdim, input_dim, kernel_size=att_stride * 2, stride=att_stride, padding=att_stride // 2)

        if lose_transpose:
            upsampling = [nn.Upsample(scale_factor=att_stride, mode="nearest") if att_stride > 1 else nn.Identity()]
            if fuseconv:
                upsampling = [nn.Conv2d(self.o_proj_inpdim, input_dim, kernel_size=1, stride=1, padding=0)] + upsampling
            self.upsampling = nn.Sequential(*upsampling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()

        xout = self.conv_proj(x)
        xout = self.pwise(xout)

        N, c, h, w = xout.size()
        qkv = xout.reshape(N, self.num_heads, self.num_keys * self.head_dim, h * w)
        qkv = qkv.permute(0, 1, 3, 2)  # [N, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=3)

        values = F.scaled_dot_product_attention(q, k, v)
        o = self.o_proj(values.permute(0, 1, 3, 2).reshape(N, self.o_proj_inpdim, h, w))

        o = self.upsampling(o)
        return o[:N, :C, :H, :W]


class CPUBoneBlock(nn.Module):
    """Attention (context) branch followed by a local conv branch, both with identity residuals."""

    def __init__(
            self,
            in_channels: int,
            expand_ratio: float = 4,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.Hardswish,
            fuseconv: bool = False,
            grouping: int = 1,
            att_stride: int = 1,
            mlpexpans: int = 4,
            smallkernel: bool = False,
            lose_transpose: bool = False,
    ):
        super().__init__()
        att_kernel = 5 if att_stride > 1 else 3

        block = ConvAttention(
            input_dim=in_channels,
            att_stride=att_stride,
            att_kernel=att_kernel,
            head_dim_mul=0.5,
            fuseconv=fuseconv,
            smallkernel=smallkernel,
            lose_transpose=lose_transpose,
        )

        context_module = ResidualBlock(nn.Sequential(nn.GroupNorm(1, in_channels), block), nn.Identity())
        mlp = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, in_channels * mlpexpans, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels * mlpexpans, in_channels, kernel_size=1),
            nn.Dropout(p=0.1),
        )
        context_module = nn.Sequential(context_module, ResidualBlock(mlp, nn.Identity()))

        if fuseconv and in_channels < 256:
            local_module = FusedMBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, False),
                kernel_size=2 if smallkernel else 3,
                groups=grouping,
                norm_layer=(norm_layer, norm_layer),
                act_layer=(act_layer, None),
            )
        else:
            local_module = MBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                grouping=grouping,
                use_bias=(True, True, False),
                kernel_size=2 if smallkernel else 3,
                norm_layer=(None, None, norm_layer),
                act_layer=(act_layer, act_layer, None),
            )

        self.total = nn.Sequential(context_module, ResidualBlock(local_module, nn.Identity()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.total(x)


class ClsHead(nn.Module):
    def __init__(
            self,
            in_channels: int,
            width_list: List[int],
            num_classes: int = 1000,
            dropout: float = 0.0,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.Hardswish,
    ):
        super().__init__()
        self.num_features = width_list[-1]
        self.dropout = dropout

        self.in_conv = ConvLayer(in_channels, width_list[0], 1, norm_layer=norm_layer, act_layer=act_layer)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)
        self.pre_classifier = LinearLayer(
            width_list[0], width_list[1], False, norm_layer=nn.LayerNorm, act_layer=act_layer)
        self.classifier = (
            LinearLayer(width_list[1], num_classes, True, dropout) if num_classes > 0 else nn.Identity()
        )

    def reset(self, num_classes: int):
        if num_classes > 0:
            self.classifier = LinearLayer(self.num_features, num_classes, True, self.dropout)
        else:
            self.classifier = nn.Identity()

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.pre_classifier(x)
        if pre_logits:
            return x
        return self.classifier(x)


class CPUBone(nn.Module):
    def __init__(
            self,
            width_list: List[int],
            depth_list: List[int],
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = "avg",
            head_widths: Tuple[int, int] = (1536, 1600),
            drop_rate: float = 0.0,
            expand_ratio: float = 4,
            norm_layer: LayerType = nn.BatchNorm2d,
            act_layer: LayerType = nn.Hardswish,
            fastit: bool = False,
            huge_model: bool = False,
            bigit: bool = False,
            grouping: int = 1,
            smallk_only_lasts: bool = False,
            lose_transpose: bool = False,
            just_unfused: bool = False,
    ) -> None:
        super().__init__()
        assert global_pool == "avg", "CPUBone only supports average pooling"
        self.num_classes = num_classes
        self.num_features = width_list[-1]
        self.head_hidden_size = head_widths[-1]
        self.grad_checkpointing = False

        self.expand_ratio = expand_ratio
        self.norm_layer = get_norm_layer(norm_layer)
        self.act_layer = get_act_layer(act_layer)
        self.fuseconv = fastit and not just_unfused
        self.fastit = fastit
        self.huge_model = huge_model
        self.bigit = bigit
        self.grouping = grouping
        self.smallk_only_lasts = smallk_only_lasts
        self.lose_transpose = lose_transpose

        self.stem, in_channels = self._build_stem(in_chans, width_list[0], depth_list[0])

        # stages 1-4: early stages use plain conv blocks, later stages add attention
        stages = []
        self.feature_info = []
        for stage_num, (width, depth) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            if stage_num >= 3:
                blocks, in_channels = self._build_attention_stage(in_channels, width, depth, stage_num)
            else:
                blocks, in_channels = self._build_conv_stage(in_channels, width, depth)
            stages.append(nn.Sequential(*blocks))
            self.feature_info.append(
                dict(num_chs=in_channels, reduction=2 ** (stage_num + 1), module=f"stages.{stage_num - 1}"))
        self.stages = nn.Sequential(*stages)

        self.head = ClsHead(
            in_channels=width_list[-1],
            width_list=list(head_widths),
            num_classes=num_classes,
            dropout=drop_rate,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
        )

    def _build_stem(self, in_channels: int, stem_width: int, depth: int) -> Tuple[nn.Sequential, int]:
        """Stem: downsample by 2, then `depth` local blocks at the stem width."""
        blocks = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=stem_width,
                kernel_size=3,
                stride=2,
                norm_layer=self.norm_layer,
                act_layer=self.act_layer,
            )
        ]
        in_channels = stem_width
        for _ in range(depth):
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=in_channels,
                stride=1,
                expand_ratio=4 if self.huge_model else 2,
                fusedmbconv=self.fuseconv,
                grouping=self.grouping,
                norm_layer=self.norm_layer,
                act_layer=self.act_layer,
            )
            blocks.append(ResidualBlock(block, nn.Identity()))
        return nn.Sequential(*blocks), in_channels

    def _build_conv_stage(self, in_channels: int, width: int, depth: int) -> Tuple[List[nn.Module], int]:
        """Stages 1-2: `depth` plain conv blocks, downsampling (stride 2) on the first one."""
        blocks = []
        for i in range(depth):
            stride = 2 if i == 0 else 1
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=width,
                stride=stride,
                expand_ratio=6 if stride == 2 and (self.bigit or self.huge_model) else self.expand_ratio,
                fusedmbconv=self.fuseconv,
                grouping=self.grouping,
                norm_layer=self.norm_layer,
                act_layer=self.act_layer,
            )
            blocks.append(ResidualBlock(block, nn.Identity() if stride == 1 else None))
            in_channels = width
        return blocks, in_channels

    def _build_attention_stage(
            self,
            in_channels: int,
            width: int,
            depth: int,
            stage_num: int,
    ) -> Tuple[List[nn.Module], int]:
        """Stages 3-4: one downsampling conv block, followed by `depth` CPUBoneBlocks (attention + local conv)."""
        downsample = self.build_local_block(
            in_channels=in_channels,
            out_channels=width,
            stride=2,
            expand_ratio=6 if self.bigit or (self.huge_model and stage_num < 4) else self.expand_ratio,
            fusedmbconv=self.fastit,
            grouping=self.grouping,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
        )
        in_channels = width
        blocks = [ResidualBlock(downsample, None)]
        for _ in range(depth):
            blocks.append(
                CPUBoneBlock(
                    in_channels=in_channels,
                    expand_ratio=self.expand_ratio,
                    norm_layer=self.norm_layer,
                    act_layer=self.act_layer,
                    fuseconv=self.fuseconv,
                    grouping=self.grouping,
                    att_stride=2 if stage_num == 3 else 1,
                    mlpexpans=4 if self.fastit else 2,
                    smallkernel=self.smallk_only_lasts,
                    lose_transpose=self.lose_transpose,
                )
            )
        return blocks, in_channels

    @staticmethod
    def build_local_block(
            in_channels: int,
            out_channels: int,
            stride: int,
            expand_ratio: float,
            norm_layer: Type[nn.Module],
            act_layer: Type[nn.Module],
            fusedmbconv: bool = False,
            grouping: int = 1,
            kernel_size: int = 3,
    ) -> nn.Module:
        if fusedmbconv:
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(False, False),
                kernel_size=kernel_size,
                groups=grouping,
                norm_layer=(norm_layer, norm_layer),
                act_layer=(act_layer, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                grouping=grouping,
                use_bias=(False, False, False),
                norm_layer=(None, None, norm_layer),
                act_layer=(act_layer, act_layer, None),
            )
        return block

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)',
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.classifier

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool == "avg", "CPUBone only supports average pooling"
        self.head.reset(num_classes)

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
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates (no-op, CPUBone has no final norm)
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
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
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(stage, x)
            else:
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
    ):
        """Prune layers not required for specified intermediates."""
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        self.stages = self.stages[:max_index + 1]
        if prune_head:
            self.reset_classifier(0)
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    state_dict = state_dict.get("state_dict", state_dict)
    return remap_legacy_state_dict(state_dict)


def _cfg(url: str = "", **kwargs: Any) -> Dict[str, Any]:
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": (7, 7),
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "stem.0.conv",
        "classifier": "head.classifier.linear",
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    "cpubone_nano.untrained": _cfg(),
    "cpubone_t0.untrained": _cfg(),
    "cpubone_s0.untrained": _cfg(),
    "cpubone_s1.untrained": _cfg(),
    "cpubone_b0.untrained": _cfg(),
    "cpubone_b1.untrained": _cfg(),
    "cpubone_b15.untrained": _cfg(),
    "cpubone_b2.untrained": _cfg(),
    "cpubone_b25.untrained": _cfg(),
    "cpubone_b3.untrained": _cfg(),
})


def _create_cpubone(variant: str, pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model = build_model_with_cfg(
        CPUBone,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )
    return model


@register_model
def cpubone_nano(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = dict(width_list=[12, 24, 48, 96, 192], depth_list=[0, 1, 1, 1, 2])
    return _create_cpubone("cpubone_nano", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cpubone_t0(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = dict(width_list=[12, 24, 48, 96, 192], depth_list=[0, 1, 1, 1, 3])
    return _create_cpubone("cpubone_t0", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cpubone_s0(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = dict(width_list=[12, 24, 48, 96, 192], depth_list=[0, 1, 1, 2, 3])
    return _create_cpubone("cpubone_s0", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cpubone_s1(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = dict(width_list=[14, 28, 56, 112, 224], depth_list=[0, 1, 1, 2, 3])
    return _create_cpubone("cpubone_s1", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cpubone_b0(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = dict(width_list=[16, 32, 64, 128, 256], depth_list=[0, 1, 1, 3, 4])
    return _create_cpubone("cpubone_b0", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cpubone_b1(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = dict(width_list=[16, 32, 64, 128, 256], depth_list=[0, 1, 1, 5, 5])
    return _create_cpubone("cpubone_b1", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cpubone_b15(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = dict(width_list=[20, 40, 80, 160, 320], depth_list=[0, 1, 1, 6, 6])
    return _create_cpubone("cpubone_b15", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cpubone_b2(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = dict(width_list=[24, 48, 96, 192, 384], depth_list=[0, 1, 1, 6, 6])
    return _create_cpubone("cpubone_b2", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cpubone_b25(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = dict(width_list=[24, 48, 96, 192, 384], depth_list=[0, 2, 3, 6, 6])
    return _create_cpubone("cpubone_b25", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cpubone_b3(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = dict(width_list=[32, 64, 128, 256, 512], depth_list=[1, 2, 3, 6, 6])
    return _create_cpubone("cpubone_b3", pretrained=pretrained, **dict(model_args, **kwargs))
