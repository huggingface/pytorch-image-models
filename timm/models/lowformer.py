"""LowFormer
LowFormer: Hardware Efficient Design for Convolutional Transformer Backbones (WACV 2025)
- paper: https://arxiv.org/abs/2409.03460
- code: https://github.com/altair199797/LowFormer
@article{Nottebaum2024LowFormerHE,
  title={LowFormer: Hardware Efficient Design for Convolutional Transformer Backbones},
  author={Moritz Nottebaum and Matteo Dunnhofer and Christian Micheloni},
  journal={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2024},
  pages={7008-7018},
}

The LowFormer-E1/E2/E3 edge GPU variants are introduced in the journal extension
Beyond MACs: Hardware Efficient Architecture Design for Vision Backbones (IJCV 2026)
- paper: https://arxiv.org/abs/2603.26551
- code: https://github.com/altair199797/LowFormer
@article{Nottebaum2026BeyondMACs,
  author  = {Nottebaum, Moritz and Dunnhofer, Matteo and Micheloni, Christian},
  title   = {Beyond {MAC}s: Hardware Efficient Architecture Design for Vision Backbones},
  journal = {International Journal of Computer Vision},
  year    = {2026},
  volume  = {134},
  number  = {6},
  pages   = {295},
  doi     = {10.1007/s11263-026-02873-5},
  url     = {https://doi.org/10.1007/s11263-026-02873-5}
}

Modifications by / Copyright 2026 Ryan Hou & Ross Wightman, original copyrights below
"""
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import (
    DropPath,
    GroupNorm1,
    Linear,
    SelectAdaptivePool2d,
    calculate_drop_path_rates,
    get_device_dtype,
    trunc_normal_,
    use_fused_attn,
)
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._features_fx import register_notrace_module
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['LowFormer']


class ResidualBlock(nn.Module):
    def __init__(
            self,
            main: nn.Module,
            shortcut: Optional[nn.Module] = None,
            drop_path: float = 0.,
    ):
        super().__init__()
        self.main = main
        self.shortcut = shortcut
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is None:
            return self.main(x)
        return self.drop_path(self.main(x)) + self.shortcut(x)


class ConvLayer(nn.Module):
    """Conv + optional norm + optional act.

    NOTE deliberately not timm's ConvNormAct: LowFormer needs activation-without-norm
    (the MBConv depthwise convs), which ConvNormAct cannot express — its norm and act are
    combined via get_norm_act_layer, which returns None for norm_layer=None.
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
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=use_bias,
            **dd,
        )
        self.norm = norm_layer(out_channels, **dd) if norm_layer is not None else nn.Identity()
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
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
            expand_groups: int = 1,
            use_bias: bool = False,
            norm_layer: Optional[Type[nn.Module]] = nn.BatchNorm2d,
            act_layer: Optional[Type[nn.Module]] = nn.ReLU6,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        # pointwise expand
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            groups=expand_groups,
            use_bias=use_bias,
            norm_layer=None,
            act_layer=act_layer,
            **dd,
        )
        # depthwise
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            use_bias=use_bias,
            norm_layer=None,
            act_layer=act_layer,
            **dd,
        )
        # pointwise project
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            groups=expand_groups,
            use_bias=False,
            norm_layer=norm_layer,
            act_layer=None,
            **dd,
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
            expand_groups: int = 1,
            use_bias: bool = False,
            norm_layer: Optional[Type[nn.Module]] = nn.BatchNorm2d,
            act_layer: Optional[Type[nn.Module]] = nn.ReLU6,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=expand_groups,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **dd,
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            groups=1,
            use_bias=False,
            norm_layer=norm_layer,
            act_layer=None,
            **dd,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


# kept as an FX leaf: the shape-dependent reshape/crop in forward() traces fine on recent torch,
# but leaf status keeps older-torch compatibility, matching other timm attention modules
@register_notrace_module
class ConvAttention(nn.Module):
    """Low-frequency / conv-projected attention with strided downsample and learned upsample."""

    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            input_dim: int,
            head_dim_mul: float = 0.5,
            att_stride: int = 4,
            att_kernel: int = 7,
            fuse_out_proj: bool = False,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.num_heads = int(max(1, (input_dim * head_dim_mul) // 30))
        self.head_dim = int((input_dim // self.num_heads) * head_dim_mul)
        self.num_keys = 3
        self.scale = self.head_dim ** -0.5
        self.att_stride = att_stride
        self.fused_attn = use_fused_attn()

        total_dim = int(self.head_dim * self.num_heads * self.num_keys)

        self.conv_proj = ConvLayer(
            input_dim,
            input_dim,
            kernel_size=att_kernel,
            stride=att_stride,
            groups=input_dim,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
            **dd,
        )
        self.pwise = nn.Conv2d(input_dim, total_dim, kernel_size=1, stride=1, padding=0, bias=False, **dd)

        self.o_proj_inpdim = self.head_dim * self.num_heads
        # With fuse_out_proj the output projection is folded into the upsampling module below, which
        # then maps o_proj_inpdim -> input_dim instead of being a depthwise / parameter-free upsample.
        if fuse_out_proj:
            self.o_proj = nn.Identity()
            if att_stride == 1:
                self.upsampling = nn.ConvTranspose2d(
                    self.o_proj_inpdim, input_dim, kernel_size=3, stride=1, padding=1, **dd)
            else:
                self.upsampling = nn.ConvTranspose2d(
                    self.o_proj_inpdim, input_dim,
                    kernel_size=att_stride * 2, stride=att_stride, padding=att_stride // 2, **dd)
        else:
            self.o_proj = nn.Conv2d(self.o_proj_inpdim, input_dim, kernel_size=1, stride=1, padding=0, **dd)
            if att_stride == 1:
                self.upsampling = nn.ConvTranspose2d(
                    input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, **dd)
            else:
                self.upsampling = nn.ConvTranspose2d(
                    input_dim, input_dim,
                    kernel_size=att_stride * 2, stride=att_stride, padding=att_stride // 2, groups=input_dim, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        xout = self.conv_proj(x)
        xout = self.pwise(xout)

        N, _, h, w = xout.size()
        qkv = xout.reshape(N, self.num_heads, self.num_keys * self.head_dim, h * w)
        qkv = qkv.permute(0, 1, 3, 2)  # [N, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=3)

        if self.fused_attn:
            values = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            values = attn @ v
        o = self.o_proj(values.permute(0, 1, 3, 2).reshape(N, self.o_proj_inpdim, h, w))

        o = self.upsampling(o)
        # Upsampling can overshoot after same-padding, crop to the input spatial size
        return o[..., :H, :W]


class LowFormerBlock(nn.Module):
    """Attention (context) and MLP branches followed by a local conv branch, all with identity residuals."""

    def __init__(
            self,
            in_channels: int,
            expand_ratio: float = 4,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.Hardswish,
            fused_conv: bool = False,
            expand_groups: int = 1,
            attn: bool = True,
            attn_mlp: bool = True,
            attn_mlp_ratio: int = 4,
            att_stride: int = 1,
            proj_drop: float = 0.,
            drop_path: float = 0.,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        att_kernel = 5 if att_stride > 1 else 3

        if attn:
            attn_module: nn.Module = ConvAttention(
                input_dim=in_channels,
                att_stride=att_stride,
                att_kernel=att_kernel,
                head_dim_mul=0.5,
                fuse_out_proj=fused_conv,
                **dd,
            )
            if attn_mlp:
                attn_module = nn.Sequential(GroupNorm1(in_channels, **dd), attn_module)
            self.attn = ResidualBlock(attn_module, nn.Identity(), drop_path)
        else:
            self.attn = nn.Identity()

        if attn_mlp:
            self.mlp = ResidualBlock(
                nn.Sequential(
                    GroupNorm1(in_channels, **dd),
                    nn.Conv2d(in_channels, in_channels * attn_mlp_ratio, kernel_size=1, **dd),
                    nn.GELU(),
                    nn.Conv2d(in_channels * attn_mlp_ratio, in_channels, kernel_size=1, **dd),
                    nn.Dropout(proj_drop),
                ),
                nn.Identity(),
                drop_path,
            )
        else:
            self.mlp = nn.Identity()

        block_cls = FusedMBConv if fused_conv and in_channels < 256 else MBConv
        local_module = block_cls(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            expand_groups=expand_groups,
            use_bias=True,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **dd,
        )
        self.local = ResidualBlock(local_module, nn.Identity(), drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.mlp(x)
        x = self.local(x)
        return x


class LowFormer(nn.Module):
    def __init__(
            self,
            width_list: List[int],
            depth_list: List[int],
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = "avg",
            head_widths: Tuple[int, int] = (1536, 1600),
            drop_rate: float = 0.0,
            proj_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            expand_ratio: float = 4,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.Hardswish,
            fused_conv: bool = True,
            attn: bool = True,
            attn_mlp: bool = True,
            attn_mlp_ratio: int = 4,
            stem_expand_ratio: float = 2,
            downsample_expand_ratios: Optional[Tuple[float, ...]] = None,
            expand_groups: int = 1,
            device=None,
            dtype=None,
    ):
        """LowFormer backbone with classification head.

        Args:
            width_list: Channel width of the stem and each of the four stages.
            depth_list: Number of blocks in the stem and each of the four stages.
            in_chans: Number of input image channels.
            num_classes: Number of classifier output classes.
            global_pool: Global pooling type, either 'avg' or '' to disable pooling.
            head_widths: Hidden widths of the classification head (in_conv, pre_classifier).
            drop_rate: Classifier dropout rate.
            proj_drop_rate: Dropout rate at the end of the attention-stage (LowFormerBlock) MLPs.
            drop_path_rate: Stochastic depth rate.
            expand_ratio: Default expand ratio of MBConv / FusedMBConv blocks.
            norm_layer: Normalization layer.
            act_layer: Activation layer.
            fused_conv: Use FusedMBConv instead of MBConv in the stem, conv stages, attention
                stage downsamples and block local branches; also folds the attention output
                projection into its upsampling layer.
            attn: Include the attention (ConvAttention) branch in LowFormerBlocks.
            attn_mlp: Include the MLP branch in LowFormerBlocks.
            attn_mlp_ratio: MLP expansion ratio in the attention (LowFormerBlock) stages.
            stem_expand_ratio: Expand ratio of the stem blocks.
            downsample_expand_ratios: Per-stage expand ratios of the four stride-2 downsample
                blocks, None uses `expand_ratio` everywhere.
            expand_groups: Groups of the expand conv in MBConv / FusedMBConv blocks.

        The ablation flags of the original implementation map onto these args as follows:
        `fastit=True` → `fused_conv=True, attn_mlp_ratio=4`; `bigit=True` →
        `downsample_expand_ratios=(6, 6, 6, 6)`; `huge_model=True` → `stem_expand_ratio=4`;
        `grouping` → `expand_groups`; `noattention=True` → `attn=False`; `mlpremoved=True` →
        `attn_mlp=False`.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_rate = drop_rate
        self.feature_info = []
        self.grad_checkpointing = False

        num_stages = len(width_list) - 1
        if downsample_expand_ratios is None:
            downsample_expand_ratios = (expand_ratio,) * num_stages
        assert len(downsample_expand_ratios) == num_stages
        downsample_expand_ratios = tuple(downsample_expand_ratios)


        # stochastic depth: linear ramp of drop rates across all blocks (downsample blocks have no
        # shortcut and ignore theirs)
        dpr = calculate_drop_path_rates(drop_path_rate, sum(depth_list))
        block_cls = FusedMBConv if fused_conv else MBConv

        # stem: stride-2 conv, then `depth_list[0]` local blocks at the stem width
        stem_blocks = [
            ConvLayer(
                in_channels=in_chans,
                out_channels=width_list[0],
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                act_layer=act_layer,
                **dd,
            )
        ]
        in_channels = width_list[0]
        for i in range(depth_list[0]):
            block = block_cls(
                in_channels=in_channels,
                out_channels=in_channels,
                stride=1,
                expand_ratio=stem_expand_ratio,
                expand_groups=expand_groups,
                use_bias=False,
                norm_layer=norm_layer,
                act_layer=act_layer,
                **dd,
            )
            stem_blocks.append(ResidualBlock(block, nn.Identity(), dpr[i]))
        self.stem = nn.Sequential(*stem_blocks)

        # stages 1-4: early stages use plain conv blocks, later stages add attention
        stages = []
        reduction = 2  # stem downsamples by 2
        block_idx = depth_list[0]
        for stage_num, (width, depth) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            stage_dpr = dpr[block_idx:block_idx + depth]
            block_idx += depth
            blocks = []
            if stage_num >= 3:
                downsample = block_cls(
                    in_channels=in_channels,
                    out_channels=width,
                    stride=2,
                    expand_ratio=downsample_expand_ratios[stage_num - 1],
                    expand_groups=expand_groups,
                    use_bias=False,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    **dd,
                )
                blocks.append(ResidualBlock(downsample, None))
                in_channels = width
                for i in range(depth):
                    blocks.append(
                        LowFormerBlock(
                            in_channels=in_channels,
                            expand_ratio=expand_ratio,
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                            fused_conv=fused_conv,
                            expand_groups=expand_groups,
                            attn=attn,
                            attn_mlp=attn_mlp,
                            attn_mlp_ratio=attn_mlp_ratio,
                            att_stride=2 if stage_num == 3 else 1,
                            proj_drop=proj_drop_rate,
                            drop_path=stage_dpr[i],
                            **dd,
                        )
                    )
            elif depth > 0:
                for i in range(depth):
                    stride = 2 if i == 0 else 1
                    block = block_cls(
                        in_channels=in_channels,
                        out_channels=width,
                        stride=stride,
                        expand_ratio=(
                            downsample_expand_ratios[stage_num - 1] if stride == 2 else expand_ratio
                        ),
                        expand_groups=expand_groups,
                        use_bias=False,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        **dd,
                    )
                    blocks.append(ResidualBlock(block, nn.Identity() if stride == 1 else None, stage_dpr[i]))
                    in_channels = width
            # zero-depth stages (e.g. LowFormer-B2) contain no downsampling block
            stages.append(nn.Sequential(*blocks))
            reduction *= 2 if blocks else 1
            self.feature_info.append(
                dict(num_chs=in_channels, reduction=reduction, module=f"stages.{stage_num - 1}"))
        self.stages = nn.Sequential(*stages)

        # head
        self.num_features = width_list[-1]
        self.head_hidden_size = head_widths[-1]
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.in_conv = ConvLayer(
            in_channels=width_list[-1],
            out_channels=head_widths[0],
            kernel_size=1,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **dd,
        )
        self.pre_classifier = nn.Sequential(
            Linear(head_widths[0], head_widths[1], bias=False, **dd),
            nn.LayerNorm(head_widths[1], **dd),
            act_layer(),
        )
        self.classifier = Linear(head_widths[1], num_classes, **dd) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return set()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        matcher = dict(
            stem=r'^stem\.\d+',
            blocks=[
                (r'^stages\.(\d+)' if coarse else r'^stages\.(\d+)\.(\d+)', None),
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.classifier

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        dd = get_device_dtype(self)
        was_training = self.training
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
            self.global_pool.train(was_training)
        self.classifier = Linear(
            self.head_hidden_size, num_classes, **dd,
        ) if num_classes > 0 else nn.Identity()
        self.classifier.train(was_training)

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
            norm: Apply norm layer to compatible intermediates (no-op, LowFormer has no final norm).
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
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        if self.global_pool.is_identity():
            # Keep the pretrained Linear/LayerNorm parameter shapes while applying them channel-wise.
            x = x.permute(0, 2, 3, 1)
        x = self.pre_classifier(x)
        x = F.dropout(x, self.drop_rate, training=self.training) if self.drop_rate else x
        if not pre_logits:
            x = self.classifier(x)
        if self.global_pool.is_identity():
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    if 'stem.0.conv.weight' in state_dict:
        return state_dict  # native timm checkpoint, no remapping needed
    state_dict = state_dict.get('state_dict', state_dict)
    out_dict = {}
    for k, v in state_dict.items():
        # backbone was a submodule → stem / stages are now top-level children
        k = k.replace("backbone.input_stem.", "stem.")
        k = k.replace("backbone.stages.", "stages.")
        # head was OpSequential([ConvLayer, AdaptiveAvgPool2d, LinearLayer, LinearLayer]) → named modules;
        # ClsHeadTorchScript checkpoints use an .opseq module name
        k = k.replace("head.opseq.", "head.")
        k = k.replace("head.op_list.0.", "in_conv.")
        k = k.replace("head.op_list.2.linear.", "pre_classifier.0.")
        k = k.replace("head.op_list.2.norm.", "pre_classifier.1.")
        k = k.replace("head.op_list.3.linear.", "classifier.")
        # input_stem / stages were OpSequential (module list under .op_list) → plain nn.Sequential
        k = k.replace(".op_list.", ".")
        # blocks were OpSequential([context, local]) → named ResidualBlock branches (longest keys first)
        k = k.replace(".total.0.0.", ".attn.")
        k = k.replace(".total.0.1.", ".mlp.")
        k = k.replace(".total.0.", ".attn.")
        k = k.replace(".total.1.", ".local.")
        # conv_proj was nn.Sequential([conv, bn]) → now ConvLayer with .conv / .norm
        k = k.replace(".conv_proj.0.", ".conv_proj.conv.")
        k = k.replace(".conv_proj.1.", ".conv_proj.norm.")
        # pwise was a single-element nn.Sequential → now a plain nn.Conv2d
        k = k.replace(".pwise.0.", ".pwise.")
        out_dict[k] = v
    return out_dict


def _cfg(url: str = "", **kwargs: Any) -> Dict[str, Any]:
    return {
        "url": url, "num_classes": 1000, "input_size": (3, 224, 224), "pool_size": (7, 7),
        "crop_pct": 0.95, "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD,
        "first_conv": "stem.0.conv", "classifier": "classifier",
        'origin_url': 'https://github.com/altair199797/LowFormer', "license": "apache-2.0",
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    "lowformer_b0.in1k": _cfg(
        hf_hub_id='timm/',
    ),
    "lowformer_b1.in1k": _cfg(
        hf_hub_id='timm/',
    ),
    "lowformer_b15.in1k": _cfg(
        hf_hub_id='timm/',
    ),
    "lowformer_b2.untrained": _cfg(),
    "lowformer_b3.in1k": _cfg(
        hf_hub_id='timm/',
    ),
    "lowformer_e1.in1k": _cfg(
        hf_hub_id='timm/',
    ),
    "lowformer_e2.in1k": _cfg(
        hf_hub_id='timm/',
    ),
    "lowformer_e3.in1k": _cfg(
        hf_hub_id='timm/',
    ),
})


def _create_lowformer(variant: str, pretrained: bool = False, **kwargs: Any) -> LowFormer:
    model = build_model_with_cfg(
        LowFormer, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )
    return model


@register_model
def lowformer_b0(pretrained: bool = False, **kwargs: Any) -> LowFormer:
    """Instantiate LowFormer-B0 model variant."""
    model_args = dict(
        width_list=[16, 32, 64, 128, 256], depth_list=[0, 1, 1, 3, 4],
    )
    return _create_lowformer("lowformer_b0", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def lowformer_b1(pretrained: bool = False, **kwargs: Any) -> LowFormer:
    """Instantiate LowFormer-B1 model variant."""
    model_args = dict(
        width_list=[16, 32, 64, 128, 256], depth_list=[0, 1, 1, 5, 5],
        downsample_expand_ratios=(6, 6, 6, 6),
    )
    return _create_lowformer("lowformer_b1", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def lowformer_b15(pretrained: bool = False, **kwargs: Any) -> LowFormer:
    """Instantiate LowFormer-B1.5 model variant."""
    model_args = dict(
        width_list=[20, 40, 80, 160, 320], depth_list=[0, 1, 1, 6, 6], head_widths=(2304, 2560),
        downsample_expand_ratios=(6, 6, 6, 6),
    )
    return _create_lowformer("lowformer_b15", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def lowformer_b2(pretrained: bool = False, **kwargs: Any) -> LowFormer:
    """Instantiate LowFormer-B2 model variant."""
    model_args = dict(
        width_list=[24, 48, 96, 192, 384], depth_list=[0, 1, 1, 6, 6], head_widths=(2304, 2560),
        downsample_expand_ratios=(6, 6, 6, 6),
    )
    return _create_lowformer("lowformer_b2", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def lowformer_b3(pretrained: bool = False, **kwargs: Any) -> LowFormer:
    """Instantiate LowFormer-B3 model variant."""
    model_args = dict(
        width_list=[32, 64, 128, 256, 512], depth_list=[1, 2, 3, 6, 6],
        stem_expand_ratio=4, downsample_expand_ratios=(6, 6, 6, 6),
    )
    return _create_lowformer("lowformer_b3", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def lowformer_e1(pretrained: bool = False, **kwargs: Any) -> LowFormer:
    """Instantiate LowFormer-E1 edge GPU model variant."""
    model_args = dict(
        width_list=[20, 40, 80, 160, 320], depth_list=[0, 1, 1, 4, 4], head_widths=(2304, 2560),
        attn=False, attn_mlp=False, downsample_expand_ratios=(6, 6, 6, 6),
    )
    return _create_lowformer("lowformer_e1", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def lowformer_e2(pretrained: bool = False, **kwargs: Any) -> LowFormer:
    """Instantiate LowFormer-E2 edge GPU model variant."""
    model_args = dict(
        width_list=[32, 64, 128, 256, 512], depth_list=[1, 2, 3, 4, 4],
        attn=False, attn_mlp=False, stem_expand_ratio=4, downsample_expand_ratios=(6, 6, 6, 6),
    )
    return _create_lowformer("lowformer_e2", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def lowformer_e3(pretrained: bool = False, **kwargs: Any) -> LowFormer:
    """Instantiate LowFormer-E3 edge GPU model variant."""
    model_args = dict(
        width_list=[32, 64, 128, 256, 512], depth_list=[1, 2, 3, 6, 6],
        attn_mlp=False, stem_expand_ratio=4, downsample_expand_ratios=(6, 6, 6, 6),
    )
    return _create_lowformer("lowformer_e3", pretrained=pretrained, **dict(model_args, **kwargs))
