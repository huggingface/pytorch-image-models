"""CPUBone

CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities
Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni
Conference on Computer Vision and Pattern Recognition (CVPR) Findings, 2026

Adapted from the original implementation at https://github.com/altair199797/CPUBone.
"""
from typing import Any, Dict, Final, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, GroupNorm1, LayerType, calculate_drop_path_rates, get_act_layer, get_norm_layer, \
    use_fused_attn
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._features_fx import register_notrace_module
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['CPUBone']


_LOCAL_MBCONV_NORM_MODES = {
    # mode: (expand, depthwise, project)
    'proj': (False, False, True),
    'depth_proj': (False, True, True),
    'all': (True, True, True),
}


def _check_local_mbconv_norm(local_mbconv_norm: str) -> None:
    if local_mbconv_norm not in _LOCAL_MBCONV_NORM_MODES:
        raise ValueError(
            f'Invalid local_mbconv_norm={local_mbconv_norm!r}; '
            f'expected one of {tuple(_LOCAL_MBCONV_NORM_MODES)}.'
        )


def _check_global_pool(global_pool: str) -> None:
    assert global_pool in ("", "avg"), "CPUBone only supports average or disabled pooling"


def remap_legacy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap keys from original CPUBone checkpoints to the current model layout."""
    remapped = {}
    for k, v in state_dict.items():
        # conv_proj was nn.Sequential([conv, bn]) → now ConvLayer with .conv / .bn
        k = k.replace(".conv_proj.0.", ".conv_proj.conv.")
        k = k.replace(".conv_proj.1.", ".conv_proj.norm.")
        # pwise was a single-element nn.Sequential → now a plain nn.Conv2d
        k = k.replace(".pwise.0.", ".pwise.")
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
            expand_groups: int = 1,
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
            groups=expand_groups,
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
            expand_groups: int = 1,
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
            groups=expand_groups,
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


# kept as an FX leaf: the shape-dependent reshape/crop in forward() traces fine on recent torch,
# but leaf status keeps older-torch compatibility, matching other timm attention modules
@register_notrace_module
class ConvAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            input_dim: int,
            head_dim_mul: float = 1.0,
            att_stride: int = 4,
            att_kernel: int = 7,
            fuse_out_proj: bool = False,
            small_kernels: bool = False,
            upsample_mode: str = 'transpose',
    ):
        super().__init__()
        self.num_heads = int(max(1, (input_dim * head_dim_mul) // 30))
        self.head_dim = int((input_dim // self.num_heads) * head_dim_mul)
        self.num_keys = 3
        self.scale = self.head_dim ** -0.5
        self.att_stride = att_stride
        self.small_kernels = small_kernels
        self.fused_attn = use_fused_attn()

        total_dim = int(self.head_dim * self.num_heads * self.num_keys)

        self.conv_proj = ConvLayer(
            input_dim,
            input_dim,
            kernel_size=2 if small_kernels else att_kernel,
            stride=att_stride,
            groups=input_dim,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
        )
        self.pwise = nn.Conv2d(input_dim, total_dim, kernel_size=1, stride=1, padding=0, bias=False)

        self.o_proj_inpdim = self.head_dim * self.num_heads
        # With fuse_out_proj the output projection is folded into the upsampling module below, which
        # then maps o_proj_inpdim -> input_dim instead of being a depthwise / parameter-free upsample.
        if fuse_out_proj:
            self.o_proj = nn.Identity()
        else:
            self.o_proj = nn.Conv2d(self.o_proj_inpdim, input_dim, kernel_size=1, stride=1, padding=0)

        if upsample_mode == 'nearest':
            upsampling = [nn.Upsample(scale_factor=att_stride, mode="nearest") if att_stride > 1 else nn.Identity()]
            if fuse_out_proj:
                upsampling = [nn.Conv2d(self.o_proj_inpdim, input_dim, kernel_size=1, stride=1, padding=0)] + upsampling
            self.upsampling = nn.Sequential(*upsampling)
        elif fuse_out_proj:
            if att_stride == 1:
                self.upsampling = nn.ConvTranspose2d(self.o_proj_inpdim, input_dim, kernel_size=3, stride=1, padding=1)
            else:
                self.upsampling = nn.ConvTranspose2d(
                    self.o_proj_inpdim, input_dim,
                    kernel_size=att_stride * 2, stride=att_stride, padding=att_stride // 2)
        else:
            if att_stride == 1:
                self.upsampling = nn.ConvTranspose2d(
                    input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim)
            else:
                self.upsampling = nn.ConvTranspose2d(
                    input_dim, input_dim,
                    kernel_size=att_stride * 2, stride=att_stride, padding=att_stride // 2, groups=input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        # The reduced 2x2 projection rounds odd spatial dimensions down, while the residual branch
        # retains them. Pad only that projection to make the subsequent upsample large enough to crop.
        if self.small_kernels and self.att_stride > 1:
            pad_h = (-H) % self.att_stride
            pad_w = (-W) % self.att_stride
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))

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
        # Upsampling can overshoot after same-padding or the odd-size projection pad above.
        return o[..., :H, :W]


class CPUBoneBlock(nn.Module):
    """Attention (context) branch followed by a local conv branch, both with identity residuals."""

    def __init__(
            self,
            in_channels: int,
            expand_ratio: float = 4,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.Hardswish,
            fused_conv: bool = False,
            expand_groups: int = 1,
            att_stride: int = 1,
            mlp_ratio: int = 4,
            small_kernels: bool = False,
            attn_upsample: str = 'transpose',
            proj_drop: float = 0.1,
            drop_path: float = 0.,
            local_mbconv_norm: str = 'proj',
    ):
        super().__init__()
        _check_local_mbconv_norm(local_mbconv_norm)
        att_kernel = 5 if att_stride > 1 else 3

        block = ConvAttention(
            input_dim=in_channels,
            att_stride=att_stride,
            att_kernel=att_kernel,
            head_dim_mul=0.5,
            fuse_out_proj=fused_conv,
            small_kernels=small_kernels,
            upsample_mode=attn_upsample,
        )

        context_module = ResidualBlock(nn.Sequential(GroupNorm1(in_channels), block), nn.Identity(), drop_path)
        mlp = nn.Sequential(
            GroupNorm1(in_channels),
            nn.Conv2d(in_channels, in_channels * mlp_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels * mlp_ratio, in_channels, kernel_size=1),
            nn.Dropout(p=proj_drop),
        )
        context_module = nn.Sequential(context_module, ResidualBlock(mlp, nn.Identity(), drop_path))

        if fused_conv and in_channels < 256:
            local_module = FusedMBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, False),
                kernel_size=2 if small_kernels else 3,
                expand_groups=expand_groups,
                norm_layer=(norm_layer, norm_layer),
                act_layer=(act_layer, None),
            )
        else:
            norm_mask = _LOCAL_MBCONV_NORM_MODES[local_mbconv_norm]
            local_norms = tuple(norm_layer if enabled else None for enabled in norm_mask)
            # A convolution bias is redundant whenever normalization follows it.
            local_biases = tuple(not enabled for enabled in norm_mask)
            local_module = MBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                expand_groups=expand_groups,
                use_bias=local_biases,
                kernel_size=2 if small_kernels else 3,
                norm_layer=local_norms,
                act_layer=(act_layer, act_layer, None),
            )

        self.total = nn.Sequential(context_module, ResidualBlock(local_module, nn.Identity(), drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.total(x)


class ClsHead(nn.Module):
    def __init__(
            self,
            in_channels: int,
            width_list: List[int],
            num_classes: int = 1000,
            global_pool: str = "avg",
            dropout: float = 0.0,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.Hardswish,
    ):
        super().__init__()
        _check_global_pool(global_pool)
        self.num_features = width_list[-1]
        self.dropout = dropout
        self.pool_type = global_pool

        self.in_conv = ConvLayer(in_channels, width_list[0], 1, norm_layer=norm_layer, act_layer=act_layer)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1) if global_pool else nn.Identity()
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.pre_classifier = LinearLayer(
            width_list[0], width_list[1], False, norm_layer=nn.LayerNorm, act_layer=act_layer)
        self.classifier = (
            LinearLayer(width_list[1], num_classes, True, dropout) if num_classes > 0 else nn.Identity()
        )

    def reset(self, num_classes: int, global_pool: Optional[str] = None):
        if global_pool is not None:
            _check_global_pool(global_pool)
            self.pool_type = global_pool
            self.global_pool = nn.AdaptiveAvgPool2d(output_size=1) if global_pool else nn.Identity()
            self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        if num_classes > 0:
            self.classifier = LinearLayer(self.num_features, num_classes, True, self.dropout)
        else:
            self.classifier = nn.Identity()

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        if not self.pool_type:
            # Keep the pretrained Linear/LayerNorm parameter shapes while applying them channel-wise.
            x = x.permute(0, 2, 3, 1)
        x = self.pre_classifier(x)
        if not pre_logits:
            x = self.classifier(x)
        if not self.pool_type:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


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
            proj_drop_rate: float = 0.1,
            drop_path_rate: float = 0.0,
            expand_ratio: float = 4,
            norm_layer: LayerType = nn.BatchNorm2d,
            act_layer: LayerType = nn.Hardswish,
            fused_conv: bool = False,
            fused_downsample: bool = False,
            attn_mlp_ratio: int = 2,
            stem_expand_ratio: float = 2,
            downsample_expand_ratios: Optional[Tuple[float, ...]] = None,
            expand_groups: int = 1,
            small_kernels: bool = False,
            attn_upsample: str = 'transpose',
            local_mbconv_norm: str = 'proj',
    ) -> None:
        """
        Args:
            width_list: Channel width of the stem and each of the four stages.
            depth_list: Number of blocks in the stem and each of the four stages.
            in_chans: Number of input image channels.
            num_classes: Number of classifier output classes.
            global_pool: Global pooling type, either 'avg' or '' to disable pooling.
            head_widths: Hidden widths of the classification head (in_conv, pre_classifier).
            drop_rate: Classifier dropout rate.
            proj_drop_rate: Dropout rate at the end of the attention-stage (CPUBoneBlock) MLPs, the
                0.1 default matches the original implementation's fixed dropout.
            drop_path_rate: Stochastic depth rate.
            expand_ratio: Default expand ratio of MBConv / FusedMBConv blocks.
            norm_layer: Normalization layer.
            act_layer: Activation layer.
            fused_conv: Use FusedMBConv instead of MBConv in the stem, conv stages, and block local
                branches; also folds the attention output projection into its upsampling layer.
            fused_downsample: Use FusedMBConv for the downsample blocks of the attention stages.
            attn_mlp_ratio: MLP expansion ratio in the attention (CPUBoneBlock) stages.
            stem_expand_ratio: Expand ratio of the stem blocks.
            downsample_expand_ratios: Per-stage expand ratios of the four stride-2 downsample
                blocks, None uses `expand_ratio` everywhere.
            expand_groups: Groups of the expand conv in MBConv / FusedMBConv blocks.
            small_kernels: Use 2x2 kernels in the attention stages (attention conv_proj and the
                local conv branch).
            attn_upsample: Upsampling mode after strided attention, 'transpose' (learned
                ConvTranspose2d) or 'nearest' (parameter-free nearest-neighbor).
            local_mbconv_norm: Normalization placement in unfused CPUBoneBlock local MBConvs;
                one of 'proj', 'depth_proj', or 'all'. Convolution biases are disabled wherever
                normalization is enabled.

        The ablation flags of the original implementation map onto these args as follows:
        `fastit=True` → `fused_conv=True, fused_downsample=True, attn_mlp_ratio=4` (adding
        `just_unfused=True` cancels only `fused_conv`); `bigit=True` →
        `downsample_expand_ratios=(6, 6, 6, 6)`; `huge_model=True` → `stem_expand_ratio=4,
        downsample_expand_ratios=(6, 6, 6, expand_ratio)`; `grouping` → `expand_groups`;
        `smallk_only_lasts` → `small_kernels`; `lose_transpose=True` → `attn_upsample='nearest'`.
        """
        super().__init__()
        _check_global_pool(global_pool)
        assert attn_upsample in ('transpose', 'nearest')
        _check_local_mbconv_norm(local_mbconv_norm)
        num_stages = len(width_list) - 1
        if downsample_expand_ratios is None:
            downsample_expand_ratios = (expand_ratio,) * num_stages
        assert len(downsample_expand_ratios) == num_stages
        self.num_classes = num_classes
        self.num_features = width_list[-1]
        self.head_hidden_size = head_widths[-1]
        self.global_pool = global_pool
        self.grad_checkpointing = False

        self.expand_ratio = expand_ratio
        self.norm_layer = get_norm_layer(norm_layer)
        self.act_layer = get_act_layer(act_layer)
        self.fused_conv = fused_conv
        self.fused_downsample = fused_downsample
        self.attn_mlp_ratio = attn_mlp_ratio
        self.proj_drop_rate = proj_drop_rate
        self.stem_expand_ratio = stem_expand_ratio
        self.downsample_expand_ratios = tuple(downsample_expand_ratios)
        self.expand_groups = expand_groups
        self.small_kernels = small_kernels
        self.attn_upsample = attn_upsample
        self.local_mbconv_norm = local_mbconv_norm

        # stochastic depth: linear ramp of drop rates across all blocks (downsample blocks have no
        # shortcut and ignore theirs)
        dpr = calculate_drop_path_rates(drop_path_rate, sum(depth_list))

        self.stem, in_channels = self._build_stem(in_chans, width_list[0], depth_list[0], dpr[:depth_list[0]])
        block_idx = depth_list[0]

        # stages 1-4: early stages use plain conv blocks, later stages add attention
        stages = []
        self.feature_info = []
        for stage_num, (width, depth) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            stage_dpr = dpr[block_idx:block_idx + depth]
            block_idx += depth
            if stage_num >= 3:
                blocks, in_channels = self._build_attention_stage(in_channels, width, depth, stage_num, stage_dpr)
            else:
                blocks, in_channels = self._build_conv_stage(in_channels, width, depth, stage_num, stage_dpr)
            stages.append(nn.Sequential(*blocks))
            self.feature_info.append(
                dict(num_chs=in_channels, reduction=2 ** (stage_num + 1), module=f"stages.{stage_num - 1}"))
        self.stages = nn.Sequential(*stages)

        self.head = ClsHead(
            in_channels=width_list[-1],
            width_list=list(head_widths),
            num_classes=num_classes,
            global_pool=global_pool,
            dropout=drop_rate,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
        )

    def _build_stem(
            self,
            in_channels: int,
            stem_width: int,
            depth: int,
            dpr: List[float],
    ) -> Tuple[nn.Sequential, int]:
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
        for i in range(depth):
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=in_channels,
                stride=1,
                expand_ratio=self.stem_expand_ratio,
                fusedmbconv=self.fused_conv,
                expand_groups=self.expand_groups,
                norm_layer=self.norm_layer,
                act_layer=self.act_layer,
            )
            blocks.append(ResidualBlock(block, nn.Identity(), dpr[i]))
        return nn.Sequential(*blocks), in_channels

    def _build_conv_stage(
            self,
            in_channels: int,
            width: int,
            depth: int,
            stage_num: int,
            dpr: List[float],
    ) -> Tuple[List[nn.Module], int]:
        """Stages 1-2: `depth` plain conv blocks, downsampling (stride 2) on the first one."""
        blocks = []
        for i in range(depth):
            stride = 2 if i == 0 else 1
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=width,
                stride=stride,
                expand_ratio=self.downsample_expand_ratios[stage_num - 1] if stride == 2 else self.expand_ratio,
                fusedmbconv=self.fused_conv,
                expand_groups=self.expand_groups,
                norm_layer=self.norm_layer,
                act_layer=self.act_layer,
            )
            blocks.append(ResidualBlock(block, nn.Identity() if stride == 1 else None, dpr[i]))
            in_channels = width
        return blocks, in_channels

    def _build_attention_stage(
            self,
            in_channels: int,
            width: int,
            depth: int,
            stage_num: int,
            dpr: List[float],
    ) -> Tuple[List[nn.Module], int]:
        """Stages 3-4: one downsampling conv block, followed by `depth` CPUBoneBlocks (attention + local conv)."""
        downsample = self.build_local_block(
            in_channels=in_channels,
            out_channels=width,
            stride=2,
            expand_ratio=self.downsample_expand_ratios[stage_num - 1],
            fusedmbconv=self.fused_downsample,
            expand_groups=self.expand_groups,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
        )
        in_channels = width
        blocks = [ResidualBlock(downsample, None)]
        for i in range(depth):
            blocks.append(
                CPUBoneBlock(
                    in_channels=in_channels,
                    expand_ratio=self.expand_ratio,
                    norm_layer=self.norm_layer,
                    act_layer=self.act_layer,
                    fused_conv=self.fused_conv,
                    expand_groups=self.expand_groups,
                    att_stride=2 if stage_num == 3 else 1,
                    mlp_ratio=self.attn_mlp_ratio,
                    small_kernels=self.small_kernels,
                    attn_upsample=self.attn_upsample,
                    proj_drop=self.proj_drop_rate,
                    drop_path=dpr[i],
                    local_mbconv_norm=self.local_mbconv_norm,
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
            expand_groups: int = 1,
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
                expand_groups=expand_groups,
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
                expand_groups=expand_groups,
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
        if global_pool is not None:
            _check_global_pool(global_pool)
            self.global_pool = global_pool
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

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
    "cpubone_nano.in1k": _cfg(hf_hub_id="Kaeruu/CPUBone", hf_hub_filename="cpubone_nano.safetensors"),
    "cpubone_b0.in1k": _cfg(hf_hub_id="Kaeruu/CPUBone", hf_hub_filename="cpubone_b0.safetensors"),
    "cpubone_b1.in1k": _cfg(hf_hub_id="Kaeruu/CPUBone", hf_hub_filename="cpubone_b1.safetensors"),
    "cpubone_b1_dwnorm.untrained": _cfg(),
    "cpubone_b1_allnorm.untrained": _cfg(),
    "cpubone_b2.in1k": _cfg(hf_hub_id="Kaeruu/CPUBone", hf_hub_filename="cpubone_b2.safetensors"),
    "cpubone_b2_dwnorm.untrained": _cfg(),
    "cpubone_b2_allnorm.untrained": _cfg(),
    "cpubone_b3.in1k": _cfg(hf_hub_id="Kaeruu/CPUBone", hf_hub_filename="cpubone_b3.safetensors"),
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


# Variant flags below were decoded from the weight shapes of the released checkpoints (see the
# CPUBone.__init__ docstring for the mapping from the original ablation flags). All released
# checkpoints use the fastit/grouping=2/smallk_only_lasts/lose_transpose combination, i.e.
# fused_conv, fused_downsample, attn_mlp_ratio=4, expand_groups=2, small_kernels, 'nearest'.

@register_model
def cpubone_nano(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = dict(
        width_list=[12, 24, 48, 96, 192],
        depth_list=[0, 1, 1, 1, 2],
        fused_conv=True,
        fused_downsample=True,
        attn_mlp_ratio=4,
        expand_groups=2,
        small_kernels=True,
        attn_upsample="nearest",
    )
    return _create_cpubone("cpubone_nano", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cpubone_b0(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = dict(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[0, 1, 1, 3, 4],
        fused_conv=True,
        fused_downsample=True,
        attn_mlp_ratio=4,
        expand_groups=2,
        small_kernels=True,
        attn_upsample="nearest",
    )
    return _create_cpubone("cpubone_b0", pretrained=pretrained, **dict(model_args, **kwargs))


def _cpubone_b1_args(local_mbconv_norm: str = 'proj') -> Dict[str, Any]:
    return dict(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[0, 1, 1, 5, 5],
        fused_conv=True,
        fused_downsample=True,
        attn_mlp_ratio=4,
        downsample_expand_ratios=(6, 6, 6, 6),
        expand_groups=2,
        small_kernels=True,
        attn_upsample="nearest",
        local_mbconv_norm=local_mbconv_norm,
    )


@register_model
def cpubone_b1(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = _cpubone_b1_args()
    return _create_cpubone("cpubone_b1", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cpubone_b1_dwnorm(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = _cpubone_b1_args(local_mbconv_norm='depth_proj')
    return _create_cpubone("cpubone_b1_dwnorm", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cpubone_b1_allnorm(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = _cpubone_b1_args(local_mbconv_norm='all')
    return _create_cpubone("cpubone_b1_allnorm", pretrained=pretrained, **dict(model_args, **kwargs))


def _cpubone_b2_args(local_mbconv_norm: str = 'proj') -> Dict[str, Any]:
    return dict(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[0, 1, 1, 6, 6],
        head_widths=(2304, 2560),
        fused_conv=True,
        fused_downsample=True,
        attn_mlp_ratio=4,
        downsample_expand_ratios=(6, 6, 6, 6),
        expand_groups=2,
        small_kernels=True,
        attn_upsample="nearest",
        local_mbconv_norm=local_mbconv_norm,
    )


@register_model
def cpubone_b2(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = _cpubone_b2_args()
    return _create_cpubone("cpubone_b2", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cpubone_b2_dwnorm(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = _cpubone_b2_args(local_mbconv_norm='depth_proj')
    return _create_cpubone("cpubone_b2_dwnorm", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cpubone_b2_allnorm(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = _cpubone_b2_args(local_mbconv_norm='all')
    return _create_cpubone("cpubone_b2_allnorm", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cpubone_b3(pretrained: bool = False, **kwargs: Any) -> CPUBone:
    model_args = dict(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 3, 6, 6],
        fused_conv=True,
        fused_downsample=True,
        attn_mlp_ratio=4,
        stem_expand_ratio=4,
        downsample_expand_ratios=(6, 6, 6, 6),
        expand_groups=2,
        small_kernels=True,
        attn_upsample="nearest",
    )
    return _create_cpubone("cpubone_b3", pretrained=pretrained, **dict(model_args, **kwargs))
