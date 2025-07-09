from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import (
    SelectAdaptivePool2d,
    Linear,
    LayerType,
    RmsNorm2d,
    ConvNormAct,
    create_conv2d,
    get_norm_layer,
    get_norm_act_layer,
    to_2tuple,
)
from ._builder import build_model_with_cfg
from ._efficientnet_blocks import SqueezeExcite, UniversalInvertedResidual
from ._efficientnet_builder import (
    BlockArgs,
    EfficientNetBuilder,
    decode_arch_def,
    efficientnet_init_weights,
    round_channels,
)
from ._features import feature_take_indices
from ._features_fx import register_notrace_module
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['MobileNetV5', 'MobileNetV5Encoder']

_GELU = partial(nn.GELU, approximate='tanh')


@register_notrace_module
class MobileNetV5MultiScaleFusionAdapter(nn.Module):
  """Multi-layer fusion token adapter.

  Args:
    in_chs: List of input channel counts for each feature scale.
    out_chs: The number of output channels.
    output_resolution: The output resolution.
    expansion_ratio: The FFN expansion ratio.
    interpolation_mode: The upsampling interpolation mode.
    layer_scale_init_value: The initial value of the layer scale, no layer scale if None.
  """

  def __init__(
        self,
        in_chs: Union[int, List[int]],
        out_chs: int,
        output_resolution: int,
        expansion_ratio: float = 2.0,
        interpolation_mode: str = "nearest",
        layer_scale_init_value: Optional[float] = None,
        noskip: bool = True,
        act_layer: Optional[LayerType] = None,
        norm_layer: Optional[LayerType] = None,
  ):
    super().__init__()
    self.in_channels = sum(in_chs) if isinstance(in_chs, Sequence) else in_chs
    self.out_channels = out_chs
    self.output_resolution = to_2tuple(output_resolution)
    self.expansion_ratio = expansion_ratio
    self.interpolation_mode = interpolation_mode
    self.layer_scale_init_value = layer_scale_init_value
    self.noskip = noskip

    act_layer = act_layer or _GELU
    norm_layer = norm_layer or RmsNorm2d
    self.ffn = UniversalInvertedResidual(
        in_chs=self.in_channels,
        out_chs=self.out_channels,
        dw_kernel_size_mid=0,
        exp_ratio=self.expansion_ratio,
        act_layer=act_layer,
        norm_layer=norm_layer,
        noskip=self.noskip,
        layer_scale_init_value=self.layer_scale_init_value,
    )

    self.norm = norm_layer(self.out_channels)

  def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
    # Inputs list of [B, C, H, W] tensors
    high_resolution = inputs[0].shape[-2:]  # Assuming the first input is the highest resolution.
    resized_inputs = []
    for _, img in enumerate(inputs):
        feat_size = img.shape[-2:]
        if feat_size[0] < high_resolution[0] or feat_size[1] < high_resolution[1]:
            img = F.interpolate(img, size=high_resolution, mode=self.interpolation_mode)
        resized_inputs.append(img)

    channel_cat_imgs = torch.cat(resized_inputs, dim=1)  # Cat on channel dim, must equal self.in_channels
    img = self.ffn(channel_cat_imgs)

    if high_resolution[0] != self.output_resolution[0] or high_resolution[1] != self.output_resolution[1]:
        # Interpolate / pool to target output_resolution if highest feature resolution differs
        if (
            high_resolution[0] % self.output_resolution[0] != 0 or
            high_resolution[1] % self.output_resolution[1] != 0
        ):
            img = F.interpolate(img, size=self.output_resolution, mode="bilinear")
        else:
            h_strides = high_resolution[0] // self.output_resolution[0]
            w_strides = high_resolution[1] // self.output_resolution[1]
            img = F.avg_pool2d(
                img,
                kernel_size=(h_strides, w_strides),
                stride=(h_strides, w_strides),
            )

    img = self.norm(img)

    return img


class MobileNetV5(nn.Module):
    """ MobiletNet-V5
    """

    def __init__(
            self,
            block_args: BlockArgs,
            num_classes: int = 1000,
            in_chans: int = 3,
            stem_size: int = 16,
            stem_bias: bool = False,
            fix_stem: bool = False,
            num_features: int = 2048,
            pad_type: str = '',
            use_msfa: bool = True,
            msfa_indices: List[int] = (-3, -2, -1),
            msfa_output_resolution: int = 16,
            act_layer: Optional[LayerType] = None,
            norm_layer: Optional[LayerType] = None,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[LayerType] = None,
            se_from_exp: bool = True,
            round_chs_fn: Callable = round_channels,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale_init_value: Optional[float] = None,
            global_pool: str = 'avg',
    ):
        """
        Args:
            block_args: Arguments for blocks of the network.
            num_classes: Number of classes for classification head.
            in_chans: Number of input image channels.
            stem_size: Number of output channels of the initial stem convolution.
            fix_stem: If True, don't scale stem by round_chs_fn.
            num_features: Number of output channels of the conv head layer.
            head_bias: If True, add a learnable bias to the conv head layer.
            pad_type: Type of padding to use for convolution layers.
            act_layer: Type of activation layer.
            norm_layer: Type of normalization layer.
            aa_layer: Type of anti-aliasing layer.
            se_layer: Type of Squeeze-and-Excite layer.
            se_from_exp: If True, calculate SE channel reduction from expanded mid channels.
            round_chs_fn: Callable to round number of filters based on depth multiplier.
            drop_rate: Dropout rate.
            drop_path_rate: Stochastic depth rate.
            layer_scale_init_value: Enable layer scale on compatible blocks if not None.
            global_pool: Type of pooling to use for global pooling features of the FC head.
        """
        super().__init__()
        act_layer = act_layer or _GELU
        norm_layer = get_norm_layer(norm_layer) or RmsNorm2d
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.msfa_indices = msfa_indices
        self.msfa_output_resolution = msfa_output_resolution

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = ConvNormAct(
            in_chans,
            stem_size,
            kernel_size=3,
            stride=2,
            padding=pad_type,
            bias=stem_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=32,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            se_from_exp=se_from_exp,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        self.stage_ends = [f['stage'] for f in self.feature_info]
        self.num_features = builder.in_chs  # features of last stage, output of forward_features()

        # Neck (aggregation) + Head + Pooling
        if use_msfa:
            self.num_features = self.head_hidden_size = num_features # output of msfa is output of forward_features()
            # Map msfa indices to feature info and calculate sum of feature channels
            self.msfa_indices = feature_take_indices(len(self.feature_info), self.msfa_indices)[0]
            self.msfa_in_chs = sum([self.feature_info[mi]['num_chs'] for mi in self.msfa_indices])

            self.msfa = MobileNetV5MultiScaleFusionAdapter(
                in_chs=self.msfa_in_chs,
                out_chs=num_features,
                output_resolution=self.msfa_output_resolution,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.conv_head = None
            self.norm_head = None
        else:
            self.num_features = builder.in_chs  # features of last stage, output of forward_features()
            self.head_hidden_size = num_features
            self.msfa = None
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            num_pooled_chs = self.num_features * self.global_pool.feat_mult()
            # mobilenet-v4 style post-pooling PW conv is followed by a norm+act layer
            self.conv_head = create_conv2d(num_pooled_chs, self.head_hidden_size, 1, padding=pad_type)
            self.norm_head = norm_act_layer(self.head_hidden_size)

        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.head_hidden_size, num_classes) if num_classes > 0 else nn.Identity()

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1]
        layers.extend(self.blocks)
        layers.append(self.global_pool)
        if self.conv_head is not None:
            layers.append(self.conv_head)
        if self.norm_head is not None:
            layers.append(self.norm_head)
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False):
        return dict(
            stem=r'^conv_stem|bn1',
            blocks=r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)'
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.classifier

    def reset_classifier(self, num_classes: int, global_pool: str = 'avg'):
        self.num_classes = num_classes
        # NOTE: cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.head_hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            extra_blocks: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
            extra_blocks: Include outputs of all blocks and head conv in output, does not align with feature_info
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        if stop_early:
            assert intermediates_only, 'Must use intermediates_only for early stopping.'
        intermediates = []
        if extra_blocks:
            take_indices, max_index = feature_take_indices(len(self.blocks) + 1, indices)
        else:
            take_indices, max_index = feature_take_indices(len(self.stage_ends), indices)
            take_indices = [self.stage_ends[i] for i in take_indices]
            max_index = self.stage_ends[max_index]

        # FIXME MFSA and forward_intermediates overlap, they both take indices from specific features
        # When a user wants to grab specific feature maps for a downstream task AND have the msfa output
        # what should we do? Accumulate two intermediates? One for msfa and one for take_indices?

        # forward pass
        feat_idx = 0  # stem is index 0
        x = self.conv_stem(x)
        if feat_idx in take_indices:
            intermediates.append(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index]
        for blk in blocks:
            feat_idx += 1
            x = blk(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        # FIXME see note above
        # self.msfa(msfa_intermediatse)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
            extra_blocks: bool = False,
    ):
        """ Prune layers not required for specified intermediates.
        """
        if extra_blocks:
            take_indices, max_index = feature_take_indices(len(self.blocks) + 1, indices)
        else:
            take_indices, max_index = feature_take_indices(len(self.stage_ends), indices)
            max_index = self.stage_ends[max_index]
        self.blocks = self.blocks[:max_index]  # truncate blocks w/ stem as idx 0
        if max_index < len(self.blocks):
            self.conv_head = None
            self.norm_head = None
        if prune_head:
            self.conv_head = None
            self.norm_head = None
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.msfa is not None:
            # When MSFA aggregation layer is present, we gather intermediates as is forward_intermediates
            feat_idx = 0  # offset by one from blocks index due to stem feature
            intermediates = []
            x = self.conv_stem(x)
            if feat_idx in self.msfa_indices:
                intermediates.append(x)
            for blk in self.blocks:
                feat_idx += 1
                # FIXME fix grad checkpointing
                x = blk(x)
                if feat_idx in self.msfa_indices:
                    intermediates.append(x)
            x = self.msfa(intermediates)
        else:
            x = self.conv_stem(x)
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(self.blocks, x, flatten=True)
            else:
                x = self.blocks(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        if self.conv_head is not None:
            x = self.conv_head(x)
        if self.norm_head is not None:
            x = self.norm_head(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        if pre_logits:
            return x
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class MobileNetV5Encoder(nn.Module):
    """MobileNetV5 Vision Encoder"""

    def __init__(
            self,
            block_args: BlockArgs,
            in_chans: int = 3,
            stem_size: int = 64,
            stem_bias: bool = True,
            fix_stem: bool = False,
            pad_type: str = '',
            msfa_indices: Sequence[int] = (-2, -1),
            msfa_output_resolution: int = 16,
            act_layer: Optional[LayerType] = None,
            norm_layer: Optional[LayerType] = None,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[LayerType] = None,
            se_from_exp: bool = True,
            round_chs_fn: Callable = round_channels,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        act_layer = act_layer or _GELU
        norm_layer = get_norm_layer(norm_layer) or RmsNorm2d
        se_layer = se_layer or SqueezeExcite
        self.num_classes = 0    # Exists to satisfy ._hub module APIs.
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = ConvNormAct(
            in_chans,
            stem_size,
            kernel_size=3,
            stride=2,
            padding=pad_type,
            bias=stem_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        builder = EfficientNetBuilder(
            output_stride=32,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            se_from_exp=se_from_exp,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        self.stage_ends = [f['stage'] for f in self.feature_info]

        self.num_features = self.head_hidden_size = 2048    # output of msfa is output of forward_features()
        # Map msfa indices to feature info and calculate sum of feature channels
        self.msfa_indices = feature_take_indices(len(self.feature_info), msfa_indices)[0]
        self.msfa_in_chs = sum([self.feature_info[mi]['num_chs'] for mi in self.msfa_indices])
        self.msfa_output_resolution = msfa_output_resolution

        self.msfa = MobileNetV5MultiScaleFusionAdapter(
            in_chs=self.msfa_in_chs,
            out_chs=self.num_features,
            output_resolution=self.msfa_output_resolution,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        efficientnet_init_weights(self)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            extra_blocks: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: (Unused) Applies norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
            extra_blocks: Include outputs of all blocks and head conv in output, does not align with feature_info
        Returns:

        """
        del norm

        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        if stop_early:
            assert intermediates_only, 'Must use intermediates_only for early stopping.'

        # MobileNet v5's MultiScaleFusionAdapter takes intermediates from specific feature indicies and uses them in
        # its computation. These MSFA indices are not guaranteed to be captured by the `indices` parameter passed to
        # this function, so we accumulate two sets of indices, one that aligns with the `indices` parameter and one
        # that is required by the MSFA block.
        intermediates = []
        msfa_intermediates = []

        if extra_blocks:
            take_indices, max_index = feature_take_indices(len(self.blocks) + 1, indices)
        else:
            take_indices, max_index = feature_take_indices(len(self.stage_ends), indices)
            take_indices = [self.stage_ends[i] for i in take_indices]
            max_index = self.stage_ends[max_index]

        # forward pass
        feat_idx = 0    # stem is index 0
        x = self.conv_stem(x)
        if feat_idx in take_indices:
            intermediates.append(x)
        if feat_idx in self.msfa_indices:
            msfa_intermediates.append(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index]

        for blk in blocks:
            feat_idx += 1
            x = blk(x)
            if feat_idx in take_indices:
                intermediates.append(x)
            if feat_idx in self.msfa_indices:
                msfa_intermediates.append(x)

        if intermediates_only:
            return intermediates

        return self.msfa(msfa_intermediates), intermediates

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat_idx = 0    # offset by one from blocks index due to stem feature
        intermediates = []

        x = self.conv_stem(x)
        if feat_idx in self.msfa_indices:
            intermediates.append(x)

        for blk in self.blocks:
            feat_idx += 1
            # FIXME fix grad checkpointing
            x = blk(x)
            if feat_idx in self.msfa_indices:
                intermediates.append(x)

        return self.msfa(intermediates)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("MobileNetV5Encoder does not support classification use cases.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def _create_mnv5_encoder(variant: str, pretrained: bool = False, **kwargs) -> MobileNetV5Encoder:
    out_indices = kwargs.pop('out_indices', (0, 1, 2, 3, 4))
    feature_cfg = dict(out_indices=out_indices, feature_cls='getter')
    kwargs_filter = (
        'num_classes',
        'num_features',
        'head_conv',
        'head_bias',
        'head_norm',
        'global_pool',
    )
    model = build_model_with_cfg(
        MobileNetV5Encoder,
        variant,
        pretrained,
        pretrained_strict=False,
        feature_cfg=feature_cfg,
        kwargs_filter=kwargs_filter,
        **kwargs,
    )
    return model


def _create_mnv5(variant: str, pretrained: bool = False, **kwargs) -> MobileNetV5Encoder:
    out_indices = kwargs.pop('out_indices', (0, 1, 2, 3, 4))
    feature_cfg = dict(out_indices=out_indices, feature_cls='getter')
    model = build_model_with_cfg(
        MobileNetV5,
        variant,
        pretrained,
        pretrained_strict=False,
        feature_cfg=feature_cfg,
        **kwargs,
    )
    return model


def _gen_mobilenet_v5(
        variant: str,
        channel_multiplier: float = 1.0,
        group_size=None,
        pretrained: bool = False,
        encoder: bool = False,
        **kwargs,
) -> MobileNetV5Encoder:
    if 'mobilenetv5_base' in variant:
        arch_def: list[list[str]] = [
            # Stage 0: 128x128 in
            [
                'er_r1_k3_s2_e4_c128',
                'er_r1_k3_s1_e4_c128',
                'er_r1_k3_s1_e4_c128',
            ],
            # Stage 1: 256x256 in
            [
                'uir_r1_a3_k5_s2_e6_c256',
                'uir_r1_a5_k0_s1_e4_c256',
                'uir_r1_a3_k0_s1_e4_c256',
                'uir_r1_a5_k0_s1_e4_c256',
                'uir_r1_a3_k0_s1_e4_c256',
            ],
            # Stage 2: 640x640 in
            [
                "uir_r1_a5_k5_s2_e6_c512",
                "uir_r1_a5_k0_s1_e4_c512",
                "uir_r1_a5_k0_s1_e4_c512",
                "uir_r1_a0_k0_s1_e1_c512",
                'mqa_r1_k3_h8_s2_d64_c512',
                "uir_r1_a0_k0_s1_e2_c512",
                'mqa_r1_k3_h8_s2_d64_c512',
                "uir_r1_a0_k0_s1_e2_c512",
                'mqa_r1_k3_h8_s2_d64_c512',
                "uir_r1_a0_k0_s1_e2_c512",
                'mqa_r1_k3_h8_s2_d64_c512',
                "uir_r1_a0_k0_s1_e2_c512",
                'mqa_r1_k3_h8_s2_d64_c512',
                "uir_r1_a0_k0_s1_e2_c512",
                'mqa_r1_k3_h8_s2_d64_c512',
                "uir_r1_a0_k0_s1_e2_c512",
            ],
            # Stage 3: 1280x1280 in
            [
                "uir_r1_a5_k5_s2_e6_c1024",
                'mqa_r1_k3_h16_s1_d64_c1024',
                "uir_r1_a0_k0_s1_e2_c1024",
                'mqa_r1_k3_h16_s1_d64_c1024',
                "uir_r1_a0_k0_s1_e2_c1024",
                'mqa_r1_k3_h16_s1_d64_c1024',
                "uir_r1_a0_k0_s1_e2_c1024",
                'mqa_r1_k3_h16_s1_d64_c1024',
                "uir_r1_a0_k0_s1_e2_c1024",
                'mqa_r1_k3_h16_s1_d64_c1024',
                "uir_r1_a0_k0_s1_e2_c1024",
                'mqa_r1_k3_h16_s1_d64_c1024',
                "uir_r1_a0_k0_s1_e2_c1024",
                'mqa_r1_k3_h16_s1_d64_c1024',
                "uir_r1_a0_k0_s1_e2_c1024",
            ],
        ]
    else:
        arch_def: list[list[str]] = [
            # Stage 0: 128x128 in
            [
                'er_r1_k3_s2_e4_c128',
                'er_r1_k3_s1_e4_c128',
                'er_r1_k3_s1_e4_c128',
            ],
            # Stage 1: 256x256 in
            [
                'uir_r1_a3_k5_s2_e6_c256',
                'uir_r1_a5_k0_s1_e4_c256',
                'uir_r1_a3_k0_s1_e4_c256',
                'uir_r1_a5_k0_s1_e4_c256',
                'uir_r1_a3_k0_s1_e4_c256',
            ],
            # Stage 2: 640x640 in
            [
                "uir_r1_a5_k5_s2_e6_c640",
                "uir_r1_a5_k0_s1_e4_c640",
                "uir_r1_a5_k0_s1_e4_c640",
                "uir_r1_a5_k0_s1_e4_c640",
                "uir_r1_a5_k0_s1_e4_c640",
                "uir_r1_a5_k0_s1_e4_c640",
                "uir_r1_a5_k0_s1_e4_c640",
                "uir_r1_a5_k0_s1_e4_c640",
                "uir_r1_a0_k0_s1_e1_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
            ],
            # Stage 3: 1280x1280 in
            [
                "uir_r1_a5_k5_s2_e6_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
            ],
        ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, group_size=group_size),
        stem_size=64,
        fix_stem=channel_multiplier < 1.0,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=RmsNorm2d,
        act_layer=_GELU,
        layer_scale_init_value=1e-5,
    )
    model_kwargs = dict(model_kwargs, **kwargs)
    if encoder:
        model = _create_mnv5_encoder(variant, pretrained, **model_kwargs)
    else:
        model = _create_mnv5(variant, pretrained, **model_kwargs)
    return model


def _cfg(url: str = '', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 256, 256), 'pool_size': (16, 16),
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'conv_stem.conv', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # encoder-only configs
    'mobilenetv5_300m_enc': _cfg(
        #hf_hub_id='timm/',
        mean=(0., 0., 0.), std=(1., 1., 1.),
        input_size=(3, 768, 768),
        num_classes=0),

    # WIP classification configs for testing
    'mobilenetv5_300m': _cfg(
        # hf_hub_id='timm/',
        mean=(0., 0., 0.), std=(1., 1., 1.),
        input_size=(3, 768, 768),
        num_classes=0),
    'mobilenetv5_base.untrained': _cfg(
        # hf_hub_id='timm/',
        num_classes=1000)
})


@register_model
def mobilenetv5_300m_enc(pretrained: bool = False, **kwargs) -> MobileNetV5Encoder:
    """MobileNet V5 Vision Encoder"""
    pad_type = kwargs.pop('pad_type', 'same')
    model = _gen_mobilenet_v5(
        'mobilenetv5_300m_enc',
        pretrained=pretrained,
        encoder=True,
        pad_type=pad_type,
        **kwargs,
    )
    return model


@register_model
def mobilenetv5_300m(pretrained: bool = False, **kwargs) -> MobileNetV5:
    model = _gen_mobilenet_v5('mobilenetv5_300m', pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv5_base(pretrained: bool = False, **kwargs) -> MobileNetV5:
    model = _gen_mobilenet_v5('mobilenetv5_base', pretrained=pretrained, **kwargs)
    return model
