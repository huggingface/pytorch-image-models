""" MobileNet V3

A PyTorch impl of MobileNet-V3, compatible with TF weights from official impl.

Paper: Searching for MobileNetV3 - https://arxiv.org/abs/1905.02244

Hacked together by / Copyright 2019, Ross Wightman
"""
from functools import partial
from typing import Any, Dict, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import SelectAdaptivePool2d, Linear, LayerType, PadType, create_conv2d, get_norm_act_layer
from ._builder import build_model_with_cfg, pretrained_cfg_for_features
from ._efficientnet_blocks import SqueezeExcite
from ._efficientnet_builder import BlockArgs, EfficientNetBuilder, decode_arch_def, efficientnet_init_weights, \
    round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from ._features import FeatureInfo, FeatureHooks, feature_take_indices
from ._manipulate import checkpoint_seq, checkpoint
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['MobileNetV3', 'MobileNetV3Features']


class MobileNetV3(nn.Module):
    """MobileNetV3.

    Based on my EfficientNet implementation and building blocks, this model utilizes the MobileNet-v3 specific
    'efficient head', where global pooling is done before the head convolution without a final batch-norm
    layer before the classifier.

    Paper: `Searching for MobileNetV3` - https://arxiv.org/abs/1905.02244

    Other architectures utilizing MobileNet-V3 efficient head that are supported by this impl include:
      * HardCoRe-NAS - https://arxiv.org/abs/2102.11646 (defn in hardcorenas.py uses this class)
      * FBNet-V3 - https://arxiv.org/abs/2006.02049
      * LCNet - https://arxiv.org/abs/2109.15099
      * MobileNet-V4 - https://arxiv.org/abs/2404.10518
    """

    def __init__(
            self,
            block_args: BlockArgs,
            num_classes: int = 1000,
            in_chans: int = 3,
            stem_size: int = 16,
            fix_stem: bool = False,
            num_features: int = 1280,
            head_bias: bool = True,
            head_norm: bool = False,
            pad_type: str = '',
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
        """Initialize MobileNetV3.

        Args:
            block_args: Arguments for blocks of the network.
            num_classes: Number of classes for classification head.
            in_chans: Number of input image channels.
            stem_size: Number of output channels of the initial stem convolution.
            fix_stem: If True, don't scale stem by round_chs_fn.
            num_features: Number of output channels of the conv head layer.
            head_bias: If True, add a learnable bias to the conv head layer.
            head_norm: If True, add normalization to the head layer.
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
        super(MobileNetV3, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_act_layer(stem_size, inplace=True)

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
        self.head_hidden_size = num_features  # features of conv_head, pre_logits output

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        num_pooled_chs = self.num_features * self.global_pool.feat_mult()
        if head_norm:
            # mobilenet-v4 post-pooling PW conv is followed by a norm+act layer
            self.conv_head = create_conv2d(num_pooled_chs, self.head_hidden_size, 1, padding=pad_type)  # never bias
            self.norm_head = norm_act_layer(self.head_hidden_size)
            self.act2 = nn.Identity()
        else:
            # mobilenet-v3 and others only have an activation after final PW conv
            self.conv_head = create_conv2d(num_pooled_chs, self.head_hidden_size, 1, padding=pad_type, bias=head_bias)
            self.norm_head = nn.Identity()
            self.act2 = act_layer(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.head_hidden_size, num_classes) if num_classes > 0 else nn.Identity()

        efficientnet_init_weights(self)

    def as_sequential(self) -> nn.Sequential:
        """Convert model to sequential form.

        Returns:
            Sequential module containing all layers.
        """
        layers = [self.conv_stem, self.bn1]
        layers.extend(self.blocks)
        layers.extend([self.global_pool, self.conv_head, self.norm_head, self.act2])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        """Group parameters for optimization."""
        return dict(
            stem=r'^conv_stem|bn1',
            blocks=r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)'
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing."""
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """Get the classifier head."""
        return self.classifier

    def reset_classifier(self, num_classes: int, global_pool: str = 'avg') -> None:
        """Reset the classifier head.

        Args:
            num_classes: Number of classes for new classifier.
            global_pool: Global pooling type.
        """
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

        # forward pass
        feat_idx = 0  # stem is index 0
        x = self.conv_stem(x)
        x = self.bn1(x)
        if feat_idx in take_indices:
            intermediates.append(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index]
        for feat_idx, blk in enumerate(blocks, start=1):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(blk, x)
            else:
                x = blk(x)
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
            extra_blocks: bool = False,
    ) -> List[int]:
        """Prune layers not required for specified intermediates.

        Args:
            indices: Indices of intermediate layers to keep.
            prune_norm: Whether to prune normalization layer.
            prune_head: Whether to prune the classifier head.
            extra_blocks: Include outputs of all blocks.

        Returns:
            List of indices that were kept.
        """
        if extra_blocks:
            take_indices, max_index = feature_take_indices(len(self.blocks) + 1, indices)
        else:
            take_indices, max_index = feature_take_indices(len(self.stage_ends), indices)
            max_index = self.stage_ends[max_index]
        self.blocks = self.blocks[:max_index]  # truncate blocks w/ stem as idx 0
        if max_index < len(self.blocks):
            self.conv_head = nn.Identity()
            self.norm_head = nn.Identity()
        if prune_head:
            self.conv_head = nn.Identity()
            self.norm_head = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction layers.

        Args:
            x: Input tensor.

        Returns:
            Feature tensor.
        """
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass through classifier head.

        Args:
            x: Input features.
            pre_logits: Return features before final linear layer.

        Returns:
            Classification logits or features.
        """
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.norm_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        if pre_logits:
            return x
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output logits.
        """
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class MobileNetV3Features(nn.Module):
    """MobileNetV3 Feature Extractor.

    A work-in-progress feature extraction module for MobileNet-V3 to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(
            self,
            block_args: BlockArgs,
            out_indices: Tuple[int, ...] = (0, 1, 2, 3, 4),
            feature_location: str = 'bottleneck',
            in_chans: int = 3,
            stem_size: int = 16,
            fix_stem: bool = False,
            output_stride: int = 32,
            pad_type: PadType = '',
            round_chs_fn: Callable = round_channels,
            se_from_exp: bool = True,
            act_layer: Optional[LayerType] = None,
            norm_layer: Optional[LayerType] = None,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[LayerType] = None,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale_init_value: Optional[float] = None,
    ):
        """Initialize MobileNetV3Features.

        Args:
            block_args: Arguments for blocks of the network.
            out_indices: Output from stages at indices.
            feature_location: Location of feature before/after each block, must be in ['bottleneck', 'expansion'].
            in_chans: Number of input image channels.
            stem_size: Number of output channels of the initial stem convolution.
            fix_stem: If True, don't scale stem by round_chs_fn.
            output_stride: Output stride of the network.
            pad_type: Type of padding to use for convolution layers.
            round_chs_fn: Callable to round number of filters based on depth multiplier.
            se_from_exp: If True, calculate SE channel reduction from expanded mid channels.
            act_layer: Type of activation layer.
            norm_layer: Type of normalization layer.
            aa_layer: Type of anti-aliasing layer.
            se_layer: Type of Squeeze-and-Excite layer.
            drop_rate: Dropout rate.
            drop_path_rate: Stochastic depth rate.
            layer_scale_init_value: Enable layer scale on compatible blocks if not None.
        """
        super(MobileNetV3Features, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            se_from_exp=se_from_exp,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            feature_location=feature_location,
        )
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = FeatureInfo(builder.features, out_indices)
        self._stage_out_idx = {f['stage']: f['index'] for f in self.feature_info.get_dicts()}

        efficientnet_init_weights(self)

        # Register feature extraction hooks with FeatureHooks helper
        self.feature_hooks = None
        if feature_location != 'bottleneck':
            hooks = self.feature_info.get_dicts(keys=('module', 'hook_type'))
            self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing."""
        self.grad_checkpointing = enable

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through feature extraction.

        Args:
            x: Input tensor.

        Returns:
            List of feature tensors.
        """
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.feature_hooks is None:
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)  # add stem out
            for i, b in enumerate(self.blocks):
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(b, x)
                else:
                    x = b(x)
                if i + 1 in self._stage_out_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            out = self.feature_hooks.get_output(x.device)
            return list(out.values())


def _create_mnv3(variant: str, pretrained: bool = False, **kwargs) -> MobileNetV3:
    """Create a MobileNetV3 model.

    Args:
        variant: Model variant name.
        pretrained: Load pretrained weights.
        **kwargs: Additional model arguments.

    Returns:
        MobileNetV3 model instance.
    """
    features_mode = ''
    model_cls = MobileNetV3
    kwargs_filter = None
    if kwargs.pop('features_only', False):
        if 'feature_cfg' in kwargs or 'feature_cls' in kwargs:
            features_mode = 'cfg'
        else:
            kwargs_filter = ('num_classes', 'num_features', 'head_conv', 'head_bias', 'head_norm', 'global_pool')
            model_cls = MobileNetV3Features
            features_mode = 'cls'

    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        features_only=features_mode == 'cfg',
        pretrained_strict=features_mode != 'cls',
        kwargs_filter=kwargs_filter,
        **kwargs,
    )
    if features_mode == 'cls':
        model.default_cfg = pretrained_cfg_for_features(model.default_cfg)
    return model


def _gen_mobilenet_v3_rw(
        variant: str, channel_multiplier: float = 1.0, pretrained: bool = False, **kwargs
) -> MobileNetV3:
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
        variant: Model variant name.
        channel_multiplier: Multiplier to number of channels per layer.
        pretrained: Load pretrained weights.
        **kwargs: Additional model arguments.

    Returns:
        MobileNetV3 model instance.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_nre_noskip'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c960'],  # hard-swish
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        head_bias=False,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_layer=partial(SqueezeExcite, gate_layer='hard_sigmoid'),
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_mobilenet_v3(
        variant: str,
        channel_multiplier: float = 1.0,
        depth_multiplier: float = 1.0,
        group_size: Optional[int] = None,
        pretrained: bool = False,
        **kwargs
) -> MobileNetV3:
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
        variant: Model variant name.
        channel_multiplier: Multiplier to number of channels per layer.
        depth_multiplier: Depth multiplier for model scaling.
        group_size: Group size for grouped convolutions.
        pretrained: Load pretrained weights.
        **kwargs: Additional model arguments.

    Returns:
        MobileNetV3 model instance.
    """
    if 'small' in variant:
        num_features = 1024
        if 'minimal' in variant:
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16'],
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24', 'ir_r1_k3_s1_e3.67_c24'],
                # stage 2, 28x28 in
                ['ir_r1_k3_s2_e4_c40', 'ir_r2_k3_s1_e6_c40'],
                # stage 3, 14x14 in
                ['ir_r2_k3_s1_e3_c48'],
                # stage 4, 14x14in
                ['ir_r3_k3_s2_e6_c96'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],
            ]
        else:
            act_layer = resolve_act_layer(kwargs, 'hard_swish')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16_se0.25_nre'],  # relu
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24_nre', 'ir_r1_k3_s1_e3.67_c24_nre'],  # relu
                # stage 2, 28x28 in
                ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r2_k5_s1_e6_c40_se0.25'],  # hard-swish
                # stage 3, 14x14 in
                ['ir_r2_k5_s1_e3_c48_se0.25'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r3_k5_s2_e6_c96_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],  # hard-swish
            ]
    else:
        num_features = 1280
        if 'minimal' in variant:
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16'],
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24', 'ir_r1_k3_s1_e3_c24'],
                # stage 2, 56x56 in
                ['ir_r3_k3_s2_e3_c40'],
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112'],
                # stage 5, 14x14in
                ['ir_r3_k3_s2_e6_c160'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],
            ]
        else:
            act_layer = resolve_act_layer(kwargs, 'hard_swish')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16_nre'],  # relu
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
                # stage 2, 56x56 in
                ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
                # stage 5, 14x14in
                ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],  # hard-swish
            ]
    se_layer = partial(SqueezeExcite, gate_layer='hard_sigmoid', force_act_layer=nn.ReLU, rd_round_fn=round_channels)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier=depth_multiplier, group_size=group_size),
        num_features=num_features,
        stem_size=16,
        fix_stem=channel_multiplier < 0.75,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        se_layer=se_layer,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_fbnetv3(variant: str, channel_multiplier: float = 1.0, pretrained: bool = False, **kwargs) -> MobileNetV3:
    """FBNetV3 model generator.

    Paper: `FBNetV3: Joint Architecture-Recipe Search using Predictor Pretraining`
        - https://arxiv.org/abs/2006.02049
    FIXME untested, this is a preliminary impl of some FBNet-V3 variants.

    Args:
        variant: Model variant name.
        channel_multiplier: Channel width multiplier.
        pretrained: Load pretrained weights.
        **kwargs: Additional model arguments.

    Returns:
        MobileNetV3 model instance.
    """
    vl = variant.split('_')[-1]
    if vl in ('a', 'b'):
        stem_size = 16
        arch_def = [
            ['ds_r2_k3_s1_e1_c16'],
            ['ir_r1_k5_s2_e4_c24', 'ir_r3_k5_s1_e2_c24'],
            ['ir_r1_k5_s2_e5_c40_se0.25', 'ir_r4_k5_s1_e3_c40_se0.25'],
            ['ir_r1_k5_s2_e5_c72', 'ir_r4_k3_s1_e3_c72'],
            ['ir_r1_k3_s1_e5_c120_se0.25', 'ir_r5_k5_s1_e3_c120_se0.25'],
            ['ir_r1_k3_s2_e6_c184_se0.25', 'ir_r5_k5_s1_e4_c184_se0.25', 'ir_r1_k5_s1_e6_c224_se0.25'],
            ['cn_r1_k1_s1_c1344'],
        ]
    elif vl == 'd':
        stem_size = 24
        arch_def = [
            ['ds_r2_k3_s1_e1_c16'],
            ['ir_r1_k3_s2_e5_c24', 'ir_r5_k3_s1_e2_c24'],
            ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r4_k3_s1_e3_c40_se0.25'],
            ['ir_r1_k3_s2_e5_c72', 'ir_r4_k3_s1_e3_c72'],
            ['ir_r1_k3_s1_e5_c128_se0.25', 'ir_r6_k5_s1_e3_c128_se0.25'],
            ['ir_r1_k3_s2_e6_c208_se0.25', 'ir_r5_k5_s1_e5_c208_se0.25', 'ir_r1_k5_s1_e6_c240_se0.25'],
            ['cn_r1_k1_s1_c1440'],
        ]
    elif vl == 'g':
        stem_size = 32
        arch_def = [
            ['ds_r3_k3_s1_e1_c24'],
            ['ir_r1_k5_s2_e4_c40', 'ir_r4_k5_s1_e2_c40'],
            ['ir_r1_k5_s2_e4_c56_se0.25', 'ir_r4_k5_s1_e3_c56_se0.25'],
            ['ir_r1_k5_s2_e5_c104', 'ir_r4_k3_s1_e3_c104'],
            ['ir_r1_k3_s1_e5_c160_se0.25', 'ir_r8_k5_s1_e3_c160_se0.25'],
            ['ir_r1_k3_s2_e6_c264_se0.25', 'ir_r6_k5_s1_e5_c264_se0.25', 'ir_r2_k5_s1_e6_c288_se0.25'],
            ['cn_r1_k1_s1_c1728'],
        ]
    else:
        raise NotImplemented
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, round_limit=0.95)
    se_layer = partial(SqueezeExcite, gate_layer='hard_sigmoid', rd_round_fn=round_chs_fn)
    act_layer = resolve_act_layer(kwargs, 'hard_swish')
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=1984,
        head_bias=False,
        stem_size=stem_size,
        round_chs_fn=round_chs_fn,
        se_from_exp=False,
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        se_layer=se_layer,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_lcnet(variant: str, channel_multiplier: float = 1.0, pretrained: bool = False, **kwargs) -> MobileNetV3:
    """LCNet model generator.

    Essentially a MobileNet-V3 crossed with a MobileNet-V1

    Paper: `PP-LCNet: A Lightweight CPU Convolutional Neural Network` - https://arxiv.org/abs/2109.15099

    Args:
        variant: Model variant name.
        channel_multiplier: Multiplier to number of channels per layer.
        pretrained: Load pretrained weights.
        **kwargs: Additional model arguments.

    Returns:
        MobileNetV3 model instance.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['dsa_r1_k3_s1_c32'],
        # stage 1, 112x112 in
        ['dsa_r2_k3_s2_c64'],
        # stage 2, 56x56 in
        ['dsa_r2_k3_s2_c128'],
        # stage 3, 28x28 in
        ['dsa_r1_k3_s2_c256', 'dsa_r1_k5_s1_c256'],
        # stage 4, 14x14in
        ['dsa_r4_k5_s1_c256'],
        # stage 5, 14x14in
        ['dsa_r2_k5_s2_c512_se0.25'],
        # 7x7
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=16,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_layer=partial(SqueezeExcite, gate_layer='hard_sigmoid', force_act_layer=nn.ReLU),
        num_features=1280,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_mobilenet_v4(
        variant: str,
        channel_multiplier: float = 1.0,
        group_size: Optional[int] = None,
        pretrained: bool = False,
        **kwargs,
) -> MobileNetV3:
    """Creates a MobileNet-V4 model.

    Paper: https://arxiv.org/abs/2404.10518

    Args:
        variant: Model variant name.
        channel_multiplier: Multiplier to number of channels per layer.
        group_size: Group size for grouped convolutions.
        pretrained: Load pretrained weights.
        **kwargs: Additional model arguments.

    Returns:
        MobileNetV3 model instance.
    """
    num_features = 1280
    if 'hybrid' in variant:
        layer_scale_init_value = 1e-5
        if 'medium' in variant:
            stem_size = 32
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                [
                    'er_r1_k3_s2_e4_c48'  # FusedIB (EdgeResidual)
                ],
                # stage 1, 56x56 in
                [
                    'uir_r1_a3_k5_s2_e4_c80',  # ExtraDW
                    'uir_r1_a3_k3_s1_e2_c80',  # ExtraDW
                ],
                # stage 2, 28x28 in
                [
                    'uir_r1_a3_k5_s2_e6_c160',  # ExtraDW
                    'uir_r1_a0_k0_s1_e2_c160',  # FFN
                    'uir_r1_a3_k3_s1_e4_c160',  # ExtraDW
                    'uir_r1_a3_k5_s1_e4_c160',  # ExtraDW
                    'mqa_r1_k3_h4_s1_v2_d64_c160',  # MQA w/ KV downsample
                    'uir_r1_a3_k3_s1_e4_c160',  # ExtraDW
                    'mqa_r1_k3_h4_s1_v2_d64_c160',  # MQA w/ KV downsample
                    'uir_r1_a3_k0_s1_e4_c160',  # ConvNeXt
                    'mqa_r1_k3_h4_s1_v2_d64_c160',  # MQA w/ KV downsample
                    'uir_r1_a3_k3_s1_e4_c160',  # ExtraDW
                    'mqa_r1_k3_h4_s1_v2_d64_c160',  # MQA w/ KV downsample
                    'uir_r1_a3_k0_s1_e4_c160',  # ConvNeXt
                ],
                # stage 3, 14x14in
                [
                    'uir_r1_a5_k5_s2_e6_c256',  # ExtraDW
                    'uir_r1_a5_k5_s1_e4_c256',  # ExtraDW
                    'uir_r2_a3_k5_s1_e4_c256',  # ExtraDW
                    'uir_r1_a0_k0_s1_e2_c256',  # FFN
                    'uir_r1_a3_k5_s1_e2_c256',  # ExtraDW
                    'uir_r1_a0_k0_s1_e2_c256',  # FFN
                    'uir_r1_a0_k0_s1_e4_c256',  # FFN
                    'mqa_r1_k3_h4_s1_d64_c256',  # MQA
                    'uir_r1_a3_k0_s1_e4_c256',  # ConvNeXt
                    'mqa_r1_k3_h4_s1_d64_c256',  # MQA
                    'uir_r1_a5_k5_s1_e4_c256',  # ExtraDW
                    'mqa_r1_k3_h4_s1_d64_c256',  # MQA
                    'uir_r1_a5_k0_s1_e4_c256',  # ConvNeXt
                    'mqa_r1_k3_h4_s1_d64_c256', # MQA
                    'uir_r1_a5_k0_s1_e4_c256',  # ConvNeXt
                ],
                # stage 4, 7x7 in
                [
                    'cn_r1_k1_s1_c960' # Conv
                ],
            ]
        elif 'large' in variant:
            stem_size = 24
            act_layer = resolve_act_layer(kwargs, 'gelu')
            arch_def = [
                # stage 0, 112x112 in
                [
                    'er_r1_k3_s2_e4_c48',  # FusedIB (EdgeResidual)
                ],
                # stage 1, 56x56 in
                [
                    'uir_r1_a3_k5_s2_e4_c96',  # ExtraDW
                    'uir_r1_a3_k3_s1_e4_c96',  # ExtraDW
                ],
                # stage 2, 28x28 in
                [
                    'uir_r1_a3_k5_s2_e4_c192',  # ExtraDW
                    'uir_r3_a3_k3_s1_e4_c192',  # ExtraDW
                    'uir_r1_a3_k5_s1_e4_c192',  # ExtraDW
                    'uir_r2_a5_k3_s1_e4_c192',  # ExtraDW
                    'mqa_r1_k3_h8_s1_v2_d48_c192',  # MQA w/ KV downsample
                    'uir_r1_a5_k3_s1_e4_c192',  # ExtraDW
                    'mqa_r1_k3_h8_s1_v2_d48_c192',  # MQA w/ KV downsample
                    'uir_r1_a5_k3_s1_e4_c192',  # ExtraDW
                    'mqa_r1_k3_h8_s1_v2_d48_c192',  # MQA w/ KV downsample
                    'uir_r1_a5_k3_s1_e4_c192',  # ExtraDW
                    'mqa_r1_k3_h8_s1_v2_d48_c192',  # MQA w/ KV downsample
                    'uir_r1_a3_k0_s1_e4_c192',  # ConvNeXt
                ],
                # stage 3, 14x14in
                [
                    'uir_r4_a5_k5_s2_e4_c512',  # ExtraDW
                    'uir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                    'uir_r1_a5_k3_s1_e4_c512',  # ExtraDW
                    'uir_r2_a5_k0_s1_e4_c512',  # ConvNeXt
                    'uir_r1_a5_k3_s1_e4_c512',  # ExtraDW
                    'uir_r1_a5_k5_s1_e4_c512',  # ExtraDW
                    'mqa_r1_k3_h8_s1_d64_c512',  # MQA
                    'uir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                    'mqa_r1_k3_h8_s1_d64_c512',  # MQA
                    'uir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                    'mqa_r1_k3_h8_s1_d64_c512',  # MQA
                    'uir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                    'mqa_r1_k3_h8_s1_d64_c512',  # MQA
                    'uir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                ],
                # stage 4, 7x7 in
                [
                    'cn_r1_k1_s1_c960',  # Conv
                ],
            ]
        else:
            assert False, f'Unknown variant {variant}.'
    else:
        layer_scale_init_value = None
        if 'small' in variant:
            stem_size = 32
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                [
                    'cn_r1_k3_s2_e1_c32',  # Conv
                    'cn_r1_k1_s1_e1_c32',  # Conv
                ],
                # stage 1, 56x56 in
                [
                    'cn_r1_k3_s2_e1_c96',  # Conv
                    'cn_r1_k1_s1_e1_c64',  # Conv
                ],
                # stage 2, 28x28 in
                [
                    'uir_r1_a5_k5_s2_e3_c96',  # ExtraDW
                    'uir_r4_a0_k3_s1_e2_c96',  # IR
                    'uir_r1_a3_k0_s1_e4_c96',  # ConvNeXt
                ],
                # stage 3, 14x14 in
                [
                    'uir_r1_a3_k3_s2_e6_c128',  # ExtraDW
                    'uir_r1_a5_k5_s1_e4_c128',  # ExtraDW
                    'uir_r1_a0_k5_s1_e4_c128',  # IR
                    'uir_r1_a0_k5_s1_e3_c128',  # IR
                    'uir_r2_a0_k3_s1_e4_c128',  # IR
                ],
                # stage 4, 7x7 in
                [
                    'cn_r1_k1_s1_c960',  # Conv
                ],
            ]
        elif 'medium' in variant:
            stem_size = 32
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                [
                    'er_r1_k3_s2_e4_c48',  # FusedIB (EdgeResidual)
                ],
                # stage 1, 56x56 in
                [
                    'uir_r1_a3_k5_s2_e4_c80',  # ExtraDW
                    'uir_r1_a3_k3_s1_e2_c80',  # ExtraDW
                ],
                # stage 2, 28x28 in
                [
                    'uir_r1_a3_k5_s2_e6_c160',  # ExtraDW
                    'uir_r2_a3_k3_s1_e4_c160',  # ExtraDW
                    'uir_r1_a3_k5_s1_e4_c160',  # ExtraDW
                    'uir_r1_a3_k3_s1_e4_c160',  # ExtraDW
                    'uir_r1_a3_k0_s1_e4_c160',  # ConvNeXt
                    'uir_r1_a0_k0_s1_e2_c160',  # ExtraDW
                    'uir_r1_a3_k0_s1_e4_c160',  # ConvNeXt
                ],
                # stage 3, 14x14in
                [
                    'uir_r1_a5_k5_s2_e6_c256',  # ExtraDW
                    'uir_r1_a5_k5_s1_e4_c256',  # ExtraDW
                    'uir_r2_a3_k5_s1_e4_c256',  # ExtraDW
                    'uir_r1_a0_k0_s1_e4_c256',  # FFN
                    'uir_r1_a3_k0_s1_e4_c256',  # ConvNeXt
                    'uir_r1_a3_k5_s1_e2_c256',  # ExtraDW
                    'uir_r1_a5_k5_s1_e4_c256',  # ExtraDW
                    'uir_r2_a0_k0_s1_e4_c256',  # FFN
                    'uir_r1_a5_k0_s1_e2_c256',  # ConvNeXt
                ],
                # stage 4, 7x7 in
                [
                    'cn_r1_k1_s1_c960',  # Conv
                ],
            ]
        elif 'large' in variant:
            stem_size = 24
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                [
                    'er_r1_k3_s2_e4_c48',  # FusedIB (EdgeResidual)
                ],
                # stage 1, 56x56 in
                [
                    'uir_r1_a3_k5_s2_e4_c96',  # ExtraDW
                    'uir_r1_a3_k3_s1_e4_c96',  # ExtraDW
                ],
                # stage 2, 28x28 in
                [
                    'uir_r1_a3_k5_s2_e4_c192',  # ExtraDW
                    'uir_r3_a3_k3_s1_e4_c192',  # ExtraDW
                    'uir_r1_a3_k5_s1_e4_c192',  # ExtraDW
                    'uir_r5_a5_k3_s1_e4_c192',  # ExtraDW
                    'uir_r1_a3_k0_s1_e4_c192',  # ConvNeXt
                ],
                # stage 3, 14x14in
                [
                    'uir_r4_a5_k5_s2_e4_c512',  # ExtraDW
                    'uir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                    'uir_r1_a5_k3_s1_e4_c512',  # ExtraDW
                    'uir_r2_a5_k0_s1_e4_c512',  # ConvNeXt
                    'uir_r1_a5_k3_s1_e4_c512',  # ExtraDW
                    'uir_r1_a5_k5_s1_e4_c512',  # ExtraDW
                    'uir_r3_a5_k0_s1_e4_c512',  # ConvNeXt

                ],
                # stage 4, 7x7 in
                [
                    'cn_r1_k1_s1_c960',  # Conv
                ],
            ]
        else:
            assert False, f'Unknown variant {variant}.'

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, group_size=group_size),
        head_bias=False,
        head_norm=True,
        num_features=num_features,
        stem_size=stem_size,
        fix_stem=channel_multiplier < 1.0,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        layer_scale_init_value=layer_scale_init_value,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    """Create default configuration dictionary.

    Args:
        url: Model weight URL.
        **kwargs: Additional configuration options.

    Returns:
        Configuration dictionary.
    """
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'mobilenetv3_large_075.untrained': _cfg(url=''),
    'mobilenetv3_large_100.ra_in1k': _cfg(
        interpolation='bicubic',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth',
        hf_hub_id='timm/'),
    'mobilenetv3_large_100.ra4_e3600_r224_in1k': _cfg(
        hf_hub_id='timm/',
        interpolation='bicubic', mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        crop_pct=0.95, test_input_size=(3, 256, 256), test_crop_pct=1.0),
    'mobilenetv3_large_100.miil_in21k_ft_in1k': _cfg(
        interpolation='bilinear', mean=(0., 0., 0.), std=(1., 1., 1.),
        origin_url='https://github.com/Alibaba-MIIL/ImageNet21K',
        paper_ids='arXiv:2104.10972v4',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mobilenetv3_large_100_1k_miil_78_0-66471c13.pth',
        hf_hub_id='timm/'),
    'mobilenetv3_large_100.miil_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mobilenetv3_large_100_in21k_miil-d71cc17b.pth',
        hf_hub_id='timm/',
        origin_url='https://github.com/Alibaba-MIIL/ImageNet21K',
        paper_ids='arXiv:2104.10972v4',
        interpolation='bilinear', mean=(0., 0., 0.), std=(1., 1., 1.), num_classes=11221),
    'mobilenetv3_large_150d.ra4_e3600_r256_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 256, 256), crop_pct=0.95, pool_size=(8, 8), test_input_size=(3, 320, 320), test_crop_pct=1.0),

    'mobilenetv3_small_050.lamb_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_050_lambc-4b7bbe87.pth',
        hf_hub_id='timm/',
        interpolation='bicubic'),
    'mobilenetv3_small_075.lamb_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_075_lambc-384766db.pth',
        hf_hub_id='timm/',
        interpolation='bicubic'),
    'mobilenetv3_small_100.lamb_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_100_lamb-266a294c.pth',
        hf_hub_id='timm/',
        interpolation='bicubic'),

    'mobilenetv3_rw.rmsp_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_100-35495452.pth',
        hf_hub_id='timm/',
        interpolation='bicubic'),

    'tf_mobilenetv3_large_075.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_075-150ee8b0.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_100.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_100-427764d5.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_minimal_100.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_minimal_100-8596ae28.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_075.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_075-da427f52.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_100.in1k': _cfg(
        url= 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_100-37f49e2b.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_minimal_100.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_minimal_100-922a7843.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),

    'fbnetv3_b.ra2_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_b_224-ead5d2a1.pth',
        hf_hub_id='timm/',
        test_input_size=(3, 256, 256), crop_pct=0.95),
    'fbnetv3_d.ra2_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_d_224-c98bce42.pth',
        hf_hub_id='timm/',
        test_input_size=(3, 256, 256), crop_pct=0.95),
    'fbnetv3_g.ra2_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_g_240-0b1df83b.pth',
        hf_hub_id='timm/',
        input_size=(3, 240, 240), test_input_size=(3, 288, 288), crop_pct=0.95, pool_size=(8, 8)),

    "lcnet_035.untrained": _cfg(),
    "lcnet_050.ra2_in1k": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_050-f447553b.pth',
        hf_hub_id='timm/',
        interpolation='bicubic',
    ),
    "lcnet_075.ra2_in1k": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_075-318cad2c.pth',
        hf_hub_id='timm/',
        interpolation='bicubic',
    ),
    "lcnet_100.ra2_in1k": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_100-a929038c.pth',
        hf_hub_id='timm/',
        interpolation='bicubic',
    ),
    "lcnet_150.untrained": _cfg(),

    'mobilenetv4_conv_small_035.untrained': _cfg(
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        test_input_size=(3, 256, 256), test_crop_pct=0.95, interpolation='bicubic'),
    'mobilenetv4_conv_small_050.e3000_r224_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        test_input_size=(3, 256, 256), test_crop_pct=0.95, interpolation='bicubic'),
    'mobilenetv4_conv_small.e2400_r224_in1k': _cfg(
        hf_hub_id='timm/',
        test_input_size=(3, 256, 256), test_crop_pct=0.95, interpolation='bicubic'),
    'mobilenetv4_conv_small.e1200_r224_in1k': _cfg(
        hf_hub_id='timm/',
        test_input_size=(3, 256, 256), test_crop_pct=0.95, interpolation='bicubic'),
    'mobilenetv4_conv_small.e3600_r256_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95,
        test_input_size=(3, 320, 320), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_medium.e500_r256_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.95, test_input_size=(3, 320, 320), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_medium.e500_r224_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 256, 256), test_crop_pct=1.0, interpolation='bicubic'),

    'mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, interpolation='bicubic'),
    'mobilenetv4_conv_medium.e180_r384_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_medium.e180_ad_r384_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_medium.e250_r384_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=1.0, interpolation='bicubic'),

    'mobilenetv4_conv_large.e600_r384_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, test_input_size=(3, 448, 448), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_large.e500_r256_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.95, test_input_size=(3, 320, 320), test_crop_pct=1.0, interpolation='bicubic'),

    'mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.95, test_input_size=(3, 320, 320), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_hybrid_medium.ix_e550_r256_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.95, test_input_size=(3, 320, 320), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_hybrid_medium.ix_e550_r384_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, test_input_size=(3, 448, 448), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_hybrid_medium.e500_r224_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 256, 256), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_hybrid_medium.e200_r256_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
        input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.95, test_input_size=(3, 320, 320), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_hybrid_large.ix_e600_r384_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, test_input_size=(3, 448, 448), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_hybrid_large.e600_r384_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, test_input_size=(3, 448, 448), test_crop_pct=1.0, interpolation='bicubic'),

    # experimental
    'mobilenetv4_conv_aa_medium.untrained': _cfg(
        # hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95, interpolation='bicubic'),
    'mobilenetv4_conv_blur_medium.e500_r224_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 256, 256), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 448, 448), pool_size=(14, 14),
        crop_pct=0.95, test_input_size=(3, 544, 544), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, test_input_size=(3, 480, 480), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_aa_large.e600_r384_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, test_input_size=(3, 480, 480), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_aa_large.e230_r384_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, test_input_size=(3, 448, 448), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_hybrid_medium_075.untrained': _cfg(
        # hf_hub_id='timm/',
        crop_pct=0.95, interpolation='bicubic'),
    'mobilenetv4_hybrid_large_075.untrained': _cfg(
        # hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95, interpolation='bicubic'),
})


@register_model
def mobilenetv3_large_075(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_large_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_large_100(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_large_100', 1.0, pretrained=pretrained, **kwargs)
    return model

@register_model
def mobilenetv3_large_150d(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_large_150d', 1.5, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model

@register_model
def mobilenetv3_small_050(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_050', 0.50, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_small_075(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_small_100(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_rw(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V3 """
    kwargs.setdefault('bn_eps', BN_EPS_TF_DEFAULT)
    model = _gen_mobilenet_v3_rw('mobilenetv3_rw', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_large_075(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V3 """
    kwargs.setdefault('bn_eps', BN_EPS_TF_DEFAULT)
    kwargs.setdefault('pad_type', 'same')
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_large_100(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V3 """
    kwargs.setdefault('bn_eps', BN_EPS_TF_DEFAULT)
    kwargs.setdefault('pad_type', 'same')
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_large_minimal_100(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V3 """
    kwargs.setdefault('bn_eps', BN_EPS_TF_DEFAULT)
    kwargs.setdefault('pad_type', 'same')
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_minimal_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_small_075(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V3 """
    kwargs.setdefault('bn_eps', BN_EPS_TF_DEFAULT)
    kwargs.setdefault('pad_type', 'same')
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_small_100(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V3 """
    kwargs.setdefault('bn_eps', BN_EPS_TF_DEFAULT)
    kwargs.setdefault('pad_type', 'same')
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_small_minimal_100(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V3 """
    kwargs.setdefault('bn_eps', BN_EPS_TF_DEFAULT)
    kwargs.setdefault('pad_type', 'same')
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_minimal_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def fbnetv3_b(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ FBNetV3-B """
    model = _gen_fbnetv3('fbnetv3_b', pretrained=pretrained, **kwargs)
    return model


@register_model
def fbnetv3_d(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ FBNetV3-D """
    model = _gen_fbnetv3('fbnetv3_d', pretrained=pretrained, **kwargs)
    return model


@register_model
def fbnetv3_g(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ FBNetV3-G """
    model = _gen_fbnetv3('fbnetv3_g', pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_035(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ PP-LCNet 0.35"""
    model = _gen_lcnet('lcnet_035', 0.35, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_050(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ PP-LCNet 0.5"""
    model = _gen_lcnet('lcnet_050', 0.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_075(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ PP-LCNet 1.0"""
    model = _gen_lcnet('lcnet_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_100(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ PP-LCNet 1.0"""
    model = _gen_lcnet('lcnet_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_150(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ PP-LCNet 1.5"""
    model = _gen_lcnet('lcnet_150', 1.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_conv_small_035(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V4 """
    model = _gen_mobilenet_v4('mobilenetv4_conv_small_035', 0.35, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_conv_small_050(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V4 """
    model = _gen_mobilenet_v4('mobilenetv4_conv_small_050', 0.50, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_conv_small(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V4 """
    model = _gen_mobilenet_v4('mobilenetv4_conv_small', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_conv_medium(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V4 """
    model = _gen_mobilenet_v4('mobilenetv4_conv_medium', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_conv_large(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V4 """
    model = _gen_mobilenet_v4('mobilenetv4_conv_large', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_hybrid_medium(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V4 Hybrid """
    model = _gen_mobilenet_v4('mobilenetv4_hybrid_medium', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_hybrid_large(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V4 Hybrid"""
    model = _gen_mobilenet_v4('mobilenetv4_hybrid_large', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_conv_aa_medium(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V4 w/ AvgPool AA """
    model = _gen_mobilenet_v4('mobilenetv4_conv_aa_medium', 1.0, pretrained=pretrained, aa_layer='avg', **kwargs)
    return model


@register_model
def mobilenetv4_conv_blur_medium(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V4 Conv w/ Blur AA """
    model = _gen_mobilenet_v4('mobilenetv4_conv_blur_medium', 1.0, pretrained=pretrained, aa_layer='blurpc', **kwargs)
    return model


@register_model
def mobilenetv4_conv_aa_large(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V4 w/ AvgPool AA """
    model = _gen_mobilenet_v4('mobilenetv4_conv_aa_large', 1.0, pretrained=pretrained, aa_layer='avg', **kwargs)
    return model


@register_model
def mobilenetv4_hybrid_medium_075(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V4 Hybrid """
    model = _gen_mobilenet_v4('mobilenetv4_hybrid_medium_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_hybrid_large_075(pretrained: bool = False, **kwargs) -> MobileNetV3:
    """ MobileNet V4 Hybrid"""
    model = _gen_mobilenet_v4('mobilenetv4_hybrid_large_075', 0.75, pretrained=pretrained, **kwargs)
    return model


register_model_deprecations(__name__, {
    'mobilenetv3_large_100_miil': 'mobilenetv3_large_100.miil_in21k_ft_in1k',
    'mobilenetv3_large_100_miil_in21k': 'mobilenetv3_large_100.miil_in21k',
})
