""" ReXNet

A PyTorch impl of `ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network` -
https://arxiv.org/abs/2007.00992

Adapted from original impl at https://github.com/clovaai/rexnet
Copyright (c) 2020-present NAVER Corp. MIT license

Changes for timm, feature extraction, and rounded channel variant hacked together by Ross Wightman
Copyright 2020 Ross Wightman
"""

from functools import partial
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import ClassifierHead, create_act_layer, ConvNormAct, DropPath, make_divisible, SEModule
from ._builder import build_model_with_cfg
from ._efficientnet_builder import efficientnet_init_weights
from ._features import feature_take_indices
from ._manipulate import checkpoint, checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['RexNet']  # model_registry will add each entrypoint fn to this


SEWithNorm = partial(SEModule, norm_layer=nn.BatchNorm2d)


class LinearBottleneck(nn.Module):
    """Linear bottleneck block for ReXNet.

    A mobile inverted residual bottleneck block as used in MobileNetV2 and subsequent models.
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int,
            dilation: Tuple[int, int] = (1, 1),
            exp_ratio: float = 1.0,
            se_ratio: float = 0.,
            ch_div: int = 1,
            act_layer: str = 'swish',
            dw_act_layer: str = 'relu6',
            drop_path: Optional[nn.Module] = None,
    ):
        """Initialize LinearBottleneck.

        Args:
            in_chs: Number of input channels.
            out_chs: Number of output channels.
            stride: Stride for depthwise conv.
            dilation: Dilation rates.
            exp_ratio: Expansion ratio.
            se_ratio: Squeeze-excitation ratio.
            ch_div: Channel divisor.
            act_layer: Activation layer for expansion.
            dw_act_layer: Activation layer for depthwise.
            drop_path: Drop path module.
        """
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and dilation[0] == dilation[1] and in_chs <= out_chs
        self.in_channels = in_chs
        self.out_channels = out_chs

        if exp_ratio != 1.:
            dw_chs = make_divisible(round(in_chs * exp_ratio), divisor=ch_div)
            self.conv_exp = ConvNormAct(in_chs, dw_chs, act_layer=act_layer)
        else:
            dw_chs = in_chs
            self.conv_exp = None

        self.conv_dw = ConvNormAct(
            dw_chs,
            dw_chs,
            kernel_size=3,
            stride=stride,
            dilation=dilation[0],
            groups=dw_chs,
            apply_act=False,
        )
        if se_ratio > 0:
            self.se = SEWithNorm(dw_chs, rd_channels=make_divisible(int(dw_chs * se_ratio), ch_div))
        else:
            self.se = None
        self.act_dw = create_act_layer(dw_act_layer)

        self.conv_pwl = ConvNormAct(dw_chs, out_chs, 1, apply_act=False)
        self.drop_path = drop_path

    def feat_channels(self, exp: bool = False) -> int:
        """Get feature channel count.

        Args:
            exp: Return expanded channels if True.

        Returns:
            Number of feature channels.
        """
        return self.conv_dw.out_channels if exp else self.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        shortcut = x
        if self.conv_exp is not None:
            x = self.conv_exp(x)
        x = self.conv_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.act_dw(x)
        x = self.conv_pwl(x)
        if self.use_shortcut:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
        return x


def _block_cfg(
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        initial_chs: int = 16,
        final_chs: int = 180,
        se_ratio: float = 0.,
        ch_div: int = 1,
) -> List[Tuple[int, float, int, float]]:
    """Generate ReXNet block configuration.

    Args:
        width_mult: Width multiplier.
        depth_mult: Depth multiplier.
        initial_chs: Initial channel count.
        final_chs: Final channel count.
        se_ratio: Squeeze-excitation ratio.
        ch_div: Channel divisor.

    Returns:
        List of tuples (out_channels, exp_ratio, stride, se_ratio).
    """
    layers = [1, 2, 2, 3, 3, 5]
    strides = [1, 2, 2, 2, 1, 2]
    layers = [ceil(element * depth_mult) for element in layers]
    strides = sum([[element] + [1] * (layers[idx] - 1) for idx, element in enumerate(strides)], [])
    exp_ratios = [1] * layers[0] + [6] * sum(layers[1:])
    depth = sum(layers[:]) * 3
    base_chs = initial_chs / width_mult if width_mult < 1.0 else initial_chs

    # The following channel configuration is a simple instance to make each layer become an expand layer.
    out_chs_list = []
    for i in range(depth // 3):
        out_chs_list.append(make_divisible(round(base_chs * width_mult), divisor=ch_div))
        base_chs += final_chs / (depth // 3 * 1.0)

    se_ratios = [0.] * (layers[0] + layers[1]) + [se_ratio] * sum(layers[2:])

    return list(zip(out_chs_list, exp_ratios, strides, se_ratios))


def _build_blocks(
        block_cfg: List[Tuple[int, float, int, float]],
        prev_chs: int,
        width_mult: float,
        ch_div: int = 1,
        output_stride: int = 32,
        act_layer: str = 'swish',
        dw_act_layer: str = 'relu6',
        drop_path_rate: float = 0.,
) -> Tuple[List[nn.Module], List[Dict[str, Any]]]:
    """Build ReXNet blocks from configuration.

    Args:
        block_cfg: Block configuration list.
        prev_chs: Previous channel count.
        width_mult: Width multiplier.
        ch_div: Channel divisor.
        output_stride: Target output stride.
        act_layer: Activation layer name.
        dw_act_layer: Depthwise activation layer name.
        drop_path_rate: Drop path rate.

    Returns:
        Tuple of (features list, feature_info list).
    """
    feat_chs = [prev_chs]
    feature_info = []
    curr_stride = 2
    dilation = 1
    features = []
    num_blocks = len(block_cfg)
    for block_idx, (chs, exp_ratio, stride, se_ratio) in enumerate(block_cfg):
        next_dilation = dilation
        if stride > 1:
            fname = 'stem' if block_idx == 0 else f'features.{block_idx - 1}'
            feature_info += [dict(num_chs=feat_chs[-1], reduction=curr_stride, module=fname)]
            if curr_stride >= output_stride:
                next_dilation = dilation * stride
                stride = 1
        block_dpr = drop_path_rate * block_idx / (num_blocks - 1)  # stochastic depth linear decay rule
        drop_path = DropPath(block_dpr) if block_dpr > 0. else None
        features.append(LinearBottleneck(
            in_chs=prev_chs,
            out_chs=chs,
            exp_ratio=exp_ratio,
            stride=stride,
            dilation=(dilation, next_dilation),
            se_ratio=se_ratio,
            ch_div=ch_div,
            act_layer=act_layer,
            dw_act_layer=dw_act_layer,
            drop_path=drop_path,
        ))
        curr_stride *= stride
        dilation = next_dilation
        prev_chs = chs
        feat_chs += [features[-1].feat_channels()]
    pen_chs = make_divisible(1280 * width_mult, divisor=ch_div)
    feature_info += [dict(num_chs=feat_chs[-1], reduction=curr_stride, module=f'features.{len(features) - 1}')]
    features.append(ConvNormAct(prev_chs, pen_chs, act_layer=act_layer))
    return features, feature_info


class RexNet(nn.Module):
    """ReXNet model architecture.

    Based on `ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network`
    - https://arxiv.org/abs/2007.00992
    """

    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            output_stride: int = 32,
            initial_chs: int = 16,
            final_chs: int = 180,
            width_mult: float = 1.0,
            depth_mult: float = 1.0,
            se_ratio: float = 1/12.,
            ch_div: int = 1,
            act_layer: str = 'swish',
            dw_act_layer: str = 'relu6',
            drop_rate: float = 0.2,
            drop_path_rate: float = 0.,
    ):
        """Initialize ReXNet.

        Args:
            in_chans: Number of input channels.
            num_classes: Number of classes for classification.
            global_pool: Global pooling type.
            output_stride: Output stride.
            initial_chs: Initial channel count.
            final_chs: Final channel count.
            width_mult: Width multiplier.
            depth_mult: Depth multiplier.
            se_ratio: Squeeze-excitation ratio.
            ch_div: Channel divisor.
            act_layer: Activation layer name.
            dw_act_layer: Depthwise activation layer name.
            drop_rate: Dropout rate.
            drop_path_rate: Drop path rate.
        """
        super(RexNet, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        assert output_stride in (32, 16, 8)
        stem_base_chs = 32 / width_mult if width_mult < 1.0 else 32
        stem_chs = make_divisible(round(stem_base_chs * width_mult), divisor=ch_div)
        self.stem = ConvNormAct(in_chans, stem_chs, 3, stride=2, act_layer=act_layer)

        block_cfg = _block_cfg(width_mult, depth_mult, initial_chs, final_chs, se_ratio, ch_div)
        features, self.feature_info = _build_blocks(
            block_cfg,
            stem_chs,
            width_mult,
            ch_div,
            output_stride,
            act_layer,
            dw_act_layer,
            drop_path_rate,
        )
        self.num_features = self.head_hidden_size = features[-1].out_channels
        self.features = nn.Sequential(*features)

        self.head = ClassifierHead(self.num_features, num_classes, global_pool, drop_rate)

        efficientnet_init_weights(self)

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        """Group matcher for parameter groups.

        Args:
            coarse: Whether to use coarse grouping.

        Returns:
            Dictionary of grouped parameters.
        """
        matcher = dict(
            stem=r'^stem',
            blocks=r'^features\.(\d+)',
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing.

        Args:
            enable: Whether to enable gradient checkpointing.
        """
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """Get the classifier module.

        Returns:
            Classifier module.
        """
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Reset the classifier.

        Args:
            num_classes: Number of classes for new classifier.
            global_pool: Global pooling type.
        """
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
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        stage_ends = [int(info['module'].split('.')[-1]) for info in self.feature_info]
        take_indices, max_index = feature_take_indices(len(stage_ends), indices)
        take_indices = [stage_ends[i] for i in take_indices]
        max_index = stage_ends[max_index]

        # forward pass
        x = self.stem(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.features
        else:
            stages = self.features[:max_index + 1]

        for feat_idx, stage in enumerate(stages):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(stage, x)
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
        stage_ends = [int(info['module'].split('.')[-1]) for info in self.feature_info]
        take_indices, max_index = feature_take_indices(len(stage_ends), indices)
        max_index = stage_ends[max_index]
        self.features = self.features[:max_index + 1]  # truncate blocks w/ stem as idx 0
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction layers.

        Args:
            x: Input tensor.

        Returns:
            Feature tensor.
        """
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.features, x)
        else:
            x = self.features(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass through head.

        Args:
            x: Input features.
            pre_logits: Return features before final linear layer.

        Returns:
            Classification logits or features.
        """
        return self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)

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


def _create_rexnet(variant: str, pretrained: bool, **kwargs) -> RexNet:
    """Create a ReXNet model.

    Args:
        variant: Model variant name.
        pretrained: Load pretrained weights.
        **kwargs: Additional model arguments.

    Returns:
        ReXNet model instance.
    """
    feature_cfg = dict(flatten_sequential=True)
    return build_model_with_cfg(
        RexNet,
        variant,
        pretrained,
        feature_cfg=feature_cfg,
        **kwargs,
    )


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
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
        'license': 'mit', **kwargs
    }


default_cfgs = generate_default_cfgs({
    'rexnet_100.nav_in1k': _cfg(hf_hub_id='timm/'),
    'rexnet_130.nav_in1k': _cfg(hf_hub_id='timm/'),
    'rexnet_150.nav_in1k': _cfg(hf_hub_id='timm/'),
    'rexnet_200.nav_in1k': _cfg(hf_hub_id='timm/'),
    'rexnet_300.nav_in1k': _cfg(hf_hub_id='timm/'),
    'rexnetr_100.untrained': _cfg(),
    'rexnetr_130.untrained': _cfg(),
    'rexnetr_150.untrained': _cfg(),
    'rexnetr_200.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_crop_pct=1.0, test_input_size=(3, 288, 288), license='apache-2.0'),
    'rexnetr_300.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_crop_pct=1.0, test_input_size=(3, 288, 288), license='apache-2.0'),
    'rexnetr_200.sw_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
        crop_pct=0.95, test_crop_pct=1.0, test_input_size=(3, 288, 288), license='apache-2.0'),
    'rexnetr_300.sw_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
        crop_pct=0.95, test_crop_pct=1.0, test_input_size=(3, 288, 288), license='apache-2.0'),
})


@register_model
def rexnet_100(pretrained: bool = False, **kwargs) -> RexNet:
    """ReXNet V1 1.0x"""
    return _create_rexnet('rexnet_100', pretrained, **kwargs)


@register_model
def rexnet_130(pretrained: bool = False, **kwargs) -> RexNet:
    """ReXNet V1 1.3x"""
    return _create_rexnet('rexnet_130', pretrained, width_mult=1.3, **kwargs)


@register_model
def rexnet_150(pretrained: bool = False, **kwargs) -> RexNet:
    """ReXNet V1 1.5x"""
    return _create_rexnet('rexnet_150', pretrained, width_mult=1.5, **kwargs)


@register_model
def rexnet_200(pretrained: bool = False, **kwargs) -> RexNet:
    """ReXNet V1 2.0x"""
    return _create_rexnet('rexnet_200', pretrained, width_mult=2.0, **kwargs)


@register_model
def rexnet_300(pretrained: bool = False, **kwargs) -> RexNet:
    """ReXNet V1 3.0x"""
    return _create_rexnet('rexnet_300', pretrained, width_mult=3.0, **kwargs)


@register_model
def rexnetr_100(pretrained: bool = False, **kwargs) -> RexNet:
    """ReXNet V1 1.0x w/ rounded (mod 8) channels"""
    return _create_rexnet('rexnetr_100', pretrained, ch_div=8, **kwargs)


@register_model
def rexnetr_130(pretrained: bool = False, **kwargs) -> RexNet:
    """ReXNet V1 1.3x w/ rounded (mod 8) channels"""
    return _create_rexnet('rexnetr_130', pretrained, width_mult=1.3, ch_div=8, **kwargs)


@register_model
def rexnetr_150(pretrained: bool = False, **kwargs) -> RexNet:
    """ReXNet V1 1.5x w/ rounded (mod 8) channels"""
    return _create_rexnet('rexnetr_150', pretrained, width_mult=1.5, ch_div=8, **kwargs)


@register_model
def rexnetr_200(pretrained: bool = False, **kwargs) -> RexNet:
    """ReXNet V1 2.0x w/ rounded (mod 8) channels"""
    return _create_rexnet('rexnetr_200', pretrained, width_mult=2.0, ch_div=8, **kwargs)


@register_model
def rexnetr_300(pretrained: bool = False, **kwargs) -> RexNet:
    """ReXNet V1 3.0x w/ rounded (mod 16) channels"""
    return _create_rexnet('rexnetr_300', pretrained, width_mult=3.0, ch_div=16, **kwargs)
