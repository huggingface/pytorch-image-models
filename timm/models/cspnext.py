"""CSPNeXt (RTMDet / OpenMMLab)

CSPNeXt backbone from RTMDet (OpenMMLab), ImageNet-1k classification models.

Paper: `RTMDet: An Empirical Study of Designing Real-Time Object Detectors`
    - https://arxiv.org/abs/2212.07784
    The paper describes this backbone (CSP blocks with 5x5 depthwise convs) without naming it;
    the name `CSPNeXt` comes from the MMDetection implementation.

Adapted from the MMDetection / MMPretrain implementation (Apache-2.0, Copyright (c) OpenMMLab):
    - https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet

The mmcv / mmengine dependent building blocks were rewritten in plain PyTorch with timm-style
model API, and the pretrained weights were converted from the OpenMMLab checkpoints.

Note: a different paper uses the same name, `CSPNeXt: A new efficient token hybrid backbone`
(Chen et al., EAAI 2024, https://doi.org/10.1016/j.engappai.2024.107886). It describes a different
architecture (parallel large-kernel / pooling mixers, CSPNeXt-T/S/M) and is not implemented here.
"""
import re
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import ClassifierHead, ConvNormAct, EffectiveSEModule, get_device_dtype
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import MATCH_PREV_GROUP, checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['CspNext']


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling bottleneck (SPP) with parallel stride-1 max pools."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_sizes: Tuple[int, ...] = (5, 9, 13),
            norm_layer: Union[str, Callable, Type[nn.Module]] = nn.BatchNorm2d,
            act_layer: Union[str, Callable, Type[nn.Module]] = nn.SiLU,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        mid_channels = in_channels // 2
        self.conv1 = ConvNormAct(in_channels, mid_channels, 1, norm_layer=norm_layer, act_layer=act_layer, **dd)
        self.poolings = nn.ModuleList([nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernel_sizes])
        self.conv2 = ConvNormAct(
            mid_channels * (len(kernel_sizes) + 1),
            out_channels,
            1,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **dd,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.poolings], dim=1)
        return self.conv2(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise conv followed by a pointwise (1x1) conv, each with norm and activation."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int = 0,
            dilation: int = 1,
            norm_layer: Union[str, Callable, Type[nn.Module]] = nn.BatchNorm2d,
            act_layer: Union[str, Callable, Type[nn.Module]] = nn.SiLU,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.depthwise_conv = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **dd,
        )
        self.pointwise_conv = ConvNormAct(
            in_channels,
            out_channels,
            1,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **dd,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class CspNextBlock(nn.Module):
    """CSPNeXt bottleneck: 3x3 conv followed by a large-kernel depthwise separable conv."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            add_identity: bool = True,
            kernel_size: int = 5,
            dilation: int = 1,
            norm_layer: Union[str, Callable, Type[nn.Module]] = nn.BatchNorm2d,
            act_layer: Union[str, Callable, Type[nn.Module]] = nn.SiLU,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.conv1 = ConvNormAct(
            in_channels,
            out_channels,
            3,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **dd,
        )
        self.conv2 = DepthwiseSeparableConv(
            out_channels,
            out_channels,
            kernel_size,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **dd,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        if self.add_identity:
            out = out + x
        return out


class CspNextLayer(nn.Module):
    """Cross stage partial layer of CSPNeXt blocks, with optional channel attention."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            add_identity: bool = True,
            channel_attention: bool = False,
            dilation: int = 1,
            norm_layer: Union[str, Callable, Type[nn.Module]] = nn.BatchNorm2d,
            act_layer: Union[str, Callable, Type[nn.Module]] = nn.SiLU,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        mid_channels = int(out_channels * expand_ratio)
        self.short_conv = ConvNormAct(in_channels, mid_channels, 1, norm_layer=norm_layer, act_layer=act_layer, **dd)
        self.main_conv = ConvNormAct(in_channels, mid_channels, 1, norm_layer=norm_layer, act_layer=act_layer, **dd)
        self.blocks = nn.ModuleList([
            CspNextBlock(
                mid_channels,
                mid_channels,
                add_identity=add_identity,
                dilation=dilation,
                norm_layer=norm_layer,
                act_layer=act_layer,
                **dd,
            )
            for _ in range(num_blocks)
        ])
        self.attention = EffectiveSEModule(2 * mid_channels, **dd) if channel_attention else nn.Identity()
        self.final_conv = ConvNormAct(
            2 * mid_channels,
            out_channels,
            1,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **dd,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short = self.short_conv(x)
        main = self.main_conv(x)
        for block in self.blocks:
            main = block(main)
        x = torch.cat([main, short], dim=1)
        x = self.attention(x)
        return self.final_conv(x)


class CspNext(nn.Module):
    """CSPNeXt, the backbone of RTMDet (OpenMMLab).

    A 3-conv stem followed by stages of a strided 3x3 conv and a CSP layer of large-kernel depthwise separable
    blocks. The last stage has an SPP bottleneck in front of its CSP layer and no residual connections.

    The default norm layer uses PyTorch's BatchNorm defaults, as in the classification configs. The RTMDet
    detection configs use BatchNorm with eps=0.001 and momentum=0.03, pass a matching `norm_layer` for those.
    """

    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            output_stride: int = 32,
            drop_rate: float = 0.,
            depths: Tuple[int, ...] = (3, 6, 6, 3),
            channels: Tuple[int, ...] = (128, 256, 512, 1024),
            deepen_factor: float = 1.0,
            widen_factor: float = 1.0,
            expand_ratio: float = 0.5,
            channel_attention: bool = True,
            norm_layer: Union[str, Callable, Type[nn.Module]] = nn.BatchNorm2d,
            act_layer: Union[str, Callable, Type[nn.Module]] = nn.SiLU,
            device=None,
            dtype=None,
    ):
        """
        Args:
            in_chans: Number of input image channels.
            num_classes: Number of classifier classes.
            global_pool: Global pooling type.
            output_stride: Network output stride. Down-sampling is replaced by dilation beyond this stride.
            drop_rate: Classifier dropout rate.
            depths: Number of CSPNeXt blocks per stage, before scaling by `deepen_factor`.
            channels: Output channels per stage, before scaling by `widen_factor`.
            deepen_factor: Depth multiplier applied to `depths`.
            widen_factor: Width multiplier applied to `channels`.
            expand_ratio: Ratio of hidden channels in the CSP layers.
            channel_attention: Use channel attention in each CSP layer.
            norm_layer: Normalization layer.
            act_layer: Activation layer.
        """
        super().__init__()
        if output_stride not in (8, 16, 32):
            raise ValueError(f'output_stride must be one of (8, 16, 32), got {output_stride}')
        if len(depths) != len(channels):
            raise ValueError('depths and channels must have the same length')
        dd = {'device': device, 'dtype': dtype}
        self.num_classes = num_classes
        self.grad_checkpointing = False

        stage_depths = [max(round(d * deepen_factor), 1) for d in depths]
        stage_channels = [int(c * widen_factor) for c in channels]
        stem_chs = max(stage_channels[0] // 2, 1)
        half_chs = max(stem_chs // 2, 1)
        self.stem = nn.Sequential(
            ConvNormAct(in_chans, half_chs, 3, stride=2, padding=1, norm_layer=norm_layer, act_layer=act_layer, **dd),
            ConvNormAct(half_chs, half_chs, 3, padding=1, norm_layer=norm_layer, act_layer=act_layer, **dd),
            ConvNormAct(half_chs, stem_chs, 3, padding=1, norm_layer=norm_layer, act_layer=act_layer, **dd),
        )

        self.stages = nn.Sequential()
        self.feature_info = [dict(num_chs=stem_chs, reduction=2, module='stem')]
        net_stride = 2
        dilation = 1
        prev_chs = stem_chs
        num_stages = len(stage_channels)
        for i, (chs, num_blocks) in enumerate(zip(stage_channels, stage_depths)):
            stride = 2
            first_dilation = dilation
            if net_stride * 2 > output_stride:
                stride = 1
                dilation *= 2
            net_stride *= stride
            is_last = i == num_stages - 1
            layers = [
                ConvNormAct(
                    prev_chs,
                    chs,
                    3,
                    stride=stride,
                    padding=first_dilation,
                    dilation=first_dilation,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    **dd,
                ),
            ]
            if is_last:
                layers.append(SPPBottleneck(chs, chs, norm_layer=norm_layer, act_layer=act_layer, **dd))
            layers.append(CspNextLayer(
                chs,
                chs,
                expand_ratio=expand_ratio,
                num_blocks=num_blocks,
                add_identity=not is_last,
                channel_attention=channel_attention,
                dilation=dilation,
                norm_layer=norm_layer,
                act_layer=act_layer,
                **dd,
            ))
            self.stages.append(nn.Sequential(*layers))
            self.feature_info.append(dict(num_chs=chs, reduction=net_stride, module=f'stages.{i}'))
            prev_chs = chs

        self.num_features = self.head_hidden_size = prev_chs
        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            **dd,
        )

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.\d+\.blocks\.(\d+)', None),
                (r'^stages\.(\d+)\.\d+\.(?:attention|final_conv)', MATCH_PREV_GROUP),  # run after the blocks
                (r'^stages\.(\d+)', (0,)),
            ],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        dd = get_device_dtype(self)
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool, **dd)

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
            indices: Take last n feature maps if int, all if None, select matching indices if sequence.
                Index 0 is the stem output, index n is the output of stage n.
            norm: Apply norm layer to compatible intermediates (unused, no final norm).
            stop_early: Stop iterating over stages when last desired intermediate hit.
            output_fmt: Shape of intermediate feature outputs.
            intermediates_only: Only return intermediate features.

        Returns:
            Features and list of intermediate features or just intermediate features.
        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages) + 1, indices)

        x = self.stem(x)
        if 0 in take_indices:
            intermediates.append(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice stages in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index]
        for i, stage in enumerate(stages):
            x = stage(x)
            if i + 1 in take_indices:
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
            prune_norm: Whether to prune normalization layers (unused, no final norm).
            prune_head: Whether to prune the classifier head.

        Returns:
            List of indices that were kept.
        """
        take_indices, max_index = feature_take_indices(len(self.stages) + 1, indices)
        self.stages = self.stages[:max_index]
        if prune_head:
            self.reset_classifier(0, '')
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


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: Optional[nn.Module] = None):
    """Convert an OpenMMLab (MMDetection / MMPretrain) checkpoint to this model's state dict.

    Strips the 'backbone.' prefix and renames 'stageN.' to 'stages.(N-1).'. Checkpoints that are already
    in timm format are returned unchanged.
    """
    state_dict = state_dict.get('state_dict', state_dict)
    out = {}
    for k, v in state_dict.items():
        k = re.sub(r'^backbone\.', '', k)
        k = re.sub(r'^stage(\d+)\.', lambda m: f'stages.{int(m.group(1)) - 1}.', k)
        out[k] = v
    return out


def _cfg(url: str = '', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': (7, 7),
        # source eval transform: resize short edge to 236 (bicubic) -> center crop 224
        'crop_pct': 0.949,
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0.conv',
        'classifier': 'head.fc',
        'license': 'apache-2.0',
        'origin_url': 'https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet',
        'paper_name': 'RTMDet: An Empirical Study of Designing Real-Time Object Detectors',
        'paper_ids': 'arXiv:2212.07784',
        **kwargs,
    }


_NAME_NOTE = "CSPNeXt as in RTMDet / MMDetection; not the EAAI 2024 'token hybrid backbone' of the same name."
_MMDET_RSB_URL = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/'

default_cfgs = generate_default_cfgs({
    # ImageNet-1k pretrained CSPNeXt from OpenMMLab (RTMDet), converted to timm format
    'cspnext_tiny.rsb_a1_in1k': _cfg(
        hf_hub_id='timm/',
        notes=(f'Converted from {_MMDET_RSB_URL}cspnext-tiny_imagenet_600e-3a2dd350.pth', _NAME_NOTE),
    ),
    'cspnext_s.rsb_a1_in1k': _cfg(
        hf_hub_id='timm/',
        notes=(f'Converted from {_MMDET_RSB_URL}cspnext-s_imagenet_600e-ea671761.pth', _NAME_NOTE),
    ),
    'cspnext_m.rsb_a1_in1k': _cfg(
        hf_hub_id='timm/',
        notes=(f'Converted from {_MMDET_RSB_URL}cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth', _NAME_NOTE),
    ),
    'cspnext_l.rsb_a1_in1k': _cfg(
        hf_hub_id='timm/',
        notes=(f'Converted from {_MMDET_RSB_URL}cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth', _NAME_NOTE),
    ),
    'cspnext_x.rsb_a1_in1k': _cfg(
        hf_hub_id='timm/',
        notes=(f'Converted from {_MMDET_RSB_URL}cspnext-x_8xb256-rsb-a1-600e_in1k-b3f78edd.pth', _NAME_NOTE),
    ),
})


def _create_cspnext(variant: str, pretrained: bool = False, **kwargs) -> CspNext:
    return build_model_with_cfg(
        CspNext,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3, 4), feature_cls='getter'),
        **kwargs,
    )


@register_model
def cspnext_tiny(pretrained: bool = False, **kwargs) -> CspNext:
    model_args = dict(deepen_factor=0.167, widen_factor=0.375)
    return _create_cspnext('cspnext_tiny', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cspnext_s(pretrained: bool = False, **kwargs) -> CspNext:
    model_args = dict(deepen_factor=0.33, widen_factor=0.5)
    return _create_cspnext('cspnext_s', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cspnext_m(pretrained: bool = False, **kwargs) -> CspNext:
    model_args = dict(deepen_factor=0.67, widen_factor=0.75)
    return _create_cspnext('cspnext_m', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cspnext_l(pretrained: bool = False, **kwargs) -> CspNext:
    model_args = dict(deepen_factor=1.0, widen_factor=1.0)
    return _create_cspnext('cspnext_l', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cspnext_x(pretrained: bool = False, **kwargs) -> CspNext:
    model_args = dict(deepen_factor=1.33, widen_factor=1.25)
    return _create_cspnext('cspnext_x', pretrained=pretrained, **dict(model_args, **kwargs))
