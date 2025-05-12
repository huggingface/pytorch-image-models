""" PyTorch implementation of DualPathNetworks
Based on original MXNet implementation https://github.com/cypw/DPNs with
many ideas from another PyTorch implementation https://github.com/oyam/pytorch-DPNs.

This implementation is compatible with the pretrained weights from cypw's MXNet implementation.

Hacked together by / Copyright 2020 Ross Wightman
"""
import re
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DPN_MEAN, IMAGENET_DPN_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import BatchNormAct2d, ConvNormAct, create_conv2d, create_classifier, get_norm_act_layer
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['DPN']


class CatBnAct(nn.Module):
    def __init__(self, in_chs, norm_layer=BatchNormAct2d):
        super(CatBnAct, self).__init__()
        self.bn = norm_layer(in_chs, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class BnActConv2d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride, groups=1, norm_layer=BatchNormAct2d):
        super(BnActConv2d, self).__init__()
        self.bn = norm_layer(in_chs, eps=0.001)
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.bn(x))


class DualPathBlock(nn.Module):
    def __init__(
            self,
            in_chs: int,
            num_1x1_a: int,
            num_3x3_b: int,
            num_1x1_c: int,
            inc: int,
            groups: int,
            block_type: str = 'normal', 
            b: bool = False,
    ):
        super(DualPathBlock, self).__init__()
        assert block_type in ('proj', 'down', 'normal')
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        self.key_stride = 2 if block_type == 'down' else 1
        self.has_proj = block_type != 'normal'

        if self.has_proj:
            self.c1x1_w = BnActConv2d(
                in_chs=in_chs,
                out_chs=num_1x1_c + (2 * inc),
                kernel_size=1,
                stride=self.key_stride,
            )
        else:
            self.c1x1_w = nn.Identity()

        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(
            in_chs=num_1x1_a,
            out_chs=num_3x3_b,
            kernel_size=3,
            stride=self.key_stride,
            groups=groups,
        )
        if self.b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = create_conv2d(num_3x3_b, num_1x1_c, kernel_size=1)
            self.c1x1_c2 = create_conv2d(num_3x3_b, inc, kernel_size=1)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)
            self.c1x1_c1 = None
            self.c1x1_c2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_s = self.c1x1_w(x)
        x_s1 = x_s[:, :self.num_1x1_c, :, :]
        x_s2 = x_s[:, self.num_1x1_c:, :, :]

        x = self.c1x1_a(x)
        x = self.c3x3_b(x)
        x = self.c1x1_c(x)

        if self.b:
            out1 = self.c1x1_c1(x)
            out2 = self.c1x1_c2(x)
        else:
            out1 = x[:, :self.num_1x1_c, :, :]
            out2 = x[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        out = torch.cat([resid, dense], dim=1)
        return out


class Stage(nn.Module):
    def __init__(
            self,
            num_blocks: int,
            in_chs: int,
            bw_param: int,
            inc: int,
            k_r: int,
            groups: int,
            block_type: str,
            b: bool,
            bw_factor: int,
    ):
        super(Stage, self).__init__()
        self.grad_checkpointing = False

        bw = bw_param * bw_factor
        r = (k_r * bw) // (64 * bw_factor)
        prev_chs = in_chs
        blocks = [DualPathBlock(prev_chs, r, r, bw, inc, groups, block_type, b)]
        prev_chs = bw + 3 * inc

        for _ in range(2, num_blocks + 1):
            blocks.append(DualPathBlock(prev_chs, r, r, bw, inc, groups, 'normal', b))
            prev_chs += inc

        self.blocks = nn.Sequential(*blocks)
        self.output_chs = prev_chs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class DPN(nn.Module):
    def __init__(
            self,
            k_sec: Tuple[int, ...] = (3, 4, 20, 3),
            inc_sec: Tuple[int, ...] = (16, 32, 24, 128),
            k_r: int = 96,
            groups: int = 32,
            num_classes: int = 1000,
            in_chans: int = 3,
            output_stride: int = 32,
            global_pool: str = 'avg',
            small: bool = False,
            num_init_features: int = 64,
            b: bool = False,
            drop_rate: float = 0.,
            norm_layer: str = 'batchnorm2d',
            act_layer: str = 'relu',
            fc_act_layer: str = 'elu',
    ):
        super(DPN, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []
        assert output_stride == 32  # FIXME look into dilation support

        norm_layer = partial(get_norm_act_layer(norm_layer, act_layer=act_layer), eps=.001)
        fc_norm_layer = partial(get_norm_act_layer(norm_layer, act_layer=fc_act_layer), eps=.001, inplace=False)

        self.stem = ConvNormAct(
            in_chans, num_init_features, kernel_size=3 if small else 7, stride=2, norm_layer=norm_layer,
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        prev_chs = num_init_features
        self.feature_info += [dict(num_chs=prev_chs, reduction=2, module=f'stem')]

        stages = []
        for i, bw_param in enumerate([64, 128, 256, 512]):
            stage = Stage(
                num_blocks=k_sec[i],
                in_chs=prev_chs,
                bw_param=bw_param,
                inc=inc_sec[i],
                k_r=k_r,
                groups=groups,
                block_type='proj' if i == 0 else 'down',
                b=b,
                bw_factor=1 if small else 4,
            )
            prev_chs = stage.output_chs
            self.feature_info += [dict(num_chs=prev_chs, reduction=2**(i+2), module=f'stages.{i}')]
            stages.append(stage)
        self.stages = nn.Sequential(*stages)
        self.norm = CatBnAct(prev_chs, norm_layer=fc_norm_layer)
        self.num_features = self.head_hidden_size = prev_chs
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool, use_conv=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        matcher = dict(
            stem=r'^stem\.\d+',
            blocks=[
                (r'^stages\.(\d+)' if coarse else r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.classifier

    def reset_classifier(self, num_classes: int, global_pool: str = 'avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool, use_conv=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()

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
        take_indices, max_index = feature_take_indices(5, indices)
        last_idx = 4

        feat_idx = 0
        x = self.stem(x)
        if feat_idx in take_indices:
            intermediates.append(x)
        x = self.pool(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index]
        for feat_idx, stage in enumerate(stages, 1):
            x = stage(x)
            if feat_idx in take_indices:
                if norm and feat_idx == last_idx:
                    x_inter = self.norm(x)  # applying final norm last intermediate
                else:
                    x_inter = x
                intermediates.append(x_inter) 

        if intermediates_only:
            return intermediates

        if feat_idx == last_idx:
            x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(5, indices)
        self.stages = self.stages[:max_index]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        if pre_logits:
            return self.flatten(x)
        x = self.classifier(x)
        return self.flatten(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    out_dict = {}
    for k, v in state_dict.items():
        if k.startswith("features.conv1_1.conv"):
            k = k.replace("features.conv1_1.conv", "stem.conv")
        if k.startswith("features.conv1_1.bn"):
            k = k.replace("features.conv1_1.bn", "stem.bn")
        if k.startswith("features.conv5_bn_ac.bn"):
            k = k.replace("features.conv5_bn_ac.bn", "norm.bn")
        m = re.match(r"features\.conv(\d)_(\d+)\.(.*)", k)
        if m:
            stage = int(m.group(1)) - 2
            block = int(m.group(2)) - 1
            rest = m.group(3).replace("_s1", "").replace("_s2", "")
            k = f"stages.{stage}.blocks.{block}.{rest}"
        out_dict[k] = v
    return out_dict


def _create_dpn(variant: str, pretrained: bool = False, **kwargs: Any) -> DPN:
    return build_model_with_cfg(
        DPN, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True),
        **kwargs,
    )


def _cfg(url: str = '', **kwargs: Any) -> Dict[str, Any]:
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DPN_MEAN, 'std': IMAGENET_DPN_STD,
        'first_conv': 'stem.conv', 'classifier': 'classifier',
        'paper_ids': 'arXiv:1707.01629',
        'paper_name': 'Dual Path Networks',
        'origin_url': 'https://github.com/cypw/DPNs',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'dpn48b.untrained': _cfg(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'dpn68.mx_in1k': _cfg(hf_hub_id='timm/'),
    'dpn68b.ra_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'dpn68b.mx_in1k': _cfg(hf_hub_id='timm/'),
    'dpn92.mx_in1k': _cfg(hf_hub_id='timm/'),
    'dpn98.mx_in1k': _cfg(hf_hub_id='timm/'),
    'dpn131.mx_in1k': _cfg(hf_hub_id='timm/'),
    'dpn107.mx_in1k': _cfg(hf_hub_id='timm/')
})


@register_model
def dpn48b(pretrained: bool = False, **kwargs: Any) -> DPN:
    model_args = dict(
        small=True, num_init_features=10, k_r=128, groups=32,
        b=True, k_sec=(3, 4, 6, 3), inc_sec=(16, 32, 32, 64), act_layer='silu')
    return _create_dpn('dpn48b', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def dpn68(pretrained: bool = False, **kwargs: Any) -> DPN:
    model_args = dict(
        small=True, num_init_features=10, k_r=128, groups=32,
        k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64))
    return _create_dpn('dpn68', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def dpn68b(pretrained: bool = False, **kwargs: Any) -> DPN:
    model_args = dict(
        small=True, num_init_features=10, k_r=128, groups=32,
        b=True, k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64))
    return _create_dpn('dpn68b', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def dpn92(pretrained: bool = False, **kwargs: Any) -> DPN:
    model_args = dict(
        num_init_features=64, k_r=96, groups=32,
        k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128))
    return _create_dpn('dpn92', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def dpn98(pretrained: bool = False, **kwargs: Any) -> DPN:
    model_args = dict(
        num_init_features=96, k_r=160, groups=40,
        k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128))
    return _create_dpn('dpn98', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def dpn131(pretrained: bool = False, **kwargs: Any) -> DPN:
    model_args = dict(
        num_init_features=128, k_r=160, groups=40,
        k_sec=(4, 8, 28, 3), inc_sec=(16, 32, 32, 128))
    return _create_dpn('dpn131', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def dpn107(pretrained: bool = False, **kwargs: Any) -> DPN:
    model_args = dict(
        num_init_features=128, k_r=200, groups=50,
        k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128))
    return _create_dpn('dpn107', pretrained=pretrained, **dict(model_args, **kwargs))
