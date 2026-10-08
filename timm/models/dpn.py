""" PyTorch implementation of DualPathNetworks
Based on original MXNet implementation https://github.com/cypw/DPNs with
many ideas from another PyTorch implementation https://github.com/oyam/pytorch-DPNs.

This implementation is compatible with the pretrained weights from cypw's MXNet implementation.

Hacked together by / Copyright 2020 Ross Wightman
"""
from collections import OrderedDict
from functools import partial
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DPN_MEAN, IMAGENET_DPN_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import BatchNormAct2d, ConvNormAct, create_conv2d, create_classifier, get_norm_act_layer, \
    get_device_dtype
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['DPN']


class BnAct(nn.Module):
    """Norm + act wrapper, kept as a module so the pretrained weight key ('.bn') is preserved."""
    def __init__(
            self,
            in_chs: int,
            norm_layer: Type[nn.Module] = BatchNormAct2d,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.bn = norm_layer(in_chs, eps=0.001, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class BnActConv2d(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int,
            stride: int,
            groups: int = 1,
            norm_layer: Type[nn.Module] = BatchNormAct2d,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.bn = norm_layer(in_chs, eps=0.001, **dd)
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, groups=groups, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.bn(x))


class DualPathBlock(nn.Module):
    """Dual path block, the residual path (first num_1x1_c channels) and dense path (remaining channels)
    are carried through the network as a single concatenated tensor.
    """
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
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        assert block_type in ('proj', 'down', 'normal')
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        self.key_stride = 2 if block_type == 'down' else 1
        self.has_proj = block_type != 'normal'

        # NOTE separate attribute names for the stride 1 / 2 projection are kept for pretrained weight compat
        self.c1x1_w_s1: Optional[nn.Module] = None
        self.c1x1_w_s2: Optional[nn.Module] = None
        if self.has_proj:
            proj = BnActConv2d(in_chs, num_1x1_c + 2 * inc, kernel_size=1, stride=self.key_stride, **dd)
            if self.key_stride == 2:
                self.c1x1_w_s2 = proj
            else:
                self.c1x1_w_s1 = proj

        self.c1x1_a = BnActConv2d(in_chs, num_1x1_a, kernel_size=1, stride=1, **dd)
        self.c3x3_b = BnActConv2d(num_1x1_a, num_3x3_b, kernel_size=3, stride=self.key_stride, groups=groups, **dd)
        self.c1x1_c1: Optional[nn.Module] = None
        self.c1x1_c2: Optional[nn.Module] = None
        if b:
            self.c1x1_c = BnAct(num_3x3_b, **dd)
            self.c1x1_c1 = create_conv2d(num_3x3_b, num_1x1_c, kernel_size=1, **dd)
            self.c1x1_c2 = create_conv2d(num_3x3_b, inc, kernel_size=1, **dd)
        else:
            self.c1x1_c = BnActConv2d(num_3x3_b, num_1x1_c + inc, kernel_size=1, stride=1, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c1x1_w_s1 is not None:
            x_s = self.c1x1_w_s1(x)
        elif self.c1x1_w_s2 is not None:
            x_s = self.c1x1_w_s2(x)
        else:
            x_s = x
        x_s1 = x_s[:, :self.num_1x1_c]
        x_s2 = x_s[:, self.num_1x1_c:]

        x = self.c1x1_a(x)
        x = self.c3x3_b(x)
        x = self.c1x1_c(x)
        if self.c1x1_c1 is not None and self.c1x1_c2 is not None:
            out1 = self.c1x1_c1(x)
            out2 = self.c1x1_c2(x)
        else:
            out1 = x[:, :self.num_1x1_c]
            out2 = x[:, self.num_1x1_c:]
        return torch.cat([x_s1 + out1, x_s2, out2], dim=1)


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
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_rate = drop_rate
        self.b = b
        self.grad_checkpointing = False
        assert output_stride == 32  # FIXME look into dilation support

        norm_layer = partial(get_norm_act_layer(norm_layer, act_layer=act_layer), eps=.001)
        fc_norm_layer = partial(get_norm_act_layer(norm_layer, act_layer=fc_act_layer), eps=.001, inplace=False)
        bw_factor = 1 if small else 4

        # NOTE the flat 'features' sequential and its conv{stage}_{block} naming are kept for pretrained weight
        # compat, stage boundaries are tracked by index in feature_ends for forward_intermediates / pruning.
        blocks = OrderedDict()
        blocks['conv1_1'] = ConvNormAct(
            in_chans,
            num_init_features,
            kernel_size=3 if small else 7,
            stride=2,
            norm_layer=norm_layer,
            **dd,
        )
        blocks['conv1_pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.feature_info = [dict(num_chs=num_init_features, reduction=2, module='features.conv1_1')]
        self.feature_ends = [0]

        in_chs = num_init_features
        for i, (bw_base, depth, inc) in enumerate(zip((64, 128, 256, 512), k_sec, inc_sec)):
            bw = bw_base * bw_factor
            r = k_r * bw_base // 64
            name = f'conv{i + 2}'
            blocks[f'{name}_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'proj' if i == 0 else 'down', b, **dd)
            in_chs = bw + 3 * inc
            for j in range(2, depth + 1):
                blocks[f'{name}_{j}'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b, **dd)
                in_chs += inc
            self.feature_info.append(dict(num_chs=in_chs, reduction=4 * 2 ** i, module=f'features.{name}_{depth}'))
            self.feature_ends.append(len(blocks) - 1)

        blocks['conv5_bn_ac'] = BnAct(in_chs, norm_layer=fc_norm_layer, **dd)
        self.features = nn.Sequential(blocks)
        self.num_features = self.head_hidden_size = in_chs

        # Using 1x1 conv for the FC layer to allow the extra pooling scheme
        self.global_pool, self.classifier = create_classifier(
            self.num_features,
            self.num_classes,
            pool_type=global_pool,
            use_conv=True,
            **dd,
        )
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False):
        matcher = dict(
            stem=r'^features\.conv1',
            blocks=[
                (r'^features\.conv(\d+)' if coarse else r'^features\.conv(\d+)_(\d+)', None),
                (r'^features\.conv5_bn_ac', (99999,))
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
        self.num_classes = num_classes
        if global_pool is None:
            global_pool = self.global_pool.pool_type
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool, use_conv=True, **dd)
        self.global_pool.train(self.training)
        self.classifier.train(self.training)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.flatten.train(self.training)

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
            indices: Take last n features if int, all if None, select matching indices if sequence.
                Index 0 is the stem output, indices 1-4 the stage outputs.
            norm: Apply the final norm + act layer to the last intermediate
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.feature_ends), indices)
        last_idx = len(self.feature_ends) - 1
        norm_idx = len(self.features) - 1  # final norm + act is the last module in features

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            layers = self.features
        else:
            layers = self.features[:self.feature_ends[max_index] + 1]
        feat_idx = 0
        for i, layer in enumerate(layers):
            x = layer(x)
            if i == self.feature_ends[feat_idx]:
                if feat_idx in take_indices:
                    if norm and feat_idx == last_idx:
                        intermediates.append(self.features[norm_idx](x))
                    else:
                        intermediates.append(x)
                feat_idx = min(feat_idx + 1, last_idx)

        if intermediates_only:
            return intermediates

        if max_index == last_idx and len(layers) <= norm_idx:
            x = self.features[norm_idx](x)  # stopped early at last stage, apply the final norm + act

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.feature_ends), indices)
        keep = list(self.features.named_children())[:self.feature_ends[max_index] + 1]
        if max_index == len(self.feature_ends) - 1 and not prune_norm:
            keep.append(('conv5_bn_ac', self.features[-1]))
        self.features = nn.Sequential(OrderedDict(keep))
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpointing and not torch.jit.is_scripting():
            return checkpoint_seq(self.features, x)
        return self.features(x)

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


def _create_dpn(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        DPN,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True),
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DPN_MEAN, 'std': IMAGENET_DPN_STD,
        'first_conv': 'features.conv1_1.conv', 'classifier': 'classifier', 'license': 'apache-2.0',
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
def dpn48b(pretrained=False, **kwargs) -> DPN:
    model_args = dict(
        small=True, num_init_features=10, k_r=128, groups=32,
        b=True, k_sec=(3, 4, 6, 3), inc_sec=(16, 32, 32, 64), act_layer='silu')
    return _create_dpn('dpn48b', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def dpn68(pretrained=False, **kwargs) -> DPN:
    model_args = dict(
        small=True, num_init_features=10, k_r=128, groups=32,
        k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64))
    return _create_dpn('dpn68', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def dpn68b(pretrained=False, **kwargs) -> DPN:
    model_args = dict(
        small=True, num_init_features=10, k_r=128, groups=32,
        b=True, k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64))
    return _create_dpn('dpn68b', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def dpn92(pretrained=False, **kwargs) -> DPN:
    model_args = dict(
        num_init_features=64, k_r=96, groups=32,
        k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128))
    return _create_dpn('dpn92', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def dpn98(pretrained=False, **kwargs) -> DPN:
    model_args = dict(
        num_init_features=96, k_r=160, groups=40,
        k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128))
    return _create_dpn('dpn98', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def dpn131(pretrained=False, **kwargs) -> DPN:
    model_args = dict(
        num_init_features=128, k_r=160, groups=40,
        k_sec=(4, 8, 28, 3), inc_sec=(16, 32, 32, 128))
    return _create_dpn('dpn131', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def dpn107(pretrained=False, **kwargs) -> DPN:
    model_args = dict(
        num_init_features=128, k_r=200, groups=50,
        k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128))
    return _create_dpn('dpn107', pretrained=pretrained, **dict(model_args, **kwargs))
