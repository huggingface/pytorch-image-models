"""Pytorch impl of Aligned Xception 41, 65, 71

This is a correct, from scratch impl of Aligned Xception (Deeplab) models compatible with TF weights at
https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md

Hacked together by / Copyright 2020 Ross Wightman
"""
from functools import partial
from typing import List, Dict, Type, Optional

import torch
import torch.nn as nn

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import ClassifierHead, ConvNormAct, DropPath, PadType, create_conv2d, get_norm_act_layer
from timm.layers.helpers import to_3tuple
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['XceptionAligned']


class SeparableConv2d(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            stride: int = 1,
            dilation: int = 1,
            padding: PadType = '',
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
    ):
        super(SeparableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        # depthwise convolution
        self.conv_dw = create_conv2d(
            in_chs, in_chs, kernel_size, stride=stride,
            padding=padding, dilation=dilation, depthwise=True)
        self.bn_dw = norm_layer(in_chs)
        self.act_dw = act_layer(inplace=True) if act_layer is not None else nn.Identity()

        # pointwise convolution
        self.conv_pw = create_conv2d(in_chs, out_chs, kernel_size=1)
        self.bn_pw = norm_layer(out_chs)
        self.act_pw = act_layer(inplace=True) if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        x = self.act_dw(x)
        x = self.conv_pw(x)
        x = self.bn_pw(x)
        x = self.act_pw(x)
        return x


class PreSeparableConv2d(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            stride: int = 1,
            dilation: int = 1,
            padding: PadType = '',
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            first_act: bool = True,
    ):
        super(PreSeparableConv2d, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer=act_layer)
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.norm = norm_act_layer(in_chs, inplace=True) if first_act else nn.Identity()
        # depthwise convolution
        self.conv_dw = create_conv2d(
            in_chs, in_chs, kernel_size, stride=stride,
            padding=padding, dilation=dilation, depthwise=True)

        # pointwise convolution
        self.conv_pw = create_conv2d(in_chs, out_chs, kernel_size=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x


class XceptionModule(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dilation: int = 1,
            pad_type: PadType = '',
            start_with_relu: bool = True,
            no_skip: bool = False,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None
    ):
        super(XceptionModule, self).__init__()
        out_chs = to_3tuple(out_chs)
        self.in_channels = in_chs
        self.out_channels = out_chs[-1]
        self.no_skip = no_skip
        if not no_skip and (self.out_channels != self.in_channels or stride != 1):
            self.shortcut = ConvNormAct(
                in_chs, self.out_channels, 1, stride=stride, norm_layer=norm_layer, apply_act=False)
        else:
            self.shortcut = None

        separable_act_layer = None if start_with_relu else act_layer
        self.stack = nn.Sequential()
        for i in range(3):
            if start_with_relu:
                self.stack.add_module(f'act{i + 1}', act_layer(inplace=i > 0))
            self.stack.add_module(f'conv{i + 1}', SeparableConv2d(
                in_chs, out_chs[i], 3, stride=stride if i == 2 else 1, dilation=dilation, padding=pad_type,
                act_layer=separable_act_layer, norm_layer=norm_layer))
            in_chs = out_chs[i]

        self.drop_path = drop_path

    def forward(self, x):
        skip = x
        x = self.stack(x)
        if self.shortcut is not None:
            skip = self.shortcut(skip)
        if not self.no_skip:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x = x + skip
        return x


class PreXceptionModule(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dilation: int = 1,
            pad_type: PadType = '',
            no_skip: bool = False,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None
    ):
        super(PreXceptionModule, self).__init__()
        out_chs = to_3tuple(out_chs)
        self.in_channels = in_chs
        self.out_channels = out_chs[-1]
        self.no_skip = no_skip
        if not no_skip and (self.out_channels != self.in_channels or stride != 1):
            self.shortcut = create_conv2d(in_chs, self.out_channels, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()

        self.norm = get_norm_act_layer(norm_layer, act_layer=act_layer)(in_chs, inplace=True)
        self.stack = nn.Sequential()
        for i in range(3):
            self.stack.add_module(f'conv{i + 1}', PreSeparableConv2d(
                in_chs,
                out_chs[i],
                3,
                stride=stride if i == 2 else 1,
                dilation=dilation,
                padding=pad_type,
                act_layer=act_layer,
                norm_layer=norm_layer,
                first_act=i > 0,
            ))
            in_chs = out_chs[i]

        self.drop_path = drop_path

    def forward(self, x):
        x = self.norm(x)
        skip = x
        x = self.stack(x)
        if not self.no_skip:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x = x + self.shortcut(skip)
        return x


class XceptionAligned(nn.Module):
    """Modified Aligned Xception
    """

    def __init__(
            self,
            block_cfg: List[Dict],
            num_classes: int = 1000,
            in_chans: int = 3,
            output_stride: int = 32,
            preact: bool = False,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            global_pool: str = 'avg',
    ):
        super(XceptionAligned, self).__init__()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        layer_args = dict(act_layer=act_layer, norm_layer=norm_layer)
        self.stem = nn.Sequential(*[
            ConvNormAct(in_chans, 32, kernel_size=3, stride=2, **layer_args),
            create_conv2d(32, 64, kernel_size=3, stride=1) if preact else
            ConvNormAct(32, 64, kernel_size=3, stride=1, **layer_args)
        ])

        curr_dilation = 1
        curr_stride = 2
        self.feature_info = []
        self.blocks = nn.Sequential()
        module_fn = PreXceptionModule if preact else XceptionModule
        net_num_blocks = len(block_cfg)
        net_block_idx = 0
        for i, b in enumerate(block_cfg):
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            b['drop_path'] = DropPath(block_dpr) if block_dpr > 0. else None
            b['dilation'] = curr_dilation
            if b['stride'] > 1:
                name = f'blocks.{i}.stack.conv2' if preact else f'blocks.{i}.stack.act3'
                self.feature_info += [dict(num_chs=to_3tuple(b['out_chs'])[-2], reduction=curr_stride, module=name)]
                next_stride = curr_stride * b['stride']
                if next_stride > output_stride:
                    curr_dilation *= b['stride']
                    b['stride'] = 1
                else:
                    curr_stride = next_stride
            self.blocks.add_module(str(i), module_fn(**b, **layer_args))
            self.num_features = self.blocks[-1].out_channels
            net_block_idx += 1

        self.feature_info += [dict(
            num_chs=self.num_features, reduction=curr_stride, module='blocks.' + str(len(self.blocks) - 1))]
        self.act = act_layer(inplace=True) if preact else nn.Identity()
        self.head_hidden_size = self.num_features
        self.head = ClassifierHead(
            in_features=self.num_features,
            num_classes=num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
        )

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^blocks\.(\d+)',
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.act(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _xception(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        XceptionAligned,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True, feature_cls='hook'),
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 299, 299), 'pool_size': (10, 10),
        'crop_pct': 0.903, 'interpolation': 'bicubic',
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'stem.0.conv', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'xception65.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.94,
    ),

    'xception41.tf_in1k': _cfg(hf_hub_id='timm/'),
    'xception65.tf_in1k': _cfg(hf_hub_id='timm/'),
    'xception71.tf_in1k': _cfg(hf_hub_id='timm/'),

    'xception41p.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.94,
    ),
    'xception65p.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.94,
    ),
})


@register_model
def xception41(pretrained=False, **kwargs) -> XceptionAligned:
    """ Modified Aligned Xception-41
    """
    block_cfg = [
        # entry flow
        dict(in_chs=64, out_chs=128, stride=2),
        dict(in_chs=128, out_chs=256, stride=2),
        dict(in_chs=256, out_chs=728, stride=2),
        # middle flow
        *([dict(in_chs=728, out_chs=728, stride=1)] * 8),
        # exit flow
        dict(in_chs=728, out_chs=(728, 1024, 1024), stride=2),
        dict(in_chs=1024, out_chs=(1536, 1536, 2048), stride=1, no_skip=True, start_with_relu=False),
    ]
    model_args = dict(block_cfg=block_cfg, norm_layer=partial(nn.BatchNorm2d, eps=.001, momentum=.1))
    return _xception('xception41', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def xception65(pretrained=False, **kwargs) -> XceptionAligned:
    """ Modified Aligned Xception-65
    """
    block_cfg = [
        # entry flow
        dict(in_chs=64, out_chs=128, stride=2),
        dict(in_chs=128, out_chs=256, stride=2),
        dict(in_chs=256, out_chs=728, stride=2),
        # middle flow
        *([dict(in_chs=728, out_chs=728, stride=1)] * 16),
        # exit flow
        dict(in_chs=728, out_chs=(728, 1024, 1024), stride=2),
        dict(in_chs=1024, out_chs=(1536, 1536, 2048), stride=1, no_skip=True, start_with_relu=False),
    ]
    model_args = dict(block_cfg=block_cfg, norm_layer=partial(nn.BatchNorm2d, eps=.001, momentum=.1))
    return _xception('xception65', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def xception71(pretrained=False, **kwargs) -> XceptionAligned:
    """ Modified Aligned Xception-71
    """
    block_cfg = [
        # entry flow
        dict(in_chs=64, out_chs=128, stride=2),
        dict(in_chs=128, out_chs=256, stride=1),
        dict(in_chs=256, out_chs=256, stride=2),
        dict(in_chs=256, out_chs=728, stride=1),
        dict(in_chs=728, out_chs=728, stride=2),
        # middle flow
        *([dict(in_chs=728, out_chs=728, stride=1)] * 16),
        # exit flow
        dict(in_chs=728, out_chs=(728, 1024, 1024), stride=2),
        dict(in_chs=1024, out_chs=(1536, 1536, 2048), stride=1, no_skip=True, start_with_relu=False),
    ]
    model_args = dict(block_cfg=block_cfg, norm_layer=partial(nn.BatchNorm2d, eps=.001, momentum=.1))
    return _xception('xception71', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def xception41p(pretrained=False, **kwargs) -> XceptionAligned:
    """ Modified Aligned Xception-41 w/ Pre-Act
    """
    block_cfg = [
        # entry flow
        dict(in_chs=64, out_chs=128, stride=2),
        dict(in_chs=128, out_chs=256, stride=2),
        dict(in_chs=256, out_chs=728, stride=2),
        # middle flow
        *([dict(in_chs=728, out_chs=728, stride=1)] * 8),
        # exit flow
        dict(in_chs=728, out_chs=(728, 1024, 1024), stride=2),
        dict(in_chs=1024, out_chs=(1536, 1536, 2048), no_skip=True, stride=1),
    ]
    model_args = dict(block_cfg=block_cfg, preact=True, norm_layer=nn.BatchNorm2d)
    return _xception('xception41p', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def xception65p(pretrained=False, **kwargs) -> XceptionAligned:
    """ Modified Aligned Xception-65 w/ Pre-Act
    """
    block_cfg = [
        # entry flow
        dict(in_chs=64, out_chs=128, stride=2),
        dict(in_chs=128, out_chs=256, stride=2),
        dict(in_chs=256, out_chs=728, stride=2),
        # middle flow
        *([dict(in_chs=728, out_chs=728, stride=1)] * 16),
        # exit flow
        dict(in_chs=728, out_chs=(728, 1024, 1024), stride=2),
        dict(in_chs=1024, out_chs=(1536, 1536, 2048), stride=1, no_skip=True),
    ]
    model_args = dict(
        block_cfg=block_cfg, preact=True, norm_layer=partial(nn.BatchNorm2d, eps=.001, momentum=.1))
    return _xception('xception65p', pretrained=pretrained, **dict(model_args, **kwargs))
