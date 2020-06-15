""" VoVNet (V1 & V2)

Papers:
* `An Energy and GPU-Computation Efficient Backbone Network` - https://arxiv.org/abs/1904.09730
* `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667

Looked at  https://github.com/youngwanLEE/vovnet-detectron2 &
https://github.com/stigma0617/VoVNet.pytorch/blob/master/models_vovnet/vovnet.py
for some reference, rewrote most of the code.

Hacked together by Ross Wightman
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .registry import register_model
from .helpers import load_pretrained
from .layers import ConvBnAct, SeparableConvBnAct, BatchNormAct2d, SelectAdaptivePool2d, \
    create_attn, create_norm_act, get_norm_act_layer


# model cfgs adapted from https://github.com/youngwanLEE/vovnet-detectron2 &
# https://github.com/stigma0617/VoVNet.pytorch/blob/master/models_vovnet/vovnet.py
model_cfgs = dict(
    vovnet39a=dict(
        stem_chs=[64, 64, 128],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[256, 512, 768, 1024],
        layer_per_block=5,
        block_per_stage=[1, 1, 2, 2],
        residual=False,
        depthwise=False,
        attn='',
    ),
    vovnet57a=dict(
        stem_chs=[64, 64, 128],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[256, 512, 768, 1024],
        layer_per_block=5,
        block_per_stage=[1, 1, 4, 3],
        residual=False,
        depthwise=False,
        attn='',

    ),
    ese_vovnet19b_slim_dw=dict(
        stem_chs=[64, 64, 64],
        stage_conv_chs=[64, 80, 96, 112],
        stage_out_chs=[112, 256, 384, 512],
        layer_per_block=3,
        block_per_stage=[1, 1, 1, 1],
        residual=True,
        depthwise=True,
        attn='ese',

    ),
    ese_vovnet19b_dw=dict(
        stem_chs=[64, 64, 64],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[256, 512, 768, 1024],
        layer_per_block=3,
        block_per_stage=[1, 1, 1, 1],
        residual=True,
        depthwise=True,
        attn='ese',
    ),
    ese_vovnet19b_slim=dict(
        stem_chs=[64, 64, 128],
        stage_conv_chs=[64, 80, 96, 112],
        stage_out_chs=[112, 256, 384, 512],
        layer_per_block=3,
        block_per_stage=[1, 1, 1, 1],
        residual=True,
        depthwise=False,
        attn='ese',
    ),
    ese_vovnet19b=dict(
        stem_chs=[64, 64, 128],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[256, 512, 768, 1024],
        layer_per_block=3,
        block_per_stage=[1, 1, 1, 1],
        residual=True,
        depthwise=False,
        attn='ese',

    ),
    ese_vovnet39b=dict(
        stem_chs=[64, 64, 128],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[256, 512, 768, 1024],
        layer_per_block=5,
        block_per_stage=[1, 1, 2, 2],
        residual=True,
        depthwise=False,
        attn='ese',
    ),
    ese_vovnet57b=dict(
        stem_chs=[64, 64, 128],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[256, 512, 768, 1024],
        layer_per_block=5,
        block_per_stage=[1, 1, 4, 3],
        residual=True,
        depthwise=False,
        attn='ese',

    ),
    ese_vovnet99b=dict(
        stem_chs=[64, 64, 128],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[256, 512, 768, 1024],
        layer_per_block=5,
        block_per_stage=[1, 3, 9, 3],
        residual=True,
        depthwise=False,
        attn='ese',
    ),
    eca_vovnet39b=dict(
        stem_chs=[64, 64, 128],
        stage_conv_chs=[128, 160, 192, 224],
        stage_out_chs=[256, 512, 768, 1024],
        layer_per_block=5,
        block_per_stage=[1, 1, 2, 2],
        residual=True,
        depthwise=False,
        attn='eca',
    ),
)
model_cfgs['ese_vovnet39b_evos'] = model_cfgs['ese_vovnet39b']
model_cfgs['ese_vovnet99b_iabn'] = model_cfgs['ese_vovnet99b']


def _cfg(url=''):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0.conv', 'classifier': 'head.fc',
    }


default_cfgs = dict(
    vovnet39a=_cfg(url=''),
    vovnet57a=_cfg(url=''),
    ese_vovnet19b_slim_dw=_cfg(url=''),
    ese_vovnet19b_dw=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ese_vovnet19b_dw-a8741004.pth'),
    ese_vovnet19b_slim=_cfg(url=''),
    ese_vovnet39b=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ese_vovnet39b-f912fe73.pth'),
    ese_vovnet57b=_cfg(url=''),
    ese_vovnet99b=_cfg(url=''),
    eca_vovnet39b=_cfg(url=''),
    ese_vovnet39b_evos=_cfg(url=''),
    ese_vovnet99b_iabn=_cfg(url=''),
)


class SequentialAppendList(nn.Sequential):
    def __init__(self, *args):
        super(SequentialAppendList, self).__init__(*args)

    def forward(self, x: torch.Tensor, concat_list: List[torch.Tensor]) -> torch.Tensor:
        for i, module in enumerate(self):
            if i == 0:
                concat_list.append(module(x))
            else:
                concat_list.append(module(concat_list[-1]))
        x = torch.cat(concat_list, dim=1)
        return x


class OsaBlock(nn.Module):

    def __init__(self, in_chs, mid_chs, out_chs, layer_per_block, residual=False,
                 depthwise=False, attn='', norm_layer=BatchNormAct2d):
        super(OsaBlock, self).__init__()

        self.residual = residual
        self.depthwise = depthwise

        next_in_chs = in_chs
        if self.depthwise and next_in_chs != mid_chs:
            assert not residual
            self.conv_reduction = ConvBnAct(next_in_chs, mid_chs, 1, norm_layer=norm_layer)
        else:
            self.conv_reduction = None

        mid_convs = []
        for i in range(layer_per_block):
            if self.depthwise:
                conv = SeparableConvBnAct(mid_chs, mid_chs, norm_layer=norm_layer)
            else:
                conv = ConvBnAct(next_in_chs, mid_chs, 3, norm_layer=norm_layer)
            next_in_chs = mid_chs
            mid_convs.append(conv)
        self.conv_mid = SequentialAppendList(*mid_convs)

        # feature aggregation
        next_in_chs = in_chs + layer_per_block * mid_chs
        self.conv_concat = ConvBnAct(next_in_chs, out_chs, norm_layer=norm_layer)

        if attn:
            self.attn = create_attn(attn, out_chs)
        else:
            self.attn = None

    def forward(self, x):
        output = [x]
        if self.conv_reduction is not None:
            x = self.conv_reduction(x)
        x = self.conv_mid(x, output)
        x = self.conv_concat(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.residual:
            x = x + output[0]
        return x


class OsaStage(nn.Module):

    def __init__(self, in_chs, mid_chs, out_chs, block_per_stage, layer_per_block,
                 downsample=True, residual=True, depthwise=False, attn='ese', norm_layer=BatchNormAct2d):
        super(OsaStage, self).__init__()

        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        else:
            self.pool = None

        blocks = []
        for i in range(block_per_stage):
            last_block = i == block_per_stage - 1
            blocks += [OsaBlock(
                in_chs if i == 0 else out_chs, mid_chs, out_chs, layer_per_block, residual=residual and i > 0,
                depthwise=depthwise, attn=attn if last_block else '', norm_layer=norm_layer)
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        x = self.blocks(x)
        return x


class ClassifierHead(nn.Module):
    """Head."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0.):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        if num_classes > 0:
            self.fc = nn.Linear(in_chs, num_classes, bias=True)
        else:
            self.fc = nn.Identity()

    def forward(self, x):
        x = self.global_pool(x).flatten(1)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


class VovNet(nn.Module):

    def __init__(self, cfg, in_chans=3, num_classes=1000, global_pool='avg', drop_rate=0., stem_stride=4,
                 norm_layer=BatchNormAct2d):
        """ VovNet (v2)
        """
        super(VovNet, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert stem_stride in (4, 2)

        stem_chs = cfg["stem_chs"]
        stage_conv_chs = cfg["stage_conv_chs"]
        stage_out_chs = cfg["stage_out_chs"]
        block_per_stage = cfg["block_per_stage"]
        layer_per_block = cfg["layer_per_block"]

        # Stem module
        last_stem_stride = stem_stride // 2
        conv_type = SeparableConvBnAct if cfg["depthwise"] else ConvBnAct
        self.stem = nn.Sequential(*[
            ConvBnAct(in_chans, stem_chs[0], 3, stride=2, norm_layer=norm_layer),
            conv_type(stem_chs[0], stem_chs[1], 3, stride=1, norm_layer=norm_layer),
            conv_type(stem_chs[1], stem_chs[2], 3, stride=last_stem_stride, norm_layer=norm_layer),
        ])

        # OSA stages
        in_ch_list = stem_chs[-1:] + stage_out_chs[:-1]
        stage_args = dict(
            residual=cfg["residual"], depthwise=cfg["depthwise"], attn=cfg["attn"], norm_layer=norm_layer)
        stages = []
        for i in range(4):  # num_stages
            downsample = stem_stride == 2 or i > 0  # first stage has no stride/downsample if stem_stride is 4
            stages += [OsaStage(
                in_ch_list[i], stage_conv_chs[i], stage_out_chs[i], block_per_stage[i], layer_per_block,
                downsample=downsample, **stage_args)
            ]
            self.num_features = stage_out_chs[i]
        self.stages = nn.Sequential(*stages)

        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        return self.stages(x)

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)


def _vovnet(variant, pretrained=False, **kwargs):
    load_strict = True
    model_class = VovNet
    if kwargs.pop('features_only', False):
        assert False, 'Not Implemented'  # TODO
        load_strict = False
        kwargs.pop('num_classes', 0)
    model_cfg = model_cfgs[variant]
    default_cfg = default_cfgs[variant]
    model = model_class(model_cfg, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(
            model, default_cfg,
            num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3), strict=load_strict)
    return model



@register_model
def vovnet39a(pretrained=False, **kwargs):
    return _vovnet('vovnet39a', pretrained=pretrained, **kwargs)


@register_model
def vovnet57a(pretrained=False, **kwargs):
    return _vovnet('vovnet57a', pretrained=pretrained, **kwargs)


@register_model
def ese_vovnet19b_slim_dw(pretrained=False, **kwargs):
    return _vovnet('ese_vovnet19b_slim_dw', pretrained=pretrained, **kwargs)


@register_model
def ese_vovnet19b_dw(pretrained=False, **kwargs):
    return _vovnet('ese_vovnet19b_dw', pretrained=pretrained, **kwargs)


@register_model
def ese_vovnet19b_slim(pretrained=False, **kwargs):
    return _vovnet('ese_vovnet19b_slim', pretrained=pretrained, **kwargs)


@register_model
def ese_vovnet39b(pretrained=False, **kwargs):
    return _vovnet('ese_vovnet39b', pretrained=pretrained, **kwargs)


@register_model
def ese_vovnet57b(pretrained=False, **kwargs):
    return _vovnet('ese_vovnet57b', pretrained=pretrained, **kwargs)


@register_model
def ese_vovnet99b(pretrained=False, **kwargs):
    return _vovnet('ese_vovnet99b', pretrained=pretrained, **kwargs)


@register_model
def eca_vovnet39b(pretrained=False, **kwargs):
    return _vovnet('eca_vovnet39b', pretrained=pretrained, **kwargs)


# Experimental Models

@register_model
def ese_vovnet39b_evos(pretrained=False, **kwargs):
    def norm_act_fn(num_features, **kwargs):
        return create_norm_act('EvoNormSample', num_features, jit=False, **kwargs)
    return _vovnet('ese_vovnet39b_evos', pretrained=pretrained, norm_layer=norm_act_fn, **kwargs)

@register_model
def ese_vovnet99b_iabn(pretrained=False, **kwargs):
    norm_layer = get_norm_act_layer('iabn')
    return _vovnet('ese_vovnet99b_iabn', pretrained=pretrained, norm_layer=norm_layer, **kwargs)
