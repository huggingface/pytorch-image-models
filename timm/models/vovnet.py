""" VoVNet (V1 & V2)

Papers:
* `An Energy and GPU-Computation Efficient Backbone Network` - https://arxiv.org/abs/1904.09730
* `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667

Looked at  https://github.com/youngwanLEE/vovnet-detectron2 &
https://github.com/stigma0617/VoVNet.pytorch/blob/master/models_vovnet/vovnet.py
for some reference, rewrote most of the code.

Hacked together by / Copyright 2020 Ross Wightman
"""

from typing import List, Optional

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import ConvNormAct, SeparableConvNormAct, BatchNormAct2d, ClassifierHead, DropPath, \
    create_attn, create_norm_act_layer
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['VovNet']  # model_registry will add each entrypoint fn to this


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

    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            layer_per_block,
            residual=False,
            depthwise=False,
            attn='',
            norm_layer=BatchNormAct2d,
            act_layer=nn.ReLU,
            drop_path=None,
    ):
        super(OsaBlock, self).__init__()

        self.residual = residual
        self.depthwise = depthwise
        conv_kwargs = dict(norm_layer=norm_layer, act_layer=act_layer)

        next_in_chs = in_chs
        if self.depthwise and next_in_chs != mid_chs:
            assert not residual
            self.conv_reduction = ConvNormAct(next_in_chs, mid_chs, 1, **conv_kwargs)
        else:
            self.conv_reduction = None

        mid_convs = []
        for i in range(layer_per_block):
            if self.depthwise:
                conv = SeparableConvNormAct(mid_chs, mid_chs, **conv_kwargs)
            else:
                conv = ConvNormAct(next_in_chs, mid_chs, 3, **conv_kwargs)
            next_in_chs = mid_chs
            mid_convs.append(conv)
        self.conv_mid = SequentialAppendList(*mid_convs)

        # feature aggregation
        next_in_chs = in_chs + layer_per_block * mid_chs
        self.conv_concat = ConvNormAct(next_in_chs, out_chs, **conv_kwargs)

        self.attn = create_attn(attn, out_chs) if attn else None

        self.drop_path = drop_path

    def forward(self, x):
        output = [x]
        if self.conv_reduction is not None:
            x = self.conv_reduction(x)
        x = self.conv_mid(x, output)
        x = self.conv_concat(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.residual:
            x = x + output[0]
        return x


class OsaStage(nn.Module):

    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_per_stage,
            layer_per_block,
            downsample=True,
            residual=True,
            depthwise=False,
            attn='ese',
            norm_layer=BatchNormAct2d,
            act_layer=nn.ReLU,
            drop_path_rates=None,
    ):
        super(OsaStage, self).__init__()
        self.grad_checkpointing = False

        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        else:
            self.pool = None

        blocks = []
        for i in range(block_per_stage):
            last_block = i == block_per_stage - 1
            if drop_path_rates is not None and drop_path_rates[i] > 0.:
                drop_path = DropPath(drop_path_rates[i])
            else:
                drop_path = None
            blocks += [OsaBlock(
                in_chs,
                mid_chs,
                out_chs,
                layer_per_block,
                residual=residual and i > 0,
                depthwise=depthwise,
                attn=attn if last_block else '',
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop_path=drop_path
            )]
            in_chs = out_chs
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class VovNet(nn.Module):

    def __init__(
            self,
            cfg,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            output_stride=32,
            norm_layer=BatchNormAct2d,
            act_layer=nn.ReLU,
            drop_rate=0.,
            drop_path_rate=0.,
            **kwargs,
    ):
        """
        Args:
            cfg (dict): Model architecture configuration
            in_chans (int): Number of input channels (default: 3)
            num_classes (int): Number of classifier classes (default: 1000)
            global_pool (str): Global pooling type (default: 'avg')
            output_stride (int): Output stride of network, one of (8, 16, 32) (default: 32)
            norm_layer (Union[str, nn.Module]): normalization layer
            act_layer (Union[str, nn.Module]): activation layer
            drop_rate (float): Dropout rate (default: 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default: 0.)
            kwargs (dict): Extra kwargs overlayed onto cfg
        """
        super(VovNet, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride == 32  # FIXME support dilation

        cfg = dict(cfg, **kwargs)
        stem_stride = cfg.get("stem_stride", 4)
        stem_chs = cfg["stem_chs"]
        stage_conv_chs = cfg["stage_conv_chs"]
        stage_out_chs = cfg["stage_out_chs"]
        block_per_stage = cfg["block_per_stage"]
        layer_per_block = cfg["layer_per_block"]
        conv_kwargs = dict(norm_layer=norm_layer, act_layer=act_layer)

        # Stem module
        last_stem_stride = stem_stride // 2
        conv_type = SeparableConvNormAct if cfg["depthwise"] else ConvNormAct
        self.stem = nn.Sequential(*[
            ConvNormAct(in_chans, stem_chs[0], 3, stride=2, **conv_kwargs),
            conv_type(stem_chs[0], stem_chs[1], 3, stride=1, **conv_kwargs),
            conv_type(stem_chs[1], stem_chs[2], 3, stride=last_stem_stride, **conv_kwargs),
        ])
        self.feature_info = [dict(
            num_chs=stem_chs[1], reduction=2, module=f'stem.{1 if stem_stride == 4 else 2}')]
        current_stride = stem_stride

        # OSA stages
        stage_dpr = torch.split(torch.linspace(0, drop_path_rate, sum(block_per_stage)), block_per_stage)
        in_ch_list = stem_chs[-1:] + stage_out_chs[:-1]
        stage_args = dict(residual=cfg["residual"], depthwise=cfg["depthwise"], attn=cfg["attn"], **conv_kwargs)
        stages = []
        for i in range(4):  # num_stages
            downsample = stem_stride == 2 or i > 0  # first stage has no stride/downsample if stem_stride is 4
            stages += [OsaStage(
                in_ch_list[i],
                stage_conv_chs[i],
                stage_out_chs[i],
                block_per_stage[i],
                layer_per_block,
                downsample=downsample,
                drop_path_rates=stage_dpr[i],
                **stage_args,
            )]
            self.num_features = stage_out_chs[i]
            current_stride *= 2 if downsample else 1
            self.feature_info += [dict(num_chs=self.num_features, reduction=current_stride, module=f'stages.{i}')]

        self.stages = nn.Sequential(*stages)

        self.head_hidden_size = self.num_features
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else r'^stages\.(\d+).blocks\.(\d+)',
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        return self.stages(x)

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


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


def _create_vovnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        VovNet,
        variant,
        pretrained,
        model_cfg=model_cfgs[variant],
        feature_cfg=dict(flatten_sequential=True),
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0.conv', 'classifier': 'head.fc', **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'vovnet39a.untrained': _cfg(url=''),
    'vovnet57a.untrained': _cfg(url=''),
    'ese_vovnet19b_slim_dw.untrained': _cfg(url=''),
    'ese_vovnet19b_dw.ra_in1k': _cfg(
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    'ese_vovnet19b_slim.untrained': _cfg(url=''),
    'ese_vovnet39b.ra_in1k': _cfg(
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    'ese_vovnet57b.untrained': _cfg(url=''),
    'ese_vovnet99b.untrained': _cfg(url=''),
    'eca_vovnet39b.untrained': _cfg(url=''),
    'ese_vovnet39b_evos.untrained': _cfg(url=''),
})


@register_model
def vovnet39a(pretrained=False, **kwargs) -> VovNet:
    return _create_vovnet('vovnet39a', pretrained=pretrained, **kwargs)


@register_model
def vovnet57a(pretrained=False, **kwargs) -> VovNet:
    return _create_vovnet('vovnet57a', pretrained=pretrained, **kwargs)


@register_model
def ese_vovnet19b_slim_dw(pretrained=False, **kwargs) -> VovNet:
    return _create_vovnet('ese_vovnet19b_slim_dw', pretrained=pretrained, **kwargs)


@register_model
def ese_vovnet19b_dw(pretrained=False, **kwargs) -> VovNet:
    return _create_vovnet('ese_vovnet19b_dw', pretrained=pretrained, **kwargs)


@register_model
def ese_vovnet19b_slim(pretrained=False, **kwargs) -> VovNet:
    return _create_vovnet('ese_vovnet19b_slim', pretrained=pretrained, **kwargs)


@register_model
def ese_vovnet39b(pretrained=False, **kwargs) -> VovNet:
    return _create_vovnet('ese_vovnet39b', pretrained=pretrained, **kwargs)


@register_model
def ese_vovnet57b(pretrained=False, **kwargs) -> VovNet:
    return _create_vovnet('ese_vovnet57b', pretrained=pretrained, **kwargs)


@register_model
def ese_vovnet99b(pretrained=False, **kwargs) -> VovNet:
    return _create_vovnet('ese_vovnet99b', pretrained=pretrained, **kwargs)


@register_model
def eca_vovnet39b(pretrained=False, **kwargs) -> VovNet:
    return _create_vovnet('eca_vovnet39b', pretrained=pretrained, **kwargs)


# Experimental Models

@register_model
def ese_vovnet39b_evos(pretrained=False, **kwargs) -> VovNet:
    def norm_act_fn(num_features, **nkwargs):
        return create_norm_act_layer('evonorms0', num_features, jit=False, **nkwargs)
    return _create_vovnet('ese_vovnet39b_evos', pretrained=pretrained, norm_layer=norm_act_fn, **kwargs)
