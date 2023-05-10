"""
TResNet: High Performance GPU-Dedicated Architecture
https://arxiv.org/pdf/2003.13630.pdf

Original model: https://github.com/mrT23/TResNet

"""
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

from timm.layers import SpaceToDepth, BlurPool2d, ClassifierHead, SEModule,\
    ConvNormActAa, ConvNormAct, DropPath
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs, register_model_deprecations

__all__ = ['TResNet']  # model_registry will add each entrypoint fn to this


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            use_se=True,
            aa_layer=None,
            drop_path_rate=0.
    ):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        act_layer = partial(nn.LeakyReLU, negative_slope=1e-3)

        if stride == 1:
            self.conv1 = ConvNormAct(inplanes, planes, kernel_size=3, stride=1, act_layer=act_layer)
        else:
            self.conv1 = ConvNormActAa(
                inplanes, planes, kernel_size=3, stride=2, act_layer=act_layer, aa_layer=aa_layer)

        self.conv2 = ConvNormAct(planes, planes, kernel_size=3, stride=1, apply_act=False, act_layer=None)
        self.act = nn.ReLU(inplace=True)

        rd_chs = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, rd_channels=rd_chs) if use_se else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out = self.drop_path(out) + shortcut
        out = self.act(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            use_se=True,
            act_layer=None,
            aa_layer=None,
            drop_path_rate=0.,
    ):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.stride = stride
        act_layer = act_layer or partial(nn.LeakyReLU, negative_slope=1e-3)

        self.conv1 = ConvNormAct(
            inplanes, planes, kernel_size=1, stride=1, act_layer=act_layer)
        if stride == 1:
            self.conv2 = ConvNormAct(
                planes, planes, kernel_size=3, stride=1, act_layer=act_layer)
        else:
            self.conv2 = ConvNormActAa(
                planes, planes, kernel_size=3, stride=2, act_layer=act_layer, aa_layer=aa_layer)

        reduction_chs = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, rd_channels=reduction_chs) if use_se else None

        self.conv3 = ConvNormAct(
            planes, planes * self.expansion, kernel_size=1, stride=1, apply_act=False, act_layer=None)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out = self.conv3(out)
        out = self.drop_path(out) + shortcut
        out = self.act(out)
        return out


class TResNet(nn.Module):
    def __init__(
            self,
            layers,
            in_chans=3,
            num_classes=1000,
            width_factor=1.0,
            v2=False,
            global_pool='fast',
            drop_rate=0.,
            drop_path_rate=0.,
    ):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        super(TResNet, self).__init__()

        aa_layer = BlurPool2d
        act_layer = nn.LeakyReLU

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        if v2:
            self.inplanes = self.inplanes // 8 * 8
            self.planes = self.planes // 8 * 8

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(layers)).split(layers)]
        conv1 = ConvNormAct(in_chans * 16, self.planes, stride=1, kernel_size=3, act_layer=act_layer)
        layer1 = self._make_layer(
            Bottleneck if v2 else BasicBlock,
            self.planes, layers[0], stride=1, use_se=True, aa_layer=aa_layer, drop_path_rate=dpr[0])
        layer2 = self._make_layer(
            Bottleneck if v2 else BasicBlock,
            self.planes * 2, layers[1], stride=2, use_se=True, aa_layer=aa_layer, drop_path_rate=dpr[1])
        layer3 = self._make_layer(
            Bottleneck,
            self.planes * 4, layers[2], stride=2, use_se=True, aa_layer=aa_layer, drop_path_rate=dpr[2])
        layer4 = self._make_layer(
            Bottleneck,
            self.planes * 8, layers[3], stride=2, use_se=False, aa_layer=aa_layer, drop_path_rate=dpr[3])

        # body
        self.body = nn.Sequential(OrderedDict([
            ('s2d', SpaceToDepth()),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4),
        ]))

        self.feature_info = [
            dict(num_chs=self.planes, reduction=2, module=''),  # Not with S2D?
            dict(num_chs=self.planes * (Bottleneck.expansion if v2 else 1), reduction=4, module='body.layer1'),
            dict(num_chs=self.planes * 2 * (Bottleneck.expansion if v2 else 1), reduction=8, module='body.layer2'),
            dict(num_chs=self.planes * 4 * Bottleneck.expansion, reduction=16, module='body.layer3'),
            dict(num_chs=self.planes * 8 * Bottleneck.expansion, reduction=32, module='body.layer4'),
        ]

        # head
        self.num_features = (self.planes * 8) * Bottleneck.expansion
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

        # model initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.conv2.bn.weight)
            if isinstance(m, Bottleneck):
                nn.init.zeros_(m.conv3.bn.weight)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, aa_layer=None, drop_path_rate=0.):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [ConvNormAct(
                self.inplanes, planes * block.expansion, kernel_size=1, stride=1, apply_act=False, act_layer=None)]
            downsample = nn.Sequential(*layers)

        layers = []
        for i in range(blocks):
            layers.append(block(
                self.inplanes,
                planes,
                stride=stride if i == 0 else 1,
                downsample=downsample if i == 0 else None,
                use_se=use_se,
                aa_layer=aa_layer,
                drop_path_rate=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
            ))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^body\.conv1', blocks=r'^body\.layer(\d+)' if coarse else r'^body\.layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_features(self, x):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = self.body.s2d(x)
            x = self.body.conv1(x)
            x = checkpoint_seq([
                self.body.layer1,
                self.body.layer2,
                self.body.layer3,
                self.body.layer4],
                x, flatten=True)
        else:
            x = self.body(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    if 'body.conv1.conv.weight' in state_dict:
        return state_dict

    import re
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    out_dict = {}
    for k, v in state_dict.items():
        k = re.sub(r'conv(\d+)\.0.0', lambda x: f'conv{int(x.group(1))}.conv', k)
        k = re.sub(r'conv(\d+)\.0.1', lambda x: f'conv{int(x.group(1))}.bn', k)
        k = re.sub(r'conv(\d+)\.0', lambda x: f'conv{int(x.group(1))}.conv', k)
        k = re.sub(r'conv(\d+)\.1', lambda x: f'conv{int(x.group(1))}.bn', k)
        k = re.sub(r'downsample\.(\d+)\.0', lambda x: f'downsample.{int(x.group(1))}.conv', k)
        k = re.sub(r'downsample\.(\d+)\.1', lambda x: f'downsample.{int(x.group(1))}.bn', k)
        if k.endswith('bn.weight'):
            # convert weight from inplace_abn to batchnorm
            v = v.abs().add(1e-5)
        out_dict[k] = v
    return out_dict


def _create_tresnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        TResNet,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(1, 2, 3, 4), flatten_sequential=True),
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': (0., 0., 0.), 'std': (1., 1., 1.),
        'first_conv': 'body.conv1.conv', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'tresnet_m.miil_in21k_ft_in1k': _cfg(hf_hub_id='timm/'),
    'tresnet_m.miil_in21k': _cfg(hf_hub_id='timm/', num_classes=11221),
    'tresnet_m.miil_in1k': _cfg(hf_hub_id='timm/'),
    'tresnet_l.miil_in1k': _cfg(hf_hub_id='timm/'),
    'tresnet_xl.miil_in1k': _cfg(hf_hub_id='timm/'),
    'tresnet_m.miil_in1k_448': _cfg(
        input_size=(3, 448, 448), pool_size=(14, 14),
        hf_hub_id='timm/'),
    'tresnet_l.miil_in1k_448': _cfg(
        input_size=(3, 448, 448), pool_size=(14, 14),
        hf_hub_id='timm/'),
    'tresnet_xl.miil_in1k_448': _cfg(
        input_size=(3, 448, 448), pool_size=(14, 14),
        hf_hub_id='timm/'),

    'tresnet_v2_l.miil_in21k_ft_in1k': _cfg(hf_hub_id='timm/'),
    'tresnet_v2_l.miil_in21k': _cfg(hf_hub_id='timm/', num_classes=11221),
})


@register_model
def tresnet_m(pretrained=False, **kwargs) -> TResNet:
    model_kwargs = dict(layers=[3, 4, 11, 3], **kwargs)
    return _create_tresnet('tresnet_m', pretrained=pretrained, **model_kwargs)


@register_model
def tresnet_l(pretrained=False, **kwargs) -> TResNet:
    model_kwargs = dict(layers=[4, 5, 18, 3], width_factor=1.2, **kwargs)
    return _create_tresnet('tresnet_l', pretrained=pretrained, **model_kwargs)


@register_model
def tresnet_xl(pretrained=False, **kwargs) -> TResNet:
    model_kwargs = dict(layers=[4, 5, 24, 3], width_factor=1.3, **kwargs)
    return _create_tresnet('tresnet_xl', pretrained=pretrained, **model_kwargs)


@register_model
def tresnet_v2_l(pretrained=False, **kwargs) -> TResNet:
    model_kwargs = dict(layers=[3, 4, 23, 3], width_factor=1.0, v2=True, **kwargs)
    return _create_tresnet('tresnet_v2_l', pretrained=pretrained, **model_kwargs)


register_model_deprecations(__name__, {
    'tresnet_m_miil_in21k': 'tresnet_m.miil_in21k',
    'tresnet_m_448': 'tresnet_m.miil_in1k_448',
    'tresnet_l_448': 'tresnet_l.miil_in1k_448',
    'tresnet_xl_448': 'tresnet_xl.miil_in1k_448',
})