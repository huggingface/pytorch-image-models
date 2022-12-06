"""
An implementation of GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations. https://arxiv.org/abs/1911.11907
The train script of the model is similar to that of MobileNetV3
Original model: https://github.com/huawei-noah/CV-backbones/tree/master/ghostnet_pytorch
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d, Linear, make_divisible
from ._builder import build_model_with_cfg
from ._efficientnet_blocks import SqueezeExcite, ConvBnAct
from ._manipulate import checkpoint_seq
from ._registry import register_model

__all__ = ['GhostNet']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'ghostnet_050': _cfg(url=''),
    'ghostnet_100': _cfg(
        url='https://github.com/huawei-noah/CV-backbones/releases/download/ghostnet_pth/ghostnet_1x.pth'),
    'ghostnet_130': _cfg(url=''),
}


_SE_LAYER = partial(SqueezeExcite, gate_layer='hard_sigmoid', rd_round_fn=partial(make_divisible, divisor=4))


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs, mid_chs, dw_kernel_size, stride=stride,
                padding=(dw_kernel_size-1)//2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        else:
            self.conv_dw = None
            self.bn_dw = None

        # Squeeze-and-excitation
        self.se = _SE_LAYER(mid_chs, rd_ratio=se_ratio) if has_se else None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs, in_chs, dw_kernel_size, stride=stride,
                    padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        shortcut = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.conv_dw is not None:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(shortcut)
        return x


class GhostNet(nn.Module):
    def __init__(
            self, cfgs, num_classes=1000, width=1.0, in_chans=3, output_stride=32, global_pool='avg', drop_rate=0.2):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        assert output_stride == 32, 'only output_stride==32 is valid, dilation not supported'
        self.cfgs = cfgs
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []

        # building first layer
        stem_chs = make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(in_chans, stem_chs, 3, 2, 1, bias=False)
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=f'conv_stem'))
        self.bn1 = nn.BatchNorm2d(stem_chs)
        self.act1 = nn.ReLU(inplace=True)
        prev_chs = stem_chs

        # building inverted residual blocks
        stages = nn.ModuleList([])
        block = GhostBottleneck
        stage_idx = 0
        net_stride = 2
        for cfg in self.cfgs:
            layers = []
            s = 1
            for k, exp_size, c, se_ratio, s in cfg:
                out_chs = make_divisible(c * width, 4)
                mid_chs = make_divisible(exp_size * width, 4)
                layers.append(block(prev_chs, mid_chs, out_chs, k, s, se_ratio=se_ratio))
                prev_chs = out_chs
            if s > 1:
                net_stride *= 2
                self.feature_info.append(dict(
                    num_chs=prev_chs, reduction=net_stride, module=f'blocks.{stage_idx}'))
            stages.append(nn.Sequential(*layers))
            stage_idx += 1

        out_chs = make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(prev_chs, out_chs, 1)))
        self.pool_dim = prev_chs = out_chs
        
        self.blocks = nn.Sequential(*stages)        

        # building last several layers
        self.num_features = out_chs = 1280
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = nn.Conv2d(prev_chs, out_chs, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(out_chs, num_classes) if num_classes > 0 else nn.Identity()

        # FIXME init

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^conv_stem|bn1',
            blocks=[
                (r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)', None),
                (r'conv_head', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.pool_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x):
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_ghostnet(variant, width=1.0, pretrained=False, **kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s 
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    model_kwargs = dict(
        cfgs=cfgs,
        width=width,
        **kwargs,
    )
    return build_model_with_cfg(
        GhostNet, variant, pretrained,
        feature_cfg=dict(flatten_sequential=True),
        **model_kwargs)


@register_model
def ghostnet_050(pretrained=False, **kwargs):
    """ GhostNet-0.5x """
    model = _create_ghostnet('ghostnet_050', width=0.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def ghostnet_100(pretrained=False, **kwargs):
    """ GhostNet-1.0x """
    model = _create_ghostnet('ghostnet_100', width=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def ghostnet_130(pretrained=False, **kwargs):
    """ GhostNet-1.3x """
    model = _create_ghostnet('ghostnet_130', width=1.3, pretrained=pretrained, **kwargs)
    return model
