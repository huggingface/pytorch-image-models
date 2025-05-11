"""
An implementation of RepGhostNet Model as defined in:
RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization. https://arxiv.org/abs/2211.06088

Original implementation: https://github.com/ChengpengChen/RepGhost
"""
import copy
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d, Linear, make_divisible
from ._builder import build_model_with_cfg
from ._efficientnet_blocks import SqueezeExcite, ConvBnAct
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['RepGhostNet']


_SE_LAYER = partial(SqueezeExcite, gate_layer='hard_sigmoid', rd_round_fn=partial(make_divisible, divisor=4))


class RepGhostModule(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=1,
            dw_size=3,
            stride=1,
            relu=True,
            reparam=True,
    ):
        super(RepGhostModule, self).__init__()
        self.out_chs = out_chs
        init_chs = out_chs
        new_chs = out_chs

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_chs, init_chs, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_chs),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

        fusion_conv = []
        fusion_bn = []
        if reparam:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.BatchNorm2d(init_chs))

        self.fusion_conv = nn.Sequential(*fusion_conv)
        self.fusion_bn = nn.Sequential(*fusion_bn)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_chs, new_chs, dw_size, 1, dw_size//2, groups=init_chs, bias=False),
            nn.BatchNorm2d(new_chs),
            # nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        self.relu = nn.ReLU(inplace=False) if relu else nn.Identity()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            x2 = x2 + bn(conv(x1))
        return self.relu(x2)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            kernel = torch.ones(in_channels, 1, 1, 1, device=device)

        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels).to(kernel.device)

    def switch_to_deploy(self):
        if len(self.fusion_conv) == 0 and len(self.fusion_bn) == 0:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cheap_operation = nn.Conv2d(
            in_channels=self.cheap_operation[0].in_channels,
            out_channels=self.cheap_operation[0].out_channels,
            kernel_size=self.cheap_operation[0].kernel_size,
            padding=self.cheap_operation[0].padding,
            dilation=self.cheap_operation[0].dilation,
            groups=self.cheap_operation[0].groups,
            bias=True)
        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias
        self.__delattr__('fusion_conv')
        self.__delattr__('fusion_bn')
        self.fusion_conv = []
        self.fusion_bn = []

    def reparameterize(self):
        self.switch_to_deploy()


class RepGhostBottleneck(nn.Module):
    """ RepGhost bottleneck w/ optional SE"""

    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            dw_kernel_size=3,
            stride=1,
            act_layer=nn.ReLU,
            se_ratio=0.,
            reparam=True,
    ):
        super(RepGhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = RepGhostModule(in_chs, mid_chs, relu=True, reparam=reparam)

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
        self.ghost2 = RepGhostModule(mid_chs, out_chs, relu=False, reparam=reparam)
        
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


class RepGhostNet(nn.Module):
    def __init__(
            self,
            cfgs,
            num_classes=1000,
            width=1.0,
            in_chans=3,
            output_stride=32,
            global_pool='avg',
            drop_rate=0.2,
            reparam=True,
    ):
        super(RepGhostNet, self).__init__()
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
        block = RepGhostBottleneck
        stage_idx = 0
        net_stride = 2
        for cfg in self.cfgs:
            layers = []
            s = 1
            for k, exp_size, c, se_ratio, s in cfg:
                out_chs = make_divisible(c * width, 4)
                mid_chs = make_divisible(exp_size * width, 4)
                layers.append(block(prev_chs, mid_chs, out_chs, k, s, se_ratio=se_ratio, reparam=reparam))
                prev_chs = out_chs
            if s > 1:
                net_stride *= 2
                self.feature_info.append(dict(
                    num_chs=prev_chs, reduction=net_stride, module=f'blocks.{stage_idx}'))
            stages.append(nn.Sequential(*layers))
            stage_idx += 1

        out_chs = make_divisible(exp_size * width * 2, 4)
        stages.append(nn.Sequential(ConvBnAct(prev_chs, out_chs, 1)))
        self.pool_dim = prev_chs = out_chs
        
        self.blocks = nn.Sequential(*stages)        

        # building last several layers
        self.num_features = prev_chs
        self.head_hidden_size = out_chs = 1280
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = nn.Conv2d(prev_chs, out_chs, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(out_chs, num_classes) if num_classes > 0 else nn.Identity()

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
    def get_classifier(self) -> nn.Module:
        return self.classifier

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            # NOTE: cannot meaningfully change pooling of efficient head after creation
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.head_hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def convert_to_deploy(self):
        repghost_model_convert(self, do_copy=False)


def repghost_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    """
    taken from from https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


def _create_repghostnet(variant, width=1.0, pretrained=False, **kwargs):
    """
    Constructs a RepGhostNet model
    """
    cfgs = [
        # k, t, c, SE, s 
        # stage1
        [[3,  8,  16, 0, 1]],
        # stage2
        [[3,  24,  24, 0, 2]],
        [[3,  36,  24, 0, 1]],
        # stage3
        [[5,  36,  40, 0.25, 2]],
        [[5, 60,  40, 0.25, 1]],
        # stage4
        [[3, 120,  80, 0, 2]],
        [[3, 100,  80, 0, 1],
         [3, 120,  80, 0, 1],
         [3, 120,  80, 0, 1],
         [3, 240, 112, 0.25, 1],
         [3, 336, 112, 0.25, 1]
        ],
        # stage5
        [[5, 336, 160, 0.25, 2]],
        [[5, 480, 160, 0, 1],
         [5, 480, 160, 0.25, 1],
         [5, 480, 160, 0, 1],
         [5, 480, 160, 0.25, 1]
        ]
    ]
    model_kwargs = dict(
        cfgs=cfgs,
        width=width,
        **kwargs,
    )
    return build_model_with_cfg(
        RepGhostNet,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True),
        **model_kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'repghostnet_050.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/ChengpengChen/RepGhost/releases/download/RepGhost/repghostnet_0_5x_43M_66.95.pth.tar'
    ),
    'repghostnet_058.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/ChengpengChen/RepGhost/releases/download/RepGhost/repghostnet_0_58x_60M_68.94.pth.tar'
    ),
    'repghostnet_080.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/ChengpengChen/RepGhost/releases/download/RepGhost/repghostnet_0_8x_96M_72.24.pth.tar'
    ),
    'repghostnet_100.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/ChengpengChen/RepGhost/releases/download/RepGhost/repghostnet_1_0x_142M_74.22.pth.tar'
    ),
    'repghostnet_111.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/ChengpengChen/RepGhost/releases/download/RepGhost/repghostnet_1_11x_170M_75.07.pth.tar'
    ),
    'repghostnet_130.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/ChengpengChen/RepGhost/releases/download/RepGhost/repghostnet_1_3x_231M_76.37.pth.tar'
    ),
    'repghostnet_150.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/ChengpengChen/RepGhost/releases/download/RepGhost/repghostnet_1_5x_301M_77.45.pth.tar'
    ),
    'repghostnet_200.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/ChengpengChen/RepGhost/releases/download/RepGhost/repghostnet_2_0x_516M_78.81.pth.tar'
    ),
})


@register_model
def repghostnet_050(pretrained=False, **kwargs) -> RepGhostNet:
    """ RepGhostNet-0.5x """
    model = _create_repghostnet('repghostnet_050', width=0.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def repghostnet_058(pretrained=False, **kwargs) -> RepGhostNet:
    """ RepGhostNet-0.58x """
    model = _create_repghostnet('repghostnet_058', width=0.58, pretrained=pretrained, **kwargs)
    return model


@register_model
def repghostnet_080(pretrained=False, **kwargs) -> RepGhostNet:
    """ RepGhostNet-0.8x """
    model = _create_repghostnet('repghostnet_080', width=0.8, pretrained=pretrained, **kwargs)
    return model


@register_model
def repghostnet_100(pretrained=False, **kwargs) -> RepGhostNet:
    """ RepGhostNet-1.0x """
    model = _create_repghostnet('repghostnet_100', width=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def repghostnet_111(pretrained=False, **kwargs) -> RepGhostNet:
    """ RepGhostNet-1.11x """
    model = _create_repghostnet('repghostnet_111', width=1.11, pretrained=pretrained, **kwargs)
    return model

@register_model
def repghostnet_130(pretrained=False, **kwargs) -> RepGhostNet:
    """ RepGhostNet-1.3x """
    model = _create_repghostnet('repghostnet_130', width=1.3, pretrained=pretrained, **kwargs)
    return model


@register_model
def repghostnet_150(pretrained=False, **kwargs) -> RepGhostNet:
    """ RepGhostNet-1.5x """
    model = _create_repghostnet('repghostnet_150', width=1.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def repghostnet_200(pretrained=False, **kwargs) -> RepGhostNet:
    """ RepGhostNet-2.0x """
    model = _create_repghostnet('repghostnet_200', width=2.0, pretrained=pretrained, **kwargs)
    return model
