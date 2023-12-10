"""Pre-Activation ResNet v2 with GroupNorm and Weight Standardization.

A PyTorch implementation of ResNetV2 adapted from the Google Big-Transfer (BiT) source code
at https://github.com/google-research/big_transfer to match timm interfaces. The BiT weights have
been included here as pretrained models from their original .NPZ checkpoints.

Additionally, supports non pre-activation bottleneck for use as a backbone for Vision Transfomers (ViT) and
extra padding support to allow porting of official Hybrid ResNet pretrained weights from
https://github.com/google-research/vision_transformer

Thanks to the Google team for the above two repositories and associated papers:
* Big Transfer (BiT): General Visual Representation Learning - https://arxiv.org/abs/1912.11370
* An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale - https://arxiv.org/abs/2010.11929
* Knowledge distillation: A good teacher is patient and consistent - https://arxiv.org/abs/2106.05237

Original copyright of Google code below, modifications by Ross Wightman, Copyright 2020.
"""
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict  # pylint: disable=g-importing-member
from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import GroupNormAct, BatchNormAct2d, EvoNorm2dS0, FilterResponseNormTlu2d, ClassifierHead, \
    DropPath, AvgPool2dSame, create_pool2d, StdConv2d, create_conv2d, get_act_layer, get_norm_act_layer, make_divisible
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq, named_apply, adapt_input_conv
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['ResNetV2']  # model_registry will add each entrypoint fn to this



class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.

    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(
            self,
            in_chs,
            out_chs=None,
            bottle_ratio=0.25,
            stride=1,
            dilation=1,
            first_dilation=None,
            groups=1,
            act_layer=None,
            conv_layer=None,
            norm_layer=None,
            proj_layer=None,
            drop_path_rate=0.,
    ):
        super().__init__()
        first_dilation = first_dilation or dilation
        conv_layer = conv_layer or StdConv2d
        norm_layer = norm_layer or partial(GroupNormAct, num_groups=32)
        out_chs = out_chs or in_chs
        mid_chs = make_divisible(out_chs * bottle_ratio)

        if proj_layer is not None:
            self.downsample = proj_layer(
                in_chs, out_chs, stride=stride, dilation=dilation, first_dilation=first_dilation, preact=True,
                conv_layer=conv_layer, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.norm1 = norm_layer(in_chs)
        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.norm2 = norm_layer(mid_chs)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        self.norm3 = norm_layer(mid_chs)
        self.conv3 = conv_layer(mid_chs, out_chs, 1)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def zero_init_last(self):
        nn.init.zeros_(self.conv3.weight)

    def forward(self, x):
        x_preact = self.norm1(x)

        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x_preact)

        # residual branch
        x = self.conv1(x_preact)
        x = self.conv2(self.norm2(x))
        x = self.conv3(self.norm3(x))
        x = self.drop_path(x)
        return x + shortcut


class Bottleneck(nn.Module):
    """Non Pre-activation bottleneck block, equiv to V1.5/V1b Bottleneck. Used for ViT.
    """
    def __init__(
            self,
            in_chs,
            out_chs=None,
            bottle_ratio=0.25,
            stride=1,
            dilation=1,
            first_dilation=None,
            groups=1,
            act_layer=None,
            conv_layer=None,
            norm_layer=None,
            proj_layer=None,
            drop_path_rate=0.,
    ):
        super().__init__()
        first_dilation = first_dilation or dilation
        act_layer = act_layer or nn.ReLU
        conv_layer = conv_layer or StdConv2d
        norm_layer = norm_layer or partial(GroupNormAct, num_groups=32)
        out_chs = out_chs or in_chs
        mid_chs = make_divisible(out_chs * bottle_ratio)

        if proj_layer is not None:
            self.downsample = proj_layer(
                in_chs, out_chs, stride=stride, dilation=dilation, preact=False,
                conv_layer=conv_layer, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.norm1 = norm_layer(mid_chs)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        self.norm2 = norm_layer(mid_chs)
        self.conv3 = conv_layer(mid_chs, out_chs, 1)
        self.norm3 = norm_layer(out_chs, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.act3 = act_layer(inplace=True)

    def zero_init_last(self):
        if getattr(self.norm3, 'weight', None) is not None:
            nn.init.zeros_(self.norm3.weight)

    def forward(self, x):
        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        # residual
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.drop_path(x)
        x = self.act3(x + shortcut)
        return x


class DownsampleConv(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            stride=1,
            dilation=1,
            first_dilation=None,
            preact=True,
            conv_layer=None,
            norm_layer=None,
    ):
        super(DownsampleConv, self).__init__()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=stride)
        self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        return self.norm(self.conv(x))


class DownsampleAvg(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            stride=1,
            dilation=1,
            first_dilation=None,
            preact=True,
            conv_layer=None,
            norm_layer=None,
    ):
        """ AvgPool Downsampling as in 'D' ResNet variants. This is not in RegNet space but I might experiment."""
        super(DownsampleAvg, self).__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=1)
        self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        return self.norm(self.conv(self.pool(x)))


class ResNetStage(nn.Module):
    """ResNet Stage."""
    def __init__(
            self,
            in_chs,
            out_chs,
            stride,
            dilation,
            depth,
            bottle_ratio=0.25,
            groups=1,
            avg_down=False,
            block_dpr=None,
            block_fn=PreActBottleneck,
            act_layer=None,
            conv_layer=None,
            norm_layer=None,
            **block_kwargs,
    ):
        super(ResNetStage, self).__init__()
        first_dilation = 1 if dilation in (1, 2) else 2
        layer_kwargs = dict(act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer)
        proj_layer = DownsampleAvg if avg_down else DownsampleConv
        prev_chs = in_chs
        self.blocks = nn.Sequential()
        for block_idx in range(depth):
            drop_path_rate = block_dpr[block_idx] if block_dpr else 0.
            stride = stride if block_idx == 0 else 1
            self.blocks.add_module(str(block_idx), block_fn(
                prev_chs,
                out_chs,
                stride=stride,
                dilation=dilation,
                bottle_ratio=bottle_ratio,
                groups=groups,
                first_dilation=first_dilation,
                proj_layer=proj_layer,
                drop_path_rate=drop_path_rate,
                **layer_kwargs,
                **block_kwargs,
            ))
            prev_chs = out_chs
            first_dilation = dilation
            proj_layer = None

    def forward(self, x):
        x = self.blocks(x)
        return x


def is_stem_deep(stem_type):
    return any([s in stem_type for s in ('deep', 'tiered')])


def create_resnetv2_stem(
        in_chs,
        out_chs=64,
        stem_type='',
        preact=True,
        conv_layer=StdConv2d,
        norm_layer=partial(GroupNormAct, num_groups=32),
):
    stem = OrderedDict()
    assert stem_type in ('', 'fixed', 'same', 'deep', 'deep_fixed', 'deep_same', 'tiered')

    # NOTE conv padding mode can be changed by overriding the conv_layer def
    if is_stem_deep(stem_type):
        # A 3 deep 3x3  conv stack as in ResNet V1D models
        if 'tiered' in stem_type:
            stem_chs = (3 * out_chs // 8, out_chs // 2)  # 'T' resnets in resnet.py
        else:
            stem_chs = (out_chs // 2, out_chs // 2)  # 'D' ResNets
        stem['conv1'] = conv_layer(in_chs, stem_chs[0], kernel_size=3, stride=2)
        stem['norm1'] = norm_layer(stem_chs[0])
        stem['conv2'] = conv_layer(stem_chs[0], stem_chs[1], kernel_size=3, stride=1)
        stem['norm2'] = norm_layer(stem_chs[1])
        stem['conv3'] = conv_layer(stem_chs[1], out_chs, kernel_size=3, stride=1)
        if not preact:
            stem['norm3'] = norm_layer(out_chs)
    else:
        # The usual 7x7 stem conv
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=7, stride=2)
        if not preact:
            stem['norm'] = norm_layer(out_chs)

    if 'fixed' in stem_type:
        # 'fixed' SAME padding approximation that is used in BiT models
        stem['pad'] = nn.ConstantPad2d(1, 0.)
        stem['pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    elif 'same' in stem_type:
        # full, input size based 'SAME' padding, used in ViT Hybrid model
        stem['pool'] = create_pool2d('max', kernel_size=3, stride=2, padding='same')
    else:
        # the usual PyTorch symmetric padding
        stem['pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    return nn.Sequential(stem)


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode.
    """

    def __init__(
            self,
            layers,
            channels=(256, 512, 1024, 2048),
            num_classes=1000,
            in_chans=3,
            global_pool='avg',
            output_stride=32,
            width_factor=1,
            stem_chs=64,
            stem_type='',
            avg_down=False,
            preact=True,
            act_layer=nn.ReLU,
            norm_layer=partial(GroupNormAct, num_groups=32),
            conv_layer=StdConv2d,
            drop_rate=0.,
            drop_path_rate=0.,
            zero_init_last=False,
    ):
        """
        Args:
            layers (List[int]) : number of layers in each block
            channels (List[int]) : number of channels in each block:
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            width_factor (int): channel (width) multiplication factor
            stem_chs (int): stem width (default: 64)
            stem_type (str): stem type (default: '' == 7x7)
            avg_down (bool): average pooling in residual downsampling (default: False)
            preact (bool): pre-activiation (default: True)
            act_layer (Union[str, nn.Module]): activation layer
            norm_layer (Union[str, nn.Module]): normalization layer
            conv_layer (nn.Module): convolution module
            drop_rate: classifier dropout rate (default: 0.)
            drop_path_rate: stochastic depth rate (default: 0.)
            zero_init_last: zero-init last weight in residual path (default: False)
        """
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        wf = width_factor
        norm_layer = get_norm_act_layer(norm_layer, act_layer=act_layer)
        act_layer = get_act_layer(act_layer)

        self.feature_info = []
        stem_chs = make_divisible(stem_chs * wf)
        self.stem = create_resnetv2_stem(
            in_chans,
            stem_chs,
            stem_type,
            preact,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
        )
        stem_feat = ('stem.conv3' if is_stem_deep(stem_type) else 'stem.conv') if preact else 'stem.norm'
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=stem_feat))

        prev_chs = stem_chs
        curr_stride = 4
        dilation = 1
        block_dprs = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(layers)).split(layers)]
        block_fn = PreActBottleneck if preact else Bottleneck
        self.stages = nn.Sequential()
        for stage_idx, (d, c, bdpr) in enumerate(zip(layers, channels, block_dprs)):
            out_chs = make_divisible(c * wf)
            stride = 1 if stage_idx == 0 else 2
            if curr_stride >= output_stride:
                dilation *= stride
                stride = 1
            stage = ResNetStage(
                prev_chs,
                out_chs,
                stride=stride,
                dilation=dilation,
                depth=d,
                avg_down=avg_down,
                act_layer=act_layer,
                conv_layer=conv_layer,
                norm_layer=norm_layer,
                block_dpr=bdpr,
                block_fn=block_fn,
            )
            prev_chs = out_chs
            curr_stride *= stride
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{stage_idx}')]
            self.stages.add_module(str(stage_idx), stage)

        self.num_features = prev_chs
        self.norm = norm_layer(self.num_features) if preact else nn.Identity()
        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=self.drop_rate,
            use_conv=True,
        )

        self.init_weights(zero_init_last=zero_init_last)
        self.grad_checkpointing = False

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix='resnet/'):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x, flatten=True)
        else:
            x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module: nn.Module, name: str = '', zero_init_last=True):
    if isinstance(module, nn.Linear) or ('head.fc' in name and isinstance(module, nn.Conv2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif zero_init_last and hasattr(module, 'zero_init_last'):
        module.zero_init_last()


@torch.no_grad()
def _load_weights(model: nn.Module, checkpoint_path: str, prefix: str = 'resnet/'):
    import numpy as np

    def t2p(conv_weights):
        """Possibly convert HWIO to OIHW."""
        if conv_weights.ndim == 4:
            conv_weights = conv_weights.transpose([3, 2, 0, 1])
        return torch.from_numpy(conv_weights)

    weights = np.load(checkpoint_path)
    stem_conv_w = adapt_input_conv(
        model.stem.conv.weight.shape[1], t2p(weights[f'{prefix}root_block/standardized_conv2d/kernel']))
    model.stem.conv.weight.copy_(stem_conv_w)
    model.norm.weight.copy_(t2p(weights[f'{prefix}group_norm/gamma']))
    model.norm.bias.copy_(t2p(weights[f'{prefix}group_norm/beta']))
    if isinstance(getattr(model.head, 'fc', None), nn.Conv2d) and \
            model.head.fc.weight.shape[0] == weights[f'{prefix}head/conv2d/kernel'].shape[-1]:
        model.head.fc.weight.copy_(t2p(weights[f'{prefix}head/conv2d/kernel']))
        model.head.fc.bias.copy_(t2p(weights[f'{prefix}head/conv2d/bias']))
    for i, (sname, stage) in enumerate(model.stages.named_children()):
        for j, (bname, block) in enumerate(stage.blocks.named_children()):
            cname = 'standardized_conv2d'
            block_prefix = f'{prefix}block{i + 1}/unit{j + 1:02d}/'
            block.conv1.weight.copy_(t2p(weights[f'{block_prefix}a/{cname}/kernel']))
            block.conv2.weight.copy_(t2p(weights[f'{block_prefix}b/{cname}/kernel']))
            block.conv3.weight.copy_(t2p(weights[f'{block_prefix}c/{cname}/kernel']))
            block.norm1.weight.copy_(t2p(weights[f'{block_prefix}a/group_norm/gamma']))
            block.norm2.weight.copy_(t2p(weights[f'{block_prefix}b/group_norm/gamma']))
            block.norm3.weight.copy_(t2p(weights[f'{block_prefix}c/group_norm/gamma']))
            block.norm1.bias.copy_(t2p(weights[f'{block_prefix}a/group_norm/beta']))
            block.norm2.bias.copy_(t2p(weights[f'{block_prefix}b/group_norm/beta']))
            block.norm3.bias.copy_(t2p(weights[f'{block_prefix}c/group_norm/beta']))
            if block.downsample is not None:
                w = weights[f'{block_prefix}a/proj/{cname}/kernel']
                block.downsample.conv.weight.copy_(t2p(w))


def _create_resnetv2(variant, pretrained=False, **kwargs):
    feature_cfg = dict(flatten_sequential=True)
    return build_model_with_cfg(
        ResNetV2, variant, pretrained,
        feature_cfg=feature_cfg,
        **kwargs,
    )


def _create_resnetv2_bit(variant, pretrained=False, **kwargs):
    return _create_resnetv2(
        variant,
        pretrained=pretrained,
        stem_type='fixed',
        conv_layer=partial(StdConv2d, eps=1e-8),
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    #  Paper: Knowledge distillation: A good teacher is patient and consistent - https://arxiv.org/abs/2106.05237
    'resnetv2_50x1_bit.goog_distilled_in1k': _cfg(
        hf_hub_id='timm/',
        interpolation='bicubic', custom_load=True),
    'resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        interpolation='bicubic', custom_load=True),
    'resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_384': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, interpolation='bicubic', custom_load=True),

    # pretrained on imagenet21k, finetuned on imagenet1k
    'resnetv2_50x1_bit.goog_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 448, 448), pool_size=(14, 14), crop_pct=1.0, custom_load=True),
    'resnetv2_50x3_bit.goog_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 448, 448), pool_size=(14, 14), crop_pct=1.0, custom_load=True),
    'resnetv2_101x1_bit.goog_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 448, 448), pool_size=(14, 14), crop_pct=1.0, custom_load=True),
    'resnetv2_101x3_bit.goog_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 448, 448), pool_size=(14, 14), crop_pct=1.0, custom_load=True),
    'resnetv2_152x2_bit.goog_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 448, 448), pool_size=(14, 14), crop_pct=1.0, custom_load=True),
    'resnetv2_152x4_bit.goog_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 480, 480), pool_size=(15, 15), crop_pct=1.0, custom_load=True),  # only one at 480x480?

    # trained on imagenet-21k
    'resnetv2_50x1_bit.goog_in21k': _cfg(
        hf_hub_id='timm/',
        num_classes=21843, custom_load=True),
    'resnetv2_50x3_bit.goog_in21k': _cfg(
        hf_hub_id='timm/',
        num_classes=21843, custom_load=True),
    'resnetv2_101x1_bit.goog_in21k': _cfg(
        hf_hub_id='timm/',
        num_classes=21843, custom_load=True),
    'resnetv2_101x3_bit.goog_in21k': _cfg(
        hf_hub_id='timm/',
        num_classes=21843, custom_load=True),
    'resnetv2_152x2_bit.goog_in21k': _cfg(
        hf_hub_id='timm/',
        num_classes=21843, custom_load=True),
    'resnetv2_152x4_bit.goog_in21k': _cfg(
        hf_hub_id='timm/',
        num_classes=21843, custom_load=True),

    'resnetv2_50.a1h_in1k': _cfg(
        hf_hub_id='timm/',
        interpolation='bicubic', crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'resnetv2_50d.untrained': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),
    'resnetv2_50t.untrained': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),
    'resnetv2_101.a1h_in1k': _cfg(
        hf_hub_id='timm/',
        interpolation='bicubic', crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'resnetv2_101d.untrained': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),
    'resnetv2_152.untrained': _cfg(
        interpolation='bicubic'),
    'resnetv2_152d.untrained': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),

    'resnetv2_50d_gn.ah_in1k': _cfg(
        hf_hub_id='timm/',
        interpolation='bicubic', first_conv='stem.conv1',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'resnetv2_50d_evos.ah_in1k': _cfg(
        hf_hub_id='timm/',
        interpolation='bicubic', first_conv='stem.conv1',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'resnetv2_50d_frn.untrained': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),
})


@register_model
def resnetv2_50x1_bit(pretrained=False, **kwargs) -> ResNetV2:
    return _create_resnetv2_bit(
        'resnetv2_50x1_bit', pretrained=pretrained, layers=[3, 4, 6, 3], width_factor=1, **kwargs)


@register_model
def resnetv2_50x3_bit(pretrained=False, **kwargs) -> ResNetV2:
    return _create_resnetv2_bit(
        'resnetv2_50x3_bit', pretrained=pretrained, layers=[3, 4, 6, 3], width_factor=3, **kwargs)


@register_model
def resnetv2_101x1_bit(pretrained=False, **kwargs) -> ResNetV2:
    return _create_resnetv2_bit(
        'resnetv2_101x1_bit', pretrained=pretrained, layers=[3, 4, 23, 3], width_factor=1, **kwargs)


@register_model
def resnetv2_101x3_bit(pretrained=False, **kwargs) -> ResNetV2:
    return _create_resnetv2_bit(
        'resnetv2_101x3_bit', pretrained=pretrained, layers=[3, 4, 23, 3], width_factor=3, **kwargs)


@register_model
def resnetv2_152x2_bit(pretrained=False, **kwargs) -> ResNetV2:
    return _create_resnetv2_bit(
        'resnetv2_152x2_bit', pretrained=pretrained, layers=[3, 8, 36, 3], width_factor=2, **kwargs)


@register_model
def resnetv2_152x4_bit(pretrained=False, **kwargs) -> ResNetV2:
    return _create_resnetv2_bit(
        'resnetv2_152x4_bit', pretrained=pretrained, layers=[3, 8, 36, 3], width_factor=4, **kwargs)


@register_model
def resnetv2_50(pretrained=False, **kwargs) -> ResNetV2:
    model_args = dict(layers=[3, 4, 6, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d)
    return _create_resnetv2('resnetv2_50', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def resnetv2_50d(pretrained=False, **kwargs) -> ResNetV2:
    model_args = dict(
        layers=[3, 4, 6, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', avg_down=True)
    return _create_resnetv2('resnetv2_50d', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def resnetv2_50t(pretrained=False, **kwargs) -> ResNetV2:
    model_args = dict(
        layers=[3, 4, 6, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='tiered', avg_down=True)
    return _create_resnetv2('resnetv2_50t', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def resnetv2_101(pretrained=False, **kwargs) -> ResNetV2:
    model_args = dict(layers=[3, 4, 23, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d)
    return _create_resnetv2('resnetv2_101', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def resnetv2_101d(pretrained=False, **kwargs) -> ResNetV2:
    model_args = dict(
        layers=[3, 4, 23, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', avg_down=True)
    return _create_resnetv2('resnetv2_101d', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def resnetv2_152(pretrained=False, **kwargs) -> ResNetV2:
    model_args = dict(layers=[3, 8, 36, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d)
    return _create_resnetv2('resnetv2_152', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def resnetv2_152d(pretrained=False, **kwargs) -> ResNetV2:
    model_args = dict(
        layers=[3, 8, 36, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', avg_down=True)
    return _create_resnetv2('resnetv2_152d', pretrained=pretrained, **dict(model_args, **kwargs))


# Experimental configs (may change / be removed)

@register_model
def resnetv2_50d_gn(pretrained=False, **kwargs) -> ResNetV2:
    model_args = dict(
        layers=[3, 4, 6, 3], conv_layer=create_conv2d, norm_layer=GroupNormAct,
        stem_type='deep', avg_down=True)
    return _create_resnetv2('resnetv2_50d_gn', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def resnetv2_50d_evos(pretrained=False, **kwargs) -> ResNetV2:
    model_args = dict(
        layers=[3, 4, 6, 3], conv_layer=create_conv2d, norm_layer=EvoNorm2dS0,
        stem_type='deep', avg_down=True)
    return _create_resnetv2('resnetv2_50d_evos', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def resnetv2_50d_frn(pretrained=False, **kwargs) -> ResNetV2:
    model_args = dict(
        layers=[3, 4, 6, 3], conv_layer=create_conv2d, norm_layer=FilterResponseNormTlu2d,
        stem_type='deep', avg_down=True)
    return _create_resnetv2('resnetv2_50d_frn', pretrained=pretrained, **dict(model_args, **kwargs))


register_model_deprecations(__name__, {
    'resnetv2_50x1_bitm': 'resnetv2_50x1_bit.goog_in21k_ft_in1k',
    'resnetv2_50x3_bitm': 'resnetv2_50x3_bit.goog_in21k_ft_in1k',
    'resnetv2_101x1_bitm': 'resnetv2_101x1_bit.goog_in21k_ft_in1k',
    'resnetv2_101x3_bitm': 'resnetv2_101x3_bit.goog_in21k_ft_in1k',
    'resnetv2_152x2_bitm': 'resnetv2_152x2_bit.goog_in21k_ft_in1k',
    'resnetv2_152x4_bitm': 'resnetv2_152x4_bit.goog_in21k_ft_in1k',
    'resnetv2_50x1_bitm_in21k': 'resnetv2_50x1_bit.goog_in21k',
    'resnetv2_50x3_bitm_in21k': 'resnetv2_50x3_bit.goog_in21k',
    'resnetv2_101x1_bitm_in21k': 'resnetv2_101x1_bit.goog_in21k',
    'resnetv2_101x3_bitm_in21k': 'resnetv2_101x3_bit.goog_in21k',
    'resnetv2_152x2_bitm_in21k': 'resnetv2_152x2_bit.goog_in21k',
    'resnetv2_152x4_bitm_in21k': 'resnetv2_152x4_bit.goog_in21k',
    'resnetv2_50x1_bit_distilled': 'resnetv2_50x1_bit.goog_distilled_in1k',
    'resnetv2_152x2_bit_teacher': 'resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k',
    'resnetv2_152x2_bit_teacher_384': 'resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_384',
})
