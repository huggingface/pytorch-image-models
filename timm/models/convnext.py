""" ConvNeXt

Papers:
* `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf
@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2022},
}

* `ConvNeXt-V2 - Co-designing and Scaling ConvNets with Masked Autoencoders` - https://arxiv.org/abs/2301.00808
@article{Woo2023ConvNeXtV2,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon and Saining Xie},
  year={2023},
  journal={arXiv preprint arXiv:2301.00808},
}

Original code and weights from:
* https://github.com/facebookresearch/ConvNeXt, original copyright below
* https://github.com/facebookresearch/ConvNeXt-V2, original copyright below

Model defs atto, femto, pico, nano and _ols / _hnf variants are timm originals.

Modifications and additions for timm hacked together by / Copyright 2022, Ross Wightman
"""
# ConvNeXt
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the MIT license

# ConvNeXt-V2
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree (Attribution-NonCommercial 4.0 International (CC BY-NC 4.0))
# No code was used directly from ConvNeXt-V2, however the weights are CC BY-NC 4.0 so beware if using commercially.

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import trunc_normal_, AvgPool2dSame, DropPath, Mlp, GlobalResponseNormMlp, \
    LayerNorm2d, LayerNorm, create_conv2d, get_act_layer, make_divisible, to_ntuple
from timm.layers import NormMlpClassifierHead, ClassifierHead
from ._builder import build_model_with_cfg
from ._manipulate import named_apply, checkpoint_seq
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['ConvNeXt']  # model_registry will add each entrypoint fn to this


class Downsample(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, dilation=1):
        super().__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()

        if in_chs != out_chs:
            self.conv = create_conv2d(in_chs, out_chs, 1, stride=1)
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: Optional[int] = None,
            kernel_size: int = 7,
            stride: int = 1,
            dilation: Union[int, Tuple[int, int]] = (1, 1),
            mlp_ratio: float = 4,
            conv_mlp: bool = False,
            conv_bias: bool = True,
            use_grn: bool = False,
            ls_init_value: Optional[float] = 1e-6,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Optional[Callable] = None,
            drop_path: float = 0.,
    ):
        """

        Args:
            in_chs: Block input channels.
            out_chs: Block output channels (same as in_chs if None).
            kernel_size: Depthwise convolution kernel size.
            stride: Stride of depthwise convolution.
            dilation: Tuple specifying input and output dilation of block.
            mlp_ratio: MLP expansion ratio.
            conv_mlp: Use 1x1 convolutions for MLP and a NCHW compatible norm layer if True.
            conv_bias: Apply bias for all convolution (linear) layers.
            use_grn: Use GlobalResponseNorm in MLP (from ConvNeXt-V2)
            ls_init_value: Layer-scale init values, layer-scale applied if not None.
            act_layer: Activation layer.
            norm_layer: Normalization layer (defaults to LN if not specified).
            drop_path: Stochastic depth probability.
        """
        super().__init__()
        out_chs = out_chs or in_chs
        dilation = to_ntuple(2)(dilation)
        act_layer = get_act_layer(act_layer)
        if not norm_layer:
            norm_layer = LayerNorm2d if conv_mlp else LayerNorm
        mlp_layer = partial(GlobalResponseNormMlp if use_grn else Mlp, use_conv=conv_mlp)
        self.use_conv_mlp = conv_mlp
        self.conv_dw = create_conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation[0],
            depthwise=True,
            bias=conv_bias,
        )
        self.norm = norm_layer(out_chs)
        self.mlp = mlp_layer(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(out_chs)) if ls_init_value is not None else None
        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = Downsample(in_chs, out_chs, stride=stride, dilation=dilation[0])
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        x = self.drop_path(x) + self.shortcut(shortcut)
        return x


class ConvNeXtStage(nn.Module):

    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=7,
            stride=2,
            depth=2,
            dilation=(1, 1),
            drop_path_rates=None,
            ls_init_value=1.0,
            conv_mlp=False,
            conv_bias=True,
            use_grn=False,
            act_layer='gelu',
            norm_layer=None,
            norm_layer_cl=None
    ):
        super().__init__()
        self.grad_checkpointing = False

        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = 'same' if dilation[1] > 1 else 0  # same padding needed if dilation used
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                create_conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=ds_ks,
                    stride=stride,
                    dilation=dilation[0],
                    padding=pad,
                    bias=conv_bias,
                ),
            )
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(ConvNeXtBlock(
                in_chs=in_chs,
                out_chs=out_chs,
                kernel_size=kernel_size,
                dilation=dilation[1],
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                conv_bias=conv_bias,
                use_grn=use_grn,
                act_layer=act_layer,
                norm_layer=norm_layer if conv_mlp else norm_layer_cl,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
    """

    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            output_stride: int = 32,
            depths: Tuple[int, ...] = (3, 3, 9, 3),
            dims: Tuple[int, ...] = (96, 192, 384, 768),
            kernel_sizes: Union[int, Tuple[int, ...]] = 7,
            ls_init_value: Optional[float] = 1e-6,
            stem_type: str = 'patch',
            patch_size: int = 4,
            head_init_scale: float = 1.,
            head_norm_first: bool = False,
            head_hidden_size: Optional[int] = None,
            conv_mlp: bool = False,
            conv_bias: bool = True,
            use_grn: bool = False,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Optional[Union[str, Callable]] = None,
            norm_eps: Optional[float] = None,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
    ):
        """
        Args:
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            global_pool: Global pooling type.
            output_stride: Output stride of network, one of (8, 16, 32).
            depths: Number of blocks at each stage.
            dims: Feature dimension at each stage.
            kernel_sizes: Depthwise convolution kernel-sizes for each stage.
            ls_init_value: Init value for Layer Scale, disabled if None.
            stem_type: Type of stem.
            patch_size: Stem patch size for patch stem.
            head_init_scale: Init scaling value for classifier weights and biases.
            head_norm_first: Apply normalization before global pool + head.
            head_hidden_size: Size of MLP hidden layer in head if not None and head_norm_first == False.
            conv_mlp: Use 1x1 conv in MLP, improves speed for small networks w/ chan last.
            conv_bias: Use bias layers w/ all convolutions.
            use_grn: Use Global Response Norm (ConvNeXt-V2) in MLP.
            act_layer: Activation layer type.
            norm_layer: Normalization layer type.
            drop_rate: Head pre-classifier dropout rate.
            drop_path_rate: Stochastic depth drop rate.
        """
        super().__init__()
        assert output_stride in (8, 16, 32)
        kernel_sizes = to_ntuple(4)(kernel_sizes)
        if norm_layer is None:
            norm_layer = LayerNorm2d
            norm_layer_cl = norm_layer if conv_mlp else LayerNorm
            if norm_eps is not None:
                norm_layer = partial(norm_layer, eps=norm_eps)
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)
        else:
            assert conv_mlp,\
                'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            norm_layer_cl = norm_layer
            if norm_eps is not None:
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        assert stem_type in ('patch', 'overlap', 'overlap_tiered')
        if stem_type == 'patch':
            # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=conv_bias),
                norm_layer(dims[0]),
            )
            stem_stride = patch_size
        else:
            mid_chs = make_divisible(dims[0] // 2) if 'tiered' in stem_type else dims[0]
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, mid_chs, kernel_size=3, stride=2, padding=1, bias=conv_bias),
                nn.Conv2d(mid_chs, dims[0], kernel_size=3, stride=2, padding=1, bias=conv_bias),
                norm_layer(dims[0]),
            )
            stem_stride = 4

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0]
        curr_stride = stem_stride
        dilation = 1
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]
            stages.append(ConvNeXtStage(
                prev_chs,
                out_chs,
                kernel_size=kernel_sizes[i],
                stride=stride,
                dilation=(first_dilation, dilation),
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                conv_bias=conv_bias,
                use_grn=use_grn,
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_layer_cl=norm_layer_cl,
            ))
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs

        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default ConvNeXt ordering (pretrained FB weights)
        if head_norm_first:
            assert not head_hidden_size
            self.norm_pre = norm_layer(self.num_features)
            self.head = ClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
            )
        else:
            self.norm_pre = nn.Identity()
            self.head = NormMlpClassifierHead(
                self.num_features,
                num_classes,
                hidden_size=head_hidden_size,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
                norm_layer=norm_layer,
                act_layer='gelu',
            )
        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),  # blocks
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm_pre', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool=None):
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        nn.init.zeros_(module.bias)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def checkpoint_filter_fn(state_dict, model):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']

    out_dict = {}
    if 'visual.trunk.stem.0.weight' in state_dict:
        out_dict = {k.replace('visual.trunk.', ''): v for k, v in state_dict.items() if k.startswith('visual.trunk.')}
        if 'visual.head.proj.weight' in state_dict:
            out_dict['head.fc.weight'] = state_dict['visual.head.proj.weight']
            out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.proj.weight'].shape[0])
        elif 'visual.head.mlp.fc1.weight' in state_dict:
            out_dict['head.pre_logits.fc.weight'] = state_dict['visual.head.mlp.fc1.weight']
            out_dict['head.pre_logits.fc.bias'] = state_dict['visual.head.mlp.fc1.bias']
            out_dict['head.fc.weight'] = state_dict['visual.head.mlp.fc2.weight']
            out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.mlp.fc2.weight'].shape[0])
        return out_dict

    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        if 'grn' in k:
            k = k.replace('grn.beta', 'mlp.grn.bias')
            k = k.replace('grn.gamma', 'mlp.grn.weight')
            v = v.reshape(v.shape[-1])
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')
        if v.ndim == 2 and 'head' not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v

    return out_dict


def _create_convnext(variant, pretrained=False, **kwargs):
    if kwargs.get('pretrained_cfg', '') == 'fcmae':
        # NOTE fcmae pretrained weights have no classifier or final norm-layer (`head.norm`)
        # This is workaround loading with num_classes=0 w/o removing norm-layer.
        kwargs.setdefault('pretrained_strict', False)

    model = build_model_with_cfg(
        ConvNeXt, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs)
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }


def _cfgv2(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        'license': 'cc-by-nc-4.0', 'paper_ids': 'arXiv:2301.00808',
        'paper_name': 'ConvNeXt-V2: Co-designing and Scaling ConvNets with Masked Autoencoders',
        'origin_url': 'https://github.com/facebookresearch/ConvNeXt-V2',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # timm specific variants
    'convnext_tiny.in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnext_small.in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),

    'convnext_atto.d2_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_atto_d2-01bb0f51.pth',
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    'convnext_atto_ols.a2_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_atto_ols_a2-78d1c8f3.pth',
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    'convnext_femto.d1_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_femto_d1-d71d5b4c.pth',
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    'convnext_femto_ols.d1_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_femto_ols_d1-246bf2ed.pth',
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    'convnext_pico.d1_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_pico_d1-10ad7f0d.pth',
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    'convnext_pico_ols.d1_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_pico_ols_d1-611f0ca7.pth',
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnext_nano.in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnext_nano.d1h_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_nano_d1h-7eb4bdea.pth',
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnext_nano_ols.d1h_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_nano_ols_d1h-ae424a9a.pth',
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnext_tiny_hnf.a2h_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_tiny_hnf_a2h-ab7e9df2.pth',
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),

    'convnext_tiny.in12k_ft_in1k_384': _cfg(
        hf_hub_id='timm/',
       input_size=(3, 384, 384), pool_size=(12, 12),  crop_pct=1.0, crop_mode='squash'),
    'convnext_small.in12k_ft_in1k_384': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,  crop_mode='squash'),

    'convnext_nano.in12k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, num_classes=11821),
    'convnext_tiny.in12k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, num_classes=11821),
    'convnext_small.in12k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, num_classes=11821),

    'convnext_tiny.fb_in22k_ft_in1k': _cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth',
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnext_small.fb_in22k_ft_in1k': _cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_224.pth',
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnext_base.fb_in22k_ft_in1k': _cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth',
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnext_large.fb_in22k_ft_in1k': _cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth',
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnext_xlarge.fb_in22k_ft_in1k': _cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth',
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),

    'convnext_tiny.fb_in1k': _cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnext_small.fb_in1k': _cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnext_base.fb_in1k': _cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnext_large.fb_in1k': _cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),

    'convnext_tiny.fb_in22k_ft_in1k_384': _cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_384.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'convnext_small.fb_in22k_ft_in1k_384': _cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'convnext_base.fb_in22k_ft_in1k_384': _cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'convnext_large.fb_in22k_ft_in1k_384': _cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'convnext_xlarge.fb_in22k_ft_in1k_384': _cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),

    'convnext_tiny.fb_in22k': _cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
        hf_hub_id='timm/',
        num_classes=21841),
    'convnext_small.fb_in22k': _cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
        hf_hub_id='timm/',
        num_classes=21841),
    'convnext_base.fb_in22k': _cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
        hf_hub_id='timm/',
        num_classes=21841),
    'convnext_large.fb_in22k': _cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
        hf_hub_id='timm/',
        num_classes=21841),
    'convnext_xlarge.fb_in22k': _cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
        hf_hub_id='timm/',
        num_classes=21841),

    'convnextv2_nano.fcmae_ft_in22k_in1k': _cfgv2(
        url='https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_224_ema.pt',
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnextv2_nano.fcmae_ft_in22k_in1k_384': _cfgv2(
        url='https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_384_ema.pt',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'convnextv2_tiny.fcmae_ft_in22k_in1k': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_224_ema.pt",
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnextv2_tiny.fcmae_ft_in22k_in1k_384': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_384_ema.pt",
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'convnextv2_base.fcmae_ft_in22k_in1k': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_224_ema.pt",
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnextv2_base.fcmae_ft_in22k_in1k_384': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt",
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'convnextv2_large.fcmae_ft_in22k_in1k': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt",
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnextv2_large.fcmae_ft_in22k_in1k_384': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt",
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'convnextv2_huge.fcmae_ft_in22k_in1k_384': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_384_ema.pt",
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'convnextv2_huge.fcmae_ft_in22k_in1k_512': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt",
        hf_hub_id='timm/',
        input_size=(3, 512, 512), pool_size=(15, 15), crop_pct=1.0, crop_mode='squash'),

    'convnextv2_atto.fcmae_ft_in1k': _cfgv2(
        url='https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt',
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    'convnextv2_femto.fcmae_ft_in1k': _cfgv2(
        url='https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt',
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    'convnextv2_pico.fcmae_ft_in1k': _cfgv2(
        url='https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt',
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    'convnextv2_nano.fcmae_ft_in1k': _cfgv2(
        url='https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_nano_1k_224_ema.pt',
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnextv2_tiny.fcmae_ft_in1k': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt",
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnextv2_base.fcmae_ft_in1k': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt",
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnextv2_large.fcmae_ft_in1k': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_large_1k_224_ema.pt",
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnextv2_huge.fcmae_ft_in1k': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_huge_1k_224_ema.pt",
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),

    'convnextv2_atto.fcmae': _cfgv2(
        url='https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_atto_1k_224_fcmae.pt',
        hf_hub_id='timm/',
        num_classes=0),
    'convnextv2_femto.fcmae': _cfgv2(
        url='https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_femto_1k_224_fcmae.pt',
        hf_hub_id='timm/',
        num_classes=0),
    'convnextv2_pico.fcmae': _cfgv2(
        url='https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_pico_1k_224_fcmae.pt',
        hf_hub_id='timm/',
        num_classes=0),
    'convnextv2_nano.fcmae': _cfgv2(
        url='https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_nano_1k_224_fcmae.pt',
        hf_hub_id='timm/',
        num_classes=0),
    'convnextv2_tiny.fcmae': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_tiny_1k_224_fcmae.pt",
        hf_hub_id='timm/',
        num_classes=0),
    'convnextv2_base.fcmae': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_base_1k_224_fcmae.pt",
        hf_hub_id='timm/',
        num_classes=0),
    'convnextv2_large.fcmae': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_large_1k_224_fcmae.pt",
        hf_hub_id='timm/',
        num_classes=0),
    'convnextv2_huge.fcmae': _cfgv2(
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_huge_1k_224_fcmae.pt",
        hf_hub_id='timm/',
        num_classes=0),

    'convnextv2_small.untrained': _cfg(),

    # CLIP weights, fine-tuned on in1k or in12k + in1k
    'convnext_base.clip_laion2b_augreg_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0),
    'convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=1.0),
    'convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),

    'convnext_base.clip_laion2b_augreg_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0),
    'convnext_base.clip_laiona_augreg_ft_in1k_384': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    'convnext_large_mlp.clip_laion2b_augreg_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0
    ),
    'convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'
    ),
    'convnext_xxlarge.clip_laion2b_soup_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0),

    'convnext_base.clip_laion2b_augreg_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821,
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0),
    'convnext_large_mlp.clip_laion2b_soup_ft_in12k_320': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821,
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=1.0),
    'convnext_large_mlp.clip_laion2b_augreg_ft_in12k_384': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821,
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'convnext_large_mlp.clip_laion2b_soup_ft_in12k_384': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821,
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),

    # CLIP original image tower weights
    'convnext_base.clip_laion2b': _cfg(
        hf_hub_id='laion/CLIP-convnext_base_w-laion2B-s13B-b82K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, num_classes=640),
    'convnext_base.clip_laion2b_augreg': _cfg(
        hf_hub_id='laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, num_classes=640),
    'convnext_base.clip_laiona': _cfg(
        hf_hub_id='laion/CLIP-convnext_base_w-laion_aesthetic-s13B-b82K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, num_classes=640),
    'convnext_base.clip_laiona_320': _cfg(
        hf_hub_id='laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=1.0, num_classes=640),
    'convnext_base.clip_laiona_augreg_320': _cfg(
        hf_hub_id='laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=1.0, num_classes=640),
    'convnext_large_mlp.clip_laion2b_augreg': _cfg(
        hf_hub_id='laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, num_classes=768),
    'convnext_large_mlp.clip_laion2b_ft_320': _cfg(
        hf_hub_id='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=1.0, num_classes=768),
    'convnext_large_mlp.clip_laion2b_ft_soup_320': _cfg(
        hf_hub_id='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=1.0, num_classes=768),
    'convnext_xxlarge.clip_laion2b_soup': _cfg(
        hf_hub_id='laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, num_classes=1024),
    'convnext_xxlarge.clip_laion2b_rewind': _cfg(
        hf_hub_id='laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-rewind',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, num_classes=1024),
})


@register_model
def convnext_atto(pretrained=False, **kwargs) -> ConvNeXt:
    # timm femto variant (NOTE: still tweaking depths, will vary between 3-4M param, current is 3.7M
    model_args = dict(depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), conv_mlp=True)
    model = _create_convnext('convnext_atto', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnext_atto_ols(pretrained=False, **kwargs) -> ConvNeXt:
    # timm femto variant with overlapping 3x3 conv stem, wider than non-ols femto above, current param count 3.7M
    model_args = dict(depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), conv_mlp=True, stem_type='overlap_tiered')
    model = _create_convnext('convnext_atto_ols', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnext_femto(pretrained=False, **kwargs) -> ConvNeXt:
    # timm femto variant
    model_args = dict(depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), conv_mlp=True)
    model = _create_convnext('convnext_femto', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnext_femto_ols(pretrained=False, **kwargs) -> ConvNeXt:
    # timm femto variant
    model_args = dict(depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), conv_mlp=True, stem_type='overlap_tiered')
    model = _create_convnext('convnext_femto_ols', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnext_pico(pretrained=False, **kwargs) -> ConvNeXt:
    # timm pico variant
    model_args = dict(depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), conv_mlp=True)
    model = _create_convnext('convnext_pico', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnext_pico_ols(pretrained=False, **kwargs) -> ConvNeXt:
    # timm nano variant with overlapping 3x3 conv stem
    model_args = dict(depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), conv_mlp=True,  stem_type='overlap_tiered')
    model = _create_convnext('convnext_pico_ols', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnext_nano(pretrained=False, **kwargs) -> ConvNeXt:
    # timm nano variant with standard stem and head
    model_args = dict(depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), conv_mlp=True)
    model = _create_convnext('convnext_nano', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnext_nano_ols(pretrained=False, **kwargs) -> ConvNeXt:
    # experimental nano variant with overlapping conv stem
    model_args = dict(depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), conv_mlp=True, stem_type='overlap')
    model = _create_convnext('convnext_nano_ols', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnext_tiny_hnf(pretrained=False, **kwargs) -> ConvNeXt:
    # experimental tiny variant with norm before pooling in head (head norm first)
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), head_norm_first=True, conv_mlp=True)
    model = _create_convnext('convnext_tiny_hnf', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnext_tiny(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
    model = _create_convnext('convnext_tiny', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnext_small(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
    model = _create_convnext('convnext_small', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnext_base(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    model = _create_convnext('convnext_base', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnext_large(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
    model = _create_convnext('convnext_large', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnext_large_mlp(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], head_hidden_size=1536)
    model = _create_convnext('convnext_large_mlp', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnext_xlarge(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048])
    model = _create_convnext('convnext_xlarge', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnext_xxlarge(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(depths=[3, 4, 30, 3], dims=[384, 768, 1536, 3072], norm_eps=kwargs.pop('norm_eps', 1e-5))
    model = _create_convnext('convnext_xxlarge', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnextv2_atto(pretrained=False, **kwargs) -> ConvNeXt:
    # timm femto variant (NOTE: still tweaking depths, will vary between 3-4M param, current is 3.7M
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), use_grn=True, ls_init_value=None, conv_mlp=True)
    model = _create_convnext('convnextv2_atto', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnextv2_femto(pretrained=False, **kwargs) -> ConvNeXt:
    # timm femto variant
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), use_grn=True, ls_init_value=None, conv_mlp=True)
    model = _create_convnext('convnextv2_femto', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnextv2_pico(pretrained=False, **kwargs) -> ConvNeXt:
    # timm pico variant
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), use_grn=True, ls_init_value=None, conv_mlp=True)
    model = _create_convnext('convnextv2_pico', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnextv2_nano(pretrained=False, **kwargs) -> ConvNeXt:
    # timm nano variant with standard stem and head
    model_args = dict(
        depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), use_grn=True, ls_init_value=None, conv_mlp=True)
    model = _create_convnext('convnextv2_nano', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnextv2_tiny(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), use_grn=True, ls_init_value=None)
    model = _create_convnext('convnextv2_tiny', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnextv2_small(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], use_grn=True, ls_init_value=None)
    model = _create_convnext('convnextv2_small', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnextv2_base(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], use_grn=True, ls_init_value=None)
    model = _create_convnext('convnextv2_base', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnextv2_large(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], use_grn=True, ls_init_value=None)
    model = _create_convnext('convnextv2_large', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def convnextv2_huge(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], use_grn=True, ls_init_value=None)
    model = _create_convnext('convnextv2_huge', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


register_model_deprecations(__name__, {
    'convnext_tiny_in22ft1k': 'convnext_tiny.fb_in22k_ft_in1k',
    'convnext_small_in22ft1k': 'convnext_small.fb_in22k_ft_in1k',
    'convnext_base_in22ft1k': 'convnext_base.fb_in22k_ft_in1k',
    'convnext_large_in22ft1k': 'convnext_large.fb_in22k_ft_in1k',
    'convnext_xlarge_in22ft1k': 'convnext_xlarge.fb_in22k_ft_in1k',
    'convnext_tiny_384_in22ft1k': 'convnext_tiny.fb_in22k_ft_in1k_384',
    'convnext_small_384_in22ft1k': 'convnext_small.fb_in22k_ft_in1k_384',
    'convnext_base_384_in22ft1k': 'convnext_base.fb_in22k_ft_in1k_384',
    'convnext_large_384_in22ft1k': 'convnext_large.fb_in22k_ft_in1k_384',
    'convnext_xlarge_384_in22ft1k': 'convnext_xlarge.fb_in22k_ft_in1k_384',
    'convnext_tiny_in22k': 'convnext_tiny.fb_in22k',
    'convnext_small_in22k': 'convnext_small.fb_in22k',
    'convnext_base_in22k': 'convnext_base.fb_in22k',
    'convnext_large_in22k': 'convnext_large.fb_in22k',
    'convnext_xlarge_in22k': 'convnext_xlarge.fb_in22k',
})
