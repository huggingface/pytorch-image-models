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

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import trunc_normal_, SelectAdaptivePool2d, DropPath, Mlp, GlobalResponseNormMlp, \
    LayerNorm2d, LayerNorm, create_conv2d, get_act_layer, make_divisible, to_ntuple
from ._builder import build_model_with_cfg
from ._manipulate import named_apply, checkpoint_seq
from ._pretrained import generate_default_cfgs
from ._registry import register_model

__all__ = ['ConvNeXt']  # model_registry will add each entrypoint fn to this


class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.

    Args:
        in_chs (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            in_chs,
            out_chs=None,
            kernel_size=7,
            stride=1,
            dilation=1,
            mlp_ratio=4,
            conv_mlp=False,
            conv_bias=True,
            use_grn=False,
            ls_init_value=1e-6,
            act_layer='gelu',
            norm_layer=None,
            drop_path=0.,
    ):
        super().__init__()
        out_chs = out_chs or in_chs
        act_layer = get_act_layer(act_layer)
        if not norm_layer:
            norm_layer = LayerNorm2d if conv_mlp else LayerNorm
        mlp_layer = partial(GlobalResponseNormMlp if use_grn else Mlp, use_conv=conv_mlp)
        self.use_conv_mlp = conv_mlp
        self.conv_dw = create_conv2d(
            in_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=dilation, depthwise=True, bias=conv_bias)
        self.norm = norm_layer(out_chs)
        self.mlp = mlp_layer(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(out_chs)) if ls_init_value is not None else None
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

        x = self.drop_path(x) + shortcut
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
                    in_chs, out_chs, kernel_size=ds_ks, stride=stride,
                    dilation=dilation[0], padding=pad, bias=conv_bias),
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
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            output_stride=32,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            kernel_sizes=7,
            ls_init_value=1e-6,
            stem_type='patch',
            patch_size=4,
            head_init_scale=1.,
            head_norm_first=False,
            conv_mlp=False,
            conv_bias=True,
            use_grn=False,
            act_layer='gelu',
            norm_layer=None,
            drop_rate=0.,
            drop_path_rate=0.,
    ):
        """
        Args:
            in_chans (int): Number of input image channels (default: 3)
            num_classes (int): Number of classes for classification head (default: 1000)
            global_pool (str): Global pooling type (default: 'avg')
            output_stride (int): Output stride of network, one of (8, 16, 32) (default: 32)
            depths (tuple(int)): Number of blocks at each stage. (default: [3, 3, 9, 3])
            dims (tuple(int)): Feature dimension at each stage. (default: [96, 192, 384, 768])
            kernel_sizes (Union[int, List[int]]: Depthwise convolution kernel-sizes for each stage (default: 7)
            ls_init_value (float): Init value for Layer Scale (default: 1e-6)
            stem_type (str): Type of stem (default: 'patch')
            patch_size (int): Stem patch size for patch stem (default: 4)
            head_init_scale (float): Init scaling value for classifier weights and biases (default: 1)
            head_norm_first (bool): Apply normalization before global pool + head (default: False)
            conv_mlp (bool): Use 1x1 conv in MLP, improves speed for small networks w/ chan last (default: False)
            conv_bias (bool): Use bias layers w/ all convolutions (default: True)
            use_grn (bool): Use Global Response Norm (ConvNeXt-V2) in MLP (default: False)
            act_layer (Union[str, nn.Module]): Activation Layer
            norm_layer (Union[str, nn.Module]): Normalization Layer
            drop_rate (float): Head dropout rate (default: 0.)
            drop_path_rate (float): Stochastic depth rate (default: 0.)
        """
        super().__init__()
        assert output_stride in (8, 16, 32)
        kernel_sizes = to_ntuple(4)(kernel_sizes)
        if norm_layer is None:
            norm_layer = LayerNorm2d
            norm_layer_cl = norm_layer if conv_mlp else LayerNorm
        else:
            assert conv_mlp,\
                'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            norm_layer_cl = norm_layer

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        assert stem_type in ('patch', 'overlap', 'overlap_tiered')
        if stem_type == 'patch':
            # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=conv_bias),
                norm_layer(dims[0])
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
        self.norm_pre = norm_layer(self.num_features) if head_norm_first else nn.Identity()
        self.head = nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                ('norm', nn.Identity() if head_norm_first else norm_layer(self.num_features)),
                ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
                ('drop', nn.Dropout(self.drop_rate)),
                ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())]))

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
        if global_pool is not None:
            self.head.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.head.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.head.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        # NOTE nn.Sequential in head broken down since can't call head[:-1](x) in torchscript :(
        x = self.head.global_pool(x)
        x = self.head.norm(x)
        x = self.head.flatten(x)
        x = self.head.drop(x)
        return x if pre_logits else self.head.fc(x)

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
    'convnext_tiny.in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'convnext_small.in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),

    'convnext_nano.in12k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, num_classes=11821),
    'convnext_tiny.in12k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, num_classes=11821),
    'convnext_small.in12k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, num_classes=11821),

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
    'convnext_xlarge.untrained': _cfg(),
    'convnext_xxlarge.untrained': _cfg(),

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
})


@register_model
def convnext_atto(pretrained=False, **kwargs):
    # timm femto variant (NOTE: still tweaking depths, will vary between 3-4M param, current is 3.7M
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), conv_mlp=True, **kwargs)
    model = _create_convnext('convnext_atto', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_atto_ols(pretrained=False, **kwargs):
    # timm femto variant with overlapping 3x3 conv stem, wider than non-ols femto above, current param count 3.7M
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), conv_mlp=True, stem_type='overlap_tiered', **kwargs)
    model = _create_convnext('convnext_atto_ols', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_femto(pretrained=False, **kwargs):
    # timm femto variant
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), conv_mlp=True, **kwargs)
    model = _create_convnext('convnext_femto', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_femto_ols(pretrained=False, **kwargs):
    # timm femto variant
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), conv_mlp=True, stem_type='overlap_tiered', **kwargs)
    model = _create_convnext('convnext_femto_ols', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_pico(pretrained=False, **kwargs):
    # timm pico variant
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), conv_mlp=True, **kwargs)
    model = _create_convnext('convnext_pico', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_pico_ols(pretrained=False, **kwargs):
    # timm nano variant with overlapping 3x3 conv stem
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), conv_mlp=True,  stem_type='overlap_tiered', **kwargs)
    model = _create_convnext('convnext_pico_ols', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_nano(pretrained=False, **kwargs):
    # timm nano variant with standard stem and head
    model_args = dict(
        depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), conv_mlp=True, **kwargs)
    model = _create_convnext('convnext_nano', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_nano_ols(pretrained=False, **kwargs):
    # experimental nano variant with overlapping conv stem
    model_args = dict(
        depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), conv_mlp=True, stem_type='overlap', **kwargs)
    model = _create_convnext('convnext_nano_ols', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_tiny_hnf(pretrained=False, **kwargs):
    # experimental tiny variant with norm before pooling in head (head norm first)
    model_args = dict(
        depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), head_norm_first=True, conv_mlp=True, **kwargs)
    model = _create_convnext('convnext_tiny_hnf', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_tiny(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    model = _create_convnext('convnext_tiny', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_small(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    model = _create_convnext('convnext_small', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_base(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_base', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_large(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model = _create_convnext('convnext_large', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_xlarge(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    model = _create_convnext('convnext_xlarge', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_xxlarge(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 4, 30, 3], dims=[384, 768, 1536, 3072], **kwargs)
    model = _create_convnext('convnext_xxlarge', pretrained=pretrained, **model_args)
    return model


@register_model
def convnextv2_atto(pretrained=False, **kwargs):
    # timm femto variant (NOTE: still tweaking depths, will vary between 3-4M param, current is 3.7M
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), use_grn=True, ls_init_value=None, conv_mlp=True, **kwargs)
    model = _create_convnext('convnextv2_atto', pretrained=pretrained, **model_args)
    return model


@register_model
def convnextv2_femto(pretrained=False, **kwargs):
    # timm femto variant
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), use_grn=True, ls_init_value=None, conv_mlp=True, **kwargs)
    model = _create_convnext('convnextv2_femto', pretrained=pretrained, **model_args)
    return model


@register_model
def convnextv2_pico(pretrained=False, **kwargs):
    # timm pico variant
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), use_grn=True, ls_init_value=None, conv_mlp=True, **kwargs)
    model = _create_convnext('convnextv2_pico', pretrained=pretrained, **model_args)
    return model


@register_model
def convnextv2_nano(pretrained=False, **kwargs):
    # timm nano variant with standard stem and head
    model_args = dict(
        depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), use_grn=True, ls_init_value=None, conv_mlp=True, **kwargs)
    model = _create_convnext('convnextv2_nano', pretrained=pretrained, **model_args)
    return model


@register_model
def convnextv2_tiny(pretrained=False, **kwargs):
    model_args = dict(
        depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), use_grn=True, ls_init_value=None, **kwargs)
    model = _create_convnext('convnextv2_tiny', pretrained=pretrained, **model_args)
    return model


@register_model
def convnextv2_small(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], use_grn=True, ls_init_value=None, **kwargs)
    model = _create_convnext('convnextv2_small', pretrained=pretrained, **model_args)
    return model


@register_model
def convnextv2_base(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], use_grn=True, ls_init_value=None, **kwargs)
    model = _create_convnext('convnextv2_base', pretrained=pretrained, **model_args)
    return model


@register_model
def convnextv2_large(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], use_grn=True, ls_init_value=None, **kwargs)
    model = _create_convnext('convnextv2_large', pretrained=pretrained, **model_args)
    return model


@register_model
def convnextv2_huge(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], use_grn=True, ls_init_value=None, **kwargs)
    model = _create_convnext('convnextv2_huge', pretrained=pretrained, **model_args)
    return model