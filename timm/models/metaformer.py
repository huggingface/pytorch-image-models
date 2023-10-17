"""
Poolformer from MetaFormer is Actually What You Need for Vision https://arxiv.org/abs/2111.11418

IdentityFormer, RandFormer, PoolFormerV2, ConvFormer, and CAFormer
from MetaFormer Baselines for Vision https://arxiv.org/abs/2210.13452

All implemented models support feature extraction and variable input resolution.

Original implementation by Weihao Yu et al.,
adapted for timm by Fredo Guan and Ross Wightman.

Adapted from https://github.com/sail-sg/metaformer, original copyright below
"""

# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.jit import Final

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import trunc_normal_, DropPath, SelectAdaptivePool2d, GroupNorm1, LayerNorm, LayerNorm2d, Mlp, \
    use_fused_attn
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['MetaFormer']


class Stem(nn.Module):
    """
    Stem implemented by a layer of convolution.
    Conv2d params constant across all models.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            norm_layer=None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=4,
            padding=2
        )
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            norm_layer=None,
    ):
        super().__init__()
        self.norm = norm_layer(in_channels) if norm_layer else nn.Identity()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True, use_nchw=True):
        super().__init__()
        self.shape = (dim, 1, 1) if use_nchw else (dim,)
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale.view(self.shape)


class SquaredReLU(nn.Module):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """

    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(
            self,
            scale_value=1.0,
            bias_value=0.0,
            scale_learnable=True,
            bias_learnable=True,
            mode=None,
            inplace=False
    ):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            head_dim=32,
            num_heads=None,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
            proj_bias=False,
            **kwargs
    ):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# custom norm modules that disable the bias term, since the original models defs
# used a custom norm with a weight term but no bias term.

class GroupNorm1NoBias(GroupNorm1):
    def __init__(self, num_channels, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.eps = kwargs.get('eps', 1e-6)
        self.bias = None


class LayerNorm2dNoBias(LayerNorm2d):
    def __init__(self, num_channels, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.eps = kwargs.get('eps', 1e-6)
        self.bias = None


class LayerNormNoBias(nn.LayerNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.eps = kwargs.get('eps', 1e-6)
        self.bias = None


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
            self,
            dim,
            expansion_ratio=2,
            act1_layer=StarReLU,
            act2_layer=nn.Identity,
            bias=False,
            kernel_size=7,
            padding=3,
            **kwargs
    ):
        super().__init__()
        mid_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, mid_channels, kernel_size=1, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=kernel_size,
            padding=padding, groups=mid_channels, bias=bias)  # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Conv2d(mid_channels, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.dwconv(x)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    """

    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        y = self.pool(x)
        return y - x


class MlpHead(nn.Module):
    """ MLP classification head
    """

    def __init__(
            self,
            dim,
            num_classes=1000,
            mlp_ratio=4,
            act_layer=SquaredReLU,
            norm_layer=LayerNorm,
            drop_rate=0.,
            bias=True
    ):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_drop(x)
        x = self.fc2(x)
        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """

    def __init__(
            self,
            dim,
            token_mixer=Pooling,
            mlp_act=StarReLU,
            mlp_bias=False,
            norm_layer=LayerNorm2d,
            proj_drop=0.,
            drop_path=0.,
            use_nchw=True,
            layer_scale_init_value=None,
            res_scale_init_value=None,
            **kwargs
    ):
        super().__init__()
        ls_layer = partial(Scale, dim=dim, init_value=layer_scale_init_value, use_nchw=use_nchw)
        rs_layer = partial(Scale, dim=dim, init_value=res_scale_init_value, use_nchw=use_nchw)

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, proj_drop=proj_drop, **kwargs)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = ls_layer() if layer_scale_init_value is not None else nn.Identity()
        self.res_scale1 = rs_layer() if res_scale_init_value is not None else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            dim,
            int(4 * dim),
            act_layer=mlp_act,
            bias=mlp_bias,
            drop=proj_drop,
            use_conv=use_nchw,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = ls_layer() if layer_scale_init_value is not None else nn.Identity()
        self.res_scale2 = rs_layer() if res_scale_init_value is not None else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x


class MetaFormerStage(nn.Module):

    def __init__(
            self,
            in_chs,
            out_chs,
            depth=2,
            token_mixer=nn.Identity,
            mlp_act=StarReLU,
            mlp_bias=False,
            downsample_norm=LayerNorm2d,
            norm_layer=LayerNorm2d,
            proj_drop=0.,
            dp_rates=[0.] * 2,
            layer_scale_init_value=None,
            res_scale_init_value=None,
            **kwargs,
    ):
        super().__init__()

        self.grad_checkpointing = False
        self.use_nchw = not issubclass(token_mixer, Attention)

        # don't downsample if in_chs and out_chs are the same
        self.downsample = nn.Identity() if in_chs == out_chs else Downsampling(
            in_chs,
            out_chs,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_layer=downsample_norm,
        )

        self.blocks = nn.Sequential(*[MetaFormerBlock(
            dim=out_chs,
            token_mixer=token_mixer,
            mlp_act=mlp_act,
            mlp_bias=mlp_bias,
            norm_layer=norm_layer,
            proj_drop=proj_drop,
            drop_path=dp_rates[i],
            layer_scale_init_value=layer_scale_init_value,
            res_scale_init_value=res_scale_init_value,
            use_nchw=self.use_nchw,
            **kwargs,
        ) for i in range(depth)])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, x: Tensor):
        x = self.downsample(x)
        B, C, H, W = x.shape

        if not self.use_nchw:
            x = x.reshape(B, C, -1).transpose(1, 2)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        if not self.use_nchw:
            x = x.transpose(1, 2).reshape(B, C, H, W)

        return x


class MetaFormer(nn.Module):
    r""" MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452

    Args:
        in_chans (int): Number of input image channels.
        num_classes (int): Number of classes for classification head.
        global_pool: Pooling for classifier head.
        depths (list or tuple): Number of blocks at each stage.
        dims (list or tuple): Feature dimension at each stage.
        token_mixers (list, tuple or token_fcn): Token mixer for each stage.
        mlp_act: Activation layer for MLP.
        mlp_bias (boolean): Enable or disable mlp bias term.
        drop_path_rate (float): Stochastic depth rate.
        drop_rate (float): Dropout rate.
        layer_scale_init_values (list, tuple, float or None): Init value for Layer Scale.
            None means not use the layer scale. Form: https://arxiv.org/abs/2103.17239.
        res_scale_init_values (list, tuple, float or None): Init value for res Scale on residual connections.
            None means not use the res scale. From: https://arxiv.org/abs/2110.09456.
        downsample_norm (nn.Module): Norm layer used in stem and downsampling layers.
        norm_layers (list, tuple or norm_fcn): Norm layers for each stage.
        output_norm: Norm layer before classifier head.
        use_mlp_head: Use MLP classification head.
    """

    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            depths=(2, 2, 6, 2),
            dims=(64, 128, 320, 512),
            token_mixers=Pooling,
            mlp_act=StarReLU,
            mlp_bias=False,
            drop_path_rate=0.,
            proj_drop_rate=0.,
            drop_rate=0.0,
            layer_scale_init_values=None,
            res_scale_init_values=(None, None, 1.0, 1.0),
            downsample_norm=LayerNorm2dNoBias,
            norm_layers=LayerNorm2dNoBias,
            output_norm=LayerNorm2d,
            use_mlp_head=True,
            **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dims[-1]
        self.drop_rate = drop_rate
        self.use_mlp_head = use_mlp_head
        self.num_stages = len(depths)

        # convert everything to lists if they aren't indexable
        if not isinstance(depths, (list, tuple)):
            depths = [depths]  # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * self.num_stages
        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * self.num_stages
        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * self.num_stages
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * self.num_stages

        self.grad_checkpointing = False
        self.feature_info = []

        self.stem = Stem(
            in_chans,
            dims[0],
            norm_layer=downsample_norm
        )

        stages = []
        prev_dim = dims[0]
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        for i in range(self.num_stages):
            stages += [MetaFormerStage(
                prev_dim,
                dims[i],
                depth=depths[i],
                token_mixer=token_mixers[i],
                mlp_act=mlp_act,
                mlp_bias=mlp_bias,
                proj_drop=proj_drop_rate,
                dp_rates=dp_rates[i],
                layer_scale_init_value=layer_scale_init_values[i],
                res_scale_init_value=res_scale_init_values[i],
                downsample_norm=downsample_norm,
                norm_layer=norm_layers[i],
                **kwargs,
            )]
            prev_dim = dims[i]
            self.feature_info += [dict(num_chs=dims[i], reduction=2, module=f'stages.{i}')]

        self.stages = nn.Sequential(*stages)

        # if using MlpHead, dropout is handled by MlpHead
        if num_classes > 0:
            if self.use_mlp_head:
                final = MlpHead(self.num_features, num_classes, drop_rate=self.drop_rate)
            else:
                final = nn.Linear(self.num_features, num_classes)
        else:
            final = nn.Identity()

        self.head = nn.Sequential(OrderedDict([
            ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
            ('norm', output_norm(self.num_features)),
            ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
            ('drop', nn.Dropout(drop_rate) if self.use_mlp_head else nn.Identity()),
            ('fc', final)
        ]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
        for stage in self.stages:
            stage.set_grad_checkpointing(enable=enable)

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool=None):
        if global_pool is not None:
            self.head.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.head.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        if num_classes > 0:
            if self.use_mlp_head:
                final = MlpHead(self.num_features, num_classes, drop_rate=self.drop_rate)
            else:
                final = nn.Linear(self.num_features, num_classes)
        else:
            final = nn.Identity()
        self.head.fc = final

    def forward_head(self, x: Tensor, pre_logits: bool = False):
        # NOTE nn.Sequential in head broken down since can't call head[:-1](x) in torchscript :(
        x = self.head.global_pool(x)
        x = self.head.norm(x)
        x = self.head.flatten(x)
        x = self.head.drop(x)
        return x if pre_logits else self.head.fc(x)

    def forward_features(self, x: Tensor):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        return x

    def forward(self, x: Tensor):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


# this works but it's long and breaks backwards compatability with weights from the poolformer-only impl
def checkpoint_filter_fn(state_dict, model):
    if 'stem.conv.weight' in state_dict:
        return state_dict

    import re
    out_dict = {}
    is_poolformerv1 = 'network.0.0.mlp.fc1.weight' in state_dict
    model_state_dict = model.state_dict()
    for k, v in state_dict.items():
        if is_poolformerv1:
            k = re.sub(r'layer_scale_([0-9]+)', r'layer_scale\1.scale', k)
            k = k.replace('network.1', 'downsample_layers.1')
            k = k.replace('network.3', 'downsample_layers.2')
            k = k.replace('network.5', 'downsample_layers.3')
            k = k.replace('network.2', 'network.1')
            k = k.replace('network.4', 'network.2')
            k = k.replace('network.6', 'network.3')
            k = k.replace('network', 'stages')

        k = re.sub(r'downsample_layers.([0-9]+)', r'stages.\1.downsample', k)
        k = k.replace('downsample.proj', 'downsample.conv')
        k = k.replace('patch_embed.proj', 'patch_embed.conv')
        k = re.sub(r'([0-9]+).([0-9]+)', r'\1.blocks.\2', k)
        k = k.replace('stages.0.downsample', 'patch_embed')
        k = k.replace('patch_embed', 'stem')
        k = k.replace('post_norm', 'norm')
        k = k.replace('pre_norm', 'norm')
        k = re.sub(r'^head', 'head.fc', k)
        k = re.sub(r'^norm', 'head.norm', k)

        if v.shape != model_state_dict[k] and v.numel() == model_state_dict[k].numel():
            v = v.reshape(model_state_dict[k].shape)

        out_dict[k] = v
    return out_dict


def _create_metaformer(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (2, 2, 6, 2))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        MetaFormer,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs,
    )

    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head.fc', 'first_conv': 'stem.conv',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'poolformer_s12.sail_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.9),
    'poolformer_s24.sail_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.9),
    'poolformer_s36.sail_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.9),
    'poolformer_m36.sail_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95),
    'poolformer_m48.sail_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95),

    'poolformerv2_s12.sail_in1k': _cfg(hf_hub_id='timm/'),
    'poolformerv2_s24.sail_in1k': _cfg(hf_hub_id='timm/'),
    'poolformerv2_s36.sail_in1k': _cfg(hf_hub_id='timm/'),
    'poolformerv2_m36.sail_in1k': _cfg(hf_hub_id='timm/'),
    'poolformerv2_m48.sail_in1k': _cfg(hf_hub_id='timm/'),

    'convformer_s18.sail_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'convformer_s18.sail_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'convformer_s18.sail_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'convformer_s18.sail_in22k_ft_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'convformer_s18.sail_in22k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', num_classes=21841),

    'convformer_s36.sail_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'convformer_s36.sail_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'convformer_s36.sail_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'convformer_s36.sail_in22k_ft_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'convformer_s36.sail_in22k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', num_classes=21841),

    'convformer_m36.sail_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'convformer_m36.sail_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'convformer_m36.sail_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'convformer_m36.sail_in22k_ft_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'convformer_m36.sail_in22k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', num_classes=21841),

    'convformer_b36.sail_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'convformer_b36.sail_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'convformer_b36.sail_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'convformer_b36.sail_in22k_ft_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'convformer_b36.sail_in22k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', num_classes=21841),

    'caformer_s18.sail_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'caformer_s18.sail_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'caformer_s18.sail_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'caformer_s18.sail_in22k_ft_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'caformer_s18.sail_in22k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', num_classes=21841),

    'caformer_s36.sail_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'caformer_s36.sail_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'caformer_s36.sail_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'caformer_s36.sail_in22k_ft_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'caformer_s36.sail_in22k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', num_classes=21841),

    'caformer_m36.sail_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'caformer_m36.sail_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'caformer_m36.sail_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'caformer_m36.sail_in22k_ft_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'caformer_m36.sail_in22k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', num_classes=21841),

    'caformer_b36.sail_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'caformer_b36.sail_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'caformer_b36.sail_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2'),
    'caformer_b36.sail_in22k_ft_in1k_384': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12, 12)),
    'caformer_b36.sail_in22k': _cfg(
        hf_hub_id='timm/',
        classifier='head.fc.fc2', num_classes=21841),
})


@register_model
def poolformer_s12(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        downsample_norm=None,
        mlp_act=nn.GELU,
        mlp_bias=True,
        norm_layers=GroupNorm1,
        layer_scale_init_values=1e-5,
        res_scale_init_values=None,
        use_mlp_head=False,
        **kwargs)
    return _create_metaformer('poolformer_s12', pretrained=pretrained, **model_kwargs)


@register_model
def poolformer_s24(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        downsample_norm=None,
        mlp_act=nn.GELU,
        mlp_bias=True,
        norm_layers=GroupNorm1,
        layer_scale_init_values=1e-5,
        res_scale_init_values=None,
        use_mlp_head=False,
        **kwargs)
    return _create_metaformer('poolformer_s24', pretrained=pretrained, **model_kwargs)


@register_model
def poolformer_s36(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        downsample_norm=None,
        mlp_act=nn.GELU,
        mlp_bias=True,
        norm_layers=GroupNorm1,
        layer_scale_init_values=1e-6,
        res_scale_init_values=None,
        use_mlp_head=False,
        **kwargs)
    return _create_metaformer('poolformer_s36', pretrained=pretrained, **model_kwargs)


@register_model
def poolformer_m36(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        downsample_norm=None,
        mlp_act=nn.GELU,
        mlp_bias=True,
        norm_layers=GroupNorm1,
        layer_scale_init_values=1e-6,
        res_scale_init_values=None,
        use_mlp_head=False,
        **kwargs)
    return _create_metaformer('poolformer_m36', pretrained=pretrained, **model_kwargs)


@register_model
def poolformer_m48(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        downsample_norm=None,
        mlp_act=nn.GELU,
        mlp_bias=True,
        norm_layers=GroupNorm1,
        layer_scale_init_values=1e-6,
        res_scale_init_values=None,
        use_mlp_head=False,
        **kwargs)
    return _create_metaformer('poolformer_m48', pretrained=pretrained, **model_kwargs)


@register_model
def poolformerv2_s12(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        norm_layers=GroupNorm1NoBias,
        use_mlp_head=False,
        **kwargs)
    return _create_metaformer('poolformerv2_s12', pretrained=pretrained, **model_kwargs)


@register_model
def poolformerv2_s24(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        norm_layers=GroupNorm1NoBias,
        use_mlp_head=False,
        **kwargs)
    return _create_metaformer('poolformerv2_s24', pretrained=pretrained, **model_kwargs)


@register_model
def poolformerv2_s36(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        norm_layers=GroupNorm1NoBias,
        use_mlp_head=False,
        **kwargs)
    return _create_metaformer('poolformerv2_s36', pretrained=pretrained, **model_kwargs)


@register_model
def poolformerv2_m36(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        norm_layers=GroupNorm1NoBias,
        use_mlp_head=False,
        **kwargs)
    return _create_metaformer('poolformerv2_m36', pretrained=pretrained, **model_kwargs)


@register_model
def poolformerv2_m48(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        norm_layers=GroupNorm1NoBias,
        use_mlp_head=False,
        **kwargs)
    return _create_metaformer('poolformerv2_m48', pretrained=pretrained, **model_kwargs)


@register_model
def convformer_s18(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        norm_layers=LayerNorm2dNoBias,
        **kwargs)
    return _create_metaformer('convformer_s18', pretrained=pretrained, **model_kwargs)


@register_model
def convformer_s36(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        norm_layers=LayerNorm2dNoBias,
        **kwargs)
    return _create_metaformer('convformer_s36', pretrained=pretrained, **model_kwargs)


@register_model
def convformer_m36(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=SepConv,
        norm_layers=LayerNorm2dNoBias,
        **kwargs)
    return _create_metaformer('convformer_m36', pretrained=pretrained, **model_kwargs)


@register_model
def convformer_b36(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=SepConv,
        norm_layers=LayerNorm2dNoBias,
        **kwargs)
    return _create_metaformer('convformer_b36', pretrained=pretrained, **model_kwargs)


@register_model
def caformer_s18(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        norm_layers=[LayerNorm2dNoBias] * 2 + [LayerNormNoBias] * 2,
        **kwargs)
    return _create_metaformer('caformer_s18', pretrained=pretrained, **model_kwargs)


@register_model
def caformer_s36(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        norm_layers=[LayerNorm2dNoBias] * 2 + [LayerNormNoBias] * 2,
        **kwargs)
    return _create_metaformer('caformer_s36', pretrained=pretrained, **model_kwargs)


@register_model
def caformer_m36(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        norm_layers=[LayerNorm2dNoBias] * 2 + [LayerNormNoBias] * 2,
        **kwargs)
    return _create_metaformer('caformer_m36', pretrained=pretrained, **model_kwargs)


@register_model
def caformer_b36(pretrained=False, **kwargs) -> MetaFormer:
    model_kwargs = dict(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        norm_layers=[LayerNorm2dNoBias] * 2 + [LayerNormNoBias] * 2,
        **kwargs)
    return _create_metaformer('caformer_b36', pretrained=pretrained, **model_kwargs)
