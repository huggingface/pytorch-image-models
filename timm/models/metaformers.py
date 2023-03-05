"""
Poolformer from MetaFormer is Actually What You Need for Vision https://arxiv.org/abs/2111.11418

IdentityFormer, RandFormer, PoolFormerV2, ConvFormer, and CAFormer
from MetaFormer Baselines for Vision https://arxiv.org/abs/2210.13452

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
from torch import Tensor

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import trunc_normal_, DropPath, SelectAdaptivePool2d, GroupNorm1, LayerNorm, LayerNorm2d
from timm.layers.helpers import to_2tuple
from ._builder import build_model_with_cfg
from ._features import FeatureInfo
from ._features_fx import register_notrace_function
from ._manipulate import checkpoint_seq
from ._pretrained import generate_default_cfgs
from ._registry import register_model



__all__ = ['MetaFormer']


class Stem(nn.Module):
    """
    Stem implemented by a layer of convolution.
    Conv2d params constant across all models.
    """
    def __init__(self,
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
        # [B, C, H, W]
        return x

class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """
    def __init__(self,
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
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim, 1, 1), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale

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
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x.permute(0, 2, 3, 1)).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape(B, C, H, W)
        return x

class RandomMixing(nn.Module):
    def __init__(self, num_tokens=196, **kwargs):
        super().__init__()
        # FIXME no grad breaks tests
        self.random_matrix = nn.parameter.Parameter(
            data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1), 
            requires_grad=False)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, H*W, C)
        # FIXME change to work with arbitrary input sizes
        x = torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = x.reshape(B, C, H, W)
        return x

# custom norm modules that disable the bias term, since the original models defs
# used a custom norm with a weight term but no bias term.

class GroupNorm1WithoutBias(GroupNorm1):
    def __init__(self, num_channels, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.bias = None

class LayerNorm2dWithoutBias(LayerNorm2d):
    def __init__(self, num_channels, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.bias = None

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=nn.Identity, 
        bias=False, kernel_size=7, padding=3,
        **kwargs, ):
        super().__init__()
        mid_channels = int(expansion_ratio * dim)
        #self.pwconv1 = nn.Linear(dim, mid_channels, bias=bias)
        self.pwconv1 = nn.Conv2d(dim, mid_channels, kernel_size=1, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=kernel_size,
            padding=padding, groups=mid_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        #self.pwconv2 = nn.Linear(mid_channels, dim, bias=bias)
        self.pwconv2 = nn.Conv2d(mid_channels, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # [B, C, H, W]
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
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        y = self.pool(x)
        return y - x

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Modified from standard timm implementation
    """
    def __init__(
        self,
        dim,
        mlp_ratio=4,
        out_features=None,
        act_layer=StarReLU,
        mlp_fn=partial(nn.Conv2d, kernel_size=1),
        drop=0.,
        bias=False
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = mlp_fn(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = mlp_fn(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU,
        norm_layer=LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x

class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(
        self, 
        dim,
        token_mixer=nn.Identity,
        mlp=Mlp,
        mlp_fn=nn.Conv2d,
        mlp_act=StarReLU,
        mlp_bias=False,
        norm_layer=LayerNorm2d,
        drop=0., 
        drop_path=0.,
        layer_scale_init_value=None,
        res_scale_init_value=None
     ):

        super().__init__()
                
        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(
            dim=dim, 
            drop=drop, 
            mlp_fn=mlp_fn, 
            act_layer=mlp_act, 
            bias=mlp_bias
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        
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
    # implementation of a single metaformer stage
    def __init__(
        self,
        in_chs,
        out_chs,
        depth=2,
        downsample_norm=LayerNorm2d,
        token_mixer=nn.Identity,
        mlp=Mlp,
        mlp_fn=nn.Linear,
        mlp_act=StarReLU,
        mlp_bias=False,
        norm_layer=LayerNorm2d,
        dp_rates=[0.]*2,
        layer_scale_init_value=None,
        res_scale_init_value=None,
    ):
        super().__init__()

        self.grad_checkpointing = False
        
        # don't downsample if in_chs and out_chs are the same
        self.downsample = nn.Identity() if in_chs == out_chs else Downsampling(
            in_chs,
            out_chs,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_layer=downsample_norm
        )
        
        self.blocks = nn.Sequential(*[MetaFormerBlock(
            dim=out_chs,
            token_mixer=token_mixer,
            mlp=mlp,
            mlp_fn=mlp_fn,
            mlp_act=mlp_act,
            mlp_bias=mlp_bias,
            norm_layer=norm_layer,
            drop_path=dp_rates[i],
            layer_scale_init_value=layer_scale_init_value,
            res_scale_init_value=res_scale_init_value
            ) for i in range(depth)])
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
    
    # Permute to channels-first for feature extraction
    def forward(self, x: Tensor):
        
        # [B, C, H, W]
        x = self.downsample(x)
        
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        
        # [B, C, H, W]
        return x

class MetaFormer(nn.Module):
    r""" MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        depths (list or tuple): Number of blocks at each stage. Default: [2, 2, 6, 2].
        dims (int): Feature dimension at each stage. Default: [64, 128, 320, 512].
        token_mixers (list, tuple or token_fcn): Token mixer for each stage. Default: nn.Identity.
        mlps (list, tuple or mlp_fcn): Mlp for each stage. Default: Mlp.
        norm_layers (list, tuple or norm_fcn): Norm layers for each stage. Default: partial(LayerNormGeneral, eps=1e-6, bias=False).
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_dropout (float): dropout for MLP classifier. Default: 0.
        layer_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: None.
            None means not use the layer scale. Form: https://arxiv.org/abs/2103.17239.
        res_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: [None, None, 1.0, 1.0].
            None means not use the layer scale. From: https://arxiv.org/abs/2110.09456.
        output_norm: norm before classifier head. Default: partial(nn.LayerNorm, eps=1e-6).
        head_fn: classification head. Default: nn.Linear.
    """
    def __init__(
        self,
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        downsample_norm=LayerNorm2dWithoutBias,
        token_mixers=nn.Identity,
        mlps=Mlp,
        mlp_fn=partial(nn.Conv2d, kernel_size=1),
        mlp_act=StarReLU,
        mlp_bias=False,
        norm_layers=GroupNorm1WithoutBias,
        drop_path_rate=0.,
        drop_rate=0.0, 
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=LayerNorm2d, 
        head_norm_first=False,
        head_fn=nn.Linear,
        global_pool = 'avg',
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.head_fn = head_fn
        self.num_features = dims[-1]
        self.drop_rate = drop_rate
        self.num_stages = len(depths)
        
        # convert everything to lists if they aren't indexable
        if not isinstance(depths, (list, tuple)):
            depths = [depths] # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * self.num_stages
        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * self.num_stages
        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * self.num_stages
        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * self.num_stages
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * self.num_stages
        
        self.grad_checkpointing = False
        self.feature_info = []
        
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        
        self.stem = Stem(
            in_chans,
            dims[0], 
            norm_layer=downsample_norm
        )
        
        stages = nn.ModuleList() # each stage consists of multiple metaformer blocks
        cur = 0
        last_dim = dims[0]
        for i in range(self.num_stages):
            stage = MetaFormerStage(
                last_dim,
                dims[i],
                depth=depths[i],
                downsample_norm=downsample_norm,
                token_mixer=token_mixers[i],
                mlp=mlps[i],
                mlp_fn=mlp_fn,
                mlp_act=mlp_act,
                mlp_bias=mlp_bias,
                norm_layer=norm_layers[i],
                dp_rates=dp_rates[i],
                layer_scale_init_value=layer_scale_init_values[i],
                res_scale_init_value=res_scale_init_values[i],
            )
            
            stages.append(stage)
            cur += depths[i]
            last_dim = dims[i]
            self.feature_info += [dict(num_chs=dims[i], reduction=2, module=f'stages.{i}')]
        
        self.stages = nn.Sequential(*stages)
        
        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, similar to ConvNeXt
        # drop removed - if using single fc layer, models have no dropout
        # if using MlpHead, dropout is handled by MlpHead
        if num_classes > 0:
            if self.drop_rate > 0.0:
                head = self.head_fn(dims[-1], num_classes, head_dropout=self.drop_rate)
            else:
                head = self.head_fn(dims[-1], num_classes)
        else:
            head = nn.Identity()
        
        self.norm_pre = output_norm(self.num_features) if head_norm_first else nn.Identity()
        self.head = nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                ('norm', nn.Identity() if head_norm_first else output_norm(self.num_features)),
                ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
                ('fc', head)]))

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
            if self.drop_rate > 0.0:
                head = self.head_fn(dims[-1], num_classes, head_dropout=self.drop_rate)
            else:
                head = self.head_fn(dims[-1], num_classes)
        else:
            head = nn.Identity()
        self.head.fc = head
    
    def forward_head(self, x: Tensor, pre_logits: bool = False):
        # NOTE nn.Sequential in head broken down since can't call head[:-1](x) in torchscript :(
        x = self.head.global_pool(x)
        x = self.head.norm(x)
        x = self.head.flatten(x)
        return x if pre_logits else self.head.fc(x)
        
    def forward_features(self, x: Tensor):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        x = self.norm_pre(x)
        return x 

    def forward(self, x: Tensor):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

# FIXME convert to group matcher
# this works but it's long and breaks backwards compatability with weights from the poolformer-only impl
def checkpoint_filter_fn(state_dict, model):
    import re
    out_dict = {}
    for k, v in state_dict.items():
        
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
        
        if ((("fc1.weight" in k) \
            or ("fc2.weight" in k)) \
            and "fc.fc" not in k) \
            or ("res_scale1.scale" in k) \
            or ("res_scale2.scale" in k) \
            or ("pwconv1.weight" in k) \
            or ("pwconv2.weight" in k):
            v = v.reshape(*v.shape, 1, 1)
        
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
        feature_cfg=dict(flatten_sequential=True, out_indices = out_indices),
        **kwargs)
        
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
    'poolformerv1_s12.sail_in1k': _cfg(
        url='https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar',
        crop_pct=0.9),
    'poolformerv1_s24.sail_in1k': _cfg(
        url='https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s24.pth.tar',
        crop_pct=0.9),
    'poolformerv1_s36.sail_in1k': _cfg(
        url='https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar',
        crop_pct=0.9),
    'poolformerv1_m36.sail_in1k': _cfg(
        url='https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m36.pth.tar',
        crop_pct=0.95),
    'poolformerv1_m48.sail_in1k': _cfg(
        url='https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m48.pth.tar',
        crop_pct=0.95),

    'identityformer_s12.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s12.pth'),
    'identityformer_s24.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s24.pth'),
    'identityformer_s36.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s36.pth'),
    'identityformer_m36.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_m36.pth'),
    'identityformer_m48.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_m48.pth'),


    'randformer_s12.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s12.pth'),
    'randformer_s24.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s24.pth'),
    'randformer_s36.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s36.pth'),
    'randformer_m36.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/randformer/randformer_m36.pth'),
    'randformer_m48.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/randformer/randformer_m48.pth'),

    'poolformerv2_s12.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s12.pth'),
    'poolformerv2_s24.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s24.pth'),
    'poolformerv2_s36.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s36.pth'),
    'poolformerv2_m36.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_m36.pth'),
    'poolformerv2_m48.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_m48.pth'),



    'convformer_s18.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18.pth',
        classifier='head.fc.fc2'),
    'convformer_s18.sail_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_384.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'convformer_s18.sail_in22k_ft_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_in21ft1k.pth',
        classifier='head.fc.fc2'),
    'convformer_s18.sail_in22k_ft_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_384_in21ft1k.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'convformer_s18.sail_in22k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_in21k.pth',
        classifier='head.fc.fc2', num_classes=21841),

    'convformer_s36.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36.pth',
        classifier='head.fc.fc2'),
    'convformer_s36.sail_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_384.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'convformer_s36.sail_in22k_ft_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_in21ft1k.pth',
        classifier='head.fc.fc2'),
    'convformer_s36.sail_in22k_ft_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_384_in21ft1k.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'convformer_s36.sail_in22k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_in21k.pth',
        classifier='head.fc.fc2', num_classes=21841),

    'convformer_m36.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36.pth',
        classifier='head.fc.fc2'),
    'convformer_m36.sail_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_384.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'convformer_m36.sail_in22k_ft_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_in21ft1k.pth',
        classifier='head.fc.fc2'),
    'convformer_m36.sail_in22k_ft_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_384_in21ft1k.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'convformer_m36.sail_in22k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_in21k.pth',
        classifier='head.fc.fc2', num_classes=21841),

    'convformer_b36.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36.pth',
        classifier='head.fc.fc2'),
    'convformer_b36.sail_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_384.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'convformer_b36.sail_in22k_ft_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_in21ft1k.pth',
        classifier='head.fc.fc2'),
    'convformer_b36.sail_in22k_ft_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_384_in21ft1k.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'convformer_b36.sail_in22k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_in21k.pth',
        classifier='head.fc.fc2', num_classes=21841),


    'caformer_s18.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18.pth',
        classifier='head.fc.fc2'),
    'caformer_s18.sail_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_384.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'caformer_s18.sail_in22k_ft_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_in21ft1k.pth',
        classifier='head.fc.fc2'),
    'caformer_s18.sail_in22k_ft_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_384_in21ft1k.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'caformer_s18.sail_in22k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_in21k.pth',
        classifier='head.fc.fc2', num_classes=21841),

    'caformer_s36.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36.pth',
        classifier='head.fc.fc2'),
    'caformer_s36.sail_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_384.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'caformer_s36.sail_in22k_ft_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_in21ft1k.pth',
        classifier='head.fc.fc2'),
    'caformer_s36.sail_in22k_ft_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_384_in21ft1k.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'caformer_s36.sail_in22k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_in21k.pth',
        classifier='head.fc.fc2', num_classes=21841),

    'caformer_m36.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36.pth',
        classifier='head.fc.fc2'),
    'caformer_m36.sail_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_384.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'caformer_m36.sail_in22k_ft_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_in21ft1k.pth',
        classifier='head.fc.fc2'),
    'caformer_m36.sail_in22k_ft_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_384_in21ft1k.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'caformer_m36.sail_in22k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_in21k.pth',
        classifier='head.fc.fc2', num_classes=21841),

    'caformer_b36.sail_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36.pth',
        classifier='head.fc.fc2'),
    'caformer_b36.sail_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_384.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'caformer_b36.sail_in22k_ft_in1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_in21ft1k.pth',
        classifier='head.fc.fc2'),
    'caformer_b36.sail_in22k_ft_in1k_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_384_in21ft1k.pth',
        classifier='head.fc.fc2', input_size=(3, 384, 384), pool_size=(12,12)),
    'caformer_b36.sail_in22k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_in21k.pth',
        classifier='head.fc.fc2', num_classes=21841),
})

# FIXME fully merge poolformerv1, rename to poolformer to succeed poolformer.py

@register_model
def poolformerv1_s12(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        downsample_norm=None,
        token_mixers=Pooling,
        mlp_act=nn.GELU,
        mlp_bias=True,
        layer_scale_init_values=1e-5,
        res_scale_init_values=None,
        **kwargs)
    return _create_metaformer('poolformerv1_s12', pretrained=pretrained, **model_kwargs)

@register_model
def poolformerv1_s24(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        downsample_norm=None,
        token_mixers=Pooling,
        mlp_act=nn.GELU,
        mlp_bias=True,
        layer_scale_init_values=1e-5,
        res_scale_init_values=None,
        **kwargs)
    return _create_metaformer('poolformerv1_s24', pretrained=pretrained, **model_kwargs)

@register_model
def poolformerv1_s36(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        downsample_norm=None,
        token_mixers=Pooling,
        mlp_act=nn.GELU,
        mlp_bias=True,
        layer_scale_init_values=1e-6,
        res_scale_init_values=None,
        **kwargs)
    return _create_metaformer('poolformerv1_s36', pretrained=pretrained, **model_kwargs)

@register_model
def poolformerv1_m36(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        downsample_norm=None,
        token_mixers=Pooling,
        mlp_act=nn.GELU,
        mlp_bias=True,
        layer_scale_init_values=1e-6,
        res_scale_init_values=None,
        **kwargs)
    return _create_metaformer('poolformerv1_m36', pretrained=pretrained, **model_kwargs)

@register_model
def poolformerv1_m48(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        downsample_norm=None,
        token_mixers=Pooling,
        mlp_act=nn.GELU,
        mlp_bias=True,
        layer_scale_init_values=1e-6,
        res_scale_init_values=None,
        **kwargs)
    return _create_metaformer('poolformerv1_m48', pretrained=pretrained, **model_kwargs)

@register_model
def identityformer_s12(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        token_mixers=nn.Identity,
        **kwargs)
    return _create_metaformer('identityformer_s12', pretrained=pretrained, **model_kwargs)

@register_model
def identityformer_s24(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        token_mixers=nn.Identity,
        **kwargs)
    return _create_metaformer('identityformer_s24', pretrained=pretrained, **model_kwargs)

@register_model
def identityformer_s36(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        token_mixers=nn.Identity,
        **kwargs)
    return _create_metaformer('identityformer_s36', pretrained=pretrained, **model_kwargs)

@register_model
def identityformer_m36(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        token_mixers=nn.Identity,
        **kwargs)
    return _create_metaformer('identityformer_m36', pretrained=pretrained, **model_kwargs)

@register_model
def identityformer_m48(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        token_mixers=nn.Identity,
        **kwargs)
    return _create_metaformer('identityformer_m48', pretrained=pretrained, **model_kwargs)

@register_model
def randformer_s12(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        token_mixers=[nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
        **kwargs)
    return _create_metaformer('randformer_s12', pretrained=pretrained, **model_kwargs)

@register_model
def randformer_s24(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        token_mixers=[nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
        **kwargs)
    return _create_metaformer('randformer_s24', pretrained=pretrained, **model_kwargs)

@register_model
def randformer_s36(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        token_mixers=[nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
        **kwargs)
    return _create_metaformer('randformer_s36', pretrained=pretrained, **model_kwargs)

@register_model
def randformer_m36(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        token_mixers=[nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
        **kwargs)
    return _create_metaformer('randformer_m36', pretrained=pretrained, **model_kwargs)

@register_model
def randformer_m48(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        token_mixers=[nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
        **kwargs)
    return _create_metaformer('randformer_m48', pretrained=pretrained, **model_kwargs)

@register_model
def poolformerv2_s12(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        token_mixers=Pooling,
        **kwargs)
    return _create_metaformer('poolformerv2_s12', pretrained=pretrained, **model_kwargs)

@register_model
def poolformerv2_s24(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        token_mixers=Pooling,
        **kwargs)
    return _create_metaformer('poolformerv2_s24', pretrained=pretrained, **model_kwargs)



@register_model
def poolformerv2_s36(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        token_mixers=Pooling,
        **kwargs)
    return _create_metaformer('poolformerv2_s36', pretrained=pretrained, **model_kwargs)



@register_model
def poolformerv2_m36(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        token_mixers=Pooling,
        **kwargs)
    return _create_metaformer('poolformerv2_m36', pretrained=pretrained, **model_kwargs)


@register_model
def poolformerv2_m48(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        token_mixers=Pooling,
        **kwargs)
    return _create_metaformer('poolformerv2_m48', pretrained=pretrained, **model_kwargs)



@register_model
def convformer_s18(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    return _create_metaformer('convformer_s18', pretrained=pretrained, **model_kwargs)




@register_model
def convformer_s36(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    return _create_metaformer('convformer_s36', pretrained=pretrained, **model_kwargs)


@register_model
def convformer_m36(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    return _create_metaformer('convformer_m36', pretrained=pretrained, **model_kwargs)



@register_model
def convformer_b36(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    return _create_metaformer('convformer_b36', pretrained=pretrained, **model_kwargs)




@register_model
def caformer_s18(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    return _create_metaformer('caformer_s18', pretrained=pretrained, **model_kwargs)



@register_model
def caformer_s36(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    return _create_metaformer('caformer_s36', pretrained=pretrained, **model_kwargs)


@register_model
def caformer_m36(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    return _create_metaformer('caformer_m36', pretrained=pretrained, **model_kwargs)


@register_model
def caformer_b36(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    return _create_metaformer('caformer_b36', pretrained=pretrained, **model_kwargs)
