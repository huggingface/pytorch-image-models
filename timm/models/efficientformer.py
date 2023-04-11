""" EfficientFormer

@article{li2022efficientformer,
  title={EfficientFormer: Vision Transformers at MobileNet Speed},
  author={Li, Yanyu and Yuan, Geng and Wen, Yang and Hu, Eric and Evangelidis, Georgios and Tulyakov,
   Sergey and Wang, Yanzhi and Ren, Jian},
  journal={arXiv preprint arXiv:2206.01191},
  year={2022}
}

Based on Apache 2.0 licensed code at https://github.com/snap-research/EfficientFormer, Copyright (c) 2022 Snap Inc.

Modifications and timm support by / Copyright 2022, Ross Wightman
"""
from typing import Dict

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_, to_2tuple, Mlp
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['EfficientFormer']  # model_registry will add each entrypoint fn to this


EfficientFormer_width = {
    'l1': (48, 96, 224, 448),
    'l3': (64, 128, 320, 512),
    'l7': (96, 192, 384, 768),
}

EfficientFormer_depth = {
    'l1': (3, 2, 6, 4),
    'l3': (4, 4, 12, 6),
    'l7': (6, 6, 18, 8),
}


class Attention(torch.nn.Module):
    attention_bias_cache: Dict[str, torch.Tensor]

    def __init__(
            self,
            dim=384,
            key_dim=32,
            num_heads=8,
            attn_ratio=4,
            resolution=7
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.key_attn_dim = key_dim * num_heads
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = self.val_dim * num_heads
        self.attn_ratio = attn_ratio

        self.qkv = nn.Linear(dim, self.key_attn_dim * 2 + self.val_attn_dim)
        self.proj = nn.Linear(self.val_attn_dim, dim)

        resolution = to_2tuple(resolution)
        pos = torch.stack(torch.meshgrid(torch.arange(resolution[0]), torch.arange(resolution[1]))).flatten(1)
        rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * resolution[1]) + rel_pos[1]
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, resolution[0] * resolution[1]))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(rel_pos))
        self.attention_bias_cache = {}  # per-device attention_biases cache (data-parallel compat)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # clear ab cache

    def get_attention_biases(self, device: torch.device) -> torch.Tensor:
        if torch.jit.is_tracing() or self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.val_dim], dim=3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.get_attention_biases(x.device)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        x = self.proj(x)
        return x


class Stem4(nn.Sequential):
    def __init__(self, in_chs, out_chs, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = 4

        self.add_module('conv1', nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1))
        self.add_module('norm1', norm_layer(out_chs // 2))
        self.add_module('act1', act_layer())
        self.add_module('conv2', nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1))
        self.add_module('norm2', norm_layer(out_chs))
        self.add_module('act2', act_layer())


class Downsample(nn.Module):
    """
    Downsampling via strided conv w/ norm
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, in_chs, out_chs, kernel_size=3, stride=2, padding=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm_layer(out_chs)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class Flat(nn.Module):

    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class ConvMlpWithNorm(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=nn.BatchNorm2d,
            drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.norm1 = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.norm2 = norm_layer(out_features) if norm_layer is not None else nn.Identity()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class MetaBlock1d(nn.Module):

    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            proj_drop=0.,
            drop_path=0.,
            layer_scale_init_value=1e-5
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Attention(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls1 = LayerScale(dim, layer_scale_init_value)
        self.ls2 = LayerScale(dim, layer_scale_init_value)

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.token_mixer(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class LayerScale2d(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


class MetaBlock2d(nn.Module):

    def __init__(
            self,
            dim,
            pool_size=3,
            mlp_ratio=4.,
            act_layer=nn.GELU,
            norm_layer=nn.BatchNorm2d,
            proj_drop=0.,
            drop_path=0.,
            layer_scale_init_value=1e-5
    ):
        super().__init__()
        self.token_mixer = Pooling(pool_size=pool_size)
        self.ls1 = LayerScale2d(dim, layer_scale_init_value)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = ConvMlpWithNorm(
            dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale2d(dim, layer_scale_init_value)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.token_mixer(x)))
        x = x + self.drop_path2(self.ls2(self.mlp(x)))
        return x


class EfficientFormerStage(nn.Module):

    def __init__(
            self,
            dim,
            dim_out,
            depth,
            downsample=True,
            num_vit=1,
            pool_size=3,
            mlp_ratio=4.,
            act_layer=nn.GELU,
            norm_layer=nn.BatchNorm2d,
            norm_layer_cl=nn.LayerNorm,
            proj_drop=.0,
            drop_path=0.,
            layer_scale_init_value=1e-5,
):
        super().__init__()
        self.grad_checkpointing = False

        if downsample:
            self.downsample = Downsample(in_chs=dim, out_chs=dim_out, norm_layer=norm_layer)
            dim = dim_out
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        blocks = []
        if num_vit and num_vit >= depth:
            blocks.append(Flat())

        for block_idx in range(depth):
            remain_idx = depth - block_idx - 1
            if num_vit and num_vit > remain_idx:
                blocks.append(
                    MetaBlock1d(
                        dim,
                        mlp_ratio=mlp_ratio,
                        act_layer=act_layer,
                        norm_layer=norm_layer_cl,
                        proj_drop=proj_drop,
                        drop_path=drop_path[block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    ))
            else:
                blocks.append(
                    MetaBlock2d(
                        dim,
                        pool_size=pool_size,
                        mlp_ratio=mlp_ratio,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        proj_drop=proj_drop,
                        drop_path=drop_path[block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    ))
                if num_vit and num_vit == remain_idx:
                    blocks.append(Flat())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class EfficientFormer(nn.Module):

    def __init__(
            self,
            depths,
            embed_dims=None,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            downsamples=None,
            num_vit=0,
            mlp_ratios=4,
            pool_size=3,
            layer_scale_init_value=1e-5,
            act_layer=nn.GELU,
            norm_layer=nn.BatchNorm2d,
            norm_layer_cl=nn.LayerNorm,
            drop_rate=0.,
            proj_drop_rate=0.,
            drop_path_rate=0.,
            **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool

        self.stem = Stem4(in_chans, embed_dims[0], norm_layer=norm_layer)
        prev_dim = embed_dims[0]

        # stochastic depth decay rule
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        downsamples = downsamples or (False,) + (True,) * (len(depths) - 1)
        stages = []
        for i in range(len(depths)):
            stage = EfficientFormerStage(
                prev_dim,
                embed_dims[i],
                depths[i],
                downsample=downsamples[i],
                num_vit=num_vit if i == 3 else 0,
                pool_size=pool_size,
                mlp_ratio=mlp_ratios,
                act_layer=act_layer,
                norm_layer_cl=norm_layer_cl,
                norm_layer=norm_layer,
                proj_drop=proj_drop_rate,
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
            )
            prev_dim = embed_dims[i]
            stages.append(stage)

        self.stages = nn.Sequential(*stages)

        # Classifier head
        self.num_features = embed_dims[-1]
        self.norm = norm_layer_cl(self.num_features)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # assuming model is always distilled (valid for current checkpoints, will split def if that changes)
        self.head_dist = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.distilled_training = False  # must set this True to train w/ distillation token

        self.apply(self._init_weights)

    # init for classification
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {k for k, _ in self.named_parameters() if 'attention_biases' in k}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',  # stem and embed
            blocks=[(r'^stages\.(\d+)', None), (r'^norm', (99999,))]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        x = self.head_drop(x)
        if pre_logits:
            return x
        x, x_dist = self.head(x), self.head_dist(x)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train/finetune, inference average the classifier predictions
            return (x + x_dist) / 2

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _checkpoint_filter_fn(state_dict, model):
    """ Remap original checkpoints -> timm """
    if 'stem.0.weight' in state_dict:
        return state_dict  # non-original checkpoint, no remapping needed

    out_dict = {}
    import re
    stage_idx = 0
    for k, v in state_dict.items():
        if k.startswith('patch_embed'):
            k = k.replace('patch_embed.0', 'stem.conv1')
            k = k.replace('patch_embed.1', 'stem.norm1')
            k = k.replace('patch_embed.3', 'stem.conv2')
            k = k.replace('patch_embed.4', 'stem.norm2')

        if re.match(r'network\.(\d+)\.proj\.weight', k):
            stage_idx += 1
        k = re.sub(r'network.(\d+).(\d+)', f'stages.{stage_idx}.blocks.\\2', k)
        k = re.sub(r'network.(\d+).proj', f'stages.{stage_idx}.downsample.conv', k)
        k = re.sub(r'network.(\d+).norm', f'stages.{stage_idx}.downsample.norm', k)

        k = re.sub(r'layer_scale_([0-9])', r'ls\1.gamma', k)
        k = k.replace('dist_head', 'head_dist')
        out_dict[k] = v
    return out_dict


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None, 'fixed_input_size': True,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1', 'classifier': ('head', 'head_dist'),
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'efficientformer_l1.snap_dist_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'efficientformer_l3.snap_dist_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'efficientformer_l7.snap_dist_in1k': _cfg(
        hf_hub_id='timm/',
    ),
})


def _create_efficientformer(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        EfficientFormer, variant, pretrained,
        pretrained_filter_fn=_checkpoint_filter_fn,
        **kwargs)
    return model


@register_model
def efficientformer_l1(pretrained=False, **kwargs):
    model_args = dict(
        depths=EfficientFormer_depth['l1'],
        embed_dims=EfficientFormer_width['l1'],
        num_vit=1,
    )
    return _create_efficientformer('efficientformer_l1', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def efficientformer_l3(pretrained=False, **kwargs):
    model_args = dict(
        depths=EfficientFormer_depth['l3'],
        embed_dims=EfficientFormer_width['l3'],
        num_vit=4,
    )
    return _create_efficientformer('efficientformer_l3', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def efficientformer_l7(pretrained=False, **kwargs):
    model_args = dict(
        depths=EfficientFormer_depth['l7'],
        embed_dims=EfficientFormer_width['l7'],
        num_vit=8,
    )
    return _create_efficientformer('efficientformer_l7', pretrained=pretrained, **dict(model_args, **kwargs))

