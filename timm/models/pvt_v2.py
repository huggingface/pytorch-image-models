""" Pyramid Vision Transformer v2

@misc{wang2021pvtv2,
      title={PVTv2: Improved Baselines with Pyramid Vision Transformer},
      author={Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and
        Tong Lu and Ping Luo and Ling Shao},
      year={2021},
      eprint={2106.13797},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Based on Apache 2.0 licensed code at https://github.com/whai362/PVT

Modifications and timm support by / Copyright 2022, Ross Wightman
"""

import math
from typing import Tuple, List, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, to_2tuple, to_ntuple, trunc_normal_, LayerNorm, use_fused_attn
from ._builder import build_model_with_cfg
from ._registry import register_model, generate_default_cfgs

__all__ = ['PyramidVisionTransformerV2']


class MlpWithDepthwiseConv(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.,
            extra_relu=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU() if extra_relu else nn.Identity()
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, feat_size: List[int]):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, feat_size[0], feat_size[1])
        x = self.relu(x)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            sr_ratio=1,
            linear_attn=False,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if not linear_attn:
            self.pool = None
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
            else:
                self.sr = None
                self.norm = None
            self.act = None
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

    def forward(self, x, feat_size: List[int]):
        B, N, C = x.shape
        H, W = feat_size
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        if self.pool is not None:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(self.pool(x)).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            x = self.act(x)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            if self.sr is not None:
                x = x.permute(0, 2, 1).reshape(B, C, H, W)
                x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
                x = self.norm(x)
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            sr_ratio=1,
            linear_attn=False,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            linear_attn=linear_attn,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MlpWithDepthwiseConv(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            extra_relu=linear_attn,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, feat_size: List[int]):
        x = x + self.drop_path1(self.attn(self.norm1(x), feat_size))
        x = x + self.drop_path2(self.mlp(self.norm2(x), feat_size))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, patch_size,
            stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class PyramidVisionTransformerStage(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            depth: int,
            downsample: bool = True,
            num_heads: int = 8,
            sr_ratio: int = 1,
            linear_attn: bool = False,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[List[float], float] = 0.0,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.grad_checkpointing = False

        if downsample:
            self.downsample = OverlapPatchEmbed(
                patch_size=3,
                stride=2,
                in_chans=dim,
                embed_dim=dim_out,
            )
        else:
            assert dim == dim_out
            self.downsample = None

        self.blocks = nn.ModuleList([Block(
            dim=dim_out,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            linear_attn=linear_attn,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
        ) for i in range(depth)])

        self.norm = norm_layer(dim_out)

    def forward(self, x):
        # x is either B, C, H, W (if downsample) or B, H, W, C if not
        if self.downsample is not None:
            # input to downsample is B, C, H, W
            x = self.downsample(x)  # output B, H, W, C
        B, H, W, C = x.shape
        feat_size = (H, W)
        x = x.reshape(B, -1, C)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x, feat_size)
            else:
                x = blk(x, feat_size)
        x = self.norm(x)
        x = x.reshape(B, feat_size[0], feat_size[1], -1).permute(0, 3, 1, 2).contiguous()
        return x


class PyramidVisionTransformerV2(nn.Module):
    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            depths=(3, 4, 6, 3),
            embed_dims=(64, 128, 256, 512),
            num_heads=(1, 2, 4, 8),
            sr_ratios=(8, 4, 2, 1),
            mlp_ratios=(8., 8., 4., 4.),
            qkv_bias=True,
            linear=False,
            drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=LayerNorm,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert global_pool in ('avg', '')
        self.global_pool = global_pool
        self.depths = depths
        num_stages = len(depths)
        mlp_ratios = to_ntuple(num_stages)(mlp_ratios)
        num_heads = to_ntuple(num_stages)(num_heads)
        sr_ratios = to_ntuple(num_stages)(sr_ratios)
        assert(len(embed_dims)) == num_stages
        self.feature_info = []

        self.patch_embed = OverlapPatchEmbed(
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        cur = 0
        prev_dim = embed_dims[0]
        stages = []
        for i in range(num_stages):
            stages += [PyramidVisionTransformerStage(
                dim=prev_dim,
                dim_out=embed_dims[i],
                depth=depths[i],
                downsample=i > 0,
                num_heads=num_heads[i],
                sr_ratio=sr_ratios[i],
                mlp_ratio=mlp_ratios[i],
                linear_attn=linear,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )]
            prev_dim = embed_dims[i]
            cur += depths[i]
            self.feature_info += [dict(num_chs=prev_dim, reduction=4 * 2**i, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        # classification head
        self.num_features = embed_dims[-1]
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^patch_embed',  # stem and embed
            blocks=r'^stages\.(\d+)'
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('avg', '')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x.mean(dim=(-1, -2))
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _checkpoint_filter_fn(state_dict, model):
    """ Remap original checkpoints -> timm """
    if 'patch_embed.proj.weight' in state_dict:
        return state_dict  # non-original checkpoint, no remapping needed

    out_dict = {}
    import re
    for k, v in state_dict.items():
        if k.startswith('patch_embed'):
            k = k.replace('patch_embed1', 'patch_embed')
            k = k.replace('patch_embed2', 'stages.1.downsample')
            k = k.replace('patch_embed3', 'stages.2.downsample')
            k = k.replace('patch_embed4', 'stages.3.downsample')
        k = k.replace('dwconv.dwconv', 'dwconv')
        k = re.sub(r'block(\d+).(\d+)', lambda x: f'stages.{int(x.group(1)) - 1}.blocks.{x.group(2)}', k)
        k = re.sub(r'^norm(\d+)', lambda x: f'stages.{int(x.group(1)) - 1}.norm', k)
        out_dict[k] = v
    return out_dict


def _create_pvt2(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(range(4))
    out_indices = kwargs.pop('out_indices', default_out_indices)
    model = build_model_with_cfg(
        PyramidVisionTransformerV2,
        variant,
        pretrained,
        pretrained_filter_fn=_checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs,
    )
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head', 'fixed_input_size': False,
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'pvt_v2_b0.in1k': _cfg(hf_hub_id='timm/'),
    'pvt_v2_b1.in1k': _cfg(hf_hub_id='timm/'),
    'pvt_v2_b2.in1k': _cfg(hf_hub_id='timm/'),
    'pvt_v2_b3.in1k': _cfg(hf_hub_id='timm/'),
    'pvt_v2_b4.in1k': _cfg(hf_hub_id='timm/'),
    'pvt_v2_b5.in1k': _cfg(hf_hub_id='timm/'),
    'pvt_v2_b2_li.in1k': _cfg(hf_hub_id='timm/'),
})


@register_model
def pvt_v2_b0(pretrained=False, **kwargs) -> PyramidVisionTransformerV2:
    model_args = dict(depths=(2, 2, 2, 2), embed_dims=(32, 64, 160, 256), num_heads=(1, 2, 5, 8))
    return _create_pvt2('pvt_v2_b0', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def pvt_v2_b1(pretrained=False, **kwargs) -> PyramidVisionTransformerV2:
    model_args = dict(depths=(2, 2, 2, 2), embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8))
    return _create_pvt2('pvt_v2_b1', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def pvt_v2_b2(pretrained=False, **kwargs) -> PyramidVisionTransformerV2:
    model_args = dict(depths=(3, 4, 6, 3), embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8))
    return _create_pvt2('pvt_v2_b2', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def pvt_v2_b3(pretrained=False, **kwargs) -> PyramidVisionTransformerV2:
    model_args = dict(depths=(3, 4, 18, 3), embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8))
    return _create_pvt2('pvt_v2_b3', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def pvt_v2_b4(pretrained=False, **kwargs) -> PyramidVisionTransformerV2:
    model_args = dict(depths=(3, 8, 27, 3), embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8))
    return _create_pvt2('pvt_v2_b4', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def pvt_v2_b5(pretrained=False, **kwargs) -> PyramidVisionTransformerV2:
    model_args = dict(
        depths=(3, 6, 40, 3), embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8), mlp_ratios=(4, 4, 4, 4))
    return _create_pvt2('pvt_v2_b5', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def pvt_v2_b2_li(pretrained=False, **kwargs) -> PyramidVisionTransformerV2:
    model_args = dict(
        depths=(3, 4, 6, 3), embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8), linear=True)
    return _create_pvt2('pvt_v2_b2_li', pretrained=pretrained, **dict(model_args, **kwargs))

