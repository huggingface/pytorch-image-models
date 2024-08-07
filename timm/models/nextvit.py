""" Next-ViT

As described in https://arxiv.org/abs/2207.05501

Next-ViT model defs and weights adapted from https://github.com/bytedance/Next-ViT, original copyright below
"""
# Copyright (c) ByteDance Inc. All rights reserved.
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_, ConvMlp, get_norm_layer, get_act_layer, use_fused_attn
from timm.layers import ClassifierHead
from ._builder import build_model_with_cfg
from ._features_fx import register_notrace_function
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['NextViT']


def merge_pre_bn(module, pre_bn_1, pre_bn_2=None):
    """ Merge pre BN to reduce inference runtime.
    """
    weight = module.weight.data
    if module.bias is None:
        zeros = torch.zeros(module.out_chs, device=weight.device).type(weight.type())
        module.bias = nn.Parameter(zeros)
    bias = module.bias.data
    if pre_bn_2 is None:
        assert pre_bn_1.track_running_stats is True, "Unsupported bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupported bn_module.affine is False"

        scale_invstd = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        extra_weight = scale_invstd * pre_bn_1.weight
        extra_bias = pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd
    else:
        assert pre_bn_1.track_running_stats is True, "Unsupported bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupported bn_module.affine is False"

        assert pre_bn_2.track_running_stats is True, "Unsupported bn_module.track_running_stats is False"
        assert pre_bn_2.affine is True, "Unsupported bn_module.affine is False"

        scale_invstd_1 = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        scale_invstd_2 = pre_bn_2.running_var.add(pre_bn_2.eps).pow(-0.5)

        extra_weight = scale_invstd_1 * pre_bn_1.weight * scale_invstd_2 * pre_bn_2.weight
        extra_bias = (
                scale_invstd_2 * pre_bn_2.weight
                * (pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd_1 - pre_bn_2.running_mean)
                + pre_bn_2.bias
        )

    if isinstance(module, nn.Linear):
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
    elif isinstance(module, nn.Conv2d):
        assert weight.shape[2] == 1 and weight.shape[3] == 1
        weight = weight.reshape(weight.shape[0], weight.shape[1])
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
        weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1)
    bias.add_(extra_bias)

    module.weight.data = weight
    module.bias.data = bias


class ConvNormAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=3,
            stride=1,
            groups=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.ReLU,
    ):
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size=kernel_size, stride=stride,
            padding=1, groups=groups, bias=False)
        self.norm = norm_layer(out_chs)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PatchEmbed(nn.Module):
    def __init__(self,
            in_chs,
            out_chs,
            stride=1,
            norm_layer = nn.BatchNorm2d,
    ):
        super(PatchEmbed, self).__init__()

        if stride == 2:
            self.pool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_chs)
        elif in_chs != out_chs:
            self.pool = nn.Identity()
            self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_chs)
        else:
            self.pool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.pool(x)))


class ConvAttention(nn.Module):
    """
    Multi-Head Convolutional Attention
    """

    def __init__(self, out_chs, head_dim, norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU):
        super(ConvAttention, self).__init__()
        self.group_conv3x3 = nn.Conv2d(
            out_chs, out_chs,
            kernel_size=3, stride=1, padding=1, groups=out_chs // head_dim, bias=False
        )
        self.norm = norm_layer(out_chs)
        self.act = act_layer()
        self.projection = nn.Conv2d(out_chs, out_chs, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out

class NextConvBlock(nn.Module):
    """
    Next Convolution Block
    """

    def __init__(
            self,
            in_chs,
            out_chs,
            stride=1,
            drop_path=0.,
            drop=0.,
            head_dim=32,
            mlp_ratio=3.,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.ReLU
    ):
        super(NextConvBlock, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        assert out_chs % head_dim == 0

        self.patch_embed = PatchEmbed(in_chs, out_chs, stride, norm_layer=norm_layer)
        self.mhca = ConvAttention(
            out_chs,
            head_dim,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.attn_drop_path = DropPath(drop_path)

        self.norm = norm_layer(out_chs)
        self.mlp = ConvMlp(
            out_chs,
            hidden_features=int(out_chs * mlp_ratio),
            drop=drop,
            bias=True,
            act_layer=act_layer,
        )
        self.mlp_drop_path = DropPath(drop_path)
        self.is_fused = False

    @torch.no_grad()
    def reparameterize(self):
        if not self.is_fused:
            merge_pre_bn(self.mlp.fc1, self.norm)
            self.norm = nn.Identity()
            self.is_fused = True

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.attn_drop_path(self.mhca(x))

        out = self.norm(x)
        x = x + self.mlp_drop_path(self.mlp(out))
        return x


class EfficientAttention(nn.Module):
    """
    Efficient Multi-Head Self Attention
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim,
            out_dim=None,
            head_dim=32,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.,
            sr_ratio=1,
            norm_layer=nn.BatchNorm1d,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio ** 2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = norm_layer(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x = self.sr(x.transpose(1, 2))
            x = self.norm(x).transpose(1, 2)

        k = self.k(x).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-1, -2)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NextTransformerBlock(nn.Module):
    """
    Next Transformer Block
    """

    def __init__(
            self,
            in_chs,
            out_chs,
            drop_path,
            stride=1,
            sr_ratio=1,
            mlp_ratio=2,
            head_dim=32,
            mix_block_ratio=0.75,
            attn_drop=0.,
            drop=0.,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.ReLU,
    ):
        super(NextTransformerBlock, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.mix_block_ratio = mix_block_ratio

        self.mhsa_out_chs = _make_divisible(int(out_chs * mix_block_ratio), 32)
        self.mhca_out_chs = out_chs - self.mhsa_out_chs

        self.patch_embed = PatchEmbed(in_chs, self.mhsa_out_chs, stride)
        self.norm1 = norm_layer(self.mhsa_out_chs)
        self.e_mhsa = EfficientAttention(
            self.mhsa_out_chs,
            head_dim=head_dim,
            sr_ratio=sr_ratio,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.mhsa_drop_path = DropPath(drop_path * mix_block_ratio)

        self.projection = PatchEmbed(self.mhsa_out_chs, self.mhca_out_chs, stride=1, norm_layer=norm_layer)
        self.mhca = ConvAttention(
            self.mhca_out_chs,
            head_dim=head_dim,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.mhca_drop_path = DropPath(drop_path * (1 - mix_block_ratio))

        self.norm2 = norm_layer(out_chs)
        self.mlp = ConvMlp(
            out_chs,
            hidden_features=int(out_chs * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_drop_path = DropPath(drop_path)
        self.is_fused = False

    @torch.no_grad()
    def reparameterize(self):
        if not self.is_fused:
            merge_pre_bn(self.e_mhsa.q, self.norm1)
            if self.e_mhsa.norm is not None:
                merge_pre_bn(self.e_mhsa.k, self.norm1, self.e_mhsa.norm)
                merge_pre_bn(self.e_mhsa.v, self.norm1, self.e_mhsa.norm)
                self.e_mhsa.norm = nn.Identity()
            else:
                merge_pre_bn(self.e_mhsa.k, self.norm1)
                merge_pre_bn(self.e_mhsa.v, self.norm1)
            self.norm1 = nn.Identity()

            merge_pre_bn(self.mlp.fc1, self.norm2)
            self.norm2 = nn.Identity()
            self.is_fused = True

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape

        out = self.norm1(x)
        out = out.reshape(B, C, -1).transpose(-1, -2)
        out = self.mhsa_drop_path(self.e_mhsa(out))
        x = x + out.transpose(-1, -2).reshape(B, C, H, W)

        out = self.projection(x)
        out = out + self.mhca_drop_path(self.mhca(out))
        x = torch.cat([x, out], dim=1)

        out = self.norm2(x)
        x = x + self.mlp_drop_path(self.mlp(out))
        return x


class NextStage(nn.Module):

    def __init__(
            self,
            in_chs,
            block_chs,
            block_types,
            stride=2,
            sr_ratio=1,
            mix_block_ratio=1.0,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            head_dim=32,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.ReLU,
    ):
        super().__init__()
        self.grad_checkpointing = False

        blocks = []
        for block_idx, block_type in enumerate(block_types):
            stride = stride if block_idx == 0 else 1
            out_chs = block_chs[block_idx]
            block_type = block_types[block_idx]
            dpr = drop_path[block_idx] if isinstance(drop_path, (list, tuple)) else drop_path
            if block_type is NextConvBlock:
                layer = NextConvBlock(
                    in_chs,
                    out_chs,
                    stride=stride,
                    drop_path=dpr,
                    drop=drop,
                    head_dim=head_dim,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                blocks.append(layer)
            elif block_type is NextTransformerBlock:
                layer = NextTransformerBlock(
                    in_chs,
                    out_chs,
                    drop_path=dpr,
                    stride=stride,
                    sr_ratio=sr_ratio,
                    head_dim=head_dim,
                    mix_block_ratio=mix_block_ratio,
                    attn_drop=attn_drop,
                    drop=drop,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                blocks.append(layer)
            in_chs = out_chs

        self.blocks = nn.Sequential(*blocks)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, x):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class NextViT(nn.Module):
    def __init__(
            self,
            in_chans,
            num_classes=1000,
            global_pool='avg',
            stem_chs=(64, 32, 64),
            depths=(3, 4, 10, 3),
            strides=(1, 2, 2, 2),
            sr_ratios=(8, 4, 2, 1),
            drop_path_rate=0.1,
            attn_drop_rate=0.,
            drop_rate=0.,
            head_dim=32,
            mix_block_ratio=0.75,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
    ):
        super(NextViT, self).__init__()
        self.grad_checkpointing = False
        self.num_classes = num_classes
        norm_layer = get_norm_layer(norm_layer)
        if act_layer is None:
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            act_layer = get_act_layer(act_layer)

        self.stage_out_chs = [
            [96] * (depths[0]),
            [192] * (depths[1] - 1) + [256],
            [384, 384, 384, 384, 512] * (depths[2] // 5),
            [768] * (depths[3] - 1) + [1024]
        ]
        self.feature_info = [dict(
            num_chs=sc[-1],
            reduction=2**(i + 2),
            module=f'stages.{i}'
        ) for i, sc in enumerate(self.stage_out_chs)]

        # Next Hybrid Strategy
        self.stage_block_types = [
            [NextConvBlock] * depths[0],
            [NextConvBlock] * (depths[1] - 1) + [NextTransformerBlock],
            [NextConvBlock, NextConvBlock, NextConvBlock, NextConvBlock, NextTransformerBlock] * (depths[2] // 5),
            [NextConvBlock] * (depths[3] - 1) + [NextTransformerBlock]]

        self.stem = nn.Sequential(
            ConvNormAct(in_chans, stem_chs[0], kernel_size=3, stride=2, norm_layer=norm_layer, act_layer=act_layer),
            ConvNormAct(stem_chs[0], stem_chs[1], kernel_size=3, stride=1, norm_layer=norm_layer, act_layer=act_layer),
            ConvNormAct(stem_chs[1], stem_chs[2], kernel_size=3, stride=1, norm_layer=norm_layer, act_layer=act_layer),
            ConvNormAct(stem_chs[2], stem_chs[2], kernel_size=3, stride=2, norm_layer=norm_layer, act_layer=act_layer),
        )
        in_chs = out_chs = stem_chs[-1]
        stages = []
        idx = 0
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        for stage_idx in range(len(depths)):
            stage = NextStage(
                in_chs=in_chs,
                block_chs=self.stage_out_chs[stage_idx],
                block_types=self.stage_block_types[stage_idx],
                stride=strides[stage_idx],
                sr_ratio=sr_ratios[stage_idx],
                mix_block_ratio=mix_block_ratio,
                head_dim=head_dim,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[stage_idx],
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            in_chs = out_chs = self.stage_out_chs[stage_idx][-1]
            stages += [stage]
            idx += depths[stage_idx]
        self.num_features = self.head_hidden_size = out_chs
        self.stages = nn.Sequential(*stages)
        self.norm = norm_layer(out_chs)
        self.head = ClassifierHead(pool_type=global_pool, in_features=out_chs, num_classes=num_classes)

        self.stage_out_idx = [sum(depths[:idx + 1]) - 1 for idx in range(len(depths))]
        self._initialize_weights()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',  # stem and embed
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
        for stage in self.stages:
            stage.set_grad_checkpointing(enable=enable)

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    """ Remap original checkpoints -> timm """
    if 'head.fc.weight' in state_dict:
        return state_dict  # non-original

    D = model.state_dict()
    out_dict = {}
    # remap originals based on order
    for ka, kb, va, vb in zip(D.keys(), state_dict.keys(), D.values(), state_dict.values()):
        out_dict[ka] = vb

    return out_dict


def _create_nextvit(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 1, 3, 1))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        NextViT,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)

    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0.conv', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'nextvit_small.bd_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'nextvit_base.bd_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'nextvit_large.bd_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'nextvit_small.bd_in1k_384': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),
    'nextvit_base.bd_in1k_384': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),
    'nextvit_large.bd_in1k_384': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),

    'nextvit_small.bd_ssld_6m_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'nextvit_base.bd_ssld_6m_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'nextvit_large.bd_ssld_6m_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'nextvit_small.bd_ssld_6m_in1k_384': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),
    'nextvit_base.bd_ssld_6m_in1k_384': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),
    'nextvit_large.bd_ssld_6m_in1k_384': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),
})


@register_model
def nextvit_small(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 4, 10, 3), drop_path_rate=0.1)
    model = _create_nextvit(
        'nextvit_small', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def nextvit_base(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 4, 20, 3), drop_path_rate=0.2)
    model = _create_nextvit(
        'nextvit_base', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def nextvit_large(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 4, 30, 3), drop_path_rate=0.2)
    model = _create_nextvit(
        'nextvit_large', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
