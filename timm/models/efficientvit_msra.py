""" EfficientViT (by MSRA)

Paper: `EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention`
    - https://arxiv.org/abs/2305.07027

Adapted from official impl at https://github.com/microsoft/Cream/tree/main/EfficientViT
"""

__all__ = ['EfficientVitMsra']
import itertools
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SqueezeExcite, SelectAdaptivePool2d, trunc_normal_, _assert
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs


class ConvNorm(torch.nn.Sequential):
    def __init__(self, in_chs, out_chs, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, ks, stride, pad, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_chs)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self.conv, self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.conv.groups, w.size(0), w.shape[2:],
            stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class NormLinear(torch.nn.Sequential):
    def __init__(self, in_features, out_features, bias=True, std=0.02, drop=0.):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.drop = nn.Dropout(drop)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        trunc_normal_(self.linear.weight, std=std)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, linear = self.bn, self.linear
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = linear.weight * w[None, :]
        if linear.bias is None:
            b = b @ self.linear.weight.T
        else:
            b = (linear.weight @ b[:, None]).view(-1) + self.linear.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = ConvNorm(dim, hid_dim, 1, 1, 0)
        self.act = torch.nn.ReLU()
        self.conv2 = ConvNorm(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = ConvNorm(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class ResidualDrop(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(
                x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class ConvMlp(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = ConvNorm(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = ConvNorm(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class CascadedGroupAttention(torch.nn.Module):
    attention_bias_cache: Dict[str, torch.Tensor]

    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4,
            resolution=14,
            kernels=(5, 5, 5, 5),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.val_dim = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(ConvNorm(dim // (num_heads), self.key_dim * 2 + self.val_dim))
            dws.append(ConvNorm(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(
            torch.nn.ReLU(),
            ConvNorm(self.val_dim * num_heads, dim, bn_weight_init=0)
        )

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N), persistent=False)
        self.attention_bias_cache = {}

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

    def forward(self, x):
        B, C, H, W = x.shape
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        attn_bias = self.get_attention_biases(x.device)
        for head_idx, (qkv, dws) in enumerate(zip(self.qkvs, self.dws)):
            if head_idx > 0:
                feat = feat + feats_in[head_idx]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.val_dim], dim=1)
            q = dws(q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
            q = q * self.scale
            attn = q.transpose(-2, -1) @ k
            attn = attn + attn_bias[head_idx]
            attn = attn.softmax(dim=-1)
            feat = v @ attn.transpose(-2, -1)
            feat = feat.view(B, self.val_dim, H, W)
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x


class LocalWindowAttention(torch.nn.Module):
    r""" Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4,
            resolution=14,
            window_resolution=7,
            kernels=(5, 5, 5, 5),
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution
        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(
            dim, key_dim, num_heads,
            attn_ratio=attn_ratio,
            resolution=window_resolution,
            kernels=kernels,
        )

    def forward(self, x):
        H = W = self.resolution
        B, C, H_, W_ = x.shape
        # Only check this for classifcation models
        _assert(H == H_, f'input feature has wrong size, expect {(H, W)}, got {(H_, W_)}')
        _assert(W == W_, f'input feature has wrong size, expect {(H, W)}, got {(H_, W_)}')
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H % self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W % self.window_resolution) % self.window_resolution
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3)
            x = x.reshape(B * nH * nW, self.window_resolution, self.window_resolution, C).permute(0, 3, 1, 2)
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution, C)
            x = x.transpose(2, 3).reshape(B, pH, pW, C)
            x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)
        return x


class EfficientVitBlock(torch.nn.Module):
    """ A basic EfficientVit building block.

    Args:
        dim (int): Number of input channels.
        key_dim (int): Dimension for query and key in the token mixer.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4,
            resolution=14,
            window_resolution=7,
            kernels=[5, 5, 5, 5],
    ):
        super().__init__()

        self.dw0 = ResidualDrop(ConvNorm(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0.))
        self.ffn0 = ResidualDrop(ConvMlp(dim, int(dim * 2)))

        self.mixer = ResidualDrop(
            LocalWindowAttention(
                dim, key_dim, num_heads,
                attn_ratio=attn_ratio,
                resolution=resolution,
                window_resolution=window_resolution,
                kernels=kernels,
            )
        )

        self.dw1 = ResidualDrop(ConvNorm(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0.))
        self.ffn1 = ResidualDrop(ConvMlp(dim, int(dim * 2)))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


class EfficientVitStage(torch.nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            key_dim,
            downsample=('', 1),
            num_heads=8,
            attn_ratio=4,
            resolution=14,
            window_resolution=7,
            kernels=[5, 5, 5, 5],
            depth=1,
    ):
        super().__init__()
        if downsample[0] == 'subsample':
            self.resolution = (resolution - 1) // downsample[1] + 1
            down_blocks = []
            down_blocks.append((
                'res1',
                torch.nn.Sequential(
                    ResidualDrop(ConvNorm(in_dim, in_dim, 3, 1, 1, groups=in_dim)),
                    ResidualDrop(ConvMlp(in_dim, int(in_dim * 2))),
                )
            ))
            down_blocks.append(('patchmerge', PatchMerging(in_dim, out_dim)))
            down_blocks.append((
                'res2',
                torch.nn.Sequential(
                    ResidualDrop(ConvNorm(out_dim, out_dim, 3, 1, 1, groups=out_dim)),
                    ResidualDrop(ConvMlp(out_dim, int(out_dim * 2))),
                )
            ))
            self.downsample = nn.Sequential(OrderedDict(down_blocks))
        else:
            assert in_dim == out_dim
            self.downsample = nn.Identity()
            self.resolution = resolution

        blocks = []
        for d in range(depth):
            blocks.append(EfficientVitBlock(out_dim, key_dim, num_heads, attn_ratio, self.resolution, window_resolution, kernels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class PatchEmbedding(torch.nn.Sequential):
    def __init__(self, in_chans, dim):
        super().__init__()
        self.add_module('conv1', ConvNorm(in_chans, dim // 8, 3, 2, 1))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('conv2', ConvNorm(dim // 8, dim // 4, 3, 2, 1))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('conv3', ConvNorm(dim // 4, dim // 2, 3, 2, 1))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('conv4', ConvNorm(dim // 2, dim, 3, 2, 1))
        self.patch_size = 16


class EfficientVitMsra(nn.Module):
    def __init__(
            self,
            img_size=224,
            in_chans=3,
            num_classes=1000,
            embed_dim=(64, 128, 192),
            key_dim=(16, 16, 16),
            depth=(1, 2, 3),
            num_heads=(4, 4, 4),
            window_size=(7, 7, 7),
            kernels=(5, 5, 5, 5),
            down_ops=(('', 1), ('subsample', 2), ('subsample', 2)),
            global_pool='avg',
            drop_rate=0.,
    ):
        super(EfficientVitMsra, self).__init__()
        self.grad_checkpointing = False
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        # Patch embedding
        self.patch_embed = PatchEmbedding(in_chans, embed_dim[0])
        stride = self.patch_embed.patch_size
        resolution = img_size // self.patch_embed.patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]

        # Build EfficientVit blocks
        self.feature_info = []
        stages = []
        pre_ed = embed_dim[0]
        for i, (ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            stage = EfficientVitStage(
                in_dim=pre_ed,
                out_dim=ed,
                key_dim=kd,
                downsample=do,
                num_heads=nh,
                attn_ratio=ar,
                resolution=resolution,
                window_resolution=wd,
                kernels=kernels,
                depth=dpth,
            )
            pre_ed = ed
            if do[0] == 'subsample' and i != 0:
                stride *= do[1]
            resolution = stage.resolution
            stages.append(stage)
            self.feature_info += [dict(num_chs=ed, reduction=stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        if global_pool == 'avg':
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        else:
            assert num_classes == 0
            self.global_pool = nn.Identity()
        self.num_features = embed_dim[-1]
        self.head = NormLinear(
            self.num_features, num_classes, drop=self.drop_rate) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^patch_embed',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+).downsample', (0,)),
                (r'^stages\.(\d+)\.\w+\.(\d+)', None),
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.linear

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            if global_pool == 'avg':
                self.global_pool = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
            else:
                assert num_classes == 0
                self.global_pool = nn.Identity()
        self.head = NormLinear(
            self.num_features, num_classes, drop=self.drop_rate) if num_classes > 0 else torch.nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


# def checkpoint_filter_fn(state_dict, model):
#     if 'model' in state_dict.keys():
#         state_dict = state_dict['model']
#     tmp_dict = {}
#     out_dict = {}
#     target_keys = model.state_dict().keys()
#     target_keys = [k for k in target_keys if k.startswith('stages.')]
#
#     for k, v in state_dict.items():
#         if 'attention_bias_idxs' in k:
#             continue
#         k = k.split('.')
#         if k[-2] == 'c':
#             k[-2] = 'conv'
#         if k[-2] == 'l':
#             k[-2] = 'linear'
#         k = '.'.join(k)
#         tmp_dict[k] = v
#
#     for k, v in tmp_dict.items():
#         if k.startswith('patch_embed'):
#             k = k.split('.')
#             k[1] = 'conv' + str(int(k[1]) // 2 + 1)
#             k = '.'.join(k)
#         elif k.startswith('blocks'):
#             kw = '.'.join(k.split('.')[2:])
#             find_kw = [a for a in list(sorted(tmp_dict.keys())) if kw in a]
#             idx = find_kw.index(k)
#             k = [a for a in target_keys if kw in a][idx]
#         out_dict[k] = v
#
#     return out_dict


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.conv1.conv',
        'classifier': 'head.linear',
        'fixed_input_size': True,
        'pool_size': (4, 4),
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'efficientvit_m0.r224_in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/xinyuliu-jeffrey/EfficientVit_Model_Zoo/releases/download/v1.0/efficientvit_m0.pth'
    ),
    'efficientvit_m1.r224_in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/xinyuliu-jeffrey/EfficientVit_Model_Zoo/releases/download/v1.0/efficientvit_m1.pth'
    ),
    'efficientvit_m2.r224_in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/xinyuliu-jeffrey/EfficientVit_Model_Zoo/releases/download/v1.0/efficientvit_m2.pth'
    ),
    'efficientvit_m3.r224_in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/xinyuliu-jeffrey/EfficientVit_Model_Zoo/releases/download/v1.0/efficientvit_m3.pth'
    ),
    'efficientvit_m4.r224_in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/xinyuliu-jeffrey/EfficientVit_Model_Zoo/releases/download/v1.0/efficientvit_m4.pth'
    ),
    'efficientvit_m5.r224_in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/xinyuliu-jeffrey/EfficientVit_Model_Zoo/releases/download/v1.0/efficientvit_m5.pth'
    ),
})


def _create_efficientvit_msra(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', (0, 1, 2))
    model = build_model_with_cfg(
        EfficientVitMsra,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs
    )
    return model


@register_model
def efficientvit_m0(pretrained=False, **kwargs):
    model_args = dict(
        img_size=224,
        embed_dim=[64, 128, 192],
        depth=[1, 2, 3],
        num_heads=[4, 4, 4],
        window_size=[7, 7, 7],
        kernels=[5, 5, 5, 5]
    )
    return _create_efficientvit_msra('efficientvit_m0', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def efficientvit_m1(pretrained=False, **kwargs):
    model_args = dict(
        img_size=224,
        embed_dim=[128, 144, 192],
        depth=[1, 2, 3],
        num_heads=[2, 3, 3],
        window_size=[7, 7, 7],
        kernels=[7, 5, 3, 3]
    )
    return _create_efficientvit_msra('efficientvit_m1', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def efficientvit_m2(pretrained=False, **kwargs):
    model_args = dict(
        img_size=224,
        embed_dim=[128, 192, 224],
        depth=[1, 2, 3],
        num_heads=[4, 3, 2],
        window_size=[7, 7, 7],
        kernels=[7, 5, 3, 3]
    )
    return _create_efficientvit_msra('efficientvit_m2', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def efficientvit_m3(pretrained=False, **kwargs):
    model_args = dict(
        img_size=224,
        embed_dim=[128, 240, 320],
        depth=[1, 2, 3],
        num_heads=[4, 3, 4],
        window_size=[7, 7, 7],
        kernels=[5, 5, 5, 5]
    )
    return _create_efficientvit_msra('efficientvit_m3', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def efficientvit_m4(pretrained=False, **kwargs):
    model_args = dict(
        img_size=224,
        embed_dim=[128, 256, 384],
        depth=[1, 2, 3],
        num_heads=[4, 4, 4],
        window_size=[7, 7, 7],
        kernels=[7, 5, 3, 3]
    )
    return _create_efficientvit_msra('efficientvit_m4', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def efficientvit_m5(pretrained=False, **kwargs):
    model_args = dict(
        img_size=224,
        embed_dim=[192, 288, 384],
        depth=[1, 3, 4],
        num_heads=[3, 3, 4],
        window_size=[7, 7, 7],
        kernels=[7, 5, 3, 3]
    )
    return _create_efficientvit_msra('efficientvit_m5', pretrained=pretrained, **dict(model_args, **kwargs))
