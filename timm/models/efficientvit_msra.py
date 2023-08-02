""" EfficientViT (by MSRA)

Paper: `EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention`
    - https://arxiv.org/abs/2305.07027

Adapted from official impl at https://github.com/microsoft/Cream/tree/main/EfficientViT
"""

__all__ = ['EfficientViTMSRA']

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite, SelectAdaptivePool2d
from ._registry import register_model, generate_default_cfgs
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
import itertools


class ConvBN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BNLinear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = ConvBN(dim, hid_dim, 1, 1, 0)
        self.act = torch.nn.ReLU()
        self.conv2 = ConvBN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = ConvBN(hid_dim, out_dim, 1, 1, 0)

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
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = ConvBN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = ConvBN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class CascadedGroupAttention(torch.nn.Module):
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(ConvBN(dim // (num_heads), self.key_dim * 2 + self.d))
            dws.append(ConvBN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(
            torch.nn.ReLU(),
            ConvBN(self.d * num_heads, dim, bn_weight_init=0)
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
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale
                +
                (trainingab[i] if self.training else self.ab[i])
            )
            attn = attn.softmax(dim=-1)
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)
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
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution
        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                           attn_ratio=attn_ratio,
                                           resolution=window_resolution,
                                           kernels=kernels,)

    def forward(self, x):
        H = W = self.resolution
        B, C, H_, W_ = x.shape
        # Only check this for classifcation models
        assert H == H_ and W == W_, 'input feature has wrong size, expect {}, got {}'.format((H, W), (H_, W_))

        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution, C).transpose(2, 3).reshape(B, pH, pW, C)
            if padding:
                x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)
        return x


class EfficientViTBlock(torch.nn.Module):
    """ A basic EfficientViT building block.

    Args:
        ed (int): Number of input channels.
        kd (int): Dimension for query and key in the token mixer.
        nh (int): Number of attention heads.
        ar (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        super().__init__()

        self.dw0 = ResidualDrop(ConvBN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn0 = ResidualDrop(FFN(ed, int(ed * 2)))

        self.mixer = ResidualDrop(
            LocalWindowAttention(ed, kd, nh, attn_ratio=ar, resolution=resolution,
                                 window_resolution=window_resolution, kernels=kernels))

        self.dw1 = ResidualDrop(ConvBN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn1 = ResidualDrop(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


class PatchEmbedding(torch.nn.Sequential):
    def __init__(self, in_chans, dim):
        super().__init__()
        self.add_module('conv1', ConvBN(in_chans, dim // 8, 3, 2, 1))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('conv2', ConvBN(dim // 8, dim // 4, 3, 2, 1))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('conv3', ConvBN(dim // 4, dim // 2, 3, 2, 1))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('conv4', ConvBN(dim // 2, dim, 3, 2, 1))
        self.patch_size = 16


class EfficientViTMSRA(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dim=[64, 128, 192],
        key_dim=[16, 16, 16],
        depth=[1, 2, 3],
        num_heads=[4, 4, 4],
        window_size=[7, 7, 7],
        kernels=[5, 5, 5, 5],
        down_ops=[[''], ['subsample', 2], ['subsample', 2]],
        global_pool='avg',
    ):
        super(EfficientViTMSRA, self).__init__()
        self.grad_checkpointing = False
        resolution = img_size
        # Patch embedding
        self.patch_embed = PatchEmbedding(in_chans, embed_dim[0])
        stride = self.patch_embed.patch_size
        resolution = img_size // self.patch_embed.patch_size

        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.feature_info = []
        stages = []

        # Build EfficientViT blocks
        for i, (ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            blocks = []
            if do[0] == 'subsample' and i != 0:
                # Build EfficientViT downsample block
                resolution_ = (resolution - 1) // do[1] + 1
                blocks.append(torch.nn.Sequential(ResidualDrop(ConvBN(embed_dim[i - 1], embed_dim[i - 1], 3, 1, 1, groups=embed_dim[i - 1])),
                              ResidualDrop(FFN(embed_dim[i - 1], int(embed_dim[i - 1] * 2))),))
                blocks.append(PatchMerging(*embed_dim[i - 1:i + 1]))
                resolution = resolution_
                blocks.append(torch.nn.Sequential(ResidualDrop(ConvBN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i])),
                              ResidualDrop(FFN(embed_dim[i], int(embed_dim[i] * 2))),))
                stride *= 2
            for d in range(dpth):
                blocks.append(EfficientViTBlock(ed, kd, nh, ar, resolution, wd, kernels))
            stages.append(nn.Sequential(*blocks))
            self.feature_info += [dict(num_chs=embed_dim[i], reduction=stride, module=f'stages.{i}')]

        self.stages = nn.Sequential(*stages)

        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool, flatten=True, input_fmt='NCHW')
        self.out_dims = embed_dim[-1]
        self.head = BNLinear(self.out_dims, num_classes) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^patch_embed',
            blocks=[(r'^stages\.(\d+)', None)]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool, flatten=True, input_fmt='NCHW')
        self.head = BNLinear(self.out_dims, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
    out_dict = {}
    for k, v in state_dict.items():
        if k.startswith('patch_embed'):
            k = k.split('.')
            k[1] = 'conv' + str(int(k[1]) // 2 + 1)
            k = '.'.join(k)
        elif k.startswith('blocks'):
            k = k.split('.')
            k[0] = 'stages.' + str(int(k[0][6:]) - 1)
            k = '.'.join(k)
        out_dict[k] = v
    return out_dict


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.in_conv.conv',
        'classifier': 'head',
        **kwargs,
    }


default_cfgs = generate_default_cfgs(
    {
        'efficientvit_m0.r224_in1k': _cfg(
            url='https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m0.pth'
        ),
        'efficientvit_m1.r224_in1k': _cfg(
            url='https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m1.pth'
        ),
        'efficientvit_m2.r224_in1k': _cfg(
            url='https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m2.pth'
        ),
        'efficientvit_m3.r224_in1k': _cfg(
            url='https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m3.pth'
        ),
        'efficientvit_m4.r224_in1k': _cfg(
            url='https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m4.pth'
        ),
        'efficientvit_m5.r224_in1k': _cfg(
            url='https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m5.pth'
        ),
    }
)


def _create_efficientvit_msra(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', (0, 1, 2))
    model = build_model_with_cfg(
        EfficientViTMSRA,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs
    )
    return model


@register_model
def efficientvit_m0(pretrained=False, **kwargs):
    model_args = dict(img_size=224,
                      embed_dim=[64, 128, 192],
                      depth=[1, 2, 3],
                      num_heads=[4, 4, 4],
                      window_size=[7, 7, 7],
                      kernels=[5, 5, 5, 5])
    return _create_efficientvit_msra('efficientvit_m0', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def efficientvit_m1(pretrained=False, **kwargs):
    model_args = dict(img_size=224,
                      embed_dim=[128, 144, 192],
                      depth=[1, 2, 3],
                      num_heads=[2, 3, 3],
                      window_size=[7, 7, 7],
                      kernels=[7, 5, 3, 3])
    return _create_efficientvit_msra('efficientvit_m1', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def efficientvit_m2(pretrained=False, **kwargs):
    model_args = dict(img_size=224,
                      embed_dim=[128, 192, 224],
                      depth=[1, 2, 3],
                      num_heads=[4, 3, 2],
                      window_size=[7, 7, 7],
                      kernels=[7, 5, 3, 3])
    return _create_efficientvit_msra('efficientvit_m2', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def efficientvit_m3(pretrained=False, **kwargs):
    model_args = dict(img_size=224,
                      embed_dim=[128, 240, 320],
                      depth=[1, 2, 3],
                      num_heads=[4, 3, 4],
                      window_size=[7, 7, 7],
                      kernels=[5, 5, 5, 5])
    return _create_efficientvit_msra('efficientvit_m3', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def efficientvit_m4(pretrained=False, **kwargs):
    model_args = dict(img_size=224,
                      embed_dim=[128, 256, 384],
                      depth=[1, 2, 3],
                      num_heads=[4, 4, 4],
                      window_size=[7, 7, 7],
                      kernels=[7, 5, 3, 3])
    return _create_efficientvit_msra('efficientvit_m4', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def efficientvit_m5(pretrained=False, **kwargs):
    model_args = dict(img_size=224,
                      embed_dim=[192, 288, 384],
                      depth=[1, 3, 4],
                      num_heads=[3, 3, 4],
                      window_size=[7, 7, 7],
                      kernels=[7, 5, 3, 3])
    return _create_efficientvit_msra('efficientvit_m5', pretrained=pretrained, **dict(model_args, **kwargs))
