""" TinyViT

Paper: `TinyViT: Fast Pretraining Distillation for Small Vision Transformers`
    - https://arxiv.org/abs/2207.10666

Adapted from official impl at https://github.com/microsoft/Cream/tree/main/TinyViT
"""

__all__ = ['TinyVit']
import math
import itertools
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, to_2tuple, trunc_normal_, resample_relative_position_bias_table, _assert
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


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        img_size = to_2tuple(resolution)
        self.patches_resolution = (math.ceil(img_size[0] / 4), math.ceil(img_size[1] / 4))
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.stride = 4
        n = embed_dim
        self.conv1 = ConvNorm(self.in_chans, n // 2, 3, 2, 1)
        self.act = activation()
        self.conv2 = ConvNorm(n // 2, n, 3, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio, activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans
        self.conv1 = ConvNorm(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()
        self.conv2 = ConvNorm(self.hidden_chans, self.hidden_chans, ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()
        self.conv3 = ConvNorm(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        x = self.act3(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = ConvNorm(dim, out_dim, 1, 1, 0)
        self.conv2 = ConvNorm(out_dim, out_dim, 3, 2, 1, groups=out_dim)
        self.conv3 = ConvNorm(out_dim, out_dim, 1, 1, 0)
        self.output_resolution = (math.ceil(input_resolution[0] / 2), math.ceil(input_resolution[1] / 2))

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, activation, drop_path=0.,
                 downsample=None, conv_expand_ratio=4.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        # build blocks
        self.blocks = nn.Sequential(*[
            MBConv(dim, dim, conv_expand_ratio, activation,
                   drop_path[i] if isinstance(drop_path, list) else drop_path,
                   )
            for i in range(depth)])

    def forward(self, x):
        x = self.blocks(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ClassifierHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes=1000
    ):
        super(ClassifierHead, self).__init__()
        self.norm_head = nn.LayerNorm(in_channels)
        self.fc = nn.Linear(in_channels, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = x.mean(1)
        x = self.norm_head(x)
        x = self.fc(x)
        return x


class Attention(torch.nn.Module):
    attention_bias_cache: Dict[str, torch.Tensor]

    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=(14, 14)):
        super().__init__()
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
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
        attn_bias = self.get_attention_biases(x.device)
        B, N, _ = x.shape
        # Normalization
        x = self.norm(x)
        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + attn_bias
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class TinyVitBlock(nn.Module):
    """ TinyViT Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.,
        drop=0.,
        drop_path=0.,
        local_conv_size=3,
        activation=nn.GELU
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads,
                              attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = ConvNorm(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        _assert(L == H * W, f"input feature has wrong size, expect {H * W}, got {L}")
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H %
                     self.window_size) % self.window_size
            pad_r = (self.window_size - W %
                     self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C
            )
            x = self.attn(x)
            # window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        x = x + self.drop_path(self.mlp(x))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"


class TinyVitStage(nn.Module):
    """ A basic TinyViT layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
        out_dim: the output dimension of the layer. Default: dim
    """

    def __init__(
        self,
        input_dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.,
        drop=0.,
        drop_path=0.,
        downsample=None,
        local_conv_size=3,
        activation=nn.GELU,
        out_dim=None,
    ):

        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.input_resolution = input_resolution
        self.depth = depth

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=input_dim, out_dim=self.out_dim, activation=activation)
            input_resolution = self.downsample.output_resolution
        else:
            self.downsample = nn.Identity()
            self.out_dim = self.input_dim

        # build blocks
        self.blocks = nn.Sequential(*[
            TinyVitBlock(dim=self.out_dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(
                             drop_path, list) else drop_path,
                         local_conv_size=local_conv_size,
                         activation=activation,
                         )
            for i in range(depth)])

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.out_dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TinyVit(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.1,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=1.0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(depths)
        self.mlp_ratio = mlp_ratio
        self.grad_checkpointing = use_checkpoint

        activation = nn.GELU

        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      resolution=img_size,
                                      activation=activation)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth rate rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build stages
        stages = nn.ModuleList()
        input_resolution = patches_resolution
        stride = self.patch_embed.stride
        self.feature_info = []
        for stage_idx in range(self.num_stages):
            if stage_idx == 0:
                out_dim = embed_dims[0]
                stage = ConvLayer(
                    dim=embed_dims[0],
                    input_resolution=input_resolution,
                    depth=depths[0],
                    activation=activation,
                    drop_path=dpr[:depths[0]],
                    downsample=None,
                    conv_expand_ratio=mbconv_expand_ratio,
                )
            else:
                out_dim = embed_dims[stage_idx]
                drop_path_rate = dpr[sum(depths[:stage_idx]):sum(depths[:stage_idx + 1])]
                stage = TinyVitStage(
                    num_heads=num_heads[stage_idx],
                    window_size=window_sizes[stage_idx],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    input_dim=embed_dims[stage_idx - 1],
                    input_resolution=input_resolution,
                    depth=depths[stage_idx],
                    drop_path=drop_path_rate,
                    downsample=PatchMerging,
                    out_dim=out_dim,
                    activation=activation,
                )
                input_resolution = (math.ceil(input_resolution[0] / 2), math.ceil(input_resolution[1] / 2))
                stride *= 2
            stages.append(stage)
            self.feature_info += [dict(num_chs=out_dim, reduction=stride, module=f'stages.{stage_idx}')]
        self.stages = nn.Sequential(*stages)

        # Classifier head
        self.num_features = embed_dims[-1]
        self.head = ClassifierHead(self.num_features, num_classes=num_classes)

        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)

    @torch.jit.ignore
    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay

        # stages -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for stage in self.stages:
            if hasattr(stage, 'downsample') and stage.downsample is not None:
                stage.downsample.apply(
                    lambda x: _set_lr_scale(x, lr_scales[i]))
            for block in stage.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
        assert i == depth
        self.head.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, 'lr_scale'), p.param_name

        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

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

    def reset_classifier(self, num_classes, **kwargs):
        self.num_classes = num_classes
        self.head = ClassifierHead(self.num_features, num_classes=num_classes)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        return x

    def forward_head(self, x):
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    # TODO: temporary use for testing, need change after weight convert
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
    targe_sd = model.state_dict()
    target_keys = list(targe_sd.keys())
    out_dict = {}
    i = 0
    for k, v in state_dict.items():
        if not k.endswith('attention_bias_idxs'):
            if 'attention_biases' in k:
                # dynamic window size by resampling relative_position_bias_table
                # TODO: whether move this func into model for dynamic input resolution? (high risk)
                v = resample_relative_position_bias_table(v, targe_sd[target_keys[i]].shape)
            out_dict[target_keys[i]] = v
        i += 1
    return out_dict


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.conv1.conv',
        'classifier': 'head.fc',
        'fixed_input_size': True,
        'pool_size': None,
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'tiny_vit_5m_224.dist_in22k': _cfg(
        url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22k_distill.pth',
        num_classes=21841
    ),
    'tiny_vit_5m_224.dist_in22k_ft_in1k': _cfg(
        url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22kto1k_distill.pth'
    ),
    'tiny_vit_5m_224.in1k': _cfg(
        url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_1k.pth'
    ),
    'tiny_vit_11m_224.dist_in22k': _cfg(
        url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22k_distill.pth',
        num_classes=21841
    ),
    'tiny_vit_11m_224.dist_in22k_ft_in1k': _cfg(
        url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22kto1k_distill.pth'
    ),
    'tiny_vit_11m_224.in1k': _cfg(
        url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_1k.pth'
    ),
    'tiny_vit_21m_224.dist_in22k': _cfg(
        url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22k_distill.pth',
        num_classes=21841
    ),
    'tiny_vit_21m_224.dist_in22k_ft_in1k': _cfg(
        url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_distill.pth'
    ),
    'tiny_vit_21m_224.in1k': _cfg(
        url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_1k.pth'
    ),
    'tiny_vit_21m_384.dist_in22k_ft_in1k': _cfg(
        url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_384_distill.pth'
    ),
    'tiny_vit_21m_512.dist_in22k_ft_in1k': _cfg(
        url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_512_distill.pth'
    ),
})


def _create_tiny_vit(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', (0, 1, 2, 3))
    model = build_model_with_cfg(
        TinyVit,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs
    )
    return model


@register_model
def tiny_vit_5m_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.0,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_5m_224', pretrained, **model_kwargs)


@register_model
def tiny_vit_11m_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dims=[64, 128, 256, 448],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 14],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.1,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_11m_224', pretrained, **model_kwargs)


@register_model
def tiny_vit_21m_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.2,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_21m_224', pretrained, **model_kwargs)


@register_model
def tiny_vit_21m_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=384,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[12, 12, 24, 12],
        drop_path_rate=0.1,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_21m_384', pretrained, **model_kwargs)


@register_model
def tiny_vit_21m_512(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=512,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[16, 16, 32, 16],
        drop_path_rate=0.1,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_21m_512', pretrained, **model_kwargs)
