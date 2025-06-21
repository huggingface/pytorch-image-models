""" TinyViT

Paper: `TinyViT: Fast Pretraining Distillation for Small Vision Transformers`
    - https://arxiv.org/abs/2207.10666

Adapted from official impl at https://github.com/microsoft/Cream/tree/main/TinyViT
"""

__all__ = ['TinyVit']

import itertools
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import LayerNorm2d, NormMlpClassifierHead, DropPath,\
    trunc_normal_, resize_rel_pos_bias_table_levit, use_fused_attn
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._features_fx import register_notrace_module
from ._manipulate import checkpoint, checkpoint_seq
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
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.conv.groups, w.size(0), w.shape[2:],
            stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchEmbed(nn.Module):
    def __init__(self, in_chs, out_chs, act_layer):
        super().__init__()
        self.stride = 4
        self.conv1 = ConvNorm(in_chs, out_chs // 2, 3, 2, 1)
        self.act = act_layer()
        self.conv2 = ConvNorm(out_chs // 2, out_chs, 3, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class MBConv(nn.Module):
    def __init__(self, in_chs, out_chs, expand_ratio, act_layer, drop_path):
        super().__init__()
        mid_chs = int(in_chs * expand_ratio)
        self.conv1 = ConvNorm(in_chs, mid_chs, ks=1)
        self.act1 = act_layer()
        self.conv2 = ConvNorm(mid_chs, mid_chs, ks=3, stride=1, pad=1, groups=mid_chs)
        self.act2 = act_layer()
        self.conv3 = ConvNorm(mid_chs, out_chs, ks=1, bn_weight_init=0.0)
        self.act3 = act_layer()
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
    def __init__(self, dim, out_dim, act_layer):
        super().__init__()
        self.conv1 = ConvNorm(dim, out_dim, 1, 1, 0)
        self.act1 = act_layer()
        self.conv2 = ConvNorm(out_dim, out_dim, 3, 2, 1, groups=out_dim)
        self.act2 = act_layer()
        self.conv3 = ConvNorm(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x


class ConvLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            act_layer,
            drop_path=0.,
            conv_expand_ratio=4.,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.Sequential(*[
            MBConv(
                dim, dim, conv_expand_ratio, act_layer,
                drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])

    def forward(self, x):
        x = self.blocks(x)
        return x


class NormMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(torch.nn.Module):
    fused_attn: torch.jit.Final[bool]
    attention_bias_cache: Dict[str, torch.Tensor]

    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4,
            resolution=(14, 14),
    ):
        super().__init__()
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.val_dim = int(attn_ratio * key_dim)
        self.out_dim = self.val_dim * num_heads
        self.attn_ratio = attn_ratio
        self.resolution = resolution
        self.fused_attn = use_fused_attn()

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, num_heads * (self.val_dim + 2 * key_dim))
        self.proj = nn.Linear(self.out_dim, dim)

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
        q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.out_dim)
        x = self.proj(x)
        return x


class TinyVitBlock(nn.Module):
    """ TinyViT Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        act_layer: the activation function. Default: nn.GELU
    """

    def __init__(
            self,
            dim,
            num_heads,
            window_size=7,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            local_conv_size=3,
            act_layer=nn.GELU
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.mlp = NormMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        pad = local_conv_size // 2
        self.local_conv = ConvNorm(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        B, H, W, C = x.shape
        L = H * W

        shortcut = x
        if H == self.window_size and W == self.window_size:
            x = x.reshape(B, L, C)
            x = self.attn(x)
            x = x.view(B, H, W, C)
        else:
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            # window partition
            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C
            )

            x = self.attn(x)

            # window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size, C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()
        x = shortcut + self.drop_path1(x)

        x = x.permute(0, 3, 1, 2)
        x = self.local_conv(x)
        x = x.reshape(B, C, L).transpose(1, 2)

        x = x + self.drop_path2(self.mlp(x))
        return x.view(B, H, W, C)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"


register_notrace_module(TinyVitBlock)


class TinyVitStage(nn.Module):
    """ A basic TinyViT layer for one stage.

    Args:
        dim (int): Number of input channels.
        out_dim: the output dimension of the layer
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
        act_layer: the activation function. Default: nn.GELU
    """

    def __init__(
            self,
            dim,
            out_dim,
            depth,
            num_heads,
            window_size,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            downsample=None,
            local_conv_size=3,
            act_layer=nn.GELU,
    ):

        super().__init__()
        self.depth = depth
        self.out_dim =  out_dim

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                dim=dim,
                out_dim=out_dim,
                act_layer=act_layer,
            )
        else:
            self.downsample = nn.Identity()
            assert dim == out_dim

        # build blocks
        self.blocks = nn.Sequential(*[
            TinyVitBlock(
                dim=out_dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                local_conv_size=local_conv_size,
                act_layer=act_layer,
            )
            for i in range(depth)])

    def forward(self, x):
        x = self.downsample(x)
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.blocks(x)
        x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
        return x

    def extra_repr(self) -> str:
        return f"dim={self.out_dim}, depth={self.depth}"


class TinyVit(nn.Module):
    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            embed_dims=(96, 192, 384, 768),
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_sizes=(7, 7, 14, 7),
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.1,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            act_layer=nn.GELU,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(depths)
        self.mlp_ratio = mlp_ratio
        self.grad_checkpointing = use_checkpoint

        self.patch_embed = PatchEmbed(
            in_chs=in_chans,
            out_chs=embed_dims[0],
            act_layer=act_layer,
        )

        # stochastic depth rate rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build stages
        self.stages = nn.Sequential()
        stride = self.patch_embed.stride
        prev_dim = embed_dims[0]
        self.feature_info = []
        for stage_idx in range(self.num_stages):
            if stage_idx == 0:
                stage = ConvLayer(
                    dim=prev_dim,
                    depth=depths[stage_idx],
                    act_layer=act_layer,
                    drop_path=dpr[:depths[stage_idx]],
                    conv_expand_ratio=mbconv_expand_ratio,
                )
            else:
                out_dim = embed_dims[stage_idx]
                drop_path_rate = dpr[sum(depths[:stage_idx]):sum(depths[:stage_idx + 1])]
                stage = TinyVitStage(
                    dim=embed_dims[stage_idx - 1],
                    out_dim=out_dim,
                    depth=depths[stage_idx],
                    num_heads=num_heads[stage_idx],
                    window_size=window_sizes[stage_idx],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    drop_path=drop_path_rate,
                    downsample=PatchMerging,
                    act_layer=act_layer,
                )
                prev_dim = out_dim
                stride *= 2
            self.stages.append(stage)
            self.feature_info += [dict(num_chs=prev_dim, reduction=stride, module=f'stages.{stage_idx}')]

        # Classifier head
        self.num_features = self.head_hidden_size = embed_dims[-1]

        norm_layer_cf = partial(LayerNorm2d, eps=1e-5)
        self.head = NormMlpClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            norm_layer=norm_layer_cf,
        )

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

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
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages), indices)

        # forward pass
        x = self.patch_embed(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]

        for feat_idx, stage in enumerate(stages):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(stage, x)
            else:
                x = stage(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        self.stages = self.stages[:max_index + 1]  # truncate blocks w/ stem as idx 0
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
    target_sd = model.state_dict()
    out_dict = {}
    for k, v in state_dict.items():
        if k.endswith('attention_bias_idxs'):
            continue
        if 'attention_biases' in k:
            # TODO: whether move this func into model for dynamic input resolution? (high risk)
            v = resize_rel_pos_bias_table_levit(v.T, target_sd[k].shape[::-1]).T
        out_dict[k] = v
    return out_dict


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.conv1.conv',
        'classifier': 'head.fc',
        'pool_size': (7, 7),
        'input_size': (3, 224, 224),
        'crop_pct': 0.95,
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'tiny_vit_5m_224.dist_in22k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22k_distill.pth',
        num_classes=21841
    ),
    'tiny_vit_5m_224.dist_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22kto1k_distill.pth'
    ),
    'tiny_vit_5m_224.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_1k.pth'
    ),
    'tiny_vit_11m_224.dist_in22k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22k_distill.pth',
        num_classes=21841
    ),
    'tiny_vit_11m_224.dist_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22kto1k_distill.pth'
    ),
    'tiny_vit_11m_224.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_1k.pth'
    ),
    'tiny_vit_21m_224.dist_in22k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22k_distill.pth',
        num_classes=21841
    ),
    'tiny_vit_21m_224.dist_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_distill.pth'
    ),
    'tiny_vit_21m_224.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_1k.pth'
    ),
    'tiny_vit_21m_384.dist_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_384_distill.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),
    'tiny_vit_21m_512.dist_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_512_distill.pth',
        input_size=(3, 512, 512), pool_size=(16, 16), crop_pct=1.0, crop_mode='squash',
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
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[16, 16, 32, 16],
        drop_path_rate=0.1,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_21m_512', pretrained, **model_kwargs)
