""" Global Context ViT

From scratch implementation of GCViT in the style of timm swin_transformer_v2_cr.py

Global Context Vision Transformers -https://arxiv.org/abs/2206.09959

@article{hatamizadeh2022global,
  title={Global Context Vision Transformers},
  author={Hatamizadeh, Ali and Yin, Hongxu and Kautz, Jan and Molchanov, Pavlo},
  journal={arXiv preprint arXiv:2206.09959},
  year={2022}
}

Free of any code related to NVIDIA GCVit impl at https://github.com/NVlabs/GCVit.
The license for this code release is Apache 2.0 with no commercial restrictions.

However, weight files adapted from NVIDIA GCVit impl ARE under a non-commercial share-alike license
(https://creativecommons.org/licenses/by-nc-sa/4.0/) until I have a chance to train new ones...

Hacked together by / Copyright 2022, Ross Wightman
"""
import math
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, to_2tuple, to_ntuple, Mlp, ClassifierHead, LayerNorm2d, \
    get_attn, get_act_layer, get_norm_layer, RelPosBias, _assert
from ._builder import build_model_with_cfg
from ._features_fx import register_notrace_function
from ._manipulate import named_apply
from ._registry import register_model, generate_default_cfgs

__all__ = ['GlobalContextVit']


class MbConvBlock(nn.Module):
    """ A depthwise separable / fused mbconv style residual block with SE, `no norm.
    """
    def __init__(
            self,
            in_chs,
            out_chs=None,
            expand_ratio=1.0,
            attn_layer='se',
            bias=False,
            act_layer=nn.GELU,
    ):
        super().__init__()
        attn_kwargs = dict(act_layer=act_layer)
        if isinstance(attn_layer, str) and attn_layer == 'se' or attn_layer == 'eca':
            attn_kwargs['rd_ratio'] = 0.25
            attn_kwargs['bias'] = False
        attn_layer = get_attn(attn_layer)
        out_chs = out_chs or in_chs
        mid_chs = int(expand_ratio * in_chs)

        self.conv_dw = nn.Conv2d(in_chs, mid_chs, 3, 1, 1, groups=in_chs, bias=bias)
        self.act = act_layer()
        self.se = attn_layer(mid_chs, **attn_kwargs)
        self.conv_pw = nn.Conv2d(mid_chs, out_chs, 1, 1, 0, bias=bias)

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.act(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = x + shortcut
        return x


class Downsample2d(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            reduction='conv',
            act_layer=nn.GELU,
            norm_layer=LayerNorm2d,  # NOTE in NCHW
    ):
        super().__init__()
        dim_out = dim_out or dim

        self.norm1 = norm_layer(dim) if norm_layer is not None else nn.Identity()
        self.conv_block = MbConvBlock(dim, act_layer=act_layer)
        assert reduction in ('conv', 'max', 'avg')
        if reduction == 'conv':
            self.reduction = nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False)
        elif reduction == 'max':
            assert dim == dim_out
            self.reduction = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            assert dim == dim_out
            self.reduction = nn.AvgPool2d(kernel_size=2)
        self.norm2 = norm_layer(dim_out) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.norm1(x)
        x = self.conv_block(x)
        x = self.reduction(x)
        x = self.norm2(x)
        return x


class FeatureBlock(nn.Module):
    def __init__(
            self,
            dim,
            levels=0,
            reduction='max',
            act_layer=nn.GELU,
    ):
        super().__init__()
        reductions = levels
        levels = max(1, levels)
        if reduction == 'avg':
            pool_fn = partial(nn.AvgPool2d, kernel_size=2)
        else:
            pool_fn = partial(nn.MaxPool2d, kernel_size=3, stride=2, padding=1)
        self.blocks = nn.Sequential()
        for i in range(levels):
            self.blocks.add_module(f'conv{i+1}', MbConvBlock(dim, act_layer=act_layer))
            if reductions:
                self.blocks.add_module(f'pool{i+1}', pool_fn())
                reductions -= 1

    def forward(self, x):
        return self.blocks(x)


class Stem(nn.Module):
    def __init__(
            self,
            in_chs: int = 3,
            out_chs: int = 96,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm2d,  # NOTE stem in NCHW
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=2, padding=1)
        self.down = Downsample2d(out_chs, act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.down(x)
        return x


class WindowAttentionGlobal(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: Tuple[int, int],
            use_global: bool = True,
            qkv_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ):
        super().__init__()
        window_size = to_2tuple(window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_global = use_global

        self.rel_pos = RelPosBias(window_size=window_size, num_heads=num_heads)
        if self.use_global:
            self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, q_global: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        if self.use_global and q_global is not None:
            _assert(x.shape[-1] == q_global.shape[-1], 'x and q_global seq lengths should be equal')

            kv = self.qkv(x)
            kv = kv.reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)

            q = q_global.repeat(B // q_global.shape[0], 1, 1, 1)
            q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        q = q * self.scale

        attn = q @ k.transpose(-2, -1).contiguous()  # NOTE contiguous() fixes an odd jit bug in PyTorch 2.0
        attn = self.rel_pos(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size: Tuple[int, int]):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class GlobalContextVitBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            feat_size: Tuple[int, int],
            num_heads: int,
            window_size: int = 7,
            mlp_ratio: float = 4.,
            use_global: bool = True,
            qkv_bias: bool = True,
            layer_scale: Optional[float] = None,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            attn_layer: Callable = WindowAttentionGlobal,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()
        feat_size = to_2tuple(feat_size)
        window_size = to_2tuple(window_size)
        self.window_size = window_size
        self.num_windows = int((feat_size[0] // window_size[0]) * (feat_size[1] // window_size[1]))

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            num_heads=num_heads,
            window_size=window_size,
            use_global=use_global,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ls1 = LayerScale(dim, layer_scale) if layer_scale is not None else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, layer_scale) if layer_scale is not None else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _window_attn(self, x, q_global: Optional[torch.Tensor] = None):
        B, H, W, C = x.shape
        x_win = window_partition(x, self.window_size)
        x_win = x_win.view(-1, self.window_size[0] * self.window_size[1], C)
        attn_win = self.attn(x_win, q_global)
        x = window_reverse(attn_win, self.window_size, (H, W))
        return x

    def forward(self, x, q_global: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(self.ls1(self._window_attn(self.norm1(x), q_global)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class GlobalContextVitStage(nn.Module):
    def __init__(
            self,
            dim,
            depth: int,
            num_heads: int,
            feat_size: Tuple[int, int],
            window_size: Tuple[int, int],
            downsample: bool = True,
            global_norm: bool = False,
            stage_norm: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            layer_scale: Optional[float] = None,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[List[float], float] = 0.0,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            norm_layer_cl: Callable = LayerNorm2d,
    ):
        super().__init__()
        if downsample:
            self.downsample = Downsample2d(
                dim=dim,
                dim_out=dim * 2,
                norm_layer=norm_layer,
            )
            dim = dim * 2
            feat_size = (feat_size[0] // 2, feat_size[1] // 2)
        else:
            self.downsample = nn.Identity()
        self.feat_size = feat_size
        window_size = to_2tuple(window_size)

        feat_levels = int(math.log2(min(feat_size) / min(window_size)))
        self.global_block = FeatureBlock(dim, feat_levels)
        self.global_norm = norm_layer_cl(dim) if global_norm else nn.Identity()

        self.blocks = nn.ModuleList([
            GlobalContextVitBlock(
                dim=dim,
                num_heads=num_heads,
                feat_size=feat_size,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                use_global=(i % 2 != 0),
                layer_scale=layer_scale,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer_cl,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer_cl(dim) if stage_norm else nn.Identity()
        self.dim = dim
        self.feat_size = feat_size
        self.grad_checkpointing = False

    def forward(self, x):
        # input NCHW, downsample & global block are 2d conv + pooling
        x = self.downsample(x)
        global_query = self.global_block(x)

        # reshape NCHW --> NHWC for transformer blocks
        x = x.permute(0, 2, 3, 1)
        global_query = self.global_norm(global_query.permute(0, 2, 3, 1))
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, global_query)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # back to NCHW
        return x


class GlobalContextVit(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            img_size: Tuple[int, int] = 224,
            window_ratio: Tuple[int, ...] = (32, 32, 16, 32),
            window_size: Tuple[int, ...] = None,
            embed_dim: int = 64,
            depths: Tuple[int, ...] = (3, 4, 19, 5),
            num_heads: Tuple[int, ...] = (2, 4, 8, 16),
            mlp_ratio: float = 3.0,
            qkv_bias: bool = True,
            layer_scale: Optional[float] = None,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init='',
            act_layer: str = 'gelu',
            norm_layer: str = 'layernorm2d',
            norm_layer_cl: str = 'layernorm',
            norm_eps: float = 1e-5,
    ):
        super().__init__()
        act_layer = get_act_layer(act_layer)
        norm_layer = partial(get_norm_layer(norm_layer), eps=norm_eps)
        norm_layer_cl = partial(get_norm_layer(norm_layer_cl), eps=norm_eps)

        img_size = to_2tuple(img_size)
        feat_size = tuple(d // 4 for d in img_size)  # stem reduction by 4
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        num_stages = len(depths)
        self.num_features = int(embed_dim * 2 ** (num_stages - 1))
        if window_size is not None:
            window_size = to_ntuple(num_stages)(window_size)
        else:
            assert window_ratio is not None
            window_size = tuple([(img_size[0] // r, img_size[1] // r) for r in to_ntuple(num_stages)(window_ratio)])

        self.stem = Stem(
            in_chs=in_chans,
            out_chs=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        for i in range(num_stages):
            last_stage = i == num_stages - 1
            stage_scale = 2 ** max(i - 1, 0)
            stages.append(GlobalContextVitStage(
                dim=embed_dim * stage_scale,
                depth=depths[i],
                num_heads=num_heads[i],
                feat_size=(feat_size[0] // stage_scale, feat_size[1] // stage_scale),
                window_size=window_size[i],
                downsample=i != 0,
                stage_norm=last_stage,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                layer_scale=layer_scale,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_layer_cl=norm_layer_cl,
            ))
        self.stages = nn.Sequential(*stages)

        # Classifier head
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

        if weight_init:
            named_apply(partial(self._init_weights, scheme=weight_init), self)

    def _init_weights(self, module, name, scheme='vit'):
        # note Conv2d left as default init
        if scheme == 'vit':
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
        else:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            k for k, _ in self.named_parameters()
            if any(n in k for n in ["relative_position_bias_table", "rel_pos.mlp"])}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',  # stem and embed
            blocks=r'^stages\.(\d+)'
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is None:
            global_pool = self.head.global_pool.pool_type
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_gcvit(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    model = build_model_with_cfg(GlobalContextVit, variant, pretrained, **kwargs)
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1', 'classifier': 'head.fc',
        'fixed_input_size': True,
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'gcvit_xxtiny.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-morevit/gcvit_xxtiny_224_nvidia-d1d86009.pth'),
    'gcvit_xtiny.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-morevit/gcvit_xtiny_224_nvidia-274b92b7.pth'),
    'gcvit_tiny.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-morevit/gcvit_tiny_224_nvidia-ac783954.pth'),
    'gcvit_small.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-morevit/gcvit_small_224_nvidia-4e98afa2.pth'),
    'gcvit_base.in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-morevit/gcvit_base_224_nvidia-f009139b.pth'),
})


@register_model
def gcvit_xxtiny(pretrained=False, **kwargs) -> GlobalContextVit:
    model_kwargs = dict(
        depths=(2, 2, 6, 2),
        num_heads=(2, 4, 8, 16),
        **kwargs)
    return _create_gcvit('gcvit_xxtiny', pretrained=pretrained, **model_kwargs)


@register_model
def gcvit_xtiny(pretrained=False, **kwargs) -> GlobalContextVit:
    model_kwargs = dict(
        depths=(3, 4, 6, 5),
        num_heads=(2, 4, 8, 16),
        **kwargs)
    return _create_gcvit('gcvit_xtiny', pretrained=pretrained, **model_kwargs)


@register_model
def gcvit_tiny(pretrained=False, **kwargs) -> GlobalContextVit:
    model_kwargs = dict(
        depths=(3, 4, 19, 5),
        num_heads=(2, 4, 8, 16),
        **kwargs)
    return _create_gcvit('gcvit_tiny', pretrained=pretrained, **model_kwargs)


@register_model
def gcvit_small(pretrained=False, **kwargs) -> GlobalContextVit:
    model_kwargs = dict(
        depths=(3, 4, 19, 5),
        num_heads=(3, 6, 12, 24),
        embed_dim=96,
        mlp_ratio=2,
        layer_scale=1e-5,
        **kwargs)
    return _create_gcvit('gcvit_small', pretrained=pretrained, **model_kwargs)


@register_model
def gcvit_base(pretrained=False, **kwargs) -> GlobalContextVit:
    model_kwargs = dict(
        depths=(3, 4, 19, 5),
        num_heads=(4, 8, 16, 32),
        embed_dim=128,
        mlp_ratio=2,
        layer_scale=1e-5,
        **kwargs)
    return _create_gcvit('gcvit_base', pretrained=pretrained, **model_kwargs)
