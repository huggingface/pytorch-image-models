""" CvT: Convolutional Vision Transformer

From-scratch implementation of CvT in PyTorch

'CvT: Introducing Convolutions to Vision Transformers'
    - https://arxiv.org/abs/2103.15808

Implementation for timm by / Copyright 2024, Fredo Guan
"""

from collections import OrderedDict
from functools import partial
from typing import List, Final, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import ConvNormAct, LayerNorm, LayerNorm2d, Mlp, QuickGELU, trunc_normal_, use_fused_attn
from ._builder import build_model_with_cfg
from ._registry import generate_default_cfgs, register_model



__all__ = ['CvT']

class ConvEmbed(nn.Module):
    def __init__(
        self,
        in_chs: int = 3,
        out_chs: int = 64,
        kernel_size: int = 7,
        stride: int = 4,
        padding: int = 2,
        norm_layer: nn.Module = LayerNorm2d,
    ) -> None:
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.norm = norm_layer(out_chs) if norm_layer else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: # [B, C, H, W] -> [B, C, H, W]
        x = self.conv(x)
        x = self.norm(x)
        return x



class ConvProj(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        stride_q: int = 1,
        stride_kv: int = 2,
        padding: int = 1,
        bias: bool = False,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.Identity,
    ) -> None:
        super().__init__()
        self.dim = dim
        
        # FIXME not working, bn layer outputs are incorrect
        '''
        self.conv_q = ConvNormAct(
            dim,
            dim,
            kernel_size,
            stride=stride_q,
            padding=padding,
            groups=dim,
            bias=bias,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        
        # TODO fuse kv conv? don't wanna do weight remap
        # TODO if act_layer is id and not cls_token (gap model?), is later projection in attn necessary?
        
        self.conv_k = ConvNormAct(
            dim,
            dim,
            kernel_size,
            stride=stride_kv,
            padding=padding,
            groups=dim,
            bias=bias,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        
        self.conv_v = ConvNormAct(
            dim,
            dim,
            kernel_size,
            stride=stride_kv,
            padding=padding,
            groups=dim,
            bias=bias,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        '''
        self.conv_q = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride_q,
                    bias=bias,
                    groups=dim
                )),
                ('bn', nn.BatchNorm2d(dim)),]))
        self.conv_k = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride_kv,
                    bias=bias,
                    groups=dim
                )),
                ('bn', nn.BatchNorm2d(dim)),]))
        self.conv_v = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride_kv,
                    bias=bias,
                    groups=dim
                )),
                ('bn', nn.BatchNorm2d(dim)),]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        # [B, C, H, W] -> [B, H*W, C]
        q = self.conv_q(x).flatten(2).transpose(1, 2)
        k = self.conv_k(x).flatten(2).transpose(1, 2)
        v = self.conv_v(x).flatten(2).transpose(1, 2)
        return q, k, v

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 1,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = dim ** -0.5
        self.fused_attn = False #use_fused_attn()
        
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, N, C = q.shape
        
        # [B, H*W, C] -> [B, H*W, n_h, d_h] -> [B, n_h, H*W, d_h]
        q = self.proj_q(q).reshape(B, q.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.proj_k(k).reshape(B, k.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.proj_v(v).reshape(B, v.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        # [B, n_h, H*W, d_h], [B, n_h, H*W/4, d_h], [B, n_h, H*W/4, d_h]

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
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

class CvTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        stride_q: int = 1,
        stride_kv: int = 2,
        padding: int = 1,
        conv_bias: bool = False,
        conv_norm_layer: nn.Module = nn.BatchNorm2d,
        conv_act_layer: nn.Module = nn.Identity,
        num_heads: int = 1,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        input_norm_layer: nn.Module = partial(LayerNorm2d, eps=1e-5),
        norm_layer: nn.Module = partial(LayerNorm, eps=1e-5),
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        mlp_layer: nn.Module = Mlp,
        mlp_ratio: float = 4.,
        mlp_act_layer: nn.Module = QuickGELU,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        self.use_cls_token = use_cls_token

        self.norm1 = input_norm_layer(dim)
        self.conv_proj = ConvProj(
            dim = dim,
            kernel_size = kernel_size,
            stride_q = stride_q,
            stride_kv = stride_kv,
            padding = padding,
            bias = conv_bias,
            norm_layer = conv_norm_layer,
            act_layer = conv_act_layer,
        )
        self.attn = Attention(
            dim = dim,
            num_heads = num_heads,
            qkv_bias = qkv_bias,
            qk_norm = qk_norm,
            attn_drop = attn_drop,
            proj_drop = proj_drop,
            norm_layer = norm_layer
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=mlp_act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.probe = nn.Identity()

    def add_cls_token(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        cls_token: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)
        return q, k, v

    def fw_attn(self, x: torch.Tensor, cls_token: Optional[torch.Tensor]) -> torch.Tensor:
        return self.attn(*self.add_cls_token(*self.conv_proj(x), cls_token))

    def forward(self, x: torch.Tensor, cls_token: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, C, H, W = x.shape

        x = torch.cat((cls_token, x.flatten(2).transpose(1, 2)), dim=1) if cls_token is not None else x.flatten(2).transpose(1, 2) \
            + self.drop_path1(self.ls1(self.fw_attn(self.norm1(x), cls_token)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        if self.use_cls_token:
            cls_token, x = torch.split(x, [1, H*W], 1)
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.probe(x)
        
        return x, cls_token

class CvTStage(nn.Module):
    def __init__(
        self,
        in_chs: int,
        dim: int,
        depth: int,
        embed_kernel_size: int = 7,
        embed_stride: int = 4,
        embed_padding: int = 2,
        kernel_size: int = 3,
        stride_q: int = 1,
        stride_kv: int = 2,
        padding: int = 1,
        conv_bias: bool = False,
        conv_norm_layer: nn.Module = nn.BatchNorm2d,
        conv_act_layer: nn.Module = nn.Identity,
        num_heads: int = 1,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        input_norm_layer = LayerNorm2d,
        norm_layer: nn.Module = LayerNorm,
        init_values: Optional[float] = None,
        drop_path_rates: List[float] = [0.],
        mlp_layer: nn.Module = Mlp,
        mlp_ratio: float = 4.,
        mlp_act_layer: nn.Module = QuickGELU,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        
        self.conv_embed = ConvEmbed(
            in_chs = in_chs,
            out_chs = dim,
            kernel_size = embed_kernel_size,
            stride = embed_stride,
            padding = embed_padding,
            norm_layer = input_norm_layer,
        )
        self.embed_drop = nn.Dropout(proj_drop)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if use_cls_token else None

        blocks = []
        for i in range(depth):
            block = CvTBlock(
                dim = dim,
                kernel_size = kernel_size,
                stride_q = stride_q,
                stride_kv = stride_kv,
                padding = padding,
                conv_bias = conv_bias,
                conv_norm_layer = conv_norm_layer,
                conv_act_layer = conv_act_layer,
                num_heads = num_heads,
                qkv_bias = qkv_bias,
                qk_norm = qk_norm,
                attn_drop = attn_drop,
                proj_drop = proj_drop,
                input_norm_layer = input_norm_layer,
                norm_layer = norm_layer,
                init_values = init_values,
                drop_path = drop_path_rates[i],
                mlp_layer = mlp_layer,
                mlp_ratio = mlp_ratio,
                mlp_act_layer = mlp_act_layer,
                use_cls_token = use_cls_token,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.probe = nn.Identity()

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.conv_embed(x)
        x = self.probe(x)
        x = self.embed_drop(x)

        cls_token = self.embed_drop(
            self.cls_token.expand(x.shape[0], -1, -1)
        ) if self.cls_token is not None else None
        for block in self.blocks: # technically possible to exploit nn.Sequential's untyped intermediate results if each block takes in a tensor
            x, cls_token = block(x, cls_token)
        
        return x, cls_token

class CvT(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        dims: Tuple[int, ...] = (64, 192, 384),
        depths: Tuple[int, ...] = (1, 2, 10),
        embed_kernel_size: Tuple[int, ...] = (7, 3, 3),
        embed_stride: Tuple[int, ...] = (4, 2, 2),
        embed_padding: Tuple[int, ...] = (2, 1, 1),
        kernel_size: int = 3,
        stride_q: int = 1,
        stride_kv: int = 2,
        padding: int = 1,
        conv_bias: bool = False,
        conv_norm_layer: nn.Module = nn.BatchNorm2d,
        conv_act_layer: nn.Module = nn.Identity,
        num_heads: Tuple[int, ...] = (1, 3, 6),
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        input_norm_layer = partial(LayerNorm2d, eps=1e-5),
        norm_layer: nn.Module = partial(LayerNorm, eps=1e-5),
        init_values: Optional[float] = None,
        drop_path_rate: float = 0.,
        mlp_layer: nn.Module = Mlp,
        mlp_ratio: float = 4.,
        mlp_act_layer: nn.Module = QuickGELU,
        use_cls_token: Tuple[bool, ...] = (False, False, True),
    ) -> None:
        super().__init__()
        num_stages = len(dims)
        assert num_stages == len(depths) == len(embed_kernel_size) == len(embed_stride)
        assert num_stages == len(embed_padding) == len(num_heads) == len(use_cls_token)
        self.num_classes = num_classes
        self.num_features = dims[-1]
        
        # FIXME only on last stage, no need for tuple
        self.use_cls_token = use_cls_token[-1]
        
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        
        in_chs = in_chans
        
        # TODO move stem
        
        stages = []
        for stage_idx in range(num_stages):
            dim = dims[stage_idx]
            stage = CvTStage(
                in_chs = in_chs,
                dim = dim,
                depth = depths[stage_idx],
                embed_kernel_size = embed_kernel_size[stage_idx],
                embed_stride = embed_stride[stage_idx],
                embed_padding = embed_padding[stage_idx],
                kernel_size = kernel_size,
                stride_q = stride_q,
                stride_kv = stride_kv,
                padding = padding,
                conv_bias = conv_bias,
                conv_norm_layer = conv_norm_layer,
                conv_act_layer = conv_act_layer,
                num_heads = num_heads[stage_idx],
                qkv_bias = qkv_bias,
                qk_norm = qk_norm,
                attn_drop = attn_drop,
                proj_drop = proj_drop,
                input_norm_layer = input_norm_layer,
                norm_layer = norm_layer,
                init_values = init_values,
                drop_path_rates = dpr[stage_idx],
                mlp_layer = mlp_layer,
                mlp_ratio = mlp_ratio,
                mlp_act_layer = mlp_act_layer,
                use_cls_token = use_cls_token[stage_idx],
            )
            in_chs = dim
            stages.append(stage)
        self.stages = nn.ModuleList(stages)
        
        self.norm = norm_layer(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for stage in self.stages:
            x, cls_token = stage(x)
            
        
        if self.use_cls_token:
            return self.head(self.norm(cls_token.flatten(1)))
        else:
            return self.head(self.norm(x.mean(dim=(2,3))))
            
        
        
def checkpoint_filter_fn(state_dict, model):
    """ Remap MSFT checkpoints -> timm """
    if 'head.fc.weight' in state_dict:
        return state_dict  # non-MSFT checkpoint

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    import re
    out_dict = {}
    for k, v in state_dict.items():
        k = re.sub(r'stage([0-9]+)', r'stages.\1', k)
        k = k.replace('patch_embed', 'conv_embed')
        k = k.replace('conv_embed.proj', 'conv_embed.conv')
        k = k.replace('attn.conv_proj', 'conv_proj.conv')
        out_dict[k] = v
    return out_dict
    
    
def _create_cvt(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 2, 10))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        CvT,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)

    return model

# TODO update first_conv
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (14, 14),
        'crop_pct': 0.95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv', 'classifier': 'head',
        **kwargs
    }
    
default_cfgs = generate_default_cfgs({
    'cvt_13.msft_in1k': _cfg(url='https://files.catbox.moe/xz97kh.pth'),
})


@register_model
def cvt_13(pretrained=False, **kwargs) -> CvT:
    model_args = dict(depths=(1, 2, 10), dims=(64, 192, 384), num_heads=(1, 3, 6))
    return _create_cvt('cvt_13', pretrained=pretrained, **dict(model_args, **kwargs))
