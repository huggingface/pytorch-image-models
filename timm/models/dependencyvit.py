from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import Mlp
from timm.models.vision_transformer import VisionTransformer
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['DependencyViT']


# FIXME there is nearly no difference between this and stock attn, allowing sdpa to be used if a workaround can be found
class ReversedAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.track_dependency_mask = False
        self.dependency_mask = None

        self.head_selector = nn.Linear(dim, num_heads, bias=False) # paper only mentions a weight matrix, assuming no bias

        self.message_controller = Mlp(
            in_features = dim,
            hidden_features = int(dim/2),
            out_features = 1,
            act_layer = nn.GELU,
            bias = False, # FIXME is there a bias term?
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # m is cumulative over all layers (m = m_i * m_i-1 * ... * m_1)
    def forward(self, in_tuple: Tuple[torch.Tensor, Union[int, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, m = in_tuple # [B, N, C], [B, 1, 1, N]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        p = self.head_selector(x).softmax(dim=-1).transpose(-2, -1).reshape(B, self.num_heads, 1, N)

        m = m * self.message_controller(x).sigmoid().reshape(B, 1, 1, N)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn).transpose(-2, -1)
        attn = attn * p * m # [B, n_h, N, N]
        x = attn @ v

        self.dependency_mask = attn.sum(1) if self.track_dependency_mask else None

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return (x, m)

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DependencyViTBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ReversedAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, in_tuple: Tuple[torch.Tensor, Union[int, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, m = in_tuple
        x, m = self.attn((self.norm1(x), m))
        x = x + self.drop_path1(self.ls1(x))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return (x, m)

class DependencyViT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, 
            **kwargs,
            block_fn = DependencyViTBlock, 
            class_token=False,
            global_pool='avg', 
            qkv_bias=False, 
            init_values=1e-6, 
            fc_norm=False,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x, _ = checkpoint_seq(self.blocks, (x,1))
        else:
            x, _ = self.blocks((x, 1))
        x = self.norm(x)
        return x


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
    }


default_cfgs = {
    'dependencyvit_tiny_patch16_224.untrained': _cfg(url=''),
}


default_cfgs = generate_default_cfgs(default_cfgs)



def _create_dependencyvit(variant: str, pretrained: bool = False, **kwargs) -> DependencyViT:
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    return build_model_with_cfg(
        DependencyViT,
        variant,
        pretrained,
        **kwargs,
    )

@register_model
def dependencyvit_tiny_patch16_224(pretrained: bool = False, **kwargs) -> DependencyViT:
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_dependencyvit('dependencyvit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
