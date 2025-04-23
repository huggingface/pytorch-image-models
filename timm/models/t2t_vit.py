"""T2T-ViT
Paper: `Tokens-to-Token ViT: Training Vision Transformers From Scratch on ImageNet`
    - https://arxiv.org/pdf/2101.11986
    - https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_Tokens-to-Token_ViT_Training_Vision_Transformers_From_Scratch_on_ImageNet_ICCV_2021_paper.pdf

Model from official source: 
    - https://github.com/yitu-opensource/T2T-ViT

@InProceedings{Yuan_2021_ICCV,
    author    = {Yuan, Li and Chen, Yunpeng and Wang, Tao and Yu, Weihao and Shi, Yujun and Jiang, Zi-Hang and Tay, Francis E.H. and Feng, Jiashi and Yan, Shuicheng},
    title     = {Tokens-to-Token ViT: Training Vision Transformers From Scratch on ImageNet},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {558-567}
}

Original implementation by Wenhui Yuan et al.,
adapted for timm by Ryan Hou and Ross Wightman, original copyright below
"""
# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import Mlp, LayerNorm, DropPath, trunc_normal_, to_2tuple

from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint
from ._registry import generate_default_cfgs, register_model

def get_sinusoid_encoding(n_position: int, d_hid: int) -> torch.Tensor:
    ''' Sinusoid position encoding table using PyTorch '''

    # Create a position tensor of shape (n_position, 1)
    position = torch.arange(n_position, dtype=torch.float32).unsqueeze(1)

    # Compute the divisor term: 1 / (10000^(2i/d_hid))
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_hid))

    # Compute the sinusoid table
    sinusoid_table = torch.zeros(n_position, d_hid)
    sinusoid_table[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
    sinusoid_table[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

    return sinusoid_table.unsqueeze(0)  # Add batch dimension

class Token_attention(nn.Module):
    def __init__(
            self, 
            dim: int,
            in_dim: int, 
            num_heads: int = 8, 
            qkv_bias: bool = False, 
            qk_scale: Optional[float] = None, 
            attn_drop: float = 0., 
            proj_drop: float = 0., 
    ):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection
        return x

class Token_transformer(nn.Module):
    def __init__(
            self, 
            dim: int, 
            in_dim: int, 
            num_heads: int = 1, 
            mlp_ratio: float = 1., 
            qkv_bias: bool = False, 
            qk_scale: Optional[float] = None, 
            drop_rate: float = 0.,
            drop_path: float = 0.,
            attn_drop: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = partial(LayerNorm, eps=1e-5),
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Token_attention(
            dim, 
            in_dim=in_dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=drop_rate
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(
            in_features=in_dim, 
            hidden_features=int(in_dim*mlp_ratio), 
            out_features=in_dim, 
            drop=drop_rate,
            act_layer=act_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Token_performer(nn.Module):
    def __init__(
            self, 
            dim: int, 
            in_dim: int, 
            head_cnt: int = 1, 
            kernel_ratio: float = 0.5, 
            dp1: float = 0.1, 
            dp2: float = 0.1,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = partial(LayerNorm, eps=1e-5),
    ):
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            act_layer(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        # self.w = torch.randn(self.m, self.emb)
        # self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)
        self.register_buffer('w', nn.init.orthogonal_(torch.randn(self.m, self.emb)) * math.sqrt(self.m))

    def prm_exp(self, x: torch.Tensor) -> torch.Tensor:
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch 
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x: torch.Tensor) -> torch.Tensor:
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(
            self, 
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            token_dim: int = 64,
            tokens_type: Literal['performer', 'transformer'] = 'performer',
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = partial(LayerNorm, eps=1e-5),
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
        self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        
        token_module = Token_performer if tokens_type == 'performer' else Token_transformer

        self.attention1 = token_module(dim=in_chans * 7 * 7, in_dim=token_dim, act_layer=act_layer, norm_layer=norm_layer)
        self.attention2 = token_module(dim=token_dim * 3 * 3, in_dim=token_dim, act_layer=act_layer, norm_layer=norm_layer)
        self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def feat_ratio(self, as_scalar: bool = True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size
    
    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        x = x.transpose(1,2).reshape(B, -1,  H // 4, W // 4)
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        x = x.transpose(1, 2).reshape(B, -1,  H // 8, W // 8)
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, 
            dim: int, 
            num_heads: int = 8, 
            qkv_bias: bool = False, 
            qk_scale: Optional[float] = None, 
            attn_drop: float = 0., 
            proj_drop: float = 0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
            self, 
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False, 
            qk_scale: Optional[float] = None, 
            drop_rate: float = 0.,
            drop_path: float = 0.,
            attn_drop: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = partial(LayerNorm, eps=1e-5),
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=drop_rate,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            drop=drop_rate,
            act_layer=act_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class T2T_ViT(nn.Module):
    def __init__(
            self, 
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            tokens_type: Literal['performer', 'transformer'] = 'performer',
            token_dim: int = 64,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_scale: Optional[float] = None, 
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = partial(LayerNorm, eps=1e-5),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1
        self.grad_checkpointing = False

        self.patch_embed = T2T_module(
            img_size=img_size, 
            patch_size=patch_size,
            in_chans=in_chans, 
            embed_dim=embed_dim, 
            tokens_type=tokens_type,
            token_dim=token_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        num_patches = self.patch_embed.num_patches
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False)
        self.register_buffer('pos_embed', get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop_rate=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]

        # class_head
        use_fc_norm = False
        self.global_pool = 'token'
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'cls_token'}
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable
    
    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if an int, if is a sequence, select by matching indices
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width = x.shape

        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]

        for i, blk in enumerate(blocks):
            x = blk(x)
            if i in take_indices:
                # normalize intermediates with final norm layer if enabled
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        
        if reshape:
            # reshape to BCHW output format
            H, W = self.patch_embed.dynamic_feat_size((height, width))
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
        if not torch.jit.is_scripting() and return_prefix_tokens:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates
    
    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)
        return x
    
    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

def _cfg(url: str = '', **kwargs: Any) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 0.9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.project', 'classifier': 'head',
        'paper_ids': 'https://arxiv.org/pdf/2101.11986',
        'paper_name': 'Tokens-to-Token ViT: Training Vision Transformers From Scratch on ImageNet',
        'origin_url': 'https://github.com/yitu-opensource/T2T-ViT',
        **kwargs
    }

def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: T2T_ViT,
) -> Dict[str, torch.Tensor]:
    if 'state_dict_ema' in state_dict:
        state_dict = state_dict['state_dict_ema']

    if 'patch_embed.project.weight' in state_dict:
        return state_dict
    
    out_dict = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        k = k.replace('tokens_to_token.', 'patch_embed.')
        out_dict[k] = v

    return out_dict

default_cfgs = generate_default_cfgs({
    't2t_vit_7.in1k': _cfg(
        # hf_hub_id='timm/',
        url='https://github.com/yitu-opensource/T2T-ViT/releases/download/main/71.7_T2T_ViT_7.pth.tar',
    ),
    't2t_vit_10.in1k': _cfg(
        # hf_hub_id='timm/',
        url='https://github.com/yitu-opensource/T2T-ViT/releases/download/main/75.2_T2T_ViT_10.pth.tar'
    ),
    't2t_vit_12.in1k': _cfg(
        # hf_hub_id='timm/',
        url='https://github.com/yitu-opensource/T2T-ViT/releases/download/main/76.5_T2T_ViT_12.pth.tar'
    ),
    't2t_vit_14.in1k': _cfg(
        # hf_hub_id='timm/',
        url='https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.5_T2T_ViT_14.pth.tar'
    ),
    't2t_vit_19.in1k': _cfg(
        # f_hub_id='timm/',
        url='https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.9_T2T_ViT_19.pth.tar'
    ),
    't2t_vit_24.in1k': _cfg(
        # hf_hub_id='timm/',
        url='https://github.com/yitu-opensource/T2T-ViT/releases/download/main/82.3_T2T_ViT_24.pth.tar'
    ),
    't2t_vit_t_14.in1k': _cfg(
        # hf_hub_id='timm/',
        url='https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.7_T2T_ViTt_14.pth.tar'
    ),
    't2t_vit_t_19.in1k': _cfg(
        # hf_hub_id='timm/',
        url='https://github.com/yitu-opensource/T2T-ViT/releases/download/main/82.4_T2T_ViTt_19.pth.tar'
    ),
    't2t_vit_t_24.in1k': _cfg(
        # hf_hub_id='timm/',
        url='https://github.com/yitu-opensource/T2T-ViT/releases/download/main/82.6_T2T_ViTt_24.pth.tar'
    ),
})

def _create_t2t_vit(variant: str, pretrained: bool, **kwargs: Any) -> T2T_ViT:
    out_indices = kwargs.pop('out_indices', 3)
    model = build_model_with_cfg(
        T2T_ViT, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model

@register_model
def t2t_vit_7(pretrained: bool = False, **kwargs: Any) -> T2T_ViT:
    model_kwargs = dict(
        tokens_type='performer', embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., **kwargs)
    model = _create_t2t_vit('t2t_vit_7', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def t2t_vit_10(pretrained: bool = False, **kwargs: Any) -> T2T_ViT:
    model_kwargs = dict(embed_dim=256, depth=10, num_heads=4, mlp_ratio=2., **kwargs)
    model = _create_t2t_vit('t2t_vit_10', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def t2t_vit_12(pretrained: bool = False, **kwargs: Any) -> T2T_ViT:
    model_kwargs = dict(embed_dim=256, depth=12, num_heads=4, mlp_ratio=2., **kwargs)
    model = _create_t2t_vit('t2t_vit_12', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def t2t_vit_14(pretrained: bool = False, **kwargs: Any) -> T2T_ViT:
    model_kwargs = dict(embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model = _create_t2t_vit('t2t_vit_14', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def t2t_vit_19(pretrained: bool = False, **kwargs: Any) -> T2T_ViT:
    model_kwargs = dict(embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model = _create_t2t_vit('t2t_vit_19', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def t2t_vit_24(pretrained: bool = False, **kwargs: Any) -> T2T_ViT:
    model_kwargs = dict(embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model = _create_t2t_vit('t2t_vit_24', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def t2t_vit_t_14(pretrained: bool = False, **kwargs: Any) -> T2T_ViT:
    model_kwargs = dict(
        tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model = _create_t2t_vit('t2t_vit_t_14', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def t2t_vit_t_19(pretrained: bool = False, **kwargs: Any) -> T2T_ViT:
    model_kwargs = dict(
        tokens_type='transformer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model = _create_t2t_vit('t2t_vit_t_19', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def t2t_vit_t_24(pretrained: bool = False, **kwargs: Any) -> T2T_ViT:
    model_kwargs = dict(
        tokens_type='transformer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model = _create_t2t_vit('t2t_vit_t_24', pretrained=pretrained, **model_kwargs)
    return model