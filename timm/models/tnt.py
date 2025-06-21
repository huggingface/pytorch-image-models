""" Transformer in Transformer (TNT) in PyTorch

A PyTorch implement of TNT as described in
'Transformer in Transformer' - https://arxiv.org/abs/2103.00112

The official mindspore code is released and available at
https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/TNT

The official pytorch code is released and available at
https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/tnt_pytorch
"""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import Mlp, DropPath, trunc_normal_, _assert, to_2tuple, resample_abs_pos_embed
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint
from ._registry import generate_default_cfgs, register_model

__all__ = ['TNT']  # model_registry will add each entrypoint fn to this


class Attention(nn.Module):
    """ Multi-Head Attention
    """

    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """ TNT Block
    """

    def __init__(
            self,
            dim,
            dim_out,
            num_pixel,
            num_heads_in=4,
            num_heads_out=12,
            mlp_ratio=4.,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            legacy=False,
    ):
        super().__init__()
        # Inner transformer
        self.norm_in = norm_layer(dim)
        self.attn_in = Attention(
            dim,
            dim,
            num_heads=num_heads_in,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm_mlp_in = norm_layer(dim)
        self.mlp_in = Mlp(
            in_features=dim,
            hidden_features=int(dim * 4),
            out_features=dim,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.legacy = legacy
        if self.legacy:
            self.norm1_proj = norm_layer(dim)
            self.proj = nn.Linear(dim * num_pixel, dim_out, bias=True)
            self.norm2_proj = None
        else:
            self.norm1_proj = norm_layer(dim * num_pixel)
            self.proj = nn.Linear(dim * num_pixel, dim_out, bias=False)
            self.norm2_proj = norm_layer(dim_out)

        # Outer transformer
        self.norm_out = norm_layer(dim_out)
        self.attn_out = Attention(
            dim_out,
            dim_out,
            num_heads=num_heads_out,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_mlp = norm_layer(dim_out)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=int(dim_out * mlp_ratio),
            out_features=dim_out,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, pixel_embed, patch_embed):
        # inner
        pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        # outer
        B, N, C = patch_embed.size()
        if self.norm2_proj is None:
            patch_embed = torch.cat([
                patch_embed[:, 0:1],
                patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1)),
            ], dim=1)
        else:
            patch_embed = torch.cat([
                patch_embed[:, 0:1],
                patch_embed[:, 1:] + self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, N - 1, -1)))),
            ], dim=1)
        patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        return pixel_embed, patch_embed


class PixelEmbed(nn.Module):
    """ Image to Pixel Embedding
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            in_dim=48,
            stride=4,
            legacy=False,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # grid_size property necessary for resizing positional embedding
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        num_patches = (self.grid_size[0]) * (self.grid_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.legacy = legacy
        self.num_patches = num_patches
        self.in_dim = in_dim
        new_patch_size = [math.ceil(ps / stride) for ps in patch_size]
        self.new_patch_size = new_patch_size

        self.proj = nn.Conv2d(in_chans, self.in_dim, kernel_size=7, padding=3, stride=stride)
        if self.legacy:
            self.unfold = nn.Unfold(kernel_size=new_patch_size, stride=new_patch_size)
        else:
            self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def forward(self, x: torch.Tensor, pixel_pos: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        _assert(
            H == self.img_size[0],
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        _assert(
            W == self.img_size[1],
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        if self.legacy:
            x = self.proj(x)
            x = self.unfold(x)
            x = x.transpose(1, 2).reshape(
                B * self.num_patches, self.in_dim, self.new_patch_size[0], self.new_patch_size[1])
        else:
            x = self.unfold(x)
            x = x.transpose(1, 2).reshape(B * self.num_patches, C, self.patch_size[0], self.patch_size[1])
            x = self.proj(x)
        x = x + pixel_pos
        x = x.reshape(B * self.num_patches, self.in_dim, -1).transpose(1, 2)
        return x


class TNT(nn.Module):
    """ Transformer in Transformer - https://arxiv.org/abs/2103.00112
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            global_pool='token',
            embed_dim=768,
            inner_dim=48,
            depth=12,
            num_heads_inner=4,
            num_heads_outer=12,
            mlp_ratio=4.,
            qkv_bias=False,
            drop_rate=0.,
            pos_drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            first_stride=4,
            legacy=False,
    ):
        super().__init__()
        assert global_pool in ('', 'token', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1
        self.grad_checkpointing = False

        self.pixel_embed = PixelEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            in_dim=inner_dim,
            stride=first_stride,
            legacy=legacy,
        )
        num_patches = self.pixel_embed.num_patches
        r = self.pixel_embed.feat_ratio() if hasattr(self.pixel_embed, 'feat_ratio') else patch_size
        self.num_patches = num_patches
        new_patch_size = self.pixel_embed.new_patch_size
        num_pixel = new_patch_size[0] * new_patch_size[1]

        self.norm1_proj = norm_layer(num_pixel * inner_dim)
        self.proj = nn.Linear(num_pixel * inner_dim, embed_dim)
        self.norm2_proj = norm_layer(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_pos = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pixel_pos = nn.Parameter(torch.zeros(1, inner_dim, new_patch_size[0], new_patch_size[1]))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        for i in range(depth):
            blocks.append(Block(
                dim=inner_dim,
                dim_out=embed_dim,
                num_pixel=num_pixel,
                num_heads_in=num_heads_inner,
                num_heads_out=num_heads_outer,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                legacy=legacy,
            ))
        self.blocks = nn.ModuleList(blocks)
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]

        self.norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.patch_pos, std=.02)
        trunc_normal_(self.pixel_pos, std=.02)
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
    def no_weight_decay(self):
        return {'patch_pos', 'pixel_pos', 'cls_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|patch_pos|pixel_pos|pixel_embed|norm[12]_proj|proj',  # stem and embed / pos
            blocks=[
                (r'^blocks\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'token', 'avg')
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

        pixel_embed = self.pixel_embed(x, self.pixel_pos)

        patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
        patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.pos_drop(patch_embed)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]

        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                pixel_embed, patch_embed = checkpoint(blk, pixel_embed, patch_embed)
            else:
                pixel_embed, patch_embed = blk(pixel_embed, patch_embed)
            if i in take_indices:
                # normalize intermediates with final norm layer if enabled
                intermediates.append(self.norm(patch_embed) if norm else patch_embed)

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]

        if reshape:
            # reshape to BCHW output format
            H, W = self.pixel_embed.dynamic_feat_size((height, width))
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
        if not torch.jit.is_scripting() and return_prefix_tokens:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        patch_embed = self.norm(patch_embed)

        return patch_embed, intermediates

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
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        B = x.shape[0]
        pixel_embed = self.pixel_embed(x, self.pixel_pos)

        patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
        patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.pos_drop(patch_embed)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                pixel_embed, patch_embed = checkpoint(blk, pixel_embed, patch_embed)
            else:
                pixel_embed, patch_embed = blk(pixel_embed, patch_embed)

        patch_embed = self.norm(patch_embed)
        return patch_embed

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'pixel_embed.proj', 'classifier': 'head',
        'paper_ids': 'arXiv:2103.00112',
        'paper_name': 'Transformer in Transformer',
        'origin_url': 'https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/tnt_pytorch',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'tnt_s_legacy_patch16_224.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/contrastive/pytorch-image-models/releases/download/TNT/tnt_s_patch16_224.pth.tar',
    ),
    'tnt_s_patch16_224.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/tnt/tnt_s_81.5.pth.tar',
    ),
    'tnt_b_patch16_224.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/tnt/tnt_b_82.9.pth.tar',
    ),
})


def checkpoint_filter_fn(state_dict, model):
    state_dict.pop('outer_tokens', None)
    if 'patch_pos' in state_dict:
        out_dict = state_dict
    else:
        out_dict = {}
        for k, v in state_dict.items():
            k = k.replace('outer_pos', 'patch_pos')
            k = k.replace('inner_pos', 'pixel_pos')
            k = k.replace('patch_embed', 'pixel_embed')
            k = k.replace('proj_norm1', 'norm1_proj')
            k = k.replace('proj_norm2', 'norm2_proj')
            k = k.replace('inner_norm1', 'norm_in')
            k = k.replace('inner_attn', 'attn_in')
            k = k.replace('inner_norm2', 'norm_mlp_in')
            k = k.replace('inner_mlp', 'mlp_in')
            k = k.replace('outer_norm1', 'norm_out')
            k = k.replace('outer_attn', 'attn_out')
            k = k.replace('outer_norm2', 'norm_mlp')
            k = k.replace('outer_mlp', 'mlp')
            if k == 'pixel_pos' and model.pixel_embed.legacy == False:
                B, N, C = v.shape
                H = W = int(N ** 0.5)
                assert H * W == N
                v = v.permute(0, 2, 1).reshape(B, C, H, W)
            out_dict[k] = v

    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    if out_dict['patch_pos'].shape != model.patch_pos.shape:
        out_dict['patch_pos'] = resample_abs_pos_embed(
            out_dict['patch_pos'],
            new_size=model.pixel_embed.grid_size,
            num_prefix_tokens=1,
        )
    return out_dict


def _create_tnt(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    model = build_model_with_cfg(
        TNT, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs)
    return model


@register_model
def tnt_s_legacy_patch16_224(pretrained=False, **kwargs) -> TNT:
    model_cfg = dict(
        patch_size=16, embed_dim=384, inner_dim=24, depth=12, num_heads_outer=6,
        qkv_bias=False, legacy=True)
    model = _create_tnt('tnt_s_legacy_patch16_224', pretrained=pretrained, **dict(model_cfg, **kwargs))
    return model


@register_model
def tnt_s_patch16_224(pretrained=False, **kwargs) -> TNT:
    model_cfg = dict(
        patch_size=16, embed_dim=384, inner_dim=24, depth=12, num_heads_outer=6,
        qkv_bias=False)
    model = _create_tnt('tnt_s_patch16_224', pretrained=pretrained, **dict(model_cfg, **kwargs))
    return model


@register_model
def tnt_b_patch16_224(pretrained=False, **kwargs) -> TNT:
    model_cfg = dict(
        patch_size=16, embed_dim=640, inner_dim=40, depth=12, num_heads_outer=10,
        qkv_bias=False)
    model = _create_tnt('tnt_b_patch16_224', pretrained=pretrained, **dict(model_cfg, **kwargs))
    return model
