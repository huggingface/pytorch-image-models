""" Transformer in Transformer (TNT) in PyTorch

A PyTorch implement of TNT as described in
'Transformer in Transformer' - https://arxiv.org/abs/2103.00112

The official mindspore code is released and available at
https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/TNT
"""
import math
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import Mlp, DropPath, trunc_normal_
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import _assert
from timm.models.registry import register_model
from timm.models.vision_transformer import resize_pos_embed


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'pixel_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'tnt_s_patch16_224': _cfg(
        url='https://github.com/contrastive/pytorch-image-models/releases/download/TNT/tnt_s_patch16_224.pth.tar',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_b_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


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
        q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
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
    def __init__(self, dim, in_dim, num_pixel, num_heads=12, in_num_head=4, mlp_ratio=4.,
            qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # Inner transformer
        self.norm_in = norm_layer(in_dim)
        self.attn_in = Attention(
            in_dim, in_dim, num_heads=in_num_head, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        
        self.norm_mlp_in = norm_layer(in_dim)
        self.mlp_in = Mlp(in_features=in_dim, hidden_features=int(in_dim * 4),
            out_features=in_dim, act_layer=act_layer, drop=drop)
        
        self.norm1_proj = norm_layer(in_dim)
        self.proj = nn.Linear(in_dim * num_pixel, dim, bias=True)
        # Outer transformer
        self.norm_out = norm_layer(dim)
        self.attn_out = Attention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
            out_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, pixel_embed, patch_embed):
        # inner
        pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        # outer
        B, N, C = patch_embed.size()
        patch_embed = torch.cat(
            [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
            dim=1)
        patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        return pixel_embed, patch_embed


class PixelEmbed(nn.Module):
    """ Image to Pixel Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, in_dim=48, stride=4):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # grid_size property necessary for resizing positional embedding
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        num_patches = (self.grid_size[0]) * (self.grid_size[1])
        self.img_size = img_size
        self.num_patches = num_patches
        self.in_dim = in_dim
        new_patch_size = [math.ceil(ps / stride) for ps in patch_size]
        self.new_patch_size = new_patch_size

        self.proj = nn.Conv2d(in_chans, self.in_dim, kernel_size=7, padding=3, stride=stride)
        self.unfold = nn.Unfold(kernel_size=new_patch_size, stride=new_patch_size)

    def forward(self, x, pixel_pos):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0],
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        _assert(W == self.img_size[1],
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        x = self.proj(x)
        x = self.unfold(x)
        x = x.transpose(1, 2).reshape(B * self.num_patches, self.in_dim, self.new_patch_size[0], self.new_patch_size[1])
        x = x + pixel_pos
        x = x.reshape(B * self.num_patches, self.in_dim, -1).transpose(1, 2)
        return x


class TNT(nn.Module):
    """ Transformer in Transformer - https://arxiv.org/abs/2103.00112
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, in_dim=48, depth=12,
                 num_heads=12, in_num_head=4, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, first_stride=4):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.pixel_embed = PixelEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, in_dim=in_dim, stride=first_stride)
        num_patches = self.pixel_embed.num_patches
        self.num_patches = num_patches
        new_patch_size = self.pixel_embed.new_patch_size
        num_pixel = new_patch_size[0] * new_patch_size[1]
        
        self.norm1_proj = norm_layer(num_pixel * in_dim)
        self.proj = nn.Linear(num_pixel * in_dim, embed_dim)
        self.norm2_proj = norm_layer(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_pos = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pixel_pos = nn.Parameter(torch.zeros(1, in_dim, new_patch_size[0], new_patch_size[1]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        for i in range(depth):
            blocks.append(Block(
                dim=embed_dim, in_dim=in_dim, num_pixel=num_pixel, num_heads=num_heads, in_num_head=in_num_head,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)

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

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        pixel_embed = self.pixel_embed(x, self.pixel_pos)
        
        patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
        patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.pos_drop(patch_embed)

        for blk in self.blocks:
            pixel_embed, patch_embed = blk(pixel_embed, patch_embed)

        patch_embed = self.norm(patch_embed)
        return patch_embed[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    if state_dict['patch_pos'].shape != model.patch_pos.shape:
        state_dict['patch_pos'] = resize_pos_embed(state_dict['patch_pos'],
            model.patch_pos, getattr(model, 'num_tokens', 1), model.pixel_embed.grid_size)
    return state_dict


def _create_tnt(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        TNT, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model


@register_model
def tnt_s_patch16_224(pretrained=False, **kwargs):
    model_cfg = dict(
        patch_size=16, embed_dim=384, in_dim=24, depth=12, num_heads=6, in_num_head=4,
        qkv_bias=False, **kwargs)
    model = _create_tnt('tnt_s_patch16_224', pretrained=pretrained, **model_cfg)
    return model


@register_model
def tnt_b_patch16_224(pretrained=False, **kwargs):
    model_cfg = dict(
        patch_size=16, embed_dim=640, in_dim=40, depth=12, num_heads=10, in_num_head=4,
        qkv_bias=False, **kwargs)
    model = _create_tnt('tnt_b_patch16_224', pretrained=pretrained, **model_cfg)
    return model
