""" Visformer

Paper: Visformer: The Vision-friendly Transformer - https://arxiv.org/abs/2104.12533

From original at https://github.com/danczs/Visformer

"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg, overlay_external_default_cfg
from .layers import to_2tuple, trunc_normal_, DropPath, PatchEmbed, LayerNorm2d, create_classifier
from .registry import register_model


__all__ = ['Visformer']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head',
        **kwargs
    }


default_cfgs = dict(
    visformer_tiny=_cfg(),
    visformer_small=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/visformer_small-839e1f5b.pth'
    ),
)


class SpatialMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., group=8, spatial_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.out_features = out_features
        self.spatial_conv = spatial_conv
        if self.spatial_conv:
            if group < 2:  # net setting
                hidden_features = in_features * 5 // 6
            else:
                hidden_features = in_features * 2
        self.hidden_features = hidden_features
        self.group = group
        self.drop = nn.Dropout(drop)
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0, bias=False)
        self.act1 = act_layer()
        if self.spatial_conv:
            self.conv2 = nn.Conv2d(
                hidden_features, hidden_features, 3, stride=1, padding=1, groups=self.group, bias=False)
            self.act2 = act_layer()
        else:
            self.conv2 = None
            self.act2 = None
        self.conv3 = nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop(x)
        if self.conv2 is not None:
            x = self.conv2(x)
            x = self.act2(x)
        x = self.conv3(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim_ratio=1., attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = round(dim // num_heads * head_dim_ratio)
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, head_dim * num_heads * 3, 1, stride=1, padding=0, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(self.head_dim * self.num_heads, dim, 1, stride=1, padding=0, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        q, k, v = x[0], x[1], x[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, head_dim_ratio=1., mlp_ratio=4.,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm2d,
                 group=8, attn_disabled=False, spatial_conv=False):
        super().__init__()
        self.spatial_conv = spatial_conv
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if attn_disabled:
            self.norm1 = None
            self.attn = None
        else:
            self.norm1 = norm_layer(dim)
            self.attn = Attention(
                dim, num_heads=num_heads, head_dim_ratio=head_dim_ratio, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SpatialMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
            group=group, spatial_conv=spatial_conv)  # new setting

    def forward(self, x):
        if self.attn is not None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Visformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, init_channels=32, embed_dim=384,
                 depth=12, num_heads=6, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=LayerNorm2d, attn_stage='111', pos_embed=True, spatial_conv='111',
                 vit_stem=False, group=8, global_pool='avg', conv_init=False, embed_norm=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.init_channels = init_channels
        self.img_size = img_size
        self.vit_stem = vit_stem
        self.conv_init = conv_init
        if isinstance(depth, (list, tuple)):
            self.stage_num1, self.stage_num2, self.stage_num3 = depth
            depth = sum(depth)
        else:
            self.stage_num1 = self.stage_num3 = depth // 3
            self.stage_num2 = depth - self.stage_num1 - self.stage_num3
        self.pos_embed = pos_embed
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # stage 1
        if self.vit_stem:
            self.stem = None
            self.patch_embed1 = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                embed_dim=embed_dim, norm_layer=embed_norm, flatten=False)
            img_size = [x // 16 for x in img_size]
        else:
            if self.init_channels is None:
                self.stem = None
                self.patch_embed1 = PatchEmbed(
                    img_size=img_size, patch_size=patch_size // 2, in_chans=in_chans,
                    embed_dim=embed_dim // 2, norm_layer=embed_norm, flatten=False)
                img_size = [x // 8 for x in img_size]
            else:
                self.stem = nn.Sequential(
                    nn.Conv2d(in_chans, self.init_channels, 7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(self.init_channels),
                    nn.ReLU(inplace=True)
                )
                img_size = [x // 2 for x in img_size]
                self.patch_embed1 = PatchEmbed(
                    img_size=img_size, patch_size=patch_size // 4, in_chans=self.init_channels,
                    embed_dim=embed_dim // 2, norm_layer=embed_norm, flatten=False)
                img_size = [x // 4 for x in img_size]

        if self.pos_embed:
            if self.vit_stem:
                self.pos_embed1 = nn.Parameter(torch.zeros(1, embed_dim, *img_size))
            else:
                self.pos_embed1 = nn.Parameter(torch.zeros(1, embed_dim//2, *img_size))
            self.pos_drop = nn.Dropout(p=drop_rate)
        self.stage1 = nn.ModuleList([
            Block(
                dim=embed_dim//2, num_heads=num_heads, head_dim_ratio=0.5, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                group=group, attn_disabled=(attn_stage[0] == '0'), spatial_conv=(spatial_conv[0] == '1')
            )
            for i in range(self.stage_num1)
        ])

        # stage2
        if not self.vit_stem:
            self.patch_embed2 = PatchEmbed(
                img_size=img_size, patch_size=patch_size // 8, in_chans=embed_dim // 2,
                embed_dim=embed_dim, norm_layer=embed_norm, flatten=False)
            img_size = [x // 2 for x in img_size]
            if self.pos_embed:
                self.pos_embed2 = nn.Parameter(torch.zeros(1, embed_dim, *img_size))
        self.stage2 = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, head_dim_ratio=1.0, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                group=group, attn_disabled=(attn_stage[1] == '0'), spatial_conv=(spatial_conv[1] == '1')
            )
            for i in range(self.stage_num1, self.stage_num1+self.stage_num2)
        ])

        # stage 3
        if not self.vit_stem:
            self.patch_embed3 = PatchEmbed(
                img_size=img_size, patch_size=patch_size // 8, in_chans=embed_dim,
                embed_dim=embed_dim * 2, norm_layer=embed_norm, flatten=False)
            img_size = [x // 2 for x in img_size]
            if self.pos_embed:
                self.pos_embed3 = nn.Parameter(torch.zeros(1, embed_dim*2, *img_size))
        self.stage3 = nn.ModuleList([
            Block(
                dim=embed_dim*2, num_heads=num_heads, head_dim_ratio=1.0, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                group=group, attn_disabled=(attn_stage[2] == '0'), spatial_conv=(spatial_conv[2] == '1')
            )
            for i in range(self.stage_num1+self.stage_num2, depth)
        ])

        # head
        self.num_features = embed_dim if self.vit_stem else embed_dim * 2
        self.norm = norm_layer(self.num_features)
        self.global_pool, self.head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        # weights init
        if self.pos_embed:
            trunc_normal_(self.pos_embed1, std=0.02)
            if not self.vit_stem:
                trunc_normal_(self.pos_embed2, std=0.02)
                trunc_normal_(self.pos_embed3, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            if self.conv_init:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        if self.stem is not None:
            x = self.stem(x)

        # stage 1
        x = self.patch_embed1(x)
        if self.pos_embed:
            x = x + self.pos_embed1
            x = self.pos_drop(x)
        for b in self.stage1:
            x = b(x)

        # stage 2
        if not self.vit_stem:
            x = self.patch_embed2(x)
            if self.pos_embed:
                x = x + self.pos_embed2
                x = self.pos_drop(x)
        for b in self.stage2:
            x = b(x)

        # stage3
        if not self.vit_stem:
            x = self.patch_embed3(x)
            if self.pos_embed:
                x = x + self.pos_embed3
                x = self.pos_drop(x)
        for b in self.stage3:
            x = b(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = self.head(x)
        return x


def _create_visformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    model = build_model_with_cfg(
        Visformer, variant, pretrained,
        default_cfg=default_cfgs[variant],
        **kwargs)
    return model


@register_model
def visformer_tiny(pretrained=False, **kwargs):
    model_cfg = dict(
        init_channels=16, embed_dim=192, depth=(7, 4, 4), num_heads=3, mlp_ratio=4., group=8,
        attn_stage='011', spatial_conv='100', norm_layer=nn.BatchNorm2d, conv_init=True,
        embed_norm=nn.BatchNorm2d, **kwargs)
    model = _create_visformer('visformer_tiny', pretrained=pretrained, **model_cfg)
    return model


@register_model
def visformer_small(pretrained=False, **kwargs):
    model_cfg = dict(
        init_channels=32, embed_dim=384, depth=(7, 4, 4), num_heads=6, mlp_ratio=4., group=8,
        attn_stage='011', spatial_conv='100', norm_layer=nn.BatchNorm2d, conv_init=True,
        embed_norm=nn.BatchNorm2d, **kwargs)
    model = _create_visformer('visformer_small', pretrained=pretrained, **model_cfg)
    return model


# @register_model
# def visformer_net1(pretrained=False, **kwargs):
#     model = Visformer(
#         init_channels=None, embed_dim=384, depth=(0, 12, 0), num_heads=6, mlp_ratio=4., attn_stage='111',
#         spatial_conv='000', vit_stem=True, conv_init=True, **kwargs)
#     model.default_cfg = _cfg()
#     return model
#
#
# @register_model
# def visformer_net2(pretrained=False, **kwargs):
#     model = Visformer(
#         init_channels=32, embed_dim=384, depth=(0, 12, 0), num_heads=6, mlp_ratio=4., attn_stage='111',
#         spatial_conv='000', vit_stem=False, conv_init=True, **kwargs)
#     model.default_cfg = _cfg()
#     return model
#
#
# @register_model
# def visformer_net3(pretrained=False, **kwargs):
#     model = Visformer(
#         init_channels=32, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., attn_stage='111',
#         spatial_conv='000', vit_stem=False, conv_init=True, **kwargs)
#     model.default_cfg = _cfg()
#     return model
#
#
# @register_model
# def visformer_net4(pretrained=False, **kwargs):
#     model = Visformer(
#         init_channels=32, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., attn_stage='111',
#         spatial_conv='000', vit_stem=False, conv_init=True, **kwargs)
#     model.default_cfg = _cfg()
#     return model
#
#
# @register_model
# def visformer_net5(pretrained=False, **kwargs):
#     model = Visformer(
#         init_channels=32, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., group=1, attn_stage='111',
#         spatial_conv='111', vit_stem=False, conv_init=True, **kwargs)
#     model.default_cfg = _cfg()
#     return model
#
#
# @register_model
# def visformer_net6(pretrained=False, **kwargs):
#     model = Visformer(
#         init_channels=32, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., group=1, attn_stage='111',
#         pos_embed=False, spatial_conv='111', conv_init=True, **kwargs)
#     model.default_cfg = _cfg()
#     return model
#
#
# @register_model
# def visformer_net7(pretrained=False, **kwargs):
#     model = Visformer(
#         init_channels=32, embed_dim=384, depth=(6, 7, 7), num_heads=6, group=1, attn_stage='000',
#         pos_embed=False, spatial_conv='111', conv_init=True, **kwargs)
#     model.default_cfg = _cfg()
#     return model




