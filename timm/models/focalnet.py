# --------------------------------------------------------
# FocalNets -- Focal Modulation Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, trunc_normal_, _assert
from ._builder import build_model_with_cfg
from ._features_fx import register_notrace_function
from ._registry import register_model

__all__ = ['FocalNet']


class FocalModulation(nn.Module):
    def __init__(
            self,
            dim,
            focal_window,
            focal_level,
            focal_factor=2,
            bias=True,
            proj_drop=0.,
            use_postln_in_modulation=False,
            normalize_modulator=False,
    ):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim, dim, kernel_size=kernel_size, stride=1,
                        groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[-1]

        # pre linear projection
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        # context aggreation
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * self.gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q * self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class FocalNetBlock(nn.Module):
    r""" Focal Modulation Network Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): Number of focal levels.
        focal_window (int): Focal window size at first focal level
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether to use layernorm after modulation
    """

    def __init__(
            self,
            dim,
            input_resolution,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            focal_level=1,
            focal_window=3,
            layerscale_value=1e-4,
            use_postln=False,
            use_postln_in_modulation=False,
            normalize_modulator=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_postln = use_postln

        self.norm1 = norm_layer(dim)
        self.modulation = FocalModulation(
            dim,
            proj_drop=drop,
            focal_window=focal_window,
            focal_level=self.focal_level,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if layerscale_value is not None:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones(dim))
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones(dim))

        self.H = None
        self.W = None

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        shortcut = x

        # Focal Modulation
        x = x if self.use_postln else self.norm1(x)
        x = x.view(B, H, W, C)
        x = self.modulation(x).view(B, H * W, C)
        x = x if not self.use_postln else self.norm1(x)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * (self.norm2(self.mlp(x)) if self.use_postln else self.mlp(self.norm2(x))))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"mlp_ratio={self.mlp_ratio}"


class BasicLayer(nn.Module):
    """ A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at first focal level
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether to use layer norm after modulation
    """

    def __init__(
            self,
            dim,
            out_dim,
            input_resolution,
            depth,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            focal_level=1,
            focal_window=1,
            use_conv_embed=False,
            layerscale_value=1e-4,
            use_postln=False,
            use_postln_in_modulation=False,
            normalize_modulator=False
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            FocalNetBlock(
                dim=dim,
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                focal_level=focal_level,
                focal_window=focal_window,
                layerscale_value=layerscale_value,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                normalize_modulator=normalize_modulator,
            )
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution,
                patch_size=2,
                in_chans=dim,
                embed_dim=out_dim,
                use_conv_embed=use_conv_embed,
                norm_layer=norm_layer,
                is_stem=False
            )
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)
            x, Ho, Wo = self.downsample(x)
        else:
            Ho, Wo = H, W
        return x, Ho, Wo

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
            self,
            img_size=(224, 224),
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            use_conv_embed=False,
            norm_layer=None,
            is_stem=False,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        padding = 0
        kernel_size = patch_size
        stride = patch_size
        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7
                padding = 2
                stride = 4
            else:
                kernel_size = 3
                padding = 1
                stride = 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, H, W


class FocalNet(nn.Module):
    r""" Focal Modulation Networks (FocalNets)

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Focal Transformer layer.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        focal_levels (list): How many focal levels at all stages. Note that this excludes the finest-grain level.
            Default: [1, 1, 1, 1]
        focal_windows (list): The focal window size at all stages. Default: [7, 5, 3, 1]
        use_conv_embed (bool): Whether to use convolutional embedding.
        layerscale_value (float): Value for layer scale. Default: 1e-4
        use_postln (bool): Whether to use layernorm after modulation (it helps stablize training of large models)
    """

    def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            focal_levels=[2, 2, 2, 2],
            focal_windows=[3, 3, 3, 3],
            use_conv_embed=False,
            layerscale_value=None,
            use_postln=False,
            use_postln_in_modulation=False,
            normalize_modulator=False,
            **kwargs,
    ):
        super().__init__()

        self.num_layers = len(depths)
        embed_dim = [embed_dim * (2 ** i) for i in range(self.num_layers)]

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio

        # split image into patches using either non-overlapped embedding or overlapped embedding
        self.patch_embed = PatchEmbed(
            img_size=to_2tuple(img_size),
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            use_conv_embed=use_conv_embed,
            norm_layer=norm_layer if self.patch_norm else None,
            is_stem=True
        )

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dim[i_layer],
                out_dim=embed_dim[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                focal_level=focal_levels[i_layer],
                focal_window=focal_windows[i_layer],
                use_conv_embed=use_conv_embed,
                use_checkpoint=use_checkpoint,
                layerscale_value=layerscale_value,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                normalize_modulator=normalize_modulator
                )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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
        return {''}

    def forward_features(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    "focalnet_tiny_srf": _cfg(),
    "focalnet_small_srf": _cfg(url="https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_small_srf.pth"),
    "focalnet_base_srf": _cfg(),
    "focalnet_tiny_lrf": _cfg(),
    "focalnet_small_lrf": _cfg(),
    "focalnet_base_lrf": _cfg(url='https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_base_lrf.pth'),
    "focalnet_large_fl3": _cfg(url='https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_large_lrf_384.pth', input_size=(3, 384, 384), num_classes=21842),
    "focalnet_large_fl4": _cfg(url="https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_large_lrf_384_fl4.pth", input_size=(3, 384, 384), num_classes=21842),
}


def checkpoint_filter_fn(state_dict, model):
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if any([n in k for n in ('relative_position_index', 'relative_coords_table')]):
            continue  # skip buffers that should not be persistent
        out_dict[k] = v
    return out_dict


def _create_focalnet(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        FocalNet, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model


@register_model
def focalnet_tiny_srf(pretrained=False, **kwargs):
    model_kwargs = dict(depths=[2, 2, 6, 2], embed_dim=96, **kwargs)
    return _create_focalnet('focalnet_tiny_srf', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_small_srf(pretrained=False, **kwargs):
    model_kwargs = dict(depths=[2, 2, 18, 2], embed_dim=96, **kwargs)
    return _create_focalnet('focalnet_small_srf', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_base_srf(pretrained=False, **kwargs):
    model_kwargs = dict(depths=[2, 2, 18, 2], embed_dim=128, **kwargs)
    return _create_focalnet('focalnet_base_srf', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_tiny_lrf(pretrained=False, **kwargs):
    model_kwargs = dict(depths=[2, 2, 6, 2], embed_dim=96, focal_levels=[3, 3, 3, 3], **kwargs)
    return _create_focalnet('focalnet_tiny_lrf', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_small_lrf(pretrained=False, **kwargs):
    model_kwargs = dict(depths=[2, 2, 18, 2], embed_dim=96, focal_levels=[3, 3, 3, 3], **kwargs)
    return _create_focalnet('focalnet_small_lrf', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_base_lrf(pretrained=False, **kwargs):
    model_kwargs = dict(depths=[2, 2, 18, 2], embed_dim=128, focal_levels=[3, 3, 3, 3], **kwargs)
    return _create_focalnet('focalnet_base_lrf', pretrained=pretrained, **model_kwargs)

# FocalNet large+ models
@register_model
def focalnet_large_fl3(pretrained=False, **kwargs):
    model_kwargs = dict(depths=[2, 2, 18, 2], embed_dim=192, focal_levels=[3, 3, 3, 3], **kwargs)
    return _create_focalnet('focalnet_large_fl3', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_large_fl4(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=[2, 2, 18, 2], embed_dim=192, focal_levels=[4, 4, 4, 4],
        use_conv_embed=True, layerscale_value=1e-4, **kwargs)
    return _create_focalnet('focalnet_large_fl4', pretrained=pretrained, **model_kwargs)

#
# @register_model
# def focalnet_large_fl4(pretrained=False, **kwargs):
#     model_kwargs = dict(depths=[2, 2, 18, 2], embed_dim=192, focal_levels=[4, 4, 4, 4], **kwargs)
#     return _create_focalnet('focalnet_large_fl4', pretrained=pretrained, **model_kwargs)



@register_model
def focalnet_xlarge_fl3(pretrained=False, **kwargs):
    model_kwargs = dict(depths=[2, 2, 18, 2], embed_dim=256, focal_levels=[3, 3, 3, 3], **kwargs)
    return _create_focalnet('focalnet_xlarge_fl3', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_xlarge_fl4(pretrained=False, **kwargs):
    model_kwargs = dict(depths=[2, 2, 18, 2], embed_dim=256, focal_levels=[4, 4, 4, 4], **kwargs)
    return _create_focalnet('focalnet_xlarge_fl4', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_huge_fl3(pretrained=False, **kwargs):
    model_kwargs = dict(depths=[2, 2, 18, 2], embed_dim=352, focal_levels=[3, 3, 3, 3], **kwargs)
    return _create_focalnet('focalnet_huge_fl3', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_huge_fl4(pretrained=False, **kwargs):
    model_kwargs = dict(depths=[2, 2, 18, 2], embed_dim=352, focal_levels=[4, 4, 4, 4], **kwargs)
    return _create_focalnet('focalnet_huge_fl4', pretrained=pretrained, **model_kwargs)

