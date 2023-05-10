""" FocalNet

As described in `Focal Modulation Networks` - https://arxiv.org/abs/2203.11926

Significant modifications and refactoring from the original impl at https://github.com/microsoft/FocalNet

This impl is/has:
* fully convolutional, NCHW tensor layout throughout, seemed to have minimal performance impact but more flexible
* re-ordered downsample / layer so that striding always at beginning of layer (stage)
* no input size constraints or input resolution/H/W tracking through the model
* torchscript fixed and a number of quirks cleaned up
* feature extraction support via `features_only=True`
"""
# --------------------------------------------------------
# FocalNets -- Focal Modulation Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import Mlp, DropPath, LayerNorm2d, trunc_normal_, ClassifierHead, NormMlpClassifierHead
from ._builder import build_model_with_cfg
from ._manipulate import named_apply
from ._registry import generate_default_cfgs, register_model

__all__ = ['FocalNet']


class FocalModulation(nn.Module):
    def __init__(
            self,
            dim: int,
            focal_window,
            focal_level: int,
            focal_factor: int = 2,
            bias: bool = True,
            use_post_norm: bool = False,
            normalize_modulator: bool = False,
            proj_drop: float = 0.,
            norm_layer: Callable = LayerNorm2d,
    ):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_post_norm = use_post_norm
        self.normalize_modulator = normalize_modulator
        self.input_split = [dim, dim, self.focal_level + 1]

        self.f = nn.Conv2d(dim, 2 * dim + (self.focal_level + 1), kernel_size=1, bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size // 2, bias=False),
                nn.GELU(),
            ))
            self.kernel_sizes.append(kernel_size)
        self.norm = norm_layer(dim) if self.use_post_norm else nn.Identity()

    def forward(self, x):
        # pre linear projection
        x = self.f(x)
        q, ctx, gates = torch.split(x, self.input_split, 1)

        # context aggreation
        ctx_all = 0
        for l, focal_layer in enumerate(self.focal_layers):
            ctx = focal_layer(ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean((2, 3), keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        x_out = q * self.h(ctx_all)
        x_out = self.norm(x_out)

        # post linear projection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


class LayerScale2d(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


class FocalNetBlock(nn.Module):
    """ Focal Modulation Network Block.
    """

    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 4.,
            focal_level: int = 1,
            focal_window: int = 3,
            use_post_norm: bool = False,
            use_post_norm_in_modulation: bool = False,
            normalize_modulator: bool = False,
            layerscale_value: float = 1e-4,
            proj_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm2d,
    ):
        """
        Args:
            dim: Number of input channels.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            focal_level: Number of focal levels.
            focal_window: Focal window size at first focal level.
            use_post_norm: Whether to use layer norm after modulation.
            use_post_norm_in_modulation: Whether to use layer norm in modulation.
            layerscale_value: Initial layerscale value.
            proj_drop: Dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_post_norm = use_post_norm

        self.norm1 = norm_layer(dim) if not use_post_norm else nn.Identity()
        self.modulation = FocalModulation(
            dim,
            focal_window=focal_window,
            focal_level=self.focal_level,
            use_post_norm=use_post_norm_in_modulation,
            normalize_modulator=normalize_modulator,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm1_post = norm_layer(dim) if use_post_norm else nn.Identity()
        self.ls1 = LayerScale2d(dim, layerscale_value) if layerscale_value is not None else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim) if not use_post_norm else nn.Identity()
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            use_conv=True,
        )
        self.norm2_post = norm_layer(dim) if use_post_norm else nn.Identity()
        self.ls2 = LayerScale2d(dim, layerscale_value) if layerscale_value is not None else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        # Focal Modulation
        x = self.norm1(x)
        x = self.modulation(x)
        x = self.norm1_post(x)
        x = shortcut + self.drop_path1(self.ls1(x))

        # FFN
        x = x + self.drop_path2(self.ls2(self.norm2_post(self.mlp(self.norm2(x)))))

        return x


class FocalNetStage(nn.Module):
    """ A basic Focal Transformer layer for one stage.
    """

    def __init__(
            self,
            dim: int,
            out_dim: int,
            depth: int,
            mlp_ratio: float = 4.,
            downsample: bool = True,
            focal_level: int = 1,
            focal_window: int = 1,
            use_overlap_down: bool = False,
            use_post_norm: bool = False,
            use_post_norm_in_modulation: bool = False,
            normalize_modulator: bool = False,
            layerscale_value: float = 1e-4,
            proj_drop: float = 0.,
            drop_path: float = 0.,
            norm_layer: Callable = LayerNorm2d,
    ):
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels.
            depth: Number of blocks.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            downsample: Downsample layer at start of the layer.
            focal_level: Number of focal levels
            focal_window: Focal window size at first focal level
            use_overlap_down: User overlapped convolution in downsample layer.
            use_post_norm: Whether to use layer norm after modulation.
            use_post_norm_in_modulation: Whether to use layer norm in modulation.
            layerscale_value: Initial layerscale value
            proj_drop: Dropout rate for projections.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.grad_checkpointing = False

        if downsample:
            self.downsample = Downsample(
                in_chs=dim,
                out_chs=out_dim,
                stride=2,
                overlap=use_overlap_down,
                norm_layer=norm_layer,
            )
        else:
            self.downsample = nn.Identity()

        # build blocks
        self.blocks = nn.ModuleList([
            FocalNetBlock(
                dim=out_dim,
                mlp_ratio=mlp_ratio,
                focal_level=focal_level,
                focal_window=focal_window,
                use_post_norm=use_post_norm,
                use_post_norm_in_modulation=use_post_norm_in_modulation,
                normalize_modulator=normalize_modulator,
                layerscale_value=layerscale_value,
                proj_drop=proj_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, x):
        x = self.downsample(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class Downsample(nn.Module):

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 4,
            overlap: bool = False,
            norm_layer: Optional[Callable] = None,
    ):
        """

        Args:
            in_chs: Number of input image channels.
            out_chs: Number of linear projection output channels.
            stride: Downsample stride.
            overlap: Use overlapping convolutions if True.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.stride = stride
        padding = 0
        kernel_size = stride
        if overlap:
            assert stride in (2, 4)
            if stride == 4:
                kernel_size, padding = 7, 2
            elif stride == 2:
                kernel_size, padding = 3, 1
        self.proj = nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm_layer(out_chs) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class FocalNet(nn.Module):
    """" Focal Modulation Networks (FocalNets)
    """

    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 96,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            mlp_ratio: float = 4.,
            focal_levels: Tuple[int, ...] = (2, 2, 2, 2),
            focal_windows: Tuple[int, ...] = (3, 3, 3, 3),
            use_overlap_down: bool = False,
            use_post_norm: bool = False,
            use_post_norm_in_modulation: bool = False,
            normalize_modulator: bool = False,
            head_hidden_size: Optional[int] = None,
            head_init_scale: float = 1.0,
            layerscale_value: Optional[float] = None,
            drop_rate: bool = 0.,
            proj_drop_rate: bool = 0.,
            drop_path_rate: bool = 0.1,
            norm_layer: Callable = partial(LayerNorm2d, eps=1e-5),
    ):
        """
        Args:
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            embed_dim: Patch embedding dimension.
            depths: Depth of each Focal Transformer layer.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            focal_levels: How many focal levels at all stages. Note that this excludes the finest-grain level.
            focal_windows: The focal window size at all stages.
            use_overlap_down: Whether to use convolutional embedding.
            use_post_norm: Whether to use layernorm after modulation (it helps stablize training of large models)
            layerscale_value: Value for layer scale.
            drop_rate: Dropout rate.
            drop_path_rate: Stochastic depth rate.
            norm_layer: Normalization layer.
        """
        super().__init__()

        self.num_layers = len(depths)
        embed_dim = [embed_dim * (2 ** i) for i in range(self.num_layers)]

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim[-1]
        self.feature_info = []

        self.stem = Downsample(
            in_chs=in_chans,
            out_chs=embed_dim[0],
            overlap=use_overlap_down,
            norm_layer=norm_layer,
        )
        in_dim = embed_dim[0]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        layers = []
        for i_layer in range(self.num_layers):
            out_dim = embed_dim[i_layer]
            layer = FocalNetStage(
                dim=in_dim,
                out_dim=out_dim,
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                downsample=i_layer > 0,
                focal_level=focal_levels[i_layer],
                focal_window=focal_windows[i_layer],
                use_overlap_down=use_overlap_down,
                use_post_norm=use_post_norm,
                use_post_norm_in_modulation=use_post_norm_in_modulation,
                normalize_modulator=normalize_modulator,
                layerscale_value=layerscale_value,
                proj_drop=proj_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
            )
            in_dim = out_dim
            layers += [layer]
            self.feature_info += [dict(num_chs=out_dim, reduction=4 * 2 ** i_layer, module=f'layers.{i_layer}')]

        self.layers = nn.Sequential(*layers)

        if head_hidden_size:
            self.norm = nn.Identity()
            self.head = NormMlpClassifierHead(
                self.num_features,
                num_classes,
                hidden_size=head_hidden_size,
                pool_type=global_pool,
                drop_rate=drop_rate,
                norm_layer=norm_layer,
            )
        else:
            self.norm = norm_layer(self.num_features)
            self.head = ClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=drop_rate
            )

        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {''}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=[
                (r'^layers\.(\d+)', None),
                (r'^norm', (99999,))
            ] if coarse else [
                (r'^layers\.(\d+).downsample', (0,)),
                (r'^layers\.(\d+)\.\w+\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
        for l in self.layers:
            l.set_grad_checkpointing(enable=enable)

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if name and 'head.fc' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.proj', 'classifier': 'head.fc',
        'license': 'mit', **kwargs
    }


default_cfgs = generate_default_cfgs({
    "focalnet_tiny_srf.ms_in1k": _cfg(
        hf_hub_id='timm/'),
    "focalnet_small_srf.ms_in1k": _cfg(
        hf_hub_id='timm/'),
    "focalnet_base_srf.ms_in1k": _cfg(
        hf_hub_id='timm/'),
    "focalnet_tiny_lrf.ms_in1k": _cfg(
        hf_hub_id='timm/'),
    "focalnet_small_lrf.ms_in1k": _cfg(
        hf_hub_id='timm/'),
    "focalnet_base_lrf.ms_in1k": _cfg(
        hf_hub_id='timm/'),

    "focalnet_large_fl3.ms_in22k": _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, num_classes=21842),
    "focalnet_large_fl4.ms_in22k": _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, num_classes=21842),
    "focalnet_xlarge_fl3.ms_in22k": _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, num_classes=21842),
    "focalnet_xlarge_fl4.ms_in22k": _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, num_classes=21842),
    "focalnet_huge_fl3.ms_in22k": _cfg(
        hf_hub_id='timm/',
        num_classes=21842),
    "focalnet_huge_fl4.ms_in22k": _cfg(
        hf_hub_id='timm/',
        num_classes=0),
})


def checkpoint_filter_fn(state_dict, model: FocalNet):
    state_dict = state_dict.get('model', state_dict)
    if 'stem.proj.weight' in state_dict:
        return state_dict
    import re
    out_dict = {}
    dest_dict = model.state_dict()
    for k, v in state_dict.items():
        k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        k = k.replace('patch_embed', 'stem')
        k = re.sub(r'layers.(\d+).downsample', lambda x: f'layers.{int(x.group(1)) + 1}.downsample', k)
        if 'norm' in k and k not in dest_dict:
            k = re.sub(r'norm([0-9])', r'norm\1_post', k)
        k = k.replace('ln.', 'norm.')
        k = k.replace('head', 'head.fc')
        if k in dest_dict and dest_dict[k].numel() == v.numel() and dest_dict[k].shape != v.shape:
            v = v.reshape(dest_dict[k].shape)
        out_dict[k] = v
    return out_dict


def _create_focalnet(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 1, 3, 1))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        FocalNet, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)
    return model


@register_model
def focalnet_tiny_srf(pretrained=False, **kwargs) -> FocalNet:
    model_kwargs = dict(depths=[2, 2, 6, 2], embed_dim=96, **kwargs)
    return _create_focalnet('focalnet_tiny_srf', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_small_srf(pretrained=False, **kwargs) -> FocalNet:
    model_kwargs = dict(depths=[2, 2, 18, 2], embed_dim=96, **kwargs)
    return _create_focalnet('focalnet_small_srf', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_base_srf(pretrained=False, **kwargs) -> FocalNet:
    model_kwargs = dict(depths=[2, 2, 18, 2], embed_dim=128, **kwargs)
    return _create_focalnet('focalnet_base_srf', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_tiny_lrf(pretrained=False, **kwargs) -> FocalNet:
    model_kwargs = dict(depths=[2, 2, 6, 2], embed_dim=96, focal_levels=[3, 3, 3, 3], **kwargs)
    return _create_focalnet('focalnet_tiny_lrf', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_small_lrf(pretrained=False, **kwargs) -> FocalNet:
    model_kwargs = dict(depths=[2, 2, 18, 2], embed_dim=96, focal_levels=[3, 3, 3, 3], **kwargs)
    return _create_focalnet('focalnet_small_lrf', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_base_lrf(pretrained=False, **kwargs) -> FocalNet:
    model_kwargs = dict(depths=[2, 2, 18, 2], embed_dim=128, focal_levels=[3, 3, 3, 3], **kwargs)
    return _create_focalnet('focalnet_base_lrf', pretrained=pretrained, **model_kwargs)


# FocalNet large+ models
@register_model
def focalnet_large_fl3(pretrained=False, **kwargs) -> FocalNet:
    model_kwargs = dict(
        depths=[2, 2, 18, 2], embed_dim=192, focal_levels=[3, 3, 3, 3], focal_windows=[5] * 4,
        use_post_norm=True, use_overlap_down=True, layerscale_value=1e-4, **kwargs)
    return _create_focalnet('focalnet_large_fl3', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_large_fl4(pretrained=False, **kwargs) -> FocalNet:
    model_kwargs = dict(
        depths=[2, 2, 18, 2], embed_dim=192, focal_levels=[4, 4, 4, 4],
        use_post_norm=True, use_overlap_down=True, layerscale_value=1e-4, **kwargs)
    return _create_focalnet('focalnet_large_fl4', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_xlarge_fl3(pretrained=False, **kwargs) -> FocalNet:
    model_kwargs = dict(
        depths=[2, 2, 18, 2], embed_dim=256, focal_levels=[3, 3, 3, 3], focal_windows=[5] * 4,
        use_post_norm=True, use_overlap_down=True, layerscale_value=1e-4, **kwargs)
    return _create_focalnet('focalnet_xlarge_fl3', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_xlarge_fl4(pretrained=False, **kwargs) -> FocalNet:
    model_kwargs = dict(
        depths=[2, 2, 18, 2], embed_dim=256, focal_levels=[4, 4, 4, 4],
        use_post_norm=True, use_overlap_down=True, layerscale_value=1e-4, **kwargs)
    return _create_focalnet('focalnet_xlarge_fl4', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_huge_fl3(pretrained=False, **kwargs) -> FocalNet:
    model_kwargs = dict(
        depths=[2, 2, 18, 2], embed_dim=352, focal_levels=[3, 3, 3, 3], focal_windows=[3] * 4,
        use_post_norm=True, use_post_norm_in_modulation=True, use_overlap_down=True, layerscale_value=1e-4, **kwargs)
    return _create_focalnet('focalnet_huge_fl3', pretrained=pretrained, **model_kwargs)


@register_model
def focalnet_huge_fl4(pretrained=False, **kwargs) -> FocalNet:
    model_kwargs = dict(
        depths=[2, 2, 18, 2], embed_dim=352, focal_levels=[4, 4, 4, 4],
        use_post_norm=True, use_post_norm_in_modulation=True, use_overlap_down=True, layerscale_value=1e-4, **kwargs)
    return _create_focalnet('focalnet_huge_fl4', pretrained=pretrained, **model_kwargs)

