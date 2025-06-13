""" EdgeNeXt

Paper: `EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications`
 - https://arxiv.org/abs/2206.10589

Original code and weights from https://github.com/mmaaz60/EdgeNeXt

Modifications and additions for timm by / Copyright 2022, Ross Wightman
"""
import math
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import trunc_normal_tf_, DropPath, LayerNorm2d, Mlp, create_conv2d, \
    NormMlpClassifierHead, ClassifierHead
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._features_fx import register_notrace_module
from ._manipulate import named_apply, checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['EdgeNeXt']  # model_registry will add each entrypoint fn to this


@register_notrace_module  # reason: FX can't symbolically trace torch.arange in forward method
class PositionalEncodingFourier(nn.Module):
    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, shape: Tuple[int, int, int]):
        device = self.token_projection.weight.device
        dtype = self.token_projection.weight.dtype
        inv_mask = ~torch.zeros(shape).to(device=device, dtype=torch.bool)
        y_embed = inv_mask.cumsum(1, dtype=torch.float32)
        x_embed = inv_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.int64, device=device).to(torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(),
             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(),
             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos.to(dtype))

        return pos


class ConvBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            kernel_size=7,
            stride=1,
            conv_bias=True,
            expand_ratio=4,
            ls_init_value=1e-6,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU, drop_path=0.,
    ):
        super().__init__()
        dim_out = dim_out or dim
        self.shortcut_after_dw = stride > 1 or dim != dim_out

        self.conv_dw = create_conv2d(
            dim, dim_out, kernel_size=kernel_size, stride=stride, depthwise=True, bias=conv_bias)
        self.norm = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(expand_ratio * dim_out), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim_out)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.shortcut_after_dw:
            shortcut = x

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = shortcut + self.drop_path(x)
        return x


class CrossCovarianceAttn(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 4, 1)
        q, k, v = qkv.unbind(0)

        # NOTE, this is NOT spatial attn, q, k, v are B, num_heads, C, L -->  C x C attn map
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)

        x = x.permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class SplitTransposeBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_scales=1,
            num_heads=8,
            expand_ratio=4,
            use_pos_emb=True,
            conv_bias=True,
            qkv_bias=True,
            ls_init_value=1e-6,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop_path=0.,
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        width = max(int(math.ceil(dim / num_scales)), int(math.floor(dim // num_scales)))
        self.width = width
        self.num_scales = max(1, num_scales - 1)

        convs = []
        for i in range(self.num_scales):
            convs.append(create_conv2d(width, width, kernel_size=3, depthwise=True, bias=conv_bias))
        self.convs = nn.ModuleList(convs)

        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=dim)
        self.norm_xca = norm_layer(dim)
        self.gamma_xca = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.xca = CrossCovarianceAttn(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

        self.norm = norm_layer(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(expand_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        # scales code re-written for torchscript as per my res2net fixes -rw
        # NOTE torch.split(x, self.width, 1) causing issues with ONNX export
        spx = x.chunk(len(self.convs) + 1, dim=1)
        spo = []
        sp = spx[0]
        for i, conv in enumerate(self.convs):
            if i > 0:
                sp = sp + spx[i]
            sp = conv(sp)
            spo.append(sp)
        spo.append(spx[-1])
        x = torch.cat(spo, 1)

        # XCA
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.pos_embd is not None:
            pos_encoding = self.pos_embd((B, H, W)).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = x + self.drop_path(self.gamma_xca * self.xca(self.norm_xca(x)))
        x = x.reshape(B, H, W, C)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = shortcut + self.drop_path(x)
        return x


class EdgeNeXtStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            stride=2,
            depth=2,
            num_global_blocks=1,
            num_heads=4,
            scales=2,
            kernel_size=7,
            expand_ratio=4,
            use_pos_emb=False,
            downsample_block=False,
            conv_bias=True,
            ls_init_value=1.0,
            drop_path_rates=None,
            norm_layer=LayerNorm2d,
            norm_layer_cl=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU
    ):
        super().__init__()
        self.grad_checkpointing = False

        if downsample_block or stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=2, stride=2, bias=conv_bias)
            )
            in_chs = out_chs

        stage_blocks = []
        for i in range(depth):
            if i < depth - num_global_blocks:
                stage_blocks.append(
                    ConvBlock(
                        dim=in_chs,
                        dim_out=out_chs,
                        stride=stride if downsample_block and i == 0 else 1,
                        conv_bias=conv_bias,
                        kernel_size=kernel_size,
                        expand_ratio=expand_ratio,
                        ls_init_value=ls_init_value,
                        drop_path=drop_path_rates[i],
                        norm_layer=norm_layer_cl,
                        act_layer=act_layer,
                    )
                )
            else:
                stage_blocks.append(
                    SplitTransposeBlock(
                        dim=in_chs,
                        num_scales=scales,
                        num_heads=num_heads,
                        expand_ratio=expand_ratio,
                        use_pos_emb=use_pos_emb,
                        conv_bias=conv_bias,
                        ls_init_value=ls_init_value,
                        drop_path=drop_path_rates[i],
                        norm_layer=norm_layer_cl,
                        act_layer=act_layer,
                    )
                )
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class EdgeNeXt(nn.Module):
    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            dims=(24, 48, 88, 168),
            depths=(3, 3, 9, 3),
            global_block_counts=(0, 1, 1, 1),
            kernel_sizes=(3, 5, 7, 9),
            heads=(8, 8, 8, 8),
            d2_scales=(2, 2, 3, 4),
            use_pos_emb=(False, True, False, False),
            ls_init_value=1e-6,
            head_init_scale=1.,
            expand_ratio=4,
            downsample_block=False,
            conv_bias=True,
            stem_type='patch',
            head_norm_first=False,
            act_layer=nn.GELU,
            drop_path_rate=0.,
            drop_rate=0.,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.drop_rate = drop_rate
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        norm_layer_cl = partial(nn.LayerNorm, eps=1e-6)
        self.feature_info = []

        assert stem_type in ('patch', 'overlap')
        if stem_type == 'patch':
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, bias=conv_bias),
                norm_layer(dims[0]),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=9, stride=4, padding=9 // 2, bias=conv_bias),
                norm_layer(dims[0]),
            )

        curr_stride = 4
        stages = []
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        in_chs = dims[0]
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            # FIXME support dilation / output_stride
            curr_stride *= stride
            stages.append(EdgeNeXtStage(
                in_chs=in_chs,
                out_chs=dims[i],
                stride=stride,
                depth=depths[i],
                num_global_blocks=global_block_counts[i],
                num_heads=heads[i],
                drop_path_rates=dp_rates[i],
                scales=d2_scales[i],
                expand_ratio=expand_ratio,
                kernel_size=kernel_sizes[i],
                use_pos_emb=use_pos_emb[i],
                ls_init_value=ls_init_value,
                downsample_block=downsample_block,
                conv_bias=conv_bias,
                norm_layer=norm_layer,
                norm_layer_cl=norm_layer_cl,
                act_layer=act_layer,
            ))
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            in_chs = dims[i]
            self.feature_info += [dict(num_chs=in_chs, reduction=curr_stride, module=f'stages.{i}')]

        self.stages = nn.Sequential(*stages)

        self.num_features = self.head_hidden_size = dims[-1]
        if head_norm_first:
            self.norm_pre = norm_layer(self.num_features)
            self.head = ClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
            )
        else:
            self.norm_pre = nn.Identity()
            self.head = NormMlpClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
                norm_layer=norm_layer,
            )

        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),  # blocks
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm_pre', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

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
        x = self.stem(x)
        last_idx = len(self.stages) - 1
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]

        for feat_idx, stage in enumerate(stages):
            x = stage(x)
            if feat_idx in take_indices:
                if norm and feat_idx == last_idx:
                    x_inter = self.norm_pre(x)  # applying final norm to last intermediate
                else:
                    x_inter = x
                intermediates.append(x_inter)

        if intermediates_only:
            return intermediates

        if feat_idx == last_idx:
            x = self.norm_pre(x)

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
        if prune_norm:
            self.norm_pre = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_tf_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_tf_(module.weight, std=.02)
        nn.init.zeros_(module.bias)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def checkpoint_filter_fn(state_dict, model):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint

    # models were released as train checkpoints... :/
    if 'model_ema' in state_dict:
        state_dict = state_dict['model_ema']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    out_dict = {}
    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')
        if v.ndim == 2 and 'head' not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v
    return out_dict


def _create_edgenext(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        EdgeNeXt, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs)
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 256, 256), 'pool_size': (8, 8),
        'crop_pct': 0.9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'edgenext_xx_small.in1k': _cfg(
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'edgenext_x_small.in1k': _cfg(
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'edgenext_small.usi_in1k': _cfg(  # USI weights
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 320, 320), test_crop_pct=1.0,
    ),
    'edgenext_base.usi_in1k': _cfg(  # USI weights
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 320, 320), test_crop_pct=1.0,
    ),
    'edgenext_base.in21k_ft_in1k': _cfg(  # USI weights
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 320, 320), test_crop_pct=1.0,
    ),
    'edgenext_small_rw.sw_in1k': _cfg(
        hf_hub_id='timm/',
        test_input_size=(3, 320, 320), test_crop_pct=1.0,
    ),
})


@register_model
def edgenext_xx_small(pretrained=False, **kwargs) -> EdgeNeXt:
    # 1.33M & 260.58M @ 256 resolution
    # 71.23% Top-1 accuracy
    # No AA, Color Jitter=0.4, No Mixup & Cutmix, DropPath=0.0, BS=4096, lr=0.006, multi-scale-sampler
    # Jetson FPS=51.66 versus 47.67 for MobileViT_XXS
    # For A100: FPS @ BS=1: 212.13 & @ BS=256: 7042.06 versus FPS @ BS=1: 96.68 & @ BS=256: 4624.71 for MobileViT_XXS
    model_args = dict(depths=(2, 2, 6, 2), dims=(24, 48, 88, 168), heads=(4, 4, 4, 4))
    return _create_edgenext('edgenext_xx_small', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def edgenext_x_small(pretrained=False, **kwargs) -> EdgeNeXt:
    # 2.34M & 538.0M @ 256 resolution
    # 75.00% Top-1 accuracy
    # No AA, No Mixup & Cutmix, DropPath=0.0, BS=4096, lr=0.006, multi-scale-sampler
    # Jetson FPS=31.61 versus 28.49 for MobileViT_XS
    # For A100: FPS @ BS=1: 179.55 & @ BS=256: 4404.95 versus FPS @ BS=1: 94.55 & @ BS=256: 2361.53 for MobileViT_XS
    model_args = dict(depths=(3, 3, 9, 3), dims=(32, 64, 100, 192), heads=(4, 4, 4, 4))
    return _create_edgenext('edgenext_x_small', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def edgenext_small(pretrained=False, **kwargs) -> EdgeNeXt:
    # 5.59M & 1260.59M @ 256 resolution
    # 79.43% Top-1 accuracy
    # AA=True, No Mixup & Cutmix, DropPath=0.1, BS=4096, lr=0.006, multi-scale-sampler
    # Jetson FPS=20.47 versus 18.86 for MobileViT_S
    # For A100: FPS @ BS=1: 172.33 & @ BS=256: 3010.25 versus FPS @ BS=1: 93.84 & @ BS=256: 1785.92 for MobileViT_S
    model_args = dict(depths=(3, 3, 9, 3), dims=(48, 96, 160, 304))
    return _create_edgenext('edgenext_small', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def edgenext_base(pretrained=False, **kwargs) -> EdgeNeXt:
    # 18.51M & 3840.93M @ 256 resolution
    # 82.5% (normal) 83.7% (USI) Top-1 accuracy
    # AA=True, Mixup & Cutmix, DropPath=0.1, BS=4096, lr=0.006, multi-scale-sampler
    # Jetson FPS=xx.xx versus xx.xx for MobileViT_S
    # For A100: FPS @ BS=1: xxx.xx & @ BS=256: xxxx.xx
    model_args = dict(depths=[3, 3, 9, 3], dims=[80, 160, 288, 584])
    return _create_edgenext('edgenext_base', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def edgenext_small_rw(pretrained=False, **kwargs) -> EdgeNeXt:
    model_args = dict(
        depths=(3, 3, 9, 3), dims=(48, 96, 192, 384),
        downsample_block=True, conv_bias=False, stem_type='overlap')
    return _create_edgenext('edgenext_small_rw', pretrained=pretrained, **dict(model_args, **kwargs))

