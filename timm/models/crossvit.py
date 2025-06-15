""" CrossViT Model

@inproceedings{
    chen2021crossvit,
    title={{CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification}},
    author={Chun-Fu (Richard) Chen and Quanfu Fan and Rameswar Panda},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2021}
}

Paper link: https://arxiv.org/abs/2103.14899
Original code: https://github.com/IBM/CrossViT/blob/main/models/crossvit.py

NOTE: model names have been renamed from originals to represent actual input res all *_224 -> *_240 and *_384 -> *_408

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
Modified from Timm. https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, to_2tuple, trunc_normal_, _assert
from ._builder import build_model_with_cfg
from ._features_fx import register_notrace_function
from ._registry import register_model, generate_default_cfgs
from .vision_transformer import Block

__all__ = ['CrossVit']  # model_registry will add each entrypoint fn to this


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        _assert(H == self.img_size[0],
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        _assert(W == self.img_size[1],
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # B1C -> B1H(C/H) -> BH1(C/H)
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        return x


class MultiScaleBlock(nn.Module):

    def __init__(
            self,
            dim,
            patches,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(Block(
                    dim=dim[d],
                    num_heads=num_heads[d],
                    mlp_ratio=mlp_ratio[d],
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                ))
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d + 1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[d]), act_layer(), nn.Linear(dim[d], dim[(d + 1) % num_branches])]
            self.projs.append(nn.Sequential(*tmp))

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d + 1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(
                    CrossAttentionBlock(
                        dim=dim[d_],
                        num_heads=nh,
                        mlp_ratio=mlp_ratio[d],
                        qkv_bias=qkv_bias,
                        proj_drop=proj_drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[-1],
                        norm_layer=norm_layer,
                    ))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(
                        dim=dim[d_],
                        num_heads=nh,
                        mlp_ratio=mlp_ratio[d],
                        qkv_bias=qkv_bias,
                        proj_drop=proj_drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[-1],
                        norm_layer=norm_layer,
                    ))
                self.fusion.append(nn.Sequential(*tmp))

        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d + 1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[(d + 1) % num_branches]), act_layer(),
                       nn.Linear(dim[(d + 1) % num_branches], dim[d])]
            self.revert_projs.append(nn.Sequential(*tmp))

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:

        outs_b = []
        for i, block in enumerate(self.blocks):
            outs_b.append(block(x[i]))

        # only take the cls token out
        proj_cls_token = torch.jit.annotate(List[torch.Tensor], [])
        for i, proj in enumerate(self.projs):
            proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))

        # cross attention
        outs = []
        for i, (fusion, revert_proj) in enumerate(zip(self.fusion, self.revert_projs)):
            tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = fusion(tmp)
            reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
        return outs


def _compute_num_patches(img_size, patches):
    return [i[0] // p * i[1] // p for i, p in zip(img_size, patches)]


@register_notrace_function
def scale_image(x, ss: Tuple[int, int], crop_scale: bool = False):  # annotations for torchscript
    """
    Pulled out of CrossViT.forward_features to bury conditional logic in a leaf node for FX tracing.
    Args:
        x (Tensor): input image
        ss (tuple[int, int]): height and width to scale to
        crop_scale (bool): whether to crop instead of interpolate to achieve the desired scale. Defaults to False
    Returns:
        Tensor: the "scaled" image batch tensor
    """
    H, W = x.shape[-2:]
    if H != ss[0] or W != ss[1]:
        if crop_scale and ss[0] <= H and ss[1] <= W:
            cu, cl = int(round((H - ss[0]) / 2.)), int(round((W - ss[1]) / 2.))
            x = x[:, :, cu:cu + ss[0], cl:cl + ss[1]]
        else:
            x = torch.nn.functional.interpolate(x, size=ss, mode='bicubic', align_corners=False)
    return x


class CrossVit(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(
            self,
            img_size=224,
            img_scale=(1.0, 1.0),
            patch_size=(8, 16),
            in_chans=3,
            num_classes=1000,
            embed_dim=(192, 384),
            depth=((1, 3, 1), (1, 3, 1), (1, 3, 1)),
            num_heads=(6, 12),
            mlp_ratio=(2., 2., 4.),
            multi_conv=False,
            crop_scale=False,
            qkv_bias=True,
            drop_rate=0.,
            pos_drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            global_pool='token',
    ):
        super().__init__()
        assert global_pool in ('token', 'avg')

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.img_size = to_2tuple(img_size)
        img_scale = to_2tuple(img_scale)
        self.img_size_scaled = [tuple([int(sj * si) for sj in self.img_size]) for si in img_scale]
        self.crop_scale = crop_scale  # crop instead of interpolate for scale
        num_patches = _compute_num_patches(self.img_size_scaled, patch_size)
        self.num_branches = len(patch_size)
        self.embed_dim = embed_dim
        self.num_features = self.head_hidden_size = sum(embed_dim)
        self.patch_embed = nn.ModuleList()

        # hard-coded for torch jit script
        for i in range(self.num_branches):
            setattr(self, f'pos_embed_{i}', nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])))
            setattr(self, f'cls_token_{i}', nn.Parameter(torch.zeros(1, 1, embed_dim[i])))

        for im_s, p, d in zip(self.img_size_scaled, patch_size, embed_dim):
            self.patch_embed.append(
                PatchEmbed(
                    img_size=im_s,
                    patch_size=p,
                    in_chans=in_chans,
                    embed_dim=d,
                    multi_conv=multi_conv,
                ))

        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(
                embed_dim,
                num_patches,
                block_cfg,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_,
                norm_layer=norm_layer,
            )
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.ModuleList([
            nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity()
            for i in range(self.num_branches)])

        for i in range(self.num_branches):
            trunc_normal_(getattr(self, f'pos_embed_{i}'), std=.02)
            trunc_normal_(getattr(self, f'cls_token_{i}'), std=.02)

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
        out = set()
        for i in range(self.num_branches):
            out.add(f'cls_token_{i}')
            pe = getattr(self, f'pos_embed_{i}', None)
            if pe is not None and pe.requires_grad:
                out.add(f'pos_embed_{i}')
        return out

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('token', 'avg')
            self.global_pool = global_pool
        self.head = nn.ModuleList([
            nn.Linear(self.embed_dim[i], num_classes) if num_classes > 0 else nn.Identity()
            for i in range(self.num_branches)
        ])

    def forward_features(self, x) -> List[torch.Tensor]:
        B = x.shape[0]
        xs = []
        for i, patch_embed in enumerate(self.patch_embed):
            x_ = x
            ss = self.img_size_scaled[i]
            x_ = scale_image(x_, ss, self.crop_scale)
            x_ = patch_embed(x_)
            cls_tokens = self.cls_token_0 if i == 0 else self.cls_token_1  # hard-coded for torch jit script
            cls_tokens = cls_tokens.expand(B, -1, -1)
            x_ = torch.cat((cls_tokens, x_), dim=1)
            pos_embed = self.pos_embed_0 if i == 0 else self.pos_embed_1  # hard-coded for torch jit script
            x_ = x_ + pos_embed
            x_ = self.pos_drop(x_)
            xs.append(x_)

        for i, blk in enumerate(self.blocks):
            xs = blk(xs)

        # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
        xs = [norm(xs[i]) for i, norm in enumerate(self.norm)]
        return xs

    def forward_head(self, xs: List[torch.Tensor], pre_logits: bool = False) -> torch.Tensor:
        xs = [x[:, 1:].mean(dim=1) for x in xs] if self.global_pool == 'avg' else [x[:, 0] for x in xs]
        xs = [self.head_drop(x) for x in xs]
        if pre_logits or isinstance(self.head[0], nn.Identity):
            return torch.cat([x for x in xs], dim=1)
        return torch.mean(torch.stack([head(xs[i]) for i, head in enumerate(self.head)], dim=0), dim=0)

    def forward(self, x):
        xs = self.forward_features(x)
        x = self.forward_head(xs)
        return x


def _create_crossvit(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    def pretrained_filter_fn(state_dict):
        new_state_dict = {}
        for key in state_dict.keys():
            if 'pos_embed' in key or 'cls_token' in key:
                new_key = key.replace(".", "_")
            else:
                new_key = key
            new_state_dict[new_key] = state_dict[key]
        return new_state_dict

    return build_model_with_cfg(
        CrossVit,
        variant,
        pretrained,
        pretrained_filter_fn=pretrained_filter_fn,
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 240, 240), 'pool_size': None, 'crop_pct': 0.875,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'fixed_input_size': True,
        'first_conv': ('patch_embed.0.proj', 'patch_embed.1.proj'),
        'classifier': ('head.0', 'head.1'),
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'crossvit_15_240.in1k': _cfg(hf_hub_id='timm/'),
    'crossvit_15_dagger_240.in1k': _cfg(
        hf_hub_id='timm/',
        first_conv=('patch_embed.0.proj.0', 'patch_embed.1.proj.0'),
    ),
    'crossvit_15_dagger_408.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 408, 408), first_conv=('patch_embed.0.proj.0', 'patch_embed.1.proj.0'), crop_pct=1.0,
    ),
    'crossvit_18_240.in1k': _cfg(hf_hub_id='timm/'),
    'crossvit_18_dagger_240.in1k': _cfg(
        hf_hub_id='timm/',
        first_conv=('patch_embed.0.proj.0', 'patch_embed.1.proj.0'),
    ),
    'crossvit_18_dagger_408.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 408, 408), first_conv=('patch_embed.0.proj.0', 'patch_embed.1.proj.0'), crop_pct=1.0,
    ),
    'crossvit_9_240.in1k': _cfg(hf_hub_id='timm/'),
    'crossvit_9_dagger_240.in1k': _cfg(
        hf_hub_id='timm/',
        first_conv=('patch_embed.0.proj.0', 'patch_embed.1.proj.0'),
    ),
    'crossvit_base_240.in1k': _cfg(hf_hub_id='timm/'),
    'crossvit_small_240.in1k': _cfg(hf_hub_id='timm/'),
    'crossvit_tiny_240.in1k': _cfg(hf_hub_id='timm/'),
})


@register_model
def crossvit_tiny_240(pretrained=False, **kwargs) -> CrossVit:
    model_args = dict(
        img_scale=(1.0, 224/240), patch_size=[12, 16], embed_dim=[96, 192], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
        num_heads=[3, 3], mlp_ratio=[4, 4, 1])
    model = _create_crossvit(variant='crossvit_tiny_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def crossvit_small_240(pretrained=False, **kwargs) -> CrossVit:
    model_args = dict(
        img_scale=(1.0, 224/240), patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
        num_heads=[6, 6], mlp_ratio=[4, 4, 1])
    model = _create_crossvit(variant='crossvit_small_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def crossvit_base_240(pretrained=False, **kwargs) -> CrossVit:
    model_args = dict(
        img_scale=(1.0, 224/240), patch_size=[12, 16], embed_dim=[384, 768], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
        num_heads=[12, 12], mlp_ratio=[4, 4, 1])
    model = _create_crossvit(variant='crossvit_base_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def crossvit_9_240(pretrained=False, **kwargs) -> CrossVit:
    model_args = dict(
        img_scale=(1.0, 224/240), patch_size=[12, 16], embed_dim=[128, 256], depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
        num_heads=[4, 4], mlp_ratio=[3, 3, 1])
    model = _create_crossvit(variant='crossvit_9_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def crossvit_15_240(pretrained=False, **kwargs) -> CrossVit:
    model_args = dict(
        img_scale=(1.0, 224/240), patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
        num_heads=[6, 6], mlp_ratio=[3, 3, 1])
    model = _create_crossvit(variant='crossvit_15_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def crossvit_18_240(pretrained=False, **kwargs) -> CrossVit:
    model_args = dict(
        img_scale=(1.0, 224 / 240), patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
        num_heads=[7, 7], mlp_ratio=[3, 3, 1], **kwargs)
    model = _create_crossvit(variant='crossvit_18_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def crossvit_9_dagger_240(pretrained=False, **kwargs) -> CrossVit:
    model_args = dict(
        img_scale=(1.0, 224 / 240), patch_size=[12, 16], embed_dim=[128, 256], depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
        num_heads=[4, 4], mlp_ratio=[3, 3, 1], multi_conv=True)
    model = _create_crossvit(variant='crossvit_9_dagger_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def crossvit_15_dagger_240(pretrained=False, **kwargs) -> CrossVit:
    model_args = dict(
        img_scale=(1.0, 224/240), patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
        num_heads=[6, 6], mlp_ratio=[3, 3, 1], multi_conv=True)
    model = _create_crossvit(variant='crossvit_15_dagger_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def crossvit_15_dagger_408(pretrained=False, **kwargs) -> CrossVit:
    model_args = dict(
        img_scale=(1.0, 384/408), patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
        num_heads=[6, 6], mlp_ratio=[3, 3, 1], multi_conv=True)
    model = _create_crossvit(variant='crossvit_15_dagger_408', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def crossvit_18_dagger_240(pretrained=False, **kwargs) -> CrossVit:
    model_args = dict(
        img_scale=(1.0, 224/240), patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
        num_heads=[7, 7], mlp_ratio=[3, 3, 1], multi_conv=True)
    model = _create_crossvit(variant='crossvit_18_dagger_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def crossvit_18_dagger_408(pretrained=False, **kwargs) -> CrossVit:
    model_args = dict(
        img_scale=(1.0, 384/408), patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
        num_heads=[7, 7], mlp_ratio=[3, 3, 1], multi_conv=True)
    model = _create_crossvit(variant='crossvit_18_dagger_408', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
