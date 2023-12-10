""" Vision OutLOoker (VOLO) implementation

Paper: `VOLO: Vision Outlooker for Visual Recognition` - https://arxiv.org/abs/2106.13112

Code adapted from official impl at https://github.com/sail-sg/volo, original copyright in comment below

Modifications and additions for timm by / Copyright 2022, Ross Wightman
"""
# Copyright 2021 Sea Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, Mlp, to_2tuple, to_ntuple, trunc_normal_
from ._builder import build_model_with_cfg
from ._registry import register_model, generate_default_cfgs

__all__ = ['VOLO']  # model_registry will add each entrypoint fn to this


class OutlookAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            kernel_size=3,
            padding=1,
            stride=1,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = head_dim ** -0.5

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W

        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unfold(v).reshape(
            B, self.num_heads, C // self.num_heads,
            self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H

        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
            self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)

        return x


class Outlooker(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size,
            padding,
            stride=1,
            num_heads=1,
            mlp_ratio=3.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            qkv_bias=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = OutlookAttention(
            dim,
            num_heads,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):

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
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Transformer(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ClassAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            head_dim=None,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, self.head_dim * self.num_heads * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_drop(cls_embed)
        return cls_embed


class ClassBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            head_dim=None,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ClassAttention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
        cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
        return torch.cat([cls_embed, x[:, 1:]], dim=1)


def get_block(block_type, **kargs):
    if block_type == 'ca':
        return ClassBlock(**kargs)


def rand_bbox(size, lam, scale=1):
    """
    get bounding box as token labeling (https://github.com/zihangJiang/TokenLabeling)
    return: bounding box
    """
    W = size[1] // scale
    H = size[2] // scale
    cut_rat = np.sqrt(1. - lam)
    cut_w = (W * cut_rat).astype(int)
    cut_h = (H * cut_rat).astype(int)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding.
    Different with ViT use 1 conv layer, we use 4 conv layers to do patch embedding
    """

    def __init__(
            self,
            img_size=224,
            stem_conv=False,
            stem_stride=1,
            patch_size=8,
            in_chans=3,
            hidden_dim=64,
            embed_dim=384,
    ):
        super().__init__()
        assert patch_size in [4, 8, 16]
        if stem_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride, padding=3, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = None

        self.proj = nn.Conv2d(
            hidden_dim, embed_dim, kernel_size=patch_size // stem_stride, stride=patch_size // stem_stride)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.proj(x)  # B, C, H, W
        return x


class Downsample(nn.Module):
    """ Image to Patch Embedding, downsampling between stage1 and stage2
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size=2):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x


def outlooker_blocks(
        block_fn,
        index,
        dim,
        layers,
        num_heads=1,
        kernel_size=3,
        padding=1,
        stride=2,
        mlp_ratio=3.,
        qkv_bias=False,
        attn_drop=0,
        drop_path_rate=0.,
        **kwargs,
):
    """
    generate outlooker layer in stage1
    return: outlooker layers
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(block_fn(
            dim,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            drop_path=block_dpr,
        ))
    blocks = nn.Sequential(*blocks)
    return blocks


def transformer_blocks(
        block_fn,
        index,
        dim,
        layers,
        num_heads,
        mlp_ratio=3.,
        qkv_bias=False,
        attn_drop=0,
        drop_path_rate=0.,
        **kwargs,
):
    """
    generate transformer layers in stage2
    return: transformer layers
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(block_fn(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            drop_path=block_dpr,
        ))
    blocks = nn.Sequential(*blocks)
    return blocks


class VOLO(nn.Module):
    """
    Vision Outlooker, the main class of our model
    """

    def __init__(
            self,
            layers,
            img_size=224,
            in_chans=3,
            num_classes=1000,
            global_pool='token',
            patch_size=8,
            stem_hidden_dim=64,
            embed_dims=None,
            num_heads=None,
            downsamples=(True, False, False, False),
            outlook_attention=(True, False, False, False),
            mlp_ratio=3.0,
            qkv_bias=False,
            drop_rate=0.,
            pos_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            post_layers=('ca', 'ca'),
            use_aux_head=True,
            use_mix_token=False,
            pooling_scale=2,
    ):
        super().__init__()
        num_layers = len(layers)
        mlp_ratio = to_ntuple(num_layers)(mlp_ratio)
        img_size = to_2tuple(img_size)

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.mix_token = use_mix_token
        self.pooling_scale = pooling_scale
        self.num_features = embed_dims[-1]
        if use_mix_token:  # enable token mixing, see token labeling for details.
            self.beta = 1.0
            assert global_pool == 'token', "return all tokens if mix_token is enabled"
        self.grad_checkpointing = False

        self.patch_embed = PatchEmbed(
            stem_conv=True,
            stem_stride=2,
            patch_size=patch_size,
            in_chans=in_chans,
            hidden_dim=stem_hidden_dim,
            embed_dim=embed_dims[0],
        )

        # inital positional encoding, we add positional encoding after outlooker blocks
        patch_grid = (img_size[0] // patch_size // pooling_scale, img_size[1] // patch_size // pooling_scale)
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_grid[0], patch_grid[1], embed_dims[-1]))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        # set the main block in network
        network = []
        for i in range(len(layers)):
            if outlook_attention[i]:
                # stage 1
                stage = outlooker_blocks(
                    Outlooker,
                    i,
                    embed_dims[i],
                    layers,
                    num_heads[i],
                    mlp_ratio=mlp_ratio[i],
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                network.append(stage)
            else:
                # stage 2
                stage = transformer_blocks(
                    Transformer,
                    i,
                    embed_dims[i],
                    layers,
                    num_heads[i],
                    mlp_ratio=mlp_ratio[i],
                    qkv_bias=qkv_bias,
                    drop_path_rate=drop_path_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                network.append(stage)

            if downsamples[i]:
                # downsampling between two stages
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], 2))

        self.network = nn.ModuleList(network)

        # set post block, for example, class attention layers
        self.post_network = None
        if post_layers is not None:
            self.post_network = nn.ModuleList([
                get_block(
                    post_layers[i],
                    dim=embed_dims[-1],
                    num_heads=num_heads[-1],
                    mlp_ratio=mlp_ratio[-1],
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    drop_path=0.,
                    norm_layer=norm_layer)
                for i in range(len(post_layers))
            ])
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))
            trunc_normal_(self.cls_token, std=.02)

        # set output type
        if use_aux_head:
            self.aux_head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.aux_head = None
        self.norm = norm_layer(self.num_features)

        # Classifier head
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[
                (r'^network\.(\d+)\.(\d+)', None),
                (r'^network\.(\d+)', (0,)),
            ],
            blocks2=[
                (r'^cls_token', (0,)),
                (r'^post_network\.(\d+)', None),
                (r'^norm', (99999,))
            ],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        if self.aux_head is not None:
            self.aux_head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            if idx == 2:
                # add positional encoding after outlooker blocks
                x = x + self.pos_embed
                x = self.pos_drop(x)
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x)
            else:
                x = block(x)

        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward_cls(self, x):
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        for block in self.post_network:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x)
            else:
                x = block(x)
        return x

    def forward_train(self, x):
        """ A separate forward fn for training with mix_token (if a train script supports).
        Combining multiple modes in as single forward with different return types is torchscript hell.
        """
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)  # B,C,H,W-> B,H,W,C

        # mix token, see token labeling for details.
        if self.mix_token and self.training:
            lam = np.random.beta(self.beta, self.beta)
            patch_h, patch_w = x.shape[1] // self.pooling_scale, x.shape[2] // self.pooling_scale
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam, scale=self.pooling_scale)
            temp_x = x.clone()
            sbbx1, sbby1 = self.pooling_scale * bbx1, self.pooling_scale * bby1
            sbbx2, sbby2 = self.pooling_scale * bbx2, self.pooling_scale * bby2
            temp_x[:, sbbx1:sbbx2, sbby1:sbby2, :] = x.flip(0)[:, sbbx1:sbbx2, sbby1:sbby2, :]
            x = temp_x
        else:
            bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0

        # step2: tokens learning in the two stages
        x = self.forward_tokens(x)

        # step3: post network, apply class attention or not
        if self.post_network is not None:
            x = self.forward_cls(x)
        x = self.norm(x)

        if self.global_pool == 'avg':
            x_cls = x.mean(dim=1)
        elif self.global_pool == 'token':
            x_cls = x[:, 0]
        else:
            x_cls = x

        if self.aux_head is None:
            return x_cls

        x_aux = self.aux_head(x[:, 1:])  # generate classes in all feature tokens, see token labeling
        if not self.training:
            return x_cls + 0.5 * x_aux.max(1)[0]

        if self.mix_token and self.training:  # reverse "mix token", see token labeling for details.
            x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])
            temp_x = x_aux.clone()
            temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
            x_aux = temp_x
            x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])

        # return these: 1. class token, 2. classes from all feature tokens, 3. bounding box
        return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)

    def forward_features(self, x):
        x = self.patch_embed(x).permute(0, 2, 3, 1)  # B,C,H,W-> B,H,W,C

        # step2: tokens learning in the two stages
        x = self.forward_tokens(x)

        # step3: post network, apply class attention or not
        if self.post_network is not None:
            x = self.forward_cls(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            out = x.mean(dim=1)
        elif self.global_pool == 'token':
            out = x[:, 0]
        else:
            out = x
        x = self.head_drop(x)
        if pre_logits:
            return out
        out = self.head(out)
        if self.aux_head is not None:
            # generate classes in all feature tokens, see token labeling
            aux = self.aux_head(x[:, 1:])
            out = out + 0.5 * aux.max(1)[0]
        return out

    def forward(self, x):
        """ simplified forward (without mix token training) """
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_volo(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    return build_model_with_cfg(
        VOLO,
        variant,
        pretrained,
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.conv.0', 'classifier': ('head', 'aux_head'),
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'volo_d1_224.sail_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/sail-sg/volo/releases/download/volo_1/d1_224_84.2.pth.tar',
        crop_pct=0.96),
    'volo_d1_384.sail_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/sail-sg/volo/releases/download/volo_1/d1_384_85.2.pth.tar',
        crop_pct=1.0, input_size=(3, 384, 384)),
    'volo_d2_224.sail_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/sail-sg/volo/releases/download/volo_1/d2_224_85.2.pth.tar',
        crop_pct=0.96),
    'volo_d2_384.sail_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/sail-sg/volo/releases/download/volo_1/d2_384_86.0.pth.tar',
        crop_pct=1.0, input_size=(3, 384, 384)),
    'volo_d3_224.sail_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/sail-sg/volo/releases/download/volo_1/d3_224_85.4.pth.tar',
        crop_pct=0.96),
    'volo_d3_448.sail_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/sail-sg/volo/releases/download/volo_1/d3_448_86.3.pth.tar',
        crop_pct=1.0, input_size=(3, 448, 448)),
    'volo_d4_224.sail_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/sail-sg/volo/releases/download/volo_1/d4_224_85.7.pth.tar',
        crop_pct=0.96),
    'volo_d4_448.sail_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/sail-sg/volo/releases/download/volo_1/d4_448_86.79.pth.tar',
        crop_pct=1.15, input_size=(3, 448, 448)),
    'volo_d5_224.sail_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/sail-sg/volo/releases/download/volo_1/d5_224_86.10.pth.tar',
        crop_pct=0.96),
    'volo_d5_448.sail_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/sail-sg/volo/releases/download/volo_1/d5_448_87.0.pth.tar',
        crop_pct=1.15, input_size=(3, 448, 448)),
    'volo_d5_512.sail_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/sail-sg/volo/releases/download/volo_1/d5_512_87.07.pth.tar',
        crop_pct=1.15, input_size=(3, 512, 512)),
})


@register_model
def volo_d1_224(pretrained=False, **kwargs) -> VOLO:
    """ VOLO-D1 model, Params: 27M """
    model_args = dict(layers=(4, 4, 8, 2), embed_dims=(192, 384, 384, 384), num_heads=(6, 12, 12, 12), **kwargs)
    model = _create_volo('volo_d1_224', pretrained=pretrained, **model_args)
    return model


@register_model
def volo_d1_384(pretrained=False, **kwargs) -> VOLO:
    """ VOLO-D1 model, Params: 27M """
    model_args = dict(layers=(4, 4, 8, 2), embed_dims=(192, 384, 384, 384), num_heads=(6, 12, 12, 12), **kwargs)
    model = _create_volo('volo_d1_384', pretrained=pretrained, **model_args)
    return model


@register_model
def volo_d2_224(pretrained=False, **kwargs) -> VOLO:
    """ VOLO-D2 model, Params: 59M """
    model_args = dict(layers=(6, 4, 10, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16), **kwargs)
    model = _create_volo('volo_d2_224', pretrained=pretrained, **model_args)
    return model


@register_model
def volo_d2_384(pretrained=False, **kwargs) -> VOLO:
    """ VOLO-D2 model, Params: 59M """
    model_args = dict(layers=(6, 4, 10, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16), **kwargs)
    model = _create_volo('volo_d2_384', pretrained=pretrained, **model_args)
    return model


@register_model
def volo_d3_224(pretrained=False, **kwargs) -> VOLO:
    """ VOLO-D3 model, Params: 86M """
    model_args = dict(layers=(8, 8, 16, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16), **kwargs)
    model = _create_volo('volo_d3_224', pretrained=pretrained, **model_args)
    return model


@register_model
def volo_d3_448(pretrained=False, **kwargs) -> VOLO:
    """ VOLO-D3 model, Params: 86M """
    model_args = dict(layers=(8, 8, 16, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16), **kwargs)
    model = _create_volo('volo_d3_448', pretrained=pretrained, **model_args)
    return model


@register_model
def volo_d4_224(pretrained=False, **kwargs) -> VOLO:
    """ VOLO-D4 model, Params: 193M """
    model_args = dict(layers=(8, 8, 16, 4), embed_dims=(384, 768, 768, 768), num_heads=(12, 16, 16, 16), **kwargs)
    model = _create_volo('volo_d4_224', pretrained=pretrained, **model_args)
    return model


@register_model
def volo_d4_448(pretrained=False, **kwargs) -> VOLO:
    """ VOLO-D4 model, Params: 193M """
    model_args = dict(layers=(8, 8, 16, 4), embed_dims=(384, 768, 768, 768), num_heads=(12, 16, 16, 16), **kwargs)
    model = _create_volo('volo_d4_448', pretrained=pretrained, **model_args)
    return model


@register_model
def volo_d5_224(pretrained=False, **kwargs) -> VOLO:
    """ VOLO-D5 model, Params: 296M
    stem_hidden_dim=128, the dim in patch embedding is 128 for VOLO-D5
    """
    model_args = dict(
        layers=(12, 12, 20, 4), embed_dims=(384, 768, 768, 768), num_heads=(12, 16, 16, 16),
        mlp_ratio=4, stem_hidden_dim=128, **kwargs)
    model = _create_volo('volo_d5_224', pretrained=pretrained, **model_args)
    return model


@register_model
def volo_d5_448(pretrained=False, **kwargs) -> VOLO:
    """ VOLO-D5 model, Params: 296M
    stem_hidden_dim=128, the dim in patch embedding is 128 for VOLO-D5
    """
    model_args = dict(
        layers=(12, 12, 20, 4), embed_dims=(384, 768, 768, 768), num_heads=(12, 16, 16, 16),
        mlp_ratio=4, stem_hidden_dim=128, **kwargs)
    model = _create_volo('volo_d5_448', pretrained=pretrained, **model_args)
    return model


@register_model
def volo_d5_512(pretrained=False, **kwargs) -> VOLO:
    """ VOLO-D5 model, Params: 296M
    stem_hidden_dim=128, the dim in patch embedding is 128 for VOLO-D5
    """
    model_args = dict(
        layers=(12, 12, 20, 4), embed_dims=(384, 768, 768, 768), num_heads=(12, 16, 16, 16),
        mlp_ratio=4, stem_hidden_dim=128, **kwargs)
    model = _create_volo('volo_d5_512', pretrained=pretrained, **model_args)
    return model
