""" Sequencer

Paper: `Sequencer: Deep LSTM for Image Classification` - https://arxiv.org/pdf/2205.01972.pdf

"""
#  Copyright (c) 2022. Yuki Tatsunami
#  Licensed under the Apache License, Version 2.0 (the "License");

import math
from functools import partial
from itertools import accumulate
from typing import Optional, Tuple

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.layers import lecun_normal_, DropPath, Mlp, PatchEmbed, ClassifierHead
from ._builder import build_model_with_cfg
from ._manipulate import named_apply
from ._registry import register_model, generate_default_cfgs

__all__ = ['Sequencer2d']  # model_registry will add each entrypoint fn to this


def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RNN, nn.GRU, nn.LSTM)):
        stdv = 1.0 / math.sqrt(module.hidden_size)
        for weight in module.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


class RNNIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RNNIdentity, self).__init__()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return x, None


class RNN2dBase(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = True,
            bidirectional: bool = True,
            union="cat",
            with_fc=True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2 * hidden_size if bidirectional else hidden_size
        self.union = union

        self.with_vertical = True
        self.with_horizontal = True
        self.with_fc = with_fc

        self.fc = None
        if with_fc:
            if union == "cat":
                self.fc = nn.Linear(2 * self.output_size, input_size)
            elif union == "add":
                self.fc = nn.Linear(self.output_size, input_size)
            elif union == "vertical":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_horizontal = False
            elif union == "horizontal":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_vertical = False
            else:
                raise ValueError("Unrecognized union: " + union)
        elif union == "cat":
            pass
            if 2 * self.output_size != input_size:
                raise ValueError(f"The output channel {2 * self.output_size} is different from the input channel {input_size}.")
        elif union == "add":
            pass
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
        elif union == "vertical":
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_horizontal = False
        elif union == "horizontal":
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_vertical = False
        else:
            raise ValueError("Unrecognized union: " + union)

        self.rnn_v = RNNIdentity()
        self.rnn_h = RNNIdentity()

    def forward(self, x):
        B, H, W, C = x.shape

        if self.with_vertical:
            v = x.permute(0, 2, 1, 3)
            v = v.reshape(-1, H, C)
            v, _ = self.rnn_v(v)
            v = v.reshape(B, W, H, -1)
            v = v.permute(0, 2, 1, 3)
        else:
            v = None

        if self.with_horizontal:
            h = x.reshape(-1, W, C)
            h, _ = self.rnn_h(h)
            h = h.reshape(B, H, W, -1)
        else:
            h = None

        if v is not None and h is not None:
            if self.union == "cat":
                x = torch.cat([v, h], dim=-1)
            else:
                x = v + h
        elif v is not None:
            x = v
        elif h is not None:
            x = h

        if self.fc is not None:
            x = self.fc(x)

        return x


class LSTM2d(RNN2dBase):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = True,
            bidirectional: bool = True,
            union="cat",
            with_fc=True,
    ):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                bias=bias,
                bidirectional=bidirectional,
            )
        if self.with_horizontal:
            self.rnn_h = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                bias=bias,
                bidirectional=bidirectional,
            )


class Sequencer2dBlock(nn.Module):
    def __init__(
            self,
            dim,
            hidden_size,
            mlp_ratio=3.0,
            rnn_layer=LSTM2d,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            num_layers=1,
            bidirectional=True,
            union="cat",
            with_fc=True,
            drop=0.,
            drop_path=0.,
    ):
        super().__init__()
        channels_dim = int(mlp_ratio * dim)
        self.norm1 = norm_layer(dim)
        self.rnn_tokens = rnn_layer(
            dim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            union=union,
            with_fc=with_fc,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.rnn_tokens(self.norm1(x)))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class Shuffle(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            B, H, W, C = x.shape
            r = torch.randperm(H * W)
            x = x.reshape(B, -1, C)
            x = x[:, r, :].reshape(B, H, W, -1)
        return x


class Downsample2d(nn.Module):
    def __init__(self, input_dim, output_dim, patch_size):
        super().__init__()
        self.down = nn.Conv2d(input_dim, output_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.down(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Sequencer2dStage(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            depth,
            patch_size,
            hidden_size,
            mlp_ratio,
            downsample=False,
            block_layer=Sequencer2dBlock,
            rnn_layer=LSTM2d,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            num_layers=1,
            bidirectional=True,
            union="cat",
            with_fc=True,
            drop=0.,
            drop_path=0.,
    ):
        super().__init__()
        if downsample:
            self.downsample = Downsample2d(dim, dim_out, patch_size)
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        blocks = []
        for block_idx in range(depth):
            blocks.append(block_layer(
                dim_out,
                hidden_size,
                mlp_ratio=mlp_ratio,
                rnn_layer=rnn_layer,
                mlp_layer=mlp_layer,
                norm_layer=norm_layer,
                act_layer=act_layer,
                num_layers=num_layers,
                bidirectional=bidirectional,
                union=union,
                with_fc=with_fc,
                drop=drop,
                drop_path=drop_path[block_idx] if isinstance(drop_path, (list, tuple)) else drop_path,
            ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class Sequencer2d(nn.Module):
    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            global_pool='avg',
            layers=(4, 3, 8, 3),
            patch_sizes=(7, 2, 2, 1),
            embed_dims=(192, 384, 384, 384),
            hidden_sizes=(48, 96, 96, 96),
            mlp_ratios=(3.0, 3.0, 3.0, 3.0),
            block_layer=Sequencer2dBlock,
            rnn_layer=LSTM2d,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            num_rnn_layers=1,
            bidirectional=True,
            union="cat",
            with_fc=True,
            drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem_norm=False,
    ):
        super().__init__()
        assert global_pool in ('', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = embed_dims[-1]  # for consistency with other models
        self.feature_dim = -1  # channel dim index for feature outputs (rank 4, NHWC)
        self.output_fmt = 'NHWC'
        self.feature_info = []

        self.stem = PatchEmbed(
            img_size=None,
            patch_size=patch_sizes[0],
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            norm_layer=norm_layer if stem_norm else None,
            flatten=False,
            output_fmt='NHWC',
        )

        assert len(layers) == len(patch_sizes) == len(embed_dims) == len(hidden_sizes) == len(mlp_ratios)
        reductions = list(accumulate(patch_sizes, lambda x, y: x * y))
        stages = []
        prev_dim = embed_dims[0]
        for i, _ in enumerate(embed_dims):
            stages += [Sequencer2dStage(
                prev_dim,
                embed_dims[i],
                depth=layers[i],
                downsample=i > 0,
                patch_size=patch_sizes[i],
                hidden_size=hidden_sizes[i],
                mlp_ratio=mlp_ratios[i],
                block_layer=block_layer,
                rnn_layer=rnn_layer,
                mlp_layer=mlp_layer,
                norm_layer=norm_layer,
                act_layer=act_layer,
                num_layers=num_rnn_layers,
                bidirectional=bidirectional,
                union=union,
                with_fc=with_fc,
                drop=drop_rate,
                drop_path=drop_path_rate,
            )]
            prev_dim = embed_dims[i]
            self.feature_info += [dict(num_chs=prev_dim, reduction=reductions[i], module=f'stages.{i}')]

        self.stages = nn.Sequential(*stages)
        self.norm = norm_layer(embed_dims[-1])
        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            input_fmt=self.output_fmt,
        )

        self.init_weights(nlhb=nlhb)

    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=[
                (r'^stages\.(\d+)', None),
                (r'^norm', (99999,))
            ] if coarse else [
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^stages\.(\d+)\.downsample', (0,)),
                (r'^norm', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    """ Remap original checkpoints -> timm """
    if 'stages.0.blocks.0.norm1.weight' in state_dict:
        return state_dict  # already translated checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']

    import re
    out_dict = {}
    for k, v in state_dict.items():
        k = re.sub(r'blocks.([0-9]+).([0-9]+).down', lambda x: f'stages.{int(x.group(1)) + 1}.downsample.down', k)
        k = re.sub(r'blocks.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = k.replace('head.', 'head.fc.')
        out_dict[k] = v

    return out_dict


def _create_sequencer2d(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(range(3))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        Sequencer2d,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs,
    )
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': DEFAULT_CROP_PCT, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.proj', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'sequencer2d_s.in1k': _cfg(hf_hub_id='timm/'),
    'sequencer2d_m.in1k': _cfg(hf_hub_id='timm/'),
    'sequencer2d_l.in1k': _cfg(hf_hub_id='timm/'),
})


@register_model
def sequencer2d_s(pretrained=False, **kwargs) -> Sequencer2d:
    model_args = dict(
        layers=[4, 3, 8, 3],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[48, 96, 96, 96],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2d,
        bidirectional=True,
        union="cat",
        with_fc=True,
    )
    model = _create_sequencer2d('sequencer2d_s', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def sequencer2d_m(pretrained=False, **kwargs) -> Sequencer2d:
    model_args = dict(
        layers=[4, 3, 14, 3],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[48, 96, 96, 96],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2d,
        bidirectional=True,
        union="cat",
        with_fc=True,
        **kwargs)
    model = _create_sequencer2d('sequencer2d_m', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def sequencer2d_l(pretrained=False, **kwargs) -> Sequencer2d:
    model_args = dict(
        layers=[8, 8, 16, 4],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[48, 96, 96, 96],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2d,
        bidirectional=True,
        union="cat",
        with_fc=True,
        **kwargs)
    model = _create_sequencer2d('sequencer2d_l', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
