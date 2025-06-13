"""
RDNet
Copyright (c) 2024-present NAVER Cloud Corp.
Apache-2.0
"""

from functools import partial
from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, NormMlpClassifierHead, ClassifierHead, EffectiveSEModule, \
    make_divisible, get_act_layer, get_norm_layer
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import named_apply
from ._registry import register_model, generate_default_cfgs

__all__ = ["RDNet"]


class Block(nn.Module):
    def __init__(self, in_chs, inter_chs, out_chs, norm_layer, act_layer):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, groups=in_chs, kernel_size=7, stride=1, padding=3),
            norm_layer(in_chs),
            nn.Conv2d(in_chs, inter_chs, kernel_size=1, stride=1, padding=0),
            act_layer(),
            nn.Conv2d(inter_chs, out_chs, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.layers(x)


class BlockESE(nn.Module):
    def __init__(self, in_chs, inter_chs, out_chs, norm_layer, act_layer):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, groups=in_chs, kernel_size=7, stride=1, padding=3),
            norm_layer(in_chs),
            nn.Conv2d(in_chs, inter_chs, kernel_size=1, stride=1, padding=0),
            act_layer(),
            nn.Conv2d(inter_chs, out_chs, kernel_size=1, stride=1, padding=0),
            EffectiveSEModule(out_chs),
        )

    def forward(self, x):
        return self.layers(x)


def _get_block_type(block: str):
    block = block.lower().strip()
    if block == "block":
        return Block
    elif block == "blockese":
        return BlockESE
    else:
        assert False, f"Unknown block type ({block})."


class DenseBlock(nn.Module):
    def __init__(
            self,
            num_input_features: int = 64,
            growth_rate: int = 64,
            bottleneck_width_ratio: float = 4.0,
            drop_path_rate: float = 0.0,
            drop_rate: float = 0.0,
            rand_gather_step_prob: float = 0.0,
            block_idx: int = 0,
            block_type: str = "Block",
            ls_init_value: float = 1e-6,
            norm_layer: str = "layernorm2d",
            act_layer: str = "gelu",
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.rand_gather_step_prob = rand_gather_step_prob
        self.block_idx = block_idx
        self.growth_rate = growth_rate

        self.gamma = nn.Parameter(ls_init_value * torch.ones(growth_rate)) if ls_init_value > 0 else None
        growth_rate = int(growth_rate)
        inter_chs = int(num_input_features * bottleneck_width_ratio / 8) * 8

        self.drop_path = DropPath(drop_path_rate)

        self.layers = _get_block_type(block_type)(
            in_chs=num_input_features,
            inter_chs=inter_chs,
            out_chs=growth_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(x, 1)
        x = self.layers(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        x = self.drop_path(x)
        return x


class DenseStage(nn.Sequential):
    def __init__(self, num_block, num_input_features, drop_path_rates, growth_rate, **kwargs):
        super().__init__()
        for i in range(num_block):
            layer = DenseBlock(
                num_input_features=num_input_features,
                growth_rate=growth_rate,
                drop_path_rate=drop_path_rates[i],
                block_idx=i,
                **kwargs,
            )
            num_input_features += growth_rate
            self.add_module(f"dense_block{i}", layer)
        self.num_out_features = num_input_features

    def forward(self, init_feature: torch.Tensor) -> torch.Tensor:
        features = [init_feature]
        for module in self:
            new_feature = module(features)
            features.append(new_feature)
        return torch.cat(features, 1)


class RDNet(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,  # timm option [--in-chans]
            num_classes: int = 1000,  # timm option [--num-classes]
            global_pool: str = 'avg',  # timm option [--gp]
            growth_rates: Union[List[int], Tuple[int]] = (64, 104, 128, 128, 128, 128, 224),
            num_blocks_list: Union[List[int], Tuple[int]] = (3, 3, 3, 3, 3, 3, 3),
            block_type: Union[List[int], Tuple[int]] = ("Block",) * 2 + ("BlockESE",) * 5,
            is_downsample_block: Union[List[bool], Tuple[bool]] = (None, True, True, False, False, False, True),
            bottleneck_width_ratio: float = 4.0,
            transition_compression_ratio: float = 0.5,
            ls_init_value: float = 1e-6,
            stem_type: str = 'patch',
            patch_size: int = 4,
            num_init_features: int = 64,
            head_init_scale: float = 1.,
            head_norm_first: bool = False,
            conv_bias: bool = True,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: str = "layernorm2d",
            norm_eps: Optional[float] = None,
            drop_rate: float = 0.0,  # timm option [--drop: dropout ratio]
            drop_path_rate: float = 0.0,  # timm option [--drop-path: drop-path ratio]
    ):
        """
        Args:
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            global_pool: Global pooling type.
            growth_rates: Growth rate at each stage.
            num_blocks_list: Number of blocks at each stage.
            is_downsample_block: Whether to downsample at each stage.
            bottleneck_width_ratio: Bottleneck width ratio (similar to mlp expansion ratio).
            transition_compression_ratio: Channel compression ratio of transition layers.
            ls_init_value: Init value for Layer Scale, disabled if None.
            stem_type: Type of stem.
            patch_size: Stem patch size for patch stem.
            num_init_features: Number of features of stem.
            head_init_scale: Init scaling value for classifier weights and biases.
            head_norm_first: Apply normalization before global pool + head.
            conv_bias: Use bias layers w/ all convolutions.
            act_layer: Activation layer type.
            norm_layer: Normalization layer type.
            norm_eps: Small value to avoid division by zero in normalization.
            drop_rate: Head pre-classifier dropout rate.
            drop_path_rate: Stochastic depth drop rate.
        """
        super().__init__()
        assert len(growth_rates) == len(num_blocks_list) == len(is_downsample_block)
        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)
        if norm_eps is not None:
            norm_layer = partial(norm_layer, eps=norm_eps)

        self.num_classes = num_classes
        self.drop_rate = drop_rate

        # stem
        assert stem_type in ('patch', 'overlap', 'overlap_tiered')
        if stem_type == 'patch':
            # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, num_init_features, kernel_size=patch_size, stride=patch_size, bias=conv_bias),
                norm_layer(num_init_features),
            )
            stem_stride = patch_size
        else:
            mid_chs = make_divisible(num_init_features // 2) if 'tiered' in stem_type else num_init_features
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, mid_chs, kernel_size=3, stride=2, padding=1, bias=conv_bias),
                nn.Conv2d(mid_chs, num_init_features, kernel_size=3, stride=2, padding=1, bias=conv_bias),
                norm_layer(num_init_features),
            )
            stem_stride = 4

        # features
        self.feature_info = []
        self.num_stages = len(growth_rates)
        curr_stride = stem_stride
        num_features = num_init_features
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(num_blocks_list)).split(num_blocks_list)]

        dense_stages = []
        for i in range(self.num_stages):
            dense_stage_layers = []
            if i != 0:
                compressed_num_features = int(num_features * transition_compression_ratio / 8) * 8
                k_size = stride = 1
                if is_downsample_block[i]:
                    curr_stride *= 2
                    k_size = stride = 2

                dense_stage_layers.append(norm_layer(num_features))
                dense_stage_layers.append(
                    nn.Conv2d(num_features, compressed_num_features, kernel_size=k_size, stride=stride, padding=0)
                )
                num_features = compressed_num_features

            stage = DenseStage(
                num_block=num_blocks_list[i],
                num_input_features=num_features,
                growth_rate=growth_rates[i],
                bottleneck_width_ratio=bottleneck_width_ratio,
                drop_rate=drop_rate,
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                block_type=block_type[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            dense_stage_layers.append(stage)
            num_features += num_blocks_list[i] * growth_rates[i]

            if i + 1 == self.num_stages or (i + 1 != self.num_stages and is_downsample_block[i + 1]):
                self.feature_info += [
                    dict(
                        num_chs=num_features,
                        reduction=curr_stride,
                        module=f'dense_stages.{i}',
                        growth_rate=growth_rates[i],
                    )
                ]
            dense_stages.append(nn.Sequential(*dense_stage_layers))
        self.dense_stages = nn.Sequential(*dense_stages)
        self.num_features = self.head_hidden_size = num_features

        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default RDNet ordering (pretrained NV weights)
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
        assert not coarse, "coarse grouping is not implemented for RDNet"
        return dict(
            stem=r'^stem',
            blocks=r'^dense_stages\.(\d+)',
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.dense_stages:
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
        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        stage_ends = [int(info['module'].split('.')[-1]) for info in self.feature_info]
        take_indices, max_index = feature_take_indices(len(stage_ends), indices)
        take_indices = [stage_ends[i] for i in take_indices]
        max_index = stage_ends[max_index]

        # forward pass
        x = self.stem(x)

        last_idx = len(self.dense_stages) - 1
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            dense_stages = self.dense_stages
        else:
            dense_stages = self.dense_stages[:max_index + 1]
        for feat_idx, stage in enumerate(dense_stages):
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
        stage_ends = [int(info['module'].split('.')[-1]) for info in self.feature_info]
        take_indices, max_index = feature_take_indices(len(stage_ends), indices)
        max_index = stage_ends[max_index]
        self.dense_stages = self.dense_stages[:max_index + 1]  # truncate blocks w/ stem as idx 0
        if prune_norm:
            self.norm_pre = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        x = self.stem(x)
        x = self.dense_stages(x)
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
        nn.init.kaiming_normal_(module.weight)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.constant_(module.bias, 0)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def checkpoint_filter_fn(state_dict, model):
    """ Remap NV checkpoints -> timm """
    if 'stem.0.weight' in state_dict:
        return state_dict  # non-NV checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']

    out_dict = {}

    for k, v in state_dict.items():
        k = k.replace('stem.stem.', 'stem.')
        out_dict[k] = v

    return out_dict


def _create_rdnet(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        RDNet, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs)
    return model


def _cfg(url='', **kwargs):
    return {
        "url": url,
        "num_classes": 1000, "input_size": (3, 224, 224), "pool_size": (7, 7),
        "crop_pct": 0.9, "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD,
        "first_conv": "stem.0", "classifier": "head.fc",
        "paper_ids": "arXiv:2403.19588",
        "paper_name": "DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs",
        "origin_url": "https://github.com/naver-ai/rdnet",
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'rdnet_tiny.nv_in1k': _cfg(
        hf_hub_id='naver-ai/rdnet_tiny.nv_in1k'),
    'rdnet_small.nv_in1k': _cfg(
        hf_hub_id='naver-ai/rdnet_small.nv_in1k'),
    'rdnet_base.nv_in1k': _cfg(
        hf_hub_id='naver-ai/rdnet_base.nv_in1k'),
    'rdnet_large.nv_in1k': _cfg(
        hf_hub_id='naver-ai/rdnet_large.nv_in1k'),
    'rdnet_large.nv_in1k_ft_in1k_384': _cfg(
        hf_hub_id='naver-ai/rdnet_large.nv_in1k_ft_in1k_384',
        input_size=(3, 384, 384), crop_pct=1.0, pool_size=(12, 12)),
})


@register_model
def rdnet_tiny(pretrained=False, **kwargs):
    n_layer = 7
    model_args = {
        "num_init_features": 64,
        "growth_rates": [64] + [104] + [128] * 4 + [224],
        "num_blocks_list": [3] * n_layer,
        "is_downsample_block": (None, True, True, False, False, False, True),
        "transition_compression_ratio": 0.5,
        "block_type": ["Block"] + ["Block"] + ["BlockESE"] * 4 + ["BlockESE"],
    }
    model = _create_rdnet("rdnet_tiny", pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def rdnet_small(pretrained=False, **kwargs):
    n_layer = 11
    model_args = {
        "num_init_features": 72,
        "growth_rates": [64] + [128] + [128] * (n_layer - 4) + [240] * 2,
        "num_blocks_list": [3] * n_layer,
        "is_downsample_block": (None, True, True, False, False, False, False, False, False, True, False),
        "transition_compression_ratio": 0.5,
        "block_type": ["Block"] + ["Block"] + ["BlockESE"] * (n_layer - 4) + ["BlockESE"] * 2,
    }
    model = _create_rdnet("rdnet_small", pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def rdnet_base(pretrained=False, **kwargs):
    n_layer = 11
    model_args = {
        "num_init_features": 120,
        "growth_rates": [96] + [128] + [168] * (n_layer - 4) + [336] * 2,
        "num_blocks_list": [3] * n_layer,
        "is_downsample_block": (None, True, True, False, False, False, False, False, False, True, False),
        "transition_compression_ratio": 0.5,
        "block_type": ["Block"] + ["Block"] + ["BlockESE"] * (n_layer - 4) + ["BlockESE"] * 2,
    }
    model = _create_rdnet("rdnet_base", pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def rdnet_large(pretrained=False, **kwargs):
    n_layer = 12
    model_args = {
        "num_init_features": 144,
        "growth_rates": [128] + [192] + [256] * (n_layer - 4) + [360] * 2,
        "num_blocks_list": [3] * n_layer,
        "is_downsample_block": (None, True, True, False, False, False, False, False, False, False, True, False),
        "transition_compression_ratio": 0.5,
        "block_type": ["Block"] + ["Block"] + ["BlockESE"] * (n_layer - 4) + ["BlockESE"] * 2,
    }
    model = _create_rdnet("rdnet_large", pretrained=pretrained, **dict(model_args, **kwargs))
    return model
