"""
Implementation of Prof-of-Concept Network: StarNet.

We make StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
    - like NO layer-scale in network design,
    - and NO EMA during training,
    - which would improve the performance further.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, SelectAdaptivePool2d, Linear, LayerType, trunc_normal_
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['StarNet']


class ConvBN(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            with_bn: bool = True,
            **kwargs,
    ):
        super().__init__()
        self.add_module('conv', nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, **kwargs))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_channels))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            mlp_ratio: int = 3,
            drop_path: float = 0.,
            act_layer: LayerType = nn.ReLU6,
    ):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=False)
        self.act = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = residual + self.drop_path(x)
        return x


class StarNet(nn.Module):
    def __init__(
            self,
            base_dim: int = 32,
            depths: List[int] = [3, 3, 12, 5],
            mlp_ratio: int = 4,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            act_layer: LayerType = nn.ReLU6,
            num_classes: int = 1000,
            in_chans: int = 3,
            global_pool: str = 'avg',
            output_stride: int = 32,
            **kwargs,
    ):
        super().__init__()
        assert output_stride == 32
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []
        stem_chs = 32

        # stem layer
        self.stem = nn.Sequential(
            ConvBN(in_chans, stem_chs, kernel_size=3, stride=2, padding=1),
            act_layer(),
        )
        prev_chs = stem_chs

        # build stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        stages = []
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(prev_chs, embed_dim, 3, stride=2, padding=1)
            blocks = [Block(embed_dim, mlp_ratio, dpr[cur + i], act_layer) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            prev_chs = embed_dim
            stages.append(nn.Sequential(down_sampler, *blocks))
            self.feature_info.append(dict(
                    num_chs=prev_chs, reduction=2**(i_layer+2), module=f'stages.{i_layer}'))
        self.stages = nn.Sequential(*stages)
        # head
        self.num_features = self.head_hidden_size = prev_chs
        self.norm = nn.BatchNorm2d(self.num_features)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.head = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return set()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        matcher = dict(
            stem=r'^stem\.\d+',
            blocks=[
                (r'^stages\.(\d+)' if coarse else r'^stages\.(\d+)\.(\d+)', None),
                (r'norm', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            # NOTE: cannot meaningfully change pooling of efficient head after creation
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.head = Linear(self.head_hidden_size, num_classes) if num_classes > 0 else nn.Identity()

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
        last_idx = len(self.stages) - 1

        # forward pass
        x = self.stem(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]

        for feat_idx, stage in enumerate(stages):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(stage, x)
            else:
                x = stage(x)
            if feat_idx in take_indices:
                if norm and feat_idx == last_idx:
                    x_inter = self.norm(x)  # applying final norm last intermediate
                else:
                    x_inter = x
                intermediates.append(x_inter)

        if intermediates_only:
            return intermediates

        if feat_idx == last_idx:
            x = self.norm(x)

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
            self.norm = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    return state_dict.get('state_dict', state_dict)


def _cfg(url: str = '', **kwargs: Any) -> Dict[str, Any]:
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0.conv', 'classifier': 'head',
        'paper_ids': 'arXiv:2403.19967',
        'paper_name': 'Rewrite the Stars',
        'origin_url': 'https://github.com/ma-xu/Rewrite-the-Stars',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'starnet_s1.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar',
    ),
    'starnet_s2.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar',
    ),
    'starnet_s3.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar',
    ),
    'starnet_s4.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar',
    ),
    'starnet_s050.untrained': _cfg(),
    'starnet_s100.untrained': _cfg(),
    'starnet_s150.untrained': _cfg(),
})


def _create_starnet(variant: str, pretrained: bool = False, **kwargs: Any) -> StarNet:
    model = build_model_with_cfg(
        StarNet, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )
    return model


@register_model
def starnet_s1(pretrained: bool = False, **kwargs: Any) -> StarNet:
    model_args = dict(base_dim=24, depths=[2, 2, 8, 3])
    return _create_starnet('starnet_s1', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def starnet_s2(pretrained: bool = False, **kwargs: Any) -> StarNet:
    model_args = dict(base_dim=32, depths=[1, 2, 6, 2])
    return _create_starnet('starnet_s2', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def starnet_s3(pretrained: bool = False, **kwargs: Any) -> StarNet:
    model_args = dict(base_dim=32, depths=[2, 2, 8, 4])
    return _create_starnet('starnet_s3', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def starnet_s4(pretrained: bool = False, **kwargs: Any) -> StarNet:
    model_args = dict(base_dim=32, depths=[3, 3, 12, 5])
    return _create_starnet('starnet_s4', pretrained=pretrained, **dict(model_args, **kwargs))


# very small networks #
@register_model
def starnet_s050(pretrained: bool = False, **kwargs: Any) -> StarNet:
    model_args = dict(base_dim=16, depths=[1, 1, 3, 1], mlp_ratio=3)
    return _create_starnet('starnet_s050', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def starnet_s100(pretrained: bool = False, **kwargs: Any) -> StarNet:
    model_args = dict(base_dim=20, depths=[1, 2, 4, 1], mlp_ratio=4)
    return _create_starnet('starnet_s100', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def starnet_s150(pretrained: bool = False, **kwargs: Any) -> StarNet:
    model_args = dict(base_dim=24, depths=[1, 2, 4, 2], mlp_ratio=3)
    return _create_starnet('starnet_s150', pretrained=pretrained, **dict(model_args, **kwargs))