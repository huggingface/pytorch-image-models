""" ConvMixer

Papers:
* `Patches Are All You Need?` - https://arxiv.org/abs/2201.09792
"""
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d, get_device_dtype
from ._registry import register_model, generate_default_cfgs
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint, checkpoint_seq

__all__ = ['ConvMixer']


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            kernel_size: int = 9,
            patch_size: int = 7,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            drop_rate: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.num_features = dim
        self.grad_checkpointing = False

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size, **dd),
            act_layer(),
            nn.BatchNorm2d(dim, **dd)
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same", **dd),
                        act_layer(),
                        nn.BatchNorm2d(dim, **dd)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1, **dd),
                    act_layer(),
                    nn.BatchNorm2d(dim, **dd)
            ) for i in range(depth)]
        )
        self.feature_info = [dict(num_chs=dim, reduction=patch_size, module=f'blocks.{i}') for i in range(depth)]
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool, flatten=bool(global_pool))  # no flatten w/o pool
        self.head_hidden_size = self.num_features * self.global_pool.feat_mult()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.head_hidden_size, num_classes, **dd) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^stem', blocks=r'^blocks\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        dd = get_device_dtype(self)
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool, flatten=bool(global_pool))
            self.global_pool.train(self.training)
        self.head_hidden_size = self.num_features * self.global_pool.feat_mult()
        self.head = nn.Linear(self.head_hidden_size, num_classes, **dd) if num_classes > 0 else nn.Identity()
        self.head.train(self.training)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor.
            indices: Take last n blocks if int, all if None, select matching indices if sequence.
                Index i is the output of block i.
            norm: Apply norm layer to compatible intermediates (unused, no final norm).
            stop_early: Stop iterating over blocks when last desired intermediate hit.
            output_fmt: Shape of intermediate feature outputs.
            intermediates_only: Only return intermediate features.

        Returns:
            Features and list of intermediate features or just intermediate features.
        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        x = self.stem(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)
            if i in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ) -> List[int]:
        """Prune layers not required for specified intermediates.

        Args:
            indices: Indices of intermediate layers to keep.
            prune_norm: Whether to prune normalization layers (unused, no final norm).
            prune_head: Whether to prune the classifier head.

        Returns:
            List of indices that were kept.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_convmixer(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    return build_model_with_cfg(
        ConvMixer,
        variant,
        pretrained,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        'first_conv': 'stem.0', 'license': 'mit',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'convmixer_1536_20.in1k': _cfg(hf_hub_id='timm/'),
    'convmixer_768_32.in1k': _cfg(hf_hub_id='timm/'),
    'convmixer_1024_20_ks9_p14.in1k': _cfg(hf_hub_id='timm/')
})



@register_model
def convmixer_1536_20(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=1536, depth=20, kernel_size=9, patch_size=7, **kwargs)
    return _create_convmixer('convmixer_1536_20', pretrained, **model_args)


@register_model
def convmixer_768_32(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=768, depth=32, kernel_size=7, patch_size=7, act_layer=nn.ReLU, **kwargs)
    return _create_convmixer('convmixer_768_32', pretrained, **model_args)


@register_model
def convmixer_1024_20_ks9_p14(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=1024, depth=20, kernel_size=9, patch_size=14, **kwargs)
    return _create_convmixer('convmixer_1024_20_ks9_p14', pretrained, **model_args)
