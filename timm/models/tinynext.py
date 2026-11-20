"""TinyNeXt
An Efficient Hybrid Vision Transformer for TinyML Applications (ICCV2025)
- `An Efficient Hybrid Vision Transformer for TinyML Applications` - https://doi.org/10.1109/ICCV51701.2025.01852
- code: https://github.com/yuffeenn/TinyNeXt

@inproceedings{tinynext_iccv2025,
	author    = {Zeng, Fanhong and Li, Huanan and Guan, Juntao and Fan, Rui and Wu, Tong and Wang, Xilong and Lai, Rui},
	title     = {An Efficient Hybrid Vision Transformer for TinyML Applications},
	booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
	month     = {October}, 
	year      = {2025},    
	pages     = {19914-19924} 
}
Modifications by / Copyright 2026 Ryan Hou & Ross Wightman, original copyrights below
"""
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn

from timm.layers import (
    ConvNormAct,
    SqueezeExcite,
    ConvMlp,
    SelectAdaptivePool2d,
    Linear,
    get_device_dtype,
    trunc_normal_,
)
from ._efficientnet_blocks import InvertedResidual

from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['TinyNeXt']


class Attention(nn.Module):
    def __init__(self, dim: int, device=None, dtype=None):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.scale = dim ** -0.5
        self.linear1 = nn.Linear(dim, dim, bias=False, **dd)
        self.linear2 = nn.Linear(dim, dim, bias=False, **dd)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(-1, -2).contiguous()
        attn = torch.matmul(self.linear1(x), x.transpose(-2, -1))
        attn = self.softmax(attn * self.scale)
        x = torch.matmul(attn, self.linear2(x))
        return x.transpose(-1, -2).view(B, C, H, W).contiguous()


class FormerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 2.0,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.attention = nn.Sequential(norm_layer(dim, **dd), Attention(dim, **dd))
        self.local = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False, **dd)
        self.mlp = nn.Sequential(
            norm_layer(dim, **dd),
            ConvMlp(
                dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                bias=True,
                **dd,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        x = x + self.local(x)
        x = x + self.mlp(x)
        return x


class SeBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 2.0,
            reduction: int = 4,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        # SqueezeExcite with hard_sigmoid gate, BN+ReLU
        self.attention = nn.Sequential(
            norm_layer(dim, **dd),
            SqueezeExcite(
                dim,
                rd_ratio=1.0/reduction,
                rd_divisor=1,
                bias=False,
                act_layer=nn.ReLU,
                norm_layer=norm_layer,
                gate_layer='hard_sigmoid',
                **dd,
            ),
        )
        self.local = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False, **dd)
        self.mlp = nn.Sequential(
            norm_layer(dim, **dd),
            ConvMlp(
                dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                bias=True,
                **dd,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        x = x + self.local(x)
        x = x + self.mlp(x)
        return x


class Stage(nn.Module):
    def __init__(
            self,
            dim: int,
            prev_dim: int,
            depth: int,
            block_type: str = 'mv2',
            mlp_ratio: float = 2.0,
            reduction: int = 4,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.GELU,
            downsample: bool = False,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.grad_checkpointing = False
        self.downsample = InvertedResidual(
            in_chs=prev_dim,
            out_chs=dim,
            dw_kernel_size=3,
            stride=2,
            exp_ratio=dim / prev_dim,
            act_layer=nn.ReLU,
            norm_layer=norm_layer,
            **dd,
        ) if downsample else nn.Identity()

        blocks = []
        for _ in range(depth):
            if block_type == 'mv2':
                blocks.append(InvertedResidual(dim, dim, exp_ratio=mlp_ratio, norm_layer=norm_layer, **dd))
            elif block_type == 'former':
                blocks.append(FormerBlock(dim, mlp_ratio, norm_layer, act_layer, **dd))
            elif block_type == 'se':
                blocks.append(SeBlock(dim, mlp_ratio, reduction, norm_layer, act_layer, **dd))
            else:
                raise ValueError(f"invalid block type {block_type}")
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class TinyNeXt(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            dims: Tuple[int, ...] = (32, 64, 96, 192),
            depths: Tuple[int, ...] = (3, 3, 6, 2),
            block_types: Tuple[str, ...] = ('mv2', 'mv2', 'former', 'se'),
            ratios: Tuple[float, ...] = (2., 2., 2., 2.),
            reduction: int = 4,
            drop_rate: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_rate = drop_rate
        self.feature_info = []

        self.stem = nn.Sequential(
            ConvNormAct(in_chans, dims[0] // 2, 3, 2, norm_layer=norm_layer, **dd),
            ConvNormAct(dims[0] // 2, dims[0], 3, 2, norm_layer=norm_layer, **dd),
        )
        prev_dim = dims[0]

        stages = []
        for i in range(len(depths)):
            stage = Stage(
                dim=dims[i],
                prev_dim=prev_dim,
                depth=depths[i],
                block_type=block_types[i],
                mlp_ratio=ratios[i],
                reduction=reduction,
                norm_layer=norm_layer,
                act_layer=act_layer,
                downsample=True if i != 0 else False,
                **dd,
            )
            stages.append(stage)
            self.feature_info.append(dict(num_chs=dims[i], reduction=2 ** (i + 2), module=f'stages.{i}'))
            prev_dim = dims[i]
        self.stages = nn.Sequential(*stages)

        # Classifier head
        self.num_features = dims[-1]
        self.norm = norm_layer(self.num_features, **dd)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.head_hidden_size = self.num_features * self.global_pool.feat_mult()
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = Linear(self.head_hidden_size, num_classes, bias=True, **dd) if num_classes > 0 else nn.Identity()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return set()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        matcher = dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        dd = get_device_dtype(self)
        self.num_classes = num_classes
        if global_pool is None:
            global_pool = self.global_pool.pool_type
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.global_pool.train(self.training)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.flatten.train(self.training)
        self.head_hidden_size = self.num_features * self.global_pool.feat_mult()
        self.head = Linear(self.head_hidden_size, num_classes, bias=True, **dd) if num_classes > 0 else nn.Identity()
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
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:
            List of intermediate features or tuple of (final features, intermediates).
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
        prune_end = max_index < len(self.stages) - 1
        if prune_norm or prune_end:
            self.norm = nn.Identity()
        self.stages = self.stages[:max_index + 1]  # truncate blocks w/ stem as idx 0
        if prune_end:
            # features now end at the last kept tap, the head is sized for the final features so it goes too
            self.num_features = self.head_hidden_size = self.feature_info[max_index]['num_chs']
        if prune_head or prune_end:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    state_dict = state_dict.get('state_dict', state_dict)
    state_dict = state_dict.get('model', state_dict)
    if 'stem.0.conv.weight' in state_dict:
        return state_dict

    key_replacements = (
        ('.layers.0.0.', '.conv_pw.'),
        ('.layers.0.1.', '.bn1.'),
        ('.layers.1.0.', '.conv_dw.'),
        ('.layers.1.1.', '.bn2.'),
        ('.layers.2.', '.conv_pwl.'),
        ('.layers.3.', '.bn3.'),
        ('.se.0.', '.attention.0.'),
        ('.se.1.se.1.', '.attention.1.fc1.'),
        ('.se.1.se.2.', '.attention.1.bn.'),
        ('.se.1.se.4.', '.attention.1.fc2.'),
        ('.mlp.1.0.', '.mlp.1.fc1.'),
        ('.mlp.1.2.', '.mlp.1.fc2.'),
    )
    out_dict = {}
    for k, v in state_dict.items():
        if k.startswith('embeds.0.'):
            m = re.match(r'^embeds\.0\.(\d+)\.(0|1)\.(.*)$', k)
            if m:
                stem_idx, layer_idx, rest = m.groups()
                layer_name = 'conv' if layer_idx == '0' else 'bn'
                k = f'stem.{stem_idx}.{layer_name}.{rest}'
        elif k.startswith('embeds.'):
            m = re.match(r'^embeds\.(\d+)\.0\.(.*)$', k)
            if m:
                stage_idx, rest = m.groups()
                k = f'stages.{stage_idx}.downsample.{rest}'
        elif k.startswith('stages.'):
            m = re.match(r'^stages\.(\d+)\.blocks\.downsample\.(.*)$', k)
            if m:
                stage_idx, rest = m.groups()
                k = f'stages.{stage_idx}.downsample.{rest}'
            else:
                m = re.match(r'^stages\.(\d+)\.(\d+)\.(.*)$', k)
                if m:
                    stage_idx, block_idx, rest = m.groups()
                    k = f'stages.{stage_idx}.blocks.{block_idx}.{rest}'

        for src, dst in key_replacements:
            k = k.replace(src, dst)
        if k.startswith('class_head.'):
            k = k.replace('class_head.', 'head.', 1)
        out_dict[k] = v
    return out_dict


def _cfg(url: str = '', **kwargs: Any) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': (128./255, 128./255, 128./255), 'std': (1./255, 1./255, 1./255),
        'first_conv': 'stem.0.conv', 'classifier': 'head',
        'paper_name': 'An Efficient Hybrid Vision Transformer for TinyML Applications',
        'origin_url': 'https://github.com/yuffeenn/TinyNeXt',
        'license': 'mit',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'tinynext_t.in1k': _cfg(
        # hf_hub_id='timm/'
        # url="https://github.com/yuffeenn/TinyNeXt/blob/main/classification/logs/tinynext_t/tinynext_t.pth",
    ),
    'tinynext_s.in1k': _cfg(
        # hf_hub_id='timm/',
        # url="https://github.com/yuffeenn/TinyNeXt/blob/main/classification/logs/tinynext_s/tinynext_s.pth",
    ),
    'tinynext_m.in1k': _cfg(
        # hf_hub_id='timm/',
        # url="https://github.com/yuffeenn/TinyNeXt/blob/main/classification/logs/tinynext_m/tinynext_m.pth",
    ),
})


def _create_tinynext(variant: str, pretrained: bool = False, **kwargs: Any) -> TinyNeXt:
    model = build_model_with_cfg(
        TinyNeXt, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs
    )
    return model


@register_model
def tinynext_t(pretrained: bool = False, **kwargs: Any) -> TinyNeXt:
    model_args = dict(dims=(32, 64, 96, 192), depths=(3, 3, 6, 2), ratios=(2., 2., 2., 2.))
    return _create_tinynext('tinynext_t', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def tinynext_s(pretrained: bool = False, **kwargs: Any) -> TinyNeXt:
    model_args = dict(dims=(32, 64, 96, 192), depths=(3, 3, 8, 3), ratios=(2., 2., 2., 2.))
    return _create_tinynext('tinynext_s', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def tinynext_m(pretrained: bool = False, **kwargs: Any) -> TinyNeXt:
    model_args = dict(dims=(32, 64, 128, 256), depths=(4, 4, 9, 4), ratios=(2., 2., 2., 1.5))
    return _create_tinynext('tinynext_m', pretrained=pretrained, **dict(model_args, **kwargs))
