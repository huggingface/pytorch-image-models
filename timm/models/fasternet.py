"""FasterNet
Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks
- paper: https://arxiv.org/abs/2303.03667
- code: https://github.com/JierunChen/FasterNet

@article{chen2023run,
  title={Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks},
  author={Chen, Jierun and Kao, Shiu-hong and He, Hao and Zhuo, Weipeng and Wen, Song and Lee, Chul-Ho and Chan, S-H Gary},
  journal={arXiv preprint arXiv:2303.03667},
  year={2023}
}

Modifications by / Copyright 2025 Ryan Hou & Ross Wightman, original copyrights below
"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d, Linear, DropPath, trunc_normal_, LayerType
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['FasterNet']


class Partial_conv3(nn.Module):
    def __init__(self, dim: int, n_div: int, forward: str):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: torch.Tensor) -> torch.Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x: torch.Tensor) -> torch.Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class MLPBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            n_div: int,
            mlp_ratio: float,
            drop_path: float,
            layer_scale_init_value: float,
            act_layer: LayerType = partial(nn.ReLU, inplace=True),
            norm_layer: LayerType = nn.BatchNorm2d,
            pconv_fw_type: str = 'split_cat',
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(*[
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False),
        ])

        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.layer_scale = None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        if self.layer_scale is not None:
            x = shortcut + self.drop_path(
                self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        else:
            x = shortcut + self.drop_path(self.mlp(x))
        return x


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            n_div: int,
            mlp_ratio: float,
            drop_path: float,
            layer_scale_init_value: float,
            act_layer: LayerType = partial(nn.ReLU, inplace=True),
            norm_layer: LayerType = nn.BatchNorm2d,
            pconv_fw_type: str = 'split_cat',
            use_merge: bool = True,
            merge_size: Union[int, Tuple[int, int]] = 2,
    ):
        super().__init__()
        self.grad_checkpointing = False
        self.blocks = nn.Sequential(*[
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type,
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(
            dim=dim // 2,
            patch_size=merge_size,
            norm_layer=norm_layer,
        ) if use_merge else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(
            self,
            in_chans: int,
            embed_dim: int,
            patch_size: Union[int, Tuple[int, int]] = 4,
            norm_layer: LayerType = nn.BatchNorm2d,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size, bias=False)
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class PatchMerging(nn.Module):
    def __init__(
            self,
            dim: int,
            patch_size: Union[int, Tuple[int, int]] = 2,
            norm_layer: LayerType = nn.BatchNorm2d,
    ):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, patch_size, patch_size, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.reduction(x))


class FasterNet(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 96,
            depths: Union[int, Tuple[int, ...]] = (1, 2, 8, 2),
            mlp_ratio: float = 2.,
            n_div: int = 4,
            patch_size: Union[int, Tuple[int, int]] = 4,
            merge_size: Union[int, Tuple[int, int]] = 2,
            patch_norm: bool = True,
            feature_dim: int = 1280,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            layer_scale_init_value: float = 0.,
            act_layer: LayerType = partial(nn.ReLU, inplace=True),
            norm_layer: LayerType = nn.BatchNorm2d,
            pconv_fw_type: str = 'split_cat',
    ):
        super().__init__()
        assert pconv_fw_type in ('split_cat', 'slicing',)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        if not isinstance(depths, (list, tuple)):
            depths = (depths)  # it means the model has only one stage
        self.num_stages = len(depths)
        self.feature_info = []

        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
            norm_layer=norm_layer if patch_norm else nn.Identity,
        )
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i in range(self.num_stages):
            dim = int(embed_dim * 2 ** i)
            stage = Block(
                dim=dim,
                depth=depths[i],
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type,
                use_merge=False if i == 0 else True,
                merge_size=merge_size,
            )
            stages_list.append(stage)
            self.feature_info += [dict(num_chs=dim, reduction=2**(i+2), module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages_list)

        # building last several layers
        self.num_features = prev_chs = int(embed_dim * 2 ** (self.num_stages - 1))
        self.head_hidden_size = out_chs = feature_dim # 1280
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = nn.Conv2d(prev_chs, out_chs, 1, 1, 0, bias=False)
        self.act = act_layer()
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(out_chs, num_classes, bias=True) if num_classes > 0 else nn.Identity()
        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return set()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        matcher = dict(
            stem=r'^patch_embed',  # stem and embed
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+).downsample', (0,)),
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^conv_head', (99999,)),
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.classifier

    def reset_classifier(self, num_classes: int, global_pool: str = 'avg'):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.head_hidden_size, num_classes) if num_classes > 0 else nn.Identity()

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
        x = self.patch_embed(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]

        for feat_idx, stage in enumerate(stages):
            x = stage(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

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
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.stages(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    # if 'avgpool_pre_head' in state_dict:
    #     return state_dict
    #
    # out_dict = {
    #     'conv_head.weight': state_dict.pop('avgpool_pre_head.1.weight'),
    #     'classifier.weight': state_dict.pop('head.weight'),
    #     'classifier.bias': state_dict.pop('head.bias')
    # }
    #
    # stage_mapping = {
    #     'stages.1.': 'stages.1.downsample.',
    #     'stages.2.': 'stages.1.',
    #     'stages.3.': 'stages.2.downsample.',
    #     'stages.4.': 'stages.2.',
    #     'stages.5.': 'stages.3.downsample.',
    #     'stages.6.': 'stages.3.'
    # }
    #
    # for k, v in state_dict.items():
    #     for old_prefix, new_prefix in stage_mapping.items():
    #         if k.startswith(old_prefix):
    #             k = k.replace(old_prefix, new_prefix)
    #             break
    #     out_dict[k] = v
    return state_dict


def _cfg(url: str = '', **kwargs: Any) -> Dict[str, Any]:
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 1.0, 'interpolation': 'bicubic', 'test_crop_pct': 0.9,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'classifier',
        'paper_ids': 'arXiv:2303.03667',
        'paper_name': "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks",
        'origin_url': 'https://github.com/JierunChen/FasterNet',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'fasternet_t0.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_t0-epoch.281-val_acc1.71.9180.pth',
    ),
    'fasternet_t1.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_t1-epoch.291-val_acc1.76.2180.pth',
    ),
    'fasternet_t2.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_t2-epoch.289-val_acc1.78.8860.pth',
    ),
    'fasternet_s.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_s-epoch.299-val_acc1.81.2840.pth',
    ),
    'fasternet_m.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_m-epoch.291-val_acc1.82.9620.pth',
    ),
    'fasternet_l.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_l-epoch.299-val_acc1.83.5060.pth',
    ),
})


def _create_fasternet(variant: str, pretrained: bool = False, **kwargs: Any) -> FasterNet:
    model = build_model_with_cfg(
        FasterNet, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )
    return model


@register_model
def fasternet_t0(pretrained: bool = False, **kwargs: Any) -> FasterNet:
    model_args = dict(embed_dim=40, depths=(1, 2, 8, 2), drop_path_rate=0.0, act_layer=nn.GELU)
    return _create_fasternet('fasternet_t0', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def fasternet_t1(pretrained: bool = False, **kwargs: Any) -> FasterNet:
    model_args = dict(embed_dim=64, depths=(1, 2, 8, 2), drop_path_rate=0.02, act_layer=nn.GELU)
    return _create_fasternet('fasternet_t1', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def fasternet_t2(pretrained: bool = False, **kwargs: Any) -> FasterNet:
    model_args = dict(embed_dim=96, depths=(1, 2, 8, 2), drop_path_rate=0.05)
    return _create_fasternet('fasternet_t2', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def fasternet_s(pretrained: bool = False, **kwargs: Any) -> FasterNet:
    model_args = dict(embed_dim=128, depths=(1, 2, 13, 2), drop_path_rate=0.1)
    return _create_fasternet('fasternet_s', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def fasternet_m(pretrained: bool = False, **kwargs: Any) -> FasterNet:
    model_args = dict(embed_dim=144, depths=(3, 4, 18, 3), drop_path_rate=0.2)
    return _create_fasternet('fasternet_m', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def fasternet_l(pretrained: bool = False, **kwargs: Any) -> FasterNet:
    model_args = dict(embed_dim=192, depths=(3, 4, 18, 3), drop_path_rate=0.3)
    return _create_fasternet('fasternet_l', pretrained=pretrained, **dict(model_args, **kwargs))
