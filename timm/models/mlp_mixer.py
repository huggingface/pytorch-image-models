""" MLP-Mixer, ResMLP, and gMLP in PyTorch

This impl originally based on MLP-Mixer paper.

Official JAX impl: https://github.com/google-research/vision_transformer/blob/linen/vit_jax/models_mixer.py

Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601

@article{tolstikhin2021,
  title={MLP-Mixer: An all-MLP Architecture for Vision},
  author={Tolstikhin, Ilya and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner,
        Thomas and Yung, Jessica and Keysers, Daniel and Uszkoreit, Jakob and Lucic, Mario and Dosovitskiy, Alexey},
  journal={arXiv preprint arXiv:2105.01601},
  year={2021}
}

Also supporting ResMlp, and a preliminary (not verified) implementations of gMLP

Code: https://github.com/facebookresearch/deit
Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
@misc{touvron2021resmlp,
      title={ResMLP: Feedforward networks for image classification with data-efficient training},
      author={Hugo Touvron and Piotr Bojanowski and Mathilde Caron and Matthieu Cord and Alaaeldin El-Nouby and
        Edouard Grave and Armand Joulin and Gabriel Synnaeve and Jakob Verbeek and Hervé Jégou},
      year={2021},
      eprint={2105.03404},
}

Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
@misc{liu2021pay,
      title={Pay Attention to MLPs},
      author={Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},
      year={2021},
      eprint={2105.08050},
}

A thank you to paper authors for releasing code and weights.

Hacked together by / Copyright 2021 Ross Wightman
"""
import math
from functools import partial
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, Mlp, GluMlp, GatedMlp, DropPath, lecun_normal_, to_2tuple
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import named_apply, checkpoint, checkpoint_seq
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['MixerBlock', 'MlpMixer']  # model_registry will add each entrypoint fn to this


class MixerBlock(nn.Module):
    """Residual Block w/ token mixing and channel MLPs.

    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self,
            dim: int,
            seq_len: int,
            mlp_ratio: Union[float, Tuple[float, float]] = (0.5, 4.0),
            mlp_layer: type = Mlp,
            norm_layer: type = partial(nn.LayerNorm, eps=1e-6),
            act_layer: type = nn.GELU,
            drop: float = 0.,
            drop_path: float = 0.,
    ) -> None:
        """Initialize MixerBlock.

        Args:
            dim: Dimension of input features.
            seq_len: Sequence length.
            mlp_ratio: Expansion ratios for token mixing and channel MLPs.
            mlp_layer: MLP layer class.
            norm_layer: Normalization layer.
            act_layer: Activation layer.
            drop: Dropout rate.
            drop_path: Drop path rate.
        """
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class Affine(nn.Module):
    """Affine transformation layer."""

    def __init__(self, dim: int) -> None:
        """Initialize Affine layer.

        Args:
            dim: Dimension of features.
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply affine transformation."""
        return torch.addcmul(self.beta, self.alpha, x)


class ResBlock(nn.Module):
    """Residual MLP block w/ LayerScale and Affine 'norm'.

    Based on: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    def __init__(
            self,
            dim: int,
            seq_len: int,
            mlp_ratio: float = 4,
            mlp_layer: type = Mlp,
            norm_layer: type = Affine,
            act_layer: type = nn.GELU,
            init_values: float = 1e-4,
            drop: float = 0.,
            drop_path: float = 0.,
    ) -> None:
        """Initialize ResBlock.

        Args:
            dim: Dimension of input features.
            seq_len: Sequence length.
            mlp_ratio: Channel MLP expansion ratio.
            mlp_layer: MLP layer class.
            norm_layer: Normalization layer.
            act_layer: Activation layer.
            init_values: Initial values for layer scale.
            drop: Dropout rate.
            drop_path: Drop path rate.
        """
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.linear_tokens = nn.Linear(seq_len, seq_len)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, drop=drop)
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        return x


class SpatialGatingUnit(nn.Module):
    """Spatial Gating Unit.

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(self, dim: int, seq_len: int, norm_layer: type = nn.LayerNorm) -> None:
        """Initialize Spatial Gating Unit.

        Args:
            dim: Dimension of input features.
            seq_len: Sequence length.
            norm_layer: Normalization layer.
        """
        super().__init__()
        gate_dim = dim // 2
        self.norm = norm_layer(gate_dim)
        self.proj = nn.Linear(seq_len, seq_len)

    def init_weights(self) -> None:
        """Initialize weights for projection gate."""
        # special init for the projection gate, called as override by base model init
        nn.init.normal_(self.proj.weight, std=1e-6)
        nn.init.ones_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial gating."""
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(-1, -2))
        return u * v.transpose(-1, -2)


class SpatialGatingBlock(nn.Module):
    """Residual Block w/ Spatial Gating.

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(
            self,
            dim: int,
            seq_len: int,
            mlp_ratio: float = 4,
            mlp_layer: type = GatedMlp,
            norm_layer: type = partial(nn.LayerNorm, eps=1e-6),
            act_layer: type = nn.GELU,
            drop: float = 0.,
            drop_path: float = 0.,
    ) -> None:
        """Initialize SpatialGatingBlock.

        Args:
            dim: Dimension of input features.
            seq_len: Sequence length.
            mlp_ratio: Channel MLP expansion ratio.
            mlp_layer: MLP layer class.
            norm_layer: Normalization layer.
            act_layer: Activation layer.
            drop: Dropout rate.
            drop_path: Drop path rate.
        """
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm = norm_layer(dim)
        sgu = partial(SpatialGatingUnit, seq_len=seq_len)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, gate_layer=sgu, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        return x


class MlpMixer(nn.Module):
    """MLP-Mixer model architecture.

    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """

    def __init__(
            self,
            num_classes: int = 1000,
            img_size: int = 224,
            in_chans: int = 3,
            patch_size: int = 16,
            num_blocks: int = 8,
            embed_dim: int = 512,
            mlp_ratio: Union[float, Tuple[float, float]] = (0.5, 4.0),
            block_layer: type = MixerBlock,
            mlp_layer: type = Mlp,
            norm_layer: type = partial(nn.LayerNorm, eps=1e-6),
            act_layer: type = nn.GELU,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            nlhb: bool = False,
            stem_norm: bool = False,
            global_pool: str = 'avg',
    ) -> None:
        """Initialize MLP-Mixer.

        Args:
            num_classes: Number of classes for classification.
            img_size: Input image size.
            in_chans: Number of input channels.
            patch_size: Patch size.
            num_blocks: Number of mixer blocks.
            embed_dim: Embedding dimension.
            mlp_ratio: MLP expansion ratio(s).
            block_layer: Block layer class.
            mlp_layer: MLP layer class.
            norm_layer: Normalization layer.
            act_layer: Activation layer.
            drop_rate: Head dropout rate.
            proj_drop_rate: Projection dropout rate.
            drop_path_rate: Drop path rate.
            nlhb: Use negative log bias initialization.
            stem_norm: Apply normalization to stem.
            global_pool: Global pooling type.
        """
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.grad_checkpointing = False

        self.stem = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if stem_norm else None,
        )
        reduction = self.stem.feat_ratio() if hasattr(self.stem, 'feat_ratio') else patch_size
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim,
                self.stem.num_patches,
                mlp_ratio,
                mlp_layer=mlp_layer,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop=proj_drop_rate,
                drop_path=drop_path_rate,
            )
            for _ in range(num_blocks)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(num_blocks)]
        self.norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(nlhb=nlhb)

    @torch.jit.ignore
    def init_weights(self, nlhb: bool = False) -> None:
        """Initialize model weights.

        Args:
            nlhb: Use negative log bias initialization for head.
        """
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        """Create regex patterns for parameter grouping.

        Args:
            coarse: Use coarse grouping.

        Returns:
            Dictionary mapping group names to regex patterns.
        """
        return dict(
            stem=r'^stem',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing.

        Args:
            enable: Whether to enable gradient checkpointing.
        """
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """Get the classifier module."""
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Reset the classifier head.

        Args:
            num_classes: Number of classes for new classifier.
            global_pool: Global pooling type.
        """
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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
            norm: Apply norm layer to all intermediates.
            stop_early: Stop iterating over blocks when last desired intermediate hit.
            output_fmt: Shape of intermediate feature outputs ('NCHW' or 'NLC').
            intermediates_only: Only return intermediate features.

        Returns:
            List of intermediate features or tuple of (final features, intermediates).
        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width = x.shape
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
                # normalize intermediates with final norm layer if enabled
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if reshape:
            # reshape to BCHW output format
            H, W = self.stem.dynamic_feat_size((height, width))
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        if intermediates_only:
            return intermediates

        x = self.norm(x)

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
            prune_norm: Whether to prune normalization layer.
            prune_head: Whether to prune the classifier head.

        Returns:
            List of indices that were kept.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction layers."""
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass through classifier head.

        Args:
            x: Feature tensor.
            pre_logits: Return features before final classifier.

        Returns:
            Output tensor.
        """
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax: bool = False) -> None:
    """Mixer weight initialization (trying to match Flax defaults).

    Args:
        module: Module to initialize.
        name: Module name.
        head_bias: Bias value for head layer.
        flax: Use Flax-style initialization.
    """
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
                # like MLP init in vit (my original init)
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
    elif hasattr(module, 'init_weights'):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in depth-first order.
        module.init_weights()


def checkpoint_filter_fn(state_dict, model):
    """ Remap checkpoints if needed """
    if 'patch_embed.proj.weight' in state_dict:
        # Remap FB ResMlp models -> timm
        out_dict = {}
        for k, v in state_dict.items():
            k = k.replace('patch_embed.', 'stem.')
            k = k.replace('attn.', 'linear_tokens.')
            k = k.replace('mlp.', 'mlp_channels.')
            k = k.replace('gamma_', 'ls')
            if k.endswith('.alpha') or k.endswith('.beta'):
                v = v.reshape(1, 1, -1)
            out_dict[k] = v
        return out_dict
    return state_dict


def _create_mixer(variant, pretrained=False, **kwargs) -> MlpMixer:
    out_indices = kwargs.pop('out_indices', 3)
    model = build_model_with_cfg(
        MlpMixer,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model


def _cfg(url='', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 0.875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'stem.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'mixer_s32_224.untrained': _cfg(),
    'mixer_s16_224.untrained': _cfg(),
    'mixer_b32_224.untrained': _cfg(),
    'mixer_b16_224.goog_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224-76587d61.pth',
    ),
    'mixer_b16_224.goog_in21k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224_in21k-617b3de2.pth',
        num_classes=21843
    ),
    'mixer_l32_224.untrained': _cfg(),
    'mixer_l16_224.goog_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224-92f9adc4.pth',
    ),
    'mixer_l16_224.goog_in21k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224_in21k-846aa33c.pth',
        num_classes=21843
    ),

    # Mixer ImageNet-21K-P pretraining
    'mixer_b16_224.miil_in21k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mixer_b16_224_miil_in21k-2a558a71.pth',
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'mixer_b16_224.miil_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mixer_b16_224_miil-9229a591.pth',
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_pct=0.875, interpolation='bilinear',
    ),

    'gmixer_12_224.untrained': _cfg(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'gmixer_24_224.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmixer_24_224_raa-7daf7ae6.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    'resmlp_12_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_12_no_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'resmlp_24_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_24_no_dist.pth',
        #url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resmlp_24_224_raa-a8256759.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'resmlp_36_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_36_no_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'resmlp_big_24_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlpB_24_no_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    'resmlp_12_224.fb_distilled_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_12_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'resmlp_24_224.fb_distilled_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_24_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'resmlp_36_224.fb_distilled_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_36_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'resmlp_big_24_224.fb_distilled_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlpB_24_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    'resmlp_big_24_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlpB_24_22k.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    'resmlp_12_224.fb_dino': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_12_dino.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'resmlp_24_224.fb_dino': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/resmlp_24_dino.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    'gmlp_ti16_224.untrained': _cfg(),
    'gmlp_s16_224.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmlp_s16_224_raa-10536d42.pth',
    ),
    'gmlp_b16_224.untrained': _cfg(),
})


@register_model
def mixer_s32_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Mixer-S/32 224x224
    Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=8, embed_dim=512, **kwargs)
    model = _create_mixer('mixer_s32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_s16_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Mixer-S/16 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512, **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b32_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Mixer-B/32 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b16_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_l32_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Mixer-L/32 224x224.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer('mixer_l32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_l16_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Mixer-L/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer('mixer_l16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmixer_12_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Glu-Mixer-12 224x224
    Experiment by Ross Wightman, adding SwiGLU to MLP-Mixer
    """
    model_args = dict(
        patch_size=16, num_blocks=12, embed_dim=384, mlp_ratio=(1.0, 4.0),
        mlp_layer=GluMlp, act_layer=nn.SiLU, **kwargs)
    model = _create_mixer('gmixer_12_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmixer_24_224(pretrained=False, **kwargs) -> MlpMixer:
    """ Glu-Mixer-24 224x224
    Experiment by Ross Wightman, adding SwiGLU to MLP-Mixer
    """
    model_args = dict(
        patch_size=16, num_blocks=24, embed_dim=384, mlp_ratio=(1.0, 4.0),
        mlp_layer=GluMlp, act_layer=nn.SiLU, **kwargs)
    model = _create_mixer('gmixer_24_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_12_224(pretrained=False, **kwargs) -> MlpMixer:
    """ ResMLP-12
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16, num_blocks=12, embed_dim=384, mlp_ratio=4, block_layer=ResBlock, norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_12_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_24_224(pretrained=False, **kwargs) -> MlpMixer:
    """ ResMLP-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16, num_blocks=24, embed_dim=384, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-5), norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_24_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_36_224(pretrained=False, **kwargs) -> MlpMixer:
    """ ResMLP-36
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16, num_blocks=36, embed_dim=384, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-6), norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_36_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_big_24_224(pretrained=False, **kwargs) -> MlpMixer:
    """ ResMLP-B-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=8, num_blocks=24, embed_dim=768, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-6), norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_big_24_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmlp_ti16_224(pretrained=False, **kwargs) -> MlpMixer:
    """ gMLP-Tiny
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=128, mlp_ratio=6, block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_ti16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmlp_s16_224(pretrained=False, **kwargs) -> MlpMixer:
    """ gMLP-Small
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=256, mlp_ratio=6, block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_s16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmlp_b16_224(pretrained=False, **kwargs) -> MlpMixer:
    """ gMLP-Base
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=512, mlp_ratio=6, block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_b16_224', pretrained=pretrained, **model_args)
    return model


register_model_deprecations(__name__, {
    'mixer_b16_224_in21k': 'mixer_b16_224.goog_in21k_ft_in1k',
    'mixer_l16_224_in21k': 'mixer_l16_224.goog_in21k_ft_in1k',
    'mixer_b16_224_miil': 'mixer_b16_224.miil_in21k_ft_in1k',
    'mixer_b16_224_miil_in21k': 'mixer_b16_224.miil_in21k',
    'resmlp_12_distilled_224': 'resmlp_12_224.fb_distilled_in1k',
    'resmlp_24_distilled_224': 'resmlp_24_224.fb_distilled_in1k',
    'resmlp_36_distilled_224': 'resmlp_36_224.fb_distilled_in1k',
    'resmlp_big_24_distilled_224': 'resmlp_big_24_224.fb_distilled_in1k',
    'resmlp_big_24_224_in22ft1k': 'resmlp_big_24_224.fb_in22k_ft_in1k',
    'resmlp_12_224_dino': 'resmlp_12_224',
    'resmlp_24_224_dino': 'resmlp_24_224',
})
