"""SwiftFormer
SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications
Code: https://github.com/Amshaker/SwiftFormer
Paper: https://arxiv.org/pdf/2303.15446

@InProceedings{Shaker_2023_ICCV,
    author    = {Shaker, Abdelrahman and Maaz, Muhammad and Rasheed, Hanoona and Khan, Salman and Yang, Ming-Hsuan and Khan, Fahad Shahbaz},
    title     = {SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023},
}
"""
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, Linear, LayerType, to_2tuple, trunc_normal_
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['SwiftFormer']


class LayerScale2d(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(
            init_values * torch.ones(dim, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(
            self,
            in_chans: int = 3,
            embed_dim: int = 768,
            patch_size: int = 16,
            stride: int = 16,
            padding: int = 0,
            norm_layer: LayerType = nn.BatchNorm2d,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride, padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


class ConvEncoder(nn.Module):
    """
    Implementation of ConvEncoder with 3*3 and 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """
    def __init__(
            self,
            dim: int,
            hidden_dim: int = 64,
            kernel_size: int = 3,
            drop_path: float = 0.,
            act_layer: LayerType = nn.GELU,
            norm_layer: LayerType = nn.BatchNorm2d,
            use_layer_scale: bool = True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = norm_layer(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = act_layer()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale = LayerScale2d(dim, 1) if use_layer_scale else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.layer_scale(x)
        x = input + self.drop_path(x)
        return x


class Mlp(nn.Module):
    """
    Implementation of MLP layer with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: LayerType = nn.GELU,
            norm_layer: LayerType = nn.BatchNorm2d,
            drop: float = 0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = norm_layer(in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientAdditiveAttention(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """
    def __init__(self, in_dims: int = 512, token_dim: int = 256, num_heads: int = 1):
        super().__init__()
        self.scale_factor = token_dim ** -0.5
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))

        self.proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        query = F.normalize(self.to_query(x), dim=-1)
        key = F.normalize(self.to_key(x), dim=-1)

        attn = F.normalize(query @ self.w_g * self.scale_factor, dim=1)
        attn = torch.sum(attn * query, dim=1, keepdim=True)

        out = self.proj(attn * key) + query
        out = self.final(out).permute(0, 2, 1).reshape(B, -1, H, W)
        return out


class LocalRepresentation(nn.Module):
    """
    Local Representation module for SwiftFormer that is implemented by 3*3 depth-wise and point-wise convolutions.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """
    def __init__(
            self,
            dim: int,
            kernel_size: int = 3,
            drop_path: float = 0.,
            use_layer_scale: bool = True,
            act_layer: LayerType = nn.GELU,
            norm_layer: LayerType = nn.BatchNorm2d,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = norm_layer(dim)
        self.pwconv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = act_layer()
        self.pwconv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale = LayerScale2d(dim, 1) if use_layer_scale else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.layer_scale(x)
        x = skip + self.drop_path(x)
        return x


class Block(nn.Module):
    """
    SwiftFormer Encoder Block for SwiftFormer. It consists of :
    (1) Local representation module, (2) EfficientAdditiveAttention, and (3) MLP block.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """
    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 4.,
            drop_rate: float = 0.,
            drop_path: float = 0.,
            act_layer: LayerType = nn.GELU,
            norm_layer: LayerType = nn.BatchNorm2d,
            use_layer_scale: bool = True,
            layer_scale_init_value: float = 1e-5,
    ):
        super().__init__()
        self.local_representation = LocalRepresentation(
            dim=dim,
            use_layer_scale=use_layer_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.attn = EfficientAdditiveAttention(in_dims=dim, token_dim=dim)
        self.linear = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop=drop_rate,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale_1 = LayerScale2d(dim, layer_scale_init_value) \
            if use_layer_scale else nn.Identity()
        self.layer_scale_2 = LayerScale2d(dim, layer_scale_init_value) \
            if use_layer_scale else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_representation(x)
        x = x + self.drop_path(self.layer_scale_1(self.attn(x)))
        x = x + self.drop_path(self.layer_scale_2(self.linear(x)))
        return x


class Stage(nn.Module):
    """
    Implementation of each SwiftFormer stages. Here, SwiftFormerEncoder used as the last block in all stages, while ConvEncoder used in the rest of the blocks.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """
    def __init__(
            self,
            dim: int,
            index: int,
            layers: List[int],
            mlp_ratio: float = 4.,
            act_layer: LayerType = nn.GELU,
            norm_layer: LayerType = nn.BatchNorm2d,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            use_layer_scale: bool = True,
            layer_scale_init_value: float = 1e-5,
            downsample: Optional[LayerType] = None,
    ):
        super().__init__()
        self.grad_checkpointing = False
        self.downsample = downsample if downsample is not None else nn.Identity()

        blocks = []
        for block_idx in range(layers[index]):
            block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
            if layers[index] - block_idx <= 1:
                blocks.append(Block(
                    dim,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path=block_dpr,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                ))
            else:
                blocks.append(ConvEncoder(
                    dim=dim,
                    hidden_dim=int(mlp_ratio * dim),
                    kernel_size=3,
                    drop_path=block_dpr,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    use_layer_scale=use_layer_scale,
                ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class SwiftFormer(nn.Module):
    def __init__(
            self,
            layers: List[int] = [3, 3, 6, 4],
            embed_dims: List[int] = [48, 56, 112, 220],
            mlp_ratios: int = 4,
            downsamples: List[bool] = [False, True, True, True],
            act_layer: LayerType = nn.GELU,
            down_patch_size: int = 3,
            down_stride: int = 2,
            down_pad: int = 1,
            num_classes: int = 1000,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            use_layer_scale: bool = True,
            layer_scale_init_value: float = 1e-5,
            global_pool: str = 'avg',
            output_stride: int = 32,
            in_chans: int = 3,
            **kwargs,
    ):
        super().__init__()
        assert output_stride == 32
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.feature_info = []

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims[0] // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dims[0] // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dims[0] // 2, embed_dims[0], 3, 2, 1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(),
        )
        prev_dim = embed_dims[0]

        stages = []
        for i in range(len(layers)):
            downsample = Embedding(
                in_chans=prev_dim,
                embed_dim=embed_dims[i],
                patch_size=down_patch_size,
                stride=down_stride,
                padding=down_pad,
            ) if downsamples[i] else nn.Identity()
            stage = Stage(
                dim=embed_dims[i],
                index=i,
                layers=layers,
                mlp_ratio=mlp_ratios,
                act_layer=act_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                downsample=downsample,
            )
            prev_dim = embed_dims[i]
            stages.append(stage)
            self.feature_info += [dict(num_chs=embed_dims[i], reduction=2**(i+2), module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        # Classifier head
        self.num_features  = self.head_hidden_size = out_chs = embed_dims[-1]
        self.norm = nn.BatchNorm2d(out_chs)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = Linear(out_chs, num_classes) if num_classes > 0 else nn.Identity()
        # assuming model is always distilled (valid for current checkpoints, will split def if that changes)
        self.head_dist = Linear(out_chs, num_classes) if num_classes > 0 else nn.Identity()
        self.distilled_training = False  # must set this True to train w/ distillation token
        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
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
            stem=r'^stem',  # stem and embed
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+).downsample', (0,)),
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> Tuple[nn.Module, nn.Module]:
        return self.head, self.head_dist

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable: bool = True):
        self.distilled_training = enable

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
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=(2, 3))
        x = self.head_drop(x)
        if pre_logits:
            return x
        x, x_dist = self.head(x), self.head_dist(x)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train/finetune, inference average the classifier predictions
            return (x + x_dist) / 2

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    state_dict = state_dict.get('model', state_dict)
    if 'stem.0.weight' in state_dict:
        return state_dict

    out_dict = {}
    for k, v in state_dict.items():
        k = k.replace('patch_embed.', 'stem.')
        k = k.replace('dist_head.', 'head_dist.')
        k = k.replace('attn.Proj.', 'attn.proj.')
        k = k.replace('.layer_scale_1', '.layer_scale_1.gamma')
        k = k.replace('.layer_scale_2', '.layer_scale_2.gamma')
        k = re.sub(r'\.layer_scale(?=$|\.)', '.layer_scale.gamma', k)
        m = re.match(r'^network\.(\d+)\.(.*)', k)
        if m:
            n_idx, rest = int(m.group(1)), m.group(2)
            stage_idx = n_idx // 2
            if n_idx % 2 == 0:
                k = f'stages.{stage_idx}.blocks.{rest}'
            else:
                k = f'stages.{stage_idx+1}.downsample.{rest}'

        out_dict[k] = v
    return out_dict


def _cfg(url: str = '', **kwargs: Any) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None, 'fixed_input_size': True,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': ('head', 'head_dist'),
        'paper_ids': 'arXiv:2303.15446',
        'paper_name': 'SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications',
        'origin_url': 'https://github.com/Amshaker/SwiftFormer',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'swiftformer_xs.dist_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'swiftformer_s.dist_in1k': _cfg(
        hf_hub_id='timm/'
    ),
    'swiftformer_l1.dist_in1k': _cfg(
        hf_hub_id='timm/'
    ),
    'swiftformer_l3.dist_in1k': _cfg(
        hf_hub_id='timm/'
    ),
})


def _create_swiftformer(variant: str, pretrained: bool = False, **kwargs: Any) -> SwiftFormer:
    model = build_model_with_cfg(
        SwiftFormer, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )
    return model


@register_model
def swiftformer_xs(pretrained: bool = False, **kwargs: Any) -> SwiftFormer:
    model_args = dict(layers=[3, 3, 6, 4], embed_dims=[48, 56, 112, 220])
    return _create_swiftformer('swiftformer_xs', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swiftformer_s(pretrained: bool = False, **kwargs: Any) -> SwiftFormer:
    model_args = dict(layers=[3, 3, 9, 6], embed_dims=[48, 64, 168, 224])
    return _create_swiftformer('swiftformer_s', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def swiftformer_l1(pretrained: bool = False, **kwargs: Any) -> SwiftFormer:
    model_args = dict(layers=[4, 3, 10, 5], embed_dims=[48, 96, 192, 384])
    return _create_swiftformer('swiftformer_l1', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swiftformer_l3(pretrained: bool = False, **kwargs: Any) -> SwiftFormer:
    model_args = dict(layers=[4, 4, 12, 6], embed_dims=[64, 128, 320, 512])
    return _create_swiftformer('swiftformer_l3', pretrained=pretrained, **dict(model_args, **kwargs))