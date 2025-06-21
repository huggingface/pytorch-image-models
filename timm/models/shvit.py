"""SHViT
SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design
Code: https://github.com/ysj9909/SHViT
Paper: https://arxiv.org/abs/2401.16456

@inproceedings{yun2024shvit,
  author={Yun, Seokju and Ro, Youngmin},
  title={SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={5756--5767},
  year={2024}
}
"""
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import GroupNorm1, SqueezeExcite, SelectAdaptivePool2d, LayerType, trunc_normal_
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['SHViT']


class Residual(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.m(x)

    @torch.no_grad()
    def fuse(self) -> nn.Module:
        if isinstance(self.m, Conv2dNorm):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = F.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class Conv2dNorm(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            bn_weight_init: int = 1,
            **kwargs,
    ):
        super().__init__()
        self.add_module('c', nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs))
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            in_channels=w.size(1) * self.c.groups,
            out_channels=w.size(0),
            kernel_size=w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            device=c.weight.device,
            dtype=c.weight.dtype,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class NormLinear(nn.Sequential):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            std: float = 0.02,
    ):
        super().__init__()
        self.add_module('bn', nn.BatchNorm1d(in_features))
        self.add_module('l', nn.Linear(in_features, out_features, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self) -> nn.Linear:
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = nn.Linear(w.size(1), w.size(0), device=l.weight.device, dtype=l.weight.dtype)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int, act_layer: LayerType = nn.ReLU):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2dNorm(dim, hid_dim)
        self.act1 = act_layer()
        self.conv2 = Conv2dNorm(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.act2 = act_layer()
        self.se = SqueezeExcite(hid_dim, 0.25)
        self.conv3 = Conv2dNorm(hid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.se(x)
        x = self.conv3(x)
        return x


class FFN(nn.Module):
    def __init__(self, dim: int, embed_dim: int, act_layer: LayerType = nn.ReLU):
        super().__init__()
        self.pw1 = Conv2dNorm(dim, embed_dim)
        self.act = act_layer()
        self.pw2 = Conv2dNorm(embed_dim, dim, bn_weight_init=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x


class SHSA(nn.Module):
    """Single-Head Self-Attention"""
    def __init__(
            self,
            dim: int,
            qk_dim: int,
            pdim: int,
            norm_layer: LayerType = GroupNorm1,
            act_layer: LayerType = nn.ReLU,
    ):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = norm_layer(pdim)

        self.qkv = Conv2dNorm(pdim, qk_dim * 2 + pdim)
        self.proj = nn.Sequential(act_layer(), Conv2dNorm(dim, dim, bn_weight_init=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim = 1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = torch.split(qkv, [self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim = 1))
        return x


class BasicBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            qk_dim: int,
            pdim: int,
            type: str,
            norm_layer: LayerType = GroupNorm1,
            act_layer: LayerType = nn.ReLU,
    ):
        super().__init__()
        self.conv = Residual(Conv2dNorm(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
        if type == "s":
            self.mixer = Residual(SHSA(dim, qk_dim, pdim, norm_layer, act_layer))
        else:
            self.mixer = nn.Identity()
        self.ffn = Residual(FFN(dim, int(dim * 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.mixer(x)
        x = self.ffn(x)
        return x


class StageBlock(nn.Module):
    def __init__(
            self,
            prev_dim: int,
            dim: int,
            qk_dim: int,
            pdim: int,
            type: str,
            depth: int,
            norm_layer: LayerType = GroupNorm1,
            act_layer: LayerType = nn.ReLU,
    ):
        super().__init__()
        self.grad_checkpointing = False
        self.downsample = nn.Sequential(
            Residual(Conv2dNorm(prev_dim, prev_dim, 3, 1, 1, groups=prev_dim)),
            Residual(FFN(prev_dim, int(prev_dim * 2), act_layer)),
            PatchMerging(prev_dim, dim, act_layer),
            Residual(Conv2dNorm(dim, dim, 3, 1, 1, groups=dim)),
            Residual(FFN(dim, int(dim * 2), act_layer)),
        ) if prev_dim != dim else nn.Identity()

        self.blocks = nn.Sequential(*[
            BasicBlock(dim, qk_dim, pdim, type, norm_layer, act_layer) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class SHViT(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: Tuple[int, int, int] = (128, 256, 384),
            partial_dim: Tuple[int, int, int] = (32, 64, 96),
            qk_dim: Tuple[int, int, int] = (16, 16, 16),
            depth: Tuple[int, int, int] = (1, 2, 3),
            types: Tuple[str, str, str] = ("s", "s", "s"),
            drop_rate: float = 0.,
            norm_layer: LayerType = GroupNorm1,
            act_layer: LayerType = nn.ReLU,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        # Patch embedding
        stem_chs = embed_dim[0]
        self.patch_embed = nn.Sequential(
            Conv2dNorm(in_chans, stem_chs // 8, 3, 2, 1),
            act_layer(),
            Conv2dNorm(stem_chs // 8, stem_chs // 4, 3, 2, 1),
            act_layer(),
            Conv2dNorm(stem_chs // 4, stem_chs // 2, 3, 2, 1),
            act_layer(),
            Conv2dNorm(stem_chs // 2, stem_chs, 3, 2, 1)
        )

        # Build SHViT blocks
        stages = []
        prev_chs = stem_chs
        for i in range(len(embed_dim)):
            stages.append(StageBlock(
                prev_dim=prev_chs,
                dim=embed_dim[i],
                qk_dim=qk_dim[i],
                pdim=partial_dim[i],
                type=types[i],
                depth=depth[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
            ))
            prev_chs = embed_dim[i]
            self.feature_info.append(dict(num_chs=prev_chs, reduction=2**(i+4), module=f'stages.{i}'))
        self.stages = nn.Sequential(*stages)

        # Classifier head
        self.num_features = self.head_hidden_size = embed_dim[-1]
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.head = NormLinear(self.head_hidden_size, num_classes) if num_classes > 0 else nn.Identity()

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
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.l

    def reset_classifier(self, num_classes: int, global_pool: str = 'avg'):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.head = NormLinear(self.head_hidden_size, num_classes) if num_classes > 0 else nn.Identity()

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
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @torch.no_grad()
    def fuse(self):
        def fuse_children(net):
            for child_name, child in net.named_children():
                if hasattr(child, 'fuse'):
                    fused = child.fuse()
                    setattr(net, child_name, fused)
                    fuse_children(fused)
                else:
                    fuse_children(child)

        fuse_children(self)


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    state_dict = state_dict.get('model', state_dict)

    # out_dict = {}
    # import re
    # replace_rules = [
    #     (re.compile(r'^blocks1\.'), 'stages.0.blocks.'),
    #     (re.compile(r'^blocks2\.'), 'stages.1.blocks.'),
    #     (re.compile(r'^blocks3\.'), 'stages.2.blocks.'),
    # ]
    # downsample_mapping = {}
    # for i in range(1, 3):
    #     downsample_mapping[f'^stages\\.{i}\\.blocks\\.0\\.0\\.'] = f'stages.{i}.downsample.0.'
    #     downsample_mapping[f'^stages\\.{i}\\.blocks\\.0\\.1\\.'] = f'stages.{i}.downsample.1.'
    #     downsample_mapping[f'^stages\\.{i}\\.blocks\\.1\\.'] = f'stages.{i}.downsample.2.'
    #     downsample_mapping[f'^stages\\.{i}\\.blocks\\.2\\.0\\.'] = f'stages.{i}.downsample.3.'
    #     downsample_mapping[f'^stages\\.{i}\\.blocks\\.2\\.1\\.'] = f'stages.{i}.downsample.4.'
    #     for j in range(3, 10):
    #         downsample_mapping[f'^stages\\.{i}\\.blocks\\.{j}\\.'] = f'stages.{i}.blocks.{j - 3}.'
    #
    # downsample_patterns = [
    #     (re.compile(pattern), replacement) for pattern, replacement in downsample_mapping.items()]
    #
    # for k, v in state_dict.items():
    #     for pattern, replacement in replace_rules:
    #         k = pattern.sub(replacement, k)
    #     for pattern, replacement in downsample_patterns:
    #         k = pattern.sub(replacement, k)
    #     out_dict[k] = v

    return state_dict


def _cfg(url: str = '', **kwargs: Any) -> Dict[str, Any]:
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (4, 4),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.0.c', 'classifier': 'head.l',
        'paper_ids': 'arXiv:2401.16456',
        'paper_name': 'SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design',
        'origin_url': 'https://github.com/ysj9909/SHViT',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'shvit_s1.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s1.pth',
    ),
    'shvit_s2.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s2.pth',
    ),
    'shvit_s3.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s3.pth',
    ),
    'shvit_s4.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s4.pth',
        input_size=(3, 256, 256),
    ),
})


def _create_shvit(variant: str, pretrained: bool = False, **kwargs: Any) -> SHViT:
    model = build_model_with_cfg(
        SHViT, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2), flatten_sequential=True),
        **kwargs,
    )
    return model


@register_model
def shvit_s1(pretrained: bool = False, **kwargs: Any) -> SHViT:
    model_args = dict(
        embed_dim=(128, 224, 320), depth=(2, 4, 5), partial_dim=(32, 48, 68), types=("i", "s", "s"))
    return _create_shvit('shvit_s1', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def shvit_s2(pretrained: bool = False, **kwargs: Any) -> SHViT:
    model_args = dict(
        embed_dim=(128, 308, 448), depth=(2, 4, 5), partial_dim=(32, 66, 96), types=("i", "s", "s"))
    return _create_shvit('shvit_s2', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def shvit_s3(pretrained: bool = False, **kwargs: Any) -> SHViT:
    model_args = dict(
        embed_dim=(192, 352, 448), depth=(3, 5, 5), partial_dim=(48, 75, 96), types=("i", "s", "s"))
    return _create_shvit('shvit_s3', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def shvit_s4(pretrained: bool = False, **kwargs: Any) -> SHViT:
    model_args = dict(
        embed_dim=(224, 336, 448), depth=(4, 7, 6), partial_dim=(48, 72, 96), types=("i", "s", "s"))
    return _create_shvit('shvit_s4', pretrained=pretrained, **dict(model_args, **kwargs))
