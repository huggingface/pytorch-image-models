"""iFormer
iFormer: Integrating ConvNet and Transformer for Mobile Application (ICLR 2025)
- paper: https://arxiv.org/abs/2501.15369
- code: https://github.com/ChuanyangZheng/iFormer
@inproceedings{zheng2025iformer,
  title={iformer: Integrating convnet and transformer for mobile application},
  author={Zheng, Chuanyang},
  booktitle={International Conference on Learning Representations},
  volume={2025},
  pages={22947--22961},
  year={2025}
}
Modifications by / Copyright 2026 Ryan Hou & Ross Wightman, original copyrights below
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, calculate_drop_path_rates, get_device_dtype, trunc_normal_
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ["iFormer"]


class ConvNorm(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            bn_weight_init: float = 1,
            device=None,
            dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.add_module(
            "c",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
                **dd,
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_channels, **dd))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
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
            device=None,
            dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.add_module("bn", nn.BatchNorm1d(in_features, **dd))
        self.add_module("l", nn.Linear(in_features, out_features, bias=bias, **dd))
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


class RepVitClassifier(nn.Module):
    def __init__(
            self,
            dim: int,
            num_classes: int,
            distillation: bool = False,
            drop: float = 0.0,
            device=None,
            dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.head_drop = nn.Dropout(drop)
        self.head = NormLinear(dim, num_classes, **dd) if num_classes > 0 else nn.Identity()
        self.distillation = distillation
        self.distilled_training = False
        self.num_classes = num_classes
        if distillation:
            self.head_dist = (
                NormLinear(dim, num_classes, **dd) if num_classes > 0 else nn.Identity()
            )
        else:
            self.head_dist = nn.Identity()

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.head_drop(x)
        if self.distillation:
            x1, x2 = self.head(x), self.head_dist(x)
            if self.training and self.distilled_training and not torch.jit.is_scripting():
                return x1, x2
            else:
                return (x1 + x2) / 2
        else:
            x = self.head(x)
            return x

    @torch.no_grad()
    def fuse(self) -> nn.Module:
        if not self.num_classes > 0:
            return nn.Identity()
        head = self.head.fuse()
        if self.distillation:
            head_dist = self.head_dist.fuse()
            head.weight += head_dist.weight
            head.bias += head_dist.bias
            head.weight /= 2
            head.bias /= 2
            return head
        else:
            return head


class SHMA(nn.Module):
    """Single-Head Modulation Attention — core of iFormer."""

    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            ratio: int = 1,
            head_dim_reduction_ratio: int = 2,
            num_heads: int = 1,
            window_size: int = 0,
            fused_attn: bool = False,
            device=None,
            dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        mid_dim = int(dim * ratio)
        dim_attn = dim // head_dim_reduction_ratio
        self.dim_head = dim_attn // num_heads
        self.scale = self.dim_head**-0.5
        self.fused_attn = fused_attn
        self.window_size = window_size
        self.q = ConvNorm(dim, dim_attn, 1, 1, 0, **dd)
        self.k = ConvNorm(dim, dim_attn, 1, 1, 0, **dd)
        self.v_gate = ConvNorm(dim, 2 * mid_dim, 1, 1, 0, **dd)
        self.gate_act = nn.Sigmoid()
        self.attn_drop = nn.Dropout(0.0)
        self.proj = ConvNorm(mid_dim, dim, 1, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        do_window = self.window_size and self.window_size > 0
        if do_window:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x_pad = F.pad(x, (0, pad_r, 0, pad_b))
            Ho, Wo = H, W
            _, _, Hp, Wp = x_pad.shape
            x = self._window_partition(x_pad, self.window_size)
            B, _, H, W = x.shape
        else:
            Ho = Wo = Hp = Wp = pad_r = pad_b = 0

        v, gate = self.gate_act(self.v_gate(x)).chunk(2, dim=1)
        q = self.q(x).flatten(2)
        k = self.k(x).flatten(2)
        v = v.flatten(2)

        if self.fused_attn:
            q_t = q.transpose(-1, -2).contiguous()
            k_t = k.transpose(-1, -2).contiguous()
            v_t = v.transpose(-1, -2).contiguous()
            x_attn = (
                F.scaled_dot_product_attention(
                    q_t,
                    k_t,
                    v_t,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
                .transpose(-1, -2)
                .reshape(B, -1, H, W)
            )
        else:
            q_s = q * self.scale
            attn = (q_s.transpose(-2, -1) @ k).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)

        x_out = self.proj(x_attn * gate)
        if do_window:
            x_out = self._window_reverse(x_out, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x_out = x_out[:, :, :Ho, :Wo].contiguous()
        return x_out

    @staticmethod
    def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
        return x

    @staticmethod
    def _window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
        _, C, _, _ = windows.shape
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, C, window_size, window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
        return x


class Residual(nn.Module):
    def __init__(
            self,
            module: nn.Module,
            drop_path: float = 0.0,
            layer_scale_init_value: float = 0,
            dim: Optional[int] = None,
            device=None,
            dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.module = module
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if layer_scale_init_value > 0 and dim is not None:

            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones((1, dim, 1, 1), **dd),
                requires_grad=True,
            )
        else:
            self.gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma is not None:
            return x + self.gamma * self.drop_path(self.module(x))
        return x + self.drop_path(self.module(x))


class SHMABlock(nn.Module):
    def __init__(
            self,
            dim: int,
            ratio: int = 1,
            head_dim_reduction_ratio: int = 2,
            drop_path: float = 0.0,
            layer_scale_init_value: float = 1e-6,
            device=None,
            dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.mixer = Residual(
            SHMA(dim, ratio, head_dim_reduction_ratio, **dd),
            drop_path,
            layer_scale_init_value,
            dim,
            **dd,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(x)


class FFN(nn.Module):
    def __init__(
            self,
            dim: int,
            ratio: int = 4,
            drop_path: float = 0.0,
            layer_scale_init_value: float = 1e-6,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        mid_channels = ratio * dim
        self.mixer = Residual(
            nn.Sequential(
                ConvNorm(dim, mid_channels, 1, **dd),
                act_layer(),
                ConvNorm(mid_channels, dim, 1, **dd),
            ),
            drop_path,
            layer_scale_init_value,
            dim,
            **dd,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(x)


class ConvBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            kernel_size: int = 7,
            ratio: int = 4,
            drop_path: float = 0.0,
            layer_scale_init_value: float = 1e-6,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        mid_channels = ratio * dim
        self.mixer = Residual(
            nn.Sequential(
                ConvNorm(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim, **dd),
                ConvNorm(dim, mid_channels, 1, **dd),
                act_layer(),
                ConvNorm(mid_channels, dim, 1, **dd),
            ),
            drop_path,
            layer_scale_init_value,
            dim,
            **dd,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(x)


class RepCPE(nn.Module):
    """Conditional Positional Encoding — depthwise residual."""

    def __init__(
            self,
            dim: int,
            kernel_size: int = 7,
            device=None,
            dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.cpe = Residual(
            ConvNorm(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim, **dd),
            **dd,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cpe(x)


class StageBlock(nn.Module):
    def __init__(
            self,
            in_dim: int,
            dim: int,
            depth: int,
            num_attn: int,
            hdrr: int,
            conv_ratio: int,
            ffn_ratio: int,
            attn_ratio: int,
            drop_path_rates: List[float],
            layer_scale_init_value: float,
            act_layer: Type[nn.Module],
            device=None,
            dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.grad_checkpointing = False
        self.downsample = nn.Identity() if in_dim == dim else ConvNorm(in_dim, dim, 3, 2, 1, **dd)

        blocks = []
        if num_attn == 0:
            for j in range(depth):
                blocks.append(
                    ConvBlock(dim, 7, conv_ratio, drop_path_rates[j], layer_scale_init_value, act_layer, **dd)
                )
        else:
            num_conv = depth - num_attn * 3
            num_prefix, num_suffix = (num_conv - 1, 1) if num_conv > 0 else (0, 0)
            for j in range(num_prefix):
                blocks.append(
                    ConvBlock(dim, 7, conv_ratio, drop_path_rates[j], layer_scale_init_value, act_layer, **dd)
                )
            for g in range(num_attn):
                offset = num_prefix + g * 3
                blocks.append(RepCPE(dim, 3, **dd))
                blocks.append(
                    SHMABlock(dim, attn_ratio, hdrr, drop_path_rates[offset + 1], layer_scale_init_value, **dd)
                )
                blocks.append(
                    FFN(dim, ffn_ratio, drop_path_rates[offset + 2], layer_scale_init_value, act_layer, **dd)
                )
            if num_suffix:
                blocks.append(
                    ConvBlock(dim, 7, conv_ratio, drop_path_rates[-1], layer_scale_init_value, act_layer, **dd)
                )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class Stem(nn.Module):
    def __init__(
            self,
            in_chans: int,
            out_chs: int,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        mid_channels = (out_chs // 2) * 4
        self.conv1 = ConvNorm(in_chans, out_chs // 2, 5, 2, 2, **dd)
        self.act1 = act_layer()
        self.conv2 = ConvNorm(out_chs // 2, mid_channels, 5, 2, 2, **dd)
        self.act2 = act_layer()
        self.conv3 = ConvNorm(mid_channels, out_chs, 1, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x


class iFormer(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = "avg",
            dims: Tuple[int, ...] = (32, 64, 128, 256),
            depths: Tuple[int, ...] = (2, 2, 16, 6),
            attn_groups: Tuple[int, ...] = (0, 0, 3, 2),
            attn_head_dim_reduction: Tuple[int, ...] = (0, 0, 2, 4),
            conv_ratio: int = 3,
            ffn_ratio: int = 2,
            attn_ratio: int = 1,
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            layer_scale_init_value: float = 0.0,
            act_layer: Type[nn.Module] = nn.GELU,
            distillation: bool = False,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.global_pool = global_pool
        self.distillation = distillation
        if not isinstance(depths, (list, tuple)):
            depths = (depths)  # it means the model has only one stage
        self.num_stages = len(depths)
        self.feature_info = []

        self.stem = Stem(in_chans, dims[0], act_layer, **dd)
        prev_dim = dims[0]

        dpr = calculate_drop_path_rates(drop_path_rate, depths, stagewise=True)

        stages = []
        for i in range(self.num_stages):
            stage = StageBlock(
                in_dim=prev_dim,
                dim=dims[i],
                depth=depths[i],
                num_attn=attn_groups[i],
                hdrr=attn_head_dim_reduction[i],
                conv_ratio=conv_ratio,
                ffn_ratio=ffn_ratio,
                attn_ratio=attn_ratio,
                drop_path_rates=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
                act_layer=act_layer,
                **dd,
            )
            prev_dim = dims[i]
            stages.append(stage)
            self.feature_info += [dict(num_chs=dims[i], reduction=2**(i+2), module=f"stages.{i}")]
        self.stages = nn.Sequential(*stages)

        self.num_features = self.head_hidden_size = dims[-1]
        self.head_drop = nn.Dropout(drop_rate)
        self.head = RepVitClassifier(dims[-1], num_classes, distillation, 0.0, **dd)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return set()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        matcher = dict(
            stem=r"^stem",
            blocks=(
                r"^stages\.(\d+)"
                if coarse
                else [
                    (r"^stages\.(\d+)\.downsample", (0,)),
                    (r"^stages\.(\d+)\.blocks\.(\d+)", None),
                    (r"^head", (99999,)),
                ]
            ),
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        for stage in self.stages:
            stage.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(
            self,
            num_classes: int,
            global_pool: Optional[str] = None,
            distillation: bool = False,
            device=None,
            dtype=None,
    ):
        dd = get_device_dtype(self, device=device, dtype=dtype)
        self.num_classes = num_classes
        self.distillation = distillation
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = RepVitClassifier(self.head_hidden_size, num_classes, distillation, **dd)
        self.head.train(self.training)

    @torch.jit.ignore
    def set_distilled_training(self, enable: bool = True):
        self.head.distilled_training = enable

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = "NCHW",
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor.
            indices: Take last n blocks if int, all if None, select matching indices if sequence.
            norm: Apply norm layer to compatible intermediates.
            stop_early: Stop iterating over blocks when last desired intermediate hit.
            output_fmt: Shape of intermediate feature outputs.
            intermediates_only: Only return intermediate features.

        Returns:
            List of intermediate features or tuple of (final features, intermediates).
        """
        assert output_fmt in ("NCHW",), "Output shape must be NCHW."
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages), indices)

        # forward pass
        x = self.stem(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[: max_index + 1]

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
    ) -> List[int]:
        """Prune layers not required for specified intermediates.
        
        Args:
            indices: Indices of intermediate layers to keep.
            prune_norm: Whether to prune normalization layer.
            prune_head: Whether to prune the classifier head.

        Returns:
            List of indices that were kept.
        """
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        self.stages = self.stages[: max_index + 1]  # truncate blocks w/ stem as idx 0
        if prune_head:
            self.reset_classifier(0, "")
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward_head(
            self, x: torch.Tensor, pre_logits: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert self.global_pool in ("avg", ""), f"Unsupported global_pool {self.global_pool}"
        if self.global_pool == "avg":
            x = x.mean((2, 3), keepdim=False)
        x = self.head_drop(x)
        if pre_logits:
            return x
        return self.head(x)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @torch.no_grad()
    def fuse(self):
        def fuse_children(net: nn.Module):
            for child_name, child in net.named_children():
                if hasattr(child, "fuse"):
                    fused = child.fuse()
                    if fused is not child:
                        setattr(net, child_name, fused)
                        fuse_children(fused)
                    else:
                        fuse_children(child)
                else:
                    fuse_children(child)

        fuse_children(self)


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    out_dict = {}
    for key, value in state_dict.items():
        # classifier: official Classfier → RepVitClassifier(head/head_dist)
        if key.startswith("classifier.classifier_dist."):
            key = key.replace("classifier.classifier_dist.", "head.head_dist.", 1)
        elif key.startswith("classifier.classifier."):
            key = key.replace("classifier.classifier.", "head.head.", 1)
        elif key.startswith("classifier."):
            key = key.replace("classifier.", "head.head.", 1)

        # stem: official downsample_layers.0 — your Stem conv1/conv2/conv3
        if key.startswith("downsample_layers.0.0."):
            key = key.replace("downsample_layers.0.0.", "stem.conv1.", 1)
        elif key.startswith("downsample_layers.0.2.conv_exp_bn1."):
            key = key.replace("downsample_layers.0.2.conv_exp_bn1.", "stem.conv2.", 1)
        elif key.startswith("downsample_layers.0.2.conv_pwl_bn2."):
            key = key.replace("downsample_layers.0.2.conv_pwl_bn2.", "stem.conv3.", 1)
        elif key.startswith("downsample_layers."):
            # other downsamples: downsample_layers.{i}.0. → stages.{i}.downsample. (Stage convention)
            m = re.match(r"^downsample_layers\.(\d+)\.0\.(.*)", key)
            if m:
                key = f"stages.{m.group(1)}.downsample.{m.group(2)}"
            else:
                m = re.match(r"^downsample_layers\.(\d+)\.(.*)", key)
                if m:
                    key = f"stages.{m.group(1)}.downsample.{m.group(2)}"
            # internal renames for downsample ConvNorm c/bn (keep c/bn, just mixer rename below)
            key = key.replace("token_channel_mixer", "mixer")
            key = key.replace("channel_mixer", "mixer")
            key = key.replace(".m.", ".module.")
            out_dict[key] = value
            continue

        # stages: official stages.{si}.{bi}.block.* → timm stages.{si}.blocks.{bi}.* (Stage convention)
        m = re.match(r"^stages\.(\d+)\.(\d+)\.block\.(.*)", key)
        if m:
            si, bi, rest = int(m.group(1)), int(m.group(2)), m.group(3)
            key = f"stages.{si}.blocks.{bi}.{rest}"
        elif key.startswith("stages."):
            m = re.match(r"^stages\.(\d+)\.(\d+)\.(.*)", key)
            if m:
                si, bi, rest = int(m.group(1)), int(m.group(2)), m.group(3)
                key = f"stages.{si}.blocks.{bi}.{rest}"
        # internal block renames: token_channel_mixer / channel_mixer → mixer, .m. → .module.
        key = key.replace("token_channel_mixer", "mixer")
        key = key.replace("channel_mixer", "mixer")
        key = key.replace(".m.", ".module.")
        out_dict[key] = value
    return out_dict


def _cfg(url: str = "", **kwargs: Any) -> Dict[str, Any]:
    return {
        "url": url, "num_classes": 1000, "input_size": (3, 224, 224), "pool_size": (7, 7),
        "crop_pct": 0.875, "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD,
        "first_conv": "stem.conv1.c", "classifier": "head.head.l",
        "paper_ids": "arXiv:2501.15369",
        "paper_name": "iFormer: Integrating ConvNet and Transformer for Mobile Application",
        "origin_url": "https://github.com/ChuanyangZheng/iFormer", "license": "mit",
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    "iformer_t.in1k": _cfg(
        url="https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_t.pth",
        # hf_hub_id='timm/',
    ),
    "iformer_s.in1k": _cfg(
        url="https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_s.pth",
        # hf_hub_id='timm/',
    ),
    "iformer_m.in1k": _cfg(
        url="https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_m.pth",
        # hf_hub_id='timm/',
    ),
    "iformer_l.in1k": _cfg(
        url="https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_l.pth",
        # hf_hub_id='timm/',
    ),
    "iformer_l2.untrained": _cfg(),
    "iformer_h.in1k": _cfg(
        url="https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_h.pth",
        # hf_hub_id='timm/',
    ),
    "iformer_m_distilled.in1k": _cfg(
        url="https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_m_distill.pth",
        classifier=('head.head.l', 'head.head_dist.l'),
        # hf_hub_id='timm/',
    ),
    "iformer_l_distilled.in1k": _cfg(
        url="https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_l_distill.pth",
        classifier=('head.head.l', 'head.head_dist.l'),
        # hf_hub_id='timm/',
    ),
    "iformer_l2_distilled.in1k": _cfg(
        url="https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_l2_distill.pth",
        classifier=('head.head.l', 'head.head_dist.l'),
        # hf_hub_id='timm/',
    ),
})


def _create_iformer(variant: str, pretrained: bool = False, **kwargs: Any) -> iFormer:
    return build_model_with_cfg(
        iFormer, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )


@register_model
def iformer_t(pretrained: bool = False, **kwargs: Any) -> iFormer:
    model_args = dict(
        dims=(32, 64, 128, 256), depths=(2, 2, 16, 6), attn_groups=(0, 0, 3, 2),
        attn_head_dim_reduction=(0, 0, 2, 4), conv_ratio=3, ffn_ratio=2,
    )
    return _create_iformer("iformer_t", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def iformer_s(pretrained: bool = False, **kwargs: Any) -> iFormer:
    model_args = dict(
        dims=(32, 64, 176, 320), depths=(2, 2, 19, 6), attn_groups=(0, 0, 3, 2),
        attn_head_dim_reduction=(0, 0, 2, 4), conv_ratio=4, ffn_ratio=3,
    )
    return _create_iformer("iformer_s", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def iformer_m(pretrained: bool = False, **kwargs: Any) -> iFormer:
    model_args = dict(
        dims=(48, 96, 192, 384), depths=(2, 2, 22, 6), attn_groups=(0, 0, 4, 2),
        attn_head_dim_reduction=(0, 0, 2, 4), conv_ratio=4, ffn_ratio=3,
    )
    return _create_iformer("iformer_m", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def iformer_l(pretrained: bool = False, **kwargs: Any) -> iFormer:
    model_args = dict(
        dims=(48, 96, 256, 384), depths=(2, 2, 33, 6), attn_groups=(0, 0, 8, 2),
        attn_head_dim_reduction=(0, 0, 2, 4), conv_ratio=4, ffn_ratio=3,
    )
    return _create_iformer("iformer_l", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def iformer_l2(pretrained: bool = False, **kwargs: Any) -> iFormer:
    model_args = dict(
        dims=(64, 128, 256, 512), depths=(3, 3, 46, 9), attn_groups=(0, 0, 11, 3),
        attn_head_dim_reduction=(0, 0, 2, 4), conv_ratio=4, ffn_ratio=3,
    )
    return _create_iformer("iformer_l2", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def iformer_h(pretrained: bool = False, **kwargs: Any) -> iFormer:
    model_args = dict(
        dims=(96, 192, 384, 768), depths=(5, 5, 60, 18), attn_groups=(0, 0, 15, 6),
        attn_head_dim_reduction=(0, 0, 1, 1), conv_ratio=4, ffn_ratio=4,
        layer_scale_init_value=1e-6,
    )
    return _create_iformer("iformer_h", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def iformer_m_distilled(pretrained: bool = False, **kwargs: Any) -> iFormer:
    model_args = dict(
        dims=(48, 96, 192, 384), depths=(2, 2, 22, 6), attn_groups=(0, 0, 4, 2),
        attn_head_dim_reduction=(0, 0, 2, 4), conv_ratio=4, ffn_ratio=3,
        distillation=True,
    )
    return _create_iformer("iformer_m_distilled", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def iformer_l_distilled(pretrained: bool = False, **kwargs: Any) -> iFormer:
    model_args = dict(
        dims=(48, 96, 256, 384), depths=(2, 2, 33, 6), attn_groups=(0, 0, 8, 2),
        attn_head_dim_reduction=(0, 0, 2, 4), conv_ratio=4, ffn_ratio=3,
        distillation=True,
    )
    return _create_iformer("iformer_l_distilled", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def iformer_l2_distilled(pretrained: bool = False, **kwargs: Any) -> iFormer:
    model_args = dict(
        dims=(64, 128, 256, 512), depths=(3, 3, 46, 9), attn_groups=(0, 0, 11, 3),
        attn_head_dim_reduction=(0, 0, 2, 4), conv_ratio=4, ffn_ratio=3,
        distillation=True,
    )
    return _create_iformer("iformer_l2_distilled", pretrained=pretrained, **dict(model_args, **kwargs))
