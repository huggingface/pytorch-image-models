""" PP-LCNetV2

Reference:
https://github.com/PaddlePaddle/PaddleClas/blob/release/2.6/docs/en/models/PP-LCNetV2_en.md
The Paddle Implement of PP-LCNetV2 (https://github.com/PaddlePaddle/PaddleClas/blob/release/2.6/ppcls/arch/backbone/legendary_models/pp_lcnet_v2.py)

PP-LCNetV2 is a CPU oriented network built on PP-LCNet (see `lcnet_*` models in mobilenetv3.py). The depthwise
convs of the later stages are re-parameterizable multi-scale branches, SE and shortcuts are used sparingly to
avoid latency penalties on CPU inference.

Adapted from the Paddle implementation for timm by Yonghye Kwon
"""
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import ConvNormAct, SelectAdaptivePool2d, SqueezeExcite, create_act_layer, create_conv2d, \
    make_divisible
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['LCNetV2']


class RepDepthwiseSeparable(nn.Module):
    """Depthwise separable block with re-parameterizable multi-scale depthwise branches.

    When `use_rep` is enabled the depthwise conv is trained as a set of parallel conv-bn branches with
    decreasing kernel size (e.g. 5x5, 3x3, 1x1) that are folded into a single depthwise conv for inference.
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dw_kernel_size: int = 3,
            split_pw: bool = False,
            use_rep: bool = False,
            use_se: bool = False,
            use_shortcut: bool = False,
            pw_ratio: float = 0.5,
            se_ratio: float = 0.25,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            inference_mode: bool = False,
            device=None,
            dtype=None,
    ) -> None:
        """
        Args:
            in_chs: Number of input channels.
            out_chs: Number of output channels.
            stride: Stride of the depthwise conv.
            dw_kernel_size: Kernel size of the (largest) depthwise conv.
            split_pw: Split the pointwise conv into a squeeze and an expand conv.
            use_rep: Train the depthwise conv as re-parameterizable multi-scale branches.
            use_se: Add a squeeze-excite module after the depthwise conv.
            use_shortcut: Add a residual shortcut when input and output shapes match.
            pw_ratio: Channel ratio of the first pointwise conv when `split_pw` is enabled.
            se_ratio: Channel reduction ratio of the squeeze-excite module.
            act_layer: Type of activation layer.
            norm_layer: Type of normalization layer.
            inference_mode: Instantiate the depthwise conv in its re-parameterized form.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.stride = stride
        self.dw_kernel_size = dw_kernel_size
        self.use_shortcut = use_shortcut and stride == 1 and in_chs == out_chs

        if use_rep:
            self.dw_conv_list = None
            if inference_mode:
                self.dw_conv = create_conv2d(
                    in_chs,
                    in_chs,
                    kernel_size=dw_kernel_size,
                    stride=stride,
                    groups=in_chs,
                    bias=True,
                    **dd,
                )
            else:
                # NOTE a 1x1 branch cannot be folded into a strided conv, it's skipped as in the original impl
                self.dw_conv = None
                self.dw_conv_list = nn.ModuleList([
                    ConvNormAct(
                        in_chs,
                        in_chs,
                        kernel_size=k,
                        stride=stride,
                        groups=in_chs,
                        apply_act=False,
                        norm_layer=norm_layer,
                        **dd,
                    ) for k in range(dw_kernel_size, 0, -2) if not (k == 1 and stride != 1)
                ])
            self.dw_act = create_act_layer(act_layer, inplace=True)
        else:
            self.dw_conv_list = None
            self.dw_conv = ConvNormAct(
                in_chs,
                in_chs,
                kernel_size=dw_kernel_size,
                stride=stride,
                groups=in_chs,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **dd,
            )
            self.dw_act = nn.Identity()

        if use_se:
            self.se = SqueezeExcite(
                in_chs,
                rd_channels=int(in_chs * se_ratio),
                act_layer=act_layer,
                gate_layer='sigmoid',
                **dd,
            )
        else:
            self.se = nn.Identity()

        if split_pw:
            mid_chs = int(out_chs * pw_ratio)
            self.pw_conv = nn.Sequential(
                ConvNormAct(in_chs, mid_chs, kernel_size=1, act_layer=act_layer, norm_layer=norm_layer, **dd),
                ConvNormAct(mid_chs, out_chs, kernel_size=1, act_layer=act_layer, norm_layer=norm_layer, **dd),
            )
        else:
            self.pw_conv = ConvNormAct(
                in_chs,
                out_chs,
                kernel_size=1,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **dd,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        if self.dw_conv is not None:
            x = self.dw_act(self.dw_conv(x))
        else:
            out = self.dw_conv_list[0](x)
            for i, dw_conv in enumerate(self.dw_conv_list):
                if i > 0:
                    out = out + dw_conv(x)
            x = self.dw_act(out)
        x = self.se(x)
        x = self.pw_conv(x)
        if self.use_shortcut:
            x = x + shortcut
        return x

    def reparameterize(self) -> None:
        """Fold the multi-scale depthwise branches into a single depthwise conv."""
        if self.dw_conv_list is None:
            return

        kernel, bias = self._get_kernel_bias()
        self.dw_conv = create_conv2d(
            self.in_chs,
            self.in_chs,
            kernel_size=self.dw_kernel_size,
            stride=self.stride,
            groups=self.in_chs,
            bias=True,
            device=kernel.device,
            dtype=kernel.dtype,
        )
        self.dw_conv.weight.data = kernel
        self.dw_conv.bias.data = bias

        # NOTE only the discarded branches are detached, the rest of the block stays trainable
        for param in self.dw_conv_list.parameters():
            param.detach_()

        self.__delattr__('dw_conv_list')
        self.dw_conv_list = None

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sum the bn folded branch kernels, zero padded to the largest kernel size."""
        kernel_final = 0
        bias_final = 0
        for dw_conv in self.dw_conv_list:
            kernel, bias = self._fuse_bn_tensor(dw_conv)
            pad = (self.dw_kernel_size - kernel.shape[-1]) // 2
            kernel_final = kernel_final + F.pad(kernel, [pad, pad, pad, pad])
            bias_final = bias_final + bias
        return kernel_final, bias_final

    @staticmethod
    def _fuse_bn_tensor(branch: ConvNormAct) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fold a conv-bn branch into an equivalent conv weight and bias."""
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


# in_chs, dw_kernel_size, split_pw, use_rep, use_se, use_shortcut
_STAGE_CFG = (
    (64, 3, False, False, False, False),
    (128, 3, False, False, False, False),
    (256, 5, True, True, True, False),
    (512, 5, False, True, False, True),
)


class LCNetV2(nn.Module):
    """PP-LCNetV2"""

    def __init__(
            self,
            scale: float = 1.0,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            head_hidden_size: Optional[int] = 1280,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_rate: float = 0.,
            inference_mode: bool = False,
            device=None,
            dtype=None,
    ) -> None:
        """
        Args:
            scale: Channel multiplier.
            depths: Number of blocks per stage.
            in_chans: Number of input image channels.
            num_classes: Number of classes for the classification head.
            global_pool: Type of pooling to use for global pooling features of the FC head.
            head_hidden_size: Number of channels of the pre-logits conv, disabled if None.
            act_layer: Type of activation layer.
            norm_layer: Type of normalization layer.
            drop_rate: Dropout rate.
            inference_mode: Instantiate the re-parameterizable blocks in their re-parameterized form.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        block_kwargs = dict(act_layer=act_layer, norm_layer=norm_layer, inference_mode=inference_mode, **dd)
        stem_chs = make_divisible(32 * scale)
        prev_chs = make_divisible(64 * scale)
        self.stem = nn.Sequential(
            ConvNormAct(
                in_chans,
                stem_chs,
                kernel_size=3,
                stride=2,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **dd,
            ),
            RepDepthwiseSeparable(stem_chs, prev_chs, stride=1, dw_kernel_size=3, **block_kwargs),
        )

        stages = []
        self.feature_info = []
        reduction = 2
        for stage_idx, (chs, dw_kernel_size, split_pw, use_rep, use_se, use_shortcut) in enumerate(_STAGE_CFG):
            out_chs = make_divisible(chs * 2 * scale)
            blocks = []
            for block_idx in range(depths[stage_idx]):
                blocks += [RepDepthwiseSeparable(
                    in_chs=prev_chs,
                    out_chs=out_chs,
                    stride=2 if block_idx == 0 else 1,
                    dw_kernel_size=dw_kernel_size,
                    split_pw=split_pw,
                    use_rep=use_rep,
                    use_se=use_se,
                    use_shortcut=use_shortcut,
                    **block_kwargs,
                )]
                prev_chs = out_chs
            stages += [nn.Sequential(*blocks)]
            reduction *= 2
            self.feature_info += [dict(num_chs=out_chs, reduction=reduction, module=f'stages.{stage_idx}')]
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs

        # Head + Pooling
        self.head_hidden_size = head_hidden_size or self.num_features
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        if head_hidden_size:
            self.conv_head = create_conv2d(self.num_features, self.head_hidden_size, 1, bias=False, **dd)
            self.act2 = create_act_layer(act_layer, inplace=True)
        else:
            self.conv_head = nn.Identity()
            self.act2 = nn.Identity()
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = nn.Linear(self.head_hidden_size, num_classes, **dd) if num_classes > 0 else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else r'^stages\.(\d+)\.(\d+)',
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.classifier

    def reset_classifier(
            self,
            num_classes: int,
            global_pool: Optional[str] = None,
            device=None,
            dtype=None,
    ) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.classifier = nn.Linear(
            self.head_hidden_size, num_classes, device=device, dtype=dtype) if num_classes > 0 else nn.Identity()

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
        x = self.stem(x)
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
    ) -> List[int]:
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        self.stages = self.stages[:max_index + 1]  # truncate blocks
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x, flatten=True)
        else:
            x = self.stages(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if pre_logits:
            return x
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_lcnetv2(variant: str, pretrained: bool = False, **kwargs) -> LCNetV2:
    out_indices = kwargs.pop('out_indices', (0, 1, 2, 3))
    return build_model_with_cfg(
        LCNetV2,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs,
    )


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0.conv', 'classifier': 'classifier',
        'origin_url': 'https://github.com/PaddlePaddle/PaddleClas',
        'license': 'apache-2.0',
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'lcnetv2_small.paddle_in1k': _cfg(hf_hub_id='timm/'),
    'lcnetv2_base.ssld_in1k': _cfg(hf_hub_id='timm/'),
    'lcnetv2_base.paddle_in1k': _cfg(hf_hub_id='timm/'),
    'lcnetv2_large.paddle_in1k': _cfg(hf_hub_id='timm/'),
})


@register_model
def lcnetv2_small(pretrained: bool = False, **kwargs) -> LCNetV2:
    model_args = dict(scale=0.75, depths=(2, 2, 4, 2))
    return _create_lcnetv2('lcnetv2_small', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def lcnetv2_base(pretrained: bool = False, **kwargs) -> LCNetV2:
    model_args = dict(scale=1.0, depths=(2, 2, 6, 2))
    return _create_lcnetv2('lcnetv2_base', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def lcnetv2_large(pretrained: bool = False, **kwargs) -> LCNetV2:
    model_args = dict(scale=1.25, depths=(2, 2, 8, 2))
    return _create_lcnetv2('lcnetv2_large', pretrained=pretrained, **dict(model_args, **kwargs))
