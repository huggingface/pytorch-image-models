"""
An implementation of GhostNet & GhostNetV2 Models as defined in:
GhostNet: More Features from Cheap Operations. https://arxiv.org/abs/1911.11907
GhostNetV2: Enhance Cheap Operation with Long-Range Attention. https://proceedings.neurips.cc/paper_files/paper/2022/file/40b60852a4abdaa696b5a1a78da34635-Paper-Conference.pdf
GhostNetV3: Exploring the Training Strategies for Compact Models. https://arxiv.org/abs/2404.11202

The train script & code of models at:
Original model: https://github.com/huawei-noah/CV-backbones/tree/master/ghostnet_pytorch
Original model: https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnetv2_pytorch/model/ghostnetv2_torch.py
Original model: https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnetv3_pytorch/ghostnetv3.py
"""
import math
from functools import partial
from typing import Any, Callable, Dict, List, Set, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d, Linear, make_divisible, LayerType
from timm.utils.model import reparameterize_model
from ._builder import build_model_with_cfg
from ._efficientnet_blocks import SqueezeExcite, ConvBnAct
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['GhostNet']


_SE_LAYER = partial(SqueezeExcite, gate_layer='hard_sigmoid', rd_round_fn=partial(make_divisible, divisor=4))


class GhostModule(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 1,
            ratio: int = 2,
            dw_size: int = 3,
            stride: int = 1,
            act_layer: LayerType = nn.ReLU,
    ):
        super(GhostModule, self).__init__()
        self.out_chs = out_chs
        init_chs = math.ceil(out_chs / ratio)
        new_chs = init_chs * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_chs, init_chs, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_chs),
            act_layer(inplace=True),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_chs, new_chs, dw_size, 1, dw_size//2, groups=init_chs, bias=False),
            nn.BatchNorm2d(new_chs),
            act_layer(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_chs, :, :]


class GhostModuleV2(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 1,
            ratio: int = 2,
            dw_size: int = 3,
            stride: int = 1,
            act_layer: LayerType = nn.ReLU,
    ):
        super().__init__()
        self.gate_fn = nn.Sigmoid()
        self.out_chs = out_chs
        init_chs = math.ceil(out_chs / ratio)
        new_chs = init_chs * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_chs, init_chs, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_chs),
            act_layer(inplace=True),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_chs, new_chs, dw_size, 1, dw_size // 2, groups=init_chs, bias=False),
            nn.BatchNorm2d(new_chs),
            act_layer(inplace=True),
        )
        self.short_conv = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.Conv2d(out_chs, out_chs, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=out_chs, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.Conv2d(out_chs, out_chs, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=out_chs, bias=False),
            nn.BatchNorm2d(out_chs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_chs, :, :] * F.interpolate(
            self.gate_fn(res), size=(out.shape[-2], out.shape[-1]), mode='nearest')


class GhostModuleV3(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 1,
            ratio: int = 2,
            dw_size: int = 3,
            stride: int = 1,
            act_layer: LayerType = nn.ReLU,
            mode: str = 'original',
    ):
        super(GhostModuleV3, self).__init__()
        self.gate_fn = nn.Sigmoid()
        self.out_chs = out_chs
        init_chs = math.ceil(out_chs / ratio)
        new_chs = init_chs * (ratio - 1)
        self.mode = mode
        self.num_conv_branches = 3
        self.infer_mode = False
        if not self.infer_mode:
            self.primary_conv = nn.Identity()
            self.cheap_operation = nn.Identity()

        self.primary_rpr_skip = None
        self.primary_rpr_scale = None
        self.primary_rpr_conv = nn.ModuleList(
            [ConvBnAct(in_chs, init_chs, kernel_size, stride, pad_type=kernel_size // 2, \
                    act_layer=None) for _ in range(self.num_conv_branches)]
        )
        # Re-parameterizable scale branch
        self.primary_activation = act_layer(inplace=True)
        self.cheap_rpr_skip = nn.BatchNorm2d(init_chs)
        self.cheap_rpr_conv = nn.ModuleList(
            [ConvBnAct(init_chs, new_chs, dw_size, 1, pad_type=dw_size // 2, group_size=1, \
                    act_layer=None) for _ in range(self.num_conv_branches)]
        )
        # Re-parameterizable scale branch
        self.cheap_rpr_scale = ConvBnAct(init_chs, new_chs, 1, 1, pad_type=0, group_size=1, act_layer=None)
        self.cheap_activation = act_layer(inplace=True)

        self.short_conv = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.Conv2d(out_chs, out_chs, kernel_size=(1,5), stride=1, padding=(0,2), groups=out_chs, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.Conv2d(out_chs, out_chs, kernel_size=(5,1), stride=1, padding=(2,0), groups=out_chs, bias=False),
            nn.BatchNorm2d(out_chs),
        ) if self.mode in ['shortcut'] else nn.Identity()

        self.in_channels = init_chs
        self.groups = init_chs
        self.kernel_size = dw_size

    def forward(self, x):
        if self.infer_mode:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
        else:
            x1 = 0
            for primary_rpr_conv in self.primary_rpr_conv:
                x1 += primary_rpr_conv(x)
            x1 = self.primary_activation(x1)

            x2 = self.cheap_rpr_scale(x1) + self.cheap_rpr_skip(x1)
            for cheap_rpr_conv in self.cheap_rpr_conv:
                x2 += cheap_rpr_conv(x1)
            x2 = self.cheap_activation(x2)

        out = torch.cat([x1,x2], dim=1)
        if self.mode not in ['shortcut']:
            return out
        else:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            return out[:,:self.out_chs,:,:] * F.interpolate(
                self.gate_fn(res), size=(out.shape[-2], out.shape[-1]), mode='nearest')

    def _get_kernel_bias_primary(self):
        kernel_scale = 0
        bias_scale = 0
        if self.primary_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.primary_rpr_scale)
            pad = self.kernel_size // 2
            kernel_scale = F.pad(kernel_scale, [pad, pad, pad, pad])

        kernel_identity = 0
        bias_identity = 0
        if self.primary_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.primary_rpr_skip)

        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.primary_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _get_kernel_bias_cheap(self):
        kernel_scale = 0
        bias_scale = 0
        if self.cheap_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.cheap_rpr_scale)
            pad = self.kernel_size // 2
            kernel_scale = F.pad(kernel_scale, [pad, pad, pad, pad])

        kernel_identity = 0
        bias_identity = 0
        if self.cheap_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.cheap_rpr_skip)

        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.cheap_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch):
        if isinstance(branch, ConvBnAct):
            kernel = branch.conv.weight
            running_mean = branch.bn1.running_mean
            running_var = branch.bn1.running_var
            gamma = branch.bn1.weight
            beta = branch.bn1.bias
            eps = branch.bn1.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if self.infer_mode:
            return
        primary_kernel, primary_bias = self._get_kernel_bias_primary()
        self.primary_conv = nn.Conv2d(
            in_channels=self.primary_rpr_conv[0].conv.in_channels,
            out_channels=self.primary_rpr_conv[0].conv.out_channels,
            kernel_size=self.primary_rpr_conv[0].conv.kernel_size,
            stride=self.primary_rpr_conv[0].conv.stride,
            padding=self.primary_rpr_conv[0].conv.padding,
            dilation=self.primary_rpr_conv[0].conv.dilation,
            groups=self.primary_rpr_conv[0].conv.groups,
            bias=True
        )
        self.primary_conv.weight.data = primary_kernel
        self.primary_conv.bias.data = primary_bias
        self.primary_conv = nn.Sequential(
            self.primary_conv,
            self.primary_activation if self.primary_activation is not None else nn.Sequential()
        )

        cheap_kernel, cheap_bias = self._get_kernel_bias_cheap()
        self.cheap_operation = nn.Conv2d(
            in_channels=self.cheap_rpr_conv[0].conv.in_channels,
            out_channels=self.cheap_rpr_conv[0].conv.out_channels,
            kernel_size=self.cheap_rpr_conv[0].conv.kernel_size,
            stride=self.cheap_rpr_conv[0].conv.stride,
            padding=self.cheap_rpr_conv[0].conv.padding,
            dilation=self.cheap_rpr_conv[0].conv.dilation,
            groups=self.cheap_rpr_conv[0].conv.groups,
            bias=True
        )
        self.cheap_operation.weight.data = cheap_kernel
        self.cheap_operation.bias.data = cheap_bias

        self.cheap_operation = nn.Sequential(
            self.cheap_operation,
            self.cheap_activation if self.cheap_activation is not None else nn.Sequential()
        )

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        if hasattr(self, 'primary_rpr_conv'):
            self.__delattr__('primary_rpr_conv')
        if hasattr(self, 'primary_rpr_scale'):
            self.__delattr__('primary_rpr_scale')
        if hasattr(self, 'primary_rpr_skip'):
            self.__delattr__('primary_rpr_skip')

        if hasattr(self, 'cheap_rpr_conv'):
            self.__delattr__('cheap_rpr_conv')
        if hasattr(self, 'cheap_rpr_scale'):
            self.__delattr__('cheap_rpr_scale')
        if hasattr(self, 'cheap_rpr_skip'):
            self.__delattr__('cheap_rpr_skip')

        self.infer_mode = True

    def reparameterize(self):
        self.switch_to_deploy()


class GhostBottleneck(nn.Module):
    """ GhostV1/V2 bottleneck w/ optional SE"""

    def __init__(
            self,
            in_chs: int,
            mid_chs: int,
            out_chs: int,
            dw_kernel_size: int = 3,
            stride: int = 1,
            act_layer: Callable = nn.ReLU,
            se_ratio: float = 0.,
            mode: str = 'original',
    ):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        if mode == 'original':
            self.ghost1 = GhostModule(in_chs, mid_chs, act_layer=act_layer)
        else:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, act_layer=act_layer)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs, mid_chs, dw_kernel_size, stride=stride,
                padding=(dw_kernel_size-1)//2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        else:
            self.conv_dw = None
            self.bn_dw = None

        # Squeeze-and-excitation
        self.se = _SE_LAYER(mid_chs, rd_ratio=se_ratio) if has_se else None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, act_layer=nn.Identity)

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs, in_chs, dw_kernel_size, stride=stride,
                    padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.conv_dw is not None:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(shortcut)
        return x


class GhostBottleneckV3(nn.Module):
    """ GhostV3 bottleneck w/ optional SE"""

    def __init__(
            self,
            in_chs: int,
            mid_chs: int,
            out_chs: int,
            dw_kernel_size: int = 3,
            stride: int = 1,
            act_layer: LayerType = nn.ReLU,
            se_ratio: float = 0.,
            mode: str = 'original',
    ):
        super(GhostBottleneckV3, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        self.num_conv_branches = 3
        self.infer_mode = False
        if not self.infer_mode:
            self.conv_dw = nn.Identity()
            self.bn_dw = nn.Identity()

        # Point-wise expansion
        self.ghost1 = GhostModuleV3(in_chs, mid_chs, act_layer=act_layer, mode=mode)

        # Depth-wise convolution
        if self.stride > 1:
            self.dw_rpr_conv = nn.ModuleList(
                [ConvBnAct(mid_chs, mid_chs, dw_kernel_size, stride, pad_type=(dw_kernel_size - 1) // 2,
                        group_size=1, act_layer=None) for _ in range(self.num_conv_branches)]
            )
            # Re-parameterizable scale branch
            self.dw_rpr_scale = ConvBnAct(mid_chs, mid_chs, 1, 2, pad_type=0, group_size=1, act_layer=None)
            self.kernel_size = dw_kernel_size
            self.in_channels = mid_chs
        else:
            self.dw_rpr_conv = nn.ModuleList()
            self.dw_rpr_scale = nn.Identity()
        self.dw_rpr_skip = None

        # Squeeze-and-excitation
        self.se = _SE_LAYER(mid_chs, rd_ratio=se_ratio) if has_se else nn.Identity()

        # Point-wise linear projection
        self.ghost2 = GhostModuleV3(mid_chs, out_chs, act_layer=nn.Identity, mode='original')

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs, in_chs, dw_kernel_size, stride=stride,
                    padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            if self.infer_mode:
                x = self.conv_dw(x)
                x = self.bn_dw(x)
            else:
                x1 = self.dw_rpr_scale(x)
                for dw_rpr_conv in self.dw_rpr_conv:
                    x1 += dw_rpr_conv(x)
                x = x1

        # Squeeze-and-excitation
        x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(shortcut)
        return x

    def _get_kernel_bias_dw(self):
        kernel_scale = 0
        bias_scale = 0
        if self.dw_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.dw_rpr_scale)
            pad = self.kernel_size // 2
            kernel_scale = F.pad(kernel_scale, [pad, pad, pad, pad])

        kernel_identity = 0
        bias_identity = 0
        if self.dw_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.dw_rpr_skip)

        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.dw_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch):
        if isinstance(branch, ConvBnAct):
            kernel = branch.conv.weight
            running_mean = branch.bn1.running_mean
            running_var = branch.bn1.running_var
            gamma = branch.bn1.weight
            beta = branch.bn1.bias
            eps = branch.bn1.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if self.infer_mode or self.stride == 1:
            return
        dw_kernel, dw_bias = self._get_kernel_bias_dw()
        self.conv_dw = nn.Conv2d(
            in_channels=self.dw_rpr_conv[0].conv.in_channels,
            out_channels=self.dw_rpr_conv[0].conv.out_channels,
            kernel_size=self.dw_rpr_conv[0].conv.kernel_size,
            stride=self.dw_rpr_conv[0].conv.stride,
            padding=self.dw_rpr_conv[0].conv.padding,
            dilation=self.dw_rpr_conv[0].conv.dilation,
            groups=self.dw_rpr_conv[0].conv.groups,
            bias=True
        )
        self.conv_dw.weight.data = dw_kernel
        self.conv_dw.bias.data = dw_bias
        self.bn_dw = nn.Identity()

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        if hasattr(self, 'dw_rpr_conv'):
            self.__delattr__('dw_rpr_conv')
        if hasattr(self, 'dw_rpr_scale'):
            self.__delattr__('dw_rpr_scale')
        if hasattr(self, 'dw_rpr_skip'):
            self.__delattr__('dw_rpr_skip')

        self.infer_mode = True

    def reparameterize(self):
        self.switch_to_deploy()


class GhostNet(nn.Module):
    def __init__(
            self,
            cfgs,
            num_classes: int = 1000,
            width: float = 1.0,
            in_chans: int = 3,
            output_stride: int = 32,
            global_pool: str = 'avg',
            drop_rate: float = 0.2,
            version: str = 'v1',
    ):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        assert output_stride == 32, 'only output_stride==32 is valid, dilation not supported'
        self.cfgs = cfgs
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []
        Bottleneck = GhostBottleneckV3 if version == 'v3' else GhostBottleneck

        # building first layer
        stem_chs = make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(in_chans, stem_chs, 3, 2, 1, bias=False)
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=f'conv_stem'))
        self.bn1 = nn.BatchNorm2d(stem_chs)
        self.act1 = nn.ReLU(inplace=True)
        prev_chs = stem_chs

        # building inverted residual blocks
        stages = nn.ModuleList([])
        stage_idx = 0
        layer_idx = 0
        net_stride = 2
        for cfg in self.cfgs:
            layers = []
            s = 1
            for k, exp_size, c, se_ratio, s in cfg:
                out_chs = make_divisible(c * width, 4)
                mid_chs = make_divisible(exp_size * width, 4)
                layer_kwargs = {}
                if version == 'v2' and layer_idx > 1:
                    layer_kwargs['mode'] = 'attn'
                if version == 'v3' and layer_idx > 1:
                    layer_kwargs['mode'] = 'shortcut'
                layers.append(Bottleneck(prev_chs, mid_chs, out_chs, k, s, se_ratio=se_ratio, **layer_kwargs))
                prev_chs = out_chs
                layer_idx += 1
            if s > 1:
                net_stride *= 2
                self.feature_info.append(dict(
                    num_chs=prev_chs, reduction=net_stride, module=f'blocks.{stage_idx}'))
            stages.append(nn.Sequential(*layers))
            stage_idx += 1

        out_chs = make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(prev_chs, out_chs, 1)))
        self.pool_dim = prev_chs = out_chs

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        self.num_features = prev_chs
        self.head_hidden_size = out_chs = 1280
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = nn.Conv2d(prev_chs, out_chs, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(out_chs, num_classes) if num_classes > 0 else nn.Identity()

        # FIXME init

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return set()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        matcher = dict(
            stem=r'^conv_stem|bn1',
            blocks=[
                (r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)', None),
                (r'conv_head', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

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
        stage_ends = [-1] + [int(info['module'].split('.')[-1]) for info in self.feature_info[1:]]
        take_indices, max_index = feature_take_indices(len(stage_ends), indices)
        take_indices = [stage_ends[i]+1 for i in take_indices]
        max_index = stage_ends[max_index]

        # forward pass
        feat_idx = 0
        x = self.conv_stem(x)
        if feat_idx in take_indices:
            intermediates.append(x)
        x = self.bn1(x)
        x = self.act1(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.blocks
        else:
            stages = self.blocks[:max_index + 1]

        for feat_idx, stage in enumerate(stages, start=1):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(stage, x)
            else:
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
        stage_ends = [-1] + [int(info['module'].split('.')[-1]) for info in self.feature_info[1:]]
        take_indices, max_index = feature_take_indices(len(stage_ends), indices)
        max_index = stage_ends[max_index]
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks w/ stem as idx 0
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def convert_to_deploy(self):
        reparameterize_model(self, inplace=False)


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    out_dict = {}
    for k, v in state_dict.items():
        if 'bn.' in k and '.ghost' in k:
            k = k.replace('bn.', 'bn1.')
        if 'bn.' in k and '.dw_rpr_' in k:
            k = k.replace('bn.', 'bn1.')
        if 'total' in k:
            continue
        out_dict[k] = v
    return out_dict


def _create_ghostnet(variant: str, width: float = 1.0, pretrained: bool = False, **kwargs: Any) -> GhostNet:
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    model_kwargs = dict(
        cfgs=cfgs,
        width=width,
        **kwargs,
    )
    return build_model_with_cfg(
        GhostNet,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True),
        **model_kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'ghostnet_050.untrained': _cfg(),
    'ghostnet_100.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/huawei-noah/CV-backbones/releases/download/ghostnet_pth/ghostnet_1x.pth'
    ),
    'ghostnet_130.untrained': _cfg(),
    'ghostnetv2_100.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/GhostNetV2/ck_ghostnetv2_10.pth.tar'
    ),
    'ghostnetv2_130.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/GhostNetV2/ck_ghostnetv2_13.pth.tar'
    ),
    'ghostnetv2_160.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/GhostNetV2/ck_ghostnetv2_16.pth.tar'
    ),
    'ghostnetv3_050.untrained': _cfg(),
    'ghostnetv3_100.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/GhostNetV3/ghostnetv3-1.0.pth.tar'
    ),
    'ghostnetv3_130.untrained': _cfg(),
    'ghostnetv3_160.untrained': _cfg(),
})


@register_model
def ghostnet_050(pretrained=False, **kwargs) -> GhostNet:
    """ GhostNet-0.5x """
    model = _create_ghostnet('ghostnet_050', width=0.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def ghostnet_100(pretrained=False, **kwargs) -> GhostNet:
    """ GhostNet-1.0x """
    model = _create_ghostnet('ghostnet_100', width=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def ghostnet_130(pretrained=False, **kwargs) -> GhostNet:
    """ GhostNet-1.3x """
    model = _create_ghostnet('ghostnet_130', width=1.3, pretrained=pretrained, **kwargs)
    return model


@register_model
def ghostnetv2_100(pretrained=False, **kwargs) -> GhostNet:
    """ GhostNetV2-1.0x """
    model = _create_ghostnet('ghostnetv2_100', width=1.0, pretrained=pretrained, version='v2', **kwargs)
    return model


@register_model
def ghostnetv2_130(pretrained=False, **kwargs) -> GhostNet:
    """ GhostNetV2-1.3x """
    model = _create_ghostnet('ghostnetv2_130', width=1.3, pretrained=pretrained, version='v2', **kwargs)
    return model


@register_model
def ghostnetv2_160(pretrained=False, **kwargs) -> GhostNet:
    """ GhostNetV2-1.6x """
    model = _create_ghostnet('ghostnetv2_160', width=1.6, pretrained=pretrained, version='v2', **kwargs)
    return model


@register_model
def ghostnetv3_050(pretrained: bool = False, **kwargs: Any) -> GhostNet:
    """ GhostNetV3-0.5x """
    model = _create_ghostnet('ghostnetv3_050', width=0.5, pretrained=pretrained, version='v3', **kwargs)
    return model


@register_model
def ghostnetv3_100(pretrained: bool = False, **kwargs: Any) -> GhostNet:
    """ GhostNetV3-1.0x """
    model = _create_ghostnet('ghostnetv3_100', width=1.0, pretrained=pretrained, version='v3', **kwargs)
    return model


@register_model
def ghostnetv3_130(pretrained: bool = False, **kwargs: Any) -> GhostNet:
    """ GhostNetV3-1.3x """
    model = _create_ghostnet('ghostnetv3_130', width=1.3, pretrained=pretrained, version='v3', **kwargs)
    return model


@register_model
def ghostnetv3_160(pretrained: bool = False, **kwargs: Any) -> GhostNet:
    """ GhostNetV3-1.6x """
    model = _create_ghostnet('ghostnetv3_160', width=1.6, pretrained=pretrained, version='v3', **kwargs)
    return model