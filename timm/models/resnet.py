"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman
"""
import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, LayerType, create_attn, \
    get_attn, get_act_layer, get_norm_layer, create_classifier
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs, register_model_deprecations

__all__ = ['ResNet', 'BasicBlock', 'Bottleneck']  # model_registry will add each entrypoint fn to this


def get_padding(kernel_size: int, stride: int, dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer: Type[nn.Module], channels: int, stride: int = 2, enable: bool = True) -> nn.Module:
    if not aa_layer or not enable:
        return nn.Identity()
    if issubclass(aa_layer, nn.AvgPool2d):
        return aa_layer(stride)
    else:
        return aa_layer(channels=channels, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            attn_layer: Attention layer.
            aa_layer: Anti-aliasing layer.
            drop_block: Class for DropBlock layer.
            drop_path: Optional DropPath layer.
        """
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn2, 'weight', None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            attn_layer: Attention layer.
            aa_layer: Anti-aliasing layer.
            drop_block: Class for DropBlock layer.
            drop_path: Optional DropPath layer.
        """
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def downsample_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_prob: float = 0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]


def make_blocks(
        block_fn: Union[BasicBlock, Bottleneck],
        channels: List[int],
        block_repeats: List[int],
        inplanes: int,
        reduce_first: int = 1,
        output_stride: int = 32,
        down_kernel_size: int = 1,
        avg_down: bool = False,
        drop_block_rate: float = 0.,
        drop_path_rate: float = 0.,
        **kwargs,
) -> Tuple[List[Tuple[str, nn.Module]], List[Dict[str, Any]]]:
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get('norm_layer'),
            )
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes,
                planes,
                stride,
                downsample,
                first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None,
                **block_kwargs,
            ))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    """

    def __init__(
            self,
            block: Union[BasicBlock, Bottleneck],
            layers: List[int],
            num_classes: int = 1000,
            in_chans: int = 3,
            output_stride: int = 32,
            global_pool: str = 'avg',
            cardinality: int = 1,
            base_width: int = 64,
            stem_width: int = 64,
            stem_type: str = '',
            replace_stem_pool: bool = False,
            block_reduce_first: int = 1,
            down_kernel_size: int = 1,
            avg_down: bool = False,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.,
            drop_block_rate: float = 0.,
            zero_init_last: bool = True,
            block_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            block (nn.Module): class for the residual block. Options are BasicBlock, Bottleneck.
            layers (List[int]) : number of layers in each block
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            cardinality (int): number of convolution groups for 3x3 conv in Bottleneck. (default 1)
            base_width (int): bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): number of channels in stem convolutions (default 64)
            stem_type (str): The type of stem (default ''):
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool (bool): replace stem max-pooling layer with a 3x3 stride-2 convolution
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            avg_down (bool): use avg pooling for projection skip connection between stages/downsample (default False)
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            aa_layer (nn.Module): anti-aliasing layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default 0.)
            drop_block_rate (float): Drop block rate (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module
        """
        super(ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        
        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [
                nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
                norm_layer(inplanes),
                act_layer(inplace=True),
            ]))
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(*[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2)])
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **block_args,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
    def init_weights(self, zero_init_last: bool = True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only: bool = False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_resnet(variant, pretrained: bool = False, **kwargs) -> ResNet:
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


def _tcfg(url='', **kwargs):
    return _cfg(url=url, **dict({'interpolation': 'bicubic'}, **kwargs))


def _ttcfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic', 'test_input_size': (3, 288, 288), 'test_crop_pct': 0.95,
        'origin_url': 'https://github.com/huggingface/pytorch-image-models',
    }, **kwargs))


def _rcfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic', 'crop_pct': 0.95, 'test_input_size': (3, 288, 288), 'test_crop_pct': 1.0,
        'origin_url': 'https://github.com/huggingface/pytorch-image-models', 'paper_ids': 'arXiv:2110.00476'
    }, **kwargs))


def _r3cfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic', 'input_size': (3, 160, 160), 'pool_size': (5, 5),
        'crop_pct': 0.95, 'test_input_size': (3, 224, 224), 'test_crop_pct': 0.95,
        'origin_url': 'https://github.com/huggingface/pytorch-image-models', 'paper_ids': 'arXiv:2110.00476',
    }, **kwargs))


def _gcfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic',
        'origin_url': 'https://cv.gluon.ai/model_zoo/classification.html',
    }, **kwargs))


default_cfgs = generate_default_cfgs({
    # ResNet and Wide ResNet trained w/ timm (RSB paper and others)
    'resnet10t.c3_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet10t_176_c3-f3215ab1.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_crop_pct=0.95, test_input_size=(3, 224, 224),
        first_conv='conv1.0'),
    'resnet14t.c3_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet14t_176_c3-c4ed2c37.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_crop_pct=0.95, test_input_size=(3, 224, 224),
        first_conv='conv1.0'),
    'resnet18.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth'),
    'resnet18.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a2_0-b61bd467.pth'),
    'resnet18.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a3_0-40c531c8.pth'),
    'resnet18d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet18d_ra2-48a79e06.pth',
        first_conv='conv1.0'),
    'resnet34.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a1_0-46f8f793.pth'),
    'resnet34.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a2_0-82d47d71.pth'),
    'resnet34.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a3_0-a20cabb6.pth',
        crop_pct=0.95),
    'resnet34.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth'),
    'resnet34d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34d_ra2-f8dcfcaf.pth',
        first_conv='conv1.0'),
    'resnet26.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26-9aa10e23.pth'),
    'resnet26d.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26d-69e92c46.pth',
        first_conv='conv1.0'),
    'resnet26t.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/resnet26t_256_ra2-6f6fa748.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.94, test_input_size=(3, 320, 320), test_crop_pct=1.0),
    'resnet50.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth'),
    'resnet50.a1h_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1h2_176-001a1197.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), crop_pct=0.9, test_input_size=(3, 224, 224), test_crop_pct=1.0),
    'resnet50.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a2_0-a2746f79.pth'),
    'resnet50.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a3_0-59cae1ef.pth'),
    'resnet50.b1k_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_b1k-532a802a.pth'),
    'resnet50.b2k_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_b2k-1ba180c1.pth'),
    'resnet50.c1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_c1-5ba5e060.pth'),
    'resnet50.c2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_c2-d01e05b2.pth'),
    'resnet50.d_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_d-f39db8af.pth'),
    'resnet50.ram_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth'),
    'resnet50.am_in1k': _tcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_am-6c502b37.pth'),
    'resnet50.ra_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ra-85ebb6e5.pth'),
    'resnet50.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/rw_resnet50-86acaeed.pth'),
    'resnet50d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth',
        first_conv='conv1.0'),
    'resnet50d.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a1_0-e20cff14.pth',
        first_conv='conv1.0'),
    'resnet50d.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a2_0-a3adc64d.pth',
        first_conv='conv1.0'),
    'resnet50d.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a3_0-403fdfad.pth',
        first_conv='conv1.0'),
    'resnet50t.untrained': _ttcfg(first_conv='conv1.0'),
    'resnet101.a1h_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth'),
    'resnet101.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1_0-cdcb52a9.pth'),
    'resnet101.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a2_0-6edb36c7.pth'),
    'resnet101.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a3_0-1db14157.pth'),
    'resnet101d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet101d_ra2-2803ffab.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95,
        test_crop_pct=1.0, test_input_size=(3, 320, 320)),
    'resnet152.a1h_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a1h-dc400468.pth'),
    'resnet152.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a1_0-2eee8a7a.pth'),
    'resnet152.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a2_0-b4c6978f.pth'),
    'resnet152.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a3_0-134d4688.pth'),
    'resnet152d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet152d_ra2-5cac0439.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95,
        test_crop_pct=1.0, test_input_size=(3, 320, 320)),
    'resnet200.untrained': _ttcfg(),
    'resnet200d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95,
        test_crop_pct=1.0, test_input_size=(3, 320, 320)),
    'wide_resnet50_2.racm_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/wide_resnet50_racm-8234f177.pth'),

    # torchvision resnet weights
    'resnet18.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet18-5c106cde.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet34.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet50.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet50-19c8e357.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet50.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet101.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet101.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet101-cd907fc2.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet152.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet152.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet152-f82ba261.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'wide_resnet50_2.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'wide_resnet50_2.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'wide_resnet101_2.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'wide_resnet101_2.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),

    # ResNets w/ alternative norm layers
    'resnet50_gn.a1h_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_gn_a1h2-8fe6c4d0.pth',
        crop_pct=0.94),

    # ResNeXt trained in timm (RSB paper and others)
    'resnext50_32x4d.a1h_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a1h-0146ab0a.pth'),
    'resnext50_32x4d.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a1_0-b5a91a1d.pth'),
    'resnext50_32x4d.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a2_0-efc76add.pth'),
    'resnext50_32x4d.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a3_0-3e450271.pth'),
    'resnext50_32x4d.ra_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnext50_32x4d_ra-d733960d.pth'),
    'resnext50d_32x4d.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50d_32x4d-103e99f8.pth',
        first_conv='conv1.0'),
    'resnext101_32x4d.untrained': _ttcfg(),
    'resnext101_64x4d.c1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnext101_64x4d_c-0d0e0cc0.pth'),

    # torchvision ResNeXt weights
    'resnext50_32x4d.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnext101_32x8d.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnext101_64x4d.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnext50_32x4d.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnext101_32x8d.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),

    #  ResNeXt models - Weakly Supervised Pretraining on Instagram Hashtags
    #  from https://github.com/facebookresearch/WSL-Images
    #  Please note the CC-BY-NC 4.0 license on these weights, non-commercial use only.
    'resnext101_32x8d.fb_wsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/WSL-Images'),
    'resnext101_32x16d.fb_wsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/WSL-Images'),
    'resnext101_32x32d.fb_wsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/WSL-Images'),
    'resnext101_32x48d.fb_wsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/WSL-Images'),

    #  Semi-Supervised ResNe*t models from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
    #  Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
    'resnet18.fb_ssl_yfcc100m_ft_in1k':  _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnet50.fb_ssl_yfcc100m_ft_in1k':  _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext50_32x4d.fb_ssl_yfcc100m_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x4d.fb_ssl_yfcc100m_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x8d.fb_ssl_yfcc100m_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),

    #  Semi-Weakly Supervised ResNe*t models from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
    #  Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
    'resnet18.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnet50.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext50_32x4d.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x4d.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x8d.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x8-b4712904.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x16d.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),

    #  Efficient Channel Attention ResNets
    'ecaresnet26t.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecaresnet26t_ra2-46609757.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        test_crop_pct=0.95, test_input_size=(3, 320, 320)),
    'ecaresnetlight.miil_in1k': _tcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnetlight-75a9c627.pth',
        test_crop_pct=0.95, test_input_size=(3, 288, 288)),
    'ecaresnet50d.miil_in1k': _tcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet50d-93c81e3b.pth',
        first_conv='conv1.0', test_crop_pct=0.95, test_input_size=(3, 288, 288)),
    'ecaresnet50d_pruned.miil_in1k': _tcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet50d_p-e4fa23c2.pth',
        first_conv='conv1.0', test_crop_pct=0.95, test_input_size=(3, 288, 288)),
    'ecaresnet50t.ra2_in1k': _tcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecaresnet50t_ra2-f7ac63c4.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        test_crop_pct=0.95, test_input_size=(3, 320, 320)),
    'ecaresnet50t.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/ecaresnet50t_a1_0-99bd76a8.pth',
        first_conv='conv1.0'),
    'ecaresnet50t.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/ecaresnet50t_a2_0-b1c7b745.pth',
        first_conv='conv1.0'),
    'ecaresnet50t.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/ecaresnet50t_a3_0-8cc311f1.pth',
        first_conv='conv1.0'),
    'ecaresnet101d.miil_in1k': _tcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet101d-153dad65.pth',
        first_conv='conv1.0', test_crop_pct=0.95, test_input_size=(3, 288, 288)),
    'ecaresnet101d_pruned.miil_in1k': _tcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet101d_p-9e74cb91.pth',
        first_conv='conv1.0', test_crop_pct=0.95, test_input_size=(3, 288, 288)),
    'ecaresnet200d.untrained': _ttcfg(
        first_conv='conv1.0', input_size=(3, 256, 256), crop_pct=0.95, pool_size=(8, 8)),
    'ecaresnet269d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecaresnet269d_320_ra2-7baa55cb.pth',
        first_conv='conv1.0', input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=0.95,
        test_crop_pct=1.0, test_input_size=(3, 352, 352)),

    #  Efficient Channel Attention ResNeXts
    'ecaresnext26t_32x4d.untrained': _tcfg(first_conv='conv1.0'),
    'ecaresnext50t_32x4d.untrained': _tcfg(first_conv='conv1.0'),

    #  Squeeze-Excitation ResNets, to eventually replace the models in senet.py
    'seresnet18.untrained': _ttcfg(),
    'seresnet34.untrained': _ttcfg(),
    'seresnet50.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/seresnet50_a1_0-ffa00869.pth',
        crop_pct=0.95),
    'seresnet50.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/seresnet50_a2_0-850de0d9.pth',
        crop_pct=0.95),
    'seresnet50.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/seresnet50_a3_0-317ecd56.pth',
        crop_pct=0.95),
    'seresnet50.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet50_ra_224-8efdb4bb.pth'),
    'seresnet50t.untrained': _ttcfg(
        first_conv='conv1.0'),
    'seresnet101.untrained': _ttcfg(),
    'seresnet152.untrained': _ttcfg(),
    'seresnet152d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet152d_ra2-04464dd2.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95,
        test_crop_pct=1.0, test_input_size=(3, 320, 320)
    ),
    'seresnet200d.untrained': _ttcfg(
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8)),
    'seresnet269d.untrained': _ttcfg(
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8)),

    #  Squeeze-Excitation ResNeXts, to eventually replace the models in senet.py
    'seresnext26d_32x4d.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26d_32x4d-80fa48a3.pth',
        first_conv='conv1.0'),
    'seresnext26t_32x4d.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26tn_32x4d-569cb627.pth',
        first_conv='conv1.0'),
    'seresnext50_32x4d.racm_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext50_32x4d_racm-a304a460.pth'),
    'seresnext101_32x4d.untrained': _ttcfg(),
    'seresnext101_32x8d.ah_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/seresnext101_32x8d_ah-e6bc4c0a.pth'),
    'seresnext101d_32x8d.ah_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/seresnext101d_32x8d_ah-191d7b94.pth',
        first_conv='conv1.0'),

    # ResNets with anti-aliasing / blur pool
    'resnetaa50d.sw_in12k_ft_in1k': _ttcfg(
        hf_hub_id='timm/',
        first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),
    'resnetaa101d.sw_in12k_ft_in1k': _ttcfg(
        hf_hub_id='timm/',
        first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),
    'seresnextaa101d_32x8d.sw_in12k_ft_in1k_288': _ttcfg(
        hf_hub_id='timm/',
        crop_pct=0.95, input_size=(3, 288, 288), pool_size=(9, 9), test_input_size=(3, 320, 320), test_crop_pct=1.0,
        first_conv='conv1.0'),
    'seresnextaa101d_32x8d.sw_in12k_ft_in1k': _ttcfg(
        hf_hub_id='timm/',
        first_conv='conv1.0', test_crop_pct=1.0),
    'seresnextaa201d_32x8d.sw_in12k_ft_in1k_384': _cfg(
        hf_hub_id='timm/',
        interpolation='bicubic', first_conv='conv1.0', pool_size=(12, 12), input_size=(3, 384, 384), crop_pct=1.0),
    'seresnextaa201d_32x8d.sw_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821, interpolation='bicubic', first_conv='conv1.0',
        crop_pct=0.95, input_size=(3, 320, 320), pool_size=(10, 10), test_input_size=(3, 384, 384), test_crop_pct=1.0),

    'resnetaa50d.sw_in12k': _ttcfg(
        hf_hub_id='timm/',
        num_classes=11821, first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),
    'resnetaa50d.d_in12k': _ttcfg(
        hf_hub_id='timm/',
        num_classes=11821, first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),
    'resnetaa101d.sw_in12k': _ttcfg(
        hf_hub_id='timm/',
        num_classes=11821, first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),
    'seresnextaa101d_32x8d.sw_in12k': _ttcfg(
        hf_hub_id='timm/',
        num_classes=11821, first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),

    'resnetblur18.untrained': _ttcfg(),
    'resnetblur50.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnetblur50-84f4748f.pth'),
    'resnetblur50d.untrained': _ttcfg(first_conv='conv1.0'),
    'resnetblur101d.untrained': _ttcfg(first_conv='conv1.0'),
    'resnetaa34d.untrained': _ttcfg(first_conv='conv1.0'),
    'resnetaa50.a1h_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnetaa50_a1h-4cf422b3.pth'),

    'seresnetaa50d.untrained': _ttcfg(first_conv='conv1.0'),
    'seresnextaa101d_32x8d.ah_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/seresnextaa101d_32x8d_ah-83c8ae12.pth',
        first_conv='conv1.0'),

    # ResNet-RS models
    'resnetrs50.tf_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs50_ema-6b53758b.pth',
        input_size=(3, 160, 160), pool_size=(5, 5), crop_pct=0.91, test_input_size=(3, 224, 224),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs101.tf_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs101_i192_ema-1509bbf6.pth',
        input_size=(3, 192, 192), pool_size=(6, 6), crop_pct=0.94, test_input_size=(3, 288, 288),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs152.tf_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs152_i256_ema-a9aff7f9.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 320, 320),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs200.tf_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnetrs200_c-6b698b88.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 320, 320),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs270.tf_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs270_ema-b40e674c.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 352, 352),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs350.tf_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs350_i256_ema-5a1aa8f1.pth',
        input_size=(3, 288, 288), pool_size=(9, 9), crop_pct=1.0, test_input_size=(3, 384, 384),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs420.tf_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs420_ema-972dee69.pth',
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=1.0, test_input_size=(3, 416, 416),
        interpolation='bicubic', first_conv='conv1.0'),

    # gluon resnet weights
    'resnet18.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet18_v1b-0757602b.pth'),
    'resnet34.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet34_v1b-c6d82d59.pth'),
    'resnet50.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1b-0ebe02e2.pth'),
    'resnet101.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1b-3b017079.pth'),
    'resnet152.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1b-c1edb0dd.pth'),
    'resnet50c.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1c-48092f55.pth',
        first_conv='conv1.0'),
    'resnet101c.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1c-1f26822a.pth',
        first_conv='conv1.0'),
    'resnet152c.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1c-a3bb0b98.pth',
        first_conv='conv1.0'),
    'resnet50d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1d-818a1b1b.pth',
        first_conv='conv1.0'),
    'resnet101d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1d-0f9c8644.pth',
        first_conv='conv1.0'),
    'resnet152d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1d-bd354e12.pth',
        first_conv='conv1.0'),
    'resnet50s.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1s-1762acc0.pth',
        first_conv='conv1.0'),
    'resnet101s.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1s-60fe0cc1.pth',
        first_conv='conv1.0'),
    'resnet152s.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1s-dcc41b81.pth',
        first_conv='conv1.0'),
    'resnext50_32x4d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext50_32x4d-e6a097c1.pth'),
    'resnext101_32x4d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext101_32x4d-b253c8c4.pth'),
    'resnext101_64x4d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext101_64x4d-f9a8e184.pth'),
    'seresnext50_32x4d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_seresnext50_32x4d-90cf2d6e.pth'),
    'seresnext101_32x4d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_seresnext101_32x4d-cf52900d.pth'),
    'seresnext101_64x4d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_seresnext101_64x4d-f9926f93.pth'),
    'senet154.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_senet154-70a1a3c0.pth',
        first_conv='conv1.0'),
})


@register_model
def resnet10t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-10-T model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 1, 1, 1], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('resnet10t', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet14t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-14-T model.
    """
    model_args = dict(block=Bottleneck, layers=[1, 1, 1, 1], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('resnet14t', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet18(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2])
    return _create_resnet('resnet18', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet18d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-18-D model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet18d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet34(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3])
    return _create_resnet('resnet34', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet34d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-34-D model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet34d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet26(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-26 model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2])
    return _create_resnet('resnet26', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet26t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-26-T model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('resnet26t', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet26d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-26-D model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet26d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3])
    return _create_resnet('resnet50', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50c(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-C model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep')
    return _create_resnet('resnet50c', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet50d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50s(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-S model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=64, stem_type='deep')
    return _create_resnet('resnet50s', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-T model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('resnet50t', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3])
    return _create_resnet('resnet101', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101c(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-C model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep')
    return _create_resnet('resnet101c', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet101d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101s(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-S model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=64, stem_type='deep')
    return _create_resnet('resnet101s', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-152 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3])
    return _create_resnet('resnet152', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152c(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-152-C model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep')
    return _create_resnet('resnet152c', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-152-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet152d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152s(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-152-S model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], stem_width=64, stem_type='deep')
    return _create_resnet('resnet152s', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet200(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-200 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3])
    return _create_resnet('resnet200', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet200d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-200-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet200d', pretrained, **dict(model_args, **kwargs))


@register_model
def wide_resnet50_2(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a Wide ResNet-50-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], base_width=128)
    return _create_resnet('wide_resnet50_2', pretrained, **dict(model_args, **kwargs))


@register_model
def wide_resnet101_2(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a Wide ResNet-101-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], base_width=128)
    return _create_resnet('wide_resnet101_2', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50_gn(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model w/ GroupNorm
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], norm_layer='groupnorm')
    return _create_resnet('resnet50_gn', pretrained, **dict(model_args, **kwargs))


@register_model
def resnext50_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt50-32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4)
    return _create_resnet('resnext50_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnext50d_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt50d-32x4d model. ResNext50 w/ deep stem & avg pool downsample
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3],  cardinality=32, base_width=4,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnext50d_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4)
    return _create_resnet('resnext101_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x8d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x8d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8)
    return _create_resnet('resnext101_32x8d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x16d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x16d model
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=16)
    return _create_resnet('resnext101_32x16d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x32d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x32d model
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=32)
    return _create_resnet('resnext101_32x32d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_64x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt101-64x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64, base_width=4)
    return _create_resnet('resnext101_64x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet26t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNeXt-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem and ECA attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet26t', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet50d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model with eca.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet50d', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet50d_pruned(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model pruned with eca.
        The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet50d_pruned', pretrained, pruned=True, **dict(model_args, **kwargs))


@register_model
def ecaresnet50t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNet-50-T model.
    Like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels in the deep stem and ECA attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet50t', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnetlight(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D light model with eca.
    """
    model_args = dict(
        block=Bottleneck, layers=[1, 1, 11, 3], stem_width=32, avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnetlight', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet101d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model with eca.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet101d', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet101d_pruned(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model pruned with eca.
       The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet101d_pruned', pretrained, pruned=True, **dict(model_args, **kwargs))


@register_model
def ecaresnet200d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-200-D model with ECA.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet200d', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet269d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-269-D model with ECA.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 30, 48, 8], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet269d', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnext26t_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNeXt-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem. This model replaces SE module with the ECA module
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnext26t_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnext50t_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNeXt-50-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem. This model replaces SE module with the ECA module
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnext50t_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet18(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet18', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet34(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet34', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet50(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet50', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet50t(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3],  stem_width=32, stem_type='deep_tiered',
        avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet50t', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet101(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet101', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet152(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet152', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet152d(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep',
        avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet152d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet200d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-200-D model with SE attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep',
        avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet200d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet269d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-269-D model with SE attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 30, 48, 8], stem_width=32, stem_type='deep',
        avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet269d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext26d_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNeXt-26-D model.`
    This is technically a 28 layer ResNet, using the 'D' modifier from Gluon / bag-of-tricks for
    combination of deep stem and avg_pool in downsample.
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep', avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext26d_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext26t_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNet-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem.
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext26t_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext50_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext50_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext101_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext101_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext101_32x8d(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext101_32x8d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext101d_32x8d(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8,
        stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext101d_32x8d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext101_64x4d(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64, base_width=4,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext101_64x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def senet154(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], cardinality=64, base_width=4, stem_type='deep',
        down_kernel_size=3, block_reduce_first=2, block_args=dict(attn_layer='se'))
    return _create_resnet('senet154', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetblur18(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-18 model with blur anti-aliasing
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], aa_layer=BlurPool2d)
    return _create_resnet('resnetblur18', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetblur50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model with blur anti-aliasing
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=BlurPool2d)
    return _create_resnet('resnetblur50', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetblur50d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model with blur anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=BlurPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnetblur50d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetblur101d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model with blur anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], aa_layer=BlurPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnetblur101d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa34d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-34-D model w/ avgpool anti-aliasing
    """
    model_args = dict(
        block=BasicBlock, layers=[3, 4, 6, 3],  aa_layer=nn.AvgPool2d, stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnetaa34d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model with avgpool anti-aliasing
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=nn.AvgPool2d)
    return _create_resnet('resnetaa50', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa50d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=nn.AvgPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnetaa50d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa101d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], aa_layer=nn.AvgPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnetaa101d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnetaa50d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE=ResNet-50-D model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=nn.AvgPool2d,
        stem_width=32, stem_type='deep', avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnetaa50d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnextaa101d_32x8d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE=ResNeXt-101-D 32x8d model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8,
        stem_width=32, stem_type='deep', avg_down=True, aa_layer=nn.AvgPool2d,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnextaa101d_32x8d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnextaa201d_32x8d(pretrained: bool = False, **kwargs):
    """Constructs a SE=ResNeXt-101-D 32x8d model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 4], cardinality=32, base_width=8,
        stem_width=64, stem_type='deep', avg_down=True, aa_layer=nn.AvgPool2d,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnextaa201d_32x8d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-50 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs50', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs101(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-101 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs101', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs152(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-152 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs152', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs200(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-200 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs200', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs270(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-270 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 29, 53, 4], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs270', pretrained, **dict(model_args, **kwargs))



@register_model
def resnetrs350(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-350 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 36, 72, 4], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs350', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs420(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-420 model
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 44, 87, 4], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs420', pretrained, **dict(model_args, **kwargs))


register_model_deprecations(__name__, {
    'tv_resnet34': 'resnet34.tv_in1k',
    'tv_resnet50': 'resnet50.tv_in1k',
    'tv_resnet101': 'resnet101.tv_in1k',
    'tv_resnet152': 'resnet152.tv_in1k',
    'tv_resnext50_32x4d' : 'resnext50_32x4d.tv_in1k',
    'ig_resnext101_32x8d': 'resnext101_32x8d.fb_wsl_ig1b_ft_in1k',
    'ig_resnext101_32x16d': 'resnext101_32x8d.fb_wsl_ig1b_ft_in1k',
    'ig_resnext101_32x32d': 'resnext101_32x8d.fb_wsl_ig1b_ft_in1k',
    'ig_resnext101_32x48d': 'resnext101_32x8d.fb_wsl_ig1b_ft_in1k',
    'ssl_resnet18': 'resnet18.fb_ssl_yfcc100m_ft_in1k',
    'ssl_resnet50': 'resnet50.fb_ssl_yfcc100m_ft_in1k',
    'ssl_resnext50_32x4d': 'resnext50_32x4d.fb_ssl_yfcc100m_ft_in1k',
    'ssl_resnext101_32x4d': 'resnext101_32x4d.fb_ssl_yfcc100m_ft_in1k',
    'ssl_resnext101_32x8d': 'resnext101_32x8d.fb_ssl_yfcc100m_ft_in1k',
    'ssl_resnext101_32x16d': 'resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k',
    'swsl_resnet18': 'resnet18.fb_swsl_ig1b_ft_in1k',
    'swsl_resnet50': 'resnet50.fb_swsl_ig1b_ft_in1k',
    'swsl_resnext50_32x4d': 'resnext50_32x4d.fb_swsl_ig1b_ft_in1k',
    'swsl_resnext101_32x4d': 'resnext101_32x4d.fb_swsl_ig1b_ft_in1k',
    'swsl_resnext101_32x8d': 'resnext101_32x8d.fb_swsl_ig1b_ft_in1k',
    'swsl_resnext101_32x16d': 'resnext101_32x16d.fb_swsl_ig1b_ft_in1k',
    'gluon_resnet18_v1b': 'resnet18.gluon_in1k',
    'gluon_resnet34_v1b': 'resnet34.gluon_in1k',
    'gluon_resnet50_v1b': 'resnet50.gluon_in1k',
    'gluon_resnet101_v1b': 'resnet101.gluon_in1k',
    'gluon_resnet152_v1b': 'resnet152.gluon_in1k',
    'gluon_resnet50_v1c': 'resnet50c.gluon_in1k',
    'gluon_resnet101_v1c': 'resnet101c.gluon_in1k',
    'gluon_resnet152_v1c': 'resnet152c.gluon_in1k',
    'gluon_resnet50_v1d': 'resnet50d.gluon_in1k',
    'gluon_resnet101_v1d': 'resnet101d.gluon_in1k',
    'gluon_resnet152_v1d': 'resnet152d.gluon_in1k',
    'gluon_resnet50_v1s': 'resnet50s.gluon_in1k',
    'gluon_resnet101_v1s': 'resnet101s.gluon_in1k',
    'gluon_resnet152_v1s': 'resnet152s.gluon_in1k',
    'gluon_resnext50_32x4d': 'resnext50_32x4d.gluon_in1k',
    'gluon_resnext101_32x4d': 'resnext101_32x4d.gluon_in1k',
    'gluon_resnext101_64x4d': 'resnext101_64x4d.gluon_in1k',
    'gluon_seresnext50_32x4d': 'seresnext50_32x4d.gluon_in1k',
    'gluon_seresnext101_32x4d': 'seresnext101_32x4d.gluon_in1k',
    'gluon_seresnext101_64x4d': 'seresnext101_64x4d.gluon_in1k',
    'gluon_senet154': 'senet154.gluon_in1k',
    'seresnext26tn_32x4d': 'seresnext26t_32x4d',
})
