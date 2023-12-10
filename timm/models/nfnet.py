""" Normalization Free Nets. NFNet, NF-RegNet, NF-ResNet (pre-activation) Models

Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
    - https://arxiv.org/abs/2101.08692

Paper: `High-Performance Large-Scale Image Recognition Without Normalization`
    - https://arxiv.org/abs/2102.06171

Official Deepmind JAX code: https://github.com/deepmind/deepmind-research/tree/master/nfnets

Status:
* These models are a work in progress, experiments ongoing.
* Pretrained weights for two models so far, more to come.
* Model details updated to closer match official JAX code now that it's released
* NF-ResNet, NF-RegNet-B, and NFNet-F models supported

Hacked together by / copyright Ross Wightman, 2021.
"""
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, Tuple, Optional

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import ClassifierHead, DropPath, AvgPool2dSame, ScaledStdConv2d, ScaledStdConv2dSame, \
    get_act_layer, get_act_fn, get_attn, make_divisible
from ._builder import build_model_with_cfg
from ._features_fx import register_notrace_module
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['NormFreeNet', 'NfCfg']  # model_registry will add each entrypoint fn to this


@dataclass
class NfCfg:
    depths: Tuple[int, int, int, int]
    channels: Tuple[int, int, int, int]
    alpha: float = 0.2
    stem_type: str = '3x3'
    stem_chs: Optional[int] = None
    group_size: Optional[int] = None
    attn_layer: Optional[str] = None
    attn_kwargs: dict = None
    attn_gain: float = 2.0  # NF correction gain to apply if attn layer is used
    width_factor: float = 1.0
    bottle_ratio: float = 0.5
    num_features: int = 0  # num out_channels for final conv, no final_conv if 0
    ch_div: int = 8  # round channels % 8 == 0 to keep tensor-core use optimal
    reg: bool = False  # enables EfficientNet-like options used in RegNet variants, expand from in_chs, se in middle
    extra_conv: bool = False  # extra 3x3 bottleneck convolution for NFNet models
    gamma_in_act: bool = False
    same_padding: bool = False
    std_conv_eps: float = 1e-5
    skipinit: bool = False  # disabled by default, non-trivial performance impact
    zero_init_fc: bool = False
    act_layer: str = 'silu'


class GammaAct(nn.Module):
    def __init__(self, act_type='relu', gamma: float = 1.0, inplace=False):
        super().__init__()
        self.act_fn = get_act_fn(act_type)
        self.gamma = gamma
        self.inplace = inplace

    def forward(self, x):
        return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)


def act_with_gamma(act_type, gamma: float = 1.):
    def _create(inplace=False):
        return GammaAct(act_type, gamma=gamma, inplace=inplace)
    return _create


class DownsampleAvg(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            conv_layer: Callable = ScaledStdConv2d,
    ):
        """ AvgPool Downsampling as in 'D' ResNet variants. Support for dilation."""
        super(DownsampleAvg, self).__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=1)

    def forward(self, x):
        return self.conv(self.pool(x))


@register_notrace_module  # reason: mul_ causes FX to drop a relevant node. https://github.com/pytorch/pytorch/issues/68301
class NormFreeBlock(nn.Module):
    """Normalization-Free pre-activation block.
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: Optional[int] = None,
            stride: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            alpha: float = 1.0,
            beta: float = 1.0,
            bottle_ratio: float = 0.25,
            group_size: Optional[int] = None,
            ch_div: int = 1,
            reg: bool = True,
            extra_conv: bool = False,
            skipinit: bool = False,
            attn_layer: Optional[Callable] = None,
            attn_gain: bool = 2.0,
            act_layer: Optional[Callable] = None,
            conv_layer: Callable = ScaledStdConv2d,
            drop_path_rate: float = 0.,
    ):
        super().__init__()
        first_dilation = first_dilation or dilation
        out_chs = out_chs or in_chs
        # RegNet variants scale bottleneck from in_chs, otherwise scale from out_chs like ResNet
        mid_chs = make_divisible(in_chs * bottle_ratio if reg else out_chs * bottle_ratio, ch_div)
        groups = 1 if not group_size else mid_chs // group_size
        if group_size and group_size % ch_div == 0:
            mid_chs = group_size * groups  # correct mid_chs if group_size divisible by ch_div, otherwise error
        self.alpha = alpha
        self.beta = beta
        self.attn_gain = attn_gain

        if in_chs != out_chs or stride != 1 or dilation != first_dilation:
            self.downsample = DownsampleAvg(
                in_chs,
                out_chs,
                stride=stride,
                dilation=dilation,
                first_dilation=first_dilation,
                conv_layer=conv_layer,
            )
        else:
            self.downsample = None

        self.act1 = act_layer()
        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.act2 = act_layer(inplace=True)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        if extra_conv:
            self.act2b = act_layer(inplace=True)
            self.conv2b = conv_layer(mid_chs, mid_chs, 3, stride=1, dilation=dilation, groups=groups)
        else:
            self.act2b = None
            self.conv2b = None
        if reg and attn_layer is not None:
            self.attn = attn_layer(mid_chs)  # RegNet blocks apply attn btw conv2 & 3
        else:
            self.attn = None
        self.act3 = act_layer()
        self.conv3 = conv_layer(mid_chs, out_chs, 1, gain_init=1. if skipinit else 0.)
        if not reg and attn_layer is not None:
            self.attn_last = attn_layer(out_chs)  # ResNet blocks apply attn after conv3
        else:
            self.attn_last = None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.skipinit_gain = nn.Parameter(torch.tensor(0.)) if skipinit else None

    def forward(self, x):
        out = self.act1(x) * self.beta

        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(out)

        # residual branch
        out = self.conv1(out)
        out = self.conv2(self.act2(out))
        if self.conv2b is not None:
            out = self.conv2b(self.act2b(out))
        if self.attn is not None:
            out = self.attn_gain * self.attn(out)
        out = self.conv3(self.act3(out))
        if self.attn_last is not None:
            out = self.attn_gain * self.attn_last(out)
        out = self.drop_path(out)

        if self.skipinit_gain is not None:
            out.mul_(self.skipinit_gain)
        out = out * self.alpha + shortcut
        return out


def create_stem(
        in_chs: int,
        out_chs: int,
        stem_type: str = '',
        conv_layer: Optional[Callable] = None,
        act_layer: Optional[Callable] = None,
        preact_feature: bool = True,
):
    stem_stride = 2
    stem_feature = dict(num_chs=out_chs, reduction=2, module='stem.conv')
    stem = OrderedDict()
    assert stem_type in ('', 'deep', 'deep_tiered', 'deep_quad', '3x3', '7x7', 'deep_pool', '3x3_pool', '7x7_pool')
    if 'deep' in stem_type:
        if 'quad' in stem_type:
            # 4 deep conv stack as in NFNet-F models
            assert not 'pool' in stem_type
            stem_chs = (out_chs // 8, out_chs // 4, out_chs // 2, out_chs)
            strides = (2, 1, 1, 2)
            stem_stride = 4
            stem_feature = dict(num_chs=out_chs // 2, reduction=2, module='stem.conv3')
        else:
            if 'tiered' in stem_type:
                stem_chs = (3 * out_chs // 8, out_chs // 2, out_chs)  # 'T' resnets in resnet.py
            else:
                stem_chs = (out_chs // 2, out_chs // 2, out_chs)  # 'D' ResNets
            strides = (2, 1, 1)
            stem_feature = dict(num_chs=out_chs // 2, reduction=2, module='stem.conv2')
        last_idx = len(stem_chs) - 1
        for i, (c, s) in enumerate(zip(stem_chs, strides)):
            stem[f'conv{i + 1}'] = conv_layer(in_chs, c, kernel_size=3, stride=s)
            if i != last_idx:
                stem[f'act{i + 2}'] = act_layer(inplace=True)
            in_chs = c
    elif '3x3' in stem_type:
        # 3x3 stem conv as in RegNet
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=3, stride=2)
    else:
        # 7x7 stem conv as in ResNet
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=7, stride=2)

    if 'pool' in stem_type:
        stem['pool'] = nn.MaxPool2d(3, stride=2, padding=1)
        stem_stride = 4

    return nn.Sequential(stem), stem_stride, stem_feature


# from https://github.com/deepmind/deepmind-research/tree/master/nfnets
_nonlin_gamma = dict(
    identity=1.0,
    celu=1.270926833152771,
    elu=1.2716004848480225,
    gelu=1.7015043497085571,
    leaky_relu=1.70590341091156,
    log_sigmoid=1.9193484783172607,
    log_softmax=1.0002083778381348,
    relu=1.7139588594436646,
    relu6=1.7131484746932983,
    selu=1.0008515119552612,
    sigmoid=4.803835391998291,
    silu=1.7881293296813965,
    softsign=2.338853120803833,
    softplus=1.9203323125839233,
    tanh=1.5939117670059204,
)


class NormFreeNet(nn.Module):
    """ Normalization-Free Network

    As described in :
    `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
        - https://arxiv.org/abs/2101.08692
    and
    `High-Performance Large-Scale Image Recognition Without Normalization` - https://arxiv.org/abs/2102.06171

    This model aims to cover both the NFRegNet-Bx models as detailed in the paper's code snippets and
    the (preact) ResNet models described earlier in the paper.

    There are a few differences:
        * channels are rounded to be divisible by 8 by default (keep tensor core kernels happy),
            this changes channel dim and param counts slightly from the paper models
        * activation correcting gamma constants are moved into the ScaledStdConv as it has less performance
            impact in PyTorch when done with the weight scaling there. This likely wasn't a concern in the JAX impl.
        * a config option `gamma_in_act` can be enabled to not apply gamma in StdConv as described above, but
            apply it in each activation. This is slightly slower, numerically different, but matches official impl.
        * skipinit is disabled by default, it seems to have a rather drastic impact on GPU memory use and throughput
            for what it is/does. Approx 8-10% throughput loss.
    """
    def __init__(
            self,
            cfg: NfCfg,
            num_classes: int = 1000,
            in_chans: int = 3,
            global_pool: str = 'avg',
            output_stride: int = 32,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            **kwargs,
    ):
        """
        Args:
            cfg: Model architecture configuration.
            num_classes: Number of classifier classes.
            in_chans: Number of input channels.
            global_pool: Global pooling type.
            output_stride: Output stride of network, one of (8, 16, 32).
            drop_rate: Dropout rate.
            drop_path_rate: Stochastic depth drop-path rate.
            **kwargs: Extra kwargs overlayed onto cfg.
        """
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        cfg = replace(cfg, **kwargs)
        assert cfg.act_layer in _nonlin_gamma, f"Please add non-linearity constants for activation ({cfg.act_layer})."
        conv_layer = ScaledStdConv2dSame if cfg.same_padding else ScaledStdConv2d
        if cfg.gamma_in_act:
            act_layer = act_with_gamma(cfg.act_layer, gamma=_nonlin_gamma[cfg.act_layer])
            conv_layer = partial(conv_layer, eps=cfg.std_conv_eps)
        else:
            act_layer = get_act_layer(cfg.act_layer)
            conv_layer = partial(conv_layer, gamma=_nonlin_gamma[cfg.act_layer], eps=cfg.std_conv_eps)
        attn_layer = partial(get_attn(cfg.attn_layer), **cfg.attn_kwargs) if cfg.attn_layer else None

        stem_chs = make_divisible((cfg.stem_chs or cfg.channels[0]) * cfg.width_factor, cfg.ch_div)
        self.stem, stem_stride, stem_feat = create_stem(
            in_chans,
            stem_chs,
            cfg.stem_type,
            conv_layer=conv_layer,
            act_layer=act_layer,
        )

        self.feature_info = [stem_feat]
        drop_path_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]
        prev_chs = stem_chs
        net_stride = stem_stride
        dilation = 1
        expected_var = 1.0
        stages = []
        for stage_idx, stage_depth in enumerate(cfg.depths):
            stride = 1 if stage_idx == 0 and stem_stride > 2 else 2
            if net_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            net_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2

            blocks = []
            for block_idx in range(cfg.depths[stage_idx]):
                first_block = block_idx == 0 and stage_idx == 0
                out_chs = make_divisible(cfg.channels[stage_idx] * cfg.width_factor, cfg.ch_div)
                blocks += [NormFreeBlock(
                    in_chs=prev_chs, out_chs=out_chs,
                    alpha=cfg.alpha,
                    beta=1. / expected_var ** 0.5,
                    stride=stride if block_idx == 0 else 1,
                    dilation=dilation,
                    first_dilation=first_dilation,
                    group_size=cfg.group_size,
                    bottle_ratio=1. if cfg.reg and first_block else cfg.bottle_ratio,
                    ch_div=cfg.ch_div,
                    reg=cfg.reg,
                    extra_conv=cfg.extra_conv,
                    skipinit=cfg.skipinit,
                    attn_layer=attn_layer,
                    attn_gain=cfg.attn_gain,
                    act_layer=act_layer,
                    conv_layer=conv_layer,
                    drop_path_rate=drop_path_rates[stage_idx][block_idx],
                )]
                if block_idx == 0:
                    expected_var = 1.  # expected var is reset after first block of each stage
                expected_var += cfg.alpha ** 2   # Even if reset occurs, increment expected variance
                first_dilation = dilation
                prev_chs = out_chs
            self.feature_info += [dict(num_chs=prev_chs, reduction=net_stride, module=f'stages.{stage_idx}')]
            stages += [nn.Sequential(*blocks)]
        self.stages = nn.Sequential(*stages)

        if cfg.num_features:
            # The paper NFRegNet models have an EfficientNet-like final head convolution.
            self.num_features = make_divisible(cfg.width_factor * cfg.num_features, cfg.ch_div)
            self.final_conv = conv_layer(prev_chs, self.num_features, 1)
            self.feature_info[-1] = dict(num_chs=self.num_features, reduction=net_stride, module=f'final_conv')
        else:
            self.num_features = prev_chs
            self.final_conv = nn.Identity()
        self.final_act = act_layer(inplace=cfg.num_features > 0)

        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=self.drop_rate,
        )

        for n, m in self.named_modules():
            if 'fc' in n and isinstance(m, nn.Linear):
                if cfg.zero_init_fc:
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.normal_(m.weight, 0., .01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',
            blocks=[
                (r'^stages\.(\d+)' if coarse else r'^stages\.(\d+)\.(\d+)', None),
                (r'^final_conv', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        x = self.final_conv(x)
        x = self.final_act(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _nfres_cfg(
        depths,
        channels=(256, 512, 1024, 2048),
        group_size=None,
        act_layer='relu',
        attn_layer=None,
        attn_kwargs=None,
):
    attn_kwargs = attn_kwargs or {}
    cfg = NfCfg(
        depths=depths,
        channels=channels,
        stem_type='7x7_pool',
        stem_chs=64,
        bottle_ratio=0.25,
        group_size=group_size,
        act_layer=act_layer,
        attn_layer=attn_layer,
        attn_kwargs=attn_kwargs,
    )
    return cfg


def _nfreg_cfg(depths, channels=(48, 104, 208, 440)):
    num_features = 1280 * channels[-1] // 440
    attn_kwargs = dict(rd_ratio=0.5)
    cfg = NfCfg(
        depths=depths,
        channels=channels,
        stem_type='3x3',
        group_size=8,
        width_factor=0.75,
        bottle_ratio=2.25,
        num_features=num_features,
        reg=True,
        attn_layer='se',
        attn_kwargs=attn_kwargs,
    )
    return cfg


def _nfnet_cfg(
        depths,
        channels=(256, 512, 1536, 1536),
        group_size=128,
        bottle_ratio=0.5,
        feat_mult=2.,
        act_layer='gelu',
        attn_layer='se',
        attn_kwargs=None,
):
    num_features = int(channels[-1] * feat_mult)
    attn_kwargs = attn_kwargs if attn_kwargs is not None else dict(rd_ratio=0.5)
    cfg = NfCfg(
        depths=depths,
        channels=channels,
        stem_type='deep_quad',
        stem_chs=128,
        group_size=group_size,
        bottle_ratio=bottle_ratio,
        extra_conv=True,
        num_features=num_features,
        act_layer=act_layer,
        attn_layer=attn_layer,
        attn_kwargs=attn_kwargs,
    )
    return cfg


def _dm_nfnet_cfg(
        depths,
        channels=(256, 512, 1536, 1536),
        act_layer='gelu',
        skipinit=True,
):
    cfg = NfCfg(
        depths=depths,
        channels=channels,
        stem_type='deep_quad',
        stem_chs=128,
        group_size=128,
        bottle_ratio=0.5,
        extra_conv=True,
        gamma_in_act=True,
        same_padding=True,
        skipinit=skipinit,
        num_features=int(channels[-1] * 2.0),
        act_layer=act_layer,
        attn_layer='se',
        attn_kwargs=dict(rd_ratio=0.5),
    )
    return cfg


model_cfgs = dict(
    # NFNet-F models w/ GELU compatible with DeepMind weights
    dm_nfnet_f0=_dm_nfnet_cfg(depths=(1, 2, 6, 3)),
    dm_nfnet_f1=_dm_nfnet_cfg(depths=(2, 4, 12, 6)),
    dm_nfnet_f2=_dm_nfnet_cfg(depths=(3, 6, 18, 9)),
    dm_nfnet_f3=_dm_nfnet_cfg(depths=(4, 8, 24, 12)),
    dm_nfnet_f4=_dm_nfnet_cfg(depths=(5, 10, 30, 15)),
    dm_nfnet_f5=_dm_nfnet_cfg(depths=(6, 12, 36, 18)),
    dm_nfnet_f6=_dm_nfnet_cfg(depths=(7, 14, 42, 21)),

    # NFNet-F models w/ GELU
    nfnet_f0=_nfnet_cfg(depths=(1, 2, 6, 3)),
    nfnet_f1=_nfnet_cfg(depths=(2, 4, 12, 6)),
    nfnet_f2=_nfnet_cfg(depths=(3, 6, 18, 9)),
    nfnet_f3=_nfnet_cfg(depths=(4, 8, 24, 12)),
    nfnet_f4=_nfnet_cfg(depths=(5, 10, 30, 15)),
    nfnet_f5=_nfnet_cfg(depths=(6, 12, 36, 18)),
    nfnet_f6=_nfnet_cfg(depths=(7, 14, 42, 21)),
    nfnet_f7=_nfnet_cfg(depths=(8, 16, 48, 24)),

    # Experimental 'light' versions of NFNet-F that are little leaner, w/ SiLU act
    nfnet_l0=_nfnet_cfg(
        depths=(1, 2, 6, 3), feat_mult=1.5, group_size=64, bottle_ratio=0.25,
        attn_kwargs=dict(rd_ratio=0.25, rd_divisor=8), act_layer='silu'),
    eca_nfnet_l0=_nfnet_cfg(
        depths=(1, 2, 6, 3), feat_mult=1.5, group_size=64, bottle_ratio=0.25,
        attn_layer='eca', attn_kwargs=dict(), act_layer='silu'),
    eca_nfnet_l1=_nfnet_cfg(
        depths=(2, 4, 12, 6), feat_mult=2, group_size=64, bottle_ratio=0.25,
        attn_layer='eca', attn_kwargs=dict(), act_layer='silu'),
    eca_nfnet_l2=_nfnet_cfg(
        depths=(3, 6, 18, 9), feat_mult=2, group_size=64, bottle_ratio=0.25,
        attn_layer='eca', attn_kwargs=dict(), act_layer='silu'),
    eca_nfnet_l3=_nfnet_cfg(
        depths=(4, 8, 24, 12), feat_mult=2, group_size=64, bottle_ratio=0.25,
        attn_layer='eca', attn_kwargs=dict(), act_layer='silu'),

    # EffNet influenced RegNet defs.
    # NOTE: These aren't quite the official ver, ch_div=1 must be set for exact ch counts. I round to ch_div=8.
    nf_regnet_b0=_nfreg_cfg(depths=(1, 3, 6, 6)),
    nf_regnet_b1=_nfreg_cfg(depths=(2, 4, 7, 7)),
    nf_regnet_b2=_nfreg_cfg(depths=(2, 4, 8, 8), channels=(56, 112, 232, 488)),
    nf_regnet_b3=_nfreg_cfg(depths=(2, 5, 9, 9), channels=(56, 128, 248, 528)),
    nf_regnet_b4=_nfreg_cfg(depths=(2, 6, 11, 11), channels=(64, 144, 288, 616)),
    nf_regnet_b5=_nfreg_cfg(depths=(3, 7, 14, 14), channels=(80, 168, 336, 704)),

    # ResNet (preact, D style deep stem/avg down) defs
    nf_resnet26=_nfres_cfg(depths=(2, 2, 2, 2)),
    nf_resnet50=_nfres_cfg(depths=(3, 4, 6, 3)),
    nf_resnet101=_nfres_cfg(depths=(3, 4, 23, 3)),

    nf_seresnet26=_nfres_cfg(depths=(2, 2, 2, 2), attn_layer='se', attn_kwargs=dict(rd_ratio=1/16)),
    nf_seresnet50=_nfres_cfg(depths=(3, 4, 6, 3), attn_layer='se', attn_kwargs=dict(rd_ratio=1/16)),
    nf_seresnet101=_nfres_cfg(depths=(3, 4, 23, 3), attn_layer='se', attn_kwargs=dict(rd_ratio=1/16)),

    nf_ecaresnet26=_nfres_cfg(depths=(2, 2, 2, 2), attn_layer='eca', attn_kwargs=dict()),
    nf_ecaresnet50=_nfres_cfg(depths=(3, 4, 6, 3), attn_layer='eca', attn_kwargs=dict()),
    nf_ecaresnet101=_nfres_cfg(depths=(3, 4, 23, 3), attn_layer='eca', attn_kwargs=dict()),
)


def _create_normfreenet(variant, pretrained=False, **kwargs):
    model_cfg = model_cfgs[variant]
    feature_cfg = dict(flatten_sequential=True)
    return build_model_with_cfg(
        NormFreeNet,
        variant,
        pretrained,
        model_cfg=model_cfg,
        feature_cfg=feature_cfg,
        **kwargs,
    )


def _dcfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'dm_nfnet_f0.dm_in1k': _dcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f0-604f9c3a.pth',
        pool_size=(6, 6), input_size=(3, 192, 192), test_input_size=(3, 256, 256), crop_pct=.9, crop_mode='squash'),
    'dm_nfnet_f1.dm_in1k': _dcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f1-fc540f82.pth',
        pool_size=(7, 7), input_size=(3, 224, 224), test_input_size=(3, 320, 320), crop_pct=0.91, crop_mode='squash'),
    'dm_nfnet_f2.dm_in1k': _dcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f2-89875923.pth',
        pool_size=(8, 8), input_size=(3, 256, 256), test_input_size=(3, 352, 352), crop_pct=0.92, crop_mode='squash'),
    'dm_nfnet_f3.dm_in1k': _dcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f3-d74ab3aa.pth',
        pool_size=(10, 10), input_size=(3, 320, 320), test_input_size=(3, 416, 416), crop_pct=0.94, crop_mode='squash'),
    'dm_nfnet_f4.dm_in1k': _dcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f4-0ac5b10b.pth',
        pool_size=(12, 12), input_size=(3, 384, 384), test_input_size=(3, 512, 512), crop_pct=0.951, crop_mode='squash'),
    'dm_nfnet_f5.dm_in1k': _dcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f5-ecb20ab1.pth',
        pool_size=(13, 13), input_size=(3, 416, 416), test_input_size=(3, 544, 544), crop_pct=0.954, crop_mode='squash'),
    'dm_nfnet_f6.dm_in1k': _dcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f6-e0f12116.pth',
        pool_size=(14, 14), input_size=(3, 448, 448), test_input_size=(3, 576, 576), crop_pct=0.956, crop_mode='squash'),

    'nfnet_f0': _dcfg(
        url='', pool_size=(6, 6), input_size=(3, 192, 192), test_input_size=(3, 256, 256)),
    'nfnet_f1': _dcfg(
        url='', pool_size=(7, 7), input_size=(3, 224, 224), test_input_size=(3, 320, 320)),
    'nfnet_f2': _dcfg(
        url='', pool_size=(8, 8), input_size=(3, 256, 256), test_input_size=(3, 352, 352)),
    'nfnet_f3': _dcfg(
        url='', pool_size=(10, 10), input_size=(3, 320, 320), test_input_size=(3, 416, 416)),
    'nfnet_f4': _dcfg(
        url='', pool_size=(12, 12), input_size=(3, 384, 384), test_input_size=(3, 512, 512)),
    'nfnet_f5': _dcfg(
        url='', pool_size=(13, 13), input_size=(3, 416, 416), test_input_size=(3, 544, 544)),
    'nfnet_f6': _dcfg(
        url='', pool_size=(14, 14), input_size=(3, 448, 448), test_input_size=(3, 576, 576)),
    'nfnet_f7': _dcfg(
        url='', pool_size=(15, 15), input_size=(3, 480, 480), test_input_size=(3, 608, 608)),

    'nfnet_l0.ra2_in1k': _dcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/nfnet_l0_ra2-45c6688d.pth',
        pool_size=(7, 7), input_size=(3, 224, 224), test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'eca_nfnet_l0.ra2_in1k': _dcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecanfnet_l0_ra2-e3e9ac50.pth',
        pool_size=(7, 7), input_size=(3, 224, 224), test_input_size=(3, 288, 288), test_crop_pct=1.0),
    'eca_nfnet_l1.ra2_in1k': _dcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecanfnet_l1_ra2-7dce93cd.pth',
        pool_size=(8, 8), input_size=(3, 256, 256), test_input_size=(3, 320, 320), test_crop_pct=1.0),
    'eca_nfnet_l2.ra3_in1k': _dcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecanfnet_l2_ra3-da781a61.pth',
        pool_size=(10, 10), input_size=(3, 320, 320), test_input_size=(3, 384, 384), test_crop_pct=1.0),
    'eca_nfnet_l3': _dcfg(
        url='',
        pool_size=(11, 11), input_size=(3, 352, 352), test_input_size=(3, 448, 448), test_crop_pct=1.0),

    'nf_regnet_b0': _dcfg(
        url='', pool_size=(6, 6), input_size=(3, 192, 192), test_input_size=(3, 256, 256), first_conv='stem.conv'),
    'nf_regnet_b1.ra2_in1k': _dcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/nf_regnet_b1_256_ra2-ad85cfef.pth',
        pool_size=(8, 8), input_size=(3, 256, 256), test_input_size=(3, 288, 288), first_conv='stem.conv'),  # NOT to paper spec
    'nf_regnet_b2': _dcfg(
        url='', pool_size=(8, 8), input_size=(3, 240, 240), test_input_size=(3, 272, 272), first_conv='stem.conv'),
    'nf_regnet_b3': _dcfg(
        url='', pool_size=(9, 9), input_size=(3, 288, 288), test_input_size=(3, 320, 320), first_conv='stem.conv'),
    'nf_regnet_b4': _dcfg(
        url='', pool_size=(10, 10), input_size=(3, 320, 320), test_input_size=(3, 384, 384), first_conv='stem.conv'),
    'nf_regnet_b5': _dcfg(
        url='', pool_size=(12, 12), input_size=(3, 384, 384), test_input_size=(3, 456, 456), first_conv='stem.conv'),

    'nf_resnet26': _dcfg(url='', first_conv='stem.conv'),
    'nf_resnet50.ra2_in1k': _dcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/nf_resnet50_ra2-9f236009.pth',
        pool_size=(8, 8), input_size=(3, 256, 256), test_input_size=(3, 288, 288), crop_pct=0.94, first_conv='stem.conv'),
    'nf_resnet101': _dcfg(url='', first_conv='stem.conv'),

    'nf_seresnet26': _dcfg(url='', first_conv='stem.conv'),
    'nf_seresnet50': _dcfg(url='', first_conv='stem.conv'),
    'nf_seresnet101': _dcfg(url='', first_conv='stem.conv'),

    'nf_ecaresnet26': _dcfg(url='', first_conv='stem.conv'),
    'nf_ecaresnet50': _dcfg(url='', first_conv='stem.conv'),
    'nf_ecaresnet101': _dcfg(url='', first_conv='stem.conv'),
})


@register_model
def dm_nfnet_f0(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-F0 (DeepMind weight compatible)
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return _create_normfreenet('dm_nfnet_f0', pretrained=pretrained, **kwargs)


@register_model
def dm_nfnet_f1(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-F1 (DeepMind weight compatible)
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return _create_normfreenet('dm_nfnet_f1', pretrained=pretrained, **kwargs)


@register_model
def dm_nfnet_f2(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-F2 (DeepMind weight compatible)
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return _create_normfreenet('dm_nfnet_f2', pretrained=pretrained, **kwargs)


@register_model
def dm_nfnet_f3(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-F3 (DeepMind weight compatible)
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return _create_normfreenet('dm_nfnet_f3', pretrained=pretrained, **kwargs)


@register_model
def dm_nfnet_f4(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-F4 (DeepMind weight compatible)
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return _create_normfreenet('dm_nfnet_f4', pretrained=pretrained, **kwargs)


@register_model
def dm_nfnet_f5(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-F5 (DeepMind weight compatible)
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return _create_normfreenet('dm_nfnet_f5', pretrained=pretrained, **kwargs)


@register_model
def dm_nfnet_f6(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-F6 (DeepMind weight compatible)
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return _create_normfreenet('dm_nfnet_f6', pretrained=pretrained, **kwargs)


@register_model
def nfnet_f0(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-F0
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return _create_normfreenet('nfnet_f0', pretrained=pretrained, **kwargs)


@register_model
def nfnet_f1(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-F1
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return _create_normfreenet('nfnet_f1', pretrained=pretrained, **kwargs)


@register_model
def nfnet_f2(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-F2
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return _create_normfreenet('nfnet_f2', pretrained=pretrained, **kwargs)


@register_model
def nfnet_f3(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-F3
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return _create_normfreenet('nfnet_f3', pretrained=pretrained, **kwargs)


@register_model
def nfnet_f4(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-F4
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return _create_normfreenet('nfnet_f4', pretrained=pretrained, **kwargs)


@register_model
def nfnet_f5(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-F5
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return _create_normfreenet('nfnet_f5', pretrained=pretrained, **kwargs)


@register_model
def nfnet_f6(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-F6
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return _create_normfreenet('nfnet_f6', pretrained=pretrained, **kwargs)


@register_model
def nfnet_f7(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-F7
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return _create_normfreenet('nfnet_f7', pretrained=pretrained, **kwargs)


@register_model
def nfnet_l0(pretrained=False, **kwargs) -> NormFreeNet:
    """ NFNet-L0b w/ SiLU
    My experimental 'light' model w/ F0 repeats, 1.5x final_conv mult, 64 group_size, .25 bottleneck & SE ratio
    """
    return _create_normfreenet('nfnet_l0', pretrained=pretrained, **kwargs)


@register_model
def eca_nfnet_l0(pretrained=False, **kwargs) -> NormFreeNet:
    """ ECA-NFNet-L0 w/ SiLU
    My experimental 'light' model w/ F0 repeats, 1.5x final_conv mult, 64 group_size, .25 bottleneck & ECA attn
    """
    return _create_normfreenet('eca_nfnet_l0', pretrained=pretrained, **kwargs)


@register_model
def eca_nfnet_l1(pretrained=False, **kwargs) -> NormFreeNet:
    """ ECA-NFNet-L1 w/ SiLU
    My experimental 'light' model w/ F1 repeats, 2.0x final_conv mult, 64 group_size, .25 bottleneck & ECA attn
    """
    return _create_normfreenet('eca_nfnet_l1', pretrained=pretrained, **kwargs)


@register_model
def eca_nfnet_l2(pretrained=False, **kwargs) -> NormFreeNet:
    """ ECA-NFNet-L2 w/ SiLU
    My experimental 'light' model w/ F2 repeats, 2.0x final_conv mult, 64 group_size, .25 bottleneck & ECA attn
    """
    return _create_normfreenet('eca_nfnet_l2', pretrained=pretrained, **kwargs)


@register_model
def eca_nfnet_l3(pretrained=False, **kwargs) -> NormFreeNet:
    """ ECA-NFNet-L3 w/ SiLU
    My experimental 'light' model w/ F3 repeats, 2.0x final_conv mult, 64 group_size, .25 bottleneck & ECA attn
    """
    return _create_normfreenet('eca_nfnet_l3', pretrained=pretrained, **kwargs)


@register_model
def nf_regnet_b0(pretrained=False, **kwargs) -> NormFreeNet:
    """ Normalization-Free RegNet-B0
    `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
        - https://arxiv.org/abs/2101.08692
    """
    return _create_normfreenet('nf_regnet_b0', pretrained=pretrained, **kwargs)


@register_model
def nf_regnet_b1(pretrained=False, **kwargs) -> NormFreeNet:
    """ Normalization-Free RegNet-B1
    `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
        - https://arxiv.org/abs/2101.08692
    """
    return _create_normfreenet('nf_regnet_b1', pretrained=pretrained, **kwargs)


@register_model
def nf_regnet_b2(pretrained=False, **kwargs) -> NormFreeNet:
    """ Normalization-Free RegNet-B2
    `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
        - https://arxiv.org/abs/2101.08692
    """
    return _create_normfreenet('nf_regnet_b2', pretrained=pretrained, **kwargs)


@register_model
def nf_regnet_b3(pretrained=False, **kwargs) -> NormFreeNet:
    """ Normalization-Free RegNet-B3
    `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
        - https://arxiv.org/abs/2101.08692
    """
    return _create_normfreenet('nf_regnet_b3', pretrained=pretrained, **kwargs)


@register_model
def nf_regnet_b4(pretrained=False, **kwargs) -> NormFreeNet:
    """ Normalization-Free RegNet-B4
    `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
        - https://arxiv.org/abs/2101.08692
    """
    return _create_normfreenet('nf_regnet_b4', pretrained=pretrained, **kwargs)


@register_model
def nf_regnet_b5(pretrained=False, **kwargs) -> NormFreeNet:
    """ Normalization-Free RegNet-B5
    `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
        - https://arxiv.org/abs/2101.08692
    """
    return _create_normfreenet('nf_regnet_b5', pretrained=pretrained, **kwargs)


@register_model
def nf_resnet26(pretrained=False, **kwargs) -> NormFreeNet:
    """ Normalization-Free ResNet-26
    `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
        - https://arxiv.org/abs/2101.08692
    """
    return _create_normfreenet('nf_resnet26', pretrained=pretrained, **kwargs)


@register_model
def nf_resnet50(pretrained=False, **kwargs) -> NormFreeNet:
    """ Normalization-Free ResNet-50
    `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
        - https://arxiv.org/abs/2101.08692
    """
    return _create_normfreenet('nf_resnet50', pretrained=pretrained, **kwargs)


@register_model
def nf_resnet101(pretrained=False, **kwargs) -> NormFreeNet:
    """ Normalization-Free ResNet-101
    `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
        - https://arxiv.org/abs/2101.08692
    """
    return _create_normfreenet('nf_resnet101', pretrained=pretrained, **kwargs)


@register_model
def nf_seresnet26(pretrained=False, **kwargs) -> NormFreeNet:
    """ Normalization-Free SE-ResNet26
    """
    return _create_normfreenet('nf_seresnet26', pretrained=pretrained, **kwargs)


@register_model
def nf_seresnet50(pretrained=False, **kwargs) -> NormFreeNet:
    """ Normalization-Free SE-ResNet50
    """
    return _create_normfreenet('nf_seresnet50', pretrained=pretrained, **kwargs)


@register_model
def nf_seresnet101(pretrained=False, **kwargs) -> NormFreeNet:
    """ Normalization-Free SE-ResNet101
    """
    return _create_normfreenet('nf_seresnet101', pretrained=pretrained, **kwargs)


@register_model
def nf_ecaresnet26(pretrained=False, **kwargs) -> NormFreeNet:
    """ Normalization-Free ECA-ResNet26
    """
    return _create_normfreenet('nf_ecaresnet26', pretrained=pretrained, **kwargs)


@register_model
def nf_ecaresnet50(pretrained=False, **kwargs) -> NormFreeNet:
    """ Normalization-Free ECA-ResNet50
    """
    return _create_normfreenet('nf_ecaresnet50', pretrained=pretrained, **kwargs)


@register_model
def nf_ecaresnet101(pretrained=False, **kwargs) -> NormFreeNet:
    """ Normalization-Free ECA-ResNet101
    """
    return _create_normfreenet('nf_ecaresnet101', pretrained=pretrained, **kwargs)
