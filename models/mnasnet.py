""" MNASNet (a1, b1, and small)

Based on offical TF implementation w/ round_channels,
decode_block_str, and model block args directly transferred
https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet

Original paper: https://arxiv.org/pdf/1807.11626.pdf.

Hacked together by Ross Wightman
"""

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.helpers import load_pretrained
from models.adaptive_avgmax_pool import SelectAdaptivePool2d
from data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

__all__ = ['MnasNet', 'mnasnet0_50', 'mnasnet0_75', 'mnasnet1_00', 'mnasnet1_40',
           'semnasnet0_50', 'semnasnet0_75', 'semnasnet1_00', 'semnasnet1_40',
           'mnasnet_small']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'layer0.conv1', 'classifier': 'last_linear',
        **kwargs
    }

default_cfgs = {
    'mnasnet0_50': _cfg(url=''),
    'mnasnet0_75': _cfg(url=''),
    'mnasnet1_00': _cfg(url=''),
    'mnasnet1_40': _cfg(url=''),
    'semnasnet0_50': _cfg(url=''),
    'semnasnet0_75': _cfg(url=''),
    'semnasnet1_00': _cfg(url=''),
    'semnasnet1_40': _cfg(url=''),
    'mnasnet_small': _cfg(url=''),
}

_BN_MOMENTUM_DEFAULT = 1 - 0.99
_BN_EPS_DEFAULT = 1e-3


def _round_channels(channels, depth_multiplier=1.0, depth_divisor=8, min_depth=None):
    """Round number of filters based on depth multiplier."""
    multiplier = depth_multiplier
    divisor = depth_divisor
    min_depth = min_depth
    if not multiplier:
        return channels

    channels *= multiplier
    min_depth = min_depth or divisor
    new_channels = max(min_depth, int(channels + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return new_channels


def _decode_block_str(block_str):
    """Gets a MNasNet block through a string notation of arguments.
    E.g. r2_k3_s2_e1_i32_o16_se0.25_noskip:
    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    i - input filters,
    o - output filters,
    se - squeeze/excitation ratio
    Args:
        block_string: a string, a string representation of block arguments.
    Returns:
        A BlockArgs instance.
    Raises:
        ValueError: if the strides option is not correctly specified.
    """
    assert isinstance(block_str, str)
    ops = block_str.split('_')
    options = {}
    for op in ops:
        splits = re.split(r'(\d.*)', op)
        if len(splits) >= 2:
            key, value = splits[:2]
            options[key] = value

    if 's' not in options or len(options['s']) != 2:
        raise ValueError('Strides options should be a pair of integers.')

    return dict(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        in_chs=int(options['i']),
        out_chs=int(options['o']),
        exp_ratio=int(options['e']),
        id_skip=('noskip' not in block_str),
        se_ratio=float(options['se']) if 'se' in options else None,
        stride=int(options['s'][0])  # TF impl passes a list of two strides
    )


def _decode_block_args(string_list):
    block_args = []
    for block_str in string_list:
        block_args.append(_decode_block_str(block_str))
    return block_args


def _initialize_weight(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        init_range = 1.0 / math.sqrt(n)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


class MnasBlock(nn.Module):
    """ MNASNet Inverted residual block w/ optional SE"""

    def __init__(self, in_chs, out_chs, kernel_size, stride,
                 exp_ratio=1.0, id_skip=True, se_ratio=0.,
                 bn_momentum=0.1, bn_eps=1e-3, act_fn=F.relu):
        super(MnasBlock, self).__init__()
        exp_chs = int(in_chs * exp_ratio)
        self.has_expansion = exp_ratio != 1
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = id_skip and (in_chs == out_chs and stride == 1)
        self.act_fn = act_fn

        # Pointwise expansion
        if self.has_expansion:
            self.conv_expand = nn.Conv2d(in_chs, exp_chs, 1, bias=False)
            self.bn0 = nn.BatchNorm2d(exp_chs, momentum=bn_momentum, eps=bn_eps)

        # Depthwise convolution
        self.conv_depthwise = nn.Conv2d(
            exp_chs, exp_chs, kernel_size, padding=kernel_size // 2,
            stride=stride, groups=exp_chs, bias=False)
        self.bn1 = nn.BatchNorm2d(exp_chs, momentum=bn_momentum, eps=bn_eps)

        # Squeeze-and-excitation
        if self.has_se:
            num_reduced_ch = max(1, int(in_chs * se_ratio))
            self.conv_se_reduce = nn.Conv2d(exp_chs, num_reduced_ch, 1, bias=True)
            self.conv_se_expand = nn.Conv2d(num_reduced_ch, exp_chs, 1, bias=True)

        # Pointwise projection
        self.conv_project = nn.Conv2d(exp_chs, out_chs, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chs, momentum=bn_momentum, eps=bn_eps)

    def forward(self, x):
        residual = x
        # Pointwise expansion
        if self.has_expansion:
            x = self.conv_expand(x)
            x = self.bn0(x)
            x = self.act_fn(x)
        # Depthwise convolution
        x = self.conv_depthwise(x)
        x = self.bn1(x)
        x = self.act_fn(x)
        # Squeeze-and-excitation
        if self.has_se:
            x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
            x_se = self.conv_se_reduce(x_se)
            x_se = F.relu(x_se)
            x_se = self.conv_se_expand(x_se)
            x = F.sigmoid(x_se) * x
        # Pointwise projection
        x = self.conv_project(x)
        x = self.bn2(x)
        # Residual
        if self.has_residual:
            return x + residual
        else:
            return x


class MnasNet(nn.Module):
    """ MNASNet

    Based on offical TF implementation
    https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet

    Original paper: https://arxiv.org/pdf/1807.11626.pdf.
    """

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=32,
                 depth_multiplier=1.0, depth_divisor=8, min_depth=None,
                 bn_momentum=_BN_MOMENTUM_DEFAULT, bn_eps=_BN_EPS_DEFAULT, drop_rate=0.,
                 global_pool='avg', act_fn=F.relu):
        super(MnasNet, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.global_pool = global_pool
        self.act_fn = act_fn
        self.depth_multiplier = depth_multiplier
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.num_features = 1280

        self.conv_stem = nn.Conv2d(in_chans, stem_size, 3, padding=1, stride=2, bias=False)
        self.bn0 = nn.BatchNorm2d(stem_size, momentum=self.bn_momentum, eps=self.bn_eps)

        blocks = []
        for i, a in enumerate(block_args):
            print(a) #FIXME debug
            a['in_chs'] = _round_channels(a['in_chs'], depth_multiplier, depth_divisor, min_depth)
            a['out_chs'] = _round_channels(a['out_chs'], depth_multiplier, depth_divisor, min_depth)
            blocks.append(self._make_stack(**a))
            out_chs = a['out_chs']
        self.blocks = nn.Sequential(*blocks)

        self.conv_head = nn.Conv2d(out_chs, self.num_features, 1, padding=0, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_features, momentum=self.bn_momentum, eps=self.bn_eps)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.num_features, self.num_classes)

        for m in self.modules():
            _initialize_weight(m)

    def _make_stack(self, in_chs, out_chs, kernel_size, stride,
                    exp_ratio, id_skip, se_ratio, num_repeat):
        blocks = [MnasBlock(
            in_chs, out_chs, kernel_size, stride, exp_ratio, id_skip, se_ratio,
            bn_momentum=self.bn_momentum, bn_eps=self.bn_eps)]
        for _ in range(1, num_repeat):
            blocks += [MnasBlock(
                out_chs, out_chs, kernel_size, 1, exp_ratio, id_skip, se_ratio,
                bn_momentum=self.bn_momentum, bn_eps=self.bn_eps)]
        return nn.Sequential(*blocks)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        #self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        del self.classifier
        if num_classes:
            self.classifier = nn.Linear(self.num_features, num_classes)
        else:
            self.classifier = None

    def forward_features(self, x, pool=True):
        x = self.conv_stem(x)
        x = self.bn0(x)
        x = self.act_fn(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn1(x)
        x = self.act_fn(x)
        if pool:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


def mnasnet_a1(depth_multiplier, num_classes=1000, **kwargs):
    """Creates a mnasnet-a1 model.
    Args:
      depth_multiplier: multiplier to number of filters per layer.
    """

    # defs from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    block_defs = [
        'r1_k3_s11_e1_i32_o16_noskip',
        'r2_k3_s22_e6_i16_o24',
        'r3_k5_s22_e3_i24_o40_se0.25',
        'r4_k3_s22_e6_i40_o80',
        'r2_k3_s11_e6_i80_o112_se0.25',
        'r3_k5_s22_e6_i112_o160_se0.25',
        'r1_k3_s11_e6_i160_o320'
    ]
    block_args = _decode_block_args(block_defs)
    model = MnasNet(
        block_args,
        num_classes=num_classes,
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None,
        stem_size=32,
        bn_momentum=_BN_MOMENTUM_DEFAULT,
        bn_eps=_BN_EPS_DEFAULT,
        #drop_rate=0.2,
        **kwargs
        )
    return model


def mnasnet_b1(depth_multiplier, num_classes=1000, **kwargs):
    """Creates a mnasnet-b1 model.
    Args:
      depth_multiplier: multiplier to number of filters per layer.
    """
    # from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    blocks_defs = [
        'r1_k3_s11_e1_i32_o16_noskip',
        'r3_k3_s22_e3_i16_o24',
        'r3_k5_s22_e3_i24_o40',
        'r3_k5_s22_e6_i40_o80',
        'r2_k3_s11_e6_i80_o96',
        'r4_k5_s22_e6_i96_o192',
        'r1_k3_s11_e6_i192_o320_noskip'
    ]
    block_args = _decode_block_args(blocks_defs)
    model = MnasNet(
        block_args,
        num_classes=num_classes,
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None,
        stem_size=32,
        bn_momentum=_BN_MOMENTUM_DEFAULT,
        bn_eps=_BN_EPS_DEFAULT,
        #drop_rate=0.2,
        **kwargs
    )
    return model


def mnasnet_small(depth_multiplier, num_classes=1000, **kwargs):
    """Creates a mnasnet-b1 model.
    Args:
      depth_multiplier: multiplier to number of filters per layer.
    """
    # from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    blocks_defs = [
        'r1_k3_s11_e1_i16_o8',
        'r1_k3_s22_e3_i8_o16',
        'r2_k3_s22_e6_i16_o16',
        'r4_k5_s22_e6_i16_o32_se0.25',
        'r3_k3_s11_e6_i32_o32_se0.25',
        'r3_k5_s22_e6_i32_o88_se0.25',
        'r1_k3_s11_e6_i88_o144'
    ]
    block_args = _decode_block_args(blocks_defs)
    model = MnasNet(
        block_args,
        num_classes=num_classes,
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None,
        stem_size=8,
        bn_momentum=_BN_MOMENTUM_DEFAULT,
        bn_eps=_BN_EPS_DEFAULT,
        #drop_rate=0.2,
        **kwargs
    )
    return model


def mnasnet0_50(num_classes=1000, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet B1, depth multiplier of 0.5. """
    default_cfg = default_cfgs['mnasnet0_50']
    model = mnasnet_b1(0.5, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    return model


def mnasnet0_75(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet B1, depth multiplier of 0.75. """
    default_cfg = default_cfgs['mnasnet0_50']
    model = mnasnet_b1(0.75, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    return model


def mnasnet1_00(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet B1, depth multiplier of 1.0. """
    default_cfg = default_cfgs['mnasnet0_50']
    model = mnasnet_b1(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    return model


def mnasnet1_40(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet B1,  depth multiplier of 1.4 """
    default_cfg = default_cfgs['mnasnet0_50']
    model = mnasnet_b1(1.4, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    return model


def semnasnet0_50(num_classes=1000, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 0.5 """
    default_cfg = default_cfgs['mnasnet0_50']
    model = mnasnet_a1(0.5, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    return model


def semnasnet0_75(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE),  depth multiplier of 0.75. """
    default_cfg = default_cfgs['mnasnet0_50']
    model = mnasnet_a1(0.75, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    return model


def semnasnet1_00(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet Small,  depth multiplier of 1.0. """
    default_cfg = default_cfgs['mnasnet0_50']
    model = mnasnet_a1(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    return model


def semnasnet1_40(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet with depth multiplier of 1.3. """
    default_cfg = default_cfgs['mnasnet0_50']
    model = mnasnet_a1(1.4, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    return model


def mnasnet_small(num_classes, in_chans=3, pretrained=False, **kwargs):
    default_cfg = default_cfgs['mnasnet_small']
    model = mnasnet_small(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    return model
