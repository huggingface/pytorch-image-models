""" Generic MobileNet

A generic MobileNet class with building blocks to support a variety of models:
* MNasNet B1, A1 (SE), Small
* MobileNetV2
* FBNet-C (TODO A & B)
* ChamNet (TODO still guessing at architecture definition)
* Single-Path NAS Pixel1
* ShuffleNetV2 (TODO add IR shuffle block)
* And likely more...

TODO not all combinations and variations have been tested. Currently working on training hyper-params...

Hacked together by Ross Wightman
"""

import math
import re
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.helpers import load_pretrained
from models.adaptive_avgmax_pool import SelectAdaptivePool2d
from models.conv2d_same import sconv2d
from data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

__all__ = ['GenMobileNet', 'mnasnet0_50', 'mnasnet0_75', 'mnasnet1_00', 'mnasnet1_40',
           'semnasnet0_50', 'semnasnet0_75', 'semnasnet1_00', 'semnasnet1_40', 'mnasnet_small',
           'mobilenetv1_1_00', 'mobilenetv2_1_00', 'chamnetv1_1_00', 'chamnetv2_1_00',
           'fbnetc_1_00', 'spnasnet1_00']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'mnasnet0_50': _cfg(url=''),
    'mnasnet0_75': _cfg(url=''),
    'mnasnet1_00': _cfg(url=''),
    'tflite_mnasnet1_00': _cfg(url='', interpolation='bicubic'),
    'mnasnet1_40': _cfg(url=''),
    'semnasnet0_50': _cfg(url=''),
    'semnasnet0_75': _cfg(url=''),
    'semnasnet1_00': _cfg(url=''),
    'tflite_semnasnet1_00': _cfg(url='', interpolation='bicubic'),
    'semnasnet1_40': _cfg(url=''),
    'mnasnet_small': _cfg(url=''),
    'mobilenetv1_1_00': _cfg(url=''),
    'mobilenetv2_1_00': _cfg(url=''),
    'chamnetv1_1_00': _cfg(url=''),
    'chamnetv2_1_00': _cfg(url=''),
    'fbnetc_1_00': _cfg(url=''),
    'spnasnet1_00': _cfg(url='https://www.dropbox.com/s/iieopt18rytkgaa/spnasnet1_00-048bc3f4.pth?dl=1'),
}

_DEBUG = True

# Default args for PyTorch BN impl
_BN_MOMENTUM_PT_DEFAULT = 0.1
_BN_EPS_PT_DEFAULT = 1e-5

# Defaults used for Google/Tensorflow training of mobile networks /w RMSprop as per
# papers and TF reference implementations. PT momentum equiv for TF decay is (1 - TF decay)
# NOTE: momentum varies btw .99 and .9997 depending on source
# .99 in official TF TPU impl
# .9997 (/w .999 in search space) for paper
_BN_MOMENTUM_TF_DEFAULT = 1 - 0.99
_BN_EPS_TF_DEFAULT = 1e-3


def _resolve_bn_params(kwargs):
    # NOTE kwargs passed as dict intentionally
    bn_momentum_default = _BN_MOMENTUM_PT_DEFAULT
    bn_eps_default = _BN_EPS_PT_DEFAULT
    bn_tf = kwargs.pop('bn_tf', False)
    if bn_tf:
        bn_momentum_default = _BN_MOMENTUM_TF_DEFAULT
        bn_eps_default = _BN_EPS_TF_DEFAULT
    bn_momentum = kwargs.pop('bn_momentum', None)
    bn_eps = kwargs.pop('bn_eps', None)
    if bn_momentum is None:
        bn_momentum = bn_momentum_default
    if bn_eps is None:
        bn_eps = bn_eps_default
    return bn_momentum, bn_eps


def _round_channels(channels, depth_multiplier=1.0, depth_divisor=8, min_depth=None):
    """Round number of filters based on depth multiplier."""
    if not depth_multiplier:
        return channels

    channels *= depth_multiplier
    min_depth = min_depth or depth_divisor
    new_channels = max(
        int(channels + depth_divisor / 2) // depth_divisor * depth_divisor,
        min_depth)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += depth_divisor
    return new_channels


def _decode_block_str(block_str):
    """ Decode block definition string

    Gets a list of block arg (dicts) through a string notation of arguments.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip

    All args can exist in any order with the exception of the leading string which
    is assumed to indicate the block type.

    leading string - block type (
      ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act,
      ca = Cascade3x3, and possibly more)
    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    c - output channels,
    se - squeeze/excitation ratio
    Args:
        block_str: a string representation of block arguments.
    Returns:
        A list of block args (dicts)
    Raises:
        ValueError: if the string def not properly specified (TODO)
    """
    assert isinstance(block_str, str)
    ops = block_str.split('_')
    block_type = ops[0]  # take the block type off the front
    ops = ops[1:]
    options = {}
    for op in ops:
        splits = re.split(r'(\d.*)', op)
        if len(splits) >= 2:
            key, value = splits[:2]
            options[key] = value

    # FIXME validate args and throw

    num_repeat = int(options['r'])
    # each type of block has different valid arguments, fill accordingly
    if block_type == 'ir':
        block_args = dict(
            block_type=block_type,
            kernel_size=int(options['k']),
            out_chs=int(options['c']),
            exp_ratio=int(options['e']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            noskip=('noskip' in block_str),
        )
        if 'g' in options:
            block_args['pw_group'] = options['g']
            if options['g'] > 1:
                block_args['shuffle_type'] = 'mid'
    elif block_type == 'ca':
        block_args = dict(
            block_type=block_type,
            out_chs=int(options['c']),
            stride=int(options['s']),
            noskip=('noskip' in block_str),
        )
    elif block_type == 'ds' or block_type == 'dsa':
        block_args = dict(
            block_type=block_type,
            kernel_size=int(options['k']),
            out_chs=int(options['c']),
            stride=int(options['s']),
            noskip=block_type == 'dsa' or 'noskip' in block_str,
            pw_act=block_type == 'dsa',
        )
    else:
        assert False, 'Unknown block type (%s)' % block_type

    # return a list of block args expanded by num_repeat
    return [deepcopy(block_args) for _ in range(num_repeat)]


def _get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def _padding_arg(default, padding_same=False):
    return 'SAME' if padding_same else default


def _decode_arch_args(string_list):
    block_args = []
    for block_str in string_list:
        block_args.append(_decode_block_str(block_str))
    return block_args


def _decode_arch_def(arch_def):
    arch_args = []
    for stack_idx, block_strings in enumerate(arch_def):
        assert isinstance(block_strings, list)
        stack_args = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            stack_args.extend(_decode_block_str(block_str))
        arch_args.append(stack_args)
    return arch_args


class _BlockBuilder:
    """ Build Trunk Blocks

    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py

    """

    def __init__(self, depth_multiplier=1.0, depth_divisor=8, min_depth=None,
                 bn_momentum=_BN_MOMENTUM_PT_DEFAULT, bn_eps=_BN_EPS_PT_DEFAULT,
                 folded_bn=False, padding_same=False):
        self.depth_multiplier = depth_multiplier
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.folded_bn = folded_bn
        self.padding_same = padding_same
        self.in_chs = None

    def _round_channels(self, chs):
        return _round_channels(chs, self.depth_multiplier, self.depth_divisor, self.min_depth)

    def _make_block(self, ba):
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = _round_channels(ba['out_chs'])
        ba['bn_momentum'] = self.bn_momentum
        ba['bn_eps'] = self.bn_eps
        ba['folded_bn'] = self.folded_bn
        ba['padding_same'] = self.padding_same
        if _DEBUG:
            print('args:', ba)
        #  could replace this with lambdas or functools binding if variety increases
        if bt == 'ir':
            block = InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'ca':
            block = CascadeConv3x3(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt
        self.in_chs = ba['out_chs']  # update in_chs for arg of next block
        return block

    def _make_stack(self, stack_args):
        blocks = []
        # each stack (stage) contains a list of block arguments
        for block_idx, ba in enumerate(stack_args):
            if _DEBUG:
                print('block', block_idx, end=', ')
            if block_idx >= 1:
                # only the first block in any stack/stage can have a stride > 1
                ba['stride'] = 1
            block = self._make_block(ba)
            blocks.append(block)
        return nn.Sequential(*blocks)

    def __call__(self, in_chs, arch_def):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            arch_def: A list of lists, outer list defines stacks (or stages), inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        arch_args = _decode_arch_def(arch_def)  # convert and expand string defs to arg dicts
        if _DEBUG:
            print('Building model trunk with %d stacks (stages)...' % len(arch_args))
        self.in_chs = in_chs
        blocks = []
        # outer list of arch_args defines the stacks ('stages' by some conventions)
        for stack_idx, stack in enumerate(arch_args):
            if _DEBUG:
                print('stack', stack_idx)
            assert isinstance(stack, list)
            stack = self._make_stack(stack)
            blocks.append(stack)
            if _DEBUG:
                print()
        return blocks


def _initialize_weight_goog(m):
    # weight init as per Tensorflow Official impl
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # fan-out
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(0)  # fan-out
        init_range = 1.0 / math.sqrt(n)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def _initialize_weight_default(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear')


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_fn=F.relu, noskip=False, pw_act=False,
                 bn_momentum=_BN_MOMENTUM_PT_DEFAULT, bn_eps=_BN_EPS_PT_DEFAULT,
                 folded_bn=False, padding_same=False):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.act_fn = act_fn
        dw_padding = _padding_arg(kernel_size // 2, padding_same)
        pw_padding = _padding_arg(0, padding_same)

        self.conv_dw = sconv2d(
            in_chs, in_chs, kernel_size,
            stride=stride, padding=dw_padding, groups=in_chs, bias=folded_bn)
        self.bn1 = None if folded_bn else nn.BatchNorm2d(in_chs, momentum=bn_momentum, eps=bn_eps)
        self.conv_pw = sconv2d(in_chs, out_chs, 1, padding=pw_padding, bias=folded_bn)
        self.bn2 = None if folded_bn else nn.BatchNorm2d(out_chs, momentum=bn_momentum, eps=bn_eps)

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act_fn(x)

        x = self.conv_pw(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        if self.has_pw_act:
            x = self.act_fn(x)

        if self.has_residual:
            x += residual
        return x


class CascadeConv3x3(nn.Sequential):
    # FIXME lifted from maskrcnn_benchmark blocks, haven't used yet
    def __init__(self, in_chs, out_chs, stride, act_fn=F.relu, noskip=False,
                 bn_momentum=_BN_MOMENTUM_PT_DEFAULT, bn_eps=_BN_EPS_PT_DEFAULT,
                 folded_bn=False, padding_same=False):
        super(CascadeConv3x3, self).__init__()
        assert stride in [1, 2]
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.act_fn = act_fn
        padding = _padding_arg(1, padding_same)

        self.conv1 = sconv2d(in_chs, in_chs, 3, stride=stride, padding=padding, bias=folded_bn)
        self.bn1 = None if folded_bn else nn.BatchNorm2d(in_chs, momentum=bn_momentum, eps=bn_eps)
        self.conv2 = sconv2d(in_chs, out_chs, 3, stride=1, padding=padding, bias=folded_bn)
        self.bn2 = None if folded_bn else nn.BatchNorm2d(out_chs, momentum=bn_momentum, eps=bn_eps)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act_fn(x)
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        if self.has_residual:
            x += residual
        return x


class ChannelShuffle(nn.Module):
    # FIXME lifted from maskrcnn_benchmark blocks, haven't used yet
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, reduce_chs=None, act_fn=F.relu):
        super(SqueezeExcite, self).__init__()
        self.act_fn = act_fn
        reduced_chs = reduce_chs or in_chs
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        # NOTE adaptiveavgpool can be used here, but seems to cause issues with NVIDIA AMP performance
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se)
        x_se = self.conv_expand(x_se)
        x = torch.sigmoid(x_se) * x
        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE"""

    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_fn=F.relu, exp_ratio=1.0, noskip=False,
                 se_ratio=0., shuffle_type=None, pw_group=1,
                 bn_momentum=_BN_MOMENTUM_PT_DEFAULT, bn_eps=_BN_EPS_PT_DEFAULT,
                 folded_bn=False, padding_same=False):
        super(InvertedResidual, self).__init__()
        mid_chs = int(in_chs * exp_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.act_fn = act_fn
        dw_padding = _padding_arg(kernel_size // 2, padding_same)
        pw_padding = _padding_arg(0, padding_same)

        # Point-wise expansion
        self.conv_pw = sconv2d(in_chs, mid_chs, 1, padding=pw_padding, groups=pw_group, bias=folded_bn)
        self.bn1 = None if folded_bn else nn.BatchNorm2d(mid_chs, momentum=bn_momentum, eps=bn_eps)

        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)

        # Depth-wise convolution
        self.conv_dw = sconv2d(
            mid_chs, mid_chs, kernel_size, padding=dw_padding, stride=stride, groups=mid_chs, bias=folded_bn)
        self.bn2 = None if folded_bn else nn.BatchNorm2d(mid_chs, momentum=bn_momentum, eps=bn_eps)

        # Squeeze-and-excitation
        if self.has_se:
            self.se = SqueezeExcite(mid_chs, reduce_chs=max(1, int(in_chs * se_ratio)))

        # Point-wise linear projection
        self.conv_pwl = sconv2d(mid_chs, out_chs, 1, padding=pw_padding, groups=pw_group, bias=folded_bn)
        self.bn3 = None if folded_bn else nn.BatchNorm2d(out_chs, momentum=bn_momentum, eps=bn_eps)

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act_fn(x)

        # FIXME haven't tried this yet
        # for channel shuffle when using groups with pointwise convs as per FBNet variants
        if self.shuffle_type == "mid":
            x = self.shuffle(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = self.act_fn(x)

        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        if self.bn3 is not None:
            x = self.bn3(x)

        if self.has_residual:
            x += residual

        # NOTE maskrcnn_benchmark building blocks have an SE module defined here for some variants

        return x


class GenMobileNet(nn.Module):
    """ Generic Mobile Net

    An implementation of mobile optimized networks that covers:
      * MobileNet-V1
      * MobileNet-V2
      * MNASNet A1, B1, and small
      * FBNet A, B, and C
      * ChamNet (arch details are murky)
      * Single-Path NAS Pixel1
    """

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=32, num_features=1280,
                 depth_multiplier=1.0, depth_divisor=8, min_depth=None,
                 bn_momentum=_BN_MOMENTUM_PT_DEFAULT, bn_eps=_BN_EPS_PT_DEFAULT,
                 drop_rate=0., act_fn=F.relu, global_pool='avg', skip_head_conv=False,
                 weight_init='goog', folded_bn=False, padding_same=False):
        super(GenMobileNet, self).__init__()
        self.num_classes = num_classes
        self.depth_multiplier = depth_multiplier
        self.drop_rate = drop_rate
        self.act_fn = act_fn
        self.num_features = num_features

        stem_size = _round_channels(stem_size, depth_multiplier, depth_divisor, min_depth)
        self.conv_stem = sconv2d(
            in_chans, stem_size, 3,
            padding=_padding_arg(1, padding_same), stride=2, bias=folded_bn)
        self.bn1 = None if folded_bn else nn.BatchNorm2d(stem_size, momentum=bn_momentum, eps=bn_eps)
        in_chs = stem_size

        builder = _BlockBuilder(
            depth_multiplier, depth_divisor, min_depth,
            bn_momentum, bn_eps, folded_bn, padding_same)
        self.blocks = nn.Sequential(*builder(in_chs, block_args))
        in_chs = builder.in_chs

        if skip_head_conv:
            self.conv_head = None
            assert in_chs == self.num_features
        else:
            self.conv_head = sconv2d(
                in_chs, self.num_features, 1,
                padding=_padding_arg(0, padding_same), bias=folded_bn)
            self.bn2 = None if folded_bn else nn.BatchNorm2d(self.num_features, momentum=bn_momentum, eps=bn_eps)

        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.classifier = nn.Linear(self.num_features, self.num_classes)

        for m in self.modules():
            if weight_init == 'goog':
                _initialize_weight_goog(m)
            else:
                _initialize_weight_default(m)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        del self.classifier
        if num_classes:
            self.classifier = nn.Linear(
                self.num_features * self.global_pool.feat_mult(), num_classes)
        else:
            self.classifier = None

    def forward_features(self, x, pool=True):
        x = self.conv_stem(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act_fn(x)
        x = self.blocks(x)
        if self.conv_head is not None:
            x = self.conv_head(x)
            if self.bn2 is not None:
                x = self.bn2(x)
            x = self.act_fn(x)
        if pool:
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


def _gen_mnasnet_a1(depth_multiplier, num_classes=1000, **kwargs):
    """Creates a mnasnet-a1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      depth_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r2_k3_s2_e6_c24'],
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25'],
        # stage 3, 28x28 in
        ['ir_r4_k3_s2_e6_c80'],
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320'],
    ]
    bn_momentum, bn_eps = _resolve_bn_params(kwargs)
    model = GenMobileNet(
        arch_def,
        num_classes=num_classes,
        stem_size=32,
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None,
        bn_momentum=bn_momentum,
        bn_eps=bn_eps,
        **kwargs
    )
    return model


def _gen_mnasnet_b1(depth_multiplier, num_classes=1000, **kwargs):
    """Creates a mnasnet-b1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      depth_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r3_k3_s2_e3_c24'],
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40'],
        # stage 3, 28x28 in
        ['ir_r3_k5_s2_e6_c80'],
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c96'],
        # stage 5, 14x14in
        ['ir_r4_k5_s2_e6_c192'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320_noskip']
    ]
    bn_momentum, bn_eps = _resolve_bn_params(kwargs)
    model = GenMobileNet(
        arch_def,
        num_classes=num_classes,
        stem_size=32,
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None,
        bn_momentum=bn_momentum,
        bn_eps=bn_eps,
        **kwargs
    )
    return model


def _gen_mnasnet_small(depth_multiplier, num_classes=1000, **kwargs):
    """Creates a mnasnet-b1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      depth_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        ['ds_r1_k3_s1_c8'],
        ['ir_r1_k3_s2_e3_c16'],
        ['ir_r2_k3_s2_e6_c16'],
        ['ir_r4_k5_s2_e6_c32_se0.25'],
        ['ir_r3_k3_s1_e6_c32_se0.25'],
        ['ir_r3_k5_s2_e6_c88_se0.25'],
        ['ir_r1_k3_s1_e6_c144']
    ]
    bn_momentum, bn_eps = _resolve_bn_params(kwargs)
    model = GenMobileNet(
        arch_def,
        num_classes=num_classes,
        stem_size=8,
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None,
        bn_momentum=bn_momentum,
        bn_eps=bn_eps,
        **kwargs
    )
    return model


def _gen_mobilenet_v1(depth_multiplier, num_classes=1000, **kwargs):
    """ Generate MobileNet-V1 network
    Ref impl: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
    Paper: https://arxiv.org/abs/1801.04381
    """
    arch_def = [
        ['dsa_r1_k3_s1_c64'],
        ['dsa_r2_k3_s2_c128'],
        ['dsa_r2_k3_s2_c256'],
        ['dsa_r6_k3_s2_c512'],
        ['dsa_r2_k3_s2_c1024'],
    ]
    bn_momentum, bn_eps = _resolve_bn_params(kwargs)
    model = GenMobileNet(
        arch_def,
        num_classes=num_classes,
        stem_size=32,
        num_features=1024,
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None,
        bn_momentum=bn_momentum,
        bn_eps=bn_eps,
        act_fn=F.relu6,
        skip_head_conv=True,
        **kwargs
        )
    return model


def _gen_mobilenet_v2(depth_multiplier, num_classes=1000, **kwargs):
    """ Generate MobileNet-V2 network
    Ref impl: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
    Paper: https://arxiv.org/abs/1801.04381
    """
    arch_def = [
        ['ds_r1_k3_s1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r3_k3_s2_e6_c32'],
        ['ir_r4_k3_s2_e6_c64'],
        ['ir_r3_k3_s1_e6_c96'],
        ['ir_r3_k3_s2_e6_c160'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    bn_momentum, bn_eps = _resolve_bn_params(kwargs)
    model = GenMobileNet(
        arch_def,
        num_classes=num_classes,
        stem_size=32,
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None,
        bn_momentum=bn_momentum,
        bn_eps=bn_eps,
        act_fn=F.relu6,
        **kwargs
    )
    return model


def _gen_chamnet_v1(depth_multiplier, num_classes=1000, **kwargs):
    """ Generate Chameleon Network (ChamNet)

    Paper: https://arxiv.org/abs/1812.08934
    Ref Impl: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_modeldef.py

    FIXME: this a bit of an educated guess based on trunkd def in maskrcnn_benchmark
    """
    arch_def = [
        ['ir_r1_k3_s1_e1_c24'],
        ['ir_r2_k7_s2_e4_c48'],
        ['ir_r5_k3_s2_e7_c64'],
        ['ir_r7_k5_s2_e12_c56'],
        ['ir_r5_k3_s1_e8_c88'],
        ['ir_r4_k3_s2_e7_c152'],
        ['ir_r1_k3_s1_e10_c104'],
    ]
    bn_momentum, bn_eps = _resolve_bn_params(kwargs)
    model = GenMobileNet(
        arch_def,
        num_classes=num_classes,
        stem_size=32,
        num_features=1280,  # no idea what this is? try mobile/mnasnet default?
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None,
        bn_momentum=bn_momentum,
        bn_eps=bn_eps,
        **kwargs
    )
    return model


def _gen_chamnet_v2(depth_multiplier, num_classes=1000, **kwargs):
    """ Generate Chameleon Network (ChamNet)

    Paper: https://arxiv.org/abs/1812.08934
    Ref Impl: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_modeldef.py

    FIXME: this a bit of an educated guess based on trunk def in maskrcnn_benchmark
    """
    arch_def = [
        ['ir_r1_k3_s1_e1_c24'],
        ['ir_r4_k5_s2_e8_c32'],
        ['ir_r6_k7_s2_e5_c48'],
        ['ir_r3_k5_s2_e9_c56'],
        ['ir_r6_k3_s1_e6_c56'],
        ['ir_r6_k3_s2_e2_c152'],
        ['ir_r1_k3_s1_e6_c112'],
    ]
    bn_momentum, bn_eps = _resolve_bn_params(kwargs)
    model = GenMobileNet(
        arch_def,
        num_classes=num_classes,
        stem_size=32,
        num_features=1280,  # no idea what this is? try mobile/mnasnet default?
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None,
        bn_momentum=bn_momentum,
        bn_eps=bn_eps,
        **kwargs
    )
    return model


def _gen_fbnetc(depth_multiplier, num_classes=1000, **kwargs):
    """ FBNet-C

        Paper: https://arxiv.org/abs/1812.03443
        Ref Impl: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_modeldef.py

        NOTE: the impl above does not relate to the 'C' variant here, that was derived from paper,
        it was used to confirm some building block details
    """
    arch_def = [
        ['ir_r1_k3_s1_e1_c16'],
        ['ir_r1_k3_s2_e6_c24', 'ir_r2_k3_s1_e1_c24'],
        ['ir_r1_k5_s2_e6_c32', 'ir_r1_k5_s1_e3_c32', 'ir_r1_k5_s1_e6_c32', 'ir_r1_k3_s1_e6_c32'],
        ['ir_r1_k5_s2_e6_c64', 'ir_r1_k5_s1_e3_c64', 'ir_r2_k5_s1_e6_c64'],
        ['ir_r3_k5_s1_e6_c112', 'ir_r1_k5_s1_e3_c112'],
        ['ir_r4_k5_s2_e6_c184'],
        ['ir_r1_k3_s1_e6_c352'],
    ]
    bn_momentum, bn_eps = _resolve_bn_params(kwargs)
    model = GenMobileNet(
        arch_def,
        num_classes=num_classes,
        stem_size=16,
        num_features=1984,  # paper suggests this, but is not 100% clear
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None,
        bn_momentum=bn_momentum,
        bn_eps=bn_eps,
        **kwargs
    )
    return model


def _gen_spnasnet(depth_multiplier, num_classes=1000, **kwargs):
    """Creates the Single-Path NAS model from search targeted for Pixel1 phone.

    Paper: https://arxiv.org/abs/1904.02877

    Args:
      depth_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r3_k3_s2_e3_c24'],
        # stage 2, 56x56 in
        ['ir_r1_k5_s2_e6_c40', 'ir_r3_k3_s1_e3_c40'],
        # stage 3, 28x28 in
        ['ir_r1_k5_s2_e6_c80', 'ir_r3_k3_s1_e3_c80'],
        # stage 4, 14x14in
        ['ir_r1_k5_s1_e6_c96', 'ir_r3_k5_s1_e3_c96'],
        # stage 5, 14x14in
        ['ir_r4_k5_s2_e6_c192'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320_noskip']
    ]
    bn_momentum, bn_eps = _resolve_bn_params(kwargs)
    model = GenMobileNet(
        arch_def,
        num_classes=num_classes,
        stem_size=32,
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None,
        bn_momentum=bn_momentum,
        bn_eps=bn_eps,
        **kwargs
    )
    return model


def mnasnet0_50(num_classes=1000, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet B1, depth multiplier of 0.5. """
    default_cfg = default_cfgs['mnasnet0_50']
    model = _gen_mnasnet_b1(0.5, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def mnasnet0_75(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet B1, depth multiplier of 0.75. """
    default_cfg = default_cfgs['mnasnet0_75']
    model = _gen_mnasnet_b1(0.75, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def mnasnet1_00(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet B1, depth multiplier of 1.0. """
    default_cfg = default_cfgs['mnasnet1_00']
    model = _gen_mnasnet_b1(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def tflite_mnasnet1_00(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet B1, depth multiplier of 1.0. """
    default_cfg = default_cfgs['tflite_mnasnet1_00']
    # these two args are for compat with tflite pretrained weights
    kwargs['folded_bn'] = True
    kwargs['padding_same'] = True
    model = _gen_mnasnet_b1(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def mnasnet1_40(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet B1,  depth multiplier of 1.4 """
    default_cfg = default_cfgs['mnasnet1_40']
    model = _gen_mnasnet_b1(1.4, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def semnasnet0_50(num_classes=1000, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 0.5 """
    default_cfg = default_cfgs['semnasnet0_50']
    model = _gen_mnasnet_a1(0.5, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def semnasnet0_75(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE),  depth multiplier of 0.75. """
    default_cfg = default_cfgs['semnasnet0_75']
    model = _gen_mnasnet_a1(0.75, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def semnasnet1_00(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 1.0. """
    default_cfg = default_cfgs['semnasnet1_00']
    model = _gen_mnasnet_a1(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def tflite_semnasnet1_00(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet A1, depth multiplier of 1.0. """
    default_cfg = default_cfgs['tflite_semnasnet1_00']
    # these two args are for compat with tflite pretrained weights
    kwargs['folded_bn'] = True
    kwargs['padding_same'] = True
    model = _gen_mnasnet_a1(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def semnasnet1_40(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 1.4. """
    default_cfg = default_cfgs['semnasnet1_40']
    model = _gen_mnasnet_a1(1.4, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def mnasnet_small(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MNASNet Small,  depth multiplier of 1.0. """
    default_cfg = default_cfgs['mnasnet_small']
    model = _gen_mnasnet_small(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def mobilenetv1_1_00(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MobileNet V1 """
    default_cfg = default_cfgs['mobilenetv1_1_00']
    model = _gen_mobilenet_v1(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def mobilenetv2_1_00(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ MobileNet V2 """
    default_cfg = default_cfgs['mobilenetv2_1_00']
    model = _gen_mobilenet_v2(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def fbnetc_1_00(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ FBNet-C """
    default_cfg = default_cfgs['fbnetc_1_00']
    model = _gen_fbnetc(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def chamnetv1_1_00(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ ChamNet """
    default_cfg = default_cfgs['chamnetv1_1_00']
    model = _gen_chamnet_v1(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def chamnetv2_1_00(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ ChamNet """
    default_cfg = default_cfgs['chamnetv2_1_00']
    model = _gen_chamnet_v2(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def spnasnet1_00(num_classes, in_chans=3, pretrained=False, **kwargs):
    """ Single-Path NAS Pixel1"""
    default_cfg = default_cfgs['spnasnet1_00']
    model = _gen_spnasnet(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model
