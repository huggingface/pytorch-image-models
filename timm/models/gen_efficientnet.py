""" Generic EfficientNets

A generic class with building blocks to support a variety of models with efficient architectures:
* EfficientNet (B0-B7)
* MixNet (Small, Medium, and Large)
* MnasNet B1, A1 (SE), Small
* MobileNet V1, V2, and V3
* FBNet-C (TODO A & B)
* ChamNet (TODO still guessing at architecture definition)
* Single-Path NAS Pixel1
* And likely more...

TODO not all combinations and variations have been tested. Currently working on training hyper-params...

Hacked together by Ross Wightman
"""

import math
import re
import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .helpers import load_pretrained
from .adaptive_avgmax_pool import SelectAdaptivePool2d
from .conv2d_helpers import select_conv2d
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


__all__ = ['GenEfficientNet']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'mnasnet_050': _cfg(url=''),
    'mnasnet_075': _cfg(url=''),
    'mnasnet_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_b1-74cb7081.pth'),
    'mnasnet_140': _cfg(url=''),
    'semnasnet_050': _cfg(url=''),
    'semnasnet_075': _cfg(url=''),
    'semnasnet_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_a1-d9418771.pth'),
    'semnasnet_140': _cfg(url=''),
    'mnasnet_small': _cfg(url=''),
    'mobilenetv1_100': _cfg(url=''),
    'mobilenetv2_100': _cfg(url=''),
    'mobilenetv3_050': _cfg(url=''),
    'mobilenetv3_075': _cfg(url=''),
    'mobilenetv3_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_100-35495452.pth'),
    'chamnetv1_100': _cfg(url=''),
    'chamnetv2_100': _cfg(url=''),
    'fbnetc_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetc_100-c345b898.pth',
        interpolation='bilinear'),
    'spnasnet_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/spnasnet_100-048bc3f4.pth',
        interpolation='bilinear'),
    'efficientnet_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0-d6904d92.pth'),
    'efficientnet_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'efficientnet_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2-cf78dc4d.pth',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'efficientnet_b3': _cfg(
        url='', input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'efficientnet_b4': _cfg(
        url='', input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'efficientnet_b5': _cfg(
        url='', input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'efficientnet_b6': _cfg(
        url='', input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'efficientnet_b7': _cfg(
        url='', input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'efficientnet_es': _cfg(
        url=''),
    'efficientnet_em': _cfg(
        url='',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'efficientnet_el': _cfg(
        url='',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_aa-827b6e33.pth',
        input_size=(3, 224, 224)),
    'tf_efficientnet_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_aa-ea7a6ee0.pth',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_aa-60c94f97.pth',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'tf_efficientnet_b3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_aa-84b4657e.pth',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_b4': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_aa-818f208c.pth',
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'tf_efficientnet_b5': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_aa-99018a74.pth',
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'tf_efficientnet_b6': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_aa-80ba17e4.pth',
        input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'tf_efficientnet_b7': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_aa-076e3472.pth',
        input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'tf_efficientnet_es': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_es-ca1afbfe.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 224, 224), ),
    'tf_efficientnet_em': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_em-e78cfe58.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_el': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_el-5143854e.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'mixnet_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_s-a907afbc.pth'),
    'mixnet_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_m-4647fc68.pth'),
    'mixnet_l': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_l-5a9a2ed8.pth'),
    'mixnet_xl': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_xl-ac5fbe8d.pth'),
    'mixnet_xxl': _cfg(),
    'tf_mixnet_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_s-89d3354b.pth'),
    'tf_mixnet_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_m-0f4d8805.pth'),
    'tf_mixnet_l': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_l-6c92e0c8.pth'),
}


_DEBUG = False

# Default args for PyTorch BN impl
_BN_MOMENTUM_PT_DEFAULT = 0.1
_BN_EPS_PT_DEFAULT = 1e-5
_BN_ARGS_PT = dict(momentum=_BN_MOMENTUM_PT_DEFAULT, eps=_BN_EPS_PT_DEFAULT)

# Defaults used for Google/Tensorflow training of mobile networks /w RMSprop as per
# papers and TF reference implementations. PT momentum equiv for TF decay is (1 - TF decay)
# NOTE: momentum varies btw .99 and .9997 depending on source
# .99 in official TF TPU impl
# .9997 (/w .999 in search space) for paper
_BN_MOMENTUM_TF_DEFAULT = 1 - 0.99
_BN_EPS_TF_DEFAULT = 1e-3
_BN_ARGS_TF = dict(momentum=_BN_MOMENTUM_TF_DEFAULT, eps=_BN_EPS_TF_DEFAULT)


def _resolve_bn_args(kwargs):
    bn_args = _BN_ARGS_TF.copy() if kwargs.pop('bn_tf', False) else _BN_ARGS_PT.copy()
    bn_momentum = kwargs.pop('bn_momentum', None)
    if bn_momentum is not None:
        bn_args['momentum'] = bn_momentum
    bn_eps = kwargs.pop('bn_eps', None)
    if bn_eps is not None:
        bn_args['eps'] = bn_eps
    return bn_args


def _round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels

    channels *= multiplier
    channel_min = channel_min or divisor
    new_channels = max(
        int(channels + divisor / 2) // divisor * divisor,
        channel_min)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return new_channels


def _parse_ksize(ss):
    if ss.isdigit():
        return int(ss)
    else:
        return [int(k) for k in ss.split('.')]


def _decode_block_str(block_str, depth_multiplier=1.0):
    """ Decode block definition string

    Gets a list of block arg (dicts) through a string notation of arguments.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip

    All args can exist in any order with the exception of the leading string which
    is assumed to indicate the block type.

    leading string - block type (
      ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act, cn = ConvBnAct)
    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    c - output channels,
    se - squeeze/excitation ratio
    n - activation fn ('re', 'r6', 'hs', or 'sw')
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
    noskip = False
    for op in ops:
        # string options being checked on individual basis, combine if they grow
        if op == 'noskip':
            noskip = True
        elif op.startswith('n'):
            # activation fn
            key = op[0]
            v = op[1:]
            if v == 're':
                value = F.relu
            elif v == 'r6':
                value = F.relu6
            elif v == 'hs':
                value = hard_swish
            elif v == 'sw':
                value = swish
            else:
                continue
            options[key] = value
        else:
            # all numeric options
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

    # if act_fn is None, the model default (passed to model init) will be used
    act_fn = options['n'] if 'n' in options else None
    exp_kernel_size = _parse_ksize(options['a']) if 'a' in options else 1
    pw_kernel_size = _parse_ksize(options['p']) if 'p' in options else 1
    fake_in_chs = int(options['fc']) if 'fc' in options else 0  # FIXME hack to deal with in_chs issue in TPU def

    num_repeat = int(options['r'])
    # each type of block has different valid arguments, fill accordingly
    if block_type == 'ir':
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            exp_ratio=float(options['e']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_fn=act_fn,
            noskip=noskip,
        )
    elif block_type == 'ds' or block_type == 'dsa':
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_fn=act_fn,
            pw_act=block_type == 'dsa',
            noskip=block_type == 'dsa' or noskip,
        )
    elif block_type == 'er':
        block_args = dict(
            block_type=block_type,
            exp_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            exp_ratio=float(options['e']),
            fake_in_chs=fake_in_chs,
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_fn=act_fn,
            noskip=noskip,
        )
    elif block_type == 'cn':
        block_args = dict(
            block_type=block_type,
            kernel_size=int(options['k']),
            out_chs=int(options['c']),
            stride=int(options['s']),
            act_fn=act_fn,
        )
    else:
        assert False, 'Unknown block type (%s)' % block_type

    return block_args, num_repeat


def _scale_stage_depth(stack_args, repeats, depth_multiplier=1.0, depth_trunc='ceil'):
    """ Per-stage depth scaling
    Scales the block repeats in each stage. This depth scaling impl maintains
    compatibility with the EfficientNet scaling method, while allowing sensible
    scaling for other models that may have multiple block arg definitions in each stage.
    """

    # We scale the total repeat count for each stage, there may be multiple
    # block arg defs per stage so we need to sum.
    num_repeat = sum(repeats)
    if depth_trunc == 'round':
        # Truncating to int by rounding allows stages with few repeats to remain
        # proportionally smaller for longer. This is a good choice when stage definitions
        # include single repeat stages that we'd prefer to keep that way as long as possible
        num_repeat_scaled = max(1, round(num_repeat * depth_multiplier))
    else:
        # The default for EfficientNet truncates repeats to int via 'ceil'.
        # Any multiplier > 1.0 will result in an increased depth for every stage.
        num_repeat_scaled = int(math.ceil(num_repeat * depth_multiplier))

    # Proportionally distribute repeat count scaling to each block definition in the stage.
    # Allocation is done in reverse as it results in the first block being less likely to be scaled.
    # The first block makes less sense to repeat in most of the arch definitions.
    repeats_scaled = []
    for r in repeats[::-1]:
        rs = max(1, round((r / num_repeat * num_repeat_scaled)))
        repeats_scaled.append(rs)
        num_repeat -= r
        num_repeat_scaled -= rs
    repeats_scaled = repeats_scaled[::-1]

    # Apply the calculated scaling to each block arg in the stage
    sa_scaled = []
    for ba, rep in zip(stack_args, repeats_scaled):
        sa_scaled.extend([deepcopy(ba) for _ in range(rep)])
    return sa_scaled


def _decode_arch_def(arch_def, depth_multiplier=1.0, depth_trunc='ceil'):
    arch_args = []
    for stack_idx, block_strings in enumerate(arch_def):
        assert isinstance(block_strings, list)
        stack_args = []
        repeats = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            ba, rep = _decode_block_str(block_str)
            stack_args.append(ba)
            repeats.append(rep)
        arch_args.append(_scale_stage_depth(stack_args, repeats, depth_multiplier, depth_trunc))
    return arch_args


def swish(x, inplace=False):
    if inplace:
        return x.mul_(x.sigmoid())
    else:
        return x * x.sigmoid()


def sigmoid(x, inplace=False):
    return x.sigmoid_() if inplace else x.sigmoid()


def hard_swish(x, inplace=False):
    if inplace:
        return x.mul_(F.relu6(x + 3.) / 6.)
    else:
        return x * F.relu6(x + 3.) / 6.


def hard_sigmoid(x, inplace=False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class _BlockBuilder:
    """ Build Trunk Blocks

    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py

    """
    def __init__(self, channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 pad_type='', act_fn=None, se_gate_fn=sigmoid, se_reduce_mid=False,
                 bn_args=_BN_ARGS_PT, drop_connect_rate=0., verbose=False):
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.pad_type = pad_type
        self.act_fn = act_fn
        self.se_gate_fn = se_gate_fn
        self.se_reduce_mid = se_reduce_mid
        self.bn_args = bn_args
        self.drop_connect_rate = drop_connect_rate
        self.verbose = verbose

        # updated during build
        self.in_chs = None
        self.block_idx = 0
        self.block_count = 0

    def _round_channels(self, chs):
        return _round_channels(chs, self.channel_multiplier, self.channel_divisor, self.channel_min)

    def _make_block(self, ba):
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self._round_channels(ba['out_chs'])
        if 'fake_in_chs' in ba and ba['fake_in_chs']:
            # FIXME this is a hack to work around mismatch in origin impl input filters
            ba['fake_in_chs'] = self._round_channels(ba['fake_in_chs'])
        ba['bn_args'] = self.bn_args
        ba['pad_type'] = self.pad_type
        # block act fn overrides the model default
        ba['act_fn'] = ba['act_fn'] if ba['act_fn'] is not None else self.act_fn
        assert ba['act_fn'] is not None
        if bt == 'ir':
            ba['drop_connect_rate'] = self.drop_connect_rate * self.block_idx / self.block_count
            ba['se_gate_fn'] = self.se_gate_fn
            ba['se_reduce_mid'] = self.se_reduce_mid
            if self.verbose:
                logging.info('  InvertedResidual {}, Args: {}'.format(self.block_idx, str(ba)))
            block = InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            ba['drop_connect_rate'] = self.drop_connect_rate * self.block_idx / self.block_count
            if self.verbose:
                logging.info('  DepthwiseSeparable {}, Args: {}'.format(self.block_idx, str(ba)))
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'er':
            ba['drop_connect_rate'] = self.drop_connect_rate * self.block_idx / self.block_count
            ba['se_gate_fn'] = self.se_gate_fn
            ba['se_reduce_mid'] = self.se_reduce_mid
            if self.verbose:
                logging.info('  EdgeResidual {}, Args: {}'.format(self.block_idx, str(ba)))
            block = EdgeResidual(**ba)
        elif bt == 'cn':
            if self.verbose:
                logging.info('  ConvBnAct {}, Args: {}'.format(self.block_idx, str(ba)))
            block = ConvBnAct(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt
        self.in_chs = ba['out_chs']  # update in_chs for arg of next block

        return block

    def _make_stack(self, stack_args):
        blocks = []
        # each stack (stage) contains a list of block arguments
        for i, ba in enumerate(stack_args):
            if self.verbose:
                logging.info(' Block: {}'.format(i))
            if i >= 1:
                # only the first block in any stack can have a stride > 1
                ba['stride'] = 1
            block = self._make_block(ba)
            blocks.append(block)
            self.block_idx += 1  # incr global idx (across all stacks)
        return nn.Sequential(*blocks)

    def __call__(self, in_chs, block_args):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            block_args: A list of lists, outer list defines stages, inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        if self.verbose:
            logging.info('Building model trunk with %d stages...' % len(block_args))
        self.in_chs = in_chs
        self.block_count = sum([len(x) for x in block_args])
        self.block_idx = 0
        blocks = []
        # outer list of block_args defines the stacks ('stages' by some conventions)
        for stack_idx, stack in enumerate(block_args):
            if self.verbose:
                logging.info('Stack: {}'.format(stack_idx))
            assert isinstance(stack, list)
            stack = self._make_stack(stack)
            blocks.append(stack)
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


def drop_connect(inputs, training=False, drop_connect_rate=0.):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


class ChannelShuffle(nn.Module):
    # FIXME haven't used yet
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
    def __init__(self, in_chs, reduce_chs=None, act_fn=F.relu, gate_fn=sigmoid):
        super(SqueezeExcite, self).__init__()
        self.act_fn = act_fn
        self.gate_fn = gate_fn
        reduced_chs = reduce_chs or in_chs
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        # NOTE adaptiveavgpool can be used here, but seems to cause issues with NVIDIA AMP performance
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se, inplace=True)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, pad_type='', act_fn=F.relu, bn_args=_BN_ARGS_PT):
        super(ConvBnAct, self).__init__()
        assert stride in [1, 2]
        self.act_fn = act_fn
        self.conv = select_conv2d(in_chs, out_chs, kernel_size, stride=stride, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(out_chs, **bn_args)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)
        return x


class EdgeResidual(nn.Module):
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride"""

    def __init__(self, in_chs, out_chs, exp_kernel_size=3, exp_ratio=1.0, fake_in_chs=0,
                 stride=1, pad_type='', act_fn=F.relu, noskip=False, pw_kernel_size=1,
                 se_ratio=0., se_reduce_mid=False, se_gate_fn=sigmoid,
                 bn_args=_BN_ARGS_PT, drop_connect_rate=0.):
        super(EdgeResidual, self).__init__()
        mid_chs = int(fake_in_chs * exp_ratio) if fake_in_chs > 0 else int(in_chs * exp_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate

        # Expansion convolution
        self.conv_exp = select_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(mid_chs, **bn_args)

        # Squeeze-and-excitation
        if self.has_se:
            se_base_chs = mid_chs if se_reduce_mid else in_chs
            self.se = SqueezeExcite(
                mid_chs, reduce_chs=max(1, int(se_base_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)

        # Point-wise linear projection
        self.conv_pwl = select_conv2d(mid_chs, out_chs, pw_kernel_size, stride=stride, padding=pad_type)
        self.bn2 = nn.BatchNorm2d(out_chs, **bn_args)

    def forward(self, x):
        residual = x

        # Expansion convolution
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)

        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn2(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual

        return x


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks with an expansion
    factor of 1.0. This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, pad_type='', act_fn=F.relu, noskip=False,
                 pw_kernel_size=1, pw_act=False,
                 se_ratio=0., se_gate_fn=sigmoid,
                 bn_args=_BN_ARGS_PT, drop_connect_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate

        self.conv_dw = select_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, padding=pad_type, depthwise=True)
        self.bn1 = nn.BatchNorm2d(in_chs, **bn_args)

        # Squeeze-and-excitation
        if self.has_se:
            self.se = SqueezeExcite(
                in_chs, reduce_chs=max(1, int(in_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)

        self.conv_pw = select_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = nn.BatchNorm2d(out_chs, **bn_args)

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)

        if self.has_se:
            x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_pw_act:
            x = self.act_fn(x, inplace=True)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, pad_type='', act_fn=F.relu, noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_reduce_mid=False, se_gate_fn=sigmoid,
                 shuffle_type=None, bn_args=_BN_ARGS_PT, drop_connect_rate=0.):
        super(InvertedResidual, self).__init__()
        mid_chs = int(in_chs * exp_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate

        # Point-wise expansion
        self.conv_pw = select_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(mid_chs, **bn_args)

        self.shuffle_type = shuffle_type
        if shuffle_type is not None and isinstance(exp_kernel_size, list):
            self.shuffle = ChannelShuffle(len(exp_kernel_size))

        # Depth-wise convolution
        self.conv_dw = select_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=pad_type, depthwise=True)
        self.bn2 = nn.BatchNorm2d(mid_chs, **bn_args)

        # Squeeze-and-excitation
        if self.has_se:
            se_base_chs = mid_chs if se_reduce_mid else in_chs
            self.se = SqueezeExcite(
                mid_chs, reduce_chs=max(1, int(se_base_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)

        # Point-wise linear projection
        self.conv_pwl = select_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn3 = nn.BatchNorm2d(out_chs, **bn_args)

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)

        # FIXME haven't tried this yet
        # for channel shuffle when using groups with pointwise convs as per FBNet variants
        if self.shuffle_type == "mid":
            x = self.shuffle(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act_fn(x, inplace=True)

        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual

        # NOTE maskrcnn_benchmark building blocks have an SE module defined here for some variants

        return x


class GenEfficientNet(nn.Module):
    """ Generic EfficientNet

    An implementation of efficient network architectures, in many cases mobile optimized networks:
      * MobileNet-V1
      * MobileNet-V2
      * MobileNet-V3
      * MnasNet A1, B1, and small
      * FBNet A, B, and C
      * ChamNet (arch details are murky)
      * Single-Path NAS Pixel1
      * EfficientNet B0-B7
      * MixNet S, M, L
    """

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=32, num_features=1280,
                 channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 pad_type='', act_fn=F.relu, drop_rate=0., drop_connect_rate=0.,
                 se_gate_fn=sigmoid, se_reduce_mid=False, bn_args=_BN_ARGS_PT,
                 global_pool='avg', head_conv='default', weight_init='goog'):
        super(GenEfficientNet, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.act_fn = act_fn
        self.num_features = num_features

        stem_size = _round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(stem_size, **bn_args)
        in_chs = stem_size

        builder = _BlockBuilder(
            channel_multiplier, channel_divisor, channel_min,
            pad_type, act_fn, se_gate_fn, se_reduce_mid,
            bn_args, drop_connect_rate, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(in_chs, block_args))
        in_chs = builder.in_chs

        if not head_conv or head_conv == 'none':
            self.efficient_head = False
            self.conv_head = None
            assert in_chs == self.num_features
        else:
            self.efficient_head = head_conv == 'efficient'
            self.conv_head = select_conv2d(in_chs, self.num_features, 1, padding=pad_type)
            self.bn2 = None if self.efficient_head else nn.BatchNorm2d(self.num_features, **bn_args)

        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), self.num_classes)

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
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)
        x = self.blocks(x)
        if self.efficient_head:
            # efficient head, currently only mobilenet-v3 performs pool before last 1x1 conv
            x = self.global_pool(x)  # always need to pool here regardless of flag
            x = self.conv_head(x)
            # no BN
            x = self.act_fn(x, inplace=True)
            if pool:
                # expect flattened output if pool is true, otherwise keep dim
                x = x.view(x.size(0), -1)
        else:
            if self.conv_head is not None:
                x = self.conv_head(x)
                x = self.bn2(x)
            x = self.act_fn(x, inplace=True)
            if pool:
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


def _gen_mnasnet_a1(channel_multiplier, num_classes=1000, **kwargs):
    """Creates a mnasnet-a1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
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
    model = GenEfficientNet(
        _decode_arch_def(arch_def),
        num_classes=num_classes,
        stem_size=32,
        channel_multiplier=channel_multiplier,
        bn_args=_resolve_bn_args(kwargs),
        **kwargs
    )
    return model


def _gen_mnasnet_b1(channel_multiplier, num_classes=1000, **kwargs):
    """Creates a mnasnet-b1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
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
    model = GenEfficientNet(
        _decode_arch_def(arch_def),
        num_classes=num_classes,
        stem_size=32,
        channel_multiplier=channel_multiplier,
        bn_args=_resolve_bn_args(kwargs),
        **kwargs
    )
    return model


def _gen_mnasnet_small(channel_multiplier, num_classes=1000, **kwargs):
    """Creates a mnasnet-b1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
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
    model = GenEfficientNet(
        _decode_arch_def(arch_def),
        num_classes=num_classes,
        stem_size=8,
        channel_multiplier=channel_multiplier,
        bn_args=_resolve_bn_args(kwargs),
        **kwargs
    )
    return model


def _gen_mobilenet_v1(channel_multiplier, num_classes=1000, **kwargs):
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
    model = GenEfficientNet(
        _decode_arch_def(arch_def),
        num_classes=num_classes,
        stem_size=32,
        num_features=1024,
        channel_multiplier=channel_multiplier,
        bn_args=_resolve_bn_args(kwargs),
        act_fn=F.relu6,
        head_conv='none',
        **kwargs
        )
    return model


def _gen_mobilenet_v2(channel_multiplier, num_classes=1000, **kwargs):
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
    model = GenEfficientNet(
        _decode_arch_def(arch_def),
        num_classes=num_classes,
        stem_size=32,
        channel_multiplier=channel_multiplier,
        bn_args=_resolve_bn_args(kwargs),
        act_fn=F.relu6,
        **kwargs
    )
    return model


def _gen_mobilenet_v3(channel_multiplier, num_classes=1000, **kwargs):
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_nre_noskip'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c960'],  # hard-swish
    ]
    model = GenEfficientNet(
        _decode_arch_def(arch_def),
        num_classes=num_classes,
        stem_size=16,
        channel_multiplier=channel_multiplier,
        bn_args=_resolve_bn_args(kwargs),
        act_fn=hard_swish,
        se_gate_fn=hard_sigmoid,
        se_reduce_mid=True,
        head_conv='efficient',
        **kwargs
    )
    return model


def _gen_chamnet_v1(channel_multiplier, num_classes=1000, **kwargs):
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
    model = GenEfficientNet(
        _decode_arch_def(arch_def),
        num_classes=num_classes,
        stem_size=32,
        num_features=1280,  # no idea what this is? try mobile/mnasnet default?
        channel_multiplier=channel_multiplier,
        bn_args=_resolve_bn_args(kwargs),
        **kwargs
    )
    return model


def _gen_chamnet_v2(channel_multiplier, num_classes=1000, **kwargs):
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
    model = GenEfficientNet(
        _decode_arch_def(arch_def),
        num_classes=num_classes,
        stem_size=32,
        num_features=1280,  # no idea what this is? try mobile/mnasnet default?
        channel_multiplier=channel_multiplier,
        bn_args=_resolve_bn_args(kwargs),
        **kwargs
    )
    return model


def _gen_fbnetc(channel_multiplier, num_classes=1000, **kwargs):
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
    model = GenEfficientNet(
        _decode_arch_def(arch_def),
        num_classes=num_classes,
        stem_size=16,
        num_features=1984,  # paper suggests this, but is not 100% clear
        channel_multiplier=channel_multiplier,
        bn_args=_resolve_bn_args(kwargs),
        **kwargs
    )
    return model


def _gen_spnasnet(channel_multiplier, num_classes=1000, **kwargs):
    """Creates the Single-Path NAS model from search targeted for Pixel1 phone.

    Paper: https://arxiv.org/abs/1904.02877

    Args:
      channel_multiplier: multiplier to number of channels per layer.
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
    model = GenEfficientNet(
        _decode_arch_def(arch_def),
        num_classes=num_classes,
        stem_size=32,
        channel_multiplier=channel_multiplier,
        bn_args=_resolve_bn_args(kwargs),
        **kwargs
    )
    return model


def _gen_efficientnet(channel_multiplier=1.0, depth_multiplier=1.0, num_classes=1000, **kwargs):
    """Creates an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    num_features = _round_channels(1280, channel_multiplier, 8, None)
    model = GenEfficientNet(
        _decode_arch_def(arch_def, depth_multiplier),
        num_classes=num_classes,
        stem_size=32,
        channel_multiplier=channel_multiplier,
        num_features=num_features,
        bn_args=_resolve_bn_args(kwargs),
        act_fn=swish,
        **kwargs
    )
    return model


def _gen_efficientnet_edge(channel_multiplier=1.0, depth_multiplier=1.0, num_classes=1000, **kwargs):
    arch_def = [
        # NOTE `fc` is present to override a mismatch between stem channels and in chs not
        # present in other models
        ['er_r1_k3_s1_e4_c24_fc24_noskip'],
        ['er_r2_k3_s2_e8_c32'],
        ['er_r4_k3_s2_e8_c48'],
        ['ir_r5_k5_s2_e8_c96'],
        ['ir_r4_k5_s1_e8_c144'],
        ['ir_r2_k5_s2_e8_c192'],
    ]
    num_features = _round_channels(1280, channel_multiplier, 8, None)
    model = GenEfficientNet(
        _decode_arch_def(arch_def, depth_multiplier),
        num_classes=num_classes,
        stem_size=32,
        channel_multiplier=channel_multiplier,
        num_features=num_features,
        bn_args=_resolve_bn_args(kwargs),
        act_fn=F.relu,
        **kwargs
    )
    return model


def _gen_mixnet_s(channel_multiplier=1.0, num_classes=1000, **kwargs):
    """Creates a MixNet Small model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
    Paper: https://arxiv.org/abs/1907.09595
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_a1.1_p1.1_s2_e6_c24', 'ir_r1_k3_a1.1_p1.1_s1_e3_c24'],  # relu
        # stage 2, 56x56 in
        ['ir_r1_k3.5.7_s2_e6_c40_se0.5_nsw', 'ir_r3_k3.5_a1.1_p1.1_s1_e6_c40_se0.5_nsw'],  # swish
        # stage 3, 28x28 in
        ['ir_r1_k3.5.7_p1.1_s2_e6_c80_se0.25_nsw', 'ir_r2_k3.5_p1.1_s1_e6_c80_se0.25_nsw'],  # swish
        # stage 4, 14x14in
        ['ir_r1_k3.5.7_a1.1_p1.1_s1_e6_c120_se0.5_nsw', 'ir_r2_k3.5.7.9_a1.1_p1.1_s1_e3_c120_se0.5_nsw'],  # swish
        # stage 5, 14x14in
        ['ir_r1_k3.5.7.9.11_s2_e6_c200_se0.5_nsw', 'ir_r2_k3.5.7.9_p1.1_s1_e6_c200_se0.5_nsw'],  # swish
        # 7x7
    ]
    model = GenEfficientNet(
        _decode_arch_def(arch_def),
        num_classes=num_classes,
        stem_size=16,
        num_features=1536,
        channel_multiplier=channel_multiplier,
        bn_args=_resolve_bn_args(kwargs),
        act_fn=F.relu,
        **kwargs
    )
    return model


def _gen_mixnet_m(channel_multiplier=1.0, depth_multiplier=1.0, num_classes=1000, **kwargs):
    """Creates a MixNet Medium-Large model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
    Paper: https://arxiv.org/abs/1907.09595
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c24'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3.5.7_a1.1_p1.1_s2_e6_c32', 'ir_r1_k3_a1.1_p1.1_s1_e3_c32'],  # relu
        # stage 2, 56x56 in
        ['ir_r1_k3.5.7.9_s2_e6_c40_se0.5_nsw', 'ir_r3_k3.5_a1.1_p1.1_s1_e6_c40_se0.5_nsw'],  # swish
        # stage 3, 28x28 in
        ['ir_r1_k3.5.7_s2_e6_c80_se0.25_nsw', 'ir_r3_k3.5.7.9_a1.1_p1.1_s1_e6_c80_se0.25_nsw'],  # swish
        # stage 4, 14x14in
        ['ir_r1_k3_s1_e6_c120_se0.5_nsw', 'ir_r3_k3.5.7.9_a1.1_p1.1_s1_e3_c120_se0.5_nsw'],  # swish
        # stage 5, 14x14in
        ['ir_r1_k3.5.7.9_s2_e6_c200_se0.5_nsw', 'ir_r3_k3.5.7.9_p1.1_s1_e6_c200_se0.5_nsw'],  # swish
        # 7x7
    ]
    model = GenEfficientNet(
        _decode_arch_def(arch_def, depth_multiplier=depth_multiplier, depth_trunc='round'),
        num_classes=num_classes,
        stem_size=24,
        num_features=1536,
        channel_multiplier=channel_multiplier,
        bn_args=_resolve_bn_args(kwargs),
        act_fn=F.relu,
        **kwargs
    )
    return model


@register_model
def mnasnet_050(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MNASNet B1, depth multiplier of 0.5. """
    default_cfg = default_cfgs['mnasnet_050']
    model = _gen_mnasnet_b1(0.5, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def mnasnet_075(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MNASNet B1, depth multiplier of 0.75. """
    default_cfg = default_cfgs['mnasnet_075']
    model = _gen_mnasnet_b1(0.75, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def mnasnet_100(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MNASNet B1, depth multiplier of 1.0. """
    default_cfg = default_cfgs['mnasnet_100']
    model = _gen_mnasnet_b1(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def mnasnet_b1(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MNASNet B1, depth multiplier of 1.0. """
    return mnasnet_100(pretrained, num_classes, in_chans, **kwargs)


@register_model
def mnasnet_140(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MNASNet B1,  depth multiplier of 1.4 """
    default_cfg = default_cfgs['mnasnet_140']
    model = _gen_mnasnet_b1(1.4, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def semnasnet_050(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 0.5 """
    default_cfg = default_cfgs['semnasnet_050']
    model = _gen_mnasnet_a1(0.5, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def semnasnet_075(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MNASNet A1 (w/ SE),  depth multiplier of 0.75. """
    default_cfg = default_cfgs['semnasnet_075']
    model = _gen_mnasnet_a1(0.75, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def semnasnet_100(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 1.0. """
    default_cfg = default_cfgs['semnasnet_100']
    model = _gen_mnasnet_a1(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def mnasnet_a1(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 1.0. """
    return semnasnet_100(pretrained, num_classes, in_chans, **kwargs)


@register_model
def semnasnet_140(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 1.4. """
    default_cfg = default_cfgs['semnasnet_140']
    model = _gen_mnasnet_a1(1.4, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def mnasnet_small(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MNASNet Small,  depth multiplier of 1.0. """
    default_cfg = default_cfgs['mnasnet_small']
    model = _gen_mnasnet_small(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def mobilenetv1_100(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MobileNet V1 """
    default_cfg = default_cfgs['mobilenetv1_100']
    model = _gen_mobilenet_v1(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def mobilenetv2_100(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MobileNet V2 """
    default_cfg = default_cfgs['mobilenetv2_100']
    model = _gen_mobilenet_v2(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def mobilenetv3_050(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MobileNet V3 """
    default_cfg = default_cfgs['mobilenetv3_050']
    model = _gen_mobilenet_v3(0.5, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def mobilenetv3_075(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MobileNet V3 """
    default_cfg = default_cfgs['mobilenetv3_075']
    model = _gen_mobilenet_v3(0.75, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def mobilenetv3_100(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ MobileNet V3 """
    default_cfg = default_cfgs['mobilenetv3_100']
    if pretrained:
        # pretrained model trained with non-default BN epsilon
        kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    model = _gen_mobilenet_v3(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def fbnetc_100(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ FBNet-C """
    default_cfg = default_cfgs['fbnetc_100']
    if pretrained:
        # pretrained model trained with non-default BN epsilon
        kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    model = _gen_fbnetc(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def chamnetv1_100(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ ChamNet """
    default_cfg = default_cfgs['chamnetv1_100']
    model = _gen_chamnet_v1(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def chamnetv2_100(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ ChamNet """
    default_cfg = default_cfgs['chamnetv2_100']
    model = _gen_chamnet_v2(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def spnasnet_100(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ Single-Path NAS Pixel1"""
    default_cfg = default_cfgs['spnasnet_100']
    model = _gen_spnasnet(1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def efficientnet_b0(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B0 """
    default_cfg = default_cfgs['efficientnet_b0']
    # NOTE for train, drop_rate should be 0.2
    #kwargs['drop_connect_rate'] = 0.2  # set when training, TODO add as cmd arg
    model = _gen_efficientnet(
        channel_multiplier=1.0, depth_multiplier=1.0,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def efficientnet_b1(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B1 """
    default_cfg = default_cfgs['efficientnet_b1']
    # NOTE for train, drop_rate should be 0.2
    #kwargs['drop_connect_rate'] = 0.2  # set when training, TODO add as cmd arg
    model = _gen_efficientnet(
        channel_multiplier=1.0, depth_multiplier=1.1,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def efficientnet_b2(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B2 """
    default_cfg = default_cfgs['efficientnet_b2']
    # NOTE for train, drop_rate should be 0.3
    #kwargs['drop_connect_rate'] = 0.2  # set when training, TODO add as cmd arg
    model = _gen_efficientnet(
        channel_multiplier=1.1, depth_multiplier=1.2,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def efficientnet_b3(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B3 """
    default_cfg = default_cfgs['efficientnet_b3']
    # NOTE for train, drop_rate should be 0.3
    #kwargs['drop_connect_rate'] = 0.2  # set when training, TODO add as cmd arg
    model = _gen_efficientnet(
        channel_multiplier=1.2, depth_multiplier=1.4,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def efficientnet_b4(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B4 """
    default_cfg = default_cfgs['efficientnet_b4']
    # NOTE for train, drop_rate should be 0.4
    #kwargs['drop_connect_rate'] = 0.2  #  set when training, TODO add as cmd arg
    model = _gen_efficientnet(
        channel_multiplier=1.4, depth_multiplier=1.8,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def efficientnet_b5(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B5 """
    # NOTE for train, drop_rate should be 0.4
    #kwargs['drop_connect_rate'] = 0.2  # set when training, TODO add as cmd arg
    default_cfg = default_cfgs['efficientnet_b5']
    model = _gen_efficientnet(
        channel_multiplier=1.6, depth_multiplier=2.2,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def efficientnet_b6(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B6 """
    # NOTE for train, drop_rate should be 0.5
    #kwargs['drop_connect_rate'] = 0.2  # set when training, TODO add as cmd arg
    default_cfg = default_cfgs['efficientnet_b6']
    model = _gen_efficientnet(
        channel_multiplier=1.8, depth_multiplier=2.6,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def efficientnet_b7(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B7 """
    # NOTE for train, drop_rate should be 0.5
    #kwargs['drop_connect_rate'] = 0.2  # set when training, TODO add as cmd arg
    default_cfg = default_cfgs['efficientnet_b7']
    model = _gen_efficientnet(
        channel_multiplier=2.0, depth_multiplier=3.1,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def efficientnet_es(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-Edge Small. """
    default_cfg = default_cfgs['efficientnet_es']
    model = _gen_efficientnet_edge(
        channel_multiplier=1.0, depth_multiplier=1.0,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def efficientnet_em(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-Edge-Medium. """
    default_cfg = default_cfgs['efficientnet_em']
    model = _gen_efficientnet_edge(
        channel_multiplier=1.0, depth_multiplier=1.1,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def efficientnet_el(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-Edge-Large. """
    default_cfg = default_cfgs['efficientnet_el']
    model = _gen_efficientnet_edge(
        channel_multiplier=1.2, depth_multiplier=1.4,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tf_efficientnet_b0(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B0. Tensorflow compatible variant  """
    default_cfg = default_cfgs['tf_efficientnet_b0']
    kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        channel_multiplier=1.0, depth_multiplier=1.0,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tf_efficientnet_b1(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B1. Tensorflow compatible variant  """
    default_cfg = default_cfgs['tf_efficientnet_b1']
    kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        channel_multiplier=1.0, depth_multiplier=1.1,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tf_efficientnet_b2(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B2. Tensorflow compatible variant  """
    default_cfg = default_cfgs['tf_efficientnet_b2']
    kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        channel_multiplier=1.1, depth_multiplier=1.2,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tf_efficientnet_b3(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B3. Tensorflow compatible variant """
    default_cfg = default_cfgs['tf_efficientnet_b3']
    kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        channel_multiplier=1.2, depth_multiplier=1.4,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tf_efficientnet_b4(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B4. Tensorflow compatible variant """
    default_cfg = default_cfgs['tf_efficientnet_b4']
    kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        channel_multiplier=1.4, depth_multiplier=1.8,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tf_efficientnet_b5(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B5. Tensorflow compatible variant """
    default_cfg = default_cfgs['tf_efficientnet_b5']
    kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        channel_multiplier=1.6, depth_multiplier=2.2,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tf_efficientnet_b6(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B6. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    default_cfg = default_cfgs['tf_efficientnet_b6']
    kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        channel_multiplier=1.8, depth_multiplier=2.6,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tf_efficientnet_b7(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B7. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    default_cfg = default_cfgs['tf_efficientnet_b7']
    kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        channel_multiplier=2.0, depth_multiplier=3.1,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tf_efficientnet_es(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-Edge Small. Tensorflow compatible variant  """
    default_cfg = default_cfgs['tf_efficientnet_es']
    kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_edge(
        channel_multiplier=1.0, depth_multiplier=1.0,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tf_efficientnet_em(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-Edge-Medium. Tensorflow compatible variant  """
    default_cfg = default_cfgs['tf_efficientnet_em']
    kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_edge(
        channel_multiplier=1.0, depth_multiplier=1.1,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tf_efficientnet_el(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-Edge-Large. Tensorflow compatible variant  """
    default_cfg = default_cfgs['tf_efficientnet_el']
    kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_edge(
        channel_multiplier=1.2, depth_multiplier=1.4,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def mixnet_s(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Creates a MixNet Small model.
    """
    default_cfg = default_cfgs['mixnet_s']
    model = _gen_mixnet_s(
        channel_multiplier=1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def mixnet_m(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Creates a MixNet Medium model.
    """
    default_cfg = default_cfgs['mixnet_m']
    model = _gen_mixnet_m(
        channel_multiplier=1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def mixnet_l(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Creates a MixNet Large model.
    """
    default_cfg = default_cfgs['mixnet_l']
    model = _gen_mixnet_m(
        channel_multiplier=1.3, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def mixnet_xl(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Creates a MixNet Extra-Large model.
    Not a paper spec, experimental def by RW w/ depth scaling.
    """
    default_cfg = default_cfgs['mixnet_xl']
    #kwargs['drop_connect_rate'] = 0.2
    model = _gen_mixnet_m(
        channel_multiplier=1.6, depth_multiplier=1.2, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def mixnet_xxl(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Creates a MixNet Double Extra Large model.
    Not a paper spec, experimental def by RW w/ depth scaling.
    """
    default_cfg = default_cfgs['mixnet_xxl']
    # kwargs['drop_connect_rate'] = 0.2
    model = _gen_mixnet_m(
        channel_multiplier=2.4, depth_multiplier=1.3, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tf_mixnet_s(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Creates a MixNet Small model. Tensorflow compatible variant
    """
    default_cfg = default_cfgs['tf_mixnet_s']
    kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mixnet_s(
        channel_multiplier=1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tf_mixnet_m(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Creates a MixNet Medium model. Tensorflow compatible variant
    """
    default_cfg = default_cfgs['tf_mixnet_m']
    kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mixnet_m(
        channel_multiplier=1.0, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tf_mixnet_l(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Creates a MixNet Large model. Tensorflow compatible variant
    """
    default_cfg = default_cfgs['tf_mixnet_l']
    kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mixnet_m(
        channel_multiplier=1.3, num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def gen_efficientnet_model_names():
    return set(_models)
