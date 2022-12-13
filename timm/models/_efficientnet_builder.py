""" EfficientNet, MobileNetV3, etc Builder

Assembles EfficieNet and related network feature blocks from string definitions.
Handles stride, dilation calculations, and selects feature extraction points.

Hacked together by / Copyright 2019, Ross Wightman
"""

import logging
import math
import re
from copy import deepcopy
from functools import partial

import torch.nn as nn

from ._efficientnet_blocks import *
from timm.layers import CondConv2d, get_condconv_initializer, get_act_layer, get_attn, make_divisible

__all__ = ["EfficientNetBuilder", "decode_arch_def", "efficientnet_init_weights",
           'resolve_bn_args', 'resolve_act_layer', 'round_channels', 'BN_MOMENTUM_TF_DEFAULT', 'BN_EPS_TF_DEFAULT']

_logger = logging.getLogger(__name__)


_DEBUG_BUILDER = False

# Defaults used for Google/Tensorflow training of mobile networks /w RMSprop as per
# papers and TF reference implementations. PT momentum equiv for TF decay is (1 - TF decay)
# NOTE: momentum varies btw .99 and .9997 depending on source
# .99 in official TF TPU impl
# .9997 (/w .999 in search space) for paper
BN_MOMENTUM_TF_DEFAULT = 1 - 0.99
BN_EPS_TF_DEFAULT = 1e-3
_BN_ARGS_TF = dict(momentum=BN_MOMENTUM_TF_DEFAULT, eps=BN_EPS_TF_DEFAULT)


def get_bn_args_tf():
    return _BN_ARGS_TF.copy()


def resolve_bn_args(kwargs):
    bn_args = {}
    bn_momentum = kwargs.pop('bn_momentum', None)
    if bn_momentum is not None:
        bn_args['momentum'] = bn_momentum
    bn_eps = kwargs.pop('bn_eps', None)
    if bn_eps is not None:
        bn_args['eps'] = bn_eps
    return bn_args


def resolve_act_layer(kwargs, default='relu'):
    return get_act_layer(kwargs.pop('act_layer', default))


def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None, round_limit=0.9):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    return make_divisible(channels * multiplier, divisor, channel_min, round_limit=round_limit)


def _log_info_if(msg, condition):
    if condition:
        _logger.info(msg)


def _parse_ksize(ss):
    if ss.isdigit():
        return int(ss)
    else:
        return [int(k) for k in ss.split('.')]


def _decode_block_str(block_str):
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
    skip = None
    for op in ops:
        # string options being checked on individual basis, combine if they grow
        if op == 'noskip':
            skip = False  # force no skip connection
        elif op == 'skip':
            skip = True  # force a skip connection
        elif op.startswith('n'):
            # activation fn
            key = op[0]
            v = op[1:]
            if v == 're':
                value = get_act_layer('relu')
            elif v == 'r6':
                value = get_act_layer('relu6')
            elif v == 'hs':
                value = get_act_layer('hard_swish')
            elif v == 'sw':
                value = get_act_layer('swish')  # aka SiLU
            elif v == 'mi':
                value = get_act_layer('mish')
            else:
                continue
            options[key] = value
        else:
            # all numeric options
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

    # if act_layer is None, the model default (passed to model init) will be used
    act_layer = options['n'] if 'n' in options else None
    exp_kernel_size = _parse_ksize(options['a']) if 'a' in options else 1
    pw_kernel_size = _parse_ksize(options['p']) if 'p' in options else 1
    force_in_chs = int(options['fc']) if 'fc' in options else 0  # FIXME hack to deal with in_chs issue in TPU def
    num_repeat = int(options['r'])

    # each type of block has different valid arguments, fill accordingly
    block_args = dict(
        block_type=block_type,
        out_chs=int(options['c']),
        stride=int(options['s']),
        act_layer=act_layer,
    )
    if block_type == 'ir':
        block_args.update(dict(
            dw_kernel_size=_parse_ksize(options['k']),
            exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size,
            exp_ratio=float(options['e']),
            se_ratio=float(options['se']) if 'se' in options else 0.,
            noskip=skip is False,
        ))
        if 'cc' in options:
            block_args['num_experts'] = int(options['cc'])
    elif block_type == 'ds' or block_type == 'dsa':
        block_args.update(dict(
            dw_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=pw_kernel_size,
            se_ratio=float(options['se']) if 'se' in options else 0.,
            pw_act=block_type == 'dsa',
            noskip=block_type == 'dsa' or skip is False,
        ))
    elif block_type == 'er':
        block_args.update(dict(
            exp_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=pw_kernel_size,
            exp_ratio=float(options['e']),
            force_in_chs=force_in_chs,
            se_ratio=float(options['se']) if 'se' in options else 0.,
            noskip=skip is False,
        ))
    elif block_type == 'cn':
        block_args.update(dict(
            kernel_size=int(options['k']),
            skip=skip is True,
        ))
    else:
        assert False, 'Unknown block type (%s)' % block_type
    if 'gs' in options:
        block_args['group_size'] = options['gs']

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


def decode_arch_def(
        arch_def,
        depth_multiplier=1.0,
        depth_trunc='ceil',
        experts_multiplier=1,
        fix_first_last=False,
        group_size=None,
):
    """ Decode block architecture definition strings -> block kwargs

    Args:
        arch_def: architecture definition strings, list of list of strings
        depth_multiplier: network depth multiplier
        depth_trunc: networ depth truncation mode when applying multiplier
        experts_multiplier: CondConv experts multiplier
        fix_first_last: fix first and last block depths when multiplier is applied
        group_size: group size override for all blocks that weren't explicitly set in arch string

    Returns:
        list of list of block kwargs
    """
    arch_args = []
    if isinstance(depth_multiplier, tuple):
        assert len(depth_multiplier) == len(arch_def)
    else:
        depth_multiplier = (depth_multiplier,) * len(arch_def)
    for stack_idx, (block_strings, multiplier) in enumerate(zip(arch_def, depth_multiplier)):
        assert isinstance(block_strings, list)
        stack_args = []
        repeats = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            ba, rep = _decode_block_str(block_str)
            if ba.get('num_experts', 0) > 0 and experts_multiplier > 1:
                ba['num_experts'] *= experts_multiplier
            if group_size is not None:
                ba.setdefault('group_size', group_size)
            stack_args.append(ba)
            repeats.append(rep)
        if fix_first_last and (stack_idx == 0 or stack_idx == len(arch_def) - 1):
            arch_args.append(_scale_stage_depth(stack_args, repeats, 1.0, depth_trunc))
        else:
            arch_args.append(_scale_stage_depth(stack_args, repeats, multiplier, depth_trunc))
    return arch_args


class EfficientNetBuilder:
    """ Build Trunk Blocks

    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py

    """
    def __init__(self, output_stride=32, pad_type='', round_chs_fn=round_channels, se_from_exp=False,
                 act_layer=None, norm_layer=None, se_layer=None, drop_path_rate=0., feature_location=''):
        self.output_stride = output_stride
        self.pad_type = pad_type
        self.round_chs_fn = round_chs_fn
        self.se_from_exp = se_from_exp  # calculate se channel reduction from expanded (mid) chs
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.se_layer = get_attn(se_layer)
        try:
            self.se_layer(8, rd_ratio=1.0)  # test if attn layer accepts rd_ratio arg
            self.se_has_ratio = True
        except TypeError:
            self.se_has_ratio = False
        self.drop_path_rate = drop_path_rate
        if feature_location == 'depthwise':
            # old 'depthwise' mode renamed 'expansion' to match TF impl, old expansion mode didn't make sense
            _logger.warning("feature_location=='depthwise' is deprecated, using 'expansion'")
            feature_location = 'expansion'
        self.feature_location = feature_location
        assert feature_location in ('bottleneck', 'expansion', '')
        self.verbose = _DEBUG_BUILDER

        # state updated during build, consumed by model
        self.in_chs = None
        self.features = []

    def _make_block(self, ba, block_idx, block_count):
        drop_path_rate = self.drop_path_rate * block_idx / block_count
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self.round_chs_fn(ba['out_chs'])
        if 'force_in_chs' in ba and ba['force_in_chs']:
            # NOTE this is a hack to work around mismatch in TF EdgeEffNet impl
            ba['force_in_chs'] = self.round_chs_fn(ba['force_in_chs'])
        ba['pad_type'] = self.pad_type
        # block act fn overrides the model default
        ba['act_layer'] = ba['act_layer'] if ba['act_layer'] is not None else self.act_layer
        assert ba['act_layer'] is not None
        ba['norm_layer'] = self.norm_layer
        ba['drop_path_rate'] = drop_path_rate
        if bt != 'cn':
            se_ratio = ba.pop('se_ratio')
            if se_ratio and self.se_layer is not None:
                if not self.se_from_exp:
                    # adjust se_ratio by expansion ratio if calculating se channels from block input
                    se_ratio /= ba.get('exp_ratio', 1.0)
                if self.se_has_ratio:
                    ba['se_layer'] = partial(self.se_layer, rd_ratio=se_ratio)
                else:
                    ba['se_layer'] = self.se_layer

        if bt == 'ir':
            _log_info_if('  InvertedResidual {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = CondConvResidual(**ba) if ba.get('num_experts', 0) else InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            _log_info_if('  DepthwiseSeparable {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'er':
            _log_info_if('  EdgeResidual {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = EdgeResidual(**ba)
        elif bt == 'cn':
            _log_info_if('  ConvBnAct {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = ConvBnAct(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt

        self.in_chs = ba['out_chs']  # update in_chs for arg of next block
        return block

    def __call__(self, in_chs, model_block_args):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            model_block_args: A list of lists, outer list defines stages, inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        _log_info_if('Building model trunk with %d stages...' % len(model_block_args), self.verbose)
        self.in_chs = in_chs
        total_block_count = sum([len(x) for x in model_block_args])
        total_block_idx = 0
        current_stride = 2
        current_dilation = 1
        stages = []
        if model_block_args[0][0]['stride'] > 1:
            # if the first block starts with a stride, we need to extract first level feat from stem
            feature_info = dict(
                module='act1', num_chs=in_chs, stage=0, reduction=current_stride,
                hook_type='forward' if self.feature_location != 'bottleneck' else '')
            self.features.append(feature_info)

        # outer list of block_args defines the stacks
        for stack_idx, stack_args in enumerate(model_block_args):
            last_stack = stack_idx + 1 == len(model_block_args)
            _log_info_if('Stack: {}'.format(stack_idx), self.verbose)
            assert isinstance(stack_args, list)

            blocks = []
            # each stack (stage of blocks) contains a list of block arguments
            for block_idx, block_args in enumerate(stack_args):
                last_block = block_idx + 1 == len(stack_args)
                _log_info_if(' Block: {}'.format(block_idx), self.verbose)

                assert block_args['stride'] in (1, 2)
                if block_idx >= 1:   # only the first block in any stack can have a stride > 1
                    block_args['stride'] = 1

                extract_features = False
                if last_block:
                    next_stack_idx = stack_idx + 1
                    extract_features = next_stack_idx >= len(model_block_args) or \
                        model_block_args[next_stack_idx][0]['stride'] > 1

                next_dilation = current_dilation
                if block_args['stride'] > 1:
                    next_output_stride = current_stride * block_args['stride']
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args['stride']
                        block_args['stride'] = 1
                        _log_info_if('  Converting stride to dilation to maintain output_stride=={}'.format(
                            self.output_stride), self.verbose)
                    else:
                        current_stride = next_output_stride
                block_args['dilation'] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation

                # create the block
                block = self._make_block(block_args, total_block_idx, total_block_count)
                blocks.append(block)

                # stash feature module name and channel info for model feature extraction
                if extract_features:
                    feature_info = dict(
                        stage=stack_idx + 1, reduction=current_stride, **block.feature_info(self.feature_location))
                    module_name = f'blocks.{stack_idx}.{block_idx}'
                    leaf_name = feature_info.get('module', '')
                    feature_info['module'] = '.'.join([module_name, leaf_name]) if leaf_name else module_name
                    self.features.append(feature_info)

                total_block_idx += 1  # incr global block idx (across all stacks)
            stages.append(nn.Sequential(*blocks))
        return stages


def _init_weight_goog(m, n='', fix_group_fanout=True):
    """ Weight initialization as per Tensorflow official implementations.

    Args:
        m (nn.Module): module to init
        n (str): module name
        fix_group_fanout (bool): enable correct (matching Tensorflow TPU impl) fanout calculation w/ group convs

    Handles layers in EfficientNet, EfficientNet-CondConv, MixNet, MnasNet, MobileNetV3, etc:
    * https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    """
    if isinstance(m, CondConv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        init_weight_fn = get_condconv_initializer(
            lambda w: nn.init.normal_(w, 0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        init_weight_fn(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        nn.init.uniform_(m.weight, -init_range, init_range)
        nn.init.zeros_(m.bias)


def efficientnet_init_weights(model: nn.Module, init_fn=None):
    init_fn = init_fn or _init_weight_goog
    for n, m in model.named_modules():
        init_fn(m, n)

