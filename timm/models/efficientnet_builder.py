import logging
import math
import re
from collections import OrderedDict
from copy import deepcopy

import torch.nn as nn

from .efficientnet_blocks import *
from .layers import CondConv2d, get_condconv_initializer

__all__ = ["EfficientNetBuilder", "decode_arch_def", "efficientnet_init_weights"]


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
                value = get_act_layer('relu')
            elif v == 'r6':
                value = get_act_layer('relu6')
            elif v == 'hs':
                value = get_act_layer('hard_swish')
            elif v == 'sw':
                value = get_act_layer('swish')
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
            act_layer=act_layer,
            noskip=noskip,
        )
        if 'cc' in options:
            block_args['num_experts'] = int(options['cc'])
    elif block_type == 'ds' or block_type == 'dsa':
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_layer=act_layer,
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
            act_layer=act_layer,
            noskip=noskip,
        )
    elif block_type == 'cn':
        block_args = dict(
            block_type=block_type,
            kernel_size=int(options['k']),
            out_chs=int(options['c']),
            stride=int(options['s']),
            act_layer=act_layer,
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


def decode_arch_def(arch_def, depth_multiplier=1.0, depth_trunc='ceil', experts_multiplier=1, fix_first_last=False):
    arch_args = []
    for stack_idx, block_strings in enumerate(arch_def):
        assert isinstance(block_strings, list)
        stack_args = []
        repeats = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            ba, rep = _decode_block_str(block_str)
            if ba.get('num_experts', 0) > 0 and experts_multiplier > 1:
                ba['num_experts'] *= experts_multiplier
            stack_args.append(ba)
            repeats.append(rep)
        if fix_first_last and (stack_idx == 0 or stack_idx == len(arch_def) - 1):
            arch_args.append(_scale_stage_depth(stack_args, repeats, 1.0, depth_trunc))
        else:
            arch_args.append(_scale_stage_depth(stack_args, repeats, depth_multiplier, depth_trunc))
    return arch_args


class EfficientNetBuilder:
    """ Build Trunk Blocks

    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py

    """
    def __init__(self, channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 output_stride=32, pad_type='', act_layer=None, se_kwargs=None,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0., feature_location='',
                 verbose=False):
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.output_stride = output_stride
        self.pad_type = pad_type
        self.act_layer = act_layer
        self.se_kwargs = se_kwargs
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs
        self.drop_path_rate = drop_path_rate
        self.feature_location = feature_location
        assert feature_location in ('bottleneck', 'depthwise', 'expansion', '')
        self.verbose = verbose

        # state updated during build, consumed by model
        self.in_chs = None
        self.features = OrderedDict()

    def _round_channels(self, chs):
        return round_channels(chs, self.channel_multiplier, self.channel_divisor, self.channel_min)

    def _make_block(self, ba, block_idx, block_count):
        drop_path_rate = self.drop_path_rate * block_idx / block_count
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self._round_channels(ba['out_chs'])
        if 'fake_in_chs' in ba and ba['fake_in_chs']:
            # FIXME this is a hack to work around mismatch in origin impl input filters
            ba['fake_in_chs'] = self._round_channels(ba['fake_in_chs'])
        ba['norm_layer'] = self.norm_layer
        ba['norm_kwargs'] = self.norm_kwargs
        ba['pad_type'] = self.pad_type
        # block act fn overrides the model default
        ba['act_layer'] = ba['act_layer'] if ba['act_layer'] is not None else self.act_layer
        assert ba['act_layer'] is not None
        if bt == 'ir':
            ba['drop_path_rate'] = drop_path_rate
            ba['se_kwargs'] = self.se_kwargs
            if self.verbose:
                logging.info('  InvertedResidual {}, Args: {}'.format(block_idx, str(ba)))
            if ba.get('num_experts', 0) > 0:
                block = CondConvResidual(**ba)
            else:
                block = InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            ba['drop_path_rate'] = drop_path_rate
            ba['se_kwargs'] = self.se_kwargs
            if self.verbose:
                logging.info('  DepthwiseSeparable {}, Args: {}'.format(block_idx, str(ba)))
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'er':
            ba['drop_path_rate'] = drop_path_rate
            ba['se_kwargs'] = self.se_kwargs
            if self.verbose:
                logging.info('  EdgeResidual {}, Args: {}'.format(block_idx, str(ba)))
            block = EdgeResidual(**ba)
        elif bt == 'cn':
            if self.verbose:
                logging.info('  ConvBnAct {}, Args: {}'.format(block_idx, str(ba)))
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
        if self.verbose:
            logging.info('Building model trunk with %d stages...' % len(model_block_args))
        self.in_chs = in_chs
        total_block_count = sum([len(x) for x in model_block_args])
        total_block_idx = 0
        current_stride = 2
        current_dilation = 1
        feature_idx = 0
        stages = []
        # outer list of block_args defines the stacks ('stages' by some conventions)
        for stage_idx, stage_block_args in enumerate(model_block_args):
            last_stack = stage_idx == (len(model_block_args) - 1)
            if self.verbose:
                logging.info('Stack: {}'.format(stage_idx))
            assert isinstance(stage_block_args, list)

            blocks = []
            # each stack (stage) contains a list of block arguments
            for block_idx, block_args in enumerate(stage_block_args):
                last_block = block_idx == (len(stage_block_args) - 1)
                extract_features = ''  # No features extracted
                if self.verbose:
                    logging.info(' Block: {}'.format(block_idx))

                # Sort out stride, dilation, and feature extraction details
                assert block_args['stride'] in (1, 2)
                if block_idx >= 1:
                    # only the first block in any stack can have a stride > 1
                    block_args['stride'] = 1

                do_extract = False
                if self.feature_location == 'bottleneck' or self.feature_location == 'depthwise':
                    if last_block:
                        next_stage_idx = stage_idx + 1
                        if next_stage_idx >= len(model_block_args):
                            do_extract = True
                        else:
                            do_extract = model_block_args[next_stage_idx][0]['stride'] > 1
                elif self.feature_location == 'expansion':
                    if block_args['stride'] > 1 or (last_stack and last_block):
                        do_extract = True
                if do_extract:
                    extract_features = self.feature_location

                next_dilation = current_dilation
                next_output_stride = current_stride
                if block_args['stride'] > 1:
                    next_output_stride = current_stride * block_args['stride']
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args['stride']
                        block_args['stride'] = 1
                        if self.verbose:
                            logging.info('  Converting stride to dilation to maintain output_stride=={}'.format(
                                self.output_stride))
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
                    feature_info = block.feature_info(extract_features)
                    if feature_info['module']:
                        feature_info['module'] = 'blocks.{}.{}.'.format(stage_idx, block_idx) + feature_info['module']
                    feature_info['stage_idx'] = stage_idx
                    feature_info['block_idx'] = block_idx
                    feature_info['reduction'] = current_stride
                    self.features[feature_idx] = feature_info
                    feature_idx += 1

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
            lambda w: w.data.normal_(0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        init_weight_fn(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def efficientnet_init_weights(model: nn.Module, init_fn=None):
    init_fn = init_fn or _init_weight_goog
    for n, m in model.named_modules():
        init_fn(m, n)

