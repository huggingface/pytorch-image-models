"""
 pnasnet5large implementation grabbed from Cadene's pretrained models
 Additional credit to https://github.com/creafz

 https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/pnasnet.py

 Papers:
    * `Progressive Neural Architecture Search` - https://arxiv.org/abs/1712.00559
"""
from collections import OrderedDict
from functools import partial
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import ConvNormAct, create_conv2d, create_pool2d, create_classifier, get_device_dtype
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._registry import register_model, generate_default_cfgs

__all__ = ['PNASNet5Large']


class SeparableConv2d(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: str = '',
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.depthwise_conv2d = create_conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            **dd,
        )
        self.pointwise_conv2d = create_conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=padding,
            **dd,
        )

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            stem_cell: bool = False,
            padding: str = '',
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        middle_channels = out_channels if stem_cell else in_channels
        self.act_1 = nn.ReLU()
        self.separable_1 = SeparableConv2d(
            in_channels,
            middle_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            **dd,
        )
        self.bn_sep_1 = nn.BatchNorm2d(middle_channels, eps=0.001, **dd)
        self.act_2 = nn.ReLU()
        self.separable_2 = SeparableConv2d(
            middle_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            **dd,
        )
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, **dd)

    def forward(self, x):
        x = self.act_1(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.act_2(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class ActConvBn(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: str = '',
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.act = nn.ReLU()
        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **dd,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, **dd)

    def forward(self, x):
        x = self.act(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class FactorizedReduction(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            padding: str = '',
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.act = nn.ReLU()
        self.path_1 = nn.Sequential(OrderedDict([
            ('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)),
            ('conv', create_conv2d(in_channels, out_channels // 2, kernel_size=1, padding=padding, **dd)),
        ]))
        self.path_2 = nn.Sequential(OrderedDict([
            ('pad', nn.ZeroPad2d((-1, 1, -1, 1))),  # shift
            ('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)),
            ('conv', create_conv2d(in_channels, out_channels // 2, kernel_size=1, padding=padding, **dd)),
        ]))
        self.final_path_bn = nn.BatchNorm2d(out_channels, eps=0.001, **dd)

    def forward(self, x):
        x = self.act(x)
        x_path1 = self.path_1(x)
        x_path2 = self.path_2(x)
        out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        return out


class CellBase(nn.Module):

    def cell_forward(self, x_left, x_right):
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_left = self.comb_iter_3_left(x_comb_iter_2)
        x_comb_iter_3_right = self.comb_iter_3_right(x_right)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_left)
        if self.comb_iter_4_right is not None:
            x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        else:
            x_comb_iter_4_right = x_right
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class CellStem0(CellBase):

    def __init__(
            self,
            in_chs_left: int,
            out_chs_left: int,
            in_chs_right: int,
            out_chs_right: int,
            pad_type: str = '',
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, kernel_size=1, padding=pad_type, **dd)

        self.comb_iter_0_left = BranchSeparables(
            in_chs_left, out_chs_left, kernel_size=5, stride=2, stem_cell=True, padding=pad_type, **dd)
        self.comb_iter_0_right = nn.Sequential(OrderedDict([
            ('max_pool', create_pool2d('max', 3, stride=2, padding=pad_type)),
            ('conv', create_conv2d(in_chs_left, out_chs_left, kernel_size=1, padding=pad_type, **dd)),
            ('bn', nn.BatchNorm2d(out_chs_left, eps=0.001, **dd)),
        ]))

        self.comb_iter_1_left = BranchSeparables(
            out_chs_right, out_chs_right, kernel_size=7, stride=2, padding=pad_type, **dd)
        self.comb_iter_1_right = create_pool2d('max', 3, stride=2, padding=pad_type)

        self.comb_iter_2_left = BranchSeparables(
            out_chs_right, out_chs_right, kernel_size=5, stride=2, padding=pad_type, **dd)
        self.comb_iter_2_right = BranchSeparables(
            out_chs_right, out_chs_right, kernel_size=3, stride=2, padding=pad_type, **dd)

        self.comb_iter_3_left = BranchSeparables(
            out_chs_right, out_chs_right, kernel_size=3, padding=pad_type, **dd)
        self.comb_iter_3_right = create_pool2d('max', 3, stride=2, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(
            in_chs_right, out_chs_right, kernel_size=3, stride=2, stem_cell=True, padding=pad_type, **dd)
        self.comb_iter_4_right = ActConvBn(
            out_chs_right, out_chs_right, kernel_size=1, stride=2, padding=pad_type, **dd)

    def forward(self, x_left):
        x_right = self.conv_1x1(x_left)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class Cell(CellBase):

    def __init__(
            self,
            in_chs_left: int,
            out_chs_left: int,
            in_chs_right: int,
            out_chs_right: int,
            pad_type: str = '',
            is_reduction: bool = False,
            match_prev_layer_dims: bool = False,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        # If `is_reduction` is set to `True` stride 2 is used for
        # convolution and pooling layers to reduce the spatial size of
        # the output of a cell approximately by a factor of 2.
        stride = 2 if is_reduction else 1

        # If `match_prev_layer_dimensions` is set to `True`
        # `FactorizedReduction` is used to reduce the spatial size
        # of the left input of a cell approximately by a factor of 2.
        self.match_prev_layer_dimensions = match_prev_layer_dims
        if match_prev_layer_dims:
            self.conv_prev_1x1 = FactorizedReduction(in_chs_left, out_chs_left, padding=pad_type, **dd)
        else:
            self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, kernel_size=1, padding=pad_type, **dd)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, kernel_size=1, padding=pad_type, **dd)

        self.comb_iter_0_left = BranchSeparables(
            out_chs_left, out_chs_left, kernel_size=5, stride=stride, padding=pad_type, **dd)
        self.comb_iter_0_right = create_pool2d('max', 3, stride=stride, padding=pad_type)

        self.comb_iter_1_left = BranchSeparables(
            out_chs_right, out_chs_right, kernel_size=7, stride=stride, padding=pad_type, **dd)
        self.comb_iter_1_right = create_pool2d('max', 3, stride=stride, padding=pad_type)

        self.comb_iter_2_left = BranchSeparables(
            out_chs_right, out_chs_right, kernel_size=5, stride=stride, padding=pad_type, **dd)
        self.comb_iter_2_right = BranchSeparables(
            out_chs_right, out_chs_right, kernel_size=3, stride=stride, padding=pad_type, **dd)

        self.comb_iter_3_left = BranchSeparables(out_chs_right, out_chs_right, kernel_size=3, **dd)
        self.comb_iter_3_right = create_pool2d('max', 3, stride=stride, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(
            out_chs_left, out_chs_left, kernel_size=3, stride=stride, padding=pad_type, **dd)
        if is_reduction:
            self.comb_iter_4_right = ActConvBn(
                out_chs_right, out_chs_right, kernel_size=1, stride=stride, padding=pad_type, **dd)
        else:
            self.comb_iter_4_right = None

    def forward(self, x_left, x_right):
        x_left = self.conv_prev_1x1(x_left)
        x_right = self.conv_1x1(x_right)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class PNASNet5Large(nn.Module):
    def __init__(
            self,
            num_classes: int = 1000,
            in_chans: int = 3,
            output_stride: int = 32,
            drop_rate: float = 0.,
            global_pool: str = 'avg',
            pad_type: str = '',
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.num_features = 4320
        assert output_stride == 32

        self.conv_0 = ConvNormAct(
            in_chans, 96, kernel_size=3, stride=2, padding=0,
            norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.1), apply_act=False, **dd)

        self.cell_stem_0 = CellStem0(
            in_chs_left=96, out_chs_left=54, in_chs_right=96, out_chs_right=54, pad_type=pad_type, **dd)

        self.cell_stem_1 = Cell(
            in_chs_left=96, out_chs_left=108, in_chs_right=270, out_chs_right=108, pad_type=pad_type,
            match_prev_layer_dims=True, is_reduction=True, **dd)
        self.cell_0 = Cell(
            in_chs_left=270, out_chs_left=216, in_chs_right=540, out_chs_right=216, pad_type=pad_type,
            match_prev_layer_dims=True, **dd)
        self.cell_1 = Cell(
            in_chs_left=540, out_chs_left=216, in_chs_right=1080, out_chs_right=216, pad_type=pad_type, **dd)
        self.cell_2 = Cell(
            in_chs_left=1080, out_chs_left=216, in_chs_right=1080, out_chs_right=216, pad_type=pad_type, **dd)
        self.cell_3 = Cell(
            in_chs_left=1080, out_chs_left=216, in_chs_right=1080, out_chs_right=216, pad_type=pad_type, **dd)

        self.cell_4 = Cell(
            in_chs_left=1080, out_chs_left=432, in_chs_right=1080, out_chs_right=432, pad_type=pad_type,
            is_reduction=True, **dd)
        self.cell_5 = Cell(
            in_chs_left=1080, out_chs_left=432, in_chs_right=2160, out_chs_right=432, pad_type=pad_type,
            match_prev_layer_dims=True, **dd)
        self.cell_6 = Cell(
            in_chs_left=2160, out_chs_left=432, in_chs_right=2160, out_chs_right=432, pad_type=pad_type, **dd)
        self.cell_7 = Cell(
            in_chs_left=2160, out_chs_left=432, in_chs_right=2160, out_chs_right=432, pad_type=pad_type, **dd)

        self.cell_8 = Cell(
            in_chs_left=2160, out_chs_left=864, in_chs_right=2160, out_chs_right=864, pad_type=pad_type,
            is_reduction=True, **dd)
        self.cell_9 = Cell(
            in_chs_left=2160, out_chs_left=864, in_chs_right=4320, out_chs_right=864, pad_type=pad_type,
            match_prev_layer_dims=True, **dd)
        self.cell_10 = Cell(
            in_chs_left=4320, out_chs_left=864, in_chs_right=4320, out_chs_right=864, pad_type=pad_type, **dd)
        self.cell_11 = Cell(
            in_chs_left=4320, out_chs_left=864, in_chs_right=4320, out_chs_right=864, pad_type=pad_type, **dd)
        self.act = nn.ReLU()
        self.feature_info = [
            dict(num_chs=96, reduction=2, module='conv_0'),
            dict(num_chs=270, reduction=4, module='cell_stem_1.conv_1x1.act'),
            dict(num_chs=1080, reduction=8, module='cell_4.conv_1x1.act'),
            dict(num_chs=2160, reduction=16, module='cell_8.conv_1x1.act'),
            dict(num_chs=4320, reduction=32, module='act'),
        ]
        # stem / cells in forward order, and the layer index of the cell output used for each feature_info tap
        self.layer_names = ['conv_0', 'cell_stem_0', 'cell_stem_1'] + [f'cell_{i}' for i in range(12)]
        tap_layers = ('conv_0', 'cell_stem_0', 'cell_3', 'cell_7', 'cell_11')
        self.feature_ends = [self.layer_names.index(n) for n in tap_layers]

        self.global_pool, self.head_drop, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool, drop_rate=drop_rate, **dd)
        self.head_hidden_size = self.num_features * self.global_pool.feat_mult()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem=r'^conv_0|cell_stem_[01]', blocks=r'^cell_(\d+)')

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.last_linear

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None, device=None, dtype=None):
        dd = get_device_dtype(self, device=device, dtype=dtype)
        self.num_classes = num_classes
        if global_pool is None:
            global_pool = self.global_pool.pool_type
        self.global_pool, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool, **dd)
        self.head_hidden_size = self.num_features * self.global_pool.feat_mult()
        self.global_pool.train(self.training)
        self.last_linear.train(self.training)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor.
            indices: Take last n features if int, all if None, select matching indices if sequence.
                Index 0 is the conv_0 stem output, indices 1-3 the ReLU activated outputs of cell_stem_0, cell_3
                and cell_7 (as consumed by the next cell), index 4 the final activated cell_11 output.
            norm: Apply norm layer to compatible intermediates (unused, no final norm).
            stop_early: Stop iterating over blocks when last desired intermediate hit.
            output_fmt: Shape of intermediate feature outputs.
            intermediates_only: Only return intermediate features.

        Returns:
            Features and list of intermediate features or just intermediate features.
        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.feature_ends), indices)
        last_idx = len(self.feature_ends) - 1

        # forward pass, each cell takes the outputs of the previous two layers
        layer_names = self.layer_names
        if stop_early:
            layer_names = layer_names[:self.feature_ends[max_index] + 1]
        x_prev = x
        feat_idx = 0
        for i, n in enumerate(layer_names):
            layer = getattr(self, n)
            if i < 2:
                x_prev, x = x, layer(x)  # conv_0 and cell_stem_0 take a single input
            else:
                x_prev, x = x, layer(x_prev, x)
            if i == self.feature_ends[feat_idx]:
                if feat_idx == last_idx:
                    x = self.act(x)
                if feat_idx in take_indices:
                    if 0 < feat_idx < last_idx:
                        # match the feature_info hook on the next cell's conv_1x1.act
                        intermediates.append(F.relu(x))
                    else:
                        intermediates.append(x)
                feat_idx += 1

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
            prune_norm: Whether to prune normalization layers (unused, no final norm).
            prune_head: Whether to prune the classifier head.

        Returns:
            List of indices that were kept.
        """
        take_indices, max_index = feature_take_indices(len(self.feature_ends), indices)
        num_layers = self.feature_ends[max_index] + 1
        prune_end = num_layers < len(self.layer_names)
        for n in self.layer_names[num_layers:]:
            setattr(self, n, None)  # cells take two inputs, can't be run through Identity (see forward_features)
        self.layer_names = self.layer_names[:num_layers]
        if prune_end:
            # features now end at the last kept tap, the head is sized for the final features so it goes too
            self.num_features = self.head_hidden_size = self.feature_info[max_index]['num_chs']
        if prune_head or prune_end:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        # the layer after each feature tap is checked for None to support pruning (prune_intermediate_layers)
        x_conv_0 = self.conv_0(x)
        if self.cell_stem_0 is None:
            return x_conv_0
        x_stem_0 = self.cell_stem_0(x_conv_0)
        if self.cell_stem_1 is None:
            return x_stem_0
        x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        if self.cell_4 is None:
            return x_cell_3
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
        if self.cell_8 is None:
            return x_cell_7
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
        x = self.act(x_cell_11)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.head_drop(x)
        return x if pre_logits else self.last_linear(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_pnasnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        PNASNet5Large,
        variant,
        pretrained,
        feature_cfg=dict(feature_cls='hook', no_rewrite=True),  # not possible to re-write this model
        **kwargs,
    )


default_cfgs = generate_default_cfgs({
    'pnasnet5large.tf_in1k': {
        'hf_hub_id': 'timm/',
        'input_size': (3, 331, 331),
        'pool_size': (11, 11),
        'crop_pct': 0.911,
        'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'num_classes': 1000,
        'first_conv': 'conv_0.conv',
        'classifier': 'last_linear',
        'license': 'apache-2.0',
    },
})


@register_model
def pnasnet5large(pretrained=False, **kwargs) -> PNASNet5Large:
    r"""PNASNet-5 model architecture from the
    `"Progressive Neural Architecture Search"
    <https://arxiv.org/abs/1712.00559>`_ paper.
    """
    model_kwargs = dict(pad_type='same', **kwargs)
    return _create_pnasnet('pnasnet5large', pretrained, **model_kwargs)
