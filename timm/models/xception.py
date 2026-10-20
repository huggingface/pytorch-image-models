"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene

Creates an Xception Model as defined in (Francois Chollet):
* `Xception: Deep Learning with Depthwise Separable Convolutions` - https://arxiv.org/abs/1610.02357

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from timm.layers import create_classifier, get_device_dtype
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._registry import register_model, generate_default_cfgs, register_model_deprecations

__all__ = ['Xception']


class SeparableConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=False,
            **dd,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=False, **dd)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            reps: int,
            strides: int = 1,
            start_with_relu: bool = True,
            grow_first: bool = True,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()

        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False, **dd)
            self.skipbn = nn.BatchNorm2d(out_channels, **dd)
        else:
            self.skip = None

        rep = []
        for i in range(reps):
            if grow_first:
                inc = in_channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = in_channels
                outc = in_channels if i < (reps - 1) else out_channels
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(inc, outc, 3, stride=1, padding=1, **dd))
            rep.append(nn.BatchNorm2d(outc, **dd))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(
            self,
            num_classes: int = 1000,
            in_chans: int = 3,
            drop_rate: float = 0.,
            global_pool: str = 'avg',
            device=None,
            dtype=None,
    ):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.drop_rate = drop_rate
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.num_features = 2048

        self.conv1 = nn.Conv2d(in_chans, 32, 3, 2, 0, bias=False, **dd)
        self.bn1 = nn.BatchNorm2d(32, **dd)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False, **dd)
        self.bn2 = nn.BatchNorm2d(64, **dd)
        self.act2 = nn.ReLU(inplace=True)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, **dd)
        self.block2 = Block(128, 256, 2, 2, **dd)
        self.block3 = Block(256, 728, 2, 2, **dd)

        self.block4 = Block(728, 728, 3, 1, **dd)
        self.block5 = Block(728, 728, 3, 1, **dd)
        self.block6 = Block(728, 728, 3, 1, **dd)
        self.block7 = Block(728, 728, 3, 1, **dd)

        self.block8 = Block(728, 728, 3, 1, **dd)
        self.block9 = Block(728, 728, 3, 1, **dd)
        self.block10 = Block(728, 728, 3, 1, **dd)
        self.block11 = Block(728, 728, 3, 1, **dd)

        self.block12 = Block(728, 1024, 2, 2, grow_first=False, **dd)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1, **dd)
        self.bn3 = nn.BatchNorm2d(1536, **dd)
        self.act3 = nn.ReLU(inplace=True)

        self.conv4 = SeparableConv2d(1536, self.num_features, 3, 1, 1, **dd)
        self.bn4 = nn.BatchNorm2d(self.num_features, **dd)
        self.act4 = nn.ReLU(inplace=True)
        self.feature_info = [
            dict(num_chs=64, reduction=2, module='act2'),
            dict(num_chs=128, reduction=4, module='block2.rep.0'),
            dict(num_chs=256, reduction=8, module='block3.rep.0'),
            dict(num_chs=728, reduction=16, module='block12.rep.0'),
            dict(num_chs=2048, reduction=32, module='act4'),
        ]
        # number of blocks run before each feature_info tap (stem, block1, block2, block11, exit flow), block taps
        # are the ReLU at the start of the next block (as hooked by features_only)
        self.feature_ends = [0, 1, 2, 11, 12]

        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool, **dd)
        self.head_hidden_size = self.num_features * self.global_pool.feat_mult()

        # #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^conv[12]|bn[12]',
            blocks=[
                (r'^block(\d+)', None),
                (r'^conv[34]|bn[34]', (99,)),
            ],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, "gradient checkpointing not supported"

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        dd = get_device_dtype(self)
        self.num_classes = num_classes
        if global_pool is None:
            global_pool = self.global_pool.pool_type
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool, **dd)
        self.head_hidden_size = self.num_features * self.global_pool.feat_mult()
        self.global_pool.train(self.training)
        self.fc.train(self.training)

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
                Index 0 is the stem output, indices 1-3 the ReLU activated outputs of block1, block2 and block11
                (the ReLU that starts the following block), index 4 the final (act4) output.
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

        # forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        if 0 in take_indices:
            intermediates.append(x)

        num_blocks = self.feature_ends[max_index] if stop_early else self.feature_ends[-1]
        feat_idx = 1
        for i in range(1, num_blocks + 1):
            x = getattr(self, f'block{i}')(x)
            if feat_idx < last_idx and i == self.feature_ends[feat_idx]:
                if feat_idx in take_indices:
                    intermediates.append(F.relu(x))
                feat_idx += 1

        if intermediates_only and max_index < last_idx:
            return intermediates

        if num_blocks == self.feature_ends[-1]:
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.act3(x)

            x = self.conv4(x)
            x = self.bn4(x)
            x = self.act4(x)
            if last_idx in take_indices:
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
        """Prune layers not required for specified intermediates.

        Args:
            indices: Indices of intermediate layers to keep.
            prune_norm: Whether to prune normalization layers (unused, no final norm).
            prune_head: Whether to prune the classifier head.

        Returns:
            List of indices that were kept.
        """
        take_indices, max_index = feature_take_indices(len(self.feature_ends), indices)
        for i in range(self.feature_ends[max_index] + 1, self.feature_ends[-1] + 1):
            setattr(self, f'block{i}', nn.Identity())
        prune_end = max_index < len(self.feature_ends) - 1
        if prune_end:
            for n in ('conv3', 'bn3', 'act3', 'conv4', 'bn4', 'act4'):
                setattr(self, n, nn.Identity())
            # features now end at the last kept tap, the head is sized for the final features so it goes too
            self.num_features = self.head_hidden_size = self.feature_info[max_index]['num_chs']
        if prune_head or prune_end:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, self.drop_rate, training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _xception(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        Xception, variant, pretrained,
        feature_cfg=dict(feature_cls='hook'),
        **kwargs)


default_cfgs = generate_default_cfgs({
    'legacy_xception.tf_in1k': {
        'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/xception-43020ad28.pth',
        'input_size': (3, 299, 299),
        'pool_size': (10, 10),
        'crop_pct': 0.8975,
        'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'num_classes': 1000,
        'first_conv': 'conv1',
        'classifier': 'fc',
        'license': 'apache-2.0',
        # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
    }
})


@register_model
def legacy_xception(pretrained=False, **kwargs) -> Xception:
    return _xception('legacy_xception', pretrained=pretrained, **kwargs)


register_model_deprecations(__name__, {
    'xception': 'legacy_xception',
})
