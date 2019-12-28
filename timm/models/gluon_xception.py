"""Pytorch impl of Gluon Xception
This is a port of the Gluon Xception code and weights, itself ported from a PyTorch DeepLab impl.

Gluon model: (https://gluon-cv.mxnet.io/_modules/gluoncv/model_zoo/xception.html)
Original PyTorch DeepLab impl: https://github.com/jfzhang95/pytorch-deeplab-xception

Hacked together by Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .registry import register_model
from .helpers import load_pretrained
from .adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

__all__ = ['Xception65', 'Xception71']

default_cfgs = {
    'gluon_xception65': {
        'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_xception-7015a15c.pth',
        'input_size': (3, 299, 299),
        'crop_pct': 0.875,
        'pool_size': (10, 10),
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'num_classes': 1000,
        'first_conv': 'conv1',
        'classifier': 'fc'
        # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
    },
    'gluon_xception71': {
        'url': '',
        'input_size': (3, 299, 299),
        'crop_pct': 0.875,
        'pool_size': (10, 10),
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'num_classes': 1000,
        'first_conv': 'conv1',
        'classifier': 'fc'
        # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
    }
}


""" PADDING NOTES
The original PyTorch and Gluon impl of these models dutifully reproduced the 
aligned padding added to Tensorflow models for Deeplab. This padding was compensating
for  Tensorflow 'SAME' padding. PyTorch symmetric padding behaves the way we'd want it to. 

So, I'm phasing out the 'fixed_padding' ported from TF and replacing with normal 
PyTorch padding, some asserts to validate the equivalence for any scenario we'd 
care about before removing altogether.
"""
_USE_FIXED_PAD = False


def _pytorch_padding(kernel_size, stride=1, dilation=1, **_):
    if _USE_FIXED_PAD:
        return 0  # FIXME remove once verified
    else:
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2

        # FIXME remove once verified
        fp = _fixed_padding(kernel_size, dilation)
        assert all(padding == p for p in fp)

        return padding


def _fixed_padding(kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return [pad_beg, pad_end, pad_beg, pad_end]


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1,
                 dilation=1, bias=False, norm_layer=None, norm_kwargs=None):
        super(SeparableConv2d, self).__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self.kernel_size = kernel_size
        self.dilation = dilation

        padding = _fixed_padding(self.kernel_size, self.dilation)
        if _USE_FIXED_PAD and any(p > 0 for p in padding):
            self.fixed_padding = nn.ZeroPad2d(padding)
        else:
            self.fixed_padding = None

        # depthwise convolution
        self.conv_dw = nn.Conv2d(
            inplanes, inplanes, kernel_size, stride=stride,
            padding=_pytorch_padding(kernel_size, stride, dilation), dilation=dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(num_features=inplanes, **norm_kwargs)
        # pointwise convolution
        self.conv_pw = nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias)

    def forward(self, x):
        if self.fixed_padding is not None:
            # FIXME remove once verified
            x = self.fixed_padding(x)
        x = self.conv_dw(x)
        x = self.bn(x)
        x = self.conv_pw(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, num_reps, stride=1, dilation=1, norm_layer=None,
                 norm_kwargs=None, start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if planes != inplanes or stride != 1:
            self.skip = nn.Sequential()
            self.skip.add_module('conv1', nn.Conv2d(
                inplanes, planes, 1, stride=stride, bias=False)),
            self.skip.add_module('bn1', norm_layer(num_features=planes, **norm_kwargs))
        else:
            self.skip = None

        rep = OrderedDict()
        l = 1
        filters = inplanes
        if grow_first:
            if start_with_relu:
                rep['act%d' % l] = nn.ReLU(inplace=False)  # NOTE: silent failure if inplace=True here
            rep['conv%d' % l] = SeparableConv2d(
                inplanes, planes, 3, 1, dilation, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            rep['bn%d' % l] = norm_layer(num_features=planes, **norm_kwargs)
            filters = planes
            l += 1

        for _ in range(num_reps - 1):
            if grow_first or start_with_relu:
                # FIXME being conservative with inplace here, think it's fine to leave True?
                rep['act%d' % l] = nn.ReLU(inplace=grow_first or not start_with_relu)
            rep['conv%d' % l] = SeparableConv2d(
                filters, filters, 3, 1, dilation, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            rep['bn%d' % l] = norm_layer(num_features=filters, **norm_kwargs)
            l += 1

        if not grow_first:
            rep['act%d' % l] = nn.ReLU(inplace=True)
            rep['conv%d' % l] = SeparableConv2d(
                inplanes, planes, 3, 1, dilation, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            rep['bn%d' % l] = norm_layer(num_features=planes, **norm_kwargs)
            l += 1

        if stride != 1:
            rep['act%d' % l] = nn.ReLU(inplace=True)
            rep['conv%d' % l] = SeparableConv2d(
                planes, planes, 3, stride, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            rep['bn%d' % l] = norm_layer(num_features=planes, **norm_kwargs)
            l += 1
        elif is_last:
            rep['act%d' % l] = nn.ReLU(inplace=True)
            rep['conv%d' % l] = SeparableConv2d(
                planes, planes, 3, 1, dilation, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            rep['bn%d' % l] = norm_layer(num_features=planes, **norm_kwargs)
            l += 1
        self.rep = nn.Sequential(rep)

    def forward(self, x):
        skip = x
        if self.skip is not None:
            skip = self.skip(skip)
        x = self.rep(x) + skip
        return x


class Xception65(nn.Module):
    """Modified Aligned Xception
    """

    def __init__(self, num_classes=1000, in_chans=3, output_stride=32, norm_layer=nn.BatchNorm2d,
                 norm_kwargs=None, drop_rate=0., global_pool='avg'):
        super(Xception65, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 1)
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(num_features=32, **norm_kwargs)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(num_features=64)

        self.block1 = Block(
            64, 128, num_reps=2, stride=2,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, start_with_relu=False)
        self.block2 = Block(
            128, 256, num_reps=2, stride=2,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, start_with_relu=False, grow_first=True)
        self.block3 = Block(
            256, 728, num_reps=2, stride=entry_block3_stride,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        self.mid = nn.Sequential(OrderedDict([('block%d' % i,  Block(
            728, 728, num_reps=3, stride=1, dilation=middle_block_dilation,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, start_with_relu=True, grow_first=True))
                                              for i in range(4, 20)]))

        # Exit flow
        self.block20 = Block(
            728, 1024, num_reps=2, stride=exit_block20_stride, dilation=exit_block_dilations[0],
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(
            1024, 1536, 3, stride=1, dilation=exit_block_dilations[1],
            norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.bn3 = norm_layer(num_features=1536, **norm_kwargs)

        self.conv4 = SeparableConv2d(
            1536, 1536, 3, stride=1, dilation=exit_block_dilations[1],
            norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.bn4 = norm_layer(num_features=1536, **norm_kwargs)

        self.num_features = 2048
        self.conv5 = SeparableConv2d(
            1536, self.num_features, 3, stride=1, dilation=exit_block_dilations[1],
            norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.bn5 = norm_layer(num_features=self.num_features, **norm_kwargs)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes) if num_classes else None

    def forward_features(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        # c1 = x
        x = self.block2(x)
        # c2 = x
        x = self.block3(x)

        # Middle flow
        x = self.mid(x)
        # c3 = x

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


class Xception71(nn.Module):
    """Modified Aligned Xception
    """

    def __init__(self, num_classes=1000, in_chans=3, output_stride=32, norm_layer=nn.BatchNorm2d,
                 norm_kwargs=None, drop_rate=0., global_pool='avg'):
        super(Xception71, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 1)
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError
        
        # Entry flow
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(num_features=32, **norm_kwargs)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(num_features=64)

        self.block1 = Block(
            64, 128, num_reps=2, stride=2, norm_layer=norm_layer,
            norm_kwargs=norm_kwargs, start_with_relu=False)
        self.block2 = nn.Sequential(*[
            Block(
                128, 256, num_reps=2, stride=1, norm_layer=norm_layer,
                norm_kwargs=norm_kwargs, start_with_relu=False, grow_first=True),
            Block(
                256, 256, num_reps=2, stride=2, norm_layer=norm_layer,
                norm_kwargs=norm_kwargs, start_with_relu=False, grow_first=True),
            Block(
                256, 728, num_reps=2, stride=2, norm_layer=norm_layer,
                norm_kwargs=norm_kwargs, start_with_relu=False, grow_first=True)])
        self.block3 = Block(
            728, 728, num_reps=2, stride=entry_block3_stride, norm_layer=norm_layer,
            norm_kwargs=norm_kwargs, start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        self.mid = nn.Sequential(OrderedDict([('block%d' % i, Block(
            728, 728, num_reps=3, stride=1, dilation=middle_block_dilation,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, start_with_relu=True, grow_first=True))
                                              for i in range(4, 20)]))

        # Exit flow
        self.block20 = Block(
            728, 1024, num_reps=2, stride=exit_block20_stride, dilation=exit_block_dilations[0],
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(
            1024, 1536, 3, stride=1, dilation=exit_block_dilations[1],
            norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.bn3 = norm_layer(num_features=1536, **norm_kwargs)

        self.conv4 = SeparableConv2d(
            1536, 1536, 3, stride=1, dilation=exit_block_dilations[1],
            norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.bn4 = norm_layer(num_features=1536, **norm_kwargs)

        self.num_features = 2048
        self.conv5 = SeparableConv2d(
            1536, self.num_features, 3, stride=1, dilation=exit_block_dilations[1],
            norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.bn5 = norm_layer(num_features=self.num_features, **norm_kwargs)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes) if num_classes else None

    def forward_features(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        # low_level_feat = x
        x = self.block2(x)
        # c2 = x
        x = self.block3(x)

        # Middle flow
        x = self.mid(x)
        # c3 = x

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


@register_model
def gluon_xception65(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ Modified Aligned Xception-65
    """
    default_cfg = default_cfgs['gluon_xception65']
    model = Xception65(num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_xception71(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ Modified Aligned Xception-71
    """
    default_cfg = default_cfgs['gluon_xception71']
    model = Xception71(num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model

