"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman
"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .helpers import load_pretrained
from .adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


__all__ = ['ResNet', 'BasicBlock', 'Bottleneck']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'resnet18': _cfg(url='https://download.pytorch.org/models/resnet18-5c106cde.pth'),
    'resnet34': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth'),
    'resnet26': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26-9aa10e23.pth',
        interpolation='bicubic'),
    'resnet26d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26d-69e92c46.pth',
        interpolation='bicubic'),
    'resnet50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth',
        interpolation='bicubic'),
    'resnet50d': _cfg(
        url='',
        interpolation='bicubic'),
    'resnet101': _cfg(url='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'),
    'resnet152': _cfg(url='https://download.pytorch.org/models/resnet152-b121ed2d.pth'),
    'tv_resnet34': _cfg(url='https://download.pytorch.org/models/resnet34-333f7ec4.pth'),
    'tv_resnet50': _cfg(url='https://download.pytorch.org/models/resnet50-19c8e357.pth'),
    'wide_resnet50_2': _cfg(url='https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth'),
    'wide_resnet101_2': _cfg(url='https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'),
    'resnext50_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50_32x4d-068914d1.pth',
        interpolation='bicubic'),
    'resnext50d_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50d_32x4d-103e99f8.pth',
        interpolation='bicubic'),
    'resnext101_32x4d': _cfg(url=''),
    'resnext101_32x8d': _cfg(url='https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'),
    'resnext101_64x4d': _cfg(url=''),
    'tv_resnext50_32x4d': _cfg(url='https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'),
    'ig_resnext101_32x8d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth'),
    'ig_resnext101_32x16d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth'),
    'ig_resnext101_32x32d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth'),
    'ig_resnext101_32x48d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth'),
    'ssl_resnet18':  _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth'),
    'ssl_resnet50':  _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth'),
    'ssl_resnext50_32x4d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth'),
    'ssl_resnext101_32x4d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth'),
    'ssl_resnext101_32x8d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth'),
    'ssl_resnext101_32x16d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth'),
    'swsl_resnet18': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth'),
    'swsl_resnet50': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth'),
    'swsl_resnext50_32x4d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth'),
    'swsl_resnext101_32x4d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth'),
    'swsl_resnext101_32x8d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x8-b4712904.pth'),
    'swsl_resnext101_32x16d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth'),
    'seresnext26d_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26d_32x4d-80fa48a3.pth',
        interpolation='bicubic'),
    'seresnext26t_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26t_32x4d-361bc1c4.pth',
        interpolation='bicubic'),
    'seresnext26tn_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26tn_32x4d-569cb627.pth',
        interpolation='bicubic'),
    'skresnet26d': _cfg()
}


def _get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class SEModule(nn.Module):

    def __init__(self, channels, reduction_channels):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            reduction_channels, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.fc1(x_se)
        x_se = self.relu(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()


class BasicBlock(nn.Module):
    __constants__ = ['se', 'downsample']  # for pre 1.4 torchscript compat
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, base_width=64, use_se=False,
                 reduce_first=1, dilation=1, previous_dilation=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=stride, padding=dilation,
            dilation=dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=previous_dilation,
            dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(outplanes)
        self.se = SEModule(outplanes, planes // 4) if use_se else None
        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act2(out)

        return out


class Bottleneck(nn.Module):
    __constants__ = ['se', 'downsample']  # for pre 1.4 torchscript compat
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, base_width=64, use_se=False,
                 reduce_first=1, dilation=1, previous_dilation=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.se = SEModule(outplanes, planes // 4) if use_se else None
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act3(out)

        return out


class SelectiveKernelAttn(nn.Module):
    def __init__(self, channels, num_paths=2, num_attn_feat=32,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(SelectiveKernelAttn, self).__init__()
        self.num_paths = num_paths
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_reduce = nn.Conv2d(channels, num_attn_feat, kernel_size=1, bias=False)
        self.bn = norm_layer(num_attn_feat)
        self.act = act_layer(inplace=True)
        self.fc_select = nn.Conv2d(num_attn_feat, channels * num_paths, kernel_size=1)

    def forward(self, x):
        assert x.shape[1] == self.num_paths
        x = torch.sum(x, dim=1)
        #print('attn sum', x.shape)
        x = self.pool(x)
        #print('attn pool', x.shape)
        x = self.fc_reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_select(x)
        #print('attn sel', x.shape)
        x = x.view((x.shape[0], self.num_paths, x.shape[1]//self.num_paths) + x.shape[-2:])
        #print('attn spl', x.shape)
        x = torch.softmax(x, dim=1)
        return x


class SelectiveKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=[3, 5], attn_reduction=16,
                 min_attn_feat=32, stride=1, dilation=1, groups=1, keep_3x3=True, use_attn=True,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(SelectiveKernelConv, self).__init__()
        if not isinstance(kernel_size, list):
            assert kernel_size >= 3 and kernel_size % 2
            kernel_size = [kernel_size] * 2
        else:
            # FIXME assert kernel sizes >=3 and odd
            pass
        if keep_3x3:
            dilation = [dilation * (k - 1) // 2 for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [dilation] * len(kernel_size)
        groups = min(out_channels // len(kernel_size), groups)

        self.conv_paths = nn.ModuleList()
        for k, d in zip(kernel_size, dilation):
            p = _get_padding(k, stride, d)
            self.conv_paths.append(nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    in_channels, out_channels, kernel_size=k, stride=stride, padding=p, dilation=d, groups=groups)),
                ('bn', norm_layer(out_channels)),
                ('act', act_layer(inplace=True))
            ])))

        if use_attn:
            num_attn_feat = max(int(out_channels / attn_reduction), min_attn_feat)
            self.attn = SelectiveKernelAttn(out_channels, len(kernel_size), num_attn_feat)
        else:
            self.attn = None

    def forward(self, x):
        x_paths = []
        for conv in self.conv_paths:
            xk = conv(x)
            x_paths.append(xk)
        if self.attn is not None:
            x_paths = torch.stack(x_paths, dim=1)
            # print('paths', x_paths.shape)
            x_attn = self.attn(x_paths)
            #print('attn', x_attn.shape)
            x = x_paths * x_attn
            #print('amul', x.shape)
            x = torch.sum(x, dim=1)
            #print('asum', x.shape)
        else:
            x = torch.cat(x_paths, dim=1)
        return x


class SelectiveKernelBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, base_width=64, use_se=False,
                 reduce_first=1, dilation=1, previous_dilation=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(SelectiveKernelBottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = SelectiveKernelConv(
            first_planes, width, stride=stride, dilation=dilation, groups=cardinality)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act3(out)

        return out


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

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int
        Numbers of layers in each block
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    use_se : bool, default False
        Enable Squeeze-Excitation module in blocks
    cardinality : int, default 1
        Number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64
        Factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    stem_width : int, default 64
        Number of channels in stem convolutions
    stem_type : str, default ''
        The type of stem:
          * '', default - a single 7x7 conv with a width of stem_width
          * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
          * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width//4 * 6, stem_width * 2
          * 'deep_tiered_narrow' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
    block_reduce_first: int, default 1
        Reduction factor for first convolution output width of residual blocks,
        1 for all archs except senets, where 2
    down_kernel_size: int, default 1
        Kernel size of residual block downsampling path, 1x1 for most archs, 3x3 for senets
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    output_stride : int, default 32
        Set the output stride of the network, 32, 16, or 8. Typically used in segmentation.
    act_layer : class, activation layer
    norm_layer : class, normalization layer
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    """
    def __init__(self, block, layers, num_classes=1000, in_chans=3, use_se=False,
                 cardinality=1, base_width=64, stem_width=64, stem_type='',
                 block_reduce_first=1, down_kernel_size=1, avg_down=False, output_stride=32,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, drop_rate=0.0, global_pool='avg',
                 zero_init_last_bn=True, block_args=None):
        block_args = block_args or dict()
        self.num_classes = num_classes
        deep_stem = 'deep' in stem_type
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.cardinality = cardinality
        self.base_width = base_width
        self.drop_rate = drop_rate
        self.expansion = block.expansion
        super(ResNet, self).__init__()

        # Stem
        if deep_stem:
            stem_chs_1 = stem_chs_2 = stem_width
            if 'tiered' in stem_type:
                stem_chs_1 = 3 * (stem_width // 4)
                stem_chs_2 = stem_width if 'narrow' in stem_type else 6 * (stem_width // 4)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs_1, 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs_1),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs_2),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs_2, self.inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.act1 = act_layer(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels, strides, dilations = [64, 128, 256, 512], [1, 2, 2, 2], [1] * 4
        if output_stride == 16:
            strides[3] = 1
            dilations[3] = 2
        elif output_stride == 8:
            strides[2:4] = [1, 1]
            dilations[2:4] = [2, 4]
        else:
            assert output_stride == 32
        llargs = list(zip(channels, layers, strides, dilations))
        lkwargs = dict(
            use_se=use_se, reduce_first=block_reduce_first, act_layer=act_layer, norm_layer=norm_layer,
            avg_down=avg_down, down_kernel_size=down_kernel_size, **block_args)
        self.layer1 = self._make_layer(block, *llargs[0], **lkwargs)
        self.layer2 = self._make_layer(block, *llargs[1], **lkwargs)
        self.layer3 = self._make_layer(block, *llargs[2], **lkwargs)
        self.layer4 = self._make_layer(block, *llargs[3], **lkwargs)

        # Head (Pooling and Classifier)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_features = 512 * block.expansion
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)

        last_bn_name = 'bn3' if 'Bottle' in block.__name__ else 'bn2'
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if zero_init_last_bn and 'layer' in n and last_bn_name in n:
                    # Initialize weight/gamma of last BN in each residual block to zero
                    nn.init.constant_(m.weight, 0.)
                else:
                    nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, reduce_first=1,
                    use_se=False, avg_down=False, down_kernel_size=1, **kwargs):
        norm_layer = kwargs.get('norm_layer')
        downsample = None
        down_kernel_size = 1 if stride == 1 and dilation == 1 else down_kernel_size
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_padding = _get_padding(down_kernel_size, stride)
            downsample_layers = []
            conv_stride = stride
            if avg_down:
                avg_stride = stride if dilation == 1 else 1
                conv_stride = 1
                downsample_layers = [nn.AvgPool2d(avg_stride, avg_stride, ceil_mode=True, count_include_pad=False)]
            downsample_layers += [
                nn.Conv2d(self.inplanes, planes * block.expansion, down_kernel_size,
                          stride=conv_stride, padding=downsample_padding, bias=False),
                norm_layer(planes * block.expansion)]
            downsample = nn.Sequential(*downsample_layers)

        first_dilation = 1 if dilation in (1, 2) else 2
        bkwargs = dict(
            cardinality=self.cardinality, base_width=self.base_width, reduce_first=reduce_first,
            use_se=use_se, **kwargs)
        layers = [block(
            self.inplanes, planes, stride, downsample, dilation=first_dilation, previous_dilation=dilation, **bkwargs)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, dilation=dilation, previous_dilation=dilation, **bkwargs))

        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        del self.fc
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes) if num_classes else None

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


@register_model
def resnet18(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-18 model.
    """
    default_cfg = default_cfgs['resnet18']
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def resnet34(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-34 model.
    """
    default_cfg = default_cfgs['resnet34']
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def resnet26(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-26 model.
    """
    default_cfg = default_cfgs['resnet26']
    model = ResNet(Bottleneck, [2, 2, 2, 2], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def skresnet26d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-26 model.
    """
    default_cfg = default_cfgs['skresnet26d']
    model = ResNet(
        SelectiveKernelBottleneck, [2, 2, 2, 2],  stem_width=32, stem_type='deep', avg_down=True,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model

@register_model
def resnet26d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-26 v1d model.
    This is technically a 28 layer ResNet, sticking with 'd' modifier from Gluon for now.
    """
    default_cfg = default_cfgs['resnet26d']
    model = ResNet(
        Bottleneck, [2, 2, 2, 2], stem_width=32, stem_type='deep', avg_down=True,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def resnet50(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-50 model.
    """
    default_cfg = default_cfgs['resnet50']
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def resnet50d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-50-D model.
    """
    default_cfg = default_cfgs['resnet50d']
    model = ResNet(
        Bottleneck, [3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def resnet101(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-101 model.
    """
    default_cfg = default_cfgs['resnet101']
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def resnet152(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-152 model.
    """
    default_cfg = default_cfgs['resnet152']
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tv_resnet34(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-34 model with original Torchvision weights.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfgs['tv_resnet34']
    if pretrained:
        load_pretrained(model, model.default_cfg, num_classes, in_chans)
    return model


@register_model
def tv_resnet50(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-50 model with original Torchvision weights.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfgs['tv_resnet50']
    if pretrained:
        load_pretrained(model, model.default_cfg, num_classes, in_chans)
    return model


@register_model
def wide_resnet50_2(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a Wide ResNet-50-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    model = ResNet(
        Bottleneck, [3, 4, 6, 3], base_width=128,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfgs['wide_resnet50_2']
    if pretrained:
        load_pretrained(model, model.default_cfg, num_classes, in_chans)
    return model


@register_model
def wide_resnet101_2(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a Wide ResNet-101-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same.
    """
    model = ResNet(
        Bottleneck, [3, 4, 23, 3], base_width=128,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfgs['wide_resnet101_2']
    if pretrained:
        load_pretrained(model, model.default_cfg, num_classes, in_chans)
    return model


@register_model
def resnext50_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNeXt50-32x4d model.
    """
    default_cfg = default_cfgs['resnext50_32x4d']
    model = ResNet(
        Bottleneck, [3, 4, 6, 3], cardinality=32, base_width=4,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def resnext50d_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNeXt50d-32x4d model. ResNext50 w/ deep stem & avg pool downsample
    """
    default_cfg = default_cfgs['resnext50d_32x4d']
    model = ResNet(
        Bottleneck, [3, 4, 6, 3], cardinality=32, base_width=4,
        stem_width=32, stem_type='deep', avg_down=True,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def resnext101_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNeXt-101 32x4d model.
    """
    default_cfg = default_cfgs['resnext101_32x4d']
    model = ResNet(
        Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=4,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def resnext101_32x8d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.
    """
    default_cfg = default_cfgs['resnext101_32x8d']
    model = ResNet(
        Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=8,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def resnext101_64x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNeXt101-64x4d model.
    """
    default_cfg = default_cfgs['resnext101_32x4d']
    model = ResNet(
        Bottleneck, [3, 4, 23, 3], cardinality=64, base_width=4,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def tv_resnext50_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNeXt50-32x4d model with original Torchvision weights.
    """
    default_cfg = default_cfgs['tv_resnext50_32x4d']
    model = ResNet(
        Bottleneck, [3, 4, 6, 3], cardinality=32, base_width=4,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def ig_resnext101_32x8d(pretrained=True, **kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=8, **kwargs)
    model.default_cfg = default_cfgs['ig_resnext101_32x8d']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def ig_resnext101_32x16d(pretrained=True, **kwargs):
    """Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=16, **kwargs)
    model.default_cfg = default_cfgs['ig_resnext101_32x16d']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def ig_resnext101_32x32d(pretrained=True, **kwargs):
    """Constructs a ResNeXt-101 32x32 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=32, **kwargs)
    model.default_cfg = default_cfgs['ig_resnext101_32x32d']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def ig_resnext101_32x48d(pretrained=True, **kwargs):
    """Constructs a ResNeXt-101 32x48 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=48, **kwargs)
    model.default_cfg = default_cfgs['ig_resnext101_32x48d']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def ssl_resnet18(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNet-18 model pre-trained on YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.default_cfg = default_cfgs['ssl_resnet18']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def ssl_resnet50(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNet-50 model pre-trained on YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model.default_cfg = default_cfgs['ssl_resnet50']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def ssl_resnext50_32x4d(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-50 32x4 model pre-trained on YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], cardinality=32, base_width=4, **kwargs)
    model.default_cfg = default_cfgs['ssl_resnext50_32x4d']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def ssl_resnext101_32x4d(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-101 32x4 model pre-trained on YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=4, **kwargs)
    model.default_cfg = default_cfgs['ssl_resnext101_32x4d']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def ssl_resnext101_32x8d(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-101 32x8 model pre-trained on YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=8, **kwargs)
    model.default_cfg = default_cfgs['ssl_resnext101_32x8d']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def ssl_resnext101_32x16d(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-101 32x16 model pre-trained on YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=16, **kwargs)
    model.default_cfg = default_cfgs['ssl_resnext101_32x16d']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def swsl_resnet18(pretrained=True, **kwargs):
    """Constructs a semi-weakly supervised Resnet-18 model pre-trained on 1B weakly supervised
       image dataset and finetuned on ImageNet.
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
       Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.default_cfg = default_cfgs['swsl_resnet18']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def swsl_resnet50(pretrained=True, **kwargs):
    """Constructs a semi-weakly supervised ResNet-50 model pre-trained on 1B weakly supervised
       image dataset and finetuned on ImageNet.
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
       Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model.default_cfg = default_cfgs['swsl_resnet50']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def swsl_resnext50_32x4d(pretrained=True, **kwargs):
    """Constructs a semi-weakly supervised ResNeXt-50 32x4 model pre-trained on 1B weakly supervised
       image dataset and finetuned on ImageNet.
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
       Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], cardinality=32, base_width=4, **kwargs)
    model.default_cfg = default_cfgs['swsl_resnext50_32x4d']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def swsl_resnext101_32x4d(pretrained=True, **kwargs):
    """Constructs a semi-weakly supervised ResNeXt-101 32x4 model pre-trained on 1B weakly supervised
       image dataset and finetuned on ImageNet.
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
       Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=4, **kwargs)
    model.default_cfg = default_cfgs['swsl_resnext101_32x4d']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def swsl_resnext101_32x8d(pretrained=True, **kwargs):
    """Constructs a semi-weakly supervised ResNeXt-101 32x8 model pre-trained on 1B weakly supervised
       image dataset and finetuned on ImageNet.
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
       Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=8, **kwargs)
    model.default_cfg = default_cfgs['swsl_resnext101_32x8d']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def swsl_resnext101_32x16d(pretrained=True, **kwargs):
    """Constructs a semi-weakly supervised ResNeXt-101 32x16 model pre-trained on 1B weakly supervised
       image dataset and finetuned on ImageNet.
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
       Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=16, **kwargs)
    model.default_cfg = default_cfgs['swsl_resnext101_32x16d']
    if pretrained:
        load_pretrained(model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def seresnext26d_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a SE-ResNeXt-26-D model.
    This is technically a 28 layer ResNet, using the 'D' modifier from Gluon / bag-of-tricks for
    combination of deep stem and avg_pool in downsample.
    """
    default_cfg = default_cfgs['seresnext26d_32x4d']
    model = ResNet(
        Bottleneck, [2, 2, 2, 2], cardinality=32, base_width=4,
        stem_width=32, stem_type='deep', avg_down=True, use_se=True,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def seresnext26t_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a SE-ResNet-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 48, 64 channels
    in the deep stem.
    """
    default_cfg = default_cfgs['seresnext26t_32x4d']
    model = ResNet(
        Bottleneck, [2, 2, 2, 2], cardinality=32, base_width=4,
        stem_width=32, stem_type='deep_tiered', avg_down=True, use_se=True,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def seresnext26tn_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a SE-ResNeXt-26-TN model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem. The channel number of the middle stem conv is narrower than the 'T' variant.
    """
    default_cfg = default_cfgs['seresnext26tn_32x4d']
    model = ResNet(
        Bottleneck, [2, 2, 2, 2], cardinality=32, base_width=4,
        stem_width=32, stem_type='deep_tiered_narrow', avg_down=True, use_se=True,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model
