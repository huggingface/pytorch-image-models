"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman
"""
import math

import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .helpers import load_pretrained, adapt_model_from_string
from .layers import SelectAdaptivePool2d, DropBlock2d, DropPath, AvgPool2dSame, create_attn
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
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50_32x4d_ra-d733960d.pth',
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
    'ecaresnext26tn_32x4d': _cfg(
        url='',
        interpolation='bicubic'),
    'ecaresnet18': _cfg(),
    'ecaresnet50': _cfg(),
    'ecaresnetlight': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45402/outputs/ECAResNetLight_4f34b35b.pth',
        interpolation='bicubic'),
    'ecaresnet50d': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45402/outputs/ECAResNet50D_833caf58.pth',
        interpolation='bicubic'),
    'ecaresnet50d_pruned': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45899/outputs/ECAResNet50D_P_9c67f710.pth',
        interpolation='bicubic'),
    'ecaresnet101d': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45402/outputs/ECAResNet101D_281c5844.pth',
        interpolation='bicubic'),
    'ecaresnet101d_pruned': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45610/outputs/ECAResNet101D_P_75a3370e.pth',
        interpolation='bicubic'),
}


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    __constants__ = ['se', 'downsample']  # for pre 1.4 torchscript compat
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act3(x)

        return x


def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


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
    def __init__(self, block, layers, num_classes=1000, in_chans=3,
                 cardinality=1, base_width=64, stem_width=64, stem_type='',
                 block_reduce_first=1, down_kernel_size=1, avg_down=False, output_stride=32,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, drop_rate=0.0, drop_path_rate=0.,
                 drop_block_rate=0., global_pool='avg', zero_init_last_bn=True, block_args=None):
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
        dp = DropPath(drop_path_rate) if drop_path_rate else None
        db_3 = DropBlock2d(drop_block_rate, 7, 0.25) if drop_block_rate else None
        db_4 = DropBlock2d(drop_block_rate, 7, 1.00) if drop_block_rate else None
        channels, strides, dilations = [64, 128, 256, 512], [1, 2, 2, 2], [1] * 4
        if output_stride == 16:
            strides[3] = 1
            dilations[3] = 2
        elif output_stride == 8:
            strides[2:4] = [1, 1]
            dilations[2:4] = [2, 4]
        else:
            assert output_stride == 32
        layer_args = list(zip(channels, layers, strides, dilations))
        layer_kwargs = dict(
            reduce_first=block_reduce_first, act_layer=act_layer, norm_layer=norm_layer,
            avg_down=avg_down, down_kernel_size=down_kernel_size, drop_path=dp, **block_args)
        self.layer1 = self._make_layer(block, *layer_args[0], **layer_kwargs)
        self.layer2 = self._make_layer(block, *layer_args[1], **layer_kwargs)
        self.layer3 = self._make_layer(block, drop_block=db_3, *layer_args[2], **layer_kwargs)
        self.layer4 = self._make_layer(block, drop_block=db_4, *layer_args[3], **layer_kwargs)

        # Head (Pooling and Classifier)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_features = 512 * block.expansion
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, reduce_first=1,
                    avg_down=False, down_kernel_size=1, **kwargs):
        downsample = None
        first_dilation = 1 if dilation in (1, 2) else 2
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_args = dict(
                in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=first_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**downsample_args) if avg_down else downsample_conv(**downsample_args)

        block_kwargs = dict(
            cardinality=self.cardinality, base_width=self.base_width, reduce_first=reduce_first,
            dilation=dilation, **kwargs)
        layers = [block(self.inplanes, planes, stride, downsample, first_dilation=first_dilation, **block_kwargs)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes, **block_kwargs) for _ in range(1, blocks)]

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
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
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
        Bottleneck, [2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32, stem_type='deep', avg_down=True,
        num_classes=num_classes, in_chans=in_chans, block_args=dict(attn_layer='se'), **kwargs)
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
        stem_width=32, stem_type='deep_tiered', avg_down=True,
        num_classes=num_classes, in_chans=in_chans, block_args=dict(attn_layer='se'), **kwargs)
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
        stem_width=32, stem_type='deep_tiered_narrow', avg_down=True,
        num_classes=num_classes, in_chans=in_chans, block_args=dict(attn_layer='se'), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def ecaresnext26tn_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs an ECA-ResNeXt-26-TN model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem. The channel number of the middle stem conv is narrower than the 'T' variant.
    this model replaces SE module with the ECA module
    """
    default_cfg = default_cfgs['ecaresnext26tn_32x4d']
    block_args = dict(attn_layer='eca')
    model = ResNet(
        Bottleneck, [2, 2, 2, 2], cardinality=32, base_width=4,
        stem_width=32, stem_type='deep_tiered_narrow', avg_down=True,
        num_classes=num_classes, in_chans=in_chans, block_args=block_args, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def ecaresnet18(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """ Constructs an ECA-ResNet-18 model.
    """
    default_cfg = default_cfgs['ecaresnet18']
    block_args = dict(attn_layer='eca')
    model = ResNet(
        BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_chans=in_chans, block_args=block_args, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def ecaresnet50(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs an ECA-ResNet-50 model.
    """
    default_cfg = default_cfgs['ecaresnet50']
    block_args = dict(attn_layer='eca')
    model = ResNet(
        Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, block_args=block_args, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model



@register_model
def ecaresnet50d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-50-D model with eca.
    """
    default_cfg = default_cfgs['ecaresnet50d']
    model = ResNet(
        Bottleneck, [3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True,
        num_classes=num_classes, in_chans=in_chans, block_args=dict(attn_layer='eca'), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model

@register_model
def ecaresnet50d_pruned(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-50-D model pruned with eca.
        The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf
    """
    default_cfg = default_cfgs['ecaresnet50d_pruned']
    model = ResNet(
        Bottleneck, [3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True,
        num_classes=num_classes, in_chans=in_chans, block_args=dict(attn_layer='eca'), **kwargs)
    model.default_cfg = default_cfg
    str_model = 'conv1.0.weight:[32, 3, 3, 3]***conv1.1.weight:[32]***conv1.3.weight:[32, 32, 3, 3]***conv1.4.weight:[32]***conv1.6.weight:[64, 32, 3, 3]***bn1.weight:[64]***layer1.0.conv1.weight:[47, 64, 1, 1]***layer1.0.bn1.weight:[47]***layer1.0.conv2.weight:[18, 47, 3, 3]***layer1.0.bn2.weight:[18]***layer1.0.conv3.weight:[19, 18, 1, 1]***layer1.0.bn3.weight:[19]***layer1.0.se.conv.weight:[1, 1, 5]***layer1.0.downsample.1.weight:[19, 64, 1, 1]***layer1.0.downsample.2.weight:[19]***layer1.1.conv1.weight:[52, 19, 1, 1]***layer1.1.bn1.weight:[52]***layer1.1.conv2.weight:[22, 52, 3, 3]***layer1.1.bn2.weight:[22]***layer1.1.conv3.weight:[19, 22, 1, 1]***layer1.1.bn3.weight:[19]***layer1.1.se.conv.weight:[1, 1, 5]***layer1.2.conv1.weight:[64, 19, 1, 1]***layer1.2.bn1.weight:[64]***layer1.2.conv2.weight:[35, 64, 3, 3]***layer1.2.bn2.weight:[35]***layer1.2.conv3.weight:[19, 35, 1, 1]***layer1.2.bn3.weight:[19]***layer1.2.se.conv.weight:[1, 1, 5]***layer2.0.conv1.weight:[85, 19, 1, 1]***layer2.0.bn1.weight:[85]***layer2.0.conv2.weight:[37, 85, 3, 3]***layer2.0.bn2.weight:[37]***layer2.0.conv3.weight:[171, 37, 1, 1]***layer2.0.bn3.weight:[171]***layer2.0.se.conv.weight:[1, 1, 5]***layer2.0.downsample.1.weight:[171, 19, 1, 1]***layer2.0.downsample.2.weight:[171]***layer2.1.conv1.weight:[107, 171, 1, 1]***layer2.1.bn1.weight:[107]***layer2.1.conv2.weight:[80, 107, 3, 3]***layer2.1.bn2.weight:[80]***layer2.1.conv3.weight:[171, 80, 1, 1]***layer2.1.bn3.weight:[171]***layer2.1.se.conv.weight:[1, 1, 5]***layer2.2.conv1.weight:[120, 171, 1, 1]***layer2.2.bn1.weight:[120]***layer2.2.conv2.weight:[85, 120, 3, 3]***layer2.2.bn2.weight:[85]***layer2.2.conv3.weight:[171, 85, 1, 1]***layer2.2.bn3.weight:[171]***layer2.2.se.conv.weight:[1, 1, 5]***layer2.3.conv1.weight:[125, 171, 1, 1]***layer2.3.bn1.weight:[125]***layer2.3.conv2.weight:[87, 125, 3, 3]***layer2.3.bn2.weight:[87]***layer2.3.conv3.weight:[171, 87, 1, 1]***layer2.3.bn3.weight:[171]***layer2.3.se.conv.weight:[1, 1, 5]***layer3.0.conv1.weight:[198, 171, 1, 1]***layer3.0.bn1.weight:[198]***layer3.0.conv2.weight:[126, 198, 3, 3]***layer3.0.bn2.weight:[126]***layer3.0.conv3.weight:[818, 126, 1, 1]***layer3.0.bn3.weight:[818]***layer3.0.se.conv.weight:[1, 1, 5]***layer3.0.downsample.1.weight:[818, 171, 1, 1]***layer3.0.downsample.2.weight:[818]***layer3.1.conv1.weight:[255, 818, 1, 1]***layer3.1.bn1.weight:[255]***layer3.1.conv2.weight:[232, 255, 3, 3]***layer3.1.bn2.weight:[232]***layer3.1.conv3.weight:[818, 232, 1, 1]***layer3.1.bn3.weight:[818]***layer3.1.se.conv.weight:[1, 1, 5]***layer3.2.conv1.weight:[256, 818, 1, 1]***layer3.2.bn1.weight:[256]***layer3.2.conv2.weight:[233, 256, 3, 3]***layer3.2.bn2.weight:[233]***layer3.2.conv3.weight:[818, 233, 1, 1]***layer3.2.bn3.weight:[818]***layer3.2.se.conv.weight:[1, 1, 5]***layer3.3.conv1.weight:[253, 818, 1, 1]***layer3.3.bn1.weight:[253]***layer3.3.conv2.weight:[235, 253, 3, 3]***layer3.3.bn2.weight:[235]***layer3.3.conv3.weight:[818, 235, 1, 1]***layer3.3.bn3.weight:[818]***layer3.3.se.conv.weight:[1, 1, 5]***layer3.4.conv1.weight:[256, 818, 1, 1]***layer3.4.bn1.weight:[256]***layer3.4.conv2.weight:[225, 256, 3, 3]***layer3.4.bn2.weight:[225]***layer3.4.conv3.weight:[818, 225, 1, 1]***layer3.4.bn3.weight:[818]***layer3.4.se.conv.weight:[1, 1, 5]***layer3.5.conv1.weight:[256, 818, 1, 1]***layer3.5.bn1.weight:[256]***layer3.5.conv2.weight:[239, 256, 3, 3]***layer3.5.bn2.weight:[239]***layer3.5.conv3.weight:[818, 239, 1, 1]***layer3.5.bn3.weight:[818]***layer3.5.se.conv.weight:[1, 1, 5]***layer4.0.conv1.weight:[492, 818, 1, 1]***layer4.0.bn1.weight:[492]***layer4.0.conv2.weight:[237, 492, 3, 3]***layer4.0.bn2.weight:[237]***layer4.0.conv3.weight:[2022, 237, 1, 1]***layer4.0.bn3.weight:[2022]***layer4.0.se.conv.weight:[1, 1, 7]***layer4.0.downsample.1.weight:[2022, 818, 1, 1]***layer4.0.downsample.2.weight:[2022]***layer4.1.conv1.weight:[512, 2022, 1, 1]***layer4.1.bn1.weight:[512]***layer4.1.conv2.weight:[500, 512, 3, 3]***layer4.1.bn2.weight:[500]***layer4.1.conv3.weight:[2022, 500, 1, 1]***layer4.1.bn3.weight:[2022]***layer4.1.se.conv.weight:[1, 1, 7]***layer4.2.conv1.weight:[512, 2022, 1, 1]***layer4.2.bn1.weight:[512]***layer4.2.conv2.weight:[490, 512, 3, 3]***layer4.2.bn2.weight:[490]***layer4.2.conv3.weight:[2022, 490, 1, 1]***layer4.2.bn3.weight:[2022]***layer4.2.se.conv.weight:[1, 1, 7]***fc.weight:[1000, 2022]***layer1_2_conv3_M.weight:[256, 19]***layer2_3_conv3_M.weight:[512, 171]***layer3_5_conv3_M.weight:[1024, 818]***layer4_2_conv3_M.weight:[2048, 2022]'
    model = adapt_model_from_string(model, str_model)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model

@register_model
def ecaresnetlight(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-50-D light model with eca.
    """
    default_cfg = default_cfgs['ecaresnetlight']
    model = ResNet(
        Bottleneck, [1, 1, 11, 3], stem_width=32, avg_down=True,
        num_classes=num_classes, in_chans=in_chans, block_args=dict(attn_layer='eca'), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model

@register_model
def ecaresnet101d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-101-D model with eca.
    """
    default_cfg = default_cfgs['ecaresnet101d']
    model = ResNet(
        Bottleneck, [3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True,
        num_classes=num_classes, in_chans=in_chans, block_args=dict(attn_layer='eca'), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model




@register_model
def ecaresnet101d_pruned(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-101-D model pruned with eca.
       The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf
    """
    default_cfg = default_cfgs['ecaresnet101d_pruned']
    model = ResNet(
        Bottleneck, [3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True,
        num_classes=num_classes, in_chans=in_chans, block_args=dict(attn_layer='eca'), **kwargs)
    model.default_cfg = default_cfg
    str_model = 'conv1.0.weight:[32, 3, 3, 3]***conv1.1.weight:[32]***conv1.3.weight:[32, 32, 3, 3]***conv1.4.weight:[32]***conv1.6.weight:[64, 32, 3, 3]***bn1.weight:[64]***layer1.0.conv1.weight:[45, 64, 1, 1]***layer1.0.bn1.weight:[45]***layer1.0.conv2.weight:[25, 45, 3, 3]***layer1.0.bn2.weight:[25]***layer1.0.conv3.weight:[26, 25, 1, 1]***layer1.0.bn3.weight:[26]***layer1.0.se.conv.weight:[1, 1, 5]***layer1.0.downsample.1.weight:[26, 64, 1, 1]***layer1.0.downsample.2.weight:[26]***layer1.1.conv1.weight:[53, 26, 1, 1]***layer1.1.bn1.weight:[53]***layer1.1.conv2.weight:[20, 53, 3, 3]***layer1.1.bn2.weight:[20]***layer1.1.conv3.weight:[26, 20, 1, 1]***layer1.1.bn3.weight:[26]***layer1.1.se.conv.weight:[1, 1, 5]***layer1.2.conv1.weight:[60, 26, 1, 1]***layer1.2.bn1.weight:[60]***layer1.2.conv2.weight:[27, 60, 3, 3]***layer1.2.bn2.weight:[27]***layer1.2.conv3.weight:[26, 27, 1, 1]***layer1.2.bn3.weight:[26]***layer1.2.se.conv.weight:[1, 1, 5]***layer2.0.conv1.weight:[81, 26, 1, 1]***layer2.0.bn1.weight:[81]***layer2.0.conv2.weight:[24, 81, 3, 3]***layer2.0.bn2.weight:[24]***layer2.0.conv3.weight:[142, 24, 1, 1]***layer2.0.bn3.weight:[142]***layer2.0.se.conv.weight:[1, 1, 5]***layer2.0.downsample.1.weight:[142, 26, 1, 1]***layer2.0.downsample.2.weight:[142]***layer2.1.conv1.weight:[93, 142, 1, 1]***layer2.1.bn1.weight:[93]***layer2.1.conv2.weight:[49, 93, 3, 3]***layer2.1.bn2.weight:[49]***layer2.1.conv3.weight:[142, 49, 1, 1]***layer2.1.bn3.weight:[142]***layer2.1.se.conv.weight:[1, 1, 5]***layer2.2.conv1.weight:[102, 142, 1, 1]***layer2.2.bn1.weight:[102]***layer2.2.conv2.weight:[54, 102, 3, 3]***layer2.2.bn2.weight:[54]***layer2.2.conv3.weight:[142, 54, 1, 1]***layer2.2.bn3.weight:[142]***layer2.2.se.conv.weight:[1, 1, 5]***layer2.3.conv1.weight:[122, 142, 1, 1]***layer2.3.bn1.weight:[122]***layer2.3.conv2.weight:[78, 122, 3, 3]***layer2.3.bn2.weight:[78]***layer2.3.conv3.weight:[142, 78, 1, 1]***layer2.3.bn3.weight:[142]***layer2.3.se.conv.weight:[1, 1, 5]***layer3.0.conv1.weight:[101, 142, 1, 1]***layer3.0.bn1.weight:[101]***layer3.0.conv2.weight:[25, 101, 3, 3]***layer3.0.bn2.weight:[25]***layer3.0.conv3.weight:[278, 25, 1, 1]***layer3.0.bn3.weight:[278]***layer3.0.se.conv.weight:[1, 1, 5]***layer3.0.downsample.1.weight:[278, 142, 1, 1]***layer3.0.downsample.2.weight:[278]***layer3.1.conv1.weight:[239, 278, 1, 1]***layer3.1.bn1.weight:[239]***layer3.1.conv2.weight:[160, 239, 3, 3]***layer3.1.bn2.weight:[160]***layer3.1.conv3.weight:[278, 160, 1, 1]***layer3.1.bn3.weight:[278]***layer3.1.se.conv.weight:[1, 1, 5]***layer3.2.conv1.weight:[234, 278, 1, 1]***layer3.2.bn1.weight:[234]***layer3.2.conv2.weight:[156, 234, 3, 3]***layer3.2.bn2.weight:[156]***layer3.2.conv3.weight:[278, 156, 1, 1]***layer3.2.bn3.weight:[278]***layer3.2.se.conv.weight:[1, 1, 5]***layer3.3.conv1.weight:[250, 278, 1, 1]***layer3.3.bn1.weight:[250]***layer3.3.conv2.weight:[176, 250, 3, 3]***layer3.3.bn2.weight:[176]***layer3.3.conv3.weight:[278, 176, 1, 1]***layer3.3.bn3.weight:[278]***layer3.3.se.conv.weight:[1, 1, 5]***layer3.4.conv1.weight:[253, 278, 1, 1]***layer3.4.bn1.weight:[253]***layer3.4.conv2.weight:[191, 253, 3, 3]***layer3.4.bn2.weight:[191]***layer3.4.conv3.weight:[278, 191, 1, 1]***layer3.4.bn3.weight:[278]***layer3.4.se.conv.weight:[1, 1, 5]***layer3.5.conv1.weight:[251, 278, 1, 1]***layer3.5.bn1.weight:[251]***layer3.5.conv2.weight:[175, 251, 3, 3]***layer3.5.bn2.weight:[175]***layer3.5.conv3.weight:[278, 175, 1, 1]***layer3.5.bn3.weight:[278]***layer3.5.se.conv.weight:[1, 1, 5]***layer3.6.conv1.weight:[230, 278, 1, 1]***layer3.6.bn1.weight:[230]***layer3.6.conv2.weight:[128, 230, 3, 3]***layer3.6.bn2.weight:[128]***layer3.6.conv3.weight:[278, 128, 1, 1]***layer3.6.bn3.weight:[278]***layer3.6.se.conv.weight:[1, 1, 5]***layer3.7.conv1.weight:[244, 278, 1, 1]***layer3.7.bn1.weight:[244]***layer3.7.conv2.weight:[154, 244, 3, 3]***layer3.7.bn2.weight:[154]***layer3.7.conv3.weight:[278, 154, 1, 1]***layer3.7.bn3.weight:[278]***layer3.7.se.conv.weight:[1, 1, 5]***layer3.8.conv1.weight:[244, 278, 1, 1]***layer3.8.bn1.weight:[244]***layer3.8.conv2.weight:[159, 244, 3, 3]***layer3.8.bn2.weight:[159]***layer3.8.conv3.weight:[278, 159, 1, 1]***layer3.8.bn3.weight:[278]***layer3.8.se.conv.weight:[1, 1, 5]***layer3.9.conv1.weight:[238, 278, 1, 1]***layer3.9.bn1.weight:[238]***layer3.9.conv2.weight:[97, 238, 3, 3]***layer3.9.bn2.weight:[97]***layer3.9.conv3.weight:[278, 97, 1, 1]***layer3.9.bn3.weight:[278]***layer3.9.se.conv.weight:[1, 1, 5]***layer3.10.conv1.weight:[244, 278, 1, 1]***layer3.10.bn1.weight:[244]***layer3.10.conv2.weight:[149, 244, 3, 3]***layer3.10.bn2.weight:[149]***layer3.10.conv3.weight:[278, 149, 1, 1]***layer3.10.bn3.weight:[278]***layer3.10.se.conv.weight:[1, 1, 5]***layer3.11.conv1.weight:[253, 278, 1, 1]***layer3.11.bn1.weight:[253]***layer3.11.conv2.weight:[181, 253, 3, 3]***layer3.11.bn2.weight:[181]***layer3.11.conv3.weight:[278, 181, 1, 1]***layer3.11.bn3.weight:[278]***layer3.11.se.conv.weight:[1, 1, 5]***layer3.12.conv1.weight:[245, 278, 1, 1]***layer3.12.bn1.weight:[245]***layer3.12.conv2.weight:[119, 245, 3, 3]***layer3.12.bn2.weight:[119]***layer3.12.conv3.weight:[278, 119, 1, 1]***layer3.12.bn3.weight:[278]***layer3.12.se.conv.weight:[1, 1, 5]***layer3.13.conv1.weight:[255, 278, 1, 1]***layer3.13.bn1.weight:[255]***layer3.13.conv2.weight:[216, 255, 3, 3]***layer3.13.bn2.weight:[216]***layer3.13.conv3.weight:[278, 216, 1, 1]***layer3.13.bn3.weight:[278]***layer3.13.se.conv.weight:[1, 1, 5]***layer3.14.conv1.weight:[256, 278, 1, 1]***layer3.14.bn1.weight:[256]***layer3.14.conv2.weight:[201, 256, 3, 3]***layer3.14.bn2.weight:[201]***layer3.14.conv3.weight:[278, 201, 1, 1]***layer3.14.bn3.weight:[278]***layer3.14.se.conv.weight:[1, 1, 5]***layer3.15.conv1.weight:[253, 278, 1, 1]***layer3.15.bn1.weight:[253]***layer3.15.conv2.weight:[149, 253, 3, 3]***layer3.15.bn2.weight:[149]***layer3.15.conv3.weight:[278, 149, 1, 1]***layer3.15.bn3.weight:[278]***layer3.15.se.conv.weight:[1, 1, 5]***layer3.16.conv1.weight:[254, 278, 1, 1]***layer3.16.bn1.weight:[254]***layer3.16.conv2.weight:[141, 254, 3, 3]***layer3.16.bn2.weight:[141]***layer3.16.conv3.weight:[278, 141, 1, 1]***layer3.16.bn3.weight:[278]***layer3.16.se.conv.weight:[1, 1, 5]***layer3.17.conv1.weight:[256, 278, 1, 1]***layer3.17.bn1.weight:[256]***layer3.17.conv2.weight:[190, 256, 3, 3]***layer3.17.bn2.weight:[190]***layer3.17.conv3.weight:[278, 190, 1, 1]***layer3.17.bn3.weight:[278]***layer3.17.se.conv.weight:[1, 1, 5]***layer3.18.conv1.weight:[256, 278, 1, 1]***layer3.18.bn1.weight:[256]***layer3.18.conv2.weight:[217, 256, 3, 3]***layer3.18.bn2.weight:[217]***layer3.18.conv3.weight:[278, 217, 1, 1]***layer3.18.bn3.weight:[278]***layer3.18.se.conv.weight:[1, 1, 5]***layer3.19.conv1.weight:[255, 278, 1, 1]***layer3.19.bn1.weight:[255]***layer3.19.conv2.weight:[156, 255, 3, 3]***layer3.19.bn2.weight:[156]***layer3.19.conv3.weight:[278, 156, 1, 1]***layer3.19.bn3.weight:[278]***layer3.19.se.conv.weight:[1, 1, 5]***layer3.20.conv1.weight:[256, 278, 1, 1]***layer3.20.bn1.weight:[256]***layer3.20.conv2.weight:[155, 256, 3, 3]***layer3.20.bn2.weight:[155]***layer3.20.conv3.weight:[278, 155, 1, 1]***layer3.20.bn3.weight:[278]***layer3.20.se.conv.weight:[1, 1, 5]***layer3.21.conv1.weight:[256, 278, 1, 1]***layer3.21.bn1.weight:[256]***layer3.21.conv2.weight:[232, 256, 3, 3]***layer3.21.bn2.weight:[232]***layer3.21.conv3.weight:[278, 232, 1, 1]***layer3.21.bn3.weight:[278]***layer3.21.se.conv.weight:[1, 1, 5]***layer3.22.conv1.weight:[256, 278, 1, 1]***layer3.22.bn1.weight:[256]***layer3.22.conv2.weight:[214, 256, 3, 3]***layer3.22.bn2.weight:[214]***layer3.22.conv3.weight:[278, 214, 1, 1]***layer3.22.bn3.weight:[278]***layer3.22.se.conv.weight:[1, 1, 5]***layer4.0.conv1.weight:[499, 278, 1, 1]***layer4.0.bn1.weight:[499]***layer4.0.conv2.weight:[289, 499, 3, 3]***layer4.0.bn2.weight:[289]***layer4.0.conv3.weight:[2042, 289, 1, 1]***layer4.0.bn3.weight:[2042]***layer4.0.se.conv.weight:[1, 1, 7]***layer4.0.downsample.1.weight:[2042, 278, 1, 1]***layer4.0.downsample.2.weight:[2042]***layer4.1.conv1.weight:[512, 2042, 1, 1]***layer4.1.bn1.weight:[512]***layer4.1.conv2.weight:[512, 512, 3, 3]***layer4.1.bn2.weight:[512]***layer4.1.conv3.weight:[2042, 512, 1, 1]***layer4.1.bn3.weight:[2042]***layer4.1.se.conv.weight:[1, 1, 7]***layer4.2.conv1.weight:[512, 2042, 1, 1]***layer4.2.bn1.weight:[512]***layer4.2.conv2.weight:[502, 512, 3, 3]***layer4.2.bn2.weight:[502]***layer4.2.conv3.weight:[2042, 502, 1, 1]***layer4.2.bn3.weight:[2042]***layer4.2.se.conv.weight:[1, 1, 7]***fc.weight:[1000, 2042]***layer1_2_conv3_M.weight:[256, 26]***layer2_3_conv3_M.weight:[512, 142]***layer3_22_conv3_M.weight:[1024, 278]***layer4_2_conv3_M.weight:[2048, 2042]'
    model = adapt_model_from_string(model, str_model)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model
