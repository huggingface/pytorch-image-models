"""Pytorch impl of MxNet Gluon ResNet/(SE)ResNeXt variants
This file evolved from https://github.com/pytorch/vision 'resnet.py' with (SE)-ResNeXt additions
and ports of Gluon variations (https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/resnet.py) 
by Ross Wightman
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .helpers import load_pretrained
from .adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


__all__ = ['GluonResNet']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'gluon_resnet18_v1b': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet18_v1b-0757602b.pth'),
    'gluon_resnet34_v1b': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet34_v1b-c6d82d59.pth'),
    'gluon_resnet50_v1b': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1b-0ebe02e2.pth'),
    'gluon_resnet101_v1b': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1b-3b017079.pth'),
    'gluon_resnet152_v1b': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1b-c1edb0dd.pth'),
    'gluon_resnet50_v1c': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1c-48092f55.pth'),
    'gluon_resnet101_v1c': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1c-1f26822a.pth'),
    'gluon_resnet152_v1c': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1c-a3bb0b98.pth'),
    'gluon_resnet50_v1d': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1d-818a1b1b.pth'),
    'gluon_resnet101_v1d': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1d-0f9c8644.pth'),
    'gluon_resnet152_v1d': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1d-bd354e12.pth'),
    'gluon_resnet50_v1e': _cfg(url=''),
    'gluon_resnet101_v1e': _cfg(url=''),
    'gluon_resnet152_v1e': _cfg(url=''),
    'gluon_resnet50_v1s': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1s-1762acc0.pth'),
    'gluon_resnet101_v1s': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1s-60fe0cc1.pth'),
    'gluon_resnet152_v1s': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1s-dcc41b81.pth'),
    'gluon_resnext50_32x4d': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext50_32x4d-e6a097c1.pth'),
    'gluon_resnext101_32x4d': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext101_32x4d-b253c8c4.pth'),
    'gluon_resnext101_64x4d': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext101_64x4d-f9a8e184.pth'),
    'gluon_seresnext50_32x4d': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_seresnext50_32x4d-90cf2d6e.pth'),
    'gluon_seresnext101_32x4d': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_seresnext101_32x4d-cf52900d.pth'),
    'gluon_seresnext101_64x4d': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_seresnext101_64x4d-f9926f93.pth'),
    'gluon_senet154': _cfg(url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_senet154-70a1a3c0.pth'),
}


def _get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class SEModule(nn.Module):

    def __init__(self, channels, reduction_channels):
        super(SEModule, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(
            reduction_channels, channels, kernel_size=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        #x = self.avg_pool(x)
        x = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BasicBlockGl(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, base_width=64, use_se=False,
                 reduce_first=1, dilation=1, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockGl, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=stride, padding=dilation,
            dilation=dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=previous_dilation,
            dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(outplanes)
        self.se = SEModule(outplanes, planes // 4) if use_se else None
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckGl(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, base_width=64, use_se=False,
                 reduce_first=1, dilation=1, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckGl, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.se = SEModule(outplanes, planes // 4) if use_se else None
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GluonResNet(nn.Module):
    """ Gluon ResNet (https://gluon-cv.mxnet.io/model_zoo/classification.html)
    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet found in the gluon model zoo that
      * have stride in 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    Included ResNet variants are:
      * v1b - 7x7 stem, stem_width=64, same as torchvision ResNet (checkpoint compatible), or NVIDIA ResNet 'v1.5'
      * v1c - 3 layer deep 3x3 stem, stem_width = 32
      * v1d - 3 layer deep 3x3 stem, stem_width = 32, average pool in downsample
      * v1e - 3 layer deep 3x3 stem, stem_width = 64, average pool in downsample  *no pretrained weights available
      * v1s - 3 layer deep 3x3 stem, stem_width = 64

    ResNeXt is standard and checkpoint compatible with torchvision pretrained models. 7x7 stem,
        stem_width = 64, standard cardinality and base width calcs

    SE-ResNeXt is standard. 7x7 stem, stem_width = 64,
        checkpoints are not compatible with Cadene pretrained, but could be with key mapping

    SENet-154 is standard. 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block

    Original ResNet-V1, ResNet-V2 (bn-act-conv), and SE-ResNet (stride in first bottleneck conv) are NOT supported.
    They do have Gluon pretrained weights but are, at best, comparable (or inferior) to the supported models.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int
        Numbers of layers in each block
    num_classes : int, default 1000
        Number of classification classes.
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    block_reduce_first: int, default 1
        Reduction factor for first convolution output width of residual blocks,
        1 for all archs except senets, where 2
    down_kernel_size: int, default 1
        Kernel size of residual block downsampling path, 1x1 for most archs, 3x3 for senets
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    """
    def __init__(self, block, layers, num_classes=1000, in_chans=3, use_se=False,
                 cardinality=1, base_width=64, stem_width=64, deep_stem=False,
                 block_reduce_first=1, down_kernel_size=1, avg_down=False, dilated=False,
                 norm_layer=nn.BatchNorm2d, drop_rate=0.0, global_pool='avg'):
        self.num_classes = num_classes
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.cardinality = cardinality
        self.base_width = base_width
        self.drop_rate = drop_rate
        self.expansion = block.expansion
        self.dilated = dilated
        super(GluonResNet, self).__init__()

        if not deep_stem:
            self.conv1 = nn.Conv2d(in_chans, stem_width, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            conv1_modules = [
                nn.Conv2d(in_chans, stem_width, 3, stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(),
                nn.Conv2d(stem_width, stem_width, 3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(),
                nn.Conv2d(stem_width, self.inplanes, 3, stride=1, padding=1, bias=False),
            ]
            self.conv1 = nn.Sequential(*conv1_modules)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stride_3_4 = 1 if self.dilated else 2
        dilation_3 = 2 if self.dilated else 1
        dilation_4 = 4 if self.dilated else 1
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=1, reduce_first=block_reduce_first,
            use_se=use_se, avg_down=avg_down, down_kernel_size=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, reduce_first=block_reduce_first,
            use_se=use_se, avg_down=avg_down, down_kernel_size=down_kernel_size, norm_layer=norm_layer)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=stride_3_4, dilation=dilation_3, reduce_first=block_reduce_first,
            use_se=use_se, avg_down=avg_down, down_kernel_size=down_kernel_size, norm_layer=norm_layer)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=stride_3_4, dilation=dilation_4, reduce_first=block_reduce_first,
            use_se=use_se, avg_down=avg_down, down_kernel_size=down_kernel_size, norm_layer=norm_layer)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_features = 512 * block.expansion
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, reduce_first=1,
                    use_se=False, avg_down=False, down_kernel_size=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_padding = _get_padding(down_kernel_size, stride)
            if avg_down:
                avg_stride = stride if dilation == 1 else 1
                downsample_layers = [
                    nn.AvgPool2d(avg_stride, avg_stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion, down_kernel_size,
                              stride=1, padding=downsample_padding, bias=False),
                    norm_layer(planes * block.expansion),
                ]
            else:
                downsample_layers = [
                    nn.Conv2d(self.inplanes, planes * block.expansion, down_kernel_size,
                              stride=stride, padding=downsample_padding, bias=False),
                    norm_layer(planes * block.expansion),
                ]
            downsample = nn.Sequential(*downsample_layers)

        first_dilation = 1 if dilation in (1, 2) else 2
        layers = [block(
            self.inplanes, planes, stride, downsample,
            cardinality=self.cardinality, base_width=self.base_width, reduce_first=reduce_first,
            use_se=use_se, dilation=first_dilation, previous_dilation=dilation, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.inplanes, planes,
                cardinality=self.cardinality, base_width=self.base_width, reduce_first=reduce_first,
                use_se=use_se, dilation=dilation, previous_dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        del self.fc
        if num_classes:
            self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)
        else:
            self.fc = None

    def forward_features(self, x, pool=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if pool:
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


@register_model
def gluon_resnet18_v1b(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-18 model.
    """
    default_cfg = default_cfgs['gluon_resnet18_v1b']
    model = GluonResNet(BasicBlockGl, [2, 2, 2, 2], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet34_v1b(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-34 model.
    """
    default_cfg = default_cfgs['gluon_resnet34_v1b']
    model = GluonResNet(BasicBlockGl, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet50_v1b(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-50 model.
    """
    default_cfg = default_cfgs['gluon_resnet50_v1b']
    model = GluonResNet(BottleneckGl, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet101_v1b(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-101 model.
    """
    default_cfg = default_cfgs['gluon_resnet101_v1b']
    model = GluonResNet(BottleneckGl, [3, 4, 23, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet152_v1b(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-152 model.
    """
    default_cfg = default_cfgs['gluon_resnet152_v1b']
    model = GluonResNet(BottleneckGl, [3, 8, 36, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet50_v1c(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-50 model.
    """
    default_cfg = default_cfgs['gluon_resnet50_v1c']
    model = GluonResNet(BottleneckGl, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans,
                        stem_width=32, deep_stem=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet101_v1c(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-101 model.
    """
    default_cfg = default_cfgs['gluon_resnet101_v1c']
    model = GluonResNet(BottleneckGl, [3, 4, 23, 3], num_classes=num_classes, in_chans=in_chans,
                        stem_width=32, deep_stem=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet152_v1c(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-152 model.
    """
    default_cfg = default_cfgs['gluon_resnet152_v1c']
    model = GluonResNet(BottleneckGl, [3, 8, 36, 3], num_classes=num_classes, in_chans=in_chans,
                        stem_width=32, deep_stem=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet50_v1d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-50 model.
    """
    default_cfg = default_cfgs['gluon_resnet50_v1d']
    model = GluonResNet(BottleneckGl, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans,
                        stem_width=32, deep_stem=True, avg_down=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet101_v1d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-101 model.
    """
    default_cfg = default_cfgs['gluon_resnet101_v1d']
    model = GluonResNet(BottleneckGl, [3, 4, 23, 3], num_classes=num_classes, in_chans=in_chans,
                        stem_width=32, deep_stem=True, avg_down=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet152_v1d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-152 model.
    """
    default_cfg = default_cfgs['gluon_resnet152_v1d']
    model = GluonResNet(BottleneckGl, [3, 8, 36, 3], num_classes=num_classes, in_chans=in_chans,
                        stem_width=32, deep_stem=True, avg_down=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet50_v1e(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-50-V1e model. No pretrained weights for any 'e' variants
    """
    default_cfg = default_cfgs['gluon_resnet50_v1e']
    model = GluonResNet(BottleneckGl, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans,
                        stem_width=64, deep_stem=True, avg_down=True, **kwargs)
    model.default_cfg = default_cfg
    #if pretrained:
    #    load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet101_v1e(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-101 model.
    """
    default_cfg = default_cfgs['gluon_resnet101_v1e']
    model = GluonResNet(BottleneckGl, [3, 4, 23, 3], num_classes=num_classes, in_chans=in_chans,
                        stem_width=64, deep_stem=True, avg_down=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet152_v1e(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-152 model.
    """
    default_cfg = default_cfgs['gluon_resnet152_v1e']
    model = GluonResNet(BottleneckGl, [3, 8, 36, 3], num_classes=num_classes, in_chans=in_chans,
                        stem_width=64, deep_stem=True, avg_down=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet50_v1s(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-50 model.
    """
    default_cfg = default_cfgs['gluon_resnet50_v1s']
    model = GluonResNet(BottleneckGl, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans,
                        stem_width=64, deep_stem=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet101_v1s(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-101 model.
    """
    default_cfg = default_cfgs['gluon_resnet101_v1s']
    model = GluonResNet(BottleneckGl, [3, 4, 23, 3], num_classes=num_classes, in_chans=in_chans,
                        stem_width=64, deep_stem=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnet152_v1s(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-152 model.
    """
    default_cfg = default_cfgs['gluon_resnet152_v1s']
    model = GluonResNet(BottleneckGl, [3, 8, 36, 3], num_classes=num_classes, in_chans=in_chans,
                        stem_width=64, deep_stem=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnext50_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNeXt50-32x4d model.
    """
    default_cfg = default_cfgs['gluon_resnext50_32x4d']
    model = GluonResNet(
        BottleneckGl, [3, 4, 6, 3], cardinality=32, base_width=4,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnext101_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNeXt-101 model.
    """
    default_cfg = default_cfgs['gluon_resnext101_32x4d']
    model = GluonResNet(
        BottleneckGl, [3, 4, 23, 3], cardinality=32, base_width=4,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_resnext101_64x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNeXt-101 model.
    """
    default_cfg = default_cfgs['gluon_resnext101_64x4d']
    model = GluonResNet(
        BottleneckGl, [3, 4, 23, 3], cardinality=64, base_width=4,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_seresnext50_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a SEResNeXt50-32x4d model.
    """
    default_cfg = default_cfgs['gluon_seresnext50_32x4d']
    model = GluonResNet(
        BottleneckGl, [3, 4, 6, 3], cardinality=32, base_width=4, use_se=True,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_seresnext101_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a SEResNeXt-101-32x4d model.
    """
    default_cfg = default_cfgs['gluon_seresnext101_32x4d']
    model = GluonResNet(
        BottleneckGl, [3, 4, 23, 3], cardinality=32, base_width=4, use_se=True,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_seresnext101_64x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a SEResNeXt-101-64x4d model.
    """
    default_cfg = default_cfgs['gluon_seresnext101_64x4d']
    model = GluonResNet(
        BottleneckGl, [3, 4, 23, 3], cardinality=64, base_width=4, use_se=True,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


@register_model
def gluon_senet154(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs an SENet-154 model.
    """
    default_cfg = default_cfgs['gluon_senet154']
    model = GluonResNet(
        BottleneckGl, [3, 8, 36, 3], cardinality=64, base_width=4, use_se=True,
        deep_stem=True, down_kernel_size=3, block_reduce_first=2,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model

