"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg, checkpoint_seq
from .layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, create_attn, get_attn, create_classifier
from .registry import register_model

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
    # ResNet and Wide ResNet
    'resnet18': _cfg(url='https://download.pytorch.org/models/resnet18-5c106cde.pth'),
    'resnet18d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet18d_ra2-48a79e06.pth',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnet34': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth'),
    'resnet34d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34d_ra2-f8dcfcaf.pth',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnet26': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26-9aa10e23.pth',
        interpolation='bicubic'),
    'resnet26d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26d-69e92c46.pth',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnet26t': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/resnet26t_256_ra2-6f6fa748.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.94),
    'resnet50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnet50d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnet50t': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnet101': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnet101d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet101d_ra2-2803ffab.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=1.0, test_input_size=(3, 320, 320)),
    'resnet152': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a1h-dc400468.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnet152d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet152d_ra2-5cac0439.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=1.0, test_input_size=(3, 320, 320)),
    'resnet200': _cfg(url='', interpolation='bicubic'),
    'resnet200d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=1.0, test_input_size=(3, 320, 320)),
    'tv_resnet34': _cfg(url='https://download.pytorch.org/models/resnet34-333f7ec4.pth'),
    'tv_resnet50': _cfg(url='https://download.pytorch.org/models/resnet50-19c8e357.pth'),
    'tv_resnet101': _cfg(url='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'),
    'tv_resnet152': _cfg(url='https://download.pytorch.org/models/resnet152-b121ed2d.pth'),
    'wide_resnet50_2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/wide_resnet50_racm-8234f177.pth',
        interpolation='bicubic'),
    'wide_resnet101_2': _cfg(url='https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'),

    # ResNets w/ alternative norm layers
    'resnet50_gn': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_gn_a1h2-8fe6c4d0.pth',
        crop_pct=0.94, interpolation='bicubic'),

    # ResNeXt
    'resnext50_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a1h-0146ab0a.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnext50d_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50d_32x4d-103e99f8.pth',
        interpolation='bicubic',
        first_conv='conv1.0'),
    'resnext101_32x4d': _cfg(url=''),
    'resnext101_32x8d': _cfg(url='https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'),
    'resnext101_64x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnext101_64x4d_c-0d0e0cc0.pth',
        interpolation='bicubic', crop_pct=1.0,  test_input_size=(3, 288, 288)),
    'tv_resnext50_32x4d': _cfg(url='https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'),

    #  ResNeXt models - Weakly Supervised Pretraining on Instagram Hashtags
    #  from https://github.com/facebookresearch/WSL-Images
    #  Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
    'ig_resnext101_32x8d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth'),
    'ig_resnext101_32x16d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth'),
    'ig_resnext101_32x32d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth'),
    'ig_resnext101_32x48d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth'),

    #  Semi-Supervised ResNe*t models from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
    #  Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
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

    #  Semi-Weakly Supervised ResNe*t models from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
    #  Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
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

    #  Squeeze-Excitation ResNets, to eventually replace the models in senet.py
    'seresnet18': _cfg(
        url='',
        interpolation='bicubic'),
    'seresnet34': _cfg(
        url='',
        interpolation='bicubic'),
    'seresnet50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet50_ra_224-8efdb4bb.pth',
        interpolation='bicubic'),
    'seresnet50t': _cfg(
        url='',
        interpolation='bicubic',
        first_conv='conv1.0'),
    'seresnet101': _cfg(
        url='',
        interpolation='bicubic'),
    'seresnet152': _cfg(
        url='',
        interpolation='bicubic'),
    'seresnet152d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet152d_ra2-04464dd2.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=1.0, test_input_size=(3, 320, 320)
    ),
    'seresnet200d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), crop_pct=0.94, pool_size=(8, 8)),
    'seresnet269d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), crop_pct=0.94, pool_size=(8, 8)),


    #  Squeeze-Excitation ResNeXts, to eventually replace the models in senet.py
    'seresnext26d_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26d_32x4d-80fa48a3.pth',
        interpolation='bicubic',
        first_conv='conv1.0'),
    'seresnext26t_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26tn_32x4d-569cb627.pth',
        interpolation='bicubic',
        first_conv='conv1.0'),
    'seresnext50_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext50_32x4d_racm-a304a460.pth',
        interpolation='bicubic'),
    'seresnext101_32x4d': _cfg(
        url='',
        interpolation='bicubic'),
    'seresnext101_32x8d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/seresnext101_32x8d_ah-e6bc4c0a.pth',
        interpolation='bicubic', test_input_size=(3, 288, 288), crop_pct=1.0),
    'senet154': _cfg(
        url='',
        interpolation='bicubic',
        first_conv='conv1.0'),

    # Efficient Channel Attention ResNets
    'ecaresnet26t': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecaresnet26t_ra2-46609757.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.95, test_input_size=(3, 320, 320)),
    'ecaresnetlight': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45402/outputs/ECAResNetLight_4f34b35b.pth',
        interpolation='bicubic'),
    'ecaresnet50d': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45402/outputs/ECAResNet50D_833caf58.pth',
        interpolation='bicubic',
        first_conv='conv1.0'),
    'ecaresnet50d_pruned': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45899/outputs/ECAResNet50D_P_9c67f710.pth',
        interpolation='bicubic',
        first_conv='conv1.0'),
    'ecaresnet50t': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecaresnet50t_ra2-f7ac63c4.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.95, test_input_size=(3, 320, 320)),
    'ecaresnet101d': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45402/outputs/ECAResNet101D_281c5844.pth',
        interpolation='bicubic', first_conv='conv1.0'),
    'ecaresnet101d_pruned': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45610/outputs/ECAResNet101D_P_75a3370e.pth',
        interpolation='bicubic',
        first_conv='conv1.0'),
    'ecaresnet200d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), crop_pct=0.94, pool_size=(8, 8)),
    'ecaresnet269d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecaresnet269d_320_ra2-7baa55cb.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 320, 320), pool_size=(10, 10),
        crop_pct=1.0, test_input_size=(3, 352, 352)),

    # Efficient Channel Attention ResNeXts
    'ecaresnext26t_32x4d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),
    'ecaresnext50t_32x4d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),

    # ResNets with anti-aliasing blur pool
    'resnetblur18': _cfg(
        interpolation='bicubic'),
    'resnetblur50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnetblur50-84f4748f.pth',
        interpolation='bicubic'),
    'resnetblur50d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetblur101d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetaa50d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetaa101d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),
    'seresnetaa50d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),

    # ResNet-RS models
    'resnetrs50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs50_ema-6b53758b.pth',
        input_size=(3, 160, 160), pool_size=(5, 5), crop_pct=0.91, test_input_size=(3, 224, 224),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs101': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs101_i192_ema-1509bbf6.pth',
        input_size=(3, 192, 192), pool_size=(6, 6), crop_pct=0.94, test_input_size=(3, 288, 288),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs152': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs152_i256_ema-a9aff7f9.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 320, 320),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs200': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnetrs200_c-6b698b88.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 320, 320),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs270': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs270_ema-b40e674c.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 352, 352),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs350': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs350_i256_ema-5a1aa8f1.pth',
        input_size=(3, 288, 288), pool_size=(9, 9), crop_pct=1.0, test_input_size=(3, 384, 384),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs420': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs420_ema-972dee69.pth',
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=1.0, test_input_size=(3, 416, 416),
        interpolation='bicubic', first_conv='conv1.0'),
}


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    return aa_layer(stride) if issubclass(aa_layer, nn.AvgPool2d) else aa_layer(channels=channels, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
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


def drop_blocks(drop_prob=0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]


def make_blocks(
        block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


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
    block : Block, class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int, number of layers in each block
    num_classes : int, default 1000, number of classification classes.
    in_chans : int, default 3, number of input (color) channels.
    output_stride : int, default 32, output stride of the network, 32, 16, or 8.
    global_pool : str, Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    cardinality : int, default 1, number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64, factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    stem_width : int, default 64, number of channels in stem convolutions
    stem_type : str, default ''
        The type of stem:
          * '', default - a single 7x7 conv with a width of stem_width
          * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
          * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
    block_reduce_first : int, default 1
        Reduction factor for first convolution output width of residual blocks, 1 for all archs except senets, where 2
    down_kernel_size : int, default 1, kernel size of residual block downsample path, 1x1 for most, 3x3 for senets
    avg_down : bool, default False, use average pooling for projection skip connection between stages/downsample.
    act_layer : nn.Module, activation layer
    norm_layer : nn.Module, normalization layer
    aa_layer : nn.Module, anti-aliasing layer
    drop_rate : float, default 0. Dropout probability before classifier, for training
    """

    def __init__(
            self, block, layers, num_classes=1000, in_chans=3, output_stride=32, global_pool='avg',
            cardinality=1, base_width=64, stem_width=64, stem_type='', replace_stem_pool=False, block_reduce_first=1,
            down_kernel_size=1, avg_down=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None,
            drop_rate=0.0, drop_path_rate=0., drop_block_rate=0., zero_init_last=True, block_args=None):
        super(ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [
                nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
                norm_layer(inplanes),
                act_layer(inplace=True)
            ]))
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(*[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2)])
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


@register_model
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return _create_resnet('resnet18', pretrained, **model_args)


@register_model
def resnet18d(pretrained=False, **kwargs):
    """Constructs a ResNet-18-D model.
    """
    model_args = dict(
        block=BasicBlock, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnet18d', pretrained, **model_args)


@register_model
def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet('resnet34', pretrained, **model_args)


@register_model
def resnet34d(pretrained=False, **kwargs):
    """Constructs a ResNet-34-D model.
    """
    model_args = dict(
        block=BasicBlock, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnet34d', pretrained, **model_args)


@register_model
def resnet26(pretrained=False, **kwargs):
    """Constructs a ResNet-26 model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2], **kwargs)
    return _create_resnet('resnet26', pretrained, **model_args)


@register_model
def resnet26t(pretrained=False, **kwargs):
    """Constructs a ResNet-26-T model.
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep_tiered', avg_down=True, **kwargs)
    return _create_resnet('resnet26t', pretrained, **model_args)


@register_model
def resnet26d(pretrained=False, **kwargs):
    """Constructs a ResNet-26-D model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnet26d', pretrained, **model_args)


@register_model
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('resnet50', pretrained, **model_args)


@register_model
def resnet50d(pretrained=False, **kwargs):
    """Constructs a ResNet-50-D model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnet50d', pretrained, **model_args)


@register_model
def resnet50t(pretrained=False, **kwargs):
    """Constructs a ResNet-50-T model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep_tiered', avg_down=True, **kwargs)
    return _create_resnet('resnet50t', pretrained, **model_args)


@register_model
def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return _create_resnet('resnet101', pretrained, **model_args)


@register_model
def resnet101d(pretrained=False, **kwargs):
    """Constructs a ResNet-101-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnet101d', pretrained, **model_args)


@register_model
def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)
    return _create_resnet('resnet152', pretrained, **model_args)


@register_model
def resnet152d(pretrained=False, **kwargs):
    """Constructs a ResNet-152-D model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnet152d', pretrained, **model_args)


@register_model
def resnet200(pretrained=False, **kwargs):
    """Constructs a ResNet-200 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3], **kwargs)
    return _create_resnet('resnet200', pretrained, **model_args)


@register_model
def resnet200d(pretrained=False, **kwargs):
    """Constructs a ResNet-200-D model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnet200d', pretrained, **model_args)


@register_model
def tv_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model with original Torchvision weights.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet('tv_resnet34', pretrained, **model_args)


@register_model
def tv_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model with original Torchvision weights.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('tv_resnet50', pretrained, **model_args)


@register_model
def tv_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model w/ Torchvision pretrained weights.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return _create_resnet('tv_resnet101', pretrained, **model_args)


@register_model
def tv_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model w/ Torchvision pretrained weights.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)
    return _create_resnet('tv_resnet152', pretrained, **model_args)


@register_model
def wide_resnet50_2(pretrained=False, **kwargs):
    """Constructs a Wide ResNet-50-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], base_width=128, **kwargs)
    return _create_resnet('wide_resnet50_2', pretrained, **model_args)


@register_model
def wide_resnet101_2(pretrained=False, **kwargs):
    """Constructs a Wide ResNet-101-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], base_width=128, **kwargs)
    return _create_resnet('wide_resnet101_2', pretrained, **model_args)


@register_model
def resnet50_gn(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model w/ GroupNorm
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('resnet50_gn', pretrained, norm_layer=GroupNorm, **model_args)


@register_model
def resnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt50-32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4, **kwargs)
    return _create_resnet('resnext50_32x4d', pretrained, **model_args)


@register_model
def resnext50d_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt50d-32x4d model. ResNext50 w/ deep stem & avg pool downsample
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3],  cardinality=32, base_width=4,
        stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnext50d_32x4d', pretrained, **model_args)


@register_model
def resnext101_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4, **kwargs)
    return _create_resnet('resnext101_32x4d', pretrained, **model_args)


@register_model
def resnext101_32x8d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8, **kwargs)
    return _create_resnet('resnext101_32x8d', pretrained, **model_args)


@register_model
def resnext101_64x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt101-64x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64, base_width=4, **kwargs)
    return _create_resnet('resnext101_64x4d', pretrained, **model_args)


@register_model
def tv_resnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt50-32x4d model with original Torchvision weights.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4, **kwargs)
    return _create_resnet('tv_resnext50_32x4d', pretrained, **model_args)


@register_model
def ig_resnext101_32x8d(pretrained=True, **kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8, **kwargs)
    return _create_resnet('ig_resnext101_32x8d', pretrained, **model_args)


@register_model
def ig_resnext101_32x16d(pretrained=True, **kwargs):
    """Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=16, **kwargs)
    return _create_resnet('ig_resnext101_32x16d', pretrained, **model_args)


@register_model
def ig_resnext101_32x32d(pretrained=True, **kwargs):
    """Constructs a ResNeXt-101 32x32 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=32, **kwargs)
    return _create_resnet('ig_resnext101_32x32d', pretrained, **model_args)


@register_model
def ig_resnext101_32x48d(pretrained=True, **kwargs):
    """Constructs a ResNeXt-101 32x48 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=48, **kwargs)
    return _create_resnet('ig_resnext101_32x48d', pretrained, **model_args)


@register_model
def ssl_resnet18(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNet-18 model pre-trained on YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return _create_resnet('ssl_resnet18', pretrained, **model_args)


@register_model
def ssl_resnet50(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNet-50 model pre-trained on YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('ssl_resnet50', pretrained, **model_args)


@register_model
def ssl_resnext50_32x4d(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-50 32x4 model pre-trained on YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4, **kwargs)
    return _create_resnet('ssl_resnext50_32x4d', pretrained, **model_args)


@register_model
def ssl_resnext101_32x4d(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-101 32x4 model pre-trained on YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4, **kwargs)
    return _create_resnet('ssl_resnext101_32x4d', pretrained, **model_args)


@register_model
def ssl_resnext101_32x8d(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-101 32x8 model pre-trained on YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8, **kwargs)
    return _create_resnet('ssl_resnext101_32x8d', pretrained, **model_args)


@register_model
def ssl_resnext101_32x16d(pretrained=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-101 32x16 model pre-trained on YFCC100M dataset and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=16, **kwargs)
    return _create_resnet('ssl_resnext101_32x16d', pretrained, **model_args)


@register_model
def swsl_resnet18(pretrained=True, **kwargs):
    """Constructs a semi-weakly supervised Resnet-18 model pre-trained on 1B weakly supervised
       image dataset and finetuned on ImageNet.
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
       Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return _create_resnet('swsl_resnet18', pretrained, **model_args)


@register_model
def swsl_resnet50(pretrained=True, **kwargs):
    """Constructs a semi-weakly supervised ResNet-50 model pre-trained on 1B weakly supervised
       image dataset and finetuned on ImageNet.
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
       Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('swsl_resnet50', pretrained, **model_args)


@register_model
def swsl_resnext50_32x4d(pretrained=True, **kwargs):
    """Constructs a semi-weakly supervised ResNeXt-50 32x4 model pre-trained on 1B weakly supervised
       image dataset and finetuned on ImageNet.
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
       Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4, **kwargs)
    return _create_resnet('swsl_resnext50_32x4d', pretrained, **model_args)


@register_model
def swsl_resnext101_32x4d(pretrained=True, **kwargs):
    """Constructs a semi-weakly supervised ResNeXt-101 32x4 model pre-trained on 1B weakly supervised
       image dataset and finetuned on ImageNet.
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
       Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4, **kwargs)
    return _create_resnet('swsl_resnext101_32x4d', pretrained, **model_args)


@register_model
def swsl_resnext101_32x8d(pretrained=True, **kwargs):
    """Constructs a semi-weakly supervised ResNeXt-101 32x8 model pre-trained on 1B weakly supervised
       image dataset and finetuned on ImageNet.
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
       Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8, **kwargs)
    return _create_resnet('swsl_resnext101_32x8d', pretrained, **model_args)


@register_model
def swsl_resnext101_32x16d(pretrained=True, **kwargs):
    """Constructs a semi-weakly supervised ResNeXt-101 32x16 model pre-trained on 1B weakly supervised
       image dataset and finetuned on ImageNet.
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
       Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=16, **kwargs)
    return _create_resnet('swsl_resnext101_32x16d', pretrained, **model_args)


@register_model
def ecaresnet26t(pretrained=False, **kwargs):
    """Constructs an ECA-ResNeXt-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem and ECA attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnet26t', pretrained, **model_args)


@register_model
def ecaresnet50d(pretrained=False, **kwargs):
    """Constructs a ResNet-50-D model with eca.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnet50d', pretrained, **model_args)


@register_model
def resnetrs50(pretrained=False, **kwargs):
    """Constructs a ResNet-RS-50 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer), **kwargs)
    return _create_resnet('resnetrs50', pretrained, **model_args)


@register_model
def resnetrs101(pretrained=False, **kwargs):
    """Constructs a ResNet-RS-101 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer), **kwargs)
    return _create_resnet('resnetrs101', pretrained, **model_args)


@register_model
def resnetrs152(pretrained=False, **kwargs):
    """Constructs a ResNet-RS-152 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer), **kwargs)
    return _create_resnet('resnetrs152', pretrained, **model_args)


@register_model
def resnetrs200(pretrained=False, **kwargs):
    """Constructs a ResNet-RS-200 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer), **kwargs)
    return _create_resnet('resnetrs200', pretrained, **model_args)


@register_model
def resnetrs270(pretrained=False, **kwargs):
    """Constructs a ResNet-RS-270 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 29, 53, 4], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer), **kwargs)
    return _create_resnet('resnetrs270', pretrained, **model_args)



@register_model
def resnetrs350(pretrained=False, **kwargs):
    """Constructs a ResNet-RS-350 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 36, 72, 4], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer), **kwargs)
    return _create_resnet('resnetrs350', pretrained, **model_args)


@register_model
def resnetrs420(pretrained=False, **kwargs):
    """Constructs a ResNet-RS-420 model
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 44, 87, 4], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer), **kwargs)
    return _create_resnet('resnetrs420', pretrained, **model_args)


@register_model
def ecaresnet50d_pruned(pretrained=False, **kwargs):
    """Constructs a ResNet-50-D model pruned with eca.
        The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnet50d_pruned', pretrained, pruned=True, **model_args)


@register_model
def ecaresnet50t(pretrained=False, **kwargs):
    """Constructs an ECA-ResNet-50-T model.
    Like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels in the deep stem and ECA attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnet50t', pretrained, **model_args)


@register_model
def ecaresnetlight(pretrained=False, **kwargs):
    """Constructs a ResNet-50-D light model with eca.
    """
    model_args = dict(
        block=Bottleneck, layers=[1, 1, 11, 3], stem_width=32, avg_down=True,
        block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnetlight', pretrained, **model_args)


@register_model
def ecaresnet101d(pretrained=False, **kwargs):
    """Constructs a ResNet-101-D model with eca.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnet101d', pretrained, **model_args)


@register_model
def ecaresnet101d_pruned(pretrained=False, **kwargs):
    """Constructs a ResNet-101-D model pruned with eca.
       The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnet101d_pruned', pretrained, pruned=True, **model_args)


@register_model
def ecaresnet200d(pretrained=False, **kwargs):
    """Constructs a ResNet-200-D model with ECA.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnet200d', pretrained, **model_args)


@register_model
def ecaresnet269d(pretrained=False, **kwargs):
    """Constructs a ResNet-269-D model with ECA.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 30, 48, 8], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnet269d', pretrained, **model_args)


@register_model
def ecaresnext26t_32x4d(pretrained=False, **kwargs):
    """Constructs an ECA-ResNeXt-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem. This model replaces SE module with the ECA module
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnext26t_32x4d', pretrained, **model_args)


@register_model
def ecaresnext50t_32x4d(pretrained=False, **kwargs):
    """Constructs an ECA-ResNeXt-50-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem. This model replaces SE module with the ECA module
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'), **kwargs)
    return _create_resnet('ecaresnext50t_32x4d', pretrained, **model_args)


@register_model
def resnetblur18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model with blur anti-aliasing
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], aa_layer=BlurPool2d, **kwargs)
    return _create_resnet('resnetblur18', pretrained, **model_args)


@register_model
def resnetblur50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model with blur anti-aliasing
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=BlurPool2d, **kwargs)
    return _create_resnet('resnetblur50', pretrained, **model_args)


@register_model
def resnetblur50d(pretrained=False, **kwargs):
    """Constructs a ResNet-50-D model with blur anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=BlurPool2d,
        stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnetblur50d', pretrained, **model_args)


@register_model
def resnetblur101d(pretrained=False, **kwargs):
    """Constructs a ResNet-101-D model with blur anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], aa_layer=BlurPool2d,
        stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnetblur101d', pretrained, **model_args)


@register_model
def resnetaa50d(pretrained=False, **kwargs):
    """Constructs a ResNet-50-D model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=nn.AvgPool2d,
        stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnetaa50d', pretrained, **model_args)


@register_model
def resnetaa101d(pretrained=False, **kwargs):
    """Constructs a ResNet-101-D model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], aa_layer=nn.AvgPool2d,
        stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('resnetaa101d', pretrained, **model_args)


@register_model
def seresnetaa50d(pretrained=False, **kwargs):
    """Constructs a SE=ResNet-50-D model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=nn.AvgPool2d,
        stem_width=32, stem_type='deep', avg_down=True, block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnetaa50d', pretrained, **model_args)


@register_model
def seresnet18(pretrained=False, **kwargs):
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet18', pretrained, **model_args)


@register_model
def seresnet34(pretrained=False, **kwargs):
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet34', pretrained, **model_args)


@register_model
def seresnet50(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet50', pretrained, **model_args)


@register_model
def seresnet50t(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3],  stem_width=32, stem_type='deep_tiered', avg_down=True,
        block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet50t', pretrained, **model_args)


@register_model
def seresnet101(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet101', pretrained, **model_args)


@register_model
def seresnet152(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet152', pretrained, **model_args)


@register_model
def seresnet152d(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet152d', pretrained, **model_args)


@register_model
def seresnet200d(pretrained=False, **kwargs):
    """Constructs a ResNet-200-D model with SE attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet200d', pretrained, **model_args)


@register_model
def seresnet269d(pretrained=False, **kwargs):
    """Constructs a ResNet-269-D model with SE attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 30, 48, 8], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnet269d', pretrained, **model_args)


@register_model
def seresnext26d_32x4d(pretrained=False, **kwargs):
    """Constructs a SE-ResNeXt-26-D model.`
    This is technically a 28 layer ResNet, using the 'D' modifier from Gluon / bag-of-tricks for
    combination of deep stem and avg_pool in downsample.
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep', avg_down=True, block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnext26d_32x4d', pretrained, **model_args)


@register_model
def seresnext26t_32x4d(pretrained=False, **kwargs):
    """Constructs a SE-ResNet-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem.
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnext26t_32x4d', pretrained, **model_args)


@register_model
def seresnext26tn_32x4d(pretrained=False, **kwargs):
    """Constructs a SE-ResNeXt-26-T model.
    NOTE I deprecated previous 't' model defs and replaced 't' with 'tn', this was the only tn model of note
    so keeping this def for backwards compat with any uses out there. Old 't' model is lost.
    """
    return seresnext26t_32x4d(pretrained=pretrained, **kwargs)


@register_model
def seresnext50_32x4d(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4,
        block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnext50_32x4d', pretrained, **model_args)


@register_model
def seresnext101_32x4d(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4,
        block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnext101_32x4d', pretrained, **model_args)


@register_model
def seresnext101_32x8d(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8,
        block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('seresnext101_32x8d', pretrained, **model_args)


@register_model
def senet154(pretrained=False, **kwargs):
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], cardinality=64, base_width=4, stem_type='deep',
        down_kernel_size=3, block_reduce_first=2, block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet('senet154', pretrained, **model_args)
