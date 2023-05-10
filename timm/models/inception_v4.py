""" Pytorch Inception-V4 implementation
Sourced from https://github.com/Cadene/tensorflow-model-zoo.torch (MIT License) which is
based upon Google's Tensorflow implementation and pretrained weights (Apache 2.0 License)
"""
from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import create_classifier, ConvNormAct
from ._builder import build_model_with_cfg
from ._registry import register_model, generate_default_cfgs

__all__ = ['InceptionV4']


class Mixed3a(nn.Module):
    def __init__(self, conv_block=ConvNormAct):
        super(Mixed3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = conv_block(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed4a(nn.Module):
    def __init__(self, conv_block=ConvNormAct):
        super(Mixed4a, self).__init__()

        self.branch0 = nn.Sequential(
            conv_block(160, 64, kernel_size=1, stride=1),
            conv_block(64, 96, kernel_size=3, stride=1)
        )

        self.branch1 = nn.Sequential(
            conv_block(160, 64, kernel_size=1, stride=1),
            conv_block(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            conv_block(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            conv_block(64, 96, kernel_size=(3, 3), stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed5a(nn.Module):
    def __init__(self, conv_block=ConvNormAct):
        super(Mixed5a, self).__init__()
        self.conv = conv_block(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class InceptionA(nn.Module):
    def __init__(self, conv_block=ConvNormAct):
        super(InceptionA, self).__init__()
        self.branch0 = conv_block(384, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            conv_block(384, 64, kernel_size=1, stride=1),
            conv_block(64, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            conv_block(384, 64, kernel_size=1, stride=1),
            conv_block(64, 96, kernel_size=3, stride=1, padding=1),
            conv_block(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            conv_block(384, 96, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class ReductionA(nn.Module):
    def __init__(self, conv_block=ConvNormAct):
        super(ReductionA, self).__init__()
        self.branch0 = conv_block(384, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            conv_block(384, 192, kernel_size=1, stride=1),
            conv_block(192, 224, kernel_size=3, stride=1, padding=1),
            conv_block(224, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class InceptionB(nn.Module):
    def __init__(self, conv_block=ConvNormAct):
        super(InceptionB, self).__init__()
        self.branch0 = conv_block(1024, 384, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            conv_block(1024, 192, kernel_size=1, stride=1),
            conv_block(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            conv_block(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        self.branch2 = nn.Sequential(
            conv_block(1024, 192, kernel_size=1, stride=1),
            conv_block(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            conv_block(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            conv_block(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            conv_block(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            conv_block(1024, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class ReductionB(nn.Module):
    def __init__(self, conv_block=ConvNormAct):
        super(ReductionB, self).__init__()

        self.branch0 = nn.Sequential(
            conv_block(1024, 192, kernel_size=1, stride=1),
            conv_block(192, 192, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            conv_block(1024, 256, kernel_size=1, stride=1),
            conv_block(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            conv_block(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            conv_block(320, 320, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class InceptionC(nn.Module):
    def __init__(self, conv_block=ConvNormAct):
        super(InceptionC, self).__init__()

        self.branch0 = conv_block(1536, 256, kernel_size=1, stride=1)

        self.branch1_0 = conv_block(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = conv_block(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_1b = conv_block(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch2_0 = conv_block(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = conv_block(384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_2 = conv_block(448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3a = conv_block(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3b = conv_block(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            conv_block(1536, 256, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):
    def __init__(
            self,
            num_classes=1000,
            in_chans=3,
            output_stride=32,
            drop_rate=0.,
            global_pool='avg',
            norm_layer='batchnorm2d',
            norm_eps=1e-3,
            act_layer='relu',
    ):
        super(InceptionV4, self).__init__()
        assert output_stride == 32
        self.num_classes = num_classes
        self.num_features = 1536
        conv_block = partial(
            ConvNormAct,
            padding=0,
            norm_layer=norm_layer,
            act_layer=act_layer,
            norm_kwargs=dict(eps=norm_eps),
            act_kwargs=dict(inplace=True),
        )

        features = [
            conv_block(in_chans, 32, kernel_size=3, stride=2),
            conv_block(32, 32, kernel_size=3, stride=1),
            conv_block(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed3a(conv_block),
            Mixed4a(conv_block),
            Mixed5a(conv_block),
        ]
        features += [InceptionA(conv_block) for _ in range(4)]
        features += [ReductionA(conv_block)]  # Mixed6a
        features += [InceptionB(conv_block) for _ in range(7)]
        features += [ReductionB(conv_block)]  # Mixed7a
        features += [InceptionC(conv_block) for _ in range(3)]
        self.features = nn.Sequential(*features)
        self.feature_info = [
            dict(num_chs=64, reduction=2, module='features.2'),
            dict(num_chs=160, reduction=4, module='features.3'),
            dict(num_chs=384, reduction=8, module='features.9'),
            dict(num_chs=1024, reduction=16, module='features.17'),
            dict(num_chs=1536, reduction=32, module='features.21'),
        ]
        self.global_pool, self.head_drop, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool, drop_rate=drop_rate)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^features\.[012]\.',
            blocks=r'^features\.(\d+)'
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        return self.features(x)

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.head_drop(x)
        return x if pre_logits else self.last_linear(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_inception_v4(variant, pretrained=False, **kwargs) -> InceptionV4:
    return build_model_with_cfg(
        InceptionV4,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True),
        **kwargs,
    )


default_cfgs = generate_default_cfgs({
    'inception_v4.tf_in1k': {
        'hf_hub_id': 'timm/',
        'num_classes': 1000, 'input_size': (3, 299, 299), 'pool_size': (8, 8),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'features.0.conv', 'classifier': 'last_linear',
    }
})


@register_model
def inception_v4(pretrained=False, **kwargs):
    return _create_inception_v4('inception_v4', pretrained, **kwargs)
