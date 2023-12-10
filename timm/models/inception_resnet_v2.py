""" Pytorch Inception-Resnet-V2 implementation
Sourced from https://github.com/Cadene/tensorflow-model-zoo.torch (MIT License) which is
based upon Google's Tensorflow implementation and pretrained weights (Apache 2.0 License)
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import create_classifier, ConvNormAct
from ._builder import build_model_with_cfg
from ._manipulate import flatten_modules
from ._registry import register_model, generate_default_cfgs, register_model_deprecations

__all__ = ['InceptionResnetV2']


class Mixed_5b(nn.Module):
    def __init__(self, conv_block=None):
        super(Mixed_5b, self).__init__()
        conv_block = conv_block or ConvNormAct

        self.branch0 = conv_block(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            conv_block(192, 48, kernel_size=1, stride=1),
            conv_block(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            conv_block(192, 64, kernel_size=1, stride=1),
            conv_block(64, 96, kernel_size=3, stride=1, padding=1),
            conv_block(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            conv_block(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):
    def __init__(self, scale=1.0, conv_block=None):
        super(Block35, self).__init__()
        self.scale = scale
        conv_block = conv_block or ConvNormAct

        self.branch0 = conv_block(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            conv_block(320, 32, kernel_size=1, stride=1),
            conv_block(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            conv_block(320, 32, kernel_size=1, stride=1),
            conv_block(32, 48, kernel_size=3, stride=1, padding=1),
            conv_block(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.act(out)
        return out


class Mixed_6a(nn.Module):
    def __init__(self, conv_block=None):
        super(Mixed_6a, self).__init__()
        conv_block = conv_block or ConvNormAct

        self.branch0 = conv_block(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            conv_block(320, 256, kernel_size=1, stride=1),
            conv_block(256, 256, kernel_size=3, stride=1, padding=1),
            conv_block(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):
    def __init__(self, scale=1.0, conv_block=None):
        super(Block17, self).__init__()
        self.scale = scale
        conv_block = conv_block or ConvNormAct

        self.branch0 = conv_block(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            conv_block(1088, 128, kernel_size=1, stride=1),
            conv_block(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            conv_block(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.act(out)
        return out


class Mixed_7a(nn.Module):
    def __init__(self, conv_block=None):
        super(Mixed_7a, self).__init__()
        conv_block = conv_block or ConvNormAct

        self.branch0 = nn.Sequential(
            conv_block(1088, 256, kernel_size=1, stride=1),
            conv_block(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            conv_block(1088, 256, kernel_size=1, stride=1),
            conv_block(256, 288, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            conv_block(1088, 256, kernel_size=1, stride=1),
            conv_block(256, 288, kernel_size=3, stride=1, padding=1),
            conv_block(288, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, no_relu=False, conv_block=None):
        super(Block8, self).__init__()
        self.scale = scale
        conv_block = conv_block or ConvNormAct

        self.branch0 = conv_block(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            conv_block(2080, 192, kernel_size=1, stride=1),
            conv_block(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            conv_block(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        self.relu = None if no_relu else nn.ReLU()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if self.relu is not None:
            out = self.relu(out)
        return out


class InceptionResnetV2(nn.Module):
    def __init__(
            self,
            num_classes=1000,
            in_chans=3,
            drop_rate=0.,
            output_stride=32,
            global_pool='avg',
            norm_layer='batchnorm2d',
            norm_eps=1e-3,
            act_layer='relu',
    ):
        super(InceptionResnetV2, self).__init__()
        self.num_classes = num_classes
        self.num_features = 1536
        assert output_stride == 32
        conv_block = partial(
            ConvNormAct,
            padding=0,
            norm_layer=norm_layer,
            act_layer=act_layer,
            norm_kwargs=dict(eps=norm_eps),
            act_kwargs=dict(inplace=True),
        )

        self.conv2d_1a = conv_block(in_chans, 32, kernel_size=3, stride=2)
        self.conv2d_2a = conv_block(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = conv_block(32, 64, kernel_size=3, stride=1, padding=1)
        self.feature_info = [dict(num_chs=64, reduction=2, module='conv2d_2b')]

        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = conv_block(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = conv_block(80, 192, kernel_size=3, stride=1)
        self.feature_info += [dict(num_chs=192, reduction=4, module='conv2d_4a')]

        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b(conv_block=conv_block)
        self.repeat = nn.Sequential(*[Block35(scale=0.17, conv_block=conv_block) for _ in range(10)])
        self.feature_info += [dict(num_chs=320, reduction=8, module='repeat')]

        self.mixed_6a = Mixed_6a(conv_block=conv_block)
        self.repeat_1 = nn.Sequential(*[Block17(scale=0.10, conv_block=conv_block) for _ in range(20)])
        self.feature_info += [dict(num_chs=1088, reduction=16, module='repeat_1')]

        self.mixed_7a = Mixed_7a(conv_block=conv_block)
        self.repeat_2 = nn.Sequential(*[Block8(scale=0.20, conv_block=conv_block) for _ in range(9)])

        self.block8 = Block8(no_relu=True, conv_block=conv_block)
        self.conv2d_7b = conv_block(2080, self.num_features, kernel_size=1, stride=1)
        self.feature_info += [dict(num_chs=self.num_features, reduction=32, module='conv2d_7b')]

        self.global_pool, self.head_drop, self.classif = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool, drop_rate=drop_rate)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        module_map = {k: i for i, (k, _) in enumerate(flatten_modules(self.named_children(), prefix=()))}
        module_map.pop(('classif',))

        def _matcher(name):
            if any([name.startswith(n) for n in ('conv2d_1', 'conv2d_2')]):
                return 0
            elif any([name.startswith(n) for n in ('conv2d_3', 'conv2d_4')]):
                return 1
            elif any([name.startswith(n) for n in ('block8', 'conv2d_7')]):
                return len(module_map) + 1
            else:
                for k in module_map.keys():
                    if k == tuple(name.split('.')[:len(k)]):
                        return module_map[k]
                return float('inf')
        return _matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, "checkpointing not supported"

    @torch.jit.ignore
    def get_classifier(self):
        return self.classif

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classif = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.head_drop(x)
        return x if pre_logits else self.classif(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_inception_resnet_v2(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(InceptionResnetV2, variant, pretrained, **kwargs)


default_cfgs = generate_default_cfgs({
    # ported from http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
    'inception_resnet_v2.tf_in1k': {
        'hf_hub_id': 'timm/',
        'num_classes': 1000, 'input_size': (3, 299, 299), 'pool_size': (8, 8),
        'crop_pct': 0.8975, 'interpolation': 'bicubic',
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'conv2d_1a.conv', 'classifier': 'classif',
    },
    # As per https://arxiv.org/abs/1705.07204 and
    # ported from http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    'inception_resnet_v2.tf_ens_adv_in1k': {
        'hf_hub_id': 'timm/',
        'num_classes': 1000, 'input_size': (3, 299, 299), 'pool_size': (8, 8),
        'crop_pct': 0.8975, 'interpolation': 'bicubic',
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'conv2d_1a.conv', 'classifier': 'classif',
    }
})


@register_model
def inception_resnet_v2(pretrained=False, **kwargs) -> InceptionResnetV2:
    return _create_inception_resnet_v2('inception_resnet_v2', pretrained=pretrained, **kwargs)


register_model_deprecations(__name__, {
    'ens_adv_inception_resnet_v2': 'inception_resnet_v2.tf_ens_adv_in1k',
})