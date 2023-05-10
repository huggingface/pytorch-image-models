""" Inception-V3

Originally from torchvision Inception3 model
Licensed BSD-Clause 3 https://github.com/pytorch/vision/blob/master/LICENSE
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import trunc_normal_, create_classifier, Linear, ConvNormAct
from ._builder import build_model_with_cfg
from ._builder import resolve_pretrained_cfg
from ._manipulate import flatten_modules
from ._registry import register_model, generate_default_cfgs, register_model_deprecations

__all__ = ['InceptionV3']  # model_registry will add each entrypoint fn to this


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        conv_block = conv_block or ConvNormAct
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        conv_block = conv_block or ConvNormAct
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        conv_block = conv_block or ConvNormAct
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        conv_block = conv_block or ConvNormAct
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        conv_block = conv_block or ConvNormAct
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        conv_block = conv_block or ConvNormAct
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class InceptionV3(nn.Module):
    """Inception-V3
    """
    aux_logits: torch.jit.Final[bool]

    def __init__(
            self,
            num_classes=1000,
            in_chans=3,
            drop_rate=0.,
            global_pool='avg',
            aux_logits=False,
            norm_layer='batchnorm2d',
            norm_eps=1e-3,
            act_layer='relu',
    ):
        super(InceptionV3, self).__init__()
        self.num_classes = num_classes
        self.aux_logits = aux_logits
        conv_block = partial(
            ConvNormAct,
            padding=0,
            norm_layer=norm_layer,
            act_layer=act_layer,
            norm_kwargs=dict(eps=norm_eps),
            act_kwargs=dict(inplace=True),
        )

        self.Conv2d_1a_3x3 = conv_block(in_chans, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.Pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.Pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = InceptionA(192, pool_features=32, conv_block=conv_block)
        self.Mixed_5c = InceptionA(256, pool_features=64, conv_block=conv_block)
        self.Mixed_5d = InceptionA(288, pool_features=64, conv_block=conv_block)
        self.Mixed_6a = InceptionB(288, conv_block=conv_block)
        self.Mixed_6b = InceptionC(768, channels_7x7=128, conv_block=conv_block)
        self.Mixed_6c = InceptionC(768, channels_7x7=160, conv_block=conv_block)
        self.Mixed_6d = InceptionC(768, channels_7x7=160, conv_block=conv_block)
        self.Mixed_6e = InceptionC(768, channels_7x7=192, conv_block=conv_block)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, conv_block=conv_block)
        else:
            self.AuxLogits = None
        self.Mixed_7a = InceptionD(768, conv_block=conv_block)
        self.Mixed_7b = InceptionE(1280, conv_block=conv_block)
        self.Mixed_7c = InceptionE(2048, conv_block=conv_block)
        self.feature_info = [
            dict(num_chs=64, reduction=2, module='Conv2d_2b_3x3'),
            dict(num_chs=192, reduction=4, module='Conv2d_4a_3x3'),
            dict(num_chs=288, reduction=8, module='Mixed_5d'),
            dict(num_chs=768, reduction=16, module='Mixed_6e'),
            dict(num_chs=2048, reduction=32, module='Mixed_7c'),
        ]

        self.num_features = 2048
        self.global_pool, self.head_drop, self.fc = create_classifier(
            self.num_features,
            self.num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                trunc_normal_(m.weight, std=stddev)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        module_map = {k: i for i, (k, _) in enumerate(flatten_modules(self.named_children(), prefix=()))}
        module_map.pop(('fc',))

        def _matcher(name):
            if any([name.startswith(n) for n in ('Conv2d_1', 'Conv2d_2')]):
                return 0
            elif any([name.startswith(n) for n in ('Conv2d_3', 'Conv2d_4')]):
                return 1
            else:
                for k in module_map.keys():
                    if k == tuple(name.split('.')[:len(k)]):
                        return module_map[k]
                return float('inf')
        return _matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_preaux(self, x):
        x = self.Conv2d_1a_3x3(x)  # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)  # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)  # N x 64 x 147 x 147
        x = self.Pool1(x)  # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)  # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)  # N x 192 x 71 x 71
        x = self.Pool2(x)  # N x 192 x 35 x 35
        x = self.Mixed_5b(x)  # N x 256 x 35 x 35
        x = self.Mixed_5c(x)  # N x 288 x 35 x 35
        x = self.Mixed_5d(x)  # N x 288 x 35 x 35
        x = self.Mixed_6a(x)  # N x 768 x 17 x 17
        x = self.Mixed_6b(x)  # N x 768 x 17 x 17
        x = self.Mixed_6c(x)  # N x 768 x 17 x 17
        x = self.Mixed_6d(x)  # N x 768 x 17 x 17
        x = self.Mixed_6e(x)  # N x 768 x 17 x 17
        return x

    def forward_postaux(self, x):
        x = self.Mixed_7a(x)  # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)  # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)  # N x 2048 x 8 x 8
        return x

    def forward_features(self, x):
        x = self.forward_preaux(x)
        if self.aux_logits:
            aux = self.AuxLogits(x)
            x = self.forward_postaux(x)
            return x, aux
        x = self.forward_postaux(x)
        return x

    def forward_head(self, x):
        x = self.global_pool(x)
        x = self.head_drop(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        if self.aux_logits:
            x, aux = self.forward_features(x)
            x = self.forward_head(x)
            return x, aux
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_inception_v3(variant, pretrained=False, **kwargs):
    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    aux_logits = kwargs.get('aux_logits', False)
    has_aux_logits = False
    if pretrained_cfg:
        # only torchvision pretrained weights have aux logits
        has_aux_logits = pretrained_cfg.tag == 'tv_in1k'
    if aux_logits:
        assert not kwargs.pop('features_only', False)
        load_strict = has_aux_logits
    else:
        load_strict = not has_aux_logits

    return build_model_with_cfg(
        InceptionV3,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_strict=load_strict,
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 299, 299), 'pool_size': (8, 8),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'Conv2d_1a_3x3.conv', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # original PyTorch weights, ported from Tensorflow but modified
    'inception_v3.tv_in1k': _cfg(
        # NOTE checkpoint has aux logit layer weights
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'),
    # my port of Tensorflow SLIM weights (http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)
    'inception_v3.tf_in1k': _cfg(hf_hub_id='timm/'),
    # my port of Tensorflow adversarially trained Inception V3 from
    # http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
    'inception_v3.tf_adv_in1k': _cfg(hf_hub_id='timm/'),
    # from gluon pretrained models, best performing in terms of accuracy/loss metrics
    # https://gluon-cv.mxnet.io/model_zoo/classification.html
    'inception_v3.gluon_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN,  # also works well with inception defaults
        std=IMAGENET_DEFAULT_STD,  # also works well with inception defaults
    )
})


@register_model
def inception_v3(pretrained=False, **kwargs) -> InceptionV3:
    model = _create_inception_v3('inception_v3', pretrained=pretrained, **kwargs)
    return model


register_model_deprecations(__name__, {
    'tf_inception_v3': 'inception_v3.tf_in1k',
    'adv_inception_v3': 'inception_v3.tf_adv_in1k',
    'gluon_inception_v3': 'inception_v3.gluon_in1k',
})