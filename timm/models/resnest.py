""" ResNeSt Models

Paper: `ResNeSt: Split-Attention Networks` - https://arxiv.org/abs/2004.08955

Adapted from original PyTorch impl w/ weights at https://github.com/zhanghang1989/ResNeSt by Hang Zhang

Modified for torchscript compat, and consistency with timm by Ross Wightman
"""
from torch import nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SplitAttn
from ._builder import build_model_with_cfg
from ._registry import register_model, generate_default_cfgs
from .resnet import ResNet


class ResNestBottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            radix=1,
            cardinality=1,
            base_width=64,
            avd=False,
            avd_first=False,
            is_first=False,
            reduce_first=1,
            dilation=1,
            first_dilation=None,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            attn_layer=None,
            aa_layer=None,
            drop_block=None,
            drop_path=None,
    ):
        super(ResNestBottleneck, self).__init__()
        assert reduce_first == 1  # not supported
        assert attn_layer is None  # not supported
        assert aa_layer is None  # TODO not yet supported
        assert drop_path is None  # TODO not yet supported

        group_width = int(planes * (base_width / 64.)) * cardinality
        first_dilation = first_dilation or dilation
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0
        self.radix = radix

        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.act1 = act_layer(inplace=True)
        self.avd_first = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and avd_first else None

        if self.radix >= 1:
            self.conv2 = SplitAttn(
                group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, radix=radix, norm_layer=norm_layer, drop_layer=drop_block)
            self.bn2 = nn.Identity()
            self.drop_block = nn.Identity()
            self.act2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)
            self.drop_block = drop_block() if drop_block is not None else nn.Identity()
            self.act2 = act_layer(inplace=True)
        self.avd_last = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and not avd_first else None

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample

    def zero_init_last(self):
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        if self.avd_first is not None:
            out = self.avd_first(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop_block(out)
        out = self.act2(out)

        if self.avd_last is not None:
            out = self.avd_last(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.act3(out)
        return out


def _create_resnest(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ResNet,
        variant,
        pretrained,
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1.0', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'resnest14d.gluon_in1k': _cfg(hf_hub_id='timm/'),
    'resnest26d.gluon_in1k': _cfg(hf_hub_id='timm/'),
    'resnest50d.in1k': _cfg(hf_hub_id='timm/'),
    'resnest101e.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8)),
    'resnest200e.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=0.909, interpolation='bicubic'),
    'resnest269e.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 416, 416), pool_size=(13, 13), crop_pct=0.928, interpolation='bicubic'),
    'resnest50d_4s2x40d.in1k': _cfg(
        hf_hub_id='timm/',
        interpolation='bicubic'),
    'resnest50d_1s4x24d.in1k': _cfg(
        hf_hub_id='timm/',
        interpolation='bicubic')
})


@register_model
def resnest14d(pretrained=False, **kwargs) -> ResNet:
    """ ResNeSt-14d model. Weights ported from GluonCV.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[1, 1, 1, 1],
        stem_type='deep', stem_width=32, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False))
    return _create_resnest('resnest14d', pretrained=pretrained, **dict(model_kwargs, **kwargs))


@register_model
def resnest26d(pretrained=False, **kwargs) -> ResNet:
    """ ResNeSt-26d model. Weights ported from GluonCV.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[2, 2, 2, 2],
        stem_type='deep', stem_width=32, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False))
    return _create_resnest('resnest26d', pretrained=pretrained, **dict(model_kwargs, **kwargs))


@register_model
def resnest50d(pretrained=False, **kwargs) -> ResNet:
    """ ResNeSt-50d model. Matches paper ResNeSt-50 model, https://arxiv.org/abs/2004.08955
    Since this codebase supports all possible variations, 'd' for deep stem, stem_width 32, avg in downsample.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 4, 6, 3],
        stem_type='deep', stem_width=32, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False))
    return _create_resnest('resnest50d', pretrained=pretrained, **dict(model_kwargs, **kwargs))


@register_model
def resnest101e(pretrained=False, **kwargs) -> ResNet:
    """ ResNeSt-101e model. Matches paper ResNeSt-101 model, https://arxiv.org/abs/2004.08955
     Since this codebase supports all possible variations, 'e' for deep stem, stem_width 64, avg in downsample.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 4, 23, 3],
        stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False))
    return _create_resnest('resnest101e', pretrained=pretrained, **dict(model_kwargs, **kwargs))


@register_model
def resnest200e(pretrained=False, **kwargs) -> ResNet:
    """ ResNeSt-200e model. Matches paper ResNeSt-200 model, https://arxiv.org/abs/2004.08955
    Since this codebase supports all possible variations, 'e' for deep stem, stem_width 64, avg in downsample.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 24, 36, 3],
        stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False))
    return _create_resnest('resnest200e', pretrained=pretrained, **dict(model_kwargs, **kwargs))


@register_model
def resnest269e(pretrained=False, **kwargs) -> ResNet:
    """ ResNeSt-269e model. Matches paper ResNeSt-269 model, https://arxiv.org/abs/2004.08955
    Since this codebase supports all possible variations, 'e' for deep stem, stem_width 64, avg in downsample.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 30, 48, 8],
        stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False))
    return _create_resnest('resnest269e', pretrained=pretrained, **dict(model_kwargs, **kwargs))


@register_model
def resnest50d_4s2x40d(pretrained=False, **kwargs) -> ResNet:
    """ResNeSt-50 4s2x40d from https://github.com/zhanghang1989/ResNeSt/blob/master/ablation.md
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 4, 6, 3],
        stem_type='deep', stem_width=32, avg_down=True, base_width=40, cardinality=2,
        block_args=dict(radix=4, avd=True, avd_first=True))
    return _create_resnest('resnest50d_4s2x40d', pretrained=pretrained, **dict(model_kwargs, **kwargs))


@register_model
def resnest50d_1s4x24d(pretrained=False, **kwargs) -> ResNet:
    """ResNeSt-50 1s4x24d from https://github.com/zhanghang1989/ResNeSt/blob/master/ablation.md
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 4, 6, 3],
        stem_type='deep', stem_width=32, avg_down=True, base_width=24, cardinality=4,
        block_args=dict(radix=1, avd=True, avd_first=True))
    return _create_resnest('resnest50d_1s4x24d', pretrained=pretrained, **dict(model_kwargs, **kwargs))
