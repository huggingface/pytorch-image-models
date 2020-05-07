"""Pytorch Densenet implementation w/ tweaks
This file is a copy of https://github.com/pytorch/vision 'densenet.py' (BSD-3-Clause) with
fixed kwargs passthrough and addition of dynamic global avg/max pool.
"""
import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.jit.annotations import List

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import load_pretrained
from .layers import SelectAdaptivePool2d
from .registry import register_model

__all__ = ['DenseNet']


def _cfg(url=''):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'features.conv0', 'classifier': 'classifier',
    }


default_cfgs = {
    'densenet121': _cfg(url='https://download.pytorch.org/models/densenet121-a639ec97.pth'),
    'densenet121d': _cfg(url=''),
    'densenet121tn': _cfg(url=''),
    'densenet169': _cfg(url='https://download.pytorch.org/models/densenet169-b2777c0a.pth'),
    'densenet201': _cfg(url='https://download.pytorch.org/models/densenet201-c1103571.pth'),
    'densenet161': _cfg(url='https://download.pytorch.org/models/densenet161-8d451a50.pth'),
}


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 drop_rate=0., memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', norm_layer(num_input_features)),
        self.add_module('relu1', act_layer(inplace=True)),
        self.add_module('conv1', nn.Conv2d(
            num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', norm_layer(bn_size * growth_rate)),
        self.add_module('relu2', act_layer(inplace=True)),
        self.add_module('conv2', nn.Conv2d(
            bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[torch.Tensor]) -> torch.Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[torch.Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[torch.Tensor]) -> torch.Tensor
        def closure(*inputs):
            return self.bn_function(*inputs)

        return cp.checkpoint(closure, input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[torch.Tensor]) -> (torch.Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (torch.Tensor) -> (torch.Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d, drop_rate=0., memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(_Transition, self).__init__()
        self.add_module('norm', norm_layer(num_input_features))
        self.add_module('relu', act_layer(inplace=True))
        self.add_module('conv', nn.Conv2d(
            num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, stem_type='', num_classes=1000, in_chans=3, global_pool='avg',
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None, drop_rate=0, memory_efficient=False):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        deep_stem = 'deep' in stem_type
        super(DenseNet, self).__init__()

        # First convolution
        if aa_layer is None:
            max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            max_pool = nn.Sequential(*[
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                aa_layer(channels=self.inplanes, stride=2)])
        if deep_stem:
            stem_chs_1 = stem_chs_2 = num_init_features // 2
            if 'tiered' in stem_type:
                stem_chs_1 = 3 * (num_init_features // 8)
                stem_chs_2 = num_init_features if 'narrow' in stem_type else 6 * (num_init_features // 8)
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(in_chans, stem_chs_1, 3, stride=2, padding=1, bias=False)),
                ('norm0', norm_layer(stem_chs_1)),
                ('relu0', act_layer(inplace=True)),
                ('conv1', nn.Conv2d(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False)),
                ('norm1', norm_layer(stem_chs_2)),
                ('relu1', act_layer(inplace=True)),
                ('conv2', nn.Conv2d(stem_chs_2, num_init_features, 3, stride=1, padding=1, bias=False)),
                ('norm2', norm_layer(num_init_features)),
                ('relu2', act_layer(inplace=True)),
                ('pool0', max_pool),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(in_chans, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', norm_layer(num_init_features)),
                ('relu0', act_layer(inplace=True)),
                ('pool0', max_pool),
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', norm_layer(num_features))
        self.act = act_layer(inplace=True)

        # Linear layer
        self.num_features = num_features
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.classifier = nn.Linear(num_features, num_classes)
        else:
            self.classifier = nn.Identity()

    def forward_features(self, x):
        x = self.features(x)
        x = self.act(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        # both classifier and block drop?
        # if self.drop_rate > 0.:
        #     x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x


def _filter_torchvision_pretrained(state_dict):
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict


def _densenet(variant, growth_rate, block_config, num_init_features, pretrained, **kwargs):
    if kwargs.pop('features_only', False):
        assert False, 'Not Implemented'  # TODO
        load_strict = False
        kwargs.pop('num_classes', 0)
        model_class = DenseNet
    else:
        load_strict = True
        model_class = DenseNet
    default_cfg = default_cfgs[variant]
    model = model_class(
        growth_rate=growth_rate, block_config=block_config, num_init_features=num_init_features, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(
            model, default_cfg,
            num_classes=kwargs.get('num_classes', 0),
            in_chans=kwargs.get('in_chans', 3),
            filter_fn=_filter_torchvision_pretrained,
            strict=load_strict)
    return model


@register_model
def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = _densenet(
        'densenet121', growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def densenet121d(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = _densenet(
        'densenet121d', growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
        stem_type='deep', pretrained=pretrained, **kwargs)
    return model


@register_model
def densenet121tn(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = _densenet(
        'densenet121tn', growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
        stem_type='deep_tiered_narrow', pretrained=pretrained, **kwargs)
    return model


@register_model
def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = _densenet(
        'densenet169', growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = _densenet(
        'densenet201', growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64,
         pretrained=pretrained, **kwargs)
    return model


@register_model
def densenet161(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = _densenet(
        'densenet161', growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96,
        pretrained=pretrained, **kwargs)
    return model
