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
from timm.layers import BatchNormAct2d, get_norm_act_layer, BlurPool2d, create_classifier
from ._builder import build_model_with_cfg
from ._manipulate import MATCH_PREV_GROUP
from ._registry import register_model, generate_default_cfgs

__all__ = ['DenseNet']


class DenseLayer(nn.Module):
    def __init__(
            self,
            num_input_features,
            growth_rate,
            bn_size,
            norm_layer=BatchNormAct2d,
            drop_rate=0.,
            grad_checkpointing=False,
    ):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', norm_layer(num_input_features)),
        self.add_module('conv1', nn.Conv2d(
            num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', norm_layer(bn_size * growth_rate)),
        self.add_module('conv2', nn.Conv2d(
            bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = float(drop_rate)
        self.grad_checkpointing = grad_checkpointing

    def bottleneck_fn(self, xs):
        # type: (List[torch.Tensor]) -> torch.Tensor
        concated_features = torch.cat(xs, 1)
        bottleneck_output = self.conv1(self.norm1(concated_features))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, x):
        # type: (List[torch.Tensor]) -> bool
        for tensor in x:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, x):
        # type: (List[torch.Tensor]) -> torch.Tensor
        def closure(*xs):
            return self.bottleneck_fn(xs)

        return cp.checkpoint(closure, *x)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (List[torch.Tensor]) -> (torch.Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (torch.Tensor) -> (torch.Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, x):  # noqa: F811
        if isinstance(x, torch.Tensor):
            prev_features = [x]
        else:
            prev_features = x

        if self.grad_checkpointing and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bottleneck_fn(prev_features)

        new_features = self.conv2(self.norm2(bottleneck_output))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
            self,
            num_layers,
            num_input_features,
            bn_size,
            growth_rate,
            norm_layer=BatchNormAct2d,
            drop_rate=0.,
            grad_checkpointing=False,
    ):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                grad_checkpointing=grad_checkpointing,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseTransition(nn.Sequential):
    def __init__(
            self,
            num_input_features,
            num_output_features,
            norm_layer=BatchNormAct2d,
            aa_layer=None,
    ):
        super(DenseTransition, self).__init__()
        self.add_module('norm', norm_layer(num_input_features))
        self.add_module('conv', nn.Conv2d(
            num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if aa_layer is not None:
            self.add_module('pool', aa_layer(num_output_features, stride=2))
        else:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate before classifier layer
        proj_drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(
            self,
            growth_rate=32,
            block_config=(6, 12, 24, 16),
            num_classes=1000,
            in_chans=3,
            global_pool='avg',
            bn_size=4,
            stem_type='',
            act_layer='relu',
            norm_layer='batchnorm2d',
            aa_layer=None,
            drop_rate=0.,
            proj_drop_rate=0.,
            memory_efficient=False,
            aa_stem_only=True,
    ):
        self.num_classes = num_classes
        super(DenseNet, self).__init__()
        norm_layer = get_norm_act_layer(norm_layer, act_layer=act_layer)

        # Stem
        deep_stem = 'deep' in stem_type  # 3x3 deep stem
        num_init_features = growth_rate * 2
        if aa_layer is None:
            stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            stem_pool = nn.Sequential(*[
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                aa_layer(channels=num_init_features, stride=2)])
        if deep_stem:
            stem_chs_1 = stem_chs_2 = growth_rate
            if 'tiered' in stem_type:
                stem_chs_1 = 3 * (growth_rate // 4)
                stem_chs_2 = num_init_features if 'narrow' in stem_type else 6 * (growth_rate // 4)
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(in_chans, stem_chs_1, 3, stride=2, padding=1, bias=False)),
                ('norm0', norm_layer(stem_chs_1)),
                ('conv1', nn.Conv2d(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False)),
                ('norm1', norm_layer(stem_chs_2)),
                ('conv2', nn.Conv2d(stem_chs_2, num_init_features, 3, stride=1, padding=1, bias=False)),
                ('norm2', norm_layer(num_init_features)),
                ('pool0', stem_pool),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(in_chans, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', norm_layer(num_init_features)),
                ('pool0', stem_pool),
            ]))
        self.feature_info = [
            dict(num_chs=num_init_features, reduction=2, module=f'features.norm{2 if deep_stem else 0}')]
        current_stride = 4

        # DenseBlocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                norm_layer=norm_layer,
                drop_rate=proj_drop_rate,
                grad_checkpointing=memory_efficient,
            )
            module_name = f'denseblock{(i + 1)}'
            self.features.add_module(module_name, block)
            num_features = num_features + num_layers * growth_rate
            transition_aa_layer = None if aa_stem_only else aa_layer
            if i != len(block_config) - 1:
                self.feature_info += [
                    dict(num_chs=num_features, reduction=current_stride, module='features.' + module_name)]
                current_stride *= 2
                trans = DenseTransition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    norm_layer=norm_layer,
                    aa_layer=transition_aa_layer,
                )
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', norm_layer(num_features))

        self.feature_info += [dict(num_chs=num_features, reduction=current_stride, module='features.norm5')]
        self.num_features = num_features

        # Linear layer
        global_pool, classifier = create_classifier(
            self.num_features,
            self.num_classes,
            pool_type=global_pool,
        )
        self.global_pool = global_pool
        self.head_drop = nn.Dropout(drop_rate)
        self.classifier = classifier

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^features\.conv[012]|features\.norm[012]|features\.pool[012]',
            blocks=r'^features\.(?:denseblock|transition)(\d+)' if coarse else [
                (r'^features\.denseblock(\d+)\.denselayer(\d+)', None),
                (r'^features\.transition(\d+)', MATCH_PREV_GROUP)  # FIXME combine with previous denselayer
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for b in self.features.modules():
            if isinstance(b, DenseLayer):
                b.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        return self.features(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = self.head_drop(x)
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


def _create_densenet(variant, growth_rate, block_config, pretrained, **kwargs):
    kwargs['growth_rate'] = growth_rate
    kwargs['block_config'] = block_config
    return build_model_with_cfg(
        DenseNet,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True),
        pretrained_filter_fn=_filter_torchvision_pretrained,
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'features.conv0', 'classifier': 'classifier', **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'densenet121.ra_in1k': _cfg(
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    'densenetblur121d.ra_in1k': _cfg(
        hf_hub_id='timm/',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    'densenet264d.untrained': _cfg(),
    'densenet121.tv_in1k': _cfg(hf_hub_id='timm/'),
    'densenet169.tv_in1k': _cfg(hf_hub_id='timm/'),
    'densenet201.tv_in1k': _cfg(hf_hub_id='timm/'),
    'densenet161.tv_in1k': _cfg(hf_hub_id='timm/'),
})


@register_model
def densenet121(pretrained=False, **kwargs) -> DenseNet:
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model_args = dict(growth_rate=32, block_config=(6, 12, 24, 16))
    model = _create_densenet('densenet121', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def densenetblur121d(pretrained=False, **kwargs) -> DenseNet:
    r"""Densenet-121 w/ blur-pooling & 3-layer 3x3 stem
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model_args = dict(growth_rate=32, block_config=(6, 12, 24, 16), stem_type='deep', aa_layer=BlurPool2d)
    model = _create_densenet('densenetblur121d', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def densenet169(pretrained=False, **kwargs) -> DenseNet:
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model_args = dict(growth_rate=32, block_config=(6, 12, 32, 32))
    model = _create_densenet('densenet169', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def densenet201(pretrained=False, **kwargs) -> DenseNet:
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model_args = dict(growth_rate=32, block_config=(6, 12, 48, 32))
    model = _create_densenet('densenet201', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def densenet161(pretrained=False, **kwargs) -> DenseNet:
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model_args = dict(growth_rate=48, block_config=(6, 12, 36, 24))
    model = _create_densenet('densenet161', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def densenet264d(pretrained=False, **kwargs) -> DenseNet:
    r"""Densenet-264 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model_args = dict(growth_rate=48, block_config=(6, 12, 64, 48), stem_type='deep')
    model = _create_densenet('densenet264d', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

