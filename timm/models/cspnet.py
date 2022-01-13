"""PyTorch CspNet

A PyTorch implementation of Cross Stage Partial Networks including:
* CSPResNet50
* CSPResNeXt50
* CSPDarkNet53
* and DarkNet53 for good measure

Based on paper `CSPNet: A New Backbone that can Enhance Learning Capability of CNN` - https://arxiv.org/abs/1911.11929

Reference impl via darknet cfg files at https://github.com/WongKinYiu/CrossStagePartialNetworks

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import ClassifierHead, ConvBnAct, DropPath, create_attn, get_norm_act_layer
from .registry import register_model


__all__ = ['CspNet']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 256, 256), 'pool_size': (8, 8),
        'crop_pct': 0.887, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1.conv', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = {
    'cspresnet50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/cspresnet50_ra-d3e8d487.pth'),
    'cspresnet50d': _cfg(url=''),
    'cspresnet50w': _cfg(url=''),
    'cspresnext50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/cspresnext50_ra_224-648b4713.pth',
        input_size=(3, 224, 224), pool_size=(7, 7), crop_pct=0.875  # FIXME I trained this at 224x224, not 256 like ref impl
    ),
    'cspresnext50_iabn': _cfg(url=''),
    'cspdarknet53': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/cspdarknet53_ra_256-d05c7c21.pth'),
    'cspdarknet53_iabn': _cfg(url=''),
    'darknet53': _cfg(url=''),
}


model_cfgs = dict(
    cspresnet50=dict(
        stem=dict(out_chs=64, kernel_size=7, stride=2, pool='max'),
        stage=dict(
            out_chs=(128, 256, 512, 1024),
            depth=(3, 3, 5, 2),
            stride=(1,) + (2,) * 3,
            exp_ratio=(2.,) * 4,
            bottle_ratio=(0.5,) * 4,
            block_ratio=(1.,) * 4,
            cross_linear=True,
        )
    ),
    cspresnet50d=dict(
        stem=dict(out_chs=[32, 32, 64], kernel_size=3, stride=2, pool='max'),
        stage=dict(
            out_chs=(128, 256, 512, 1024),
            depth=(3, 3, 5, 2),
            stride=(1,) + (2,) * 3,
            exp_ratio=(2.,) * 4,
            bottle_ratio=(0.5,) * 4,
            block_ratio=(1.,) * 4,
            cross_linear=True,
        )
    ),
    cspresnet50w=dict(
        stem=dict(out_chs=[32, 32, 64], kernel_size=3, stride=2, pool='max'),
        stage=dict(
            out_chs=(256, 512, 1024, 2048),
            depth=(3, 3, 5, 2),
            stride=(1,) + (2,) * 3,
            exp_ratio=(1.,) * 4,
            bottle_ratio=(0.25,) * 4,
            block_ratio=(0.5,) * 4,
            cross_linear=True,
        )
    ),
    cspresnext50=dict(
        stem=dict(out_chs=64, kernel_size=7, stride=2, pool='max'),
        stage=dict(
            out_chs=(256, 512, 1024, 2048),
            depth=(3, 3, 5, 2),
            stride=(1,) + (2,) * 3,
            groups=(32,) * 4,
            exp_ratio=(1.,) * 4,
            bottle_ratio=(1.,) * 4,
            block_ratio=(0.5,) * 4,
            cross_linear=True,
        )
    ),
    cspdarknet53=dict(
        stem=dict(out_chs=32, kernel_size=3, stride=1, pool=''),
        stage=dict(
            out_chs=(64, 128, 256, 512, 1024),
            depth=(1, 2, 8, 8, 4),
            stride=(2,) * 5,
            exp_ratio=(2.,) + (1.,) * 4,
            bottle_ratio=(0.5,) + (1.0,) * 4,
            block_ratio=(1.,) + (0.5,) * 4,
            down_growth=True,
        )
    ),
    darknet53=dict(
        stem=dict(out_chs=32, kernel_size=3, stride=1, pool=''),
        stage=dict(
            out_chs=(64, 128, 256, 512, 1024),
            depth=(1, 2, 8, 8, 4),
            stride=(2,) * 5,
            bottle_ratio=(0.5,) * 5,
            block_ratio=(1.,) * 5,
        )
    )
)


def create_stem(
        in_chans=3, out_chs=32, kernel_size=3, stride=2, pool='',
        act_layer=None, norm_layer=None, aa_layer=None):
    stem = nn.Sequential()
    if not isinstance(out_chs, (tuple, list)):
        out_chs = [out_chs]
    assert len(out_chs)
    in_c = in_chans
    for i, out_c in enumerate(out_chs):
        conv_name = f'conv{i + 1}'
        stem.add_module(conv_name, ConvBnAct(
            in_c, out_c, kernel_size, stride=stride if i == 0 else 1,
            act_layer=act_layer, norm_layer=norm_layer))
        in_c = out_c
        last_conv = conv_name
    if pool:
        if aa_layer is not None:
            stem.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            stem.add_module('aa', aa_layer(channels=in_c, stride=2))
        else:
            stem.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    return stem, dict(num_chs=in_c, reduction=stride, module='.'.join(['stem', last_conv]))


class ResBottleneck(nn.Module):
    """ ResNe(X)t Bottleneck Block
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.25, groups=1,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_last=False,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(ResBottleneck, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer, drop_block=drop_block)

        self.conv1 = ConvBnAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.conv2 = ConvBnAct(mid_chs, mid_chs, kernel_size=3, dilation=dilation, groups=groups, **ckwargs)
        self.attn2 = create_attn(attn_layer, channels=mid_chs) if not attn_last else None
        self.conv3 = ConvBnAct(mid_chs, out_chs, kernel_size=1, apply_act=False, **ckwargs)
        self.attn3 = create_attn(attn_layer, channels=out_chs) if attn_last else None
        self.drop_path = drop_path
        self.act3 = act_layer(inplace=True)

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.attn2 is not None:
            x = self.attn2(x)
        x = self.conv3(x)
        if self.attn3 is not None:
            x = self.attn3(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        x = x + shortcut
        # FIXME partial shortcut needed if first block handled as per original, not used for my current impl
        #x[:, :shortcut.size(1)] += shortcut
        x = self.act3(x)
        return x


class DarkBlock(nn.Module):
    """ DarkNet Block
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.5, groups=1,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, aa_layer=None,
                 drop_block=None, drop_path=None):
        super(DarkBlock, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer, drop_block=drop_block)
        self.conv1 = ConvBnAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.conv2 = ConvBnAct(mid_chs, out_chs, kernel_size=3, dilation=dilation, groups=groups, **ckwargs)
        self.attn = create_attn(attn_layer, channels=out_chs)
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        x = x + shortcut
        return x


class CrossStage(nn.Module):
    """Cross Stage."""
    def __init__(self, in_chs, out_chs, stride, dilation, depth, block_ratio=1., bottle_ratio=1., exp_ratio=1.,
                 groups=1, first_dilation=None, down_growth=False, cross_linear=False, block_dpr=None,
                 block_fn=ResBottleneck, **block_kwargs):
        super(CrossStage, self).__init__()
        first_dilation = first_dilation or dilation
        down_chs = out_chs if down_growth else in_chs  # grow downsample channels to output channels
        exp_chs = int(round(out_chs * exp_ratio))
        block_out_chs = int(round(out_chs * block_ratio))
        conv_kwargs = dict(act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'))

        if stride != 1 or first_dilation != dilation:
            self.conv_down = ConvBnAct(
                in_chs, down_chs, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups,
                aa_layer=block_kwargs.get('aa_layer', None), **conv_kwargs)
            prev_chs = down_chs
        else:
            self.conv_down = None
            prev_chs = in_chs

        # FIXME this 1x1 expansion is pushed down into the cross and block paths in the darknet cfgs. Also,
        # there is also special case for the first stage for some of the model that results in uneven split
        # across the two paths. I did it this way for simplicity for now.
        self.conv_exp = ConvBnAct(prev_chs, exp_chs, kernel_size=1, apply_act=not cross_linear, **conv_kwargs)
        prev_chs = exp_chs // 2  # output of conv_exp is always split in two

        self.blocks = nn.Sequential()
        for i in range(depth):
            drop_path = DropPath(block_dpr[i]) if block_dpr and block_dpr[i] else None
            self.blocks.add_module(str(i), block_fn(
                prev_chs, block_out_chs, dilation, bottle_ratio, groups, drop_path=drop_path, **block_kwargs))
            prev_chs = block_out_chs

        # transition convs
        self.conv_transition_b = ConvBnAct(prev_chs, exp_chs // 2, kernel_size=1, **conv_kwargs)
        self.conv_transition = ConvBnAct(exp_chs, out_chs, kernel_size=1, **conv_kwargs)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        x = self.conv_exp(x)
        split = x.shape[1] // 2
        xs, xb = x[:, :split], x[:, split:]
        xb = self.blocks(xb)
        xb = self.conv_transition_b(xb).contiguous()
        out = self.conv_transition(torch.cat([xs, xb], dim=1))
        return out


class DarkStage(nn.Module):
    """DarkNet stage."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, block_ratio=1., bottle_ratio=1., groups=1,
                 first_dilation=None, block_fn=ResBottleneck, block_dpr=None, **block_kwargs):
        super(DarkStage, self).__init__()
        first_dilation = first_dilation or dilation

        self.conv_down = ConvBnAct(
            in_chs, out_chs, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups,
            act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'),
            aa_layer=block_kwargs.get('aa_layer', None))

        prev_chs = out_chs
        block_out_chs = int(round(out_chs * block_ratio))
        self.blocks = nn.Sequential()
        for i in range(depth):
            drop_path = DropPath(block_dpr[i]) if block_dpr and block_dpr[i] else None
            self.blocks.add_module(str(i), block_fn(
                prev_chs, block_out_chs, dilation, bottle_ratio, groups, drop_path=drop_path, **block_kwargs))
            prev_chs = block_out_chs

    def forward(self, x):
        x = self.conv_down(x)
        x = self.blocks(x)
        return x


def _cfg_to_stage_args(cfg, curr_stride=2, output_stride=32, drop_path_rate=0.):
    # get per stage args for stage and containing blocks, calculate strides to meet target output_stride
    num_stages = len(cfg['depth'])
    if 'groups' not in cfg:
        cfg['groups'] = (1,) * num_stages
    if 'down_growth' in cfg and not isinstance(cfg['down_growth'], (list, tuple)):
        cfg['down_growth'] = (cfg['down_growth'],) * num_stages
    if 'cross_linear' in cfg and not isinstance(cfg['cross_linear'], (list, tuple)):
        cfg['cross_linear'] = (cfg['cross_linear'],) * num_stages
    cfg['block_dpr'] = [None] * num_stages if not drop_path_rate else \
        [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg['depth'])).split(cfg['depth'])]
    stage_strides = []
    stage_dilations = []
    stage_first_dilations = []
    dilation = 1
    for cfg_stride in cfg['stride']:
        stage_first_dilations.append(dilation)
        if curr_stride >= output_stride:
            dilation *= cfg_stride
            stride = 1
        else:
            stride = cfg_stride
            curr_stride *= stride
        stage_strides.append(stride)
        stage_dilations.append(dilation)
    cfg['stride'] = stage_strides
    cfg['dilation'] = stage_dilations
    cfg['first_dilation'] = stage_first_dilations
    stage_args = [dict(zip(cfg.keys(), values)) for values in zip(*cfg.values())]
    return stage_args


class CspNet(nn.Module):
    """Cross Stage Partial base model.

    Paper: `CSPNet: A New Backbone that can Enhance Learning Capability of CNN` - https://arxiv.org/abs/1911.11929
    Ref Impl: https://github.com/WongKinYiu/CrossStagePartialNetworks

    NOTE: There are differences in the way I handle the 1x1 'expansion' conv in this impl vs the
    darknet impl. I did it this way for simplicity and less special cases.
    """

    def __init__(self, cfg, in_chans=3, num_classes=1000, output_stride=32, global_pool='avg', drop_rate=0.,
                 act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm2d, aa_layer=None, drop_path_rate=0.,
                 zero_init_last_bn=True, stage_fn=CrossStage, block_fn=ResBottleneck):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)
        layer_args = dict(act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer)

        # Construct the stem
        self.stem, stem_feat_info = create_stem(in_chans, **cfg['stem'], **layer_args)
        self.feature_info = [stem_feat_info]
        prev_chs = stem_feat_info['num_chs']
        curr_stride = stem_feat_info['reduction']  # reduction does not include pool
        if cfg['stem']['pool']:
            curr_stride *= 2

        # Construct the stages
        per_stage_args = _cfg_to_stage_args(
            cfg['stage'], curr_stride=curr_stride, output_stride=output_stride, drop_path_rate=drop_path_rate)
        self.stages = nn.Sequential()
        for i, sa in enumerate(per_stage_args):
            self.stages.add_module(
                str(i), stage_fn(prev_chs, **sa, **layer_args, block_fn=block_fn))
            prev_chs = sa['out_chs']
            curr_stride *= sa['stride']
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]

        # Construct the head
        self.num_features = prev_chs
        self.head = ClassifierHead(
            in_chs=prev_chs, num_classes=num_classes, pool_type=global_pool, drop_rate=drop_rate)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _create_cspnet(variant, pretrained=False, **kwargs):
    cfg_variant = variant.split('_')[0]
    # NOTE: DarkNet is one of few models with stride==1 features w/ 6 out_indices [0..5]
    out_indices = kwargs.pop('out_indices', (0, 1, 2, 3, 4, 5) if 'darknet' in variant else (0, 1, 2, 3, 4))
    return build_model_with_cfg(
        CspNet, variant, pretrained,
        default_cfg=default_cfgs[variant],
        model_cfg=model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)


@register_model
def cspresnet50(pretrained=False, **kwargs):
    return _create_cspnet('cspresnet50', pretrained=pretrained, **kwargs)


@register_model
def cspresnet50d(pretrained=False, **kwargs):
    return _create_cspnet('cspresnet50d', pretrained=pretrained, **kwargs)


@register_model
def cspresnet50w(pretrained=False, **kwargs):
    return _create_cspnet('cspresnet50w', pretrained=pretrained, **kwargs)


@register_model
def cspresnext50(pretrained=False, **kwargs):
    return _create_cspnet('cspresnext50', pretrained=pretrained, **kwargs)


@register_model
def cspresnext50_iabn(pretrained=False, **kwargs):
    norm_layer = get_norm_act_layer('iabn')
    return _create_cspnet('cspresnext50_iabn', pretrained=pretrained, norm_layer=norm_layer, **kwargs)


@register_model
def cspdarknet53(pretrained=False, **kwargs):
    return _create_cspnet('cspdarknet53', pretrained=pretrained, block_fn=DarkBlock, **kwargs)


@register_model
def cspdarknet53_iabn(pretrained=False, **kwargs):
    norm_layer = get_norm_act_layer('iabn')
    return _create_cspnet('cspdarknet53_iabn', pretrained=pretrained, block_fn=DarkBlock, norm_layer=norm_layer, **kwargs)


@register_model
def darknet53(pretrained=False, **kwargs):
    return _create_cspnet('darknet53', pretrained=pretrained, block_fn=DarkBlock, stage_fn=DarkStage, **kwargs)
