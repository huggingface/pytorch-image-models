"""
MambaOut models for image classification.
Some implementations are modified from:
timm (https://github.com/rwightman/pytorch-image-models),
MetaFormer (https://github.com/sail-sg/metaformer),
InceptionNeXt (https://github.com/sail-sg/inceptionnext)
"""
from collections import OrderedDict
from typing import Optional

import torch
from torch import nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import trunc_normal_, DropPath, LayerNorm, LayerScale, ClNormMlpClassifierHead, get_act_layer
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs


class Stem(nn.Module):
    r""" Code modified from InternImage:
        https://github.com/OpenGVLab/InternImage
    """

    def __init__(
            self,
            in_chs=3,
            out_chs=96,
            mid_norm: bool = True,
            act_layer=nn.GELU,
            norm_layer=LayerNorm,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_chs,
            out_chs // 2,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.norm1 = norm_layer(out_chs // 2) if mid_norm else None
        self.act = act_layer()
        self.conv2 = nn.Conv2d(
            out_chs // 2,
            out_chs,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.norm2 = norm_layer(out_chs)

    def forward(self, x):
        x = self.conv1(x)
        if self.norm1 is not None:
            x = x.permute(0, 2, 3, 1)
            x = self.norm1(x)
            x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        return x


class DownsampleNormFirst(nn.Module):

    def __init__(
            self,
            in_chs=96,
            out_chs=198,
            norm_layer=LayerNorm,
    ):
        super().__init__()
        self.norm = norm_layer(in_chs)
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Downsample(nn.Module):

    def __init__(
            self,
            in_chs=96,
            out_chs=198,
            norm_layer=LayerNorm,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.norm = norm_layer(out_chs)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """

    def __init__(
            self,
            in_features,
            num_classes=1000,
            pool_type='avg',
            act_layer=nn.GELU,
            mlp_ratio=4,
            norm_layer=LayerNorm,
            drop_rate=0.,
            bias=True,
    ):
        super().__init__()
        if mlp_ratio is not None:
            hidden_size = int(mlp_ratio * in_features)
        else:
            hidden_size = None
        self.pool_type = pool_type
        self.in_features = in_features
        self.hidden_size = hidden_size or in_features

        self.norm = norm_layer(in_features)
        if hidden_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(in_features, hidden_size)),
                ('act', act_layer()),
                ('norm', norm_layer(hidden_size))
            ]))
            self.num_features = hidden_size
        else:
            self.num_features = in_features
            self.pre_logits = nn.Identity()

        self.fc = nn.Linear(self.num_features, num_classes, bias=bias) if num_classes > 0 else nn.Identity()
        self.head_dropout = nn.Dropout(drop_rate)

    def reset(self, num_classes: int, pool_type: Optional[str] = None, reset_other: bool = False):
        if pool_type is not None:
            self.pool_type = pool_type
        if reset_other:
            self.norm = nn.Identity()
            self.pre_logits = nn.Identity()
            self.num_features = self.in_features
        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, pre_logits: bool = False):
        if self.pool_type == 'avg':
            x = x.mean((1, 2))
        x = self.norm(x)
        x = self.pre_logits(x)
        x = self.head_dropout(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x


class GatedConvBlock(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve paraitcal efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """

    def __init__(
            self,
            dim,
            expansion_ratio=8 / 3,
            kernel_size=7,
            conv_ratio=1.0,
            ls_init_value=None,
            norm_layer=LayerNorm,
            act_layer=nn.GELU,
            drop_path=0.,
            **kwargs
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=conv_channels
        )
        self.fc2 = nn.Linear(hidden, dim)
        self.ls = LayerScale(dim) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x  # [B, H, W, C]
        x = self.norm(x)
        x = self.fc1(x)
        g, i, c = torch.split(x, self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.ls(x)
        x = self.drop_path(x)
        return x + shortcut


class MambaOutStage(nn.Module):

    def __init__(
            self,
            dim,
            dim_out: Optional[int] = None,
            depth: int = 4,
            expansion_ratio=8 / 3,
            kernel_size=7,
            conv_ratio=1.0,
            downsample: str = '',
            ls_init_value: Optional[float] = None,
            norm_layer=LayerNorm,
            act_layer=nn.GELU,
            drop_path=0.,
    ):
        super().__init__()
        dim_out = dim_out or dim
        self.grad_checkpointing = False

        if downsample == 'conv':
            self.downsample = Downsample(dim, dim_out, norm_layer=norm_layer)
        elif downsample == 'conv_nf':
            self.downsample = DownsampleNormFirst(dim, dim_out, norm_layer=norm_layer)
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        self.blocks = nn.Sequential(*[
            GatedConvBlock(
                dim=dim_out,
                expansion_ratio=expansion_ratio,
                kernel_size=kernel_size,
                conv_ratio=conv_ratio,
                ls_init_value=ls_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop_path=drop_path[j] if isinstance(drop_path, (list, tuple)) else drop_path,
            )
            for j in range(depth)
        ])

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class MambaOut(nn.Module):
    r""" MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        depths (list or tuple): Number of blocks at each stage. Default: [3, 3, 9, 3].
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 576].
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        output_norm: norm before classifier head. Default: partial(nn.LayerNorm, eps=1e-6).
        head_fn: classification head. Default: nn.Linear.
        head_dropout (float): dropout for MLP classifier. Default: 0.
    """

    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 576),
            norm_layer=LayerNorm,
            act_layer=nn.GELU,
            conv_ratio=1.0,
            expansion_ratio=8/3,
            kernel_size=7,
            stem_mid_norm=True,
            ls_init_value=None,
            downsample='conv',
            drop_path_rate=0.,
            drop_rate=0.,
            head_fn='default',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.output_fmt = 'NHWC'
        if not isinstance(depths, (list, tuple)):
            depths = [depths]  # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]
        act_layer = get_act_layer(act_layer)

        num_stage = len(depths)
        self.num_stage = num_stage
        self.feature_info = []

        self.stem = Stem(
            in_chans,
            dims[0],
            mid_norm=stem_mid_norm,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        prev_dim = dims[0]
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        cur = 0
        curr_stride = 4
        self.stages = nn.Sequential()
        for i in range(num_stage):
            dim = dims[i]
            stride = 2 if curr_stride == 2 or i > 0 else 1
            curr_stride *= stride
            stage = MambaOutStage(
                dim=prev_dim,
                dim_out=dim,
                depth=depths[i],
                kernel_size=kernel_size,
                conv_ratio=conv_ratio,
                expansion_ratio=expansion_ratio,
                downsample=downsample if i > 0 else '',
                ls_init_value=ls_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop_path=dp_rates[i],
            )
            self.stages.append(stage)
            prev_dim = dim
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [dict(num_chs=prev_dim, reduction=curr_stride, module=f'stages.{i}')]
            cur += depths[i]

        if head_fn == 'default':
            # specific to this model, unusual norm -> pool -> fc -> act -> norm -> fc combo
            self.head = MlpHead(
                prev_dim,
                num_classes,
                pool_type=global_pool,
                drop_rate=drop_rate,
                norm_layer=norm_layer,
            )
        else:
            # more typical norm -> pool -> fc -> act -> fc
            self.head = ClNormMlpClassifierHead(
                prev_dim,
                num_classes,
                hidden_size=int(prev_dim * 4),
                pool_type=global_pool,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
            )
        self.num_features = prev_dim
        self.head_hidden_size = self.head.num_features

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),  # blocks
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    if 'model' in state_dict:
        state_dict = state_dict['model']
    if 'stem.conv1.weight' in state_dict:
        return state_dict

    import re
    out_dict = {}
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+)', r'stages.\1.downsample', k)
        # remap head names
        if k.startswith('norm.'):
            # this is moving to head since it's after the pooling
            k = k.replace('norm.', 'head.norm.')
        elif k.startswith('head.'):
            k = k.replace('head.fc1.', 'head.pre_logits.fc.')
            k = k.replace('head.norm.', 'head.pre_logits.norm.')
            k = k.replace('head.fc2.', 'head.fc.')
        out_dict[k] = v

    return out_dict


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'test_input_size': (3, 288, 288),
        'pool_size': (7, 7), 'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # original weights
    'mambaout_femto.in1k': _cfg(
        hf_hub_id='timm/'),
    'mambaout_kobe.in1k': _cfg(
        hf_hub_id='timm/'),
    'mambaout_tiny.in1k': _cfg(
        hf_hub_id='timm/'),
    'mambaout_small.in1k': _cfg(
        hf_hub_id='timm/'),
    'mambaout_base.in1k': _cfg(
        hf_hub_id='timm/'),

    # timm experiments below
    'mambaout_small_rw.sw_e450_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'mambaout_base_short_rw.sw_e500_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_crop_pct=1.0,
    ),
    'mambaout_base_tall_rw.sw_e500_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_crop_pct=1.0,
    ),
    'mambaout_base_wide_rw.sw_e500_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_crop_pct=1.0,
    ),
    'mambaout_base_plus_rw.sw_e150_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), test_input_size=(3, 384, 384), crop_mode='squash', pool_size=(12, 12),
    ),
    'mambaout_base_plus_rw.sw_e150_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
    ),
    'test_mambaout': _cfg(input_size=(3, 160, 160), test_input_size=(3, 192, 192), pool_size=(5, 5)),
})


def _create_mambaout(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        MambaOut, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )
    return model


# a series of MambaOut models
@register_model
def mambaout_femto(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 9, 3), dims=(48, 96, 192, 288))
    return _create_mambaout('mambaout_femto', pretrained=pretrained, **dict(model_args, **kwargs))

# Kobe Memorial Version with 24 Gated CNN blocks
@register_model
def mambaout_kobe(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 15, 3], dims=[48, 96, 192, 288])
    return _create_mambaout('mambaout_kobe', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def mambaout_tiny(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 576])
    return _create_mambaout('mambaout_tiny', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def mambaout_small(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 4, 27, 3], dims=[96, 192, 384, 576])
    return _create_mambaout('mambaout_small', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def mambaout_base(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 4, 27, 3], dims=[128, 256, 512, 768])
    return _create_mambaout('mambaout_base', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def mambaout_small_rw(pretrained=False, **kwargs):
    model_args = dict(
        depths=[3, 4, 27, 3],
        dims=[96, 192, 384, 576],
        stem_mid_norm=False,
        downsample='conv_nf',
        ls_init_value=1e-6,
        head_fn='norm_mlp',
    )
    return _create_mambaout('mambaout_small_rw', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def mambaout_base_short_rw(pretrained=False, **kwargs):
    model_args = dict(
        depths=(3, 3, 25, 3),
        dims=(128, 256, 512, 768),
        expansion_ratio=3.0,
        conv_ratio=1.25,
        stem_mid_norm=False,
        downsample='conv_nf',
        ls_init_value=1e-6,
        head_fn='norm_mlp',
    )
    return _create_mambaout('mambaout_base_short_rw', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def mambaout_base_tall_rw(pretrained=False, **kwargs):
    model_args = dict(
        depths=(3, 4, 30, 3),
        dims=(128, 256, 512, 768),
        expansion_ratio=2.5,
        conv_ratio=1.25,
        stem_mid_norm=False,
        downsample='conv_nf',
        ls_init_value=1e-6,
        head_fn='norm_mlp',
    )
    return _create_mambaout('mambaout_base_tall_rw', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def mambaout_base_wide_rw(pretrained=False, **kwargs):
    model_args = dict(
        depths=(3, 4, 27, 3),
        dims=(128, 256, 512, 768),
        expansion_ratio=3.0,
        conv_ratio=1.5,
        stem_mid_norm=False,
        downsample='conv_nf',
        ls_init_value=1e-6,
        act_layer='silu',
        head_fn='norm_mlp',
    )
    return _create_mambaout('mambaout_base_wide_rw', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def mambaout_base_plus_rw(pretrained=False, **kwargs):
    model_args = dict(
        depths=(3, 4, 30, 3),
        dims=(128, 256, 512, 768),
        expansion_ratio=3.0,
        conv_ratio=1.5,
        stem_mid_norm=False,
        downsample='conv_nf',
        ls_init_value=1e-6,
        act_layer='silu',
        head_fn='norm_mlp',
    )
    return _create_mambaout('mambaout_base_plus_rw', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def test_mambaout(pretrained=False, **kwargs):
    model_args = dict(
        depths=(1, 1, 3, 1),
        dims=(16, 32, 48, 64),
        expansion_ratio=3,
        stem_mid_norm=False,
        downsample='conv_nf',
        ls_init_value=1e-4,
        act_layer='silu',
        head_fn='norm_mlp',
    )
    return _create_mambaout('test_mambaout', pretrained=pretrained, **dict(model_args, **kwargs))
