"""
InceptionNeXt paper: https://arxiv.org/abs/2303.16900
Original implementation & weights from: https://github.com/sail-sg/inceptionnext
"""

from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import trunc_normal_, DropPath, to_2tuple, get_padding, SelectAdaptivePool2d
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs


class InceptionDWConv2d(nn.Module):
    """ Inception depthwise convolution
    """

    def __init__(
            self,
            in_chs,
            square_kernel_size=3,
            band_kernel_size=11,
            branch_ratio=0.125,
            dilation=1,
    ):
        super().__init__()

        gc = int(in_chs * branch_ratio)  # channel numbers of a convolution branch
        square_padding = get_padding(square_kernel_size, dilation=dilation)
        band_padding = get_padding(band_kernel_size, dilation=dilation)
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size,
            padding=square_padding, dilation=dilation, groups=gc)
        self.dwconv_w = nn.Conv2d(
            gc, gc, (1, band_kernel_size),
            padding=(0, band_padding), dilation=(1, dilation), groups=gc)
        self.dwconv_h = nn.Conv2d(
            gc, gc, (band_kernel_size, 1),
            padding=(band_padding, 0), dilation=(dilation, 1), groups=gc)
        self.split_indexes = (in_chs - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((
            x_id,
            self.dwconv_hw(x_hw),
            self.dwconv_w(x_w),
            self.dwconv_h(x_h)
            ), dim=1,
        )


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.ReLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MlpClassifierHead(nn.Module):
    """ MLP classification head
    """

    def __init__(
            self,
            dim,
            num_classes=1000,
            pool_type='avg',
            mlp_ratio=3,
            act_layer=nn.GELU,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop=0.,
            bias=True
    ):
        super().__init__()
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=True)
        in_features = dim * self.global_pool.feat_mult()
        hidden_features = int(mlp_ratio * in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.global_pool(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MetaNeXtBlock(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            dilation=1,
            token_mixer=InceptionDWConv2d,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,

    ):
        super().__init__()
        self.token_mixer = token_mixer(dim, dilation=dilation)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class MetaNeXtStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            stride=2,
            depth=2,
            dilation=(1, 1),
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer=InceptionDWConv2d,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        self.grad_checkpointing = False
        if stride > 1 or dilation[0] != dilation[1]:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=2,
                    stride=stride,
                    dilation=dilation[0],
                ),
            )
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock(
                dim=out_chs,
                dilation=dilation[1],
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                token_mixer=token_mixer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class MetaNeXt(nn.Module):
    r""" MetaNeXt
        A PyTorch impl of : `InceptionNeXt: When Inception Meets ConvNeXt` - https://arxiv.org/abs/2303.16900

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: (3, 3, 9, 3)
        dims (tuple(int)): Feature dimension at each stage. Default: (96, 192, 384, 768)
        token_mixers: Token mixer function. Default: nn.Identity
        norm_layer: Normalization layer. Default: nn.BatchNorm2d
        act_layer: Activation function for MLP. Default: nn.GELU
        mlp_ratios (int or tuple(int)): MLP ratios. Default: (4, 4, 4, 3)
        head_fn: classifier head
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            output_stride=32,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            token_mixers=InceptionDWConv2d,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.GELU,
            mlp_ratios=(4, 4, 4, 3),
            head_fn=MlpClassifierHead,
            drop_rate=0.,
            drop_path_rate=0.,
            ls_init_value=1e-6,
    ):
        super().__init__()

        num_stage = len(depths)
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage
        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.drop_rate = drop_rate
        self.feature_info = []

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            norm_layer(dims[0])
        )

        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_chs = dims[0]
        curr_stride = 4
        dilation = 1
        # feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.Sequential()
        for i in range(num_stage):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]
            self.stages.append(MetaNeXtStage(
                prev_chs,
                out_chs,
                stride=stride if i > 0 else 1,
                dilation=(first_dilation, dilation),
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                token_mixer=token_mixers[i],
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratios[i],
            ))
            prev_chs = out_chs
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.num_features = prev_chs
        if self.num_classes > 0:
            if issubclass(head_fn, MlpClassifierHead):
                assert self.global_pool, 'Cannot disable global pooling with MLP head present.'
            self.head = head_fn(self.num_features, num_classes, pool_type=self.global_pool, drop=drop_rate)
        else:
            if self.global_pool:
                self.head = SelectAdaptivePool2d(pool_type=self.global_pool, flatten=True)
            else:
                self.head = nn.Identity()
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
    def get_classifier(self):
        return self.head.fc2

    def reset_classifier(self, num_classes=0, global_pool=None, head_fn=MlpClassifierHead):
        if global_pool is not None:
            self.global_pool = global_pool
        if num_classes > 0:
            if issubclass(head_fn, MlpClassifierHead):
                assert self.global_pool, 'Cannot disable global pooling with MLP head present.'
            self.head = head_fn(self.num_features, num_classes, pool_type=self.global_pool, drop=self.drop_rate)
        else:
            if self.global_pool:
                self.head = SelectAdaptivePool2d(pool_type=self.global_pool, flatten=True)
            else:
                self.head = nn.Identity()

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if pre_logits:
            if hasattr(self.head, 'global_pool'):
                x = self.head.global_pool(x)
            return x
        return self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc2',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'inception_next_tiny.sail_in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_tiny.pth',
    ),
    'inception_next_small.sail_in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_small.pth',
    ),
    'inception_next_base.sail_in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_base.pth',
        crop_pct=0.95,
    ),
    'inception_next_base.sail_in1k_384': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_base_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),
})


def _create_inception_next(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        MetaNeXt, variant, pretrained,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )
    return model


@register_model
def inception_next_tiny(pretrained=False, **kwargs):
    model_args = dict(
        depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
        token_mixers=InceptionDWConv2d,
    )
    return _create_inception_next('inception_next_tiny', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def inception_next_small(pretrained=False, **kwargs):
    model_args = dict(
        depths=(3, 3, 27, 3), dims=(96, 192, 384, 768),
        token_mixers=InceptionDWConv2d,
    )
    return _create_inception_next('inception_next_small', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def inception_next_base(pretrained=False, **kwargs):
    model_args = dict(
        depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024),
        token_mixers=InceptionDWConv2d,
    )
    return _create_inception_next('inception_next_base', pretrained=pretrained, **dict(model_args, **kwargs))
