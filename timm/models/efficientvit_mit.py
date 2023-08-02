""" EfficientViT (by MIT Song Han's Lab)

Paper: `Efficientvit: Enhanced linear attention for high-resolution low-computation visual recognition`
    - https://arxiv.org/abs/2205.14756

Adapted from official impl at https://github.com/mit-han-lab/efficientvit
"""

__all__ = ['EfficientViT']

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ._registry import register_model, generate_default_cfgs
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from timm.layers import SelectAdaptivePool2d
from collections import OrderedDict


def val2list(x: list or tuple or any, repeat_time=1):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1):
    # repeat elements if necessary
    x = val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm=nn.BatchNorm2d,
        act_func=nn.ReLU,
    ):
        super(ConvNormAct, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func(inplace=True) if act_func else None

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=(nn.BatchNorm2d, nn.BatchNorm2d),
        act_func=(nn.ReLU6, None),
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvNormAct(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=(nn.BatchNorm2d, nn.BatchNorm2d, nn.BatchNorm2d),
        act_func=(nn.ReLU6, nn.ReLU6, None),
    ):
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvNormAct(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvNormAct(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvNormAct(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x):
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class LiteMSA(nn.Module):
    """Lightweight multi-scale attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, nn.BatchNorm2d),
        act_func=(None, None),
        kernel_func=nn.ReLU,
        scales=(5,),
    ):
        super(LiteMSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvNormAct(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = kernel_func(inplace=False)

        self.proj = ConvNormAct(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x):
        B, _, H, W = list(x.size())

        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        multi_scale_qkv = torch.reshape(
            multi_scale_qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)
        q, k, v = (
            multi_scale_qkv[..., 0: self.dim],
            multi_scale_qkv[..., self.dim: 2 * self.dim],
            multi_scale_qkv[..., 2 * self.dim:],
        )

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)

        # final projecttion
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        out = self.proj(out)

        return out


class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        heads_ratio=1.0,
        dim=32,
        expand_ratio=4,
        norm=nn.BatchNorm2d,
        act_func=nn.Hardswish,
    ):
        super(EfficientViTBlock, self).__init__()
        self.context_module = ResidualBlock(
            LiteMSA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
            ),
            nn.Identity(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, nn.Identity())

    def forward(self, x):
        x = self.context_module(x)
        x = self.local_module(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: nn.Module or None,
        shortcut: nn.Module or None,
        post_act=None,
        pre_norm: nn.Module or None = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = post_act(inplace=True) if post_act else nn.Identity()

    def forward_main(self, x):
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x):
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


class ClsHead(nn.Module):
    def __init__(
        self,
        in_channels,
        width_list,
        n_classes=1000,
        dropout=0,
        norm=nn.BatchNorm2d,
        act_func=nn.Hardswish,
        global_pool='avg',
    ):
        super(ClsHead, self).__init__()
        self.ops = nn.Sequential(
            ConvNormAct(in_channels, width_list[0], 1, norm=norm, act_func=act_func),
            SelectAdaptivePool2d(pool_type=global_pool, flatten=True, input_fmt='NCHW'),
            nn.Linear(width_list[0], width_list[1], bias=False),
            nn.LayerNorm(width_list[1]),
            act_func(inplace=True),
            nn.Dropout(dropout, inplace=False) if dropout else nn.Identity(),
            nn.Linear(width_list[1], n_classes, bias=True),
        )

    def forward(self, x):
        x = self.ops(x)
        return x


class EfficientViT(nn.Module):
    def __init__(
        self,
        in_chans=3,
        width_list=[],
        depth_list=[],
        dim=32,
        expand_ratio=4,
        norm=nn.BatchNorm2d,
        act_func=nn.Hardswish,
        global_pool='avg',
        head_width_list=[],
        head_dropout=0.0,
        num_classes=1000,
    ):
        super(EfficientViT, self).__init__()
        self.grad_checkpointing = False
        self.global_pool = global_pool
        # input stem
        input_stem = [
            ('in_conv', ConvNormAct(
                in_channels=3,
                out_channels=width_list[0],
                kernel_size=3,
                stride=2,
                norm=norm,
                act_func=act_func,
            ))
        ]
        stem_block = 0
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            input_stem.append((f'res{stem_block}', ResidualBlock(block, nn.Identity())))
            stem_block += 1
        in_channels = width_list[0]
        self.stem = nn.Sequential(OrderedDict(input_stem))
        stride = 2
        self.feature_info = []
        stages = []
        stage_idx = 0
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stage_stride = 2 if i == 0 else 1
                stride *= stage_stride
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stage_stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, nn.Identity() if stage_stride == 1 else None)
                stage.append(block)
                in_channels = w
            stages.append(nn.Sequential(*stage))
            self.feature_info += [dict(num_chs=in_channels, reduction=stride, module=f'stages.{stage_idx}')]
            stage_idx += 1

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            stages.append(nn.Sequential(*stage))
            stride *= 2
            self.feature_info += [dict(num_chs=in_channels, reduction=stride, module=f'stages.{stage_idx}')]
            stage_idx += 1

        self.stages = nn.Sequential(*stages)
        self.num_features = in_channels
        self.head_width_list = head_width_list
        self.head_dropout = head_dropout
        if num_classes > 0:
            self.head = ClsHead(self.num_features, self.head_width_list, n_classes=num_classes, dropout=self.head_dropout, global_pool=self.global_pool)
        else:
            if global_pool is not None:
                self.head = SelectAdaptivePool2d(pool_type=global_pool, flatten=True, input_fmt='NCHW')
            else:
                self.head = nn.Identity()

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ):
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',  # stem and embed
            blocks=[(r'^stages\.(\d+)', None)]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None, dropout=0):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        if num_classes > 0:
            self.head = ClsHead(self.num_features, self.head_width_list, n_classes=num_classes, dropout=self.head_dropout, global_pool=global_pool)
        else:
            if global_pool is not None:
                self.head = SelectAdaptivePool2d(pool_type=global_pool, flatten=True, input_fmt='NCHW')
            else:
                self.head = nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    target_keys = list(model.state_dict().keys())
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    out_dict = {}
    for i, (k, v) in enumerate(state_dict.items()):
        out_dict[target_keys[i]] = v
    return out_dict


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.in_conv.conv',
        'classifier': 'head',
        **kwargs,
    }


default_cfgs = generate_default_cfgs(
    {
        'efficientvit_b0.r224_in1k': _cfg(
            # url='https://drive.google.com/file/d/1ganFBZmmvCTpgUwiLb8ePD6NBNxRyZDk/view?usp=drive_link'
        ),
        'efficientvit_b1.r224_in1k': _cfg(
            # url='https://drive.google.com/file/d/1hKN_hvLG4nmRzbfzKY7GlqwpR5uKpOOk/view?usp=share_link'
        ),
        'efficientvit_b1.r256_in1k': _cfg(
            # url='https://drive.google.com/file/d/1hXcG_jB0ODMOESsSkzVye-58B4F3Cahs/view?usp=share_link'
        ),
        'efficientvit_b1.r288_in1k': _cfg(
            # url='https://drive.google.com/file/d/1sE_Suz9gOOUO7o5r9eeAT4nKK8Hrbhsu/view?usp=share_link'
        ),
        'efficientvit_b2.r224_in1k': _cfg(
            # url='https://drive.google.com/file/d/1DiM-iqVGTrq4te8mefHl3e1c12u4qR7d/view?usp=share_link'
        ),
        'efficientvit_b2.r256_in1k': _cfg(
            # url='https://drive.google.com/file/d/192OOk4ISitwlyW979M-FSJ_fYMMW9HQz/view?usp=share_link'
        ),
        'efficientvit_b2.r288_in1k': _cfg(
            # url='https://drive.google.com/file/d/1aodcepOyne667hvBAGpf9nDwmd5g0NpU/view?usp=share_link'
        ),
        'efficientvit_b3.r224_in1k': _cfg(
            # url='https://drive.google.com/file/d/18RZDGLiY8KsyJ7LGic4mg1JHwd-a_ky6/view?usp=share_link'
        ),
        'efficientvit_b3.r256_in1k': _cfg(
            # url='https://drive.google.com/file/d/1y1rnir4I0XiId-oTCcHhs7jqnrHGFi-g/view?usp=share_link'
        ),
        'efficientvit_b3.r288_in1k': _cfg(
            # url='https://drive.google.com/file/d/1KfwbGtlyFgslNr4LIHERv6aCfkItEvRk/view?usp=share_link'
        ),
    }
)


def _create_efficientvit(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', (0, 1, 2, 3))
    model = build_model_with_cfg(
        EfficientViT,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs
    )
    return model


@register_model
def efficientvit_b0(pretrained=False, **kwargs):
    model_args = dict(width_list=[8, 16, 32, 64, 128], depth_list=[1, 2, 2, 2, 2], dim=16, head_width_list=[1024, 1280])
    return _create_efficientvit('efficientvit_b0', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def efficientvit_b1(pretrained=False, **kwargs):
    model_args = dict(width_list=[16, 32, 64, 128, 256], depth_list=[1, 2, 3, 3, 4], dim=16, head_width_list=[1536, 1600])
    return _create_efficientvit('efficientvit_b0', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def efficientvit_b2(pretrained=False, **kwargs):
    model_args = dict(width_list=[24, 48, 96, 192, 384], depth_list=[1, 3, 4, 4, 6], dim=32, head_width_list=[2304, 2560])
    return _create_efficientvit('efficientvit_b0', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def efficientvit_b3(pretrained=False, **kwargs):
    model_args = dict(width_list=[32, 64, 128, 256, 512], depth_list=[1, 4, 6, 6, 9], dim=32, head_width_list=[2304, 2560])
    return _create_efficientvit('efficientvit_b0', pretrained=pretrained, **dict(model_args, **kwargs))
