""" RepViT

Paper: `RepViT: Revisiting Mobile CNN From ViT Perspective`
    - https://arxiv.org/abs/2307.09283

@misc{wang2023repvit,
      title={RepViT: Revisiting Mobile CNN From ViT Perspective}, 
      author={Ao Wang and Hui Chen and Zijia Lin and Hengjun Pu and Guiguang Ding},
      year={2023},
      eprint={2307.09283},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Adapted from official impl at https://github.com/jameslahm/RepViT
"""

__all__ = ['RepVit']

import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ._registry import register_model, generate_default_cfgs
from ._builder import build_model_with_cfg
from timm.layers import SqueezeExcite, trunc_normal_, to_ntuple, to_2tuple
from ._manipulate import checkpoint_seq

import torch


class ConvNorm(nn.Sequential):
    def __init__(self, in_dim, out_dim, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(in_dim, out_dim, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_dim))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            device=c.weight.device,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class NormLinear(nn.Sequential):
    def __init__(self, in_dim, out_dim, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', nn.BatchNorm1d(in_dim))
        self.add_module('l', nn.Linear(in_dim, out_dim, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepVggDw(nn.Module):
    def __init__(self, ed, kernel_size, legacy=False):
        super().__init__()
        self.conv = ConvNorm(ed, ed, kernel_size, 1, (kernel_size - 1) // 2, groups=ed)
        if legacy:
            self.conv1 = ConvNorm(ed, ed, 1, 1, 0, groups=ed)
            # Make torchscript happy.
            self.bn = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
            self.bn = nn.BatchNorm2d(ed)
        self.dim = ed
        self.legacy = legacy

    def forward(self, x):
        return self.bn(self.conv(x) + self.conv1(x) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()

        if self.legacy:
            conv1 = self.conv1.fuse()
        else:
            conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = nn.functional.pad(
            torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1, 1, 1, 1]
        )

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        if not self.legacy:
            bn = self.bn
            w = bn.weight / (bn.running_var + bn.eps) ** 0.5
            w = conv.weight * w[:, None, None, None]
            b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / (bn.running_var + bn.eps) ** 0.5
            conv.weight.data.copy_(w)
            conv.bias.data.copy_(b)
        return conv


class RepVitMlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, act_layer):
        super().__init__()
        self.conv1 = ConvNorm(in_dim, hidden_dim, 1, 1, 0)
        self.act = act_layer()
        self.conv2 = ConvNorm(hidden_dim, in_dim, 1, 1, 0, bn_weight_init=0)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class RepViTBlock(nn.Module):
    def __init__(self, in_dim, mlp_ratio, kernel_size, use_se, act_layer, legacy=False):
        super(RepViTBlock, self).__init__()

        self.token_mixer = RepVggDw(in_dim, kernel_size, legacy)
        self.se = SqueezeExcite(in_dim, 0.25) if use_se else nn.Identity()
        self.channel_mixer = RepVitMlp(in_dim, in_dim * mlp_ratio, act_layer)

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.se(x)
        identity = x
        x = self.channel_mixer(x)
        return identity + x


class RepVitStem(nn.Module):
    def __init__(self, in_chs, out_chs, act_layer):
        super().__init__()
        self.conv1 = ConvNorm(in_chs, out_chs // 2, 3, 2, 1)
        self.act1 = act_layer()
        self.conv2 = ConvNorm(out_chs // 2, out_chs, 3, 2, 1)
        self.stride = 4

    def forward(self, x):
        return self.conv2(self.act1(self.conv1(x)))


class RepVitDownsample(nn.Module):
    def __init__(self, in_dim, mlp_ratio, out_dim, kernel_size, act_layer, legacy=False):
        super().__init__()
        self.pre_block = RepViTBlock(in_dim, mlp_ratio, kernel_size, use_se=False, act_layer=act_layer, legacy=legacy)
        self.spatial_downsample = ConvNorm(in_dim, in_dim, kernel_size, 2, (kernel_size - 1) // 2, groups=in_dim)
        self.channel_downsample = ConvNorm(in_dim, out_dim, 1, 1)
        self.ffn = RepVitMlp(out_dim, out_dim * mlp_ratio, act_layer)

    def forward(self, x):
        x = self.pre_block(x)
        x = self.spatial_downsample(x)
        x = self.channel_downsample(x)
        identity = x
        x = self.ffn(x)
        return x + identity


class RepVitClassifier(nn.Module):
    def __init__(self, dim, num_classes, distillation=False, drop=0.0):
        super().__init__()
        self.head_drop = nn.Dropout(drop)
        self.head = NormLinear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.distillation = distillation
        self.distilled_training = False
        self.num_classes = num_classes
        if distillation:
            self.head_dist = NormLinear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.head_drop(x)
        if self.distillation:
            x1, x2 = self.head(x), self.head_dist(x)
            if self.training and self.distilled_training and not torch.jit.is_scripting():
                return x1, x2
            else:
                return (x1 + x2) / 2
        else:
            x = self.head(x)
            return x

    @torch.no_grad()
    def fuse(self):
        if not self.num_classes > 0:
            return nn.Identity()
        head = self.head.fuse()
        if self.distillation:
            head_dist = self.head_dist.fuse()
            head.weight += head_dist.weight
            head.bias += head_dist.bias
            head.weight /= 2
            head.bias /= 2
            return head
        else:
            return head


class RepVitStage(nn.Module):
    def __init__(self, in_dim, out_dim, depth, mlp_ratio, act_layer, kernel_size=3, downsample=True, legacy=False):
        super().__init__()
        if downsample:
            self.downsample = RepVitDownsample(in_dim, mlp_ratio, out_dim, kernel_size, act_layer, legacy)
        else:
            assert in_dim == out_dim
            self.downsample = nn.Identity()

        blocks = []
        use_se = True
        for _ in range(depth):
            blocks.append(RepViTBlock(out_dim, mlp_ratio, kernel_size, use_se, act_layer, legacy))
            use_se = not use_se

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class RepVit(nn.Module):
    def __init__(
        self,
        in_chans=3,
        img_size=224,
        embed_dim=(48,),
        depth=(2,),
        mlp_ratio=2,
        global_pool='avg',
        kernel_size=3,
        num_classes=1000,
        act_layer=nn.GELU,
        distillation=True,
        drop_rate=0.0,
        legacy=False,
    ):
        super(RepVit, self).__init__()
        self.grad_checkpointing = False
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        in_dim = embed_dim[0]
        self.stem = RepVitStem(in_chans, in_dim, act_layer)
        stride = self.stem.stride
        resolution = tuple([i // p for i, p in zip(to_2tuple(img_size), to_2tuple(stride))])

        num_stages = len(embed_dim)
        mlp_ratios = to_ntuple(num_stages)(mlp_ratio)

        self.feature_info = []
        stages = []
        for i in range(num_stages):
            downsample = True if i != 0 else False
            stages.append(
                RepVitStage(
                    in_dim,
                    embed_dim[i],
                    depth[i],
                    mlp_ratio=mlp_ratios[i],
                    act_layer=act_layer,
                    kernel_size=kernel_size,
                    downsample=downsample,
                    legacy=legacy,
                )
            )
            stage_stride = 2 if downsample else 1
            stride *= stage_stride
            resolution = tuple([(r - 1) // stage_stride + 1 for r in resolution])
            self.feature_info += [dict(num_chs=embed_dim[i], reduction=stride, module=f'stages.{i}')]
            in_dim = embed_dim[i]
        self.stages = nn.Sequential(*stages)

        self.num_features = embed_dim[-1]
        self.head_drop = nn.Dropout(drop_rate)
        self.head = RepVitClassifier(embed_dim[-1], num_classes, distillation)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^stem', blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))])  # stem and embed
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None, distillation=False):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = (
            RepVitClassifier(self.embed_dim[-1], num_classes, distillation) if num_classes > 0 else nn.Identity()
        )

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.head.distilled_training = enable

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean((2, 3), keepdim=False)
        x = self.head_drop(x)
        return self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @torch.no_grad()
    def fuse(self):
        def fuse_children(net):
            for child_name, child in net.named_children():
                if hasattr(child, 'fuse'):
                    fused = child.fuse()
                    setattr(net, child_name, fused)
                    fuse_children(fused)
                else:
                    fuse_children(child)

        fuse_children(self)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': (7, 7),
        'crop_pct': 0.95,
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1.c',
        'classifier': ('head.head.l', 'head.head_dist.l'),
        **kwargs,
    }


default_cfgs = generate_default_cfgs(
    {
        'repvit_m1.dist_in1k': _cfg(
            hf_hub_id='timm/',
        ),
        'repvit_m2.dist_in1k': _cfg(
            hf_hub_id='timm/',
        ),
        'repvit_m3.dist_in1k': _cfg(
            hf_hub_id='timm/',
        ),
        'repvit_m0_9.dist_300e_in1k': _cfg(
            hf_hub_id='timm/',
        ),
        'repvit_m0_9.dist_450e_in1k': _cfg(
            hf_hub_id='timm/',
        ),
        'repvit_m1_0.dist_300e_in1k': _cfg(
            hf_hub_id='timm/',
        ),
        'repvit_m1_0.dist_450e_in1k': _cfg(
            hf_hub_id='timm/',
        ),
        'repvit_m1_1.dist_300e_in1k': _cfg(
            hf_hub_id='timm/',
        ),
        'repvit_m1_1.dist_450e_in1k': _cfg(
            hf_hub_id='timm/',
        ),
        'repvit_m1_5.dist_300e_in1k': _cfg(
            hf_hub_id='timm/',
        ),
        'repvit_m1_5.dist_450e_in1k': _cfg(
            hf_hub_id='timm/',
        ),
        'repvit_m2_3.dist_300e_in1k': _cfg(
            hf_hub_id='timm/',
        ),
        'repvit_m2_3.dist_450e_in1k': _cfg(
            hf_hub_id='timm/',
        ),
    }
)


def _create_repvit(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', (0, 1, 2, 3))
    model = build_model_with_cfg(
        RepVit,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs,
    )
    return model


@register_model
def repvit_m1(pretrained=False, **kwargs):
    """
    Constructs a RepViT-M1 model
    """
    model_args = dict(embed_dim=(48, 96, 192, 384), depth=(2, 2, 14, 2), legacy=True)
    return _create_repvit('repvit_m1', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def repvit_m2(pretrained=False, **kwargs):
    """
    Constructs a RepViT-M2 model
    """
    model_args = dict(embed_dim=(64, 128, 256, 512), depth=(2, 2, 12, 2), legacy=True)
    return _create_repvit('repvit_m2', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def repvit_m3(pretrained=False, **kwargs):
    """
    Constructs a RepViT-M3 model
    """
    model_args = dict(embed_dim=(64, 128, 256, 512), depth=(4, 4, 18, 2), legacy=True)
    return _create_repvit('repvit_m3', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def repvit_m0_9(pretrained=False, **kwargs):
    """
    Constructs a RepViT-M0.9 model
    """
    model_args = dict(embed_dim=(48, 96, 192, 384), depth=(2, 2, 14, 2))
    return _create_repvit('repvit_m0_9', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def repvit_m1_0(pretrained=False, **kwargs):
    """
    Constructs a RepViT-M1.0 model
    """
    model_args = dict(embed_dim=(56, 112, 224, 448), depth=(2, 2, 14, 2))
    return _create_repvit('repvit_m1_0', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def repvit_m1_1(pretrained=False, **kwargs):
    """
    Constructs a RepViT-M1.1 model
    """
    model_args = dict(embed_dim=(64, 128, 256, 512), depth=(2, 2, 12, 2))
    return _create_repvit('repvit_m1_1', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def repvit_m1_5(pretrained=False, **kwargs):
    """
    Constructs a RepViT-M1.5 model
    """
    model_args = dict(embed_dim=(64, 128, 256, 512), depth=(4, 4, 24, 4))
    return _create_repvit('repvit_m1_5', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def repvit_m2_3(pretrained=False, **kwargs):
    """
    Constructs a RepViT-M2.3 model
    """
    model_args = dict(embed_dim=(80, 160, 320, 640), depth=(6, 6, 34, 2))
    return _create_repvit('repvit_m2_3', pretrained=pretrained, **dict(model_args, **kwargs))
