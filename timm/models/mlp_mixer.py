""" MLP-Mixer in PyTorch

Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601

NOTE this is a very early stage first run through, the param counts aren't matching paper so
something is up...
"""
from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg, overlay_external_default_cfg
from .layers import DropPath, to_2tuple, trunc_normal_, lecun_normal_
from .registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_grid = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class MixerBlock(nn.Module):

    def __init__(
            self, dim, seq_len, tokens_dim, channels_dim,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mlp_token = Mlp(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = Mlp(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp_token(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class MlpMixer(nn.Module):

    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            hidden_dim=512,
            tokens_dim=256,
            channels_dim=2048,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop=0.,
            drop_path=0.,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.stem = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=hidden_dim)
        # FIXME drop_path (stochastic depth scaling rule?)
        self.blocks = nn.Sequential(*[
            MixerBlock(
                hidden_dim, self.stem.num_patches, tokens_dim, channels_dim,
                norm_layer=norm_layer, act_layer=act_layer, drop=drop, drop_path=drop_path)
            for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, self.num_classes)  # zero init

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


@register_model
def mixer_small_p16(pretrained=False, **kwargs):
    model = MlpMixer()
    model.default_cfg = _cfg()
    return model


@register_model
def mixer_base_p16(pretrained=False, **kwargs):
    model = MlpMixer(num_blocks=12, hidden_dim=768, tokens_dim=384, channels_dim=3072)
    model.default_cfg = _cfg()
    return model