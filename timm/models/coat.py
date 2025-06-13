"""
CoaT architecture.

Paper: Co-Scale Conv-Attentional Image Transformers - https://arxiv.org/abs/2104.06399

Official CoaT code at: https://github.com/mlpc-ucsd/CoaT

Modified from timm/models/vision_transformer.py
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, trunc_normal_, _assert, LayerNorm
from ._builder import build_model_with_cfg
from ._registry import register_model, generate_default_cfgs

__all__ = ['CoaT']


class ConvRelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """
    def __init__(self, head_chs, num_heads, window):
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                1. An integer of window size, which assigns all attention heads with the same window s
                    size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits (
                    e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                    It will apply different window size to the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: num_heads}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1
            # Determine padding size.
            # Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * head_chs,
                cur_head_split * head_chs,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * head_chs,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * head_chs for x in self.head_splits]

    def forward(self, q, v, size: Tuple[int, int]):
        B, num_heads, N, C = q.shape
        H, W = size
        _assert(N == 1 + H * W, '')

        # Convolutional relative position encoding.
        q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
        v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]

        v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
        conv_v_img_list = []
        for i, conv in enumerate(self.conv_list):
            conv_v_img_list.append(conv(v_img_list[i]))
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)

        EV_hat = q_img * conv_v_img
        EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
        return EV_hat


class FactorAttnConvRelPosEnc(nn.Module):
    """ Factorized attention with convolutional relative position encoding class. """
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
            shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, h, N, Ch]

        # Factorized attention.
        k_softmax = k.softmax(dim=2)
        factor_att = k_softmax.transpose(-1, -2) @ v
        factor_att = q @ factor_att

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)  # [B, h, N, Ch]

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ConvPosEnc(nn.Module):
    """ Convolutional Position Encoding.
        Note: This module is similar to the conditional position encoding in CPVT.
    """
    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim)

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        _assert(N == 1 + H * W, '')

        # Extract CLS token and image tokens.
        cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]

        # Depthwise convolution.
        feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)

        # Combine with CLS token.
        x = torch.cat((cls_token, x), dim=1)

        return x


class SerialBlock(nn.Module):
    """ Serial block class.
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module. """
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            shared_cpe=None,
            shared_crpe=None,
    ):
        super().__init__()

        # Conv-Attention.
        self.cpe = shared_cpe

        self.norm1 = norm_layer(dim)
        self.factoratt_crpe = FactorAttnConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            shared_crpe=shared_crpe,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP.
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x, size: Tuple[int, int]):
        # Conv-Attention.
        x = self.cpe(x, size)
        cur = self.norm1(x)
        cur = self.factoratt_crpe(cur, size)
        x = x + self.drop_path(cur)

        # MLP.
        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path(cur)

        return x


class ParallelBlock(nn.Module):
    """ Parallel block class. """
    def __init__(
            self,
            dims,
            num_heads,
            mlp_ratios=[],
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            shared_crpes=None,
    ):
        super().__init__()

        # Conv-Attention.
        self.norm12 = norm_layer(dims[1])
        self.norm13 = norm_layer(dims[2])
        self.norm14 = norm_layer(dims[3])
        self.factoratt_crpe2 = FactorAttnConvRelPosEnc(
            dims[1],
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            shared_crpe=shared_crpes[1],
        )
        self.factoratt_crpe3 = FactorAttnConvRelPosEnc(
            dims[2],
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            shared_crpe=shared_crpes[2],
        )
        self.factoratt_crpe4 = FactorAttnConvRelPosEnc(
            dims[3],
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            shared_crpe=shared_crpes[3],
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP.
        self.norm22 = norm_layer(dims[1])
        self.norm23 = norm_layer(dims[2])
        self.norm24 = norm_layer(dims[3])
        # In parallel block, we assume dimensions are the same and share the linear transformation.
        assert dims[1] == dims[2] == dims[3]
        assert mlp_ratios[1] == mlp_ratios[2] == mlp_ratios[3]
        mlp_hidden_dim = int(dims[1] * mlp_ratios[1])
        self.mlp2 = self.mlp3 = self.mlp4 = Mlp(
            in_features=dims[1],
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def upsample(self, x, factor: float, size: Tuple[int, int]):
        """ Feature map up-sampling. """
        return self.interpolate(x, scale_factor=factor, size=size)

    def downsample(self, x, factor: float, size: Tuple[int, int]):
        """ Feature map down-sampling. """
        return self.interpolate(x, scale_factor=1.0/factor, size=size)

    def interpolate(self, x, scale_factor: float, size: Tuple[int, int]):
        """ Feature map interpolation. """
        B, N, C = x.shape
        H, W = size
        _assert(N == 1 + H * W, '')

        cls_token = x[:, :1, :]
        img_tokens = x[:, 1:, :]

        img_tokens = img_tokens.transpose(1, 2).reshape(B, C, H, W)
        img_tokens = F.interpolate(
            img_tokens,
            scale_factor=scale_factor,
            recompute_scale_factor=False,
            mode='bilinear',
            align_corners=False,
        )
        img_tokens = img_tokens.reshape(B, C, -1).transpose(1, 2)

        out = torch.cat((cls_token, img_tokens), dim=1)

        return out

    def forward(self, x1, x2, x3, x4, sizes: List[Tuple[int, int]]):
        _, S2, S3, S4 = sizes
        cur2 = self.norm12(x2)
        cur3 = self.norm13(x3)
        cur4 = self.norm14(x4)
        cur2 = self.factoratt_crpe2(cur2, size=S2)
        cur3 = self.factoratt_crpe3(cur3, size=S3)
        cur4 = self.factoratt_crpe4(cur4, size=S4)
        upsample3_2 = self.upsample(cur3, factor=2., size=S3)
        upsample4_3 = self.upsample(cur4, factor=2., size=S4)
        upsample4_2 = self.upsample(cur4, factor=4., size=S4)
        downsample2_3 = self.downsample(cur2, factor=2., size=S2)
        downsample3_4 = self.downsample(cur3, factor=2., size=S3)
        downsample2_4 = self.downsample(cur2, factor=4., size=S2)
        cur2 = cur2 + upsample3_2 + upsample4_2
        cur3 = cur3 + upsample4_3 + downsample2_3
        cur4 = cur4 + downsample3_4 + downsample2_4
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)

        # MLP.
        cur2 = self.norm22(x2)
        cur3 = self.norm23(x3)
        cur4 = self.norm24(x4)
        cur2 = self.mlp2(cur2)
        cur3 = self.mlp3(cur3)
        cur4 = self.mlp4(cur4)
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)

        return x1, x2, x3, x4


class CoaT(nn.Module):
    """ CoaT class. """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dims=(64, 128, 320, 512),
            serial_depths=(3, 4, 6, 3),
            parallel_depth=0,
            num_heads=8,
            mlp_ratios=(4, 4, 4, 4),
            qkv_bias=True,
            drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=LayerNorm,
            return_interm_layers=False,
            out_features=None,
            crpe_window=None,
            global_pool='token',
    ):
        super().__init__()
        assert global_pool in ('token', 'avg')
        crpe_window = crpe_window or {3: 2, 5: 3, 7: 3}
        self.return_interm_layers = return_interm_layers
        self.out_features = out_features
        self.embed_dims = embed_dims
        self.num_features = self.head_hidden_size = embed_dims[-1]
        self.num_classes = num_classes
        self.global_pool = global_pool

        # Patch embeddings.
        img_size = to_2tuple(img_size)
        self.patch_embed1 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dims[0], norm_layer=nn.LayerNorm)
        self.patch_embed2 = PatchEmbed(
            img_size=[x // 4 for x in img_size], patch_size=2, in_chans=embed_dims[0],
            embed_dim=embed_dims[1], norm_layer=nn.LayerNorm)
        self.patch_embed3 = PatchEmbed(
            img_size=[x // 8 for x in img_size], patch_size=2, in_chans=embed_dims[1],
            embed_dim=embed_dims[2], norm_layer=nn.LayerNorm)
        self.patch_embed4 = PatchEmbed(
            img_size=[x // 16 for x in img_size], patch_size=2, in_chans=embed_dims[2],
            embed_dim=embed_dims[3], norm_layer=nn.LayerNorm)

        # Class tokens.
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # Convolutional position encodings.
        self.cpe1 = ConvPosEnc(dim=embed_dims[0], k=3)
        self.cpe2 = ConvPosEnc(dim=embed_dims[1], k=3)
        self.cpe3 = ConvPosEnc(dim=embed_dims[2], k=3)
        self.cpe4 = ConvPosEnc(dim=embed_dims[3], k=3)

        # Convolutional relative position encodings.
        self.crpe1 = ConvRelPosEnc(head_chs=embed_dims[0] // num_heads, num_heads=num_heads, window=crpe_window)
        self.crpe2 = ConvRelPosEnc(head_chs=embed_dims[1] // num_heads, num_heads=num_heads, window=crpe_window)
        self.crpe3 = ConvRelPosEnc(head_chs=embed_dims[2] // num_heads, num_heads=num_heads, window=crpe_window)
        self.crpe4 = ConvRelPosEnc(head_chs=embed_dims[3] // num_heads, num_heads=num_heads, window=crpe_window)

        # Disable stochastic depth.
        dpr = drop_path_rate
        assert dpr == 0.0
        skwargs = dict(
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr,
            norm_layer=norm_layer,
        )

        # Serial blocks 1.
        self.serial_blocks1 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[0],
                mlp_ratio=mlp_ratios[0],
                shared_cpe=self.cpe1,
                shared_crpe=self.crpe1,
                **skwargs,
            )
            for _ in range(serial_depths[0])]
        )

        # Serial blocks 2.
        self.serial_blocks2 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[1],
                mlp_ratio=mlp_ratios[1],
                shared_cpe=self.cpe2,
                shared_crpe=self.crpe2,
                **skwargs,
            )
            for _ in range(serial_depths[1])]
        )

        # Serial blocks 3.
        self.serial_blocks3 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[2],
                mlp_ratio=mlp_ratios[2],
                shared_cpe=self.cpe3,
                shared_crpe=self.crpe3,
                **skwargs,
            )
            for _ in range(serial_depths[2])]
        )

        # Serial blocks 4.
        self.serial_blocks4 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[3],
                mlp_ratio=mlp_ratios[3],
                shared_cpe=self.cpe4,
                shared_crpe=self.crpe4,
                **skwargs,
            )
            for _ in range(serial_depths[3])]
        )

        # Parallel blocks.
        self.parallel_depth = parallel_depth
        if self.parallel_depth > 0:
            self.parallel_blocks = nn.ModuleList([
                ParallelBlock(
                    dims=embed_dims,
                    mlp_ratios=mlp_ratios,
                    shared_crpes=(self.crpe1, self.crpe2, self.crpe3, self.crpe4),
                    **skwargs,
                )
                for _ in range(parallel_depth)]
            )
        else:
            self.parallel_blocks = None

        # Classification head(s).
        if not self.return_interm_layers:
            if self.parallel_blocks is not None:
                self.norm2 = norm_layer(embed_dims[1])
                self.norm3 = norm_layer(embed_dims[2])
            else:
                self.norm2 = self.norm3 = None
            self.norm4 = norm_layer(embed_dims[3])

            if self.parallel_depth > 0:
                # CoaT series: Aggregate features of last three scales for classification.
                assert embed_dims[1] == embed_dims[2] == embed_dims[3]
                self.aggregate = torch.nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
                self.head_drop = nn.Dropout(drop_rate)
                self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            else:
                # CoaT-Lite series: Use feature of last scale for classification.
                self.aggregate = None
                self.head_drop = nn.Dropout(drop_rate)
                self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights.
        trunc_normal_(self.cls_token1, std=.02)
        trunc_normal_(self.cls_token2, std=.02)
        trunc_normal_(self.cls_token3, std=.02)
        trunc_normal_(self.cls_token4, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token1', 'cls_token2', 'cls_token3', 'cls_token4'}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem1=r'^cls_token1|patch_embed1|crpe1|cpe1',
            serial_blocks1=r'^serial_blocks1\.(\d+)',
            stem2=r'^cls_token2|patch_embed2|crpe2|cpe2',
            serial_blocks2=r'^serial_blocks2\.(\d+)',
            stem3=r'^cls_token3|patch_embed3|crpe3|cpe3',
            serial_blocks3=r'^serial_blocks3\.(\d+)',
            stem4=r'^cls_token4|patch_embed4|crpe4|cpe4',
            serial_blocks4=r'^serial_blocks4\.(\d+)',
            parallel_blocks=[  # FIXME (partially?) overlap parallel w/ serial blocks??
                (r'^parallel_blocks\.(\d+)', None),
                (r'^norm|aggregate', (99999,)),
            ]
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('token', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x0):
        B = x0.shape[0]

        # Serial blocks 1.
        x1 = self.patch_embed1(x0)
        H1, W1 = self.patch_embed1.grid_size
        x1 = insert_cls(x1, self.cls_token1)
        for blk in self.serial_blocks1:
            x1 = blk(x1, size=(H1, W1))
        x1_nocls = remove_cls(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        # Serial blocks 2.
        x2 = self.patch_embed2(x1_nocls)
        H2, W2 = self.patch_embed2.grid_size
        x2 = insert_cls(x2, self.cls_token2)
        for blk in self.serial_blocks2:
            x2 = blk(x2, size=(H2, W2))
        x2_nocls = remove_cls(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        # Serial blocks 3.
        x3 = self.patch_embed3(x2_nocls)
        H3, W3 = self.patch_embed3.grid_size
        x3 = insert_cls(x3, self.cls_token3)
        for blk in self.serial_blocks3:
            x3 = blk(x3, size=(H3, W3))
        x3_nocls = remove_cls(x3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()

        # Serial blocks 4.
        x4 = self.patch_embed4(x3_nocls)
        H4, W4 = self.patch_embed4.grid_size
        x4 = insert_cls(x4, self.cls_token4)
        for blk in self.serial_blocks4:
            x4 = blk(x4, size=(H4, W4))
        x4_nocls = remove_cls(x4).reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()

        # Only serial blocks: Early return.
        if self.parallel_blocks is None:
            if not torch.jit.is_scripting() and self.return_interm_layers:
                # Return intermediate features for down-stream tasks (e.g. Deformable DETR and Detectron2).
                feat_out = {}
                if 'x1_nocls' in self.out_features:
                    feat_out['x1_nocls'] = x1_nocls
                if 'x2_nocls' in self.out_features:
                    feat_out['x2_nocls'] = x2_nocls
                if 'x3_nocls' in self.out_features:
                    feat_out['x3_nocls'] = x3_nocls
                if 'x4_nocls' in self.out_features:
                    feat_out['x4_nocls'] = x4_nocls
                return feat_out
            else:
                # Return features for classification.
                x4 = self.norm4(x4)
                return x4

        # Parallel blocks.
        for blk in self.parallel_blocks:
            x2, x3, x4 = self.cpe2(x2, (H2, W2)), self.cpe3(x3, (H3, W3)), self.cpe4(x4, (H4, W4))
            x1, x2, x3, x4 = blk(x1, x2, x3, x4, sizes=[(H1, W1), (H2, W2), (H3, W3), (H4, W4)])

        if not torch.jit.is_scripting() and self.return_interm_layers:
            # Return intermediate features for down-stream tasks (e.g. Deformable DETR and Detectron2).
            feat_out = {}
            if 'x1_nocls' in self.out_features:
                x1_nocls = remove_cls(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x1_nocls'] = x1_nocls
            if 'x2_nocls' in self.out_features:
                x2_nocls = remove_cls(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x2_nocls'] = x2_nocls
            if 'x3_nocls' in self.out_features:
                x3_nocls = remove_cls(x3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x3_nocls'] = x3_nocls
            if 'x4_nocls' in self.out_features:
                x4_nocls = remove_cls(x4).reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x4_nocls'] = x4_nocls
            return feat_out
        else:
            x2 = self.norm2(x2)
            x3 = self.norm3(x3)
            x4 = self.norm4(x4)
            return [x2, x3, x4]

    def forward_head(self, x_feat: Union[torch.Tensor, List[torch.Tensor]], pre_logits: bool = False):
        if isinstance(x_feat, list):
            assert self.aggregate is not None
            if self.global_pool == 'avg':
                x = torch.cat([xl[:, 1:].mean(dim=1, keepdim=True) for xl in x_feat], dim=1)  # [B, 3, C]
            else:
                x = torch.stack([xl[:, 0] for xl in x_feat], dim=1)  # [B, 3, C]
            x = self.aggregate(x).squeeze(dim=1)  # Shape: [B, C]
        else:
            x = x_feat[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x_feat[:, 0]
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x) -> torch.Tensor:
        if not torch.jit.is_scripting() and self.return_interm_layers:
            # Return intermediate features (for down-stream tasks).
            return self.forward_features(x)
        else:
            # Return features for classification.
            x_feat = self.forward_features(x)
            x = self.forward_head(x_feat)
            return x


def insert_cls(x, cls_token):
    """ Insert CLS token. """
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    return x


def remove_cls(x):
    """ Remove CLS token. """
    return x[:, 1:, :]


def checkpoint_filter_fn(state_dict, model):
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    for k, v in state_dict.items():
        # original model had unused norm layers, removing them requires filtering pretrained checkpoints
        if k.startswith('norm1') or \
                (k.startswith('norm2') and getattr(model, 'norm2', None) is None) or \
                (k.startswith('norm3') and getattr(model, 'norm3', None) is None) or \
                (k.startswith('norm4') and getattr(model, 'norm4', None) is None) or \
                (k.startswith('aggregate') and getattr(model, 'aggregate', None) is None) or \
                (k.startswith('head') and getattr(model, 'head', None) is None):
            continue
        out_dict[k] = v
    return out_dict


def _create_coat(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        CoaT,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    )
    return model


def _cfg_coat(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed1.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'coat_tiny.in1k': _cfg_coat(hf_hub_id='timm/'),
    'coat_mini.in1k': _cfg_coat(hf_hub_id='timm/'),
    'coat_small.in1k': _cfg_coat(hf_hub_id='timm/'),
    'coat_lite_tiny.in1k': _cfg_coat(hf_hub_id='timm/'),
    'coat_lite_mini.in1k': _cfg_coat(hf_hub_id='timm/'),
    'coat_lite_small.in1k': _cfg_coat(hf_hub_id='timm/'),
    'coat_lite_medium.in1k': _cfg_coat(hf_hub_id='timm/'),
    'coat_lite_medium_384.in1k': _cfg_coat(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0, crop_mode='squash',
    ),
})


@register_model
def coat_tiny(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        patch_size=4, embed_dims=[152, 152, 152, 152], serial_depths=[2, 2, 2, 2], parallel_depth=6)
    model = _create_coat('coat_tiny', pretrained=pretrained, **dict(model_cfg, **kwargs))
    return model


@register_model
def coat_mini(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        patch_size=4, embed_dims=[152, 216, 216, 216], serial_depths=[2, 2, 2, 2], parallel_depth=6)
    model = _create_coat('coat_mini', pretrained=pretrained, **dict(model_cfg, **kwargs))
    return model


@register_model
def coat_small(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        patch_size=4, embed_dims=[152, 320, 320, 320], serial_depths=[2, 2, 2, 2], parallel_depth=6, **kwargs)
    model = _create_coat('coat_small', pretrained=pretrained, **dict(model_cfg, **kwargs))
    return model


@register_model
def coat_lite_tiny(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        patch_size=4, embed_dims=[64, 128, 256, 320], serial_depths=[2, 2, 2, 2], mlp_ratios=[8, 8, 4, 4])
    model = _create_coat('coat_lite_tiny', pretrained=pretrained, **dict(model_cfg, **kwargs))
    return model


@register_model
def coat_lite_mini(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        patch_size=4, embed_dims=[64, 128, 320, 512], serial_depths=[2, 2, 2, 2], mlp_ratios=[8, 8, 4, 4])
    model = _create_coat('coat_lite_mini', pretrained=pretrained, **dict(model_cfg, **kwargs))
    return model


@register_model
def coat_lite_small(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        patch_size=4, embed_dims=[64, 128, 320, 512], serial_depths=[3, 4, 6, 3], mlp_ratios=[8, 8, 4, 4])
    model = _create_coat('coat_lite_small', pretrained=pretrained, **dict(model_cfg, **kwargs))
    return model


@register_model
def coat_lite_medium(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        patch_size=4, embed_dims=[128, 256, 320, 512], serial_depths=[3, 6, 10, 8])
    model = _create_coat('coat_lite_medium', pretrained=pretrained, **dict(model_cfg, **kwargs))
    return model


@register_model
def coat_lite_medium_384(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        img_size=384, patch_size=4, embed_dims=[128, 256, 320, 512], serial_depths=[3, 6, 10, 8])
    model = _create_coat('coat_lite_medium_384', pretrained=pretrained, **dict(model_cfg, **kwargs))
    return model