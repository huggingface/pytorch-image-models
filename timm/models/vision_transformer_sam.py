""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'Exploring Plain Vision Transformer Backbones for Object Detection'
    - https://arxiv.org/abs/2203.16527

'Segment Anything Model (SAM)'
    - https://github.com/facebookresearch/segment-anything/

"""
import logging
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import PatchEmbed, Mlp, DropPath, PatchDropout, LayerNorm2d, ClassifierHead, NormMlpClassifierHead, \
    Format, resample_abs_pos_embed_nhwc, RotaryEmbeddingCat, apply_rot_embed_cat, to_2tuple, use_fused_attn
from torch.jit import Final

from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._features_fx import register_notrace_function
from ._manipulate import checkpoint, checkpoint_seq
from ._registry import generate_default_cfgs, register_model

# model_registry will add each entrypoint fn to this
__all__ = ['VisionTransformerSAM']


_logger = logging.getLogger(__name__)


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

register_notrace_function(get_rel_pos)


def get_decomposed_rel_pos_bias(
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py
    Args:
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        bias (Tensor): attention bias to add to attention map
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn_bias = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    return attn_bias.reshape(-1, q_h * q_w, k_h * k_w)


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            use_rel_pos: bool = False,
            input_size: Optional[Tuple[int, int]] = None,
            rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert rope is None
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(
                2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(
                2 * input_size[1] - 1, self.head_dim))
        self.rope = rope

    def forward(self, x):
        B, H, W, _ = x.shape
        N = H * W
        x = x.reshape(B, N, -1)
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # qkv with shape (3, B, nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_rel_pos:
            attn_bias = get_decomposed_rel_pos_bias(q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        else:
            attn_bias = None
            if self.rope is not None:
                rope = self.rope.get_embed()
                q = apply_rot_embed_cat(q, rope).type_as(v)
                k = apply_rot_embed_cat(k, rope).type_as(v)

        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.view(B, self.num_heads, N, -1).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, H, W, -1)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            use_rel_pos=False,
            window_size=0,
            input_size=None,
            rope=None,
    ):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, H, W, _ = x.shape

        shortcut = x
        x = self.norm1(x)
        # Window partition
        pad_hw: Optional[Tuple[int, int]] = None
        if self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)

        x = self.drop_path1(self.ls1(self.attn(x)))

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, (H, W), pad_hw)

        x = shortcut + x

        x = x.reshape(B, H * W, -1)  # MLP is faster for N, L, C tensor
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = x.reshape(B, H, W, -1)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, hw: Tuple[int, int], pad_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw if pad_hw is not None else hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    x = x[:, :H, :W, :].contiguous()
    return x


class VisionTransformerSAM(nn.Module):
    """ Vision Transformer for Segment-Anything Model(SAM)

    A PyTorch impl of : `Exploring Plain Vision Transformer Backbones for Object Detection` or `Segment Anything Model (SAM)`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            num_classes: int = 768,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            pre_norm: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = partial(PatchEmbed, output_fmt=Format.NHWC, strict_img_size=False),
            norm_layer: Optional[Callable] = nn.LayerNorm,
            act_layer: Optional[Callable] = nn.GELU,
            block_fn: Callable = Block,
            mlp_layer: Callable = Mlp,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            use_rope: bool = False,
            window_size: int = 14,
            global_attn_indexes: Tuple[int, ...] = (),
            neck_chans: int = 256,
            global_pool: str = 'avg',
            head_hidden_size: Optional[int] = None,
            ref_feat_shape: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
            use_abs_pos: If True, use absolute positional embeddings.
            use_rel_pos: If True, add relative positional embeddings to the attention map.
            use_rope: If True, add rotary position embeddings to q/k in attention block.
            window_size: Window size for window attention blocks. If 0, not use window attention.
            global_attn_indexes: Indexes for blocks using global attention. Used when window_size > 0.
            global_pool: Global pooling type.
            head_hidden_size: If set, use NormMlpHead
            ref_feat_shape: Tuple of reference feature shapes for ROPE, (global, local)
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used
        )
        grid_size = self.patch_embed.grid_size
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(torch.zeros(1, grid_size[0], grid_size[1], embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=0,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        if use_rope:
            assert not use_rel_pos, "ROPE and relative pos embeddings should not be enabled at same time"
            if ref_feat_shape is not None:
                assert len(ref_feat_shape) == 2
                ref_feat_shape_global = to_2tuple(ref_feat_shape[0])
                ref_feat_shape_window = to_2tuple(ref_feat_shape[1])
            else:
                ref_feat_shape_global = ref_feat_shape_window = None
            self.rope_global = RotaryEmbeddingCat(
                embed_dim // num_heads,
                in_pixels=False,
                feat_shape=grid_size,
                ref_feat_shape=ref_feat_shape_global,
            )
            self.rope_window = RotaryEmbeddingCat(
                embed_dim // num_heads,
                in_pixels=False,
                feat_shape=to_2tuple(window_size),
                ref_feat_shape=ref_feat_shape_window,
            )
        else:
            self.rope_global = None
            self.rope_window = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                use_rel_pos=use_rel_pos,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=grid_size,
                rope=self.rope_window if i not in global_attn_indexes else self.rope_global,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]

        if neck_chans:
            self.neck = nn.Sequential(
                nn.Conv2d(
                    embed_dim,
                    neck_chans,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(neck_chans),
                nn.Conv2d(
                    neck_chans,
                    neck_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(neck_chans),
            )
            self.num_features = neck_chans
        else:
            if head_hidden_size:
                self.neck = nn.Identity()
            else:
                # should have a final norm with standard ClassifierHead
                self.neck = LayerNorm2d(embed_dim)
            neck_chans = embed_dim

        # Classifier Head
        if head_hidden_size:
            self.head = NormMlpClassifierHead(
                neck_chans,
                num_classes,
                hidden_size=head_hidden_size,
                pool_type=global_pool,
                drop_rate=drop_rate,
            )
        else:
            self.head = ClassifierHead(
                neck_chans,
                num_classes,
                pool_type=global_pool,
                drop_rate=drop_rate,
            )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt == 'NCHW', 'Output shape for ViT-SAM must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass, collect intermediates
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            # dynamically resize abs pos embedding if needed
            x = x + resample_abs_pos_embed_nhwc(self.pos_embed, x.shape[1:3])
        x = self.pos_drop(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)
            if i in take_indices:
                # make output BCHW
                if norm:
                    # norm is intertwined with neck convs so apply both, changes the dim
                    # FIXME only apply to final? Need experiments
                    intermediates.append(self.neck(x.permute(0, 3, 1, 2)))
                else:
                    intermediates.append(x.permute(0, 3, 1, 2))

        if intermediates_only:
            return intermediates

        x = self.neck(x.permute(0, 3, 1, 2))

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Optional[Union[int, List[int]]] = None,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            # neck is being treated as equivalent to final norm here
            self.neck = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            # dynamically resize abs pos embedding if needed
            x = x + resample_abs_pos_embed_nhwc(self.pos_embed, x.shape[1:3])
        x = self.pos_drop(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(
        state_dict,
        model,
):
    """ Remap SAM checkpoints -> timm """
    sam_checkpoint = 'image_encoder.patch_embed.proj.weight' in state_dict
    out_dict = {}
    for k, v in state_dict.items():
        if k.startswith('image_encoder.'):
            k = k[14:]
            k = k.replace('mlp.lin', 'mlp.fc')
        else:
            if sam_checkpoint:
                continue
        out_dict[k] = v
    return out_dict


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 1024, 1024), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({

    # Segment-Anything Model (SAM) pretrained - https://github.com/facebookresearch/segment-anything (no classifier head, for fine-tune/features only)
    'samvit_base_patch16.sa1b': _cfg(
        url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 1024, 1024), crop_pct=1.0),
    'samvit_large_patch16.sa1b': _cfg(
        url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 1024, 1024), crop_pct=1.0),
    'samvit_huge_patch16.sa1b': _cfg(
        url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 1024, 1024), crop_pct=1.0),

    'samvit_base_patch16_224': _cfg(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=1000,
        input_size=(3, 224, 224), crop_pct=0.9),
})


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    return build_model_with_cfg(
        VisionTransformerSAM,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )


@register_model
def samvit_base_patch16(pretrained=False, **kwargs) -> VisionTransformerSAM:
    """ ViT-B/16 for Segment-Anything
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, global_attn_indexes=[2, 5, 8, 11],
        window_size=14, use_rel_pos=True, img_size=1024,
    )
    model = _create_vision_transformer(
        'samvit_base_patch16', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def samvit_large_patch16(pretrained=False, **kwargs) -> VisionTransformerSAM:
    """ ViT-L/16 for Segment-Anything
    """
    model_args = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, global_attn_indexes=[5, 11, 17, 23],
        window_size=14, use_rel_pos=True, img_size=1024,
    )
    model = _create_vision_transformer(
        'samvit_large_patch16', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def samvit_huge_patch16(pretrained=False, **kwargs) -> VisionTransformerSAM:
    """ ViT-H/16 for Segment-Anything
    """
    model_args = dict(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, global_attn_indexes=[7, 15, 23, 31],
        window_size=14, use_rel_pos=True, img_size=1024,
    )
    model = _create_vision_transformer(
        'samvit_huge_patch16', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def samvit_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformerSAM:
    """ ViT-B/16 based on samvit arch
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, global_attn_indexes=[2, 5, 8, 11],
        window_size=14, use_rel_pos=True, use_abs_pos=False, img_size=224, neck_chans=None,
    )
    model = _create_vision_transformer(
        'samvit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

