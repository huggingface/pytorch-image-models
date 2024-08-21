""" An PyTorch implementation of Hiera

Adapted for timm from originals at https://github.com/facebookresearch/hiera
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles
#
# Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan,
# Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed,
# Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer.
#
# Paper: https://arxiv.org/abs/2306.00989/
#
# References:
# slowfast: https://github.com/facebookresearch/SlowFast
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------
import math
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, Mlp, LayerScale, ClNormMlpClassifierHead, use_fused_attn, \
    _assert, get_norm_layer, to_2tuple, init_weight_vit, init_weight_jax

from ._registry import generate_default_cfgs, register_model
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._features_fx import register_notrace_function
from ._manipulate import named_apply


__all__ = ['Hiera']


def conv_nd(n: int) -> Type[nn.Module]:
    """
    Returns a conv with nd (e.g., Conv2d for n=2). Work up to n=3.
    If you wanted a 4d Hiera, you could probably just implement this for n=4. (no promises)
    """
    return [nn.Identity, nn.Conv1d, nn.Conv2d, nn.Conv3d][n]


@register_notrace_function
def get_resized_mask(target_size: List[int], mask: torch.Tensor) -> torch.Tensor:
    # target_size: [(T), (H), W]
    # (spatial) mask: [B, C, (t), (h), w]
    if mask is None:
        return mask

    _assert(len(mask.shape[2:]) == len(target_size), "mask spatial shape and target_size must match.")
    if mask.shape[2:] != target_size:
        return F.interpolate(mask.float(), size=target_size)
    return mask


def undo_windowing(
        x: torch.Tensor,
        shape: List[int],
        mu_shape: List[int],
) -> torch.Tensor:
    """
    Restore spatial organization by undoing windowed organization of mask units.

    Args:
        x: organized by mask units windows, e.g. in 2d [B, #MUy*#MUx, MUy, MUx, C]
        shape: current spatial shape, if it were not organized into mask unit
            windows, e.g. in 2d [B, #MUy*MUy, #MUx*MUx, C].
        mu_shape: current mask unit shape, e.g. in 2d [MUy, MUx]
    Returns:
        x: e.g. in 2d, [B, #MUy*MUy, #MUx*MUx, C]
    """
    D = len(shape)
    B, C = x.shape[0], x.shape[-1]
    # [B, #MUy*#MUx, MUy, MUx, C] -> [B, #MUy, #MUx, MUy, MUx, C]
    num_MUs = [s // mu for s, mu in zip(shape, mu_shape)]
    x = x.view(B, *num_MUs, *mu_shape, C)

    # [B, #MUy, #MUx, MUy, MUx, C] -> [B, #MUy*MUy, #MUx*MUx, C]
    permute = (
        [0]
        + sum([list(p) for p in zip(range(1, 1 + D), range(1 + D, 1 + 2 * D))], [])
        + [len(x.shape) - 1]
    )
    x = x.permute(permute).reshape(B, *shape, C)

    return x


class Unroll(nn.Module):
    """
    Reorders the tokens such that patches are contiguous in memory.
    E.g., given [B, (H, W), C] and stride of (Sy, Sx), this will re-order the tokens as
                           [B, (Sy, Sx, H // Sy, W // Sx), C]

    This allows operations like Max2d to be computed as x.view(B, Sx*Sy, -1, C).max(dim=1).
    Not only is this faster, but it also makes it easy to support inputs of arbitrary
    dimensions in addition to patch-wise sparsity.

    Performing this operation multiple times in sequence puts entire windows as contiguous
    in memory. For instance, if you applied the stride (2, 2) 3 times, entire windows of
    size 8x8 would be contiguous in memory, allowing operations like mask unit attention
    computed easily and efficiently, while also allowing max to be applied sequentially.

    Note: This means that intermediate values of the model are not in HxW order, so they
    need to be re-rolled if you want to use the intermediate values as a HxW feature map.
    The last block of the network is fine though, since by then the strides are all consumed.
    """

    def __init__(
            self,
            input_size: Tuple[int, ...],
            patch_stride: Tuple[int, ...],
            unroll_schedule: List[Tuple[int, ...]],
    ):
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]
        self.schedule = unroll_schedule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Flattened patch embeddings [B, N, C]
        Output: Patch embeddings [B, N, C] permuted such that [B, 4, N//4, C].max(1) etc. performs MaxPoolNd
        """
        B, _, C = x.shape
        cur_size = self.size
        x = x.view(*([B] + cur_size + [C]))

        for strides in self.schedule:
            # Move patches with the given strides to the batch dimension

            # Create a view of the tensor with the patch stride as separate dims
            # For example in 2d: [B, H // Sy, Sy, W // Sx, Sx, C]
            cur_size = [i // s for i, s in zip(cur_size, strides)]
            new_shape = [B] + sum([[i, s] for i, s in zip(cur_size, strides)], []) + [C]
            x = x.view(new_shape)

            # Move the patch stride into the batch dimension
            # For example in 2d: [B, Sy, Sx, H // Sy, W // Sx, C]
            L = len(new_shape)
            permute = [0] + list(range(2, L - 1, 2)) + list(range(1, L - 1, 2)) + [L - 1]
            x = x.permute(permute)

            # Now finally flatten the relevant dims into the batch dimension
            x = x.flatten(0, len(strides))
            B *= math.prod(strides)

        x = x.reshape(-1, math.prod(self.size), C)
        return x


class Reroll(nn.Module):
    """
    Undos the "unroll" operation so that you can use intermediate features.
    """

    def __init__(
            self,
            input_size: Tuple[int, ...],
            patch_stride: Tuple[int, ...],
            unroll_schedule: List[Tuple[int, ...]],
            stage_ends: List[int],
            q_pool: int,
    ):
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]

        # The first stage has to reverse everything
        # The next stage has to reverse all but the first unroll, etc.
        self.schedule = {}
        size = self.size
        for i in range(stage_ends[-1] + 1):
            self.schedule[i] = unroll_schedule, size
            # schedule unchanged if no pooling at a stage end
            if i in stage_ends[:q_pool]:
                if len(unroll_schedule) > 0:
                    size = [n // s for n, s in zip(size, unroll_schedule[0])]
                unroll_schedule = unroll_schedule[1:]

    def forward(
            self,
            x: torch.Tensor,
            block_idx: int,
            mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Roll the given tensor back up to spatial order assuming it's from the given block.

        If no mask is provided:
            - Returns [B, H, W, C] for 2d, [B, T, H, W, C] for 3d, etc.
        If a mask is provided:
            - Returns [B, #MUs, MUy, MUx, C] for 2d, etc.
        """
        schedule, size = self.schedule[block_idx]
        B, N, C = x.shape

        D = len(size)
        cur_mu_shape = [1] * D

        for strides in schedule:
            # Extract the current patch from N
            x = x.view(B, *strides, N // math.prod(strides), *cur_mu_shape, C)

            # Move that patch into the current MU
            # Example in 2d: [B, Sy, Sx, N//(Sy*Sx), MUy, MUx, C] -> [B, N//(Sy*Sx), Sy, MUy, Sx, MUx, C]
            L = len(x.shape)
            permute = (
                [0, 1 + D]
                + sum([list(p) for p in zip(range(1, 1 + D), range(1 + D + 1, L - 1))], [])
                + [L - 1]
            )
            x = x.permute(permute)

            # Reshape to [B, N//(Sy*Sx), *MU, C]
            for i in range(D):
                cur_mu_shape[i] *= strides[i]
            x = x.reshape(B, -1, *cur_mu_shape, C)
            N = x.shape[1]

        # Current shape (e.g., 2d: [B, #MUy*#MUx, MUy, MUx, C])
        x = x.view(B, N, *cur_mu_shape, C)

        # If masked, return [B, #MUs, MUy, MUx, C]
        if mask is not None:
            return x

        # If not masked, we can return [B, H, W, C]
        x = undo_windowing(x, size, cur_mu_shape)

        return x


class MaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See `Unroll` for more details.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            dim_out: int,
            heads: int,
            q_stride: int = 1,
            window_size: int = 0,
            use_mask_unit_attn: bool = False,
    ):
        """
        Args:
        - dim, dim_out: The input and output feature dimensions.
        - heads: The number of attention heads.
        - q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        - window_size: The current (flattened) size of a mask unit *after* pooling (if any).
        - use_mask_unit_attn: Use Mask Unit or Global Attention.
        """
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride
        self.head_dim = dim_out // heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Input should be of shape [batch, tokens, channels]. """
        B, N, _ = x.shape
        num_windows = (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1
        qkv = self.qkv(x).reshape(B, -1, num_windows, 3, self.heads, self.head_dim).permute(3, 0, 4, 2, 1, 5)
        q, k, v = qkv.unbind(0)

        if self.q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            q = q.view(B, self.heads, num_windows, self.q_stride, -1, self.head_dim).amax(dim=3)

        if self.fused_attn:
            # Note: the original paper did *not* use SDPA, it's a free boost!
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q * self.scale) @ k.transpose(-1, -2)
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = x.transpose(1, 3).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x


class HieraBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            heads: int,
            mlp_ratio: float = 4.0,
            drop_path: float = 0.0,
            init_values: Optional[float] = None,
            norm_layer: nn.Module = nn.LayerNorm,
            act_layer: nn.Module = nn.GELU,
            q_stride: int = 1,
            window_size: int = 0,
            use_expand_proj: bool = True,
            use_mask_unit_attn: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out

        self.norm1 = norm_layer(dim)
        if dim != dim_out:
            self.do_expand = True
            if use_expand_proj:
                self.proj = nn.Linear(dim, dim_out)
            else:
                assert dim_out == dim * 2
                self.proj = None
        else:
            self.do_expand = False
            self.proj = None
        self.attn = MaskUnitAttention(
            dim,
            dim_out,
            heads,
            q_stride,
            window_size,
            use_mask_unit_attn
        )
        self.ls1 = LayerScale(dim_out, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)
        self.ls2 = LayerScale(dim_out, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Q Pooling
        x_norm = self.norm1(x)
        if self.do_expand:
            if self.proj is not None:
                x = self.proj(x_norm)
                x = x.view(x.shape[0], self.attn.q_stride, -1, x.shape[-1]).amax(dim=1)  # max-pool
            else:
                x = torch.cat([
                    x.view(x.shape[0], self.attn.q_stride, -1, x.shape[-1]).amax(dim=1),  # max-pool
                    x.view(x.shape[0], self.attn.q_stride, -1, x.shape[-1]).mean(dim=1),  # avg-pool
                    ],
                    dim=-1,
                )
        x = x + self.drop_path1(self.ls1(self.attn(x_norm)))

        # MLP
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PatchEmbed(nn.Module):
    """Patch embed that supports any number of spatial dimensions (1d, 2d, 3d)."""

    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            kernel: Tuple[int, ...],
            stride: Tuple[int, ...],
            padding: Tuple[int, ...],
            reshape: bool = True,
    ):
        super().__init__()

        # Support any number of spatial dimensions
        self.spatial_dims = len(kernel)
        self.reshape = reshape
        self.proj = conv_nd(self.spatial_dims)(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask = get_resized_mask(target_size=x.shape[2:], mask=mask)
            x = self.proj(x * mask.to(torch.bool))
        else:
            x = self.proj(x)
        if self.reshape:
            x = x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)
        return x


class Hiera(nn.Module):

    def __init__(
            self,
            img_size: Tuple[int, ...] = (224, 224),
            in_chans: int = 3,
            embed_dim: int = 96,  # initial embed dim
            num_heads: int = 1,  # initial number of heads
            num_classes: int = 1000,
            global_pool: str = 'avg',
            stages: Tuple[int, ...] = (2, 3, 16, 3),
            q_pool: int = 3,  # number of q_pool stages
            q_stride: Tuple[int, ...] = (2, 2),
            mask_unit_size: Tuple[int, ...] = (8, 8),  # must divide q_stride ** (#stages-1)
            # mask_unit_attn: which stages use mask unit attention?
            mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),
            use_expand_proj: bool = True,
            dim_mul: float = 2.0,
            head_mul: float = 2.0,
            patch_kernel: Tuple[int, ...] = (7, 7),
            patch_stride: Tuple[int, ...] = (4, 4),
            patch_padding: Tuple[int, ...] = (3, 3),
            mlp_ratio: float = 4.0,
            drop_path_rate: float = 0.0,
            init_values: Optional[float] = None,
            fix_init: bool = True,
            weight_init: str = '',
            norm_layer: Union[str, nn.Module] = "LayerNorm",
            drop_rate: float = 0.0,
            patch_drop_rate: float = 0.0,
            head_init_scale: float = 0.001,
            sep_pos_embed: bool = False,
            abs_win_pos_embed: bool = False,
            global_pos_size: Tuple[int, int] = (14, 14),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.grad_checkpointing = False
        norm_layer = get_norm_layer(norm_layer)
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)

        self.patch_stride = patch_stride
        self.tokens_spatial_shape = [i // s for i, s in zip(img_size, patch_stride)]
        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(mask_unit_size)
        flat_q_stride = math.prod(q_stride)
        assert q_pool < len(stages)
        self.q_pool, self.q_stride = q_pool, q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, mask_unit_size
        self.mask_spatial_shape = [i // s for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size)]
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        self.patch_drop_rate = patch_drop_rate

        self.patch_embed = PatchEmbed(
            in_chans,
            embed_dim,
            patch_kernel,
            patch_stride,
            patch_padding,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        self.pos_embed_win: Optional[nn.Parameter] = None
        self.pos_embed_spatial: Optional[nn.Parameter] = None
        self.pos_embed_temporal: Optional[nn.Parameter] = None
        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.tokens_spatial_shape[0], embed_dim)
            )
        else:
            if abs_win_pos_embed:
                # absolute win, params NCHW to make tile & interpolate more natural before add & reshape
                self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *global_pos_size))
                self.pos_embed_win = nn.Parameter(torch.zeros(1, embed_dim, *mask_unit_size))
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        # Setup roll and reroll modules
        self.unroll = Unroll(
            img_size,
            patch_stride,
            [q_stride] * len(self.stage_ends[:-1])
        )
        self.reroll = Reroll(
            img_size,
            patch_stride,
            [q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            q_pool,
        )
        # q_pool locations
        q_pool_blocks = [x + 1 for x in self.stage_ends[:q_pool]]

        # Transformer blocks
        cur_stage = 0
        depth = sum(stages)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        self.feature_info = []
        for i in range(depth):
            dim_out = embed_dim
            # Mask unit or global attention.
            # Lag by 1 block, so that global attention,
            # applied post pooling on lower resolution
            use_mask_unit_attn = mask_unit_attn[cur_stage]

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride

            block = HieraBlock(
                dim=embed_dim,
                dim_out=dim_out,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                init_values=init_values,
                norm_layer=norm_layer,
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_expand_proj=use_expand_proj,
                use_mask_unit_attn=use_mask_unit_attn,
            )
            embed_dim = dim_out
            if i in self.stage_ends:
                self.feature_info += [
                    dict(num_chs=dim_out, reduction=2**(cur_stage+2), module=f'blocks.{self.stage_ends[cur_stage]}')]
            self.blocks.append(block)

        self.num_features = self.head_hidden_size = embed_dim
        self.head = ClNormMlpClassifierHead(
            embed_dim,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            norm_layer=norm_layer,
            input_fmt='NLC',
        )

        # Initialize everything
        if sep_pos_embed:
            nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            if self.pos_embed is not None:
                nn.init.trunc_normal_(self.pos_embed, std=0.02)
            if self.pos_embed_win is not None:
                nn.init.trunc_normal_(self.pos_embed_win, std=0.02)

        if weight_init != 'skip':
            init_fn = init_weight_jax if weight_init == 'jax' else init_weight_vit
            init_fn = partial(init_fn, classifier_name='head.fc')
            named_apply(init_fn, self)
        if fix_init:
            self.fix_init_weight()
        if isinstance(self.head.fc, nn.Linear):
            self.head.fc.weight.data.mul_(head_init_scale)
            self.head.fc.bias.data.mul_(head_init_scale)

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.pos_embed is not None:
            return ["pos_embed"]
        elif self.pos_embed_abs is not None:
            return ['pos_embed_abs', 'pos_embed_win']
        else:
            return ["pos_embed_spatial", "pos_embed_temporal"]

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r'^pos_embed|pos_embed_spatial|pos_embed_temporal|pos_embed_abs|pos_embed_win|patch_embed',
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None, reset_other: bool = False):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool, reset_other=reset_other)

    def get_random_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Generates a random mask, mask_ratio fraction are dropped.
        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.
        """
        B = x.shape[0]
        # Tokens selected for masking at mask unit level
        num_windows = math.prod(self.mask_spatial_shape)  # num_mask_units
        len_keep = int(num_windows * (1 - mask_ratio))
        noise = torch.rand(B, num_windows, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 1 is *keep*, 0 is *remove*
        # Note this is opposite to original MAE
        mask = torch.zeros([B, num_windows], device=x.device)
        mask[:, :len_keep] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.bool()

    def _pos_embed(self, x) -> torch.Tensor:
        if self.pos_embed_win is not None:
            # absolute win position embedding, from
            # Window Attention is Bugged: How not to Interpolate Position Embeddings (https://arxiv.org/abs/2311.05613)
            pos_embed_win = self.pos_embed_win.tile(self.mask_spatial_shape)
            pos_embed = F.interpolate(
                self.pos_embed,
                size=pos_embed_win.shape[-2:],
                mode='bicubic',
                antialias=True,
            )
            pos_embed = pos_embed + pos_embed_win
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
        elif self.pos_embed is not None:
            pos_embed = self.pos_embed
        else:
            pos_embed = (
                self.pos_embed_spatial.repeat(1, self.tokens_spatial_shape[0], 1)
                +
                torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                    dim=1,
                )
            )
        x = x + pos_embed
        return x

    def forward_intermediates(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = True,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            coarse: bool = True,
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
        assert not norm, 'normalization of features not supported'
        assert output_fmt in ('NCHW', 'NHWC'), 'Output format must be one of NCHW, NHWC.'
        if coarse:
            take_indices, max_index = feature_take_indices(len(self.stage_ends), indices)
            take_indices = [self.stage_ends[i] for i in take_indices]
            max_index = self.stage_ends[max_index]
        else:
            take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        if mask is not None:
            patch_mask = mask.view(x.shape[0], 1, *self.mask_spatial_shape)  # B, C, *mask_spatial_shape
        else:
            patch_mask = None
        x = self.patch_embed(x, mask=patch_mask)
        x = self._pos_embed(x)
        x = self.unroll(x)

        # Discard masked tokens
        if mask is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(x.shape[0], -1, x.shape[-1])

        intermediates = []
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            x = blk(x)
            if i in take_indices:
                x_int = self.reroll(x, i, mask=mask)
                intermediates.append(x_int.permute(0, 3, 1, 2) if output_fmt == 'NCHW' else x_int)

        if intermediates_only:
            return intermediates

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
            coarse: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        if coarse:
            take_indices, max_index = feature_take_indices(len(self.stage_ends), indices)
            max_index = self.stage_ends[max_index]
        else:
            take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_head:
            self.head.reset(0, reset_other=True)
        return take_indices

    def forward_features(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        mask should be a boolean tensor of shape [B, #MUt*#MUy*#MUx] where #MU are the number of mask units in that dim.
        Note: 1 in mask is *keep*, 0 is *remove*; mask.sum(dim=-1) should be the same across the batch.
        """
        if self.training and self.patch_drop_rate > 0:
            # using mask for something like 'patch dropout' via mask-units in supervised train / fine-tune
            assert mask is None
            mask = self.get_random_mask(x, mask_ratio=self.patch_drop_rate)

        if mask is not None:
            patch_mask = mask.view(x.shape[0], 1, *self.mask_spatial_shape)  # B, C, *mask_spatial_shape
        else:
            patch_mask = None
        x = self.patch_embed(x, mask=patch_mask)
        x = self._pos_embed(x)
        x = self.unroll(x)

        # Discard masked tokens
        if mask is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(x.shape[0], -1, x.shape[-1])

        intermediates = []
        for i, blk in enumerate(self.blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)
            if return_intermediates and i in self.stage_ends:
                intermediates.append(self.reroll(x, i, mask=mask))

        # x may not always be in spatial order here.
        # e.g. if q_pool = 2, mask_unit_size = (8, 8), and
        # q_stride = (2, 2), not all unrolls were consumed,
        # intermediates[-1] is x in spatial order
        if return_intermediates:
            return x, intermediates

        return x

    def forward_head(self, x, pre_logits: bool = False) -> torch.Tensor:
        x = self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)
        return x

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.forward_features(x, mask=mask)
        if mask is None:
            x = self.forward_head(x)
        return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    "hiera_tiny_224.mae_in1k_ft_in1k": _cfg(
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
    ),
    "hiera_tiny_224.mae": _cfg(
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        num_classes=0,
    ),

    "hiera_small_224.mae_in1k_ft_in1k": _cfg(
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
    ),
    "hiera_small_224.mae": _cfg(
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        num_classes=0,
    ),

    "hiera_base_224.mae_in1k_ft_in1k": _cfg(
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
    ),
    "hiera_base_224.mae": _cfg(
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        num_classes=0,
    ),

    "hiera_base_plus_224.mae_in1k_ft_in1k": _cfg(
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
    ),
    "hiera_base_plus_224.mae": _cfg(
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        num_classes=0,
    ),

    "hiera_large_224.mae_in1k_ft_in1k": _cfg(
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
    ),
    "hiera_large_224.mae": _cfg(
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        num_classes=0,
    ),

    "hiera_huge_224.mae_in1k_ft_in1k": _cfg(
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
    ),
    "hiera_huge_224.mae": _cfg(
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        num_classes=0,
    ),

    "hiera_small_abswin_256.sbb2_e200_in12k_ft_in1k": _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), crop_pct=0.95,
    ),
    "hiera_small_abswin_256.sbb2_pd_e200_in12k_ft_in1k": _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), crop_pct=0.95,
    ),
    "hiera_small_abswin_256.sbb2_e200_in12k": _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
        input_size=(3, 256, 256), crop_pct=0.95,
    ),
    "hiera_small_abswin_256.sbb2_pd_e200_in12k": _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
        input_size=(3, 256, 256), crop_pct=0.95,
    ),
    "hiera_base_abswin_256.untrained": _cfg(
        # hf_hub_id='timm/',
        input_size=(3, 256, 256), crop_pct=0.95,
    ),
})


def checkpoint_filter_fn(state_dict, model=None):
    state_dict = state_dict.get('model_state', state_dict)
    output = {}
    for k, v in state_dict.items():
        # if k == 'pos_embed' and  v.shape[1] != model.pos_embed.shape[1]:
        #     # To resize pos embedding when using model at different size from pretrained weights
        #     from timm.layers import resample_abs_pos_embed
        #     v = resample_abs_pos_embed(
        #         v,
        #         new_size=(64, 64),
        #         num_prefix_tokens=0,
        #         verbose=True,
        #     )
        if 'head.projection.' in k:
            k = k.replace('head.projection.', 'head.fc.')
        if k.startswith('encoder_norm.'):
            k = k.replace('encoder_norm.', 'head.norm.')
        elif k.startswith('norm.'):
            k = k.replace('norm.', 'head.norm.')
        if k == 'pos_embed_abs':
            k = 'pos_embed'
        output[k] = v
    return output


def _create_hiera(variant: str, pretrained: bool = False, **kwargs) -> Hiera:
    out_indices = kwargs.pop('out_indices', 4)

    return build_model_with_cfg(
        Hiera,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )


@register_model
def hiera_tiny_224(pretrained=False, **kwargs):
    model_args = dict(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2))
    return _create_hiera('hiera_tiny_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def hiera_small_224(pretrained=False, **kwargs):
    model_args = dict(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2))
    return _create_hiera('hiera_small_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def hiera_base_224(pretrained=False, **kwargs):
    model_args = dict(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3))
    return _create_hiera('hiera_base_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def hiera_base_plus_224(pretrained=False, **kwargs):
    model_args = dict(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3))
    return _create_hiera('hiera_base_plus_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def hiera_large_224(pretrained=False, **kwargs):
    model_args = dict(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4))
    return _create_hiera('hiera_large_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def hiera_huge_224(pretrained=False, **kwargs):
    model_args = dict(embed_dim=256, num_heads=4, stages=(2, 6, 36, 4))
    return _create_hiera('hiera_huge_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def hiera_small_abswin_256(pretrained=False, **kwargs):
    model_args = dict(
        embed_dim=96, num_heads=1, stages=(1, 2, 11, 2), abs_win_pos_embed=True, global_pos_size=(16, 16),
        init_values=1e-5, weight_init='jax', use_expand_proj=False,
    )
    return _create_hiera('hiera_small_abswin_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def hiera_base_abswin_256(pretrained=False, **kwargs):
    model_args = dict(
        embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), abs_win_pos_embed=True, init_values=1e-5, weight_init='jax')
    return _create_hiera('hiera_base_abswin_256', pretrained=pretrained, **dict(model_args, **kwargs))
