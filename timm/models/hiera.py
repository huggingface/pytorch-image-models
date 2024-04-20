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
from typing import List, Tuple, Type, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, Mlp, use_fused_attn


from ._registry import generate_default_cfgs, register_model
from ._builder import build_model_with_cfg


def conv_nd(n: int) -> Type[nn.Module]:
    """
    Returns a conv with nd (e.g., Conv2d for n=2). Work up to n=3.
    If you wanted a 4d Hiera, you could probably just implement this for n=4. (no promises)
    """
    return [nn.Identity, nn.Conv1d, nn.Conv2d, nn.Conv3d][n]


def get_resized_mask(target_size: torch.Size, mask: torch.Tensor) -> torch.Tensor:
    # target_size: [(T), (H), W]
    # (spatial) mask: [B, C, (t), (h), w]
    if mask is None:
        return mask

    assert len(mask.shape[2:]) == len(target_size)
    if mask.shape[2:] != target_size:
        return F.interpolate(mask.float(), size=target_size)
    return mask


def do_masked_conv(
        x: torch.Tensor,
        conv: nn.Module,
        mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Zero-out the masked regions of the input before conv.
    Prevents leakage of masked regions when using overlapping kernels.
    """
    if conv is None:
        return x
    if mask is None:
        return conv(x)

    mask = get_resized_mask(target_size=x.shape[2:], mask=mask)
    return conv(x * mask.bool())


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
            norm_layer: nn.Module = nn.LayerNorm,
            act_layer: nn.Module = nn.GELU,
            q_stride: int = 1,
            window_size: int = 0,
            use_mask_unit_attn: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.norm1 = norm_layer(dim)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
        else:
            self.proj = None
        self.attn = MaskUnitAttention(
            dim,
            dim_out,
            heads,
            q_stride,
            window_size,
            use_mask_unit_attn
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Q Pooling
        x_norm = self.norm1(x)
        if self.proj is not None:
            x = self.proj(x_norm)
            x = x.view(x.shape[0], self.attn.q_stride, -1, x.shape[-1]).amax(dim=1)  # max-pool
        x = x + self.drop_path1(self.attn(x_norm))

        # MLP
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class Head(nn.Module):
    def __init__(
            self,
            dim: int,
            num_classes: int,
            drop_rate: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.projection = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.projection(x)
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
        x = do_masked_conv(x, self.proj, mask)
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
            stages: Tuple[int, ...] = (2, 3, 16, 3),
            q_pool: int = 3,  # number of q_pool stages
            q_stride: Tuple[int, ...] = (2, 2),
            mask_unit_size: Tuple[int, ...] = (8, 8),  # must divide q_stride ** (#stages-1)
            # mask_unit_attn: which stages use mask unit attention?
            mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),
            dim_mul: float = 2.0,
            head_mul: float = 2.0,
            patch_kernel: Tuple[int, ...] = (7, 7),
            patch_stride: Tuple[int, ...] = (4, 4),
            patch_padding: Tuple[int, ...] = (3, 3),
            mlp_ratio: float = 4.0,
            drop_path_rate: float = 0.0,
            norm_layer: Union[str, nn.Module] = "LayerNorm",
            drop_rate: float = 0.0,
            head_init_scale: float = 0.001,
            sep_pos_embed: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Do it this way to ensure that the init args are all PoD (for config usage)
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        depth = sum(stages)
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

        self.patch_embed = PatchEmbed(
            in_chans,
            embed_dim,
            patch_kernel,
            patch_stride,
            patch_padding,
            #reshape=False,  # leave spatial / temporal dims in output
        )

        if sep_pos_embed:
            self.pos_embed = None
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.tokens_spatial_shape[0], embed_dim)
            )
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
            self.pos_embed_spatial = None
            self.pos_embed_temporal = None

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

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        cur_stage = 0
        self.blocks = nn.ModuleList()

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
                norm_layer=norm_layer,
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_mask_unit_attn=use_mask_unit_attn,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.norm = norm_layer(embed_dim)
        self.head = Head(embed_dim, num_classes, drop_rate=drop_rate)

        # Initialize everything
        if sep_pos_embed:
            nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(partial(self._init_weights))
        self.head.projection.weight.data.mul_(head_init_scale)
        self.head.projection.bias.data.mul_(head_init_scale)

    def _init_weights(self, m, init_bias=0.02):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, init_bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, init_bias)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.sep_pos_embed:
            return ["pos_embed_spatial", "pos_embed_temporal"]
        else:
            return ["pos_embed"]

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
        if self.pos_embed is not None:
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

    def forward_features(
            self,
            x: torch.Tensor,
            mask: torch.Tensor = None,
            return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        mask should be a boolean tensor of shape [B, #MUt*#MUy*#MUx] where #MU are the number of mask units in that dim.
        Note: 1 in mask is *keep*, 0 is *remove*; mask.sum(dim=-1) should be the same across the batch.
        """
        x = self.patch_embed(
            x,
            mask=mask.view(x.shape[0], 1, *self.mask_spatial_shape)  # B, C, *mask_spatial_shape
            if mask is not None else None,
        )
        x = self._pos_embed(x)
        x = self.unroll(x)

        # Discard masked tokens
        if mask is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(x.shape[0], -1, x.shape[-1])

        intermediates = []
        for i, blk in enumerate(self.blocks):
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
        x = x.mean(dim=1)
        x = self.norm(x)
        if pre_logits:
            return x
        x = self.head(x)
        return x

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor = None,
            return_intermediates: bool = False,
    ) -> torch.Tensor:
        if return_intermediates:
            x, intermediates = self.forward_features(x, mask=mask, return_intermediates=return_intermediates)
            if mask is not None:
                x = self.forward_head(x)
            return x, intermediates
        else:
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
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = generate_default_cfgs({
    "hiera_tiny_224.mae_in1k_ft_in1k": _cfg(
        url="https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth",
        #hf_hb='timm/',
    ),
    "hiera_tiny_224.mae": _cfg(
        url="https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth",
        #hf_hb='timm/',
        num_classes=0,
    ),

    "hiera_small_224.mae_in1k_ft_in1k": _cfg(
        url="https://dl.fbaipublicfiles.com/hiera/hiera_small_224.pth",
        #hf_hb='timm/',
    ),
    "hiera_small_224.mae": _cfg(
        url="https://dl.fbaipublicfiles.com/hiera/mae_hiera_small_224.pth",
        #hf_hb='timm/',
        num_classes=0,
    ),

    "hiera_base_224.mae_in1k_ft_in1k": _cfg(
        url="https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth",
        #hf_hb='timm/',
    ),
    "hiera_base_224.mae": _cfg(
        url="https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_224.pth",
        #hf_hb='timm/',
        num_classes=0,
    ),

    "hiera_base_plus_224.mae_in1k_ft_in1k": _cfg(
        url="https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_224.pth",
        #hf_hb='timm/',
    ),
    "hiera_base_plus_224.mae": _cfg(
        url="https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_224.pth",
        #hf_hb='timm/',
        num_classes=0,
    ),

    "hiera_large_224.mae_in1k_ft_in1k": _cfg(
        url="https://dl.fbaipublicfiles.com/hiera/hiera_large_224.pth",
        #hf_hb='timm/',
    ),
    "hiera_large_224.mae": _cfg(
        url="https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_224.pth",
        #hf_hb='timm/',
        num_classes=0,
    ),

    "hiera_huge_224.mae_in1k_ft_in1k": _cfg(
        url="https://dl.fbaipublicfiles.com/hiera/hiera_huge_224.pth",
        #hf_hb='timm/',
    ),
    "hiera_huge_224.mae": _cfg(
        url="https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_224.pth",
        #hf_hb='timm/',
        num_classes=0,
    ),
})

def checkpoint_filter_fn(state_dict, model=None):
    state_dict = state_dict.get('model_state', state_dict)
    output = {}
    for k, v in state_dict.items():
        if k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # # To resize pos embedding when using model at different size from pretrained weights
            # from timm.layers import resample_abs_pos_embed
            # v = resample_abs_pos_embed(
            #     v,
            #     new_size=(64, 64),
            #     num_prefix_tokens=0,
            #     verbose=True,
            # )
            #v = F.interpolate(v.transpose(1, 2), (model.pos_embed.shape[1],)).transpose(1, 2)
            pass
        output[k] = v
    return output


def _create_hiera(variant: str, pretrained: bool = False, **kwargs) -> Hiera:
    out_indices = kwargs.pop('out_indices', 4)

    return build_model_with_cfg(
        Hiera,
        variant,
        pretrained,
        #pretrained_strict=False,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )

@register_model
def hiera_tiny_224(pretrained = False, **kwargs):
    model_args = dict(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2))
    return _create_hiera('hiera_tiny_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def hiera_small_224(pretrained = False, **kwargs):
    model_args = dict(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2))
    return _create_hiera('hiera_small_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def hiera_base_224(pretrained = False, **kwargs):
    model_args = dict(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3))
    return _create_hiera('hiera_base_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def hiera_base_plus_224(pretrained = False, **kwargs):
    model_args = dict(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3))
    return _create_hiera('hiera_base_plus_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def hiera_large_224(pretrained = False, **kwargs):
    model_args = dict(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4))
    return _create_hiera('hiera_large_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def hiera_huge_224(pretrained = False, **kwargs):
    model_args = dict(embed_dim=256, num_heads=4, stages=(2, 6, 36, 4))
    return _create_hiera('hiera_huge_224', pretrained=pretrained, **dict(model_args, **kwargs))
