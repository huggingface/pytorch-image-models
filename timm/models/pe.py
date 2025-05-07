from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal

import torch
import torch.nn as nn
from torch import nn, Tensor, broadcast_tensors, einsum
from torch.nn import functional as F
from torch.nn import Module, ModuleList
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint
from torch.jit import Final


### Import timm layers
from timm.layers import (
    DropPath,
    AttentionPoolLatent,
    LayerType,
    LayerScale,
    use_fused_attn,
)

# from timm.layers import RotaryEmbeddingCat, RotaryEmbedding # not compatible
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._registry import generate_default_cfgs, register_model, register_model_deprecations
from ._features_fx import register_notrace_module


__all__ = ['PE']


######## PE Rope (Simplified) ########
@register_notrace_module
class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        freqs_for: Union[Literal["lang"], Literal["pixel"], Literal["constant"]] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        theta_rescale_factor=1.0,
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        theta *= theta_rescale_factor ** (dim / (dim - 2))
        if freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * torch.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            assert False
        if learned_freq:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs, persistent=False)

    def forward(self, t: Tensor):
        freqs = self.freqs
        freqs = t.type(freqs.dtype).unsqueeze(-1) * freqs
        freqs = freqs.repeat_interleave(2, dim=-1)
        return freqs


@register_notrace_module
class Rope2D(Module):
    def __init__(self, dim, grid_size, use_cls_token=False):
        super().__init__()
        self.dim = dim
        self.use_cls_token = use_cls_token
        self.grid_size = grid_size
        self.rope = RotaryEmbedding(self.dim // 2)
        self.init_tensors()

    def init_tensors(self):
        self.update_grid(self.grid_size[0], self.grid_size[1])

    def update_grid(self, grid_h, grid_w):
        if self.use_cls_token:
            # +1 to leave space for the cls token to be (0, 0)
            grid_y_range = torch.arange(grid_h) + 1
            grid_x_range = torch.arange(grid_w) + 1
        else:
            grid_y_range = torch.arange(grid_h)
            grid_x_range = torch.arange(grid_w)
        freqs_y = self.rope(grid_y_range)[:, None].expand(grid_h, grid_w, -1)
        freqs_x = self.rope(grid_x_range)[None, :].expand(grid_h, grid_w, -1)
        freq = torch.cat([freqs_x, freqs_y], dim=-1).reshape(grid_h * grid_w, -1)

        if self.use_cls_token:
            freq = torch.cat([torch.zeros(1, freq.shape[-1]), freq], dim=0)
        self.register_buffer('freq', freq[None, ...], persistent=False)

    def rotate_half(self, x):
        shape = x.shape
        x = x.view(shape[:-1] + (-1, 2))
        x1, x2 = x[..., 0], x[..., 1]
        x = torch.stack((-x2, x1), dim=-1)
        return x.view(shape[:-1] + (-1,))

    def apply_rotary_emb(self, freqs, t):
        start_index = 0
        scale = 1.0
        seq_dim = -2
        dtype = t.dtype

        # if len(t.shape) == 3:
        #     seq_len = t.shape[seq_dim]
        #     freqs = freqs[-seq_len:]

        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim

        t_left, t, t_right = (
            t[..., :start_index],
            t[..., start_index:end_index],
            t[..., end_index:],
        )
        t = (t * freqs.cos() * scale) + (self.rotate_half(t) * freqs.sin() * scale)
        out = torch.cat((t_left, t, t_right), dim=-1)

        return out.type(dtype)

    def forward(self, q, k):
        # batch, heads, seq, dim = q.shape
        q = self.apply_rotary_emb(self.freq[:, None, :, :], q)
        k = self.apply_rotary_emb(self.freq[:, None, :, :], k)
        return q, k


######## PE Modules ########
class AttentionPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_probe: int = 1,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.probe = nn.Parameter(torch.randn(1, num_probe, self.embed_dim))
        self.attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        self.layernorm = norm_layer(embed_dim)
        self.mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(self.embed_dim, self.mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(self.mlp_width, self.embed_dim)),
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        batch, _, _ = x.shape
        q = self.probe.repeat((batch, 1, 1)).to(x.dtype)
        x = self.attn(q, x, x, need_weights=False)[0]
        x = x + self.mlp(self.layernorm(x))
        return x


class SelfAttention(nn.Module):
    r"""
    Implements sequence packed attention and RoPe
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rope: Optional[nn.Module] = None,
    ):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # To make this compatibile with nn.MultiHeadAttention
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.rope = rope
        self.scale = self.head_dim ** (-0.5)
        self.fused_attn = use_fused_attn()

    def init_tensors(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        batch, seq, embed_dim = x.shape
        proj = F.linear(x, self.in_proj_weight, self.in_proj_bias)

        # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
        proj = proj.unflatten(-1, (3, embed_dim)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = proj[0], proj[1], proj[2]

        # Use "q_" so that we don't accidentally quit in pdb :)
        q = q.view(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.rope is not None:
            q, k = self.rope(q, k)

        if self.fused_attn:
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = attn @ v

        attn = attn.permute(0, 2, 1, 3).contiguous().view(batch, seq, -1)

        return F.linear(attn, self.out_proj.weight, self.out_proj.bias)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.attn = SelfAttention(d_model, n_head, rope=rope)

        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )

    def _call_attn(self, q_x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if attn_mask is not None:
            # Leave boolean masks as is
            if not attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(q_x.dtype)

        return self.attn(q_x, attn_mask=attn_mask)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        x = x + self.drop_path1(self.ls_1(self._call_attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                    rope=rope,
                )
                for _ in range(layers)
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def truncate(self, layer_idx: int):
        """Delete layers so the last layer is the given layer index."""
        self.layers = ((self.layers + layer_idx) % self.layers) + 1
        self.resblocks = nn.ModuleList(self.resblocks[: self.layers])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        for i, r in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class PE(nn.Module):
    def __init__(
        self,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-5),
        use_ln_pre: bool = True,
        use_ln_post: bool = True,
        ls_init_value: float = None,
        drop_path: float = 0.0,
        img_size: int = 448,  # Pretrain image size only; you can pass in any image size
        use_abs_posemb: bool = True,
        use_rope2d: bool = True,
        use_cls_token: bool = False,
        use_proj: bool = True,
        output_dim: Optional[int] = 1280,
        num_classes: int = 0,
        attn_pooler_heads: int = 8,
        use_attn_pool: bool = True,
        in_chans: int = 3,
        drop_rate: float = 0.0,  # Expected to be here, TODO add a final drop layer once head finalized
    ):
        super().__init__()
        self.patch_size = patch_size
        self.heads = heads
        self.width = width
        self.layers = layers
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.emb_dim = width

        # PE contains an (optional) projection layer
        # Flow: x -> Transfomer(x) -> pool -> proj -> head (for timm).
        # forward_features: x -> Transfomer(x)
        # forward_head: pool -> proj -> head
        # output_dim is the final output dim of the model (keep it for clarity)
        self.use_proj = use_proj
        if self.use_proj:
            self.proj_dim = output_dim
            self.head_hidden_size = self.proj_dim
            self.num_features = width  # self.proj_dim
        else:
            self.proj_dim = 0
            assert output_dim == width
            self.head_hidden_size = width
            self.num_features = width

        self.num_classes = num_classes
        self.output_dim = output_dim

        self.use_abs_posemb = use_abs_posemb
        self.use_cls_token = use_cls_token
        self.use_rope2d = use_rope2d

        if isinstance(img_size, (tuple, list)):
            img_size = img_size[0]
        self.img_size = img_size
        self.grid_size = self.img_size // self.patch_size

        self.conv1 = nn.Conv2d(
            in_channels=in_chans,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        self.rope = (
            Rope2D(
                dim=width // heads,
                use_cls_token=self.use_cls_token,
                grid_size=(img_size // patch_size, img_size // patch_size),
            )
            if self.use_rope2d
            else None
        )

        self.ln_pre = norm_layer(width) if use_ln_pre else nn.Identity()
        self.ln_post = norm_layer(self.width) if use_ln_post else nn.Identity()

        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path,
            rope=self.rope,
        )

        self.feature_info = [dict(module=f'blocks.{i}', num_chs=width, reduction=patch_size) for i in range(layers)]

        if use_attn_pool:
            self.attn_pool = AttentionPooling(
                embed_dim=width,
                num_heads=attn_pooler_heads,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None

        self.head_act_layer = None  # =act_layer if to add an additional activation between fc1(proj) and fc2(head)
        self.init_tensors()

    def init_tensors(self):
        def init_submodule_tensors(module):
            for name, child in module.named_children():
                if hasattr(child, "init_tensors"):
                    # logger.debug(f"Initializing tensors for submodule: {name}")
                    child.init_tensors()
                init_submodule_tensors(child)

        init_submodule_tensors(self)
        self.rope.init_tensors()

        # class embeddings and positional embeddings
        init_scale = self.width**-0.5

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(init_scale * torch.randn(self.width))
        else:
            self.class_embedding = None

        if self.use_abs_posemb:
            self.positional_embedding = nn.Parameter(
                init_scale * torch.randn(int(self.use_cls_token) + self.grid_size**2, self.width)
            )
        else:
            self.positional_embedding = None

        # PE's: Transfomer(x) -> pool -> proj -> head (for timm). (PE contains an additional projection layer)
        if self.use_proj:
            self.proj = nn.Parameter(init_scale * torch.randn(self.width, self.proj_dim))
        else:  # no projection (eg PE-lang and PE-spatial)
            self.proj = None

        if self.num_classes > 0:
            self.head = nn.Linear(self.head_hidden_size, self.num_classes)  # no proj. input dim = self.width (pooled)
        else:
            self.head = nn.Identity()

    def truncate(self, layer_idx: int):
        """Delete layers so the last layer is the given layer index."""
        self.transformer.truncate(layer_idx)
        self.layers = self.transformer.layers

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.set_grad_checkpointing(enable=enable)

    def forward_pool_and_proj(self, x: torch.Tensor):
        if self.attn_pool is not None:
            x = self.attn_pool(x).squeeze(1)
        if self.proj is not None:
            x = x @ self.proj
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        # PE has an additional proj layer: Transfomer(x) -> pool -> proj -> head (for timm).
        # To discuss with Ross where to split
        x = self.forward_pool_and_proj(x)
        if self.head_act_layer is not None:
            x = self.head_act_layer(x)
        return x if pre_logits else self.head(x)

    def forward_features(self, x: torch.Tensor, norm: bool = False):
        batch, _, h, w = x.shape

        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).reshape(batch, -1, self.width)

        if self.class_embedding is not None:
            x = torch.cat(
                [self.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x],
                dim=1,
            )

        if self.positional_embedding is not None:
            x = x + self.positional_embedding[None, ...]

        x = self.ln_pre(x)
        x = self.transformer(x)
        if norm:
            x = self.ln_post(x)

        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x, norm=True)
        x = self.forward_head(x)
        return x

    def reset_classifier(self, num_classes: int):
        self.num_classes = num_classes
        if num_classes > 0:
            if self.proj is not None:
                self.head = nn.Parameter(self.proj_dim, num_classes)
            else:  # no projection (eg PE-lang and PE-spatial)
                self.head = nn.Parameter(self.width, num_classes)
        else:
            self.head = nn.Identity()

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: Optional[Union[int, List[int]]] = None,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = 'NCHW',
        intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(self.layers, indices)

        # forward pass
        B, _, height, width = x.shape
        # patch embedgging
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, self.width)  # NLC

        if self.class_embedding is not None:
            x = torch.cat(
                [self.class_embedding.view(1, 1, -1).expand(B, -1, -1), x],
                dim=1,
            )

        if self.positional_embedding is not None:
            x = x + self.positional_embedding[None, ...]

        x = self.ln_pre(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.transformer.resblocks
        else:
            blocks = self.transformer.resblocks[: max_index + 1]

        for i, blk in enumerate(blocks):
            x = blk(x)
            if i in take_indices:
                # normalize intermediates with final norm layer if enabled
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if self.class_embedding is not None:
            prefix_tokens = [y[:, 0] for y in intermediates]  # only one cls token in PE
            intermediates = [y[:, 1:] for y in intermediates]
        else:
            prefix_tokens = None

        if reshape:
            # reshape to BCHW output format
            H = W = self.grid_size
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
        if not torch.jit.is_scripting() and return_prefix_tokens and prefix_tokens is not None:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.ln_post(x)

        return x, intermediates


def checkpoint_filter_fn(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    state_dict = state_dict.get('model', state_dict)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    if any(k.startswith("visual.") for k in state_dict):
        state_dict = {k.replace("visual.", ""): v for k, v in state_dict.items() if "visual" in k}
    return state_dict


######## PE Config ########
def _cfg(url='', **kwargs):
    return {
        'license': 'apache-2.0',
        'num_classes': 0,
        'interpolation': 'bilinear',
        'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN,  # (0.5, 0.5, 0.5)
        'std': IMAGENET_INCEPTION_STD,  # (0.5, 0.5, 0.5)
        'first_conv': 'conv1',
        'classifier': 'head',
        **kwargs,
    }


default_cfgs = generate_default_cfgs(
    {
        # TODO finalize locations
        'vit_pe_core_base_patch16_224': _cfg(
            hf_hub_id='facebook/pe_core_base_patch16_224_timm', input_size=(3, 224, 224)
        ),
        'vit_pe_core_large_patch14_336': _cfg(hf_hub_id='timm/', input_size=(3, 336, 336)),
        'vit_pe_core_gigantic_patch14_448': _cfg(hf_hub_id='timm/', input_size=(3, 448, 448)),
        'vit_pe_lang_large_patch14_448': _cfg(hf_hub_id='timm/', input_size=(3, 448, 448)),
        'vit_pe_lang_gigantic_patch14_448': _cfg(hf_hub_id='timm/', input_size=(3, 448, 448)),
        'vit_pe_spatial_gigantic_patch14_448': _cfg(hf_hub_id='timm/', input_size=(3, 448, 448)),
    }
)


def _create_pe(variant: str, pretrained: bool = False, **kwargs) -> PE:
    out_indices = kwargs.pop('out_indices', 3)
    return build_model_with_cfg(
        PE,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_strict=True,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )


@register_model
def vit_pe_core_base_patch16_224(pretrained=False, **kwargs):
    model_args = dict(
        img_size=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        mlp_ratio=4.0,
        output_dim=1024,
        num_classes=0,
        use_cls_token=True,
        use_attn_pool=True,
        use_proj=True,
    )
    return _create_pe('vit_pe_core_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_pe_core_large_patch14_336(pretrained=False, **kwargs):
    model_args = dict(
        img_size=336,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        mlp_ratio=4.0,
        output_dim=1024,
        num_classes=0,
        use_cls_token=True,
        use_attn_pool=True,
        use_proj=True,
    )
    return _create_pe('vit_pe_core_large_patch14_336', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_pe_core_gigantic_patch14_448(pretrained=False, **kwargs):
    model_args = dict(
        img_size=448,
        patch_size=14,
        width=1536,
        layers=50,
        heads=16,
        mlp_ratio=8960 / 1536,
        output_dim=1280,
        num_classes=0,
        use_cls_token=False,
        use_attn_pool=True,
        use_proj=True,
    )
    return _create_pe('vit_pe_core_gigantic_patch14_448', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_pe_lang_large_patch14_448(pretrained=False, **kwargs):
    model_args = dict(
        img_size=448,
        patch_size=14,
        width=1024,
        layers=23,
        heads=16,
        mlp_ratio=4.0,
        output_dim=1024,
        num_classes=0,
        use_cls_token=True,
        use_ln_post=False,
        use_attn_pool=False,
        ls_init_value=0.1,
        use_proj=False,
    )
    return _create_pe('vit_pe_lang_large_patch14_448', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_pe_lang_gigantic_patch14_448(pretrained=False, **kwargs):
    model_args = dict(
        img_size=448,
        patch_size=14,
        width=1536,
        layers=47,
        heads=16,
        mlp_ratio=8960 / 1536,
        output_dim=1536,
        num_classes=0,
        use_cls_token=False,
        use_ln_post=False,
        use_attn_pool=False,
        ls_init_value=0.1,
        use_proj=False,
    )
    return _create_pe('vit_pe_lang_gigantic_patch14_448', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_pe_spatial_gigantic_patch14_448(pretrained=False, **kwargs):
    model_args = dict(
        img_size=448,
        patch_size=14,
        width=1536,
        layers=50,
        heads=16,
        mlp_ratio=8960 / 1536,
        output_dim=1536,
        num_classes=0,
        use_cls_token=False,
        use_ln_post=False,
        use_attn_pool=False,
        ls_init_value=0.1,
        use_proj=False,
    )
    return _create_pe('vit_pe_spatial_gigantic_patch14_448', pretrained=pretrained, **dict(model_args, **kwargs))
