""" BEiT3
Paper: `Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks`
    - https://arxiv.org/abs/2208.10442
    - https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Image_as_a_Foreign_Language_BEiT_Pretraining_for_Vision_and_CVPR_2023_paper.pdf

Model from official source: 
    - https://github.com/microsoft/unilm/tree/master/beit3
    - https://github.com/microsoft/torchscale/blob/main/torchscale/model/BEiT3.py

@inproceedings{beit3,
    title={Image as a foreign language: {BEiT} pretraining for vision and vision-language tasks},
    author={Wenhui Wang and Hangbo Bao and Li Dong and Johan Bjorck and Zhiliang Peng and Qiang Liu and Kriti Aggarwal and Owais Khan Mohammed and Saksham Singhal and Subhojit Som and Furu Wei},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2023}
}
@InProceedings{Wang_2023_CVPR,
    author    = {Wang, Wenhui and Bao, Hangbo and Dong, Li and Bjorck, Johan and Peng, Zhiliang and Liu, Qiang and Aggarwal, Kriti and Mohammed, Owais Khan and Singhal, Saksham and Som, Subhojit and Wei, Furu},
    title     = {Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {19175-19186}
}

Original implementation by Wenhui Wang et al.,
adapted for timm by Ryan Hou and Ross Wightman.

At this point only the 1k fine-tuned classification weights and model configs have been added,
see original source above for pre-training models and procedure.

Adapted from https://github.com/microsoft/torchscale/blob/main/torchscale/model/BEiT3.py, original copyright below
"""
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import math
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, Mlp, LayerNorm, DropPath, trunc_normal_, LayerType

from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint
from ._registry import generate_default_cfgs, register_model

__all__ = ['BEiT3']


class PositionalEmbedding(nn.Embedding):
    """
    Reference from:
    https://github.com/microsoft/torchscale/blob/main/torchscale/component/embedding.py#L99-L119
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            torch.arange(2, self.num_embeddings).long().unsqueeze(0).to(x.device),
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class Attention(nn.Module):
    """
    Reference from:
    https://github.com/microsoft/torchscale/blob/main/torchscale/component/multihead_attention.py#L20-L171
    """
    def __init__(
            self, 
            dim: int, 
            num_heads: int, 
            drop_rate: float = 0., 
            norm_layer: LayerType = partial(LayerNorm, eps=1e-5)
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.inner_attn_ln = norm_layer(dim)
        self.attn_drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q *= self.scaling

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.reshape(B * self.num_heads, N, self.head_dim)
        k = k.reshape(B * self.num_heads, N, self.head_dim)
        v = v.reshape(B * self.num_heads, N, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        attn_probs = self.attn_drop(attn_weights)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).reshape(N, B, C).transpose(0, 1)
        attn = self.inner_attn_ln(attn)
        attn = self.out_proj(attn)
        return attn


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            drop_rate: float = 0.,
            drop_path: float = 0.,
            attn_drop: float = 0.,
            act_layer: LayerType = nn.GELU,
            norm_layer: LayerType = partial(LayerNorm, eps=1e-5),
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, drop_rate=attn_drop, norm_layer=norm_layer)
        self.attn_drop = nn.Dropout(drop_rate)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop=drop_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn_drop(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BEiT3(nn.Module):
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            act_layer: LayerType = nn.GELU,
            norm_layer: LayerType = partial(LayerNorm, eps=1e-5),
            head_init_scale: float = 0.001,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1
        self.grad_checkpointing = False

        # vision_embed
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # encoder
        self.pos_embed = PositionalEmbedding(num_patches + 3, embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                attn_drop=attn_drop_rate,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]

        # class_head
        use_fc_norm = self.global_pool == 'avg'
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=.02)

        self.fix_init_weight(depth)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
    
    def fix_init_weight(self, depth: int):
        init_scale = math.sqrt(math.log(depth * 2))
        for name, p in self.named_parameters():
            if (
                "fc1" in name
                or "fc2" in name
                or "out_proj" in name
                or "v_proj" in name
            ):
                p.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable
    
    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if an int, if is a sequence, select by matching indices
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
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width = x.shape

        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed(x)
        x = self.pos_drop(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]

        for i, blk in enumerate(blocks):
            x = blk(x)
            if i in take_indices:
                # normalize intermediates with final norm layer if enabled
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        
        if reshape:
            # reshape to BCHW output format
            H, W = self.patch_embed.dynamic_feat_size((height, width))
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
        if not torch.jit.is_scripting() and return_prefix_tokens:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed(x)
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _cfg(url: str = '', **kwargs: Any) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        'paper_ids': 'arXiv:2208.10442',
        'paper_name': 'Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks',
        'origin_url': 'https://github.com/microsoft/unilm/tree/master/beit3',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'beit3_base_patch16_224.in1k': _cfg(
        url='https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_224_in1k.pth',
        # hf_hub_id='timm/',
    ),
    'beit3_base_patch16_224.indomain_in1k': _cfg(
        url='https://github.com/addf400/files/releases/download/beit3/beit3_base_indomain_patch16_224_in1k.pth',
        # hf_hub_id='timm/',
    ),
    'beit3_large_patch16_224.in1k': _cfg(
        url='https://github.com/addf400/files/releases/download/beit3/beit3_large_patch16_224_in1k.pth',
        # hf_hub_id='timm/',
    ),
    'beit3_large_patch16_224.indomain_in1k': _cfg(
        url='https://github.com/addf400/files/releases/download/beit3/beit3_large_indomain_patch16_224_in1k.pth',
        # hf_hub_id='timm/',
    ),
})


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: BEiT3,
) -> Dict[str, torch.Tensor]:
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    if 'patch_embed.proj.weight' in state_dict:
        return state_dict
    
    state_dict.pop('beit3.text_embed.weight')
    state_dict.pop('beit3.vision_embed.mask_token')

    out_dict = {}

    for k, v in state_dict.items():
        if '.B.' in k:
            continue
        elif 'vision_embed.cls_token' in k:
            k = 'cls_token'
        else:
            k = k.replace('beit3.', '')
            k = k.replace('embed_positions.', 'pos_embed.')
            k = k.replace('vision_embed.', 'patch_embed.')
            k = k.replace('encoder.', '')
            k = k.replace('layers.', 'blocks.')
            k = k.replace('ffn.', 'mlp.')
            k = k.replace('ffn_layernorm.', 'norm.')
            k = k.replace('self_attn.', 'attn.')
            k = k.replace('self_attn_layer_norm.', 'norm1.')
            k = k.replace('final_layer_norm.', 'norm2.')
            k = k.replace('A.', '')
        
        out_dict[k] = v
    
    return out_dict


def _create_beit3(variant: str, pretrained: bool, **kwargs: Any) -> BEiT3:
    out_indices = kwargs.pop('out_indices', 3)
    model = build_model_with_cfg(
        BEiT3, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model


@register_model
def beit3_base_patch16_224(pretrained: bool = False, **kwargs: Any) -> BEiT3:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4)
    model = _create_beit3('beit3_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def beit3_large_patch16_224(pretrained: bool = False, **kwargs: Any) -> BEiT3:
    model_args = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4)
    model = _create_beit3('beit3_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
