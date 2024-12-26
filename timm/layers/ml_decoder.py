from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_activation_fn
from torch.jit import Final

from timm.layers import Mlp, use_fused_attn
from timm.layers.classifier import _create_pool


class MLDecoderHead(nn.Module):
    """MLDecoder wrapper with forward compatible with ClassifierHead"""

    def __init__(self, in_features, num_classes, pool_type='avg', use_conv=False, input_fmt='NCHW'):
        super(MLDecoderHead, self).__init__()
        self.in_features = in_features
        self.use_conv = use_conv
        self.input_fmt = input_fmt

        self.global_pool, num_pooled_features = _create_pool(in_features, num_classes, pool_type, use_conv=use_conv, input_fmt=input_fmt)
        self.head = MLDecoder(in_features=in_features, num_classes=num_classes)
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()


    def reset(self, num_classes, global_pool=None):
        if global_pool is not None:
            if global_pool != self.global_pool.pool_type:
                self.global_pool, _ = _create_pool(self.in_features, num_classes, global_pool, use_conv=self.use_conv)
            self.flatten = nn.Flatten(1) if self.use_conv and global_pool else nn.Identity()
        num_pooled_features = self.in_features * self.global_pool.feat_mult()
        self.head = MLDecoder(in_features=in_features, num_classes=num_classes)


    def forward(self, x, pre_logits: bool = False):
        # pool for compatibility with ClassifierHead
        if self.input_fmt == 'NHWC':
            x = x.permute(0, 3, 1, 2)
        if pre_logits:
            x = self.global_pool(x)
            return x.flatten(1)
        else:
            x = self.head(x)
            return self.flatten(x)

def add_ml_decoder_head(model):

    # ignore CoaT, crossvit
    # ignore distillation models: deit_distilled, efficientformer V2
    num_classes = model.num_classes
    num_features = model.num_features

    assert num_classes > 0, "MLDecoder requires a model to have num_classes > 0"

    if hasattr(model, 'global_pool') and hasattr(model, 'fc'):  # most CNN models, like Resnet50
        model.global_pool = nn.Identity()
        del model.fc

        model.fc = MLDecoder(num_classes=num_classes, in_features=num_features)

    elif hasattr(model, 'fc_norm') or 'Cait' in model._get_name(): # ViT, BEiT, EVA
        model.global_pool = None # disable any pooling, model instantiation leaves 1 norm layer after features, [B, n + K x K, C]
        if hasattr(model, 'attn_pool'):
            model.attn_pool = None
        model.head_drop = nn.Identity()
        model.head = MLDecoder(num_classes=num_classes, in_features=num_features)

    elif 'MetaFormer' in model._get_name():
        if hasattr(model.head, 'flatten'):  # ConvNext case
            model.head.flatten = nn.Identity()
        model.head.global_pool = nn.Identity()
        model.head.drop = nn.Identity()
        del model.head.fc
        model.head.fc = MLDecoder(num_classes=num_classes, in_features=num_features)

    # maybe  and isinstance(model.head, (NormMlpClassifierHead, ClassifierHead) ?
    elif hasattr(model, 'head'):    # ClassifierHead, nn.Sequential
        input_fmt = getattr(model.head, 'input_fmt', 'NCHW')
        model.head = MLDecoderHead(num_features, num_classes)
        if hasattr(model, 'global_pool'):
            if(isinstance(model.global_pool, nn.Module)):
                model.global_pool = nn.Identity()
            else:
                model.global_pool = None
        if hasattr(model, 'head_drop'):
            model.head_drop = nn.Identity()

    elif 'MobileNetV3' in model._get_name(): # mobilenetv3 - conflict with efficientnet

        model.flatten = nn.Identity()
        del model.classifier
        model.classifier = MLDecoder(num_classes=num_classes, in_features=num_features)

    elif hasattr(model, 'global_pool') and hasattr(model, 'classifier'):  # EfficientNet
        model.global_pool = nn.Identity()
        del model.classifier
        model.classifier = MLDecoder(num_classes=num_classes, in_features=num_features)
    elif hasattr(model, 'global_pool') and hasattr(model, 'last_linear'):  # InceptionV4
        model.global_pool = nn.Identity()
        del model.last_linear
        model.last_linear = MLDecoder(num_classes=num_classes, in_features=num_features)

    elif hasattr(model, 'global_pool') and hasattr(model, 'classif'):  # InceptionResnetV2
        model.global_pool = nn.Identity()
        del model.classif
        model.classif = MLDecoder(num_classes=num_classes, in_features=num_features)

    else:
        raise Exception("Model code-writing is not aligned currently with ml-decoder")

    # FIXME does not work
    if hasattr(model, 'drop_rate'):  # Ml-Decoder has inner dropout
        model.drop_rate = 0
    return model


class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False) -> Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.self_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


# class ExtrapClasses(object):
#     def __init__(self, num_queries: int, group_size: int):
#         self.num_queries = num_queries
#         self.group_size = group_size
#
#     def __call__(self, h: torch.Tensor, class_embed_w: torch.Tensor, class_embed_b: torch.Tensor, out_extrap:
#     torch.Tensor):
#         # h = h.unsqueeze(-1).expand(-1, -1, -1, self.group_size)
#         h = h[..., None].repeat(1, 1, 1, self.group_size) # torch.Size([bs, 5, 768, groups])
#         w = class_embed_w.view((self.num_queries, h.shape[2], self.group_size))
#         out = (h * w).sum(dim=2) + class_embed_b
#         out = out.view((h.shape[0], self.group_size * self.num_queries))
#         return out


@torch.jit.script
class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor): # [B, K, C], [K, C, N/K], [B, K, N/K]
        for i in range(self.embed_len_decoder):
            h_i = h[:, i, :] # [B, 1, C]
            w_i = duplicate_pooling[i, :, :] # [1, C, N/K]
            out_extrap[:, i, :] = torch.matmul(h_i, w_i) # [B, 1, N/K]


class MLDecoderLegacy(nn.Module):
    def __init__(self, num_classes, num_of_groups=-1, decoder_embedding=768, in_features=2048, simple_group_fc = True):
        super(MLDecoderLegacy, self).__init__()
        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        if embed_len_decoder > num_classes:
            embed_len_decoder = num_classes
        self.embed_len_decoder = embed_len_decoder

        # switching to 768 initial embeddings
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding
        self.embed_standart = nn.Linear(in_features, decoder_embedding)

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 2048
        layer_decode = TransformerDecoderLayerOptimal(d_model=decoder_embedding,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)

        # non-learnable queries
        self.query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
        self.query_embed.requires_grad_(False)

        # group fully-connected
        self.simple_group_fc = simple_group_fc
        self.num_classes = num_classes
        self.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)
        self.duplicate_pooling = torch.nn.Parameter(
            torch.Tensor(embed_len_decoder, decoder_embedding, self.duplicate_factor))
        self.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))
        torch.nn.init.xavier_normal_(self.duplicate_pooling)
        torch.nn.init.constant_(self.duplicate_pooling_bias, 0)
        self.group_fc = None if simple_group_fc else GroupFC(embed_len_decoder)

    def forward(self, x):
        if len(x.shape) == 4:  # [bs,2048, 7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:  # [bs, 197,468]
            embedding_spatial = x
        embedding_spatial_786 = self.embed_standart(embedding_spatial)
        embedding_spatial_786 = torch.nn.functional.relu(embedding_spatial_786, inplace=True)

        bs = embedding_spatial_786.shape[0]
        query_embed = self.query_embed.weight
        # tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1))  # [embed_len_decoder, batch, 768]

        if(self.simple_group_fc):
            out_extrap = (h @ self.duplicate_pooling).permute(1,0,2) # [B, K, N/K]
        else:
            h = h.transpose(0, 1)
            out_extrap = torch.zeros(h.shape[0], h.shape[1], self.duplicate_factor, device=h.device, dtype=h.dtype)
            self.group_fc(h, self.duplicate_pooling, out_extrap)

        h_out = out_extrap.flatten(1)[:, :self.num_classes]
        h_out += self.duplicate_pooling_bias
        logits = h_out
        return logits


class CrossAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            attn_drop: float = 0.1,
            proj_drop: float = 0.1,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, x) -> torch.Tensor:
        K, _ = q.shape # [K, C]
        B, N, C = x.shape # [B, N, C]
        q = self.q(q).reshape(1, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [1, n_h, K, d_h]
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # [2, B, n_h, N, d_h]
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1) # [B, n_h, K, N]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v # [B, n_h, K, d_h]

        x = x.permute(2, 0, 1, 3).reshape(K, B, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GroupLinear(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        num_groups,
    ):
        super().__init__()
        self.num_classes = num_classes
        duplicate_factor = int(num_classes / num_groups + 0.999)
        self.weight = nn.Parameter(torch.Tensor(num_groups, dim, duplicate_factor))
        self.bias = nn.Parameter(torch.Tensor(num_classes))
        nn.init.xavier_normal_(self.weight)
        nn.init.constant_(self.bias, 0)
    
    def forward(self, x): # [B, K, C]
        x = (x @ self.weight).permute(1, 0, 2).flatten(1)[:, :self.num_classes]
        x += self.bias
        return x

class MLDecoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dim: int = 768,
        num_groups: int = 100,
        num_heads: int = 8,
        embed_drop: float = 0.1,
        embed_norm: bool = True,
        k_norm: bool = False,
        attn_drop: float = 0.1,
        mlp_ratio: float = 8/3,
        proj_drop: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        
    ):
        super().__init__()
        
        
        # non-learnable queries
        self.query_embed = nn.Embedding(num_groups, dim)
        self.query_embed.requires_grad_(False)
        self.embed_drop = nn.Dropout(embed_drop)
        self.embed_norm = norm_layer(dim)
        
        self.proj = nn.Linear(in_features, dim)
        self.act = act_layer()
        self.norm1 = norm_layer(dim)
        
        
        self.attn = CrossAttention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.fc = GroupLinear(dim, num_classes, num_groups)
    
    def forward(self, x):
        # BCHW to BNC
        if(len(x.shape) == 4):
            x = x.flatten(2).transpose(1, 2)
                
        x = self.act(self.proj(x))
        q = self.embed_norm(self.embed_drop(self.query_embed.weight))
        x = self.attn(q, self.norm1(x))# + q.unsqueeze(1)
        x = x + self.mlp(self.norm2(x))
        x = self.fc(x)
        return x
        

            

        

        
