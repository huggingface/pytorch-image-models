from typing import Any, Literal, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_activation_fn
from torch.jit import Final

from timm.layers import Mlp, use_fused_attn
from timm.layers.classifier import _create_pool


class MLDecoderHead(nn.Module):
    """MLDecoder wrapper with forward compatible with ClassifierHead"""

    def __init__(self, head, in_features, num_classes, pool_type='avg', use_conv=False, input_fmt='NCHW'):
        super(MLDecoderHead, self).__init__()
        self.in_features = in_features
        self.use_conv = use_conv
        self.input_fmt = input_fmt

        self.global_pool, num_pooled_features = _create_pool(in_features, num_classes, pool_type, use_conv=use_conv, input_fmt=input_fmt)
        self.head = head
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()


    def reset(self, num_classes, global_pool=None):
        if global_pool is not None:
            if global_pool != self.global_pool.pool_type:
                self.global_pool, _ = _create_pool(self.in_features, num_classes, global_pool, use_conv=self.use_conv, input_fmt=self.input_fmt)
            self.flatten = nn.Flatten(1) if self.use_conv and global_pool else nn.Identity()
        num_pooled_features = self.in_features * self.global_pool.feat_mult()
        # TODO fix this it is incorrect, need to impl a reset for mldecoder itself i think
        self.head = type(self.head)(in_features=in_features, num_classes=num_classes)


    def forward(self, x, q=None, pre_logits: bool = False):
        # pool for compatibility with ClassifierHead
        if self.input_fmt == 'NHWC':
            x = x.permute(0, 3, 1, 2)
        if pre_logits:
            x = self.global_pool(x)
            return x.flatten(1)
        else:
            x = self.head(x, q=q)
            return self.flatten(x)

def add_ml_decoder_head(model, head_version='new', **kwargs):
    # ignore CoaT, crossvit
    # ignore distillation models: deit_distilled, efficientformer V2
    num_classes = model.num_classes
    num_features = model.num_features
    
    if head_version == 'old':
        head_fn = MLDecoderLegacy
    else:
        head_fn = MLDecoder
    
    head = head_fn(num_features, num_classes, **kwargs)

    assert num_classes > 0, "MLDecoder requires a model to have num_classes > 0"

    if hasattr(model, 'global_pool') and hasattr(model, 'fc'):  # most CNN models, like Resnet50
        model.global_pool = nn.Identity()
        del model.fc

        model.fc = head

    elif hasattr(model, 'fc_norm') or 'Cait' in model._get_name(): # ViT, BEiT, EVA
        model.global_pool = None # disable any pooling, model instantiation leaves 1 norm layer after features, [B, n + K x K, C]
        if hasattr(model, 'attn_pool'):
            model.attn_pool = None
        model.head_drop = nn.Identity()
        model.head = head

    elif 'MetaFormer' in model._get_name():
        '''
        if hasattr(model.head, 'flatten'):  # ConvNext case
            model.head.flatten = nn.Identity()
        model.head.global_pool = nn.Identity()
        model.head.drop = nn.Identity()
        del model.head.fc
        model.head.fc = head
        '''
        target_type = type(model.forward_head)
        model.head = MLDecoderHead(head, num_features, num_classes)
        model.forward_head = target_type(model.head.forward, model, type(model))

    # maybe  and isinstance(model.head, (NormMlpClassifierHead, ClassifierHead) ?
    elif hasattr(model, 'head'):    # ClassifierHead, nn.Sequential
        input_fmt = getattr(model.head, 'input_fmt', 'NCHW')
        model.head = MLDecoderHead(head, num_features, num_classes)
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
        model.classifier = head

    elif hasattr(model, 'global_pool') and hasattr(model, 'classifier'):  # EfficientNet
        model.global_pool = nn.Identity()
        del model.classifier
        model.classifier = head
    elif hasattr(model, 'global_pool') and hasattr(model, 'last_linear'):  # InceptionV4
        model.global_pool = nn.Identity()
        del model.last_linear
        model.last_linear = head

    elif hasattr(model, 'global_pool') and hasattr(model, 'classif'):  # InceptionResnetV2
        model.global_pool = nn.Identity()
        del model.classif
        model.classif = head

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
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dim: int = 768,
        num_groups: int = 100,
        simple_group_fc = True,
    ):
        super(MLDecoderLegacy, self).__init__()
        embed_len_decoder = 100 if num_groups < 0 else num_groups
        if embed_len_decoder > num_classes:
            embed_len_decoder = num_classes
        self.embed_len_decoder = embed_len_decoder

        # switching to 768 initial embeddings
        dim = 768 if dim < 0 else dim
        self.embed_standart = nn.Linear(in_features, dim)

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 2048
        layer_decode = TransformerDecoderLayerOptimal(d_model=dim,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)

        # non-learnable queries
        self.query_embed = nn.Embedding(embed_len_decoder, dim)
        self.query_embed.requires_grad_(False)

        # group fully-connected
        self.simple_group_fc = simple_group_fc
        self.num_classes = num_classes
        self.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)
        self.duplicate_pooling = torch.nn.Parameter(
            torch.Tensor(embed_len_decoder, dim, self.duplicate_factor))
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
            query_dim: Optional[int] = None,
            kv_dim: Optional[int] = None,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            attn_drop: float = 0.1,
            proj_drop: float = 0.1,
            use_out_proj: bool = True,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.query_dim = query_dim or dim
        self.kv_dim = kv_dim or dim
        
        self.q = nn.Linear(self.query_dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(self.kv_dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) if use_out_proj else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop) if use_out_proj else nn.Identity()

    def forward(self, q, x) -> torch.Tensor:
        K, _ = q.shape # [K, C_q]
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
        shared: bool = False,
    ):
        super().__init__()
        # 1 group for all queries (shared_fc, use with class_embed) or 1 group for each query (default, used in paper)
        self.num_classes = num_classes
        duplicate_factor = int(num_classes / num_groups + 0.999)
        num_biases = duplicate_factor if shared else num_classes
        num_groups = 1 if shared else num_groups
        self.weight = nn.Parameter(torch.Tensor(num_groups, dim, duplicate_factor))
        self.bias = nn.Parameter(torch.Tensor(num_biases))
        nn.init.xavier_normal_(self.weight)
        nn.init.constant_(self.bias, 0)
    
    def forward(self, x): # [K, B, C]
        x = (x @ self.weight).permute(1, 0, 2).flatten(1)[:, :self.num_classes].contiguous()
        x += self.bias
        return x

class Zero(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> int:
        return 0

class MLDecoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dim: int = 768,
        num_groups: int = 100,
        num_heads: int = 8,
        class_embed: Optional[torch.Tensor] = None,
        class_embed_merge: Literal['', 'add', 'concat'] = 'add',
        learnable_embed: bool = False,
        learnable_class_embed: bool = False,
        embed_drop: float = 0.1,
        embed_norm: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.1,
        mlp_ratio: float = 8/3,
        use_mlp: bool = True,
        proj_drop: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        post_input_proj_act: bool = True,
        use_input_proj: bool = True,
        attn_out_proj: bool = True,
        shared_fc: bool = False,
    ):
        super().__init__()
        self.have_class_embed = class_embed is not None
        self.class_embed_merge = class_embed_merge
        self.class_embed = None
        self.query_embed = None
        self.query_dim = 0
        self.shared_fc = shared_fc
        num_groups = num_classes if num_groups < 1 else num_groups
        
        # case using class embed
        if self.have_class_embed:
            assert len(class_embed) == num_classes, 'ML-Decoder got class_embed where dim 0 != num_classes'
            class_embed = class_embed.clone().detach() # copy instead of reference, detach gradient flow
            self.query_dim += class_embed.shape[-1] # [K , D]
            duplicate_factor = int(num_classes / num_groups + 0.999)
            class_embed_pad_length = (duplicate_factor - num_classes % duplicate_factor) % duplicate_factor
            
            # pad and reshape into groups
            class_embed = torch.cat([class_embed, torch.zeros(class_embed_pad_length, class_embed.shape[1])])
            class_embed = class_embed.reshape(num_groups, duplicate_factor, -1)
            
            # TODO different merging strategies
            # reduce each group to a single embed with mean
            class_embed = class_embed.mean(1)
            self.class_embed = nn.Embedding.from_pretrained(class_embed)
            
            
            # TODO can use tensor instead of nn.Embedding and simply register as either a parameter or a buffer for learnability
            self.class_embed.requires_grad_(learnable_class_embed)
            
            # resolve query embed
            # case add: use same shape as class embed
            if class_embed_merge == 'add':
                self.query_embed = nn.Embedding(num_groups, class_embed.shape[-1])
                self.query_embed.requires_grad_(learnable_embed)
            # case concat: use default shape
            elif class_embed_merge == 'concat':
                self.query_dim += dim
                self.query_embed = nn.Embedding(num_groups, dim)
                self.query_embed.requires_grad_(learnable_embed)
            # no query embed otherwise, only class embed

        # case no class embed, only using query embed
        else:
            self.query_dim += dim
            self.query_embed = nn.Embedding(num_groups, dim)
            # TODO can use tensor instead of nn.Embedding and simply register as either a parameter or a buffer for learnability
            self.query_embed.requires_grad_(learnable_embed)
                    
        self.embed_drop = nn.Dropout(embed_drop)
        self.embed_norm = norm_layer(self.query_dim)
        

        self.proj = nn.Linear(in_features, dim) if use_input_proj else nn.Identity()
        self.act = act_layer() if post_input_proj_act else nn.Identity()
        self.norm1 = norm_layer(in_features if use_input_proj else dim)
        
        
        self.attn = CrossAttention(
            dim, 
            query_dim=self.query_dim, 
            kv_dim=dim if use_input_proj else in_features, 
            num_heads=num_heads,
            use_out_proj=attn_out_proj)
        self.norm2 = norm_layer(dim) if use_mlp else nn.Identity()
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        ) if use_mlp else Zero()
        
        self.fc = GroupLinear(dim, num_classes, num_groups, shared=self.shared_fc)
    
    def _resolve_query(self, q):
        if q is not None:
            return q
        if not self.have_class_embed:
            return self.query_embed.weight
        else:
            if self.class_embed_merge == 'add':
                return self.query_embed.weight + self.class_embed.weight
            elif self.class_embed_merge == 'concat':
                return torch.cat([x.weight for x in [self.query_embed, self.class_embed] if x is not None], dim=1)
            else:
                return self.class_embed.weight
    
    def forward(self, x, q=None):
        # BCHW to BNC
        if(len(x.shape) == 4):
            x = x.flatten(2).transpose(1, 2)

        x = self.act(self.proj(x))
        q = self._resolve_query(q)
        q = self.embed_norm(self.embed_drop(q))
        x = self.attn(q, self.norm1(x))# + q.unsqueeze(1)
        x = x + self.mlp(self.norm2(x))
        x = self.fc(x)
        return x
        
