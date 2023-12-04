import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

def prepare_input(self, x):
    """
    Prepares the input tensor for different neural network architectures (Transformers and CNNs).
    
    This function adjusts the shape of the input tensor based on its dimensionality.
    It supports input tensors for Transformers (3D) and CNNs (4D), ensuring they are
    correctly formatted for these architectures.

    For a Transformer, it expects a tensor of shape (B, N, d), where B is the batch size,
    N are patch tokens, and d is the depth (channels). The tensor is returned as is.

    For a CNN, it expects a tensor of shape (B, d, H, W), where B is the batch size,
    d is the depth (channels), H is the height, and W is the width. The tensor is reshaped
    and permuted to the shape (B, H*W, d) to match CNN input requirements.

    Parameters:
    x (torch.Tensor): The input tensor to be preprocessed.
    """
    if len(x.shape) == 3: # Transformer
        # Input tensor dimensions:
        # x: (B, N, d), where B is batch size, N are patch tokens, d is depth (channels)
        B, N, d = x.shape
        return x
    if len(x.shape) == 4: # CNN
        # Input tensor dimensions:
        # x: (B, d, H, W), where B is batch size, d is depth (channels), H is height, and W is width
        B, d, H, W = x.shape
        x = x.reshape(B, d, H*W).permute(0, 2, 1) # (B, d, H, W) -> (B, d, H*W) -> (B, H*W, d)
        return x
    else:
        raise ValueError(f"Unsupported number of dimensions in input tensor: {len(x.shape)}")

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn

class ViTPooling(nn.Module):
    """
    Multi-Head Attention proposed in "Attention Is All You Need"
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)
    Args:
        d_model (int): The dimension of keys / values / quries (default: 512)
        num_heads (int): The number of attention heads. (default: 8)
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): In transformer, three different ways:
            Case 1: come from previoys decoder layer
            Case 2: come from the input embedding
            Case 3: come from the output embedding (masked)
        - **key** (batch, k_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)
        - **value** (batch, v_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)
        - **mask** (-): tensor containing indices to be masked
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super(ViTPooling, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:

        x = prepare_input(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        query = self.query_proj(x).view(B, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
        key = self.key_proj(x).view(B, -1, self.num_heads, self.d_head)      # BxK_LENxNxD
        value = self.value_proj(x).view(B, -1, self.num_heads, self.d_head)  # BxV_LENxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(B * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(B * self.num_heads, -1, self.d_head)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(B * self.num_heads, -1, self.d_head)  # BNxV_LENxD

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, B, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(B, -1, self.num_heads * self.d_head)  # BxTxND

        return context[:, 0]