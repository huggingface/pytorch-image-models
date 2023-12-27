import torch
import torch.nn
from torch import Tensor

from timm.layers import LayerNorm2d, Mlp, ConvNormAct

class ConvEmbed(nn.Module):
    def __init__(
        self,
        in_chs=3,
        out_chs=64,
        kernel_size=7,
        stride=4,
        padding=2,
        norm_layer=LayerNorm2d,
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.norm = norm_layer(out_chs) if norm_layer else nn.Identity()
        
    def forward(self, x: Tensor): # [B, C, H, W] -> [B, C, H, W]
        x = self.conv(x)
        x = self.norm(x)
        return x



class Attention(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        num_heads,
        kernel_size=3,
        stride_q=1,
        stride_kv=1,
        padding_q=1,
        padding_kv=1,
        qkv_bias=False,
        conv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        conv_norm_layer=nn.BatchNorm2d,
        conv_act_layer=nn.Identity(),
        
        cls_token=True
    ):
        assert out_chs % num_heads == 0, 'dim should be divisible by num_heads'
        self.out_chs = out_chs
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = out_chs ** -0.5
        
        self.conv_q = ConvNormAct(
            in_chs,
            out_chs,
            kernel_size,
            stride=stride_q,
            padding=padding_q,
            groups=in_chs,
            bias=conv_bias,
            norm_layer=conv_norm_layer,
            act_layer=conv_act_layer
        )
        
        self.conv_k = ConvNormAct(
            in_chs,
            out_chs * 2,
            kernel_size,
            stride=stride_kv,
            padding=padding_kv,
            groups=in_chs,
            bias=conv_bias,
            norm_layer=conv_norm_layer,
            act_layer=conv_act_layer
        )
        
        self.conv_v = ConvNormAct(
            in_chs,
            out_chs * 2,
            kernel_size,
            stride=stride_kv,
            padding=padding_kv,
            groups=in_chs,
            bias=conv_bias,
            norm_layer=conv_norm_layer,
            act_layer=conv_act_layer
        )
        
        # FIXME better way to do this? iirc 1 is better than 3
        self.proj_q = nn.Linear(in_chs, out_chs, bias=qkv_bias)
        self.proj_k = nn.Linear(in_chs, out_chs, bias=qkv_bias)
        self.proj_v = nn.Linear(in_chs, out_chs, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_chs, out_chs)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: Tensor):
        # [B, C_in, H, W] -> [B, H*W, C_out]
        q = self.conv_q(x).flatten(2).transpose(1, 2)
        k = self.conv_k(x).flatten(2).transpose(1, 2)
        v = self.conv_v(x).flatten(2).transpose(1, 2)
        
        # need to handle cls token here
        
        # [B, H*W, C_out] -> [B, H*W, n_h, d_h] -> [B, n_h, H*W, d_h]
        q = self.proj_q(q).reshape(B, q.shape[2], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.proj_k(k).reshape(B, k.shape[2], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.proj_v(v).reshape(B, v.shape[2], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # FIXME F.sdpa
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, self.out_chs)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class QuickGELU(nn.Module):
    def forward(self, x: Tensor):
        return x * torch.sigmoid(1.702 * x)
        
        