import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from timm.models._builder import build_model_with_cfg
from ._registry import register_model, generate_default_cfgs
from typing import List
import numpy as np
import math

__all__ = ['DCT2D']

# Helper Functions
mean=[[932.42657,-0.00260,0.33415,-0.02840,0.00003,-0.02792,-0.00183,0.00006,0.00032,0.03402,-0.00571,0.00020,0.00006,-0.00038,-0.00558,-0.00116,-0.00000,-0.00047,-0.00008,-0.00030,0.00942,0.00161,-0.00009,-0.00006,-0.00014,-0.00035,0.00001,-0.00220,0.00033,-0.00002,-0.00003,-0.00020,0.00007,-0.00000,0.00005,0.00293,-0.00004,0.00006,0.00019,0.00004,0.00006,-0.00015,-0.00002,0.00007,0.00010,-0.00004,0.00008,0.00000,0.00008,-0.00001,0.00015,0.00002,0.00007,0.00003,0.00004,-0.00001,0.00004,-0.00000,0.00002,-0.00000,-0.00008,-0.00000,-0.00003,0.00003],
[962.34735,-0.00428,0.09835,0.00152,-0.00009,0.00312,-0.00141,-0.00001,-0.00013,0.01050,0.00065,0.00006,-0.00000,0.00003,0.00264,0.00000,0.00001,0.00007,-0.00006,0.00003,0.00341,0.00163,0.00004,0.00003,-0.00001,0.00008,-0.00000,0.00090,0.00018,-0.00006,-0.00001,0.00007,-0.00003,-0.00001,0.00006,0.00084,-0.00000,-0.00001,0.00000,0.00004,-0.00001,-0.00002,0.00000,0.00001,0.00002,0.00001,0.00004,0.00011,0.00000,-0.00003,0.00011,-0.00002,0.00001,0.00001,0.00001,0.00001,-0.00007,-0.00003,0.00001,0.00000,0.00001,0.00002,0.00001,0.00000],
[1053.16101,-0.00213,-0.09207,0.00186,0.00013,0.00034,-0.00119,0.00002,0.00011,-0.00984,0.00046,-0.00007,-0.00001,-0.00005,0.00180,0.00042,0.00002,-0.00010,0.00004,0.00003,-0.00301,0.00125,-0.00002,-0.00003,-0.00001,-0.00001,-0.00001,0.00056,0.00021,0.00001,-0.00001,0.00002,-0.00001,-0.00001,0.00005,-0.00070,-0.00002,-0.00002,0.00005,-0.00004,-0.00000,0.00002,-0.00002,0.00001,0.00000,-0.00003,0.00004,0.00007,0.00001,0.00000,0.00013,-0.00000,0.00000,0.00002,-0.00000,-0.00001,-0.00004,-0.00003,0.00000,0.00001,-0.00001,0.00001,-0.00000,0.00000]]

var=[[270372.37500,6287.10645,5974.94043,1653.10889,1463.91748,1832.58997,755.92468,692.41528,648.57184,641.46881,285.79288,301.62100,380.43405,349.84027,374.15891,190.30960,190.76746,221.64578,200.82646,145.87979,126.92046,62.14622,67.75562,102.42001,129.74922,130.04631,103.12189,97.76417,53.17402,54.81048,73.48712,81.04342,69.35100,49.06024,33.96053,37.03279,20.48858,24.94830,33.90822,44.54912,47.56363,40.03160,30.43313,22.63899,26.53739,26.57114,21.84404,17.41557,15.18253,10.69678,11.24111,12.97229,15.08971,15.31646,8.90409,7.44213,6.66096,6.97719,4.17834,3.83882,4.51073,2.36646,2.41363,1.48266],
[18839.21094,321.70932,300.15259,77.47830,76.02293,89.04748,33.99642,34.74807,32.12333,28.19588,12.04675,14.26871,18.45779,16.59588,15.67892,7.37718,8.56312,10.28946,9.41013,6.69090,5.16453,2.55186,3.03073,4.66765,5.85418,5.74644,4.33702,3.66948,1.95107,2.26034,3.06380,3.50705,3.06359,2.19284,1.54454,1.57860,0.97078,1.13941,1.48653,1.89996,1.95544,1.64950,1.24754,0.93677,1.09267,1.09516,0.94163,0.78966,0.72489,0.50841,0.50909,0.55664,0.63111,0.64125,0.38847,0.33378,0.30918,0.33463,0.20875,0.19298,0.21903,0.13380,0.13444,0.09554],
[17127.39844,292.81421,271.45209,66.64056,63.60253,76.35437,28.06587,27.84831,25.96656,23.60370,9.99173,11.34992,14.46955,12.92553,12.69353,5.91537,6.60187,7.90891,7.32825,5.32785,4.29660,2.13459,2.44135,3.66021,4.50335,4.38959,3.34888,2.97181,1.60633,1.77010,2.35118,2.69018,2.38189,1.74596,1.26014,1.31684,0.79327,0.92046,1.17670,1.47609,1.50914,1.28725,0.99898,0.74832,0.85736,0.85800,0.74663,0.63508,0.58748,0.41098,0.41121,0.44663,0.50277,0.51519,0.31729,0.27336,0.25399,0.27241,0.17353,0.16255,0.18440,0.11602,0.11511,0.08450]]


def _zigzag_permutation(rows: int, cols: int) -> List[int]:
    idx_matrix = np.arange(0, rows * cols, 1).reshape(rows, cols).tolist()
    dia = [[] for _ in range(rows + cols - 1)]
    zigzag = []
    for i in range(rows):
        for j in range(cols):
            s = i + j
            if s % 2 == 0:
                dia[s].insert(0, idx_matrix[i][j])
            else:
                dia[s].append(idx_matrix[i][j])
    for d in dia:
        zigzag.extend(d)
    return zigzag


def _dct_kernel_type_2(kernel_size: int, orthonormal: bool, device=None, dtype=None) -> torch.Tensor:
    factory_kwargs = dict(device=device, dtype=dtype)
    x = torch.eye(kernel_size, **factory_kwargs)
    v = x.clone().contiguous().view(-1, kernel_size)
    v = torch.cat([v, v.flip([1])], dim=-1)
    v = torch.fft.fft(v, dim=-1)[:, :kernel_size]
    try:
        k = torch.tensor(-1j, **factory_kwargs) * torch.pi * torch.arange(kernel_size, **factory_kwargs)[None, :]
    except:
        k = torch.tensor(-1j, **factory_kwargs) * math.pi * torch.arange(kernel_size, **factory_kwargs)[None, :]
    k = torch.exp(k / (kernel_size * 2))
    v = v * k
    v = v.real
    if orthonormal:
        v[:, 0] = v[:, 0] * torch.sqrt(torch.tensor(1 / (kernel_size * 4), **factory_kwargs))
        v[:, 1:] = v[:, 1:] * torch.sqrt(torch.tensor(1 / (kernel_size * 2), **factory_kwargs))
    v = v.contiguous().view(*x.shape)
    return v


def _dct_kernel_type_3(kernel_size: int, orthonormal: bool, device=None, dtype=None) -> torch.Tensor:
    return torch.linalg.inv(_dct_kernel_type_2(kernel_size, orthonormal, device, dtype))


class _DCT1D(nn.Module):
    def __init__(self, kernel_size: int, kernel_type: int = 2, orthonormal: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(_DCT1D, self).__init__()
        kernel = {'2': _dct_kernel_type_2, '3': _dct_kernel_type_3}
        self.weights = nn.Parameter(kernel[f'{kernel_type}'](kernel_size, orthonormal, **factory_kwargs).T, False)
        self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weights, self.bias)


class _DCT2D(nn.Module):
    def __init__(self, kernel_size: int, kernel_type: int = 2, orthonormal: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(_DCT2D, self).__init__()
        self.transform = _DCT1D(kernel_size, kernel_type, orthonormal, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(self.transform(x).transpose(-1, -2)).transpose(-1, -2)


class Learnable_DCT2D(nn.Module):
    def __init__(self, kernel_size: int, kernel_type: int = 2, orthonormal: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(Learnable_DCT2D, self).__init__()
        self.k = kernel_size
        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), stride=(kernel_size, kernel_size))
        self.transform = _DCT2D(kernel_size, kernel_type, orthonormal, **factory_kwargs)
        self.permutation = _zigzag_permutation(kernel_size, kernel_size)
        self.Y_Conv = nn.Conv2d(kernel_size ** 2, 24, kernel_size=1, padding=0)
        self.Cb_Conv = nn.Conv2d(kernel_size ** 2, 4, kernel_size=1, padding=0)
        self.Cr_Conv = nn.Conv2d(kernel_size ** 2, 4, kernel_size=1, padding=0)
        self.mean = torch.tensor(mean, requires_grad=False)
        self.var = torch.tensor(var, requires_grad=False)
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], requires_grad=False)
        self.imagenet_var = torch.tensor([0.229, 0.224, 0.225], requires_grad=False)

    def denormalize(self, x):
        x = x.multiply(self.imagenet_var.to(x.device)).add_(self.imagenet_mean.to(x.device)) * 255
        return x

    def rgb2ycbcr(self, x):
        y = (x[:, :, :, 0] * 0.299) + (x[:, :, :, 1] * 0.587) + (x[:, :, :, 2] * 0.114)
        cb = 0.564 * (x[:, :, :, 2] - y) + 128
        cr = 0.713 * (x[:, :, :, 0] - y) + 128
        x = torch.stack([y, cb, cr], dim=-1)
        return x

    def frequncy_normalize(self, x):
        x[:, 0, ].sub_(self.mean.to(x.device)[0]).div_((self.var.to(x.device)[0] ** 0.5 + 1e-8))
        x[:, 1, ].sub_(self.mean.to(x.device)[1]).div_((self.var.to(x.device)[1] ** 0.5 + 1e-8))
        x[:, 2, ].sub_(self.mean.to(x.device)[2]).div_((self.var.to(x.device)[2] ** 0.5 + 1e-8))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.denormalize(x)
        x = self.rgb2ycbcr(x)
        x = x.permute(0, 3, 1, 2)
        x = self.unfold(x).transpose(-1, -2)
        x = x.reshape(b, h // self.k, w // self.k, c, self.k, self.k)
        x = self.transform(x)
        x = x.reshape(-1, c, self.k * self.k)
        x = x[:, :, self.permutation]
        x = self.frequncy_normalize(x)
        x = x.reshape(b, h // self.k, w // self.k, c, -1)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        x_Y = self.Y_Conv(x[:, 0, ])
        x_Cb = self.Cb_Conv(x[:, 1, ])
        x_Cr = self.Cr_Conv(x[:, 2, ])
        x = torch.cat([x_Y, x_Cb, x_Cr], dim=1)
        return x


class DCT2D(nn.Module):
    def __init__(self, kernel_size: int, kernel_type: int = 2, orthonormal: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(DCT2D, self).__init__()
        self.k = kernel_size
        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), stride=(kernel_size, kernel_size))
        self.transform = _DCT2D(kernel_size, kernel_type, orthonormal, **factory_kwargs)
        self.permutation = _zigzag_permutation(kernel_size, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.unfold(x).transpose(-1, -2)
        x = x.reshape(b, h // self.k, w // self.k, c, self.k, self.k)
        x = self.transform(x)
        x = x.reshape(-1, c, self.k * self.k)
        x = x[:, :, self.permutation]
        x = x.reshape(b * (h // self.k) * (w // self.k), c, -1)

        mean_list = torch.zeros([3, 64])
        var_list = torch.zeros([3, 64])
        mean_list[0] = torch.mean(x[:, 0, ], dim=0)
        mean_list[1] = torch.mean(x[:, 1, ], dim=0)
        mean_list[2] = torch.mean(x[:, 2, ], dim=0)
        var_list[0] = torch.var(x[:, 0, ], dim=0)
        var_list[1] = torch.var(x[:, 1, ], dim=0)
        var_list[2] = torch.var(x[:, 2, ], dim=0)
        return mean_list, var_list


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attention = Spatial_Attention()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        # Spatial Attention logic
        attention = self.attention(x)

        # [Fix] nn.UpsamplingBilinear2d 클래스 생성 -> F.interpolate 함수 사용
        # align_corners=False가 최신 기본값에 가깝습니다. (성능 차이는 미미함)
        attention = F.interpolate(attention, size=x.shape[2:], mode='bilinear', align_corners=False)

        x = x * attention

        x = input + self.drop_path(x)
        return x


class Spatial_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        # downsample=False, dropout=0. by default
        self.attention = TransformerBlock(1, 1, heads=1, dim_head=1, img_size=[7, 7], downsample=False)

    def forward(self, x):
        x_avg = x.mean([1]).unsqueeze(1)
        x_max = x.max(dim=1).values.unsqueeze(1)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.attention(x)
        return x


class TransformerBlock(nn.Module):
    """
    Refactored TransformerBlock without einops and PreNorm class wrapper.
    Manual reshaping is performed in forward().
    """

    def __init__(self, inp, oup, heads=8, dim_head=32, img_size=None, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.downsample = downsample
        self.ih, self.iw = img_size

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        else:
            # [Fix] JIT 컴파일 에러 방지: 사용하지 않더라도 속성을 정의해야 함
            self.pool1 = nn.Identity()
            self.pool2 = nn.Identity()
            self.proj = nn.Identity()

        # Attention block components
        # Note: In old code, PreNorm wrapped Attention. Here we split them.
        self.attn_norm = nn.LayerNorm(inp)
        self.attn = Attention(inp, oup, heads, dim_head, dropout)

        # FeedForward block components
        self.ff_norm = nn.LayerNorm(oup)
        self.ff = FeedForward(oup, hidden_dim, dropout)

    def forward(self, x):
        # x shape: (B, C, H, W)
        if self.downsample:
            # Identity path with projection
            shortcut = self.proj(self.pool1(x))

            # Attention path
            x_t = self.pool2(x)
            B, C, H, W = x_t.shape

            # Flatten spatial: (B, C, H, W) -> (B, H*W, C)
            x_t = x_t.flatten(2).transpose(1, 2)

            # PreNorm -> Attention
            x_t = self.attn_norm(x_t)
            x_t = self.attn(x_t)

            # Unflatten: (B, H*W, C) -> (B, C, H, W)
            x_t = x_t.transpose(1, 2).reshape(B, C, H, W)

            x = shortcut + x_t
        else:
            # Standard PreNorm Residual Attention
            B, C, H, W = x.shape
            shortcut = x

            # Flatten
            x_t = x.flatten(2).transpose(1, 2)

            # PreNorm -> Attention
            x_t = self.attn_norm(x_t)
            x_t = self.attn(x_t)

            # Unflatten
            x_t = x_t.transpose(1, 2).reshape(B, C, H, W)

            x = shortcut + x_t

        # FeedForward Block
        B, C, H, W = x.shape
        shortcut = x

        # Flatten
        x_t = x.flatten(2).transpose(1, 2)

        # PreNorm -> FeedForward
        x_t = self.ff_norm(x_t)
        x_t = self.ff(x_t)

        # Unflatten
        x_t = x_t.transpose(1, 2).reshape(B, C, H, W)

        x = shortcut + x_t

        return x


class Attention(nn.Module):
    """
    Refactored Attention without einops.rearrange.
    """

    def __init__(self, inp, oup, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.pos_embed = PosCNN(in_chans=inp)

    def forward(self, x):
        # x shape: (B, N, C)
        # Positional Embedding (expects B, N, C -> internally converts to spatial)
        x = self.pos_embed(x)

        B, N, C = x.shape

        # Generate Q, K, V
        # qkv: (B, N, 3 * heads * dim_head)
        qkv = self.to_qkv(x)

        # Reshape to (B, N, 3, heads, dim_head) and permute to (3, B, heads, N, dim_head)
        qkv = qkv.reshape(B, N, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, heads, N, dim_head)

        # Attention Score
        # (B, heads, N, dim_head) @ (B, heads, dim_head, N) -> (B, heads, N, N)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        # Weighted Sum
        # (B, heads, N, N) @ (B, heads, N, dim_head) -> (B, heads, N, dim_head)
        out = torch.matmul(attn, v)

        # Rearrange output: (B, heads, N, dim_head) -> (B, N, heads*dim_head)
        out = out.transpose(1, 2).reshape(B, N, -1)

        out = self.to_out(out)
        return out


class CSATv2(nn.Module):
    def __init__(
            self,
            img_size=512,
            num_classes=1000,
            in_chans=3,
            drop_path_rate=0.0,
            global_pool='avg',
            **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.global_pool = global_pool

        if isinstance(img_size, (tuple, list)):
            img_size = img_size[0]

        self.img_size = img_size

        dims = [32, 72, 168, 386]
        channel_order = "channels_first"
        depths = [2, 2, 6, 4]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.dct = Learnable_DCT2D(8)

        self.stages1 = nn.Sequential(
            Block(dim=dims[0], drop_path=dp_rates[0]),
            Block(dim=dims[0], drop_path=dp_rates[1]),
            LayerNorm(dims[0], eps=1e-6, data_format=channel_order),
            nn.Conv2d(dims[0], dims[0 + 1], kernel_size=2, stride=2),
        )

        self.stages2 = nn.Sequential(
            Block(dim=dims[1], drop_path=dp_rates[0]),
            Block(dim=dims[1], drop_path=dp_rates[1]),
            LayerNorm(dims[1], eps=1e-6, data_format=channel_order),
            nn.Conv2d(dims[1], dims[1 + 1], kernel_size=2, stride=2),
        )

        self.stages3 = nn.Sequential(
            Block(dim=dims[2], drop_path=dp_rates[0]),
            Block(dim=dims[2], drop_path=dp_rates[1]),
            Block(dim=dims[2], drop_path=dp_rates[2]),
            Block(dim=dims[2], drop_path=dp_rates[3]),
            Block(dim=dims[2], drop_path=dp_rates[4]),
            Block(dim=dims[2], drop_path=dp_rates[5]),
            TransformerBlock(
                inp=dims[2],
                oup=dims[2],
                img_size=[int(self.img_size / 32), int(self.img_size / 32)],
            ),
            TransformerBlock(
                inp=dims[2],
                oup=dims[2],
                img_size=[int(self.img_size / 32), int(self.img_size / 32)],
            ),
            LayerNorm(dims[2], eps=1e-6, data_format=channel_order),
            nn.Conv2d(dims[2], dims[2 + 1], kernel_size=2, stride=2),
        )

        self.stages4 = nn.Sequential(
            Block(dim=dims[3], drop_path=dp_rates[0]),
            Block(dim=dims[3], drop_path=dp_rates[1]),
            Block(dim=dims[3], drop_path=dp_rates[2]),
            Block(dim=dims[3], drop_path=dp_rates[3]),
            TransformerBlock(
                inp=dims[3],
                oup=dims[3],
                img_size=[int(self.img_size / 64), int(self.img_size / 64)],
            ),
            TransformerBlock(
                inp=dims[3],
                oup=dims[3],
                img_size=[int(self.img_size / 64), int(self.img_size / 64)],
            ),
        )

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.dct(x)
        x = self.stages1(x)
        x = self.stages2(x)
        x = self.stages3(x)
        x = self.stages4(x)
        x = x.mean(dim=(-2, -1))
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if pre_logits:
            return x
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


# --- Components like LayerNorm, GRN, DropPath, FeedForward, PosCNN, trunc_normal_ ---
# (이 부분은 einops와 무관하므로 위 코드와 동일하게 유지합니다. 여기서는 공간 절약을 위해 생략)
# 기존 코드의 LayerNorm, GRN, DropPath, FeedForward, PosCNN, trunc_normal_ 함수를 그대로 사용하세요.

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            # [Fix] elif -> else로 변경
            # JIT이 "모든 경로에서 Tensor가 반환됨"을 알 수 있게 함
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PosCNN(nn.Module):
    def __init__(self, in_chans):
        super(PosCNN, self).__init__()
        self.proj = nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, padding=1, bias=True, groups=in_chans)

    def forward(self, x):
        B, N, C = x.shape
        feat_token = x
        H, W = int(N ** 0.5), int(N ** 0.5)
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        return x


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


default_cfgs = generate_default_cfgs({
    'csatv2': {
        'url': 'https://huggingface.co/Hyunil/CSATv2/resolve/main/CSATv2_ImageNet_timm.pth',
        'num_classes': 1000,
        'input_size': (3, 512, 512),
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'interpolation': 'bilinear',
        'crop_pct': 1.0,
    },
})


def _create_csatv2(variant: str, pretrained: bool = False, **kwargs) -> CSATv2:
    return build_model_with_cfg(
        CSATv2,
        variant,
        pretrained,
        default_cfg=default_cfgs[variant],
        **kwargs,
    )


@register_model
def csatv2(pretrained: bool = False, **kwargs) -> CSATv2:
    model_args = dict(
        img_size=kwargs.pop('img_size', 512),
        num_classes=kwargs.pop('num_classes', 1000),
    )
    return _create_csatv2('csatv2', pretrained, **dict(model_args, **kwargs))
