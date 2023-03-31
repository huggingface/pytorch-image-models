"""
An implementation of ViG Model as defined in:
Vision GNN: An Image is Worth Graph of Nodes.
https://arxiv.org/abs/2206.00272
The imagenet-21k pretrained weights:
https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/pyramid-vig/pvig_m_im21k_90e.pth
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, DyGraphConv2d
from timm.layers.pos_embed_sincos import build_sincos2d_pos_embed
from .helpers import load_pretrained, build_model_with_cfgx
from .registry import register_model
from .fx_features import register_notrace_function, register_notrace_module


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.convs.0', 'classifier': 'prediction.4',
        'min_input_size': (3, 224, 224),
        **kwargs
    }


default_cfgs = {
    'pvig_ti_224_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/pyramid-vig/pvig_ti_78.5.pth.tar',
    ),
    'pvig_s_224_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/pyramid-vig/pvig_s_82.1.pth.tar',
    ),
    'pvig_m_224_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/pyramid-vig/pvig_m_83.1.pth.tar',
    ),
    'pvig_b_224_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/pyramid-vig/pvig_b_83.66.pth.tar',
    ),
}



def get_2d_relative_pos_embed(embed_dim, grid_size):
    """
    relative position embedding
    References: https://arxiv.org/abs/2009.13658
    """
    pos_embed = build_sincos2d_pos_embed([grid_size, grid_size], embed_dim)
    relative_pos = 2 * torch.matmul(pos_embed, pos_embed.transpose(0, 1)) / pos_embed.shape[1]
    return relative_pos


@register_notrace_module  # reason: FX can't symbolically trace control flow in forward method
class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='mr', act_layer=nn.GELU, norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act_layer, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if relative_pos:
            relative_pos_tensor = get_2d_relative_pos_embed(in_channels,
                    int(n**0.5)).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.register_buffer('relative_pos', -relative_pos_tensor.squeeze(1))
        else:
            self.relative_pos = None

    @register_notrace_function  # reason: int argument is a Proxy
    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
            act_layer=nn.GELU, drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act_layer=nn.GELU):
        super().__init__()        
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DeepGCN(torch.nn.Module):
    def __init__(self, opt, num_classes=1000, in_chans=3):
        super(DeepGCN, self).__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        k = opt.k
        act_layer = nn.GELU
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        drop_path = opt.drop_path
        channels = opt.channels
        self.num_features = channels[-1]  # num_features for consistency with other models
        
        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay 
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)
        
        self.stem = Stem(in_dim=in_chans, out_dim=channels[0], act_layer=act_layer)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224//4, 224//4))
        HW = 224 // 4 * 224 // 4

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i-1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act_layer,
                                norm, bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                relative_pos=True),
                          FFN(channels[i], channels[i] * 4, act_layer=act_layer, drop_path=dpr[idx])
                         )]
                idx += 1
        self.backbone = Seq(*self.backbone)

        if num_classes > 0:
            self.prediction = Seq(nn.Conv2d(self.num_features, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, num_classes, 1, bias=True))
        else:
            self.prediction = nn.Identity()
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    @register_notrace_function  # reason: int argument is a Proxy
    def _get_pos_embed(self, pos_embed, H, W):
        if pos_embed is None or (H == pos_embed.size(-2) and W == pos_embed.size(-1)):
            return pos_embed
        else:
            return F.interpolate(pos_embed, size=(H, W), mode="bicubic")

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        if num_classes > 0:
            self.prediction = Seq(nn.Conv2d(self.num_features, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, num_classes, 1, bias=True))
        else:
            self.prediction = nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        B, C, H, W = x.shape
        x = x + self._get_pos_embed(self.pos_embed, H, W)
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)


def _create_pvig(variant, opt, pretrained=False, **kwargs):
    """
    Constructs a GhostNet model
    """
    model_kwargs = dict(
        opt=opt,
        **kwargs,
    )
    return build_model_with_cfg(
        DeepGCN, variant, pretrained,
        feature_cfg=dict(flatten_sequential=True),
        **model_kwargs)


@register_model
def pvig_ti_224_gelu(pretrained=False, num_classes=1000, **kwargs):
    class OptInit:
        def __init__(self, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 6, 2] # number of basic blocks in the backbone
            self.channels = [48, 96, 240, 384] # number of channels of deep features

    opt = OptInit(**kwargs)
    model = _create_pvig('pvig_ti_224_gelu', opt, pretrained, num_classes=num_classes)
    model.default_cfg = default_cfgs['pvig_ti_224_gelu']
    return model


@register_model
def pvig_s_224_gelu(pretrained=False, num_classes=1000, **kwargs):
    class OptInit:
        def __init__(self, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 6, 2] # number of basic blocks in the backbone
            self.channels = [80, 160, 400, 640] # number of channels of deep features

    opt = OptInit(**kwargs)
    model = _create_pvig('pvig_s_224_gelu', opt, pretrained, num_classes=num_classes)
    model.default_cfg = default_cfgs['pvig_s_224_gelu']
    return model


@register_model
def pvig_m_224_gelu(pretrained=False, num_classes=1000, **kwargs):
    class OptInit:
        def __init__(self, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,16,2] # number of basic blocks in the backbone
            self.channels = [96, 192, 384, 768] # number of channels of deep features

    opt = OptInit(**kwargs)
    model = _create_pvig('pvig_m_224_gelu', opt, pretrained, num_classes=num_classes)
    model.default_cfg = default_cfgs['pvig_m_224_gelu']
    return model


@register_model
def pvig_b_224_gelu(pretrained=False, num_classes=1000, **kwargs):
    class OptInit:
        def __init__(self, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,18,2] # number of basic blocks in the backbone
            self.channels = [128, 256, 512, 1024] # number of channels of deep features

    opt = OptInit(**kwargs)
    model = _create_pvig('pvig_b_224_gelu', opt, pretrained, num_classes=num_classes)
    model.default_cfg = default_cfgs['pvig_b_224_gelu']
    return model
