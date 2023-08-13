"""
An implementation of GhostNet Model as defined in:
GhostNetV2: Enhance Cheap Operation with Long-Range Attention. https://proceedings.neurips.cc/paper_files/paper/2022/file/40b60852a4abdaa696b5a1a78da34635-Paper-Conference.pdf
The train script of the model is similar to that of GhostNet.
Original model: https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnetv2_pytorch/model/ghostnetv2_torch.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d, Linear, make_divisible
from ._builder import build_model_with_cfg
from ._registry import register_model
from ._registry import register_model, generate_default_cfgs

__all__ = ['GhostNetV2']



def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    
  
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x
    
class GhostModuleV2(nn.Module):

    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,mode=None):
        super(GhostModuleV2, self).__init__()
        self.mode=mode
        self.gate_fn=nn.Sigmoid()

        if self.mode in ['original']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        elif self.mode in ['attn']: 
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            ) 
            self.short_conv = nn.Sequential( 
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1,5), stride=1, padding=(0,2), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5,1), stride=1, padding=(2,0), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
            ) 
      
    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.oup,:,:]         
        elif self.mode in ['attn']:  
            res=self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2))  
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.oup,:,:]*F.interpolate(self.gate_fn(res),size=(out.shape[-2],out.shape[-1]),mode='nearest') 


class GhostBottleneckV2(nn.Module): 

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.,layer_id=None):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        if layer_id<=1:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True,mode='original')
        else:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True,mode='attn') 

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None
            
        self.ghost2 = GhostModuleV2(mid_chs, out_chs, relu=False,mode='original')
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )
    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

   
class GhostNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, in_chans=3,output_stride=32, drop_rate=0.2,global_pool='avg',block=GhostBottleneckV2):
        super(GhostNetV2, self).__init__()
        # setting of inverted residual blocks
        assert output_stride == 32, 'only output_stride==32 is valid, dilation not supported'
        
        self.cfgs = cfgs
        self.drop_rate = drop_rate

        # building first layer
        output_channel = make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(in_chans, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        #block = block
        layer_id=0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = make_divisible(c * width, 4)
                hidden_channel = make_divisible(exp_size * width, 4)
                if block==GhostBottleneckV2:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                  se_ratio=se_ratio,layer_id=layer_id))
                input_channel = output_channel
                layer_id+=1
            stages.append(nn.Sequential(*layers))

        output_channel = make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        
        self.blocks = nn.Sequential(*stages)        

        # building last several layers
        output_channel = 1280
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.drop_rate > 0.:
            x = F.drop_rate(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x

def _create_ghostnetv2(variant, width=1.0, pretrained=False, **kwargs):
    
    """
    Constructs a GhostNetV2 model
    """
    cfgs = [   
        # k, t, c, SE, s 
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    model_kwargs = dict(
        cfgs=cfgs,
        width=width,
        **kwargs,
    )

    return build_model_with_cfg(
        GhostNetV2,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True),
        **model_kwargs,
    )

    # return GhostNetV2(cfgs, num_classes=kwargs['num_classes'], 
    #                 width=kwargs['width'], 
    #                 drop_rate=kwargs['drop_rate'],
    #                 args=kwargs['args'])


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }

default_cfgs = generate_default_cfgs({
    'ghostnetv2_100.in1k': _cfg(
        url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/GhostNetV2/ck_ghostnetv2_10.pth.tar'),
    'ghostnetv2_130.in1k': _cfg(
        url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/GhostNetV2/ck_ghostnetv2_13.pth.tar'),    
    'ghostnetv2_160.in1k': _cfg(
        url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/GhostNetV2/ck_ghostnetv2_16.pth.tar'),         
})


@register_model
def ghostnetv2_100(pretrained=False, **kwargs) -> GhostNetV2:
    """ GhostNetV2-1.0x """
    model = _create_ghostnetv2('ghostnetv2_100', width=1.0, pretrained=pretrained, **kwargs)
    return model

@register_model
def ghostnetv2_130(pretrained=False, **kwargs) -> GhostNetV2:
    """ GhostNetV2-1.3x """
    model = _create_ghostnetv2('ghostnetv2_130', width=1.3, pretrained=pretrained, **kwargs)
    return model

@register_model
def ghostnetv2_160(pretrained=False, **kwargs) -> GhostNetV2:
    """ GhostNetV2-1.6x """
    model = _create_ghostnetv2('ghostnetv2_160', width=1.6, pretrained=pretrained, **kwargs)
    return model
