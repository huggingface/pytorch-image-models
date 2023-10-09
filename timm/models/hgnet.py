""" PP-HGNet (V1 & V2)

Reference:
The Paddle Implement of PP-HGNet (https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5.1/docs/en/models/PP-HGNet_en.md)
PP-HGNet: https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5.1/ppcls/arch/backbone/legendary_models/pp_hgnet.py
PP-HGNetv2: https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5.1/ppcls/arch/backbone/legendary_models/pp_hgnet_v2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d
from ._builder import build_model_with_cfg
from ._registry import register_model, generate_default_cfgs

__all__ = ['PPHGNet']


class LearnableAffineBlock(nn.Module):
    def __init__(self,
                 scale_value=1.0,
                 bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]))
        self.bias = nn.Parameter(torch.tensor([bias_value]))

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        use_act=True,
        use_lab=False
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups=1,
        use_lab=False
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab)
        self.conv2 = ConvBNAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            use_act=True,
            use_lab=use_lab)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ESEModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class StemV1(nn.Module):
    # for PP-HGNet
    def __init__(self, stem_channels):
        super().__init__()
        self.stem = nn.Sequential(*[
            ConvBNAct(
                in_channels=stem_channels[i],
                out_channels=stem_channels[i + 1],
                kernel_size=3,
                stride=2 if i == 0 else 1) for i in range(
                    len(stem_channels) - 1)
        ])
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        return x


class StemV2(nn.Module):
    # for PP-HGNetv2
    def __init__(self, in_channels, mid_channels, out_channels, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab)
        self.stem2a = ConvBNAct(
            in_channels=mid_channels,
            out_channels=mid_channels // 2,
            kernel_size=2,
            stride=1,
            use_lab=use_lab)
        self.stem2b = ConvBNAct(
            in_channels=mid_channels // 2,
            out_channels=mid_channels,
            kernel_size=2,
            stride=1,
            use_lab=use_lab)
        self.stem3 = ConvBNAct(
            in_channels=mid_channels * 2,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab)
        self.stem4 = ConvBNAct(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        layer_num,
        kernel_size=3,
        residual=False,
        light_block=False,
        use_lab=False,
        agg='ese',
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_channels=in_channels if i == 0 else mid_channels,
                        out_channels=mid_channels,
                        kernel_size=kernel_size,
                        use_lab=use_lab,))
            else:
                self.layers.append(
                    ConvBNAct(
                        in_channels=in_channels if i == 0 else mid_channels,
                        out_channels=mid_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,))

        # feature aggregation
        total_channels = in_channels + layer_num * mid_channels
        if agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(
                in_channels=total_channels,
                out_channels=out_channels // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab)
            aggregation_excitation_conv = ConvBNAct(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                use_lab=use_lab)
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv)
        else:
            aggregation_conv = ConvBNAct(
                in_channels=total_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                use_lab=use_lab)
            att = ESEModule(out_channels)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att)

    def forward(self, x):
        identity = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x += identity
        return x


class HGStage(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        block_num,
        layer_num,
        downsample=True,
        stride=2,
        light_block=False,
        kernel_size=3,
        use_lab=False,
        agg='ese',
    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=stride,
                groups=in_channels,
                use_act=False,
                use_lab=use_lab,)
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HGBlock(
                    in_channels if i == 0 else out_channels,
                    mid_channels,
                    out_channels,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg))
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class ClassifierHead(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        pool_type='avg',
        drop_rate=0.,
        use_last_conv=True,
        class_expand=2048,
        use_lab=False
    ):
        super(ClassifierHead, self).__init__()
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=False, input_fmt='NCHW')
        if use_last_conv:
            last_conv = nn.Conv2d(
                in_channels=num_features,
                out_channels=class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False)
            act = nn.ReLU()
            if use_lab:
                lab = LearnableAffineBlock()
                self.last_conv = nn.Sequential(last_conv, act, lab)
            else:
                self.last_conv = nn.Sequential(last_conv, act)
        else:
            self.last_conv = nn.Indentity()

        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        else:
            self.dropout = nn.Identity()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(class_expand if use_last_conv else num_features, num_classes)

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.last_conv(x)
        x = self.dropout(x)
        x = self.flatten(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x


class PPHGNet(nn.Module):

    def __init__(
        self,
        cfg,
        in_chans=3,
        num_classes=1000,
        global_pool='avg',
        use_last_conv=True,
        class_expand=2048,
        drop_rate=0.,
        use_lab=False,
        **kwargs,
    ):
        super(PPHGNet, self).__init__()
        stem_type = cfg["stem_type"]
        stem_channels = cfg["stem_channels"]
        stages_cfg = [cfg["stage1"], cfg["stage2"], cfg["stage3"], cfg["stage4"]]
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.use_last_conv = use_last_conv
        self.class_expand = class_expand
        self.use_lab = use_lab

        assert stem_type in ['v1', 'v2']
        if stem_type == 'v2':
            self.stem = StemV2(
                in_channels=in_chans,
                mid_channels=stem_channels[0],
                out_channels=stem_channels[1],
                use_lab=use_lab)
        else:
            self.stem = StemV1([in_chans] + stem_channels)

        current_stride = 4

        stages = []
        self.feature_info = []
        for i, stage_config in enumerate(stages_cfg):
            in_channels, mid_channels, out_channels, block_num, is_downsample, light_block, kernel_size, layer_num = stage_config
            stages += [HGStage(
                in_channels=in_channels,
                mid_channels=mid_channels,
                out_channels=out_channels,
                block_num=block_num,
                layer_num=layer_num,
                downsample=is_downsample,
                light_block=light_block,
                kernel_size=kernel_size,
                use_lab=use_lab,
                agg='ese' if stem_type == 'v1' else 'se'
            )]
            self.num_features = out_channels
            if is_downsample:
                current_stride *= 2
            self.feature_info += [dict(num_chs=self.num_features, reduction=current_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        self.head = ClassifierHead(
            num_features=self.num_features,
            num_classes=num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            use_last_conv=use_last_conv,
            class_expand=class_expand,
            use_lab=use_lab
        )

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else r'^stages\.(\d+).blocks\.(\d+)',
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = ClassifierHead(
                num_features=self.num_features,
                num_classes=num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
                use_last_conv=self.use_last_conv,
                class_expand=self.class_expand,
                use_lab=self.use_lab)
        else:
            if self.global_pool == 'avg':
                self.head = SelectAdaptivePool2d(pool_type=self.global_pool, flatten=True)
            else:
                self.head = nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        return self.stages(x)

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


model_cfgs = dict(
    # PP-HGNet
    hgnet_tiny={
        "stem_type": 'v1',
        "stem_channels": [48, 48, 96],
        # in_channels, mid_channels, out_channels, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [96, 96, 224, 1, False, False, 3, 5],
        "stage2": [224, 128, 448, 1, True, False, 3, 5],
        "stage3": [448, 160, 512, 2, True, False, 3, 5],
        "stage4": [512, 192, 768, 1, True, False, 3, 5],
    },
    hgnet_small={
        "stem_type": 'v1',
        "stem_channels": [64, 64, 128],
        # in_channels, mid_channels, out_channels, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [128, 128, 256, 1, False, False, 3, 6],
        "stage2": [256, 160, 512, 1, True, False, 3, 6],
        "stage3": [512, 192, 768, 2, True, False, 3, 6],
        "stage4": [768, 224, 1024, 1, True, False, 3, 6],
    },
    hgnet_base={
        "stem_type": 'v1',
        "stem_channels": [96, 96, 160],
        # in_channels, mid_channels, out_channels, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [160, 192, 320, 1, False, False, 3, 7],
        "stage2": [320, 224, 640, 2, True, False, 3, 7],
        "stage3": [640, 256, 960, 3, True, False, 3, 7],
        "stage4": [960, 288, 1280, 2, True, False, 3, 7],
    },
    # PP-HGNetv2
    hgnetv2_b0={
        "stem_type": 'v2',
        "stem_channels": [16, 16],
        # in_channels, mid_channels, out_channels, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [16, 16, 64, 1, False, False, 3, 3],
        "stage2": [64, 32, 256, 1, True, False, 3, 3],
        "stage3": [256, 64, 512, 2, True, True, 5, 3],
        "stage4": [512, 128, 1024, 1, True, True, 5, 3],
    },
    hgnetv2_b1={
        "stem_type": 'v2',
        "stem_channels": [24, 32],
        # in_channels, mid_channels, out_channels, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 64, 1, False, False, 3, 3],
        "stage2": [64, 48, 256, 1, True, False, 3, 3],
        "stage3": [256, 96, 512, 2, True, True, 5, 3],
        "stage4": [512, 192, 1024, 1, True, True, 5, 3],
    },
    hgnetv2_b2={
        "stem_type": 'v2',
        "stem_channels": [24, 32],
        # in_channels, mid_channels, out_channels, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 96, 1, False, False, 3, 4],
        "stage2": [96, 64, 384, 1, True, False, 3, 4],
        "stage3": [384, 128, 768, 3, True, True, 5, 4],
        "stage4": [768, 256, 1536, 1, True, True, 5, 4],
    },
    hgnetv2_b3={
        "stem_type": 'v2',
        "stem_channels": [24, 32],
        # in_channels, mid_channels, out_channels, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 128, 1, False, False, 3, 5],
        "stage2": [128, 64, 512, 1, True, False, 3, 5],
        "stage3": [512, 128, 1024, 3, True, True, 5, 5],
        "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
    },
    hgnetv2_b4={
        "stem_type": 'v2',
        "stem_channels": [32, 48],
        # in_channels, mid_channels, out_channels, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [48, 48, 128, 1, False, False, 3, 6],
        "stage2": [128, 96, 512, 1, True, False, 3, 6],
        "stage3": [512, 192, 1024, 3, True, True, 5, 6],
        "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
    },
    hgnetv2_b5={
        "stem_type": 'v2',
        "stem_channels": [32, 64],
        # in_channels, mid_channels, out_channels, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [64, 64, 128, 1, False, False, 3, 6],
        "stage2": [128, 128, 512, 2, True, False, 3, 6],
        "stage3": [512, 256, 1024, 5, True, True, 5, 6],
        "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
    },
    hgnetv2_b6={
        "stem_type": 'v2',
        "stem_channels": [48, 96],
        # in_channels, mid_channels, out_channels, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [96, 96, 192, 2, False, False, 3, 6],
        "stage2": [192, 192, 512, 3, True, False, 3, 6],
        "stage3": [512, 384, 1024, 6, True, True, 5, 6],
        "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
    },
)


def _create_hgnet(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', (0, 1, 2, 3))
    return build_model_with_cfg(
        PPHGNet,
        variant,
        pretrained,
        model_cfg=model_cfgs[variant],
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.95, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head.fc', 'first_conv': 'stem.0.conv',
        'test_crop_pct': 1.0, 'test_input_size': (3, 288, 288),
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'hgnet_tiny.paddle_in1k': _cfg(
        first_conv='stem.0.conv',
        hf_hub_id='timm/'),
    'hgnet_small.paddle_in1k': _cfg(
        first_conv='stem.0.conv',
        hf_hub_id='timm/'),
    'hgnet_base': _cfg(
        first_conv='stem.0.conv'),
    'hgnetv2_b0.ssld_in1k': _cfg(
        first_conv='stem.stem1.conv',
        hf_hub_id='timm/'),
    'hgnetv2_b0.ssld_stage1': _cfg(
        first_conv='stem.stem1.conv',
        hf_hub_id='timm/'),
    'hgnetv2_b1.ssld_in1k': _cfg(
        first_conv='stem.stem1.conv',
        hf_hub_id='timm/'),
    'hgnetv2_b1.ssld_stage1': _cfg(
        first_conv='stem.stem1.conv',
        hf_hub_id='timm/'),
    'hgnetv2_b2.ssld_in1k': _cfg(
        first_conv='stem.stem1.conv',
        hf_hub_id='timm/'),
    'hgnetv2_b2.ssld_stage1': _cfg(
        first_conv='stem.stem1.conv',
        hf_hub_id='timm/'),
    'hgnetv2_b3.ssld_in1k': _cfg(
        first_conv='stem.stem1.conv',
        hf_hub_id='timm/'),
    'hgnetv2_b3.ssld_stage1': _cfg(
        first_conv='stem.stem1.conv',
        hf_hub_id='timm/'),
    'hgnetv2_b4.ssld_in1k': _cfg(
        first_conv='stem.stem1.conv',
        hf_hub_id='timm/'),
    'hgnetv2_b4.ssld_stage1': _cfg(
        first_conv='stem.stem1.conv',
        hf_hub_id='timm/'),
    'hgnetv2_b5.ssld_in1k': _cfg(
        first_conv='stem.stem1.conv',
        hf_hub_id='timm/'),
    'hgnetv2_b5.ssld_stage1': _cfg(
        first_conv='stem.stem1.conv',
        hf_hub_id='timm/'),
    'hgnetv2_b6.ssld_in1k': _cfg(
        first_conv='stem.stem1.conv',
        hf_hub_id='timm/'),
    'hgnetv2_b6.ssld_stage1': _cfg(
        first_conv='stem.stem1.conv',
        hf_hub_id='timm/'),
})


@register_model
def hgnet_tiny(pretrained=False, **kwargs) -> PPHGNet:
    return _create_hgnet('hgnet_tiny', pretrained=pretrained, **kwargs)


@register_model
def hgnet_small(pretrained=False, **kwargs) -> PPHGNet:
    return _create_hgnet('hgnet_small', pretrained=pretrained, **kwargs)


@register_model
def hgnet_base(pretrained=False, **kwargs) -> PPHGNet:
    return _create_hgnet('hgnet_base', pretrained=pretrained, **kwargs)


@register_model
def hgnetv2_b0(pretrained=False, **kwargs) -> PPHGNet:
    return _create_hgnet('hgnetv2_b0', pretrained=pretrained, use_lab=True, **kwargs)


@register_model
def hgnetv2_b1(pretrained=False, **kwargs) -> PPHGNet:
    return _create_hgnet('hgnetv2_b1', pretrained=pretrained, use_lab=True, **kwargs)


@register_model
def hgnetv2_b2(pretrained=False, **kwargs) -> PPHGNet:
    return _create_hgnet('hgnetv2_b2', pretrained=pretrained, use_lab=True, **kwargs)


@register_model
def hgnetv2_b3(pretrained=False, **kwargs) -> PPHGNet:
    return _create_hgnet('hgnetv2_b3', pretrained=pretrained, use_lab=True, **kwargs)


@register_model
def hgnetv2_b4(pretrained=False, **kwargs) -> PPHGNet:
    return _create_hgnet('hgnetv2_b4', pretrained=pretrained, **kwargs)


@register_model
def hgnetv2_b5(pretrained=False, **kwargs) -> PPHGNet:
    return _create_hgnet('hgnetv2_b5', pretrained=pretrained, **kwargs)


@register_model
def hgnetv2_b6(pretrained=False, **kwargs) -> PPHGNet:
    return _create_hgnet('hgnetv2_b6', pretrained=pretrained, **kwargs)
