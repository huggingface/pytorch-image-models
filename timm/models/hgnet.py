""" PP-HGNet (V1 & V2)

Reference:
https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/models/ImageNet1k/PP-HGNetV2.md
The Paddle Implement of PP-HGNet (https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5.1/docs/en/models/PP-HGNet_en.md)
PP-HGNet: https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5.1/ppcls/arch/backbone/legendary_models/pp_hgnet.py
PP-HGNetv2: https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5.1/ppcls/arch/backbone/legendary_models/pp_hgnet_v2.py
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d, DropPath, create_conv2d
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._registry import register_model, generate_default_cfgs
from ._manipulate import checkpoint_seq

__all__ = ['HighPerfGpuNet']


class LearnableAffineBlock(nn.Module):
    def __init__(
            self,
            scale_value=1.0,
            bias_value=0.0
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        self.conv = create_conv2d(
            in_chs,
            out_chs,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_chs)
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
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(
            chs,
            chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class StemV1(nn.Module):
    # for PP-HGNet
    def __init__(self, stem_chs):
        super().__init__()
        self.stem = nn.Sequential(*[
            ConvBNAct(
                stem_chs[i],
                stem_chs[i + 1],
                kernel_size=3,
                stride=2 if i == 0 else 1) for i in range(
                len(stem_chs) - 1)
        ])
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        return x


class StemV2(nn.Module):
    # for PP-HGNetv2
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem2a = ConvBNAct(
            mid_chs,
            mid_chs // 2,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem2b = ConvBNAct(
            mid_chs // 2,
            mid_chs,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem3 = ConvBNAct(
            mid_chs * 2,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem4 = ConvBNAct(
            mid_chs,
            out_chs,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

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


class HighPerfGpuBlock(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            layer_num,
            kernel_size=3,
            residual=False,
            light_block=False,
            use_lab=False,
            agg='ese',
            drop_path=0.,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                    )
                )

        # feature aggregation
        total_chs = in_chs + layer_num * mid_chs
        if agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )

        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class HighPerfGpuStage(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            stride=2,
            light_block=False,
            kernel_size=3,
            use_lab=False,
            agg='ese',
            drop_path=0.,
    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=stride,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HighPerfGpuBlock(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)
        self.grad_checkpointing= False

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class ClassifierHead(nn.Module):
    def __init__(
            self,
            in_features: int,
            num_classes: int,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            hidden_size: Optional[int] = 2048,
            use_lab: bool = False
    ):
        super(ClassifierHead, self).__init__()
        self.num_features = in_features
        if pool_type is not None:
            if not pool_type:
                assert num_classes == 0, 'Classifier head must be removed if pooling is disabled'

        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        if hidden_size is not None:
            self.num_features = hidden_size
            last_conv = nn.Conv2d(
                in_features,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            act = nn.ReLU()
            if use_lab:
                lab = LearnableAffineBlock()
                self.last_conv = nn.Sequential(last_conv, act, lab)
            else:
                self.last_conv = nn.Sequential(last_conv, act)
        else:
            self.last_conv = nn.Identity()

        self.dropout = nn.Dropout(drop_rate)
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()  # don't flatten if pooling disabled
        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def reset(self, num_classes: int, pool_type: Optional[str] = None):
        if pool_type is not None:
            if not pool_type:
                assert num_classes == 0, 'Classifier head must be removed if pooling is disabled'
            self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
            self.flatten = nn.Flatten(1) if pool_type else nn.Identity()  # don't flatten if pooling disabled

        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.last_conv(x)
        x = self.dropout(x)
        x = self.flatten(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x


class HighPerfGpuNet(nn.Module):

    def __init__(
            self,
            cfg: Dict,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            head_hidden_size: Optional[int] = 2048,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            use_lab: bool = False,
            **kwargs,
    ):
        super(HighPerfGpuNet, self).__init__()
        stem_type = cfg["stem_type"]
        stem_chs = cfg["stem_chs"]
        stages_cfg = [cfg["stage1"], cfg["stage2"], cfg["stage3"], cfg["stage4"]]
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.use_lab = use_lab

        assert stem_type in ['v1', 'v2']
        if stem_type == 'v2':
            self.stem = StemV2(
                in_chs=in_chans,
                mid_chs=stem_chs[0],
                out_chs=stem_chs[1],
                use_lab=use_lab)
        else:
            self.stem = StemV1([in_chans] + stem_chs)

        current_stride = 4

        stages = []
        self.feature_info = []
        block_depths = [c[3] for c in stages_cfg]
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(block_depths)).split(block_depths)]
        for i, stage_config in enumerate(stages_cfg):
            in_chs, mid_chs, out_chs, block_num, downsample, light_block, kernel_size, layer_num = stage_config
            stages += [HighPerfGpuStage(
                in_chs=in_chs,
                mid_chs=mid_chs,
                out_chs=out_chs,
                block_num=block_num,
                layer_num=layer_num,
                downsample=downsample,
                light_block=light_block,
                kernel_size=kernel_size,
                use_lab=use_lab,
                agg='ese' if stem_type == 'v1' else 'se',
                drop_path=dpr[i],
            )]
            self.num_features = out_chs
            if downsample:
                current_stride *= 2
            self.feature_info += [dict(num_chs=self.num_features, reduction=current_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        self.head = ClassifierHead(
            self.num_features,
            num_classes=num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            hidden_size=head_hidden_size,
            use_lab=use_lab
        )
        self.head_hidden_size = self.head.num_features

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
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages), indices)

        # forward pass
        x = self.stem(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]

        for feat_idx, stage in enumerate(stages):
            x = stage(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        self.stages = self.stages[:max_index + 1]  # truncate blocks w/ stem as idx 0
        if prune_head:
            self.reset_classifier(0, 'avg')
        return take_indices

    def forward_features(self, x):
        x = self.stem(x)
        return self.stages(x)

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


model_cfgs = dict(
    # PP-HGNet
    hgnet_tiny={
        "stem_type": 'v1',
        "stem_chs": [48, 48, 96],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [96, 96, 224, 1, False, False, 3, 5],
        "stage2": [224, 128, 448, 1, True, False, 3, 5],
        "stage3": [448, 160, 512, 2, True, False, 3, 5],
        "stage4": [512, 192, 768, 1, True, False, 3, 5],
    },
    hgnet_small={
        "stem_type": 'v1',
        "stem_chs": [64, 64, 128],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [128, 128, 256, 1, False, False, 3, 6],
        "stage2": [256, 160, 512, 1, True, False, 3, 6],
        "stage3": [512, 192, 768, 2, True, False, 3, 6],
        "stage4": [768, 224, 1024, 1, True, False, 3, 6],
    },
    hgnet_base={
        "stem_type": 'v1',
        "stem_chs": [96, 96, 160],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [160, 192, 320, 1, False, False, 3, 7],
        "stage2": [320, 224, 640, 2, True, False, 3, 7],
        "stage3": [640, 256, 960, 3, True, False, 3, 7],
        "stage4": [960, 288, 1280, 2, True, False, 3, 7],
    },
    # PP-HGNetv2
    hgnetv2_b0={
        "stem_type": 'v2',
        "stem_chs": [16, 16],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [16, 16, 64, 1, False, False, 3, 3],
        "stage2": [64, 32, 256, 1, True, False, 3, 3],
        "stage3": [256, 64, 512, 2, True, True, 5, 3],
        "stage4": [512, 128, 1024, 1, True, True, 5, 3],
    },
    hgnetv2_b1={
        "stem_type": 'v2',
        "stem_chs": [24, 32],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 64, 1, False, False, 3, 3],
        "stage2": [64, 48, 256, 1, True, False, 3, 3],
        "stage3": [256, 96, 512, 2, True, True, 5, 3],
        "stage4": [512, 192, 1024, 1, True, True, 5, 3],
    },
    hgnetv2_b2={
        "stem_type": 'v2',
        "stem_chs": [24, 32],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 96, 1, False, False, 3, 4],
        "stage2": [96, 64, 384, 1, True, False, 3, 4],
        "stage3": [384, 128, 768, 3, True, True, 5, 4],
        "stage4": [768, 256, 1536, 1, True, True, 5, 4],
    },
    hgnetv2_b3={
        "stem_type": 'v2',
        "stem_chs": [24, 32],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 128, 1, False, False, 3, 5],
        "stage2": [128, 64, 512, 1, True, False, 3, 5],
        "stage3": [512, 128, 1024, 3, True, True, 5, 5],
        "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
    },
    hgnetv2_b4={
        "stem_type": 'v2',
        "stem_chs": [32, 48],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [48, 48, 128, 1, False, False, 3, 6],
        "stage2": [128, 96, 512, 1, True, False, 3, 6],
        "stage3": [512, 192, 1024, 3, True, True, 5, 6],
        "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
    },
    hgnetv2_b5={
        "stem_type": 'v2',
        "stem_chs": [32, 64],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [64, 64, 128, 1, False, False, 3, 6],
        "stage2": [128, 128, 512, 2, True, False, 3, 6],
        "stage3": [512, 256, 1024, 5, True, True, 5, 6],
        "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
    },
    hgnetv2_b6={
        "stem_type": 'v2',
        "stem_chs": [48, 96],
        # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
        "stage1": [96, 96, 192, 2, False, False, 3, 6],
        "stage2": [192, 192, 512, 3, True, False, 3, 6],
        "stage3": [512, 384, 1024, 6, True, True, 5, 6],
        "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
    },
)


def _create_hgnet(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', (0, 1, 2, 3))
    return build_model_with_cfg(
        HighPerfGpuNet,
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
        'crop_pct': 0.965, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head.fc', 'first_conv': 'stem.stem1.conv',
        'test_crop_pct': 1.0, 'test_input_size': (3, 288, 288),
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'hgnet_tiny.paddle_in1k': _cfg(
        first_conv='stem.stem.0.conv',
        hf_hub_id='timm/'),
    'hgnet_tiny.ssld_in1k': _cfg(
        first_conv='stem.stem.0.conv',
        hf_hub_id='timm/'),
    'hgnet_small.paddle_in1k': _cfg(
        first_conv='stem.stem.0.conv',
        hf_hub_id='timm/'),
    'hgnet_small.ssld_in1k': _cfg(
        first_conv='stem.stem.0.conv',
        hf_hub_id='timm/'),
    'hgnet_base.ssld_in1k': _cfg(
        first_conv='stem.stem.0.conv',
        hf_hub_id='timm/'),
    'hgnetv2_b0.ssld_stage2_ft_in1k': _cfg(
        hf_hub_id='timm/'),
    'hgnetv2_b0.ssld_stage1_in22k_in1k': _cfg(
        hf_hub_id='timm/'),
    'hgnetv2_b1.ssld_stage2_ft_in1k': _cfg(
        hf_hub_id='timm/'),
    'hgnetv2_b1.ssld_stage1_in22k_in1k': _cfg(
        hf_hub_id='timm/'),
    'hgnetv2_b2.ssld_stage2_ft_in1k': _cfg(
        hf_hub_id='timm/'),
    'hgnetv2_b2.ssld_stage1_in22k_in1k': _cfg(
        hf_hub_id='timm/'),
    'hgnetv2_b3.ssld_stage2_ft_in1k': _cfg(
        hf_hub_id='timm/'),
    'hgnetv2_b3.ssld_stage1_in22k_in1k': _cfg(
        hf_hub_id='timm/'),
    'hgnetv2_b4.ssld_stage2_ft_in1k': _cfg(
        hf_hub_id='timm/'),
    'hgnetv2_b4.ssld_stage1_in22k_in1k': _cfg(
        hf_hub_id='timm/'),
    'hgnetv2_b5.ssld_stage2_ft_in1k': _cfg(
        hf_hub_id='timm/'),
    'hgnetv2_b5.ssld_stage1_in22k_in1k': _cfg(
        hf_hub_id='timm/'),
    'hgnetv2_b6.ssld_stage2_ft_in1k': _cfg(
        hf_hub_id='timm/'),
    'hgnetv2_b6.ssld_stage1_in22k_in1k': _cfg(
        hf_hub_id='timm/'),
})


@register_model
def hgnet_tiny(pretrained=False, **kwargs) -> HighPerfGpuNet:
    return _create_hgnet('hgnet_tiny', pretrained=pretrained, **kwargs)


@register_model
def hgnet_small(pretrained=False, **kwargs) -> HighPerfGpuNet:
    return _create_hgnet('hgnet_small', pretrained=pretrained, **kwargs)


@register_model
def hgnet_base(pretrained=False, **kwargs) -> HighPerfGpuNet:
    return _create_hgnet('hgnet_base', pretrained=pretrained, **kwargs)


@register_model
def hgnetv2_b0(pretrained=False, **kwargs) -> HighPerfGpuNet:
    return _create_hgnet('hgnetv2_b0', pretrained=pretrained, use_lab=True, **kwargs)


@register_model
def hgnetv2_b1(pretrained=False, **kwargs) -> HighPerfGpuNet:
    return _create_hgnet('hgnetv2_b1', pretrained=pretrained, use_lab=True, **kwargs)


@register_model
def hgnetv2_b2(pretrained=False, **kwargs) -> HighPerfGpuNet:
    return _create_hgnet('hgnetv2_b2', pretrained=pretrained, use_lab=True, **kwargs)


@register_model
def hgnetv2_b3(pretrained=False, **kwargs) -> HighPerfGpuNet:
    return _create_hgnet('hgnetv2_b3', pretrained=pretrained, use_lab=True, **kwargs)


@register_model
def hgnetv2_b4(pretrained=False, **kwargs) -> HighPerfGpuNet:
    return _create_hgnet('hgnetv2_b4', pretrained=pretrained, **kwargs)


@register_model
def hgnetv2_b5(pretrained=False, **kwargs) -> HighPerfGpuNet:
    return _create_hgnet('hgnetv2_b5', pretrained=pretrained, **kwargs)


@register_model
def hgnetv2_b6(pretrained=False, **kwargs) -> HighPerfGpuNet:
    return _create_hgnet('hgnetv2_b6', pretrained=pretrained, **kwargs)
