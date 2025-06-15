""" HRNet

Copied from https://github.com/HRNet/HRNet-Image-Classification

Original header:
  Copyright (c) Microsoft
  Licensed under the MIT License.
  Written by Bin Xiao (Bin.Xiao@microsoft.com)
  Modified by Ke Sun (sunk@mail.ustc.edu.cn)
"""
import logging
from typing import List

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import create_classifier
from ._builder import build_model_with_cfg, pretrained_cfg_for_features
from ._features import FeatureInfo
from ._registry import register_model, generate_default_cfgs
from .resnet import BasicBlock, Bottleneck  # leveraging ResNet block_types w/ additional features like SE

__all__ = ['HighResolutionNet', 'HighResolutionNetFeatures']  # model_registry will add each entrypoint fn to this

_BN_MOMENTUM = 0.1
_logger = logging.getLogger(__name__)


cfg_cls = dict(
    hrnet_w18_small=dict(
        stem_width=64,
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block_type='BOTTLENECK',
            num_blocks=(1,),
            num_channels=(32,),
            fuse_method='SUM',
        ),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block_type='BASIC',
            num_blocks=(2, 2),
            num_channels=(16, 32),
            fuse_method='SUM'
        ),
        stage3=dict(
            num_modules=1,
            num_branches=3,
            block_type='BASIC',
            num_blocks=(2, 2, 2),
            num_channels=(16, 32, 64),
            fuse_method='SUM'
        ),
        stage4=dict(
            num_modules=1,
            num_branches=4,
            block_type='BASIC',
            num_blocks=(2, 2, 2, 2),
            num_channels=(16, 32, 64, 128),
            fuse_method='SUM',
        ),
    ),

    hrnet_w18_small_v2=dict(
        stem_width=64,
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block_type='BOTTLENECK',
            num_blocks=(2,),
            num_channels=(64,),
            fuse_method='SUM',
        ),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block_type='BASIC',
            num_blocks=(2, 2),
            num_channels=(18, 36),
            fuse_method='SUM'
        ),
        stage3=dict(
            num_modules=3,
            num_branches=3,
            block_type='BASIC',
            num_blocks=(2, 2, 2),
            num_channels=(18, 36, 72),
            fuse_method='SUM'
        ),
        stage4=dict(
            num_modules=2,
            num_branches=4,
            block_type='BASIC',
            num_blocks=(2, 2, 2, 2),
            num_channels=(18, 36, 72, 144),
            fuse_method='SUM',
        ),
    ),

    hrnet_w18=dict(
        stem_width=64,
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block_type='BOTTLENECK',
            num_blocks=(4,),
            num_channels=(64,),
            fuse_method='SUM',
        ),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block_type='BASIC',
            num_blocks=(4, 4),
            num_channels=(18, 36),
            fuse_method='SUM'
        ),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block_type='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(18, 36, 72),
            fuse_method='SUM'
        ),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block_type='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(18, 36, 72, 144),
            fuse_method='SUM',
        ),
    ),

    hrnet_w30=dict(
        stem_width=64,
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block_type='BOTTLENECK',
            num_blocks=(4,),
            num_channels=(64,),
            fuse_method='SUM',
        ),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block_type='BASIC',
            num_blocks=(4, 4),
            num_channels=(30, 60),
            fuse_method='SUM'
        ),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block_type='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(30, 60, 120),
            fuse_method='SUM'
        ),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block_type='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(30, 60, 120, 240),
            fuse_method='SUM',
        ),
    ),

    hrnet_w32=dict(
        stem_width=64,
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block_type='BOTTLENECK',
            num_blocks=(4,),
            num_channels=(64,),
            fuse_method='SUM',
        ),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block_type='BASIC',
            num_blocks=(4, 4),
            num_channels=(32, 64),
            fuse_method='SUM'
        ),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block_type='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(32, 64, 128),
            fuse_method='SUM'
        ),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block_type='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(32, 64, 128, 256),
            fuse_method='SUM',
        ),
    ),

    hrnet_w40=dict(
        stem_width=64,
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block_type='BOTTLENECK',
            num_blocks=(4,),
            num_channels=(64,),
            fuse_method='SUM',
        ),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block_type='BASIC',
            num_blocks=(4, 4),
            num_channels=(40, 80),
            fuse_method='SUM'
        ),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block_type='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(40, 80, 160),
            fuse_method='SUM'
        ),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block_type='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(40, 80, 160, 320),
            fuse_method='SUM',
        ),
    ),

    hrnet_w44=dict(
        stem_width=64,
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block_type='BOTTLENECK',
            num_blocks=(4,),
            num_channels=(64,),
            fuse_method='SUM',
        ),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block_type='BASIC',
            num_blocks=(4, 4),
            num_channels=(44, 88),
            fuse_method='SUM'
        ),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block_type='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(44, 88, 176),
            fuse_method='SUM'
        ),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block_type='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(44, 88, 176, 352),
            fuse_method='SUM',
        ),
    ),

    hrnet_w48=dict(
        stem_width=64,
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block_type='BOTTLENECK',
            num_blocks=(4,),
            num_channels=(64,),
            fuse_method='SUM',
        ),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block_type='BASIC',
            num_blocks=(4, 4),
            num_channels=(48, 96),
            fuse_method='SUM'
        ),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block_type='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(48, 96, 192),
            fuse_method='SUM'
        ),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block_type='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(48, 96, 192, 384),
            fuse_method='SUM',
        ),
    ),

    hrnet_w64=dict(
        stem_width=64,
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block_type='BOTTLENECK',
            num_blocks=(4,),
            num_channels=(64,),
            fuse_method='SUM',
        ),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block_type='BASIC',
            num_blocks=(4, 4),
            num_channels=(64, 128),
            fuse_method='SUM'
        ),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block_type='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(64, 128, 256),
            fuse_method='SUM'
        ),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block_type='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(64, 128, 256, 512),
            fuse_method='SUM',
        ),
    )
)


class HighResolutionModule(nn.Module):
    def __init__(
            self,
            num_branches,
            block_types,
            num_blocks,
            num_in_chs,
            num_channels,
            fuse_method,
            multi_scale_output=True,
    ):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches,
            block_types,
            num_blocks,
            num_in_chs,
            num_channels,
        )

        self.num_in_chs = num_in_chs
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches,
            block_types,
            num_blocks,
            num_channels,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.fuse_act = nn.ReLU(False)

    def _check_branches(self, num_branches, block_types, num_blocks, num_in_chs, num_channels):
        error_msg = ''
        if num_branches != len(num_blocks):
            error_msg = 'num_branches({}) <> num_blocks({})'.format(num_branches, len(num_blocks))
        elif num_branches != len(num_channels):
            error_msg = 'num_branches({}) <> num_channels({})'.format(num_branches, len(num_channels))
        elif num_branches != len(num_in_chs):
            error_msg = 'num_branches({}) <> num_in_chs({})'.format(num_branches, len(num_in_chs))
        if error_msg:
            _logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block_type, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_in_chs[branch_index] != num_channels[branch_index] * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_in_chs[branch_index], num_channels[branch_index] * block_type.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block_type.expansion, momentum=_BN_MOMENTUM),
            )

        layers = [block_type(self.num_in_chs[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_in_chs[branch_index] = num_channels[branch_index] * block_type.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block_type(self.num_in_chs[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block_type, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block_type, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return nn.Identity()

        num_branches = self.num_branches
        num_in_chs = self.num_in_chs
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_in_chs[j], num_in_chs[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_in_chs[i], momentum=_BN_MOMENTUM),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_out_chs_conv3x3 = num_in_chs[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_in_chs[j], num_out_chs_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_out_chs_conv3x3, momentum=_BN_MOMENTUM)
                            ))
                        else:
                            num_out_chs_conv3x3 = num_in_chs[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_in_chs[j], num_out_chs_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_out_chs_conv3x3, momentum=_BN_MOMENTUM),
                                nn.ReLU(False)
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_in_chs(self):
        return self.num_in_chs

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i, branch in enumerate(self.branches):
            x[i] = branch(x[i])

        x_fuse = []
        for i, fuse_outer in enumerate(self.fuse_layers):
            y = None
            for j, f in enumerate(fuse_outer):
                if y is None:
                    y = f(x[j])
                else:
                    y = y + f(x[j])
            x_fuse.append(self.fuse_act(y))
        return x_fuse


class SequentialList(nn.Sequential):

    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (List[torch.Tensor]) -> (List[torch.Tensor])
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (torch.Tensor) -> (List[torch.Tensor])
        pass

    def forward(self, x) -> List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x


@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor: # `input` has a same name in Sequential forward
        pass


block_types_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(
            self,
            cfg,
            in_chans=3,
            num_classes=1000,
            output_stride=32,
            global_pool='avg',
            drop_rate=0.0,
            head='classification',
            **kwargs,
    ):
        super(HighResolutionNet, self).__init__()
        self.num_classes = num_classes
        assert output_stride == 32  # FIXME support dilation

        cfg.update(**kwargs)
        stem_width = cfg['stem_width']
        self.conv1 = nn.Conv2d(in_chans, stem_width, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_width, momentum=_BN_MOMENTUM)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(stem_width, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.act2 = nn.ReLU(inplace=True)

        self.stage1_cfg = cfg['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = block_types_dict[self.stage1_cfg['block_type']]
        num_blocks = self.stage1_cfg['num_blocks'][0]
        self.layer1 = self._make_layer(block_type, 64, num_channels, num_blocks)
        stage1_out_channel = block_type.expansion * num_channels

        self.stage2_cfg = cfg['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = block_types_dict[self.stage2_cfg['block_type']]
        num_channels = [num_channels[i] * block_type.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = block_types_dict[self.stage3_cfg['block_type']]
        num_channels = [num_channels[i] * block_type.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = block_types_dict[self.stage4_cfg['block_type']]
        num_channels = [num_channels[i] * block_type.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        self.head = head
        self.head_channels = None  # set if _make_head called
        head_conv_bias = cfg.pop('head_conv_bias', True)
        if head == 'classification':
            # Classification Head
            self.num_features = self.head_hidden_size = 2048
            self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(
                pre_stage_channels,
                conv_bias=head_conv_bias,
            )
            self.global_pool, self.head_drop, self.classifier = create_classifier(
                self.num_features,
                self.num_classes,
                pool_type=global_pool,
                drop_rate=drop_rate,
            )
        else:
            if head == 'incre':
                self.num_features = self.head_hidden_size = 2048
                self.incre_modules, _, _ = self._make_head(pre_stage_channels, incre_only=True)
            else:
                self.num_features = self.head_hidden_size = 256
                self.incre_modules = None
            self.global_pool = nn.Identity()
            self.head_drop = nn.Identity()
            self.classifier = nn.Identity()

        curr_stride = 2
        # module names aren't actually valid here, hook or FeatureNet based extraction would not work
        self.feature_info = [dict(num_chs=64, reduction=curr_stride, module='stem')]
        for i, c in enumerate(self.head_channels if self.head_channels else num_channels):
            curr_stride *= 2
            c = c * 4 if self.head_channels else c  # head block_type expansion factor of 4
            self.feature_info += [dict(num_chs=c, reduction=curr_stride, module=f'stage{i + 1}')]

        self.init_weights()

    def _make_head(self, pre_stage_channels, incre_only=False, conv_bias=True):
        head_block_type = Bottleneck
        self.head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_modules.append(self._make_layer(head_block_type, channels, self.head_channels[i], 1, stride=1))
        incre_modules = nn.ModuleList(incre_modules)
        if incre_only:
            return incre_modules, None, None

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = self.head_channels[i] * head_block_type.expansion
            out_channels = self.head_channels[i + 1] * head_block_type.expansion
            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3, stride=2, padding=1, bias=conv_bias),
                nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.head_channels[3] * head_block_type.expansion, out_channels=self.num_features,
                kernel_size=1, stride=1, padding=0, bias=conv_bias),
            nn.BatchNorm2d(self.num_features, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=_BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    _in_chs = num_channels_pre_layer[-1]
                    _out_chs = num_channels_cur_layer[i] if j == i - num_branches_pre else _in_chs
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(_in_chs, _out_chs, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(_out_chs, momentum=_BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block_type, inplanes, planes, block_types, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block_type.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_type.expansion, momentum=_BN_MOMENTUM),
            )

        layers = [block_type(inplanes, planes, stride, downsample)]
        inplanes = planes * block_type.expansion
        for i in range(1, block_types):
            layers.append(block_type(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_in_chs, multi_scale_output=True):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block_type = block_types_dict[layer_config['block_type']]
        fuse_method = layer_config['fuse_method']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            reset_multi_scale_output = multi_scale_output or i < num_modules - 1
            modules.append(HighResolutionModule(
                num_branches, block_type, num_blocks, num_in_chs, num_channels, fuse_method, reset_multi_scale_output)
            )
            num_in_chs = modules[-1].get_num_in_chs()

        return SequentialList(*modules), num_in_chs

    @torch.jit.ignore
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^conv[12]|bn[12]',
            block_types=r'^(?:layer|stage|transition)(\d+)' if coarse else [
                (r'^layer(\d+)\.(\d+)', None),
                (r'^stage(\d+)\.(\d+)', None),
                (r'^transition(\d+)', (99999,)),
            ],
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, "gradient checkpointing not supported"

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.classifier

    def reset_classifier(self, num_classes: int, global_pool: str = 'avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def stages(self, x) -> List[torch.Tensor]:
        x = self.layer1(x)

        xl = [t(x) for i, t in enumerate(self.transition1)]
        yl = self.stage2(xl)

        xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(self.transition2)]
        yl = self.stage3(xl)

        xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(self.transition3)]
        yl = self.stage4(xl)
        return yl

    def forward_features(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Stages
        yl = self.stages(x)
        if self.incre_modules is None or self.downsamp_modules is None:
            return yl

        y = None
        for i, incre in enumerate(self.incre_modules):
            if y is None:
                y = incre(yl[i])
            else:
                down: ModuleInterface = self.downsamp_modules[i - 1]  # needed for torchscript module indexing
                y = incre(yl[i]) + down.forward(y)

        y = self.final_layer(y)
        return y

    def forward_head(self, x, pre_logits: bool = False):
        # Classification Head
        x = self.global_pool(x)
        x = self.head_drop(x)
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        y = self.forward_features(x)
        x = self.forward_head(y)
        return x


class HighResolutionNetFeatures(HighResolutionNet):
    """HighResolutionNet feature extraction

    The design of HRNet makes it easy to grab feature maps, this class provides a simple wrapper to do so.
    It would be more complicated to use the FeatureNet helpers.

    The `feature_location=incre` allows grabbing increased channel count features using part of the
    classification head. If `feature_location=''` the default HRNet features are returned. First stem
    conv is used for stride 2 features.
    """

    def __init__(
            self,
            cfg,
            in_chans=3,
            num_classes=1000,
            output_stride=32,
            global_pool='avg',
            drop_rate=0.0,
            feature_location='incre',
            out_indices=(0, 1, 2, 3, 4),
            **kwargs,
    ):
        assert feature_location in ('incre', '')
        super(HighResolutionNetFeatures, self).__init__(
            cfg,
            in_chans=in_chans,
            num_classes=num_classes,
            output_stride=output_stride,
            global_pool=global_pool,
            drop_rate=drop_rate,
            head=feature_location,
            **kwargs,
        )
        self.feature_info = FeatureInfo(self.feature_info, out_indices)
        self._out_idx = {f['index'] for f in self.feature_info.get_dicts()}

    def forward_features(self, x):
        assert False, 'Not supported'

    def forward(self, x) -> List[torch.Tensor]:
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if 0 in self._out_idx:
            out.append(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.stages(x)
        if self.incre_modules is not None:
            x = [incre(f) for f, incre in zip(x, self.incre_modules)]
        for i, f in enumerate(x):
            if i + 1 in self._out_idx:
                out.append(f)
        return out


def _create_hrnet(variant, pretrained=False, cfg_variant=None, **model_kwargs):
    model_cls = HighResolutionNet
    features_only = False
    kwargs_filter = None
    if model_kwargs.pop('features_only', False):
        model_cls = HighResolutionNetFeatures
        kwargs_filter = ('num_classes', 'global_pool')
        features_only = True
    cfg_variant = cfg_variant or variant

    pretrained_strict = model_kwargs.pop(
        'pretrained_strict',
        not features_only and model_kwargs.get('head', 'classification') == 'classification'
    )
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        model_cfg=cfg_cls[cfg_variant],
        pretrained_strict=pretrained_strict,
        kwargs_filter=kwargs_filter,
        **model_kwargs,
    )
    if features_only:
        model.pretrained_cfg = pretrained_cfg_for_features(model.default_cfg)
        model.default_cfg = model.pretrained_cfg  # backwards compat
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'hrnet_w18_small.gluon_in1k': _cfg(hf_hub_id='timm/', interpolation='bicubic'),
    'hrnet_w18_small.ms_in1k': _cfg(hf_hub_id='timm/'),
    'hrnet_w18_small_v2.gluon_in1k': _cfg(hf_hub_id='timm/', interpolation='bicubic'),
    'hrnet_w18_small_v2.ms_in1k': _cfg(hf_hub_id='timm/'),
    'hrnet_w18.ms_aug_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95,
    ),
    'hrnet_w18.ms_in1k': _cfg(hf_hub_id='timm/'),
    'hrnet_w30.ms_in1k': _cfg(hf_hub_id='timm/'),
    'hrnet_w32.ms_in1k': _cfg(hf_hub_id='timm/'),
    'hrnet_w40.ms_in1k': _cfg(hf_hub_id='timm/'),
    'hrnet_w44.ms_in1k': _cfg(hf_hub_id='timm/'),
    'hrnet_w48.ms_in1k': _cfg(hf_hub_id='timm/'),
    'hrnet_w64.ms_in1k': _cfg(hf_hub_id='timm/'),

    'hrnet_w18_ssld.paddle_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_crop_pct=1.0, test_input_size=(3, 288, 288)
    ),
    'hrnet_w48_ssld.paddle_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_crop_pct=1.0, test_input_size=(3, 288, 288)
    ),
})


@register_model
def hrnet_w18_small(pretrained=False, **kwargs) -> HighResolutionNet:
    return _create_hrnet('hrnet_w18_small', pretrained, **kwargs)


@register_model
def hrnet_w18_small_v2(pretrained=False, **kwargs) -> HighResolutionNet:
    return _create_hrnet('hrnet_w18_small_v2', pretrained, **kwargs)


@register_model
def hrnet_w18(pretrained=False, **kwargs) -> HighResolutionNet:
    return _create_hrnet('hrnet_w18', pretrained, **kwargs)


@register_model
def hrnet_w30(pretrained=False, **kwargs) -> HighResolutionNet:
    return _create_hrnet('hrnet_w30', pretrained, **kwargs)


@register_model
def hrnet_w32(pretrained=False, **kwargs) -> HighResolutionNet:
    return _create_hrnet('hrnet_w32', pretrained, **kwargs)


@register_model
def hrnet_w40(pretrained=False, **kwargs) -> HighResolutionNet:
    return _create_hrnet('hrnet_w40', pretrained, **kwargs)


@register_model
def hrnet_w44(pretrained=False, **kwargs) -> HighResolutionNet:
    return _create_hrnet('hrnet_w44', pretrained, **kwargs)


@register_model
def hrnet_w48(pretrained=False, **kwargs) -> HighResolutionNet:
    return _create_hrnet('hrnet_w48', pretrained, **kwargs)


@register_model
def hrnet_w64(pretrained=False, **kwargs) -> HighResolutionNet:
    return _create_hrnet('hrnet_w64', pretrained, **kwargs)


@register_model
def hrnet_w18_ssld(pretrained=False, **kwargs) -> HighResolutionNet:
    kwargs.setdefault('head_conv_bias', False)
    return _create_hrnet('hrnet_w18_ssld', cfg_variant='hrnet_w18', pretrained=pretrained, **kwargs)


@register_model
def hrnet_w48_ssld(pretrained=False, **kwargs) -> HighResolutionNet:
    kwargs.setdefault('head_conv_bias', False)
    return _create_hrnet('hrnet_w48_ssld', cfg_variant='hrnet_w48', pretrained=pretrained, **kwargs)

