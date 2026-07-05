# CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities
# Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni
# Conference on Computer Vision and Pattern Recognition (CVPR), 2025

import os, yaml, math
from typing import Any
from inspect import signature
from copy import deepcopy
import torch

import torch.nn.functional as F
import torch.nn as nn
from functools import partial


class SafeLoaderWithTuple(yaml.SafeLoader):
    """A yaml safe loader with python tuple loading capabilities."""

    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


SafeLoaderWithTuple.add_constructor("tag:yaml.org,2002:python/tuple", SafeLoaderWithTuple.construct_python_tuple)

def val2tuple(x: list | tuple | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def val2list(x: list | tuple | Any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]

def build_kwargs_from_config(config: dict, target_func: callable) -> dict[str, any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs

def partial_update_config(config: dict, partial_config: dict) -> dict:
    for key in partial_config:
        if key in config and isinstance(partial_config[key], dict) and isinstance(config[key], dict):
            partial_update_config(config[key], partial_config[key])
        else:
            config[key] = partial_config[key]
    return config

def load_config(filename: str) -> dict:
    """Load a yaml file."""
    filename = os.path.realpath(os.path.expanduser(filename))
    return yaml.load(open(filename), Loader=SafeLoaderWithTuple)


def setup_exp_config(config_path: str, recursive=True, opt_args: dict | None = None) -> dict:
    # load config
    if not os.path.isfile(config_path):
        raise ValueError(config_path)

    fpaths = [config_path]
    if recursive:
        extension = os.path.splitext(config_path)[1]
        while os.path.dirname(config_path) != config_path:
            config_path = os.path.dirname(config_path)
            fpath = os.path.join(config_path, "default" + extension)
            if os.path.isfile(fpath):
                fpaths.append(fpath)
        fpaths = fpaths[::-1]

    default_config = load_config(fpaths[0])
    exp_config = deepcopy(default_config)
    for fpath in fpaths[1:]:
        partial_update_config(exp_config, load_config(fpath))
    # update config via args
    if opt_args is not None:
        partial_update_config(exp_config, opt_args)

    return exp_config


def load_state_dict_from_file(file: str, only_state_dict=True) -> dict[str, torch.Tensor]:
    # file = os.path.realpath(os.path.expanduser(file))
    checkpoint = torch.load(file, map_location="cpu")
    if "epoch" in checkpoint:
        print("checkpoint from epoch %d and its best validation result is %.3f" % (checkpoint["epoch"],checkpoint["best_val"]))
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint


def remap_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap keys from older checkpoint formats to the current model layout."""
    remapped = {}
    for k, v in state_dict.items():
        # conv_proj was nn.Sequential([conv, bn]) → now ConvLayer with .conv / .norm
        k = k.replace(".conv_proj.0.", ".conv_proj.conv.")
        k = k.replace(".conv_proj.1.", ".conv_proj.norm.")
        remapped[k] = v
    return remapped


def build_norm(name="bn2d", num_features=None, **kwargs) -> nn.Module | None:
    
    REGISTERED_NORM_DICT: dict[str, type] = {
        "bn2d": nn.BatchNorm2d,
        "ln": nn.LayerNorm,
        # "ln2d": LayerNorm2d,
    }
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None




def build_act(name: str, **kwargs) -> nn.Module | None:
    REGISTERED_ACT_DICT: dict[str, type] = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "hswish": nn.Hardswish,
        "silu": nn.SiLU,
        "gelu": partial(nn.GELU, approximate="tanh"),
    }
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None


def get_same_padding(kernel_size: int | tuple[int, ...], stride=1) -> int | tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    elif kernel_size == 2:
        if stride==2:
            return 0
        return -1 #(1,1)#(1,0)
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


#       /\
#      /||\
#       ||
#       ||
#       ||
# helper functions only

#########################################    MODEL MODULES     ######################################################

class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout=0,
        norm=None,
        act_func=None,
        squeeze_it=False,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)
        self.squeeze_it = squeeze_it

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if self.squeeze_it:# or x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if not self.dropout is None:
            x = self.dropout(x)
        x = self.linear(x)
        if not self.norm is None:
            x = self.norm(x)
        if not self.act is None:
            x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: nn.Module | None,
        shortcut: nn.Module | None,
        post_act=None,
        pre_norm: nn.Module | None = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if not self.post_act is None:
                res = self.post_act(res)
        return res



class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size, stride)
        # padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        if padding == -1:
            self.conv = nn.Sequential(torch.nn.ZeroPad2d((1,0,1,0)),
                nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
                padding=0,
                dilation=(dilation, dilation),
                groups=groups,
                bias=use_bias,
            ))
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
                padding=padding,
                dilation=(dilation, dilation),
                groups=groups,
                bias=use_bias,
            )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if not self.norm is None:
            x = self.norm(x)
        if not self.act is None:
            x = self.act(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        grouping=1,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super(MBConv, self).__init__()

        self.stride = stride
        self.in_channels = in_channels
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        # pwise
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            groups=grouping,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        # dwise
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

        # pwise
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            groups=1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
    
        x = self.depth_conv(x)

        x = self.point_conv(x)
        return x
    

class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            groups=1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        
        return x


class SDALayer(nn.Module):
    def __init__(self):
        super().__init__()

    def scaled_dot_product(self, q, k, v):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values

    def forward(self, q, k, v) -> torch.Tensor:
        return self.scaled_dot_product(q, k, v)


class ConvAttention(nn.Module):
    def __init__(self, input_dim, head_dim_mul=1.0, att_stride=4, att_kernel=7, fuseconv=False, smallkernel=False, lose_transpose=False):
        super().__init__()

        self.head_dim_mul = head_dim_mul
        self.num_heads = int(max(1, (input_dim * self.head_dim_mul) // 30))
        self.input_dim = input_dim
        self.head_dim = int((input_dim // self.num_heads) * self.head_dim_mul)
        self.num_keys = 3
        self.att_stride = att_stride

        total_dim = int(self.head_dim * self.num_heads * self.num_keys)

        self.conv_proj = ConvLayer(input_dim, input_dim, kernel_size=2 if smallkernel else att_kernel, norm="bn2d", act_func=None, stride=att_stride, groups=input_dim)
        self.pwise = nn.Sequential(nn.Conv2d(input_dim, total_dim, kernel_size=1, stride=1, padding=0, bias=False))
        self.sda = SDALayer()

        self.o_proj_inpdim = self.head_dim * self.num_heads
        self.o_proj = nn.Conv2d(self.o_proj_inpdim, input_dim, kernel_size=1, stride=1, padding=0)

        self.upsampling = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=att_stride*2, stride=att_stride, padding=att_stride//2, groups=input_dim)
        if att_stride == 1:
            self.upsampling = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim)

        if fuseconv:
            self.o_proj = nn.Identity()
            if att_stride == 1:
                self.upsampling = nn.ConvTranspose2d(self.o_proj_inpdim, input_dim, kernel_size=3, stride=1, padding=1)
            else:
                self.upsampling = nn.ConvTranspose2d(self.o_proj_inpdim, input_dim, kernel_size=att_stride*2, stride=att_stride, padding=att_stride//2)

        if lose_transpose:
            upsampling = [nn.Upsample(scale_factor=att_stride, mode="nearest") if att_stride > 1 else nn.Identity()]
            if fuseconv:
                upsampling = [nn.Conv2d(self.o_proj_inpdim, input_dim, kernel_size=1, stride=1, padding=0)] + upsampling
            self.upsampling = nn.Sequential(*upsampling)

    def forward(self, x):
        N, C, H, W = x.size()

        xout = self.conv_proj(x)
        xout = self.pwise(xout)

        N, c, h, w = xout.size()
        qkv = xout.reshape(N, self.num_heads, self.num_keys * self.head_dim, h * w)
        qkv = qkv.permute(0, 1, 3, 2)  # [N, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=3)

        values = self.sda(q, k, v)
        o = self.o_proj(values.permute(0, 1, 3, 2).reshape(N, self.o_proj_inpdim, h, w))

        o = self.upsampling(o)
        return o[:N, :C, :H, :W]

class CPUBoneBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expand_ratio: float = 4,
        norm="bn2d",
        act_func="hswish",
        fuseconv=False,
        bb_convattention=False,
        bb_convin2=False,
        grouping=1,
        att_stride=1,
        mlpexpans=4,
        smallkernel=False,
        lose_transpose=False,
    ):
        super(CPUBoneBlock, self).__init__()

        att_kernel = 5 if att_stride > 1 else 3

        block = ConvAttention(
            input_dim=in_channels,
            att_stride=att_stride,
            att_kernel=att_kernel,
            head_dim_mul=0.5,
            fuseconv=fuseconv,
            smallkernel=smallkernel,
            lose_transpose=lose_transpose,
        )

        context_module = ResidualBlock(nn.Sequential(nn.GroupNorm(1, in_channels), block), IdentityLayer())
        mlp = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, in_channels * mlpexpans, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels * mlpexpans, in_channels, kernel_size=1),
            nn.Dropout(p=0.1),
        )
        context_module = nn.Sequential(context_module, ResidualBlock(mlp, IdentityLayer()))

        if fuseconv and in_channels < 256:
            local_module = FusedMBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, False),
                kernel_size=2 if smallkernel else 3,
                groups=grouping,
                norm=norm,
                act_func=(act_func, None),
            )
        else:
            local_module = MBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                grouping=grouping,
                use_bias=(True, True, False),
                kernel_size=2 if smallkernel else 3,
                norm=(None, None, norm),
                act_func=(act_func, act_func, None),
            )

        self.total = nn.Sequential(context_module, ResidualBlock(local_module, IdentityLayer()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.total(x)
        return x


##############################    Classification specific Classes      ###############################


class ClsHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        width_list: list[int],
        n_classes=1000,
        dropout=0.0,
        norm="bn2d",
        no_spatial=False,
        act_func="hswish",
        fid="stage_final",
    ):
        ops = [
            ConvLayer(in_channels, width_list[0], 1, norm=norm, act_func=act_func),
            nn.AdaptiveAvgPool2d(output_size=1),
            LinearLayer(in_channels if no_spatial else width_list[0], width_list[1], False, norm="ln", act_func=act_func, squeeze_it=True),
            LinearLayer(width_list[1], n_classes, True, dropout, None, None),
        ]
        if no_spatial:
            ops = ops[2:]
        super().__init__(*ops)

        self.fid = fid

    def forward(self, feed_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        x = feed_dict[self.fid]
        return nn.Sequential.forward(self, x)


class CPUBoneCls(nn.Module):
    def __init__(self, backbone, head, name: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.name = name
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)
        output = self.head(feed_dict)
        
        return output


##############################    CPUBone Backbone Architecture      ###############################

class CPUBoneBackbone(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels=3,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
        bb_convattention=False,
        bb_convin2=False,
        fastit=False,
        huge_model=False,
        bigit=False,
        grouping=1,
        smallk_only_lasts=False,
        lose_transpose=False,
        just_unfused=False,
    ) -> None:
        super().__init__()

        self.expand_ratio = expand_ratio
        self.norm = norm
        self.act_func = act_func
        self.bb_convattention = bb_convattention
        self.bb_convin2 = bb_convin2
        self.fuseconv = fastit and not just_unfused
        self.fastit = fastit
        self.huge_model = huge_model
        self.bigit = bigit
        self.grouping = grouping
        self.smallk_only_lasts = smallk_only_lasts
        self.lose_transpose = lose_transpose

        self.input_stem, in_channels = self._build_stem(in_channels, width_list[0], depth_list[0])

        ### Stages 1-4: early stages use plain conv blocks, later stages add attention
        self.stages = nn.ModuleList()
        for stage_num, (width, depth) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            if stage_num >= 3:
                blocks, in_channels = self._build_attention_stage(in_channels, width, depth, stage_num)
            else:
                blocks, in_channels = self._build_conv_stage(in_channels, width, depth)
            self.stages.append(nn.Sequential(*blocks))

    def _build_stem(self, in_channels: int, stem_width: int, depth: int) -> tuple[nn.Sequential, int]:
        """Stem: downsample by 2, then `depth` local blocks at the stem width."""
        blocks = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=stem_width,
                kernel_size=3,
                stride=2,
                norm=self.norm,
                act_func=self.act_func,
            )
        ]
        in_channels = stem_width
        for _ in range(depth):
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=in_channels,
                stride=1,
                expand_ratio=4 if self.huge_model else 2,
                fusedmbconv=self.fuseconv,
                grouping=self.grouping,
                norm=self.norm,
                act_func=self.act_func,
            )
            blocks.append(ResidualBlock(block, IdentityLayer()))
        return nn.Sequential(*blocks), in_channels

    def _build_conv_stage(self, in_channels: int, width: int, depth: int) -> tuple[list[nn.Module], int]:
        """Stages 1-2: `depth` plain conv blocks, downsampling (stride 2) on the first one."""
        blocks = []
        for i in range(depth):
            stride = 2 if i == 0 else 1
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=width,
                stride=stride,
                expand_ratio=6 if stride == 2 and (self.bigit or self.huge_model) else self.expand_ratio,
                fusedmbconv=self.fuseconv,
                grouping=self.grouping,
                norm=self.norm,
                act_func=self.act_func,
            )
            blocks.append(ResidualBlock(block, IdentityLayer() if stride == 1 else None))
            in_channels = width
        return blocks, in_channels

    def _build_attention_stage(self, in_channels: int, width: int, depth: int, stage_num: int) -> tuple[list[nn.Module], int]:
        """Stages 3-4: one downsampling conv block, followed by `depth` CPUBoneBlocks (attention + local conv)."""
        downsample = self.build_local_block(
            in_channels=in_channels,
            out_channels=width,
            stride=2,
            expand_ratio=6 if self.bigit or (self.huge_model and stage_num < 4) else self.expand_ratio,
            fusedmbconv=self.fastit and (not self.huge_model or stage_num < 5),
            grouping=self.grouping,
            norm=self.norm,
            act_func=self.act_func,
        )
        in_channels = width
        blocks = [ResidualBlock(downsample, None)]
        for _ in range(depth):
            blocks.append(
                CPUBoneBlock(
                    in_channels=in_channels,
                    expand_ratio=self.expand_ratio,
                    norm=self.norm,
                    act_func=self.act_func,
                    bb_convattention=self.bb_convattention,
                    fuseconv=self.fuseconv,
                    bb_convin2=self.bb_convin2,
                    grouping=self.grouping,
                    att_stride=2 if stage_num == 3 else 1,
                    mlpexpans=4 if self.fastit else 2,
                    smallkernel=self.smallk_only_lasts,
                    lose_transpose=self.lose_transpose,
                )
            )
        return blocks, in_channels

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fusedmbconv: bool = False,
        grouping: int = 1,
        kernel_size: int = 3,
    ) -> nn.Module:
        if fusedmbconv:
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=False,
                kernel_size=kernel_size,
                groups=grouping,
                norm=norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                grouping=grouping,
                use_bias=False,
                norm=(None, None, norm),
                act_func=(act_func, act_func, None),
            )
        return block

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.input_stem(x)
        output_dict = {"stage0": x}
        num_stages = len(self.stages)
        for stage_num, stage in enumerate(self.stages, start=1):
            x = stage(x)
            key = "stage_final" if stage_num == num_stages else f"stage{stage_num}"
            output_dict[key] = x
        return output_dict

## Model function
def cpubone_backbone_b1(**kwargs) -> CPUBoneBackbone:
    width_list = [16, 32, 64, 128, 256]
    depth_list = [1, 2, 3, 3, 4]

    if "name" in kwargs:
        
        if kwargs["name"] == "nano":
            width_list =  [12, 24, 48, 96, 192]
            depth_list = [0,1,1,1,2]
        
        if kwargs["name"] == "t0":
            width_list =  [12, 24, 48, 96, 192]
            depth_list = [0,1,1,1,3]
            
        if kwargs["name"] == "s0": # t0 for grp_smklstag_notrans
            width_list =  [12, 24, 48, 96, 192]
            depth_list = [0,1,1,2,3]
        
        
        if kwargs["name"] == "s1":
            width_list =  [14, 28, 56, 112, 224]
            depth_list = [0,1,1,2,3]
        
        
        if kwargs["name"] == "b0":
            width_list =  [16, 32, 64, 128, 256]
            depth_list = [0,1,1,3,4]

        if kwargs["name"] == "b1":
            width_list =  [16, 32, 64, 128, 256]
            depth_list = [0,1,1,5,5]
        
        if kwargs["name"] == "b15":
            width_list =  [20, 40, 80, 160, 320]
            depth_list = [0,1,1,6,6]
        
        if kwargs["name"] == "b2":
            width_list =  [24, 48, 96, 192, 384]
            depth_list = [0,1,1,6,6]
        
        if kwargs["name"] == "b25":
            width_list =  [24, 48, 96, 192, 384]
            depth_list = [0,2,3,6,6]
        
        if kwargs["name"] == "b3":
            width_list =  [32, 64, 128, 256, 512]
            depth_list = [1,2,3,6,6]

        if kwargs["name"] == "b4":
            depth_list = [2, 3, 6, 12, 8]
            width_list = [64, 128, 256, 512, 1024]

        if kwargs["name"] == "b5":
            depth_list = [2, 4, 5, 20, 10]
            width_list = [128, 256, 512, 1024, 2048]

        if kwargs["name"] == "custom":
            width_list = kwargs.pop("width_list")
            depth_list = kwargs.pop("depth_list")
    
    backbone = CPUBoneBackbone(
        width_list=width_list,
        depth_list=depth_list,
        **build_kwargs_from_config(kwargs, CPUBoneBackbone),
    )
    return backbone, width_list


def cpubone_cls_b1(**kwargs) -> CPUBoneCls:

    backbone, width_list = cpubone_backbone_b1(**kwargs)

    cls_widths = [1536, 1600]
    if "bighead" in kwargs and kwargs["bighead"]:
        cls_widths = [2304, 2560]

    head = ClsHead(
        in_channels=width_list[-1],
        width_list=cls_widths,
        act_func="hswish",
        **build_kwargs_from_config(kwargs, ClsHead),
    )

    model = CPUBoneCls(backbone, head, **build_kwargs_from_config(kwargs, CPUBoneCls))
    return model



# Model retrieve function
def get_cpubone(config_path="configs/cls/imagenet/cpubone_b1.yaml", checkpoint_path=".exp/cls/imagenet/cpubone_b1/checkpoint/evalmodel.pt", pretrained=True):

    # load config as dict 
    config = setup_exp_config(config_path, recursive=True, opt_args=None)
    
    # get classification model
    model = cpubone_cls_b1(**config["net_config"])
    
    name = config["net_config"]["name"]
    print("Model:",name)

    try:
        if pretrained:
            if checkpoint_path is None:
                raise ValueError(f"Do not find the pretrained weight of {name}.")
            else:
                weight = load_state_dict_from_file(checkpoint_path)
                weight = remap_legacy_state_dict(weight)
                model.load_state_dict(weight)
    except Exception as e:
        print("Model weights could not be loaded!!!!!!!!!!!!!!!!!!!",e)
    
    return model



def get_model_by_name(name, pretrained=True):
    return get_cpubone(config_path="configs/cls/imagenet/cpubone_%s.yaml" % name, checkpoint_path=".exp/cls/imagenet/cpubone_%s/checkpoint/evalmodel.pt" % name, pretrained=pretrained)



def get_cpubone_nano(pretrained=True):
    return get_model_by_name("nano", pretrained=pretrained)

def get_cpubone_t0(pretrained=True):
    return get_model_by_name("t0", pretrained=pretrained)

def get_cpubone_s0(pretrained=True):
    return get_model_by_name("s0", pretrained=pretrained)

def get_cpubone_b0(pretrained=True):
    return get_model_by_name("b0", pretrained=pretrained)

def get_cpubone_b1(pretrained=True):
    return get_model_by_name("b1", pretrained=pretrained)

def get_cpubone_b15(pretrained=True):
    return get_model_by_name("b15", pretrained=pretrained)

def get_cpubone_b2(pretrained=True):
    return get_model_by_name("b2", pretrained=pretrained)

def get_cpubone_b25(pretrained=True):
    return get_model_by_name("b25", pretrained=pretrained)

def get_cpubone_b3(pretrained=True):
    return get_model_by_name("b3", pretrained=pretrained)
    


