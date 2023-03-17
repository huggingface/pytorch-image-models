""" SimpleNet

Paper: `Lets Keep it simple, Using simple architectures to outperform deeper and more complex architectures`
    - https://arxiv.org/abs/1608.06037

@article{hasanpour2016lets,
  title={Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures},
  author={Hasanpour, Seyyed Hossein and Rouhani, Mohammad and Fayyaz, Mohsen and Sabokrou, Mohammad},
  journal={arXiv preprint arXiv:1608.06037},
  year={2016}
}

Official Caffe impl at https://github.com/Coderx7/SimpleNet
Official Pythorch impl at https://github.com/Coderx7/SimpleNet_Pytorch
Seyyed Hossein Hasanpour
"""
import math
from typing import Union, Tuple, List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import ClassifierHead, create_act_layer, ConvNormAct, DropPath, make_divisible
from ._builder import build_model_with_cfg
from ._efficientnet_builder import efficientnet_init_weights
from ._manipulate import checkpoint_seq
from ._builder import build_model_with_cfg
from ._registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

__all__ = [
    "simplenet",
    "simplenetv1_small_m1_05",  # 1.5m
    "simplenetv1_small_m2_05",  # 1.5m
    "simplenetv1_small_m1_075",  # 3m
    "simplenetv1_small_m2_075",  # 3m
    "simplenetv1_5m_m1",  # 5m
    "simplenetv1_5m_m2",  # 5m
    "simplenetv1_9m_m1",  # 9m
    "simplenetv1_9m_m2",  # 9m
]  # model_registry will add each entrypoint fn to this


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "crop_pct": 0.875,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv',
        'classifier': 'head.fc',
        **kwargs,
    }


default_cfgs: Dict[str, Dict[str, Any]] = {
    "simplenetv1_small_m1_05": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_small_m1_05-a7ec600b.pth"
    ),
    "simplenetv1_small_m2_05": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_small_m2_05-62617ea1.pth"
    ),
    "simplenetv1_small_m1_075": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_small_m1_075-8427bf60.pth"
    ),
    "simplenetv1_small_m2_075": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_small_m2_075-da714eb5.pth"
    ),
    "simplenetv1_5m_m1": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_5m_m1-cc6b3ad1.pth"
    ),
    "simplenetv1_5m_m2": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_5m_m2-c35297bf.pth"
    ),
    "simplenetv1_9m_m1": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_9m_m1-8c98a0a5.pth"
    ),
    "simplenetv1_9m_m2": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_9m_m2-6b01be1e.pth"
    ),
}


class View(nn.Module):
    def forward(self, x):
        print(f"{x.shape}")
        return x


class Downsample(nn.Module):
    def __init__(self, pool='max', kernel_size=2, stride=2, dropout=0.0, inplace=True) -> None:
        super().__init__()
        self.pool = (
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
            if pool == 'max'
            else nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        )
        self.dropout = nn.Identity() if dropout is None else nn.Dropout2d(dropout, inplace=inplace)

    def forward(self, x):
        x = self.pool(x)
        x = self.dropout(x)
        return x
        # return View()(x)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, Dropout=0.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.05)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Identity() if Dropout is None else nn.Dropout2d(Dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
        # return View()(x)


class SimpleNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        in_chans: int = 3,
        scale: float = 1,
        network_idx: int = 0,
        mode: int = 2,
        drop_rates: Dict[int, float] = {},
        **kwargs,
    ):
        """Instantiates a SimpleNet model. SimpleNet is comprised of the most basic building blocks of a CNN architecture.
        It uses basic principles to maximize the network performance both in terms of feature representation and speed without
        resorting to complex design or operators.

        Args:
            num_classes (int, optional): number of classes. Defaults to 1000.
            in_chans (int, optional): number of input channels. Defaults to 3.
            scale (float, optional): scale of the architecture width. Defaults to 1.0.
            network_idx (int, optional): the network index indicating the 5 million or 8 million version(0 and 1 respectively). Defaults to 0.
            mode (int, optional): stride mode of the architecture. specifies how fast the input shrinks.
                This is used for larger input sizes such as the 224x224 in imagenet training where the
                input size incurs a lot of overhead if not downsampled properly.
                you can choose between 0 meaning no change and 4. where each number denotes a specific
                downsampling strategy. For imagenet use 1-4.
                the larger the stride mode, usually the higher accuracy and the slower
                the network gets. stride mode 1 is the fastest and achives very good accuracy.
                Defaults to 2.
            drop_rates (Dict[int,float], optional): custom drop out rates specified per layer.
                each rate should be paired with the corrosponding layer index(pooling and cnn layers are counted only). Defaults to {}.
        """
        super(SimpleNet, self).__init__()
        self.output_stride = 32
        self.grad_checkpointing = False
        # (channels or layer-type, stride=1, drp=0.)
        self.cfg = {
            "simplenetv1_imagenet": {
                'stem': [(64, 1, 0.0)],
                'stage_': [
                    (128, 1, 0.0),
                    (128, 1, 0.0),
                    (128, 1, 0.0),
                    (128, 1, 0.0),
                    (128, 1, 0.0),
                    ("p", 2, 0.0),
                    (256, 1, 0.0),
                    (256, 1, 0.0),
                    (256, 1, 0.0),
                    (512, 1, 0.0),
                    ("p", 2, 0.0),
                    (2048, 1, 0.0, "k1"),
                    (256, 1, 0.0, "k1"),
                    (256, 1, 0.0),
                ],
            },
            "simplenetv1_imagenet_9m": {
                'stem': [(128, 1, 0.0)],
                'stage_': [
                    (192, 1, 0.0),
                    (192, 1, 0.0),
                    (192, 1, 0.0),
                    (192, 1, 0.0),
                    (192, 1, 0.0),
                    ("p", 2, 0.0),
                    (320, 1, 0.0),
                    (320, 1, 0.0),
                    (320, 1, 0.0),
                    (640, 1, 0.0),
                    ("p", 2, 0.0),
                    (2560, 1, 0.0, "k1"),
                    (320, 1, 0.0, "k1"),
                    (320, 1, 0.0),
                ],
            },
        }

        self.networks = [
            "simplenetv1_imagenet",  # 0
            "simplenetv1_imagenet_9m",  # 1
            # other archs
        ]
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.scale = scale
        self.network_idx = network_idx
        self.mode = mode
        self.selected_network = self.cfg[self.networks[self.network_idx]]
        # making sure all values are in correct form
        self.dropout_rates = {int(key): float(value) for key, value in drop_rates.items()}
        # 15 is the last layer of the network(including two previous pooling layers)
        # basically specifying the dropout rate for the very last layer to be used after the pooling
        # but if we add or remove some layers later on, it will mess thing up, so lets do it dynamically
        last_layer_idx = sum(len(v) for _, v in self.selected_network.items())
        self.last_dropout_rate = self.dropout_rates.get(last_layer_idx, 0.0)
        self.strides = {
            0: {},
            1: {0: 2, 1: 2, 2: 2},
            2: {0: 2, 1: 2, 2: 1, 3: 2},
            3: {0: 2, 1: 2, 2: 1, 3: 1, 4: 2},
            4: {0: 2, 1: 1, 2: 2, 3: 1, 4: 2, 5: 1},
        }

        self.features, self.feature_info = self._build_blocks()
        self.num_features = round(self.selected_network['stage_'][-1][0] * scale)
        self.head = ClassifierHead(self.num_features, num_classes, 'max', self.last_dropout_rate)

    def _build_blocks(self):
        net_id = self.network_idx
        features = nn.Sequential()
        feature_info = []
        in_chan = self.in_chans
        current_stride = 1
        for idx, (block_key, block_info) in enumerate(self.cfg[self.networks[net_id]].items()):
            block_strides = self.extract_block_strides(block_key)
            block_dropouts = self.extract_block_dropouts(block_key)
            if block_key == 'stem':
                filters, default_stride, defaul_dropout_rate = block_info[0]
                self.stem_chs = round(filters * self.scale)
                self.stem_stride = block_strides.get(idx, default_stride)
                custom_dropout = self.get_final_dropout(idx, block_dropouts, defaul_dropout_rate)
                self.stem = ConvBNReLU(
                    in_chan, self.stem_chs, 3, stride=self.stem_stride, padding=1, Dropout=custom_dropout
                )
                feature_info += [dict(num_chs=self.stem_chs, reduction=self.stem_stride, module='stem')]
                in_chan = self.stem_chs
                reduction_rate = self.stem_stride
            else:
                stage_index = -1
                stage_list = []
                stage_id = f'stage_0'
                for idx, (filter, current_stride, current_dropout, *layer_type) in enumerate(block_info):
                    stage_id = f'stage_{stage_index}'
                    # check the current_stride
                    final_stride = block_strides.get(idx, current_stride)
                    # check final dropout
                    custom_dropout = self.get_final_dropout(idx, block_dropouts, current_dropout)
                    pad = 1
                    if layer_type == []:
                        kernel_size = 3
                        pad = 1
                    else:
                        kernel_size = 1
                        pad = 0

                    if final_stride > 1 or filter == 'p':
                        if stage_list:
                            features.add_module(stage_id, nn.Sequential(*stage_list))
                            feature_info += [
                                dict(num_chs=filters, reduction=reduction_rate, module=f'features.{stage_id}')
                            ]

                        stage_index += 1
                        reduction_rate *= final_stride
                        stage_list = []

                    if filter == 'p':
                        stage_list.append(Downsample(dropout=custom_dropout))
                    else:
                        filters = round(filter * self.scale)
                        if custom_dropout is None:
                            stage_list.append(
                                ConvBNReLU(
                                    in_chan,
                                    filters,
                                    kernel_size=kernel_size,
                                    stride=final_stride,
                                    padding=pad,
                                    Dropout=None,
                                )
                            )
                        else:
                            stage_list.append(
                                ConvBNReLU(
                                    in_chan,
                                    filters,
                                    kernel_size=kernel_size,
                                    stride=final_stride,
                                    padding=pad,
                                    Dropout=custom_dropout,
                                )
                            )
                        in_chan = filters

                if stage_id:
                    features.add_module(stage_id, nn.Sequential(*stage_list))
                    feature_info += [
                        dict(num_chs=filters, reduction=int(reduction_rate), module=f'features.{stage_id}')
                    ]

        # init the model weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

        return features, feature_info

    def get_final_dropout(self, idx, block_dropouts, current_dropout):
        custom_dropout = block_dropouts.get(idx, None)
        custom_dropout = current_dropout if custom_dropout is None else custom_dropout
        # dropout values must be strictly decimal. while 0 doesnt introduce any issues here
        # i.e. during training and inference, if you try to jit trace your model it will crash
        # due to using 0 as dropout value so here is an explicit
        # check to convert any possible integer value to its decimal counterpart.
        custom_dropout = None if custom_dropout is None else float(custom_dropout)
        return custom_dropout

    def extract_block_strides(self, block_key):
        strides = self.strides[self.mode]
        return self.process_block_info(block_key, strides)

    def extract_block_dropouts(self, block_key):
        return self.process_block_info(block_key, self.dropout_rates)

    def get_stage_info(self):
        stage_info = {}
        idx = 0
        for k, v in self.cfg[self.networks[self.network_idx]].items():
            layer_cnt = len(v)
            stage_info[k] = (idx, idx + layer_cnt, list(range(idx, idx + layer_cnt)))
            idx += layer_cnt
        return stage_info

    def process_block_info(self, block_key, data_dict):
        stage_info = self.get_stage_info()
        block_rates = {}
        (_, _, idx_list) = stage_info[block_key]
        for k, v in data_dict.items():
            if k in idx_list:
                key_in_block = idx_list.index(k)
                block_rates[key_in_block] = v
        return block_rates

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',
            blocks=r'^features\._stage_(\d+)\.(\d+)',
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, drop_rate=0.0, global_pool='max'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.features, x, flatten=True)
        else:
            x = self.features(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _checkpoint_filter_fn(state_dict, model):
    """Remaps original checkpoints -> timm"""
    # shamelessly taken from https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/levit.py#L696
    if 'stem.0.weight' in state_dict:
        return state_dict  # non-original checkpoint, no remapping needed

    out_dict = {}
    import re

    D = model.state_dict()
    out_dict = {}
    for ka, kb, va, vb in zip(D.keys(), state_dict.keys(), D.values(), state_dict.values()):
        if va.ndim == 4 and vb.ndim == 2:
            vb = vb[:, :, None, None]
        if va.shape != vb.shape:
            # head or first-conv shapes may change for fine-tune
            assert 'head' in ka or 'stem.conv1.linear' in ka
        out_dict[ka] = vb
    return out_dict


def _gen_simplenet(
    model_variant: str = "simplenetv1_m2",
    num_classes: int = 1000,
    in_chans: int = 3,
    scale: float = 1.0,
    network_idx: int = 0,
    mode: int = 2,
    pretrained: bool = False,
    drop_rates: Dict[int, float] = {},
    **kwargs,
) -> SimpleNet:
    model_args = dict(
        in_chans=in_chans,
        scale=scale,
        network_idx=network_idx,
        mode=mode,
        drop_rates=drop_rates,
        **kwargs,
    )
    model = build_model_with_cfg(
        SimpleNet,
        model_variant,
        pretrained,
        pretrained_filter_fn=_checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True),
        **model_args,
    )
    return model


@register_model
def simplenet(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Generic simplenet model builder. by default it returns `simplenetv1_5m_m2` model
    but specifying different arguments such as `netidx`, `scale` or `mode` will result in
    the corrosponding network variant.

    when pretrained is specified, if the combination of settings resemble any known variants
    specified in the `default_cfg`, their respective pretrained weights will be loaded, otherwise
    an exception will be thrown denoting Unknown model variant being specified.

    Args:
        pretrained (bool, optional): loads the model with pretrained weights only if the model is a known variant specified in default_cfg. Defaults to False.

    Raises:
        Exception: if pretrained is used with an unknown/custom model variant and exception is raised.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation.
    """
    num_classes = kwargs.get("num_classes", 1000)
    in_chans = kwargs.get("in_chans", 3)
    scale = kwargs.get("scale", 1.0)
    network_idx = kwargs.get("network_idx", 0)
    mode = kwargs.get("mode", 2)
    drop_rates = kwargs.get("drop_rates", {})
    model_variant = "simplenetv1_5m_m2"
    if pretrained:
        # check if the model specified is a known variant
        model_base = None
        if network_idx == 0:
            model_base = 5
        elif network_idx == 1:
            model_base = 9
        config = ""
        if math.isclose(scale, 1.0):
            config = f"{model_base}m_m{mode}"
        elif math.isclose(scale, 0.75):
            config = f"small_m{mode}_075"
        elif math.isclose(scale, 0.5):
            config = f"small_m{mode}_05"
        else:
            config = f"m{mode}_{scale:.2f}".replace(".", "")
        model_variant = f"simplenetv1_{config}"

        cfg = default_cfgs.get(model_variant, None)
        if cfg is None:
            raise Exception(f"Unknown model variant ('{model_variant}') specified!")

    return _gen_simplenet(model_variant, num_classes, in_chans, scale, network_idx, mode, pretrained, drop_rates)


def remove_network_settings(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Removes network related settings passed in kwargs for predefined network configruations below

    Returns:
        Dict[str,Any]: cleaned kwargs
    """
    model_args = {k: v for k, v in kwargs.items() if k not in ["scale", "network_idx", "mode", "drop_rate"]}
    return model_args


# imagenet models
@register_model
def simplenetv1_small_m1_05(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates a small variant of simplenetv1_5m, with 1.5m parameters. This uses m1 stride mode
    which makes it the fastest variant available.

    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation.
    """
    model_variant = "simplenetv1_small_m1_05"
    model_args = remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=0.5, network_idx=0, mode=1, pretrained=pretrained, **model_args)


@register_model
def simplenetv1_small_m2_05(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates a second small variant of simplenetv1_5m, with 1.5m parameters. This uses m2 stride mode
    which makes it the second fastest variant available.

    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation.
    """
    model_variant = "simplenetv1_small_m2_05"
    model_args = remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=0.5, network_idx=0, mode=2, pretrained=pretrained, **model_args)


@register_model
def simplenetv1_small_m1_075(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates a third small variant of simplenetv1_5m, with 3m parameters. This uses m1 stride mode
    which makes it the third fastest variant available.

    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation.
    """
    model_variant = "simplenetv1_small_m1_075"
    model_args = remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=0.75, network_idx=0, mode=1, pretrained=pretrained, **model_args)


@register_model
def simplenetv1_small_m2_075(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates a forth small variant of simplenetv1_5m, with 3m parameters. This uses m2 stride mode
    which makes it the forth fastest variant available.

    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation.
    """
    model_variant = "simplenetv1_small_m2_075"
    model_args = remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=0.75, network_idx=0, mode=2, pretrained=pretrained, **model_args)


@register_model
def simplenetv1_5m_m1(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates the base simplement model known as simplenetv1_5m, with 5m parameters. This variant uses m1 stride mode
    which makes it a fast and performant model.

    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation.
    """
    model_variant = "simplenetv1_5m_m1"
    model_args = remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=1.0, network_idx=0, mode=1, pretrained=pretrained, **model_args)


@register_model
def simplenetv1_5m_m2(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates the base simplement model known as simplenetv1_5m, with 5m parameters. This variant uses m2 stride mode
    which makes it a bit more performant model compared to the m1 variant of the same variant at the expense of a bit slower inference.

    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation.
    """
    model_variant = "simplenetv1_5m_m2"
    model_args = remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=1.0, network_idx=0, mode=2, pretrained=pretrained, **model_args)


@register_model
def simplenetv1_9m_m1(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates a variant of the simplenetv1_5m, with 9m parameters. This variant uses m1 stride mode
    which makes it run faster.

    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation.
    """
    model_variant = "simplenetv1_9m_m1"
    model_args = remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=1.0, network_idx=1, mode=1, pretrained=pretrained, **model_args)


@register_model
def simplenetv1_9m_m2(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates a variant of the simplenetv1_5m, with 9m parameters. This variant uses m2 stride mode
    which makes it a bit more performant model compared to the m1 variant of the same variant at the expense of a bit slower inference.

    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation.
    """
    model_variant = "simplenetv1_9m_m2"
    model_args = remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=1.0, network_idx=1, mode=2, pretrained=pretrained, **model_args)


if __name__ == "__main__":
    model = simplenet(num_classes=1000, pretrained=True)
    input_dummy = torch.randn(size=(1, 224, 224, 3))
    out = model(input_dummy)
    print(f"output: {out.size()}")
