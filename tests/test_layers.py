import torch
import torch.nn as nn

from timm.layers import create_act_layer, set_layer_config

import importlib
import os

torch_backend = os.environ.get('TORCH_BACKEND')
if torch_backend is not None:
    importlib.import_module(torch_backend)
torch_device = os.environ.get('TORCH_DEVICE', 'cpu')

class MLP(nn.Module):
    def __init__(self, act_layer="relu", inplace=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1000, 100)
        self.act = create_act_layer(act_layer, inplace=inplace)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def _run_act_layer_grad(act_type, inplace=True):
    x = torch.rand(10, 1000) * 10
    m = MLP(act_layer=act_type, inplace=inplace)

    def _run(x, act_layer=''):
        if act_layer:
            # replace act layer if set
            m.act = create_act_layer(act_layer, inplace=inplace)
        out = m(x)
        l = (out - 0).pow(2).sum()
        return l

    x = x.to(device=torch_device)
    m.to(device=torch_device)

    out_me = _run(x)

    with set_layer_config(scriptable=True):
        out_jit = _run(x, act_type)

    assert torch.isclose(out_jit, out_me)

    with set_layer_config(no_jit=True):
        out_basic = _run(x, act_type)

    assert torch.isclose(out_basic, out_jit)


def test_swish_grad():
    for _ in range(100):
        _run_act_layer_grad('swish')


def test_mish_grad():
    for _ in range(100):
        _run_act_layer_grad('mish')


def test_hard_sigmoid_grad():
    for _ in range(100):
        _run_act_layer_grad('hard_sigmoid', inplace=None)


def test_hard_swish_grad():
    for _ in range(100):
        _run_act_layer_grad('hard_swish')


def test_hard_mish_grad():
    for _ in range(100):
        _run_act_layer_grad('hard_mish')


MLDECODER_EXCLUDE_FILTERS = [
    '*efficientnet_l2*', '*resnext101_32x48d', '*in21k', '*152x4_bitm', '*101x3_bitm', '*50x3_bitm',
    '*nfnet_f3*', '*nfnet_f4*', '*nfnet_f5*', '*nfnet_f6*', '*nfnet_f7*', '*efficientnetv2_xl*',
    '*resnetrs350*', '*resnetrs420*', 'xcit_large_24_p8*', '*huge*', '*giant*', '*gigantic*',
    '*enormous*', 'maxvit_xlarge*', 'regnet*1280', 'regnet*2560']

def test_ml_decoder():
    for modelName in timm.list_models(pretrained=False, exclude_filters = MLDECODER_EXCLUDE_FILTERS):
        model = timm.create_model(modelName, num_classes=1000)
        model = add_ml_decoder_head(model)
        model.eval()
        with torch.set_grad_enabled(False):
            model(torch.randn([1,*model.default_cfg['input_size']]))