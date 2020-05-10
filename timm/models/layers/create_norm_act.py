import torch
import torch.nn as nn

from .evo_norm import EvoNormBatch2d, EvoNormSample2d
from .norm_act import BatchNormAct2d
try:
    from inplace_abn import InPlaceABN
    has_iabn = True
except ImportError:
    has_iabn = False


def create_norm_act(layer_type, num_features, jit=False, **kwargs):
    layer_parts = layer_type.split('_')
    assert len(layer_parts) in (1, 2)
    layer_class = layer_parts[0].lower()
    #activation_class = layer_parts[1].lower() if len(layer_parts) > 1 else ''   # FIXME support string act selection

    if layer_class == "batchnormact":
        layer = BatchNormAct2d(num_features, **kwargs) # defaults to RELU of no kwargs override
    elif layer_class == "batchnormrelu":
        assert 'act_layer' not in kwargs
        layer = BatchNormAct2d(num_features, act_layer=nn.ReLU, **kwargs)
    elif layer_class == "evonormbatch":
        layer = EvoNormBatch2d(num_features, **kwargs)
    elif layer_class == "evonormsample":
        layer = EvoNormSample2d(num_features, **kwargs)
    elif layer_class == "iabn" or layer_class == "inplaceabn":
        if not has_iabn:
            raise ImportError(
                "Pplease install InplaceABN:'pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.11'")
        layer = InPlaceABN(num_features, **kwargs)
    else:
        assert False, "Invalid norm_act layer (%s)" % layer_class
    if jit:
        layer = torch.jit.script(layer)
    return layer
