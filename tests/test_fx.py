import pytest

import os

import torch

try:
    from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names, NodePathTracer
    has_fx_feature_extraction = True
except ImportError:
    has_fx_feature_extraction = False

from timm import list_models, create_model, get_model_default_value, set_scriptable
from timm.models.fx_features import _leaf_modules, _autowrap_functions


TARGET_FWD_SIZE = MAX_FWD_SIZE = 384
TARGET_BWD_SIZE = 128
MAX_BWD_SIZE = 320
MAX_FWD_OUT_SIZE = 448
TARGET_JIT_SIZE = 128
MAX_JIT_SIZE = 320
TARGET_FFEAT_SIZE = 96
MAX_FFEAT_SIZE = 256
TARGET_FWD_FX_SIZE = 128
MAX_FWD_FX_SIZE = 224
TARGET_BWD_FX_SIZE = 128
MAX_BWD_FX_SIZE = 224


# exclude models that cause specific test failures
if 'GITHUB_ACTIONS' in os.environ:  # and 'Linux' in platform.system():
    # GitHub Linux runner is slower and hits memory limits sooner than MacOS, exclude bigger models
    EXCLUDE_FILTERS = [
        '*efficientnet_l2*', '*resnext101_32x48d', '*in21k', '*152x4_bitm', '*101x3_bitm', '*50x3_bitm',
        '*nfnet_f3*', '*nfnet_f4*', '*nfnet_f5*', '*nfnet_f6*', '*nfnet_f7*', '*efficientnetv2_xl*',
        '*resnetrs350*', '*resnetrs420*', 'xcit_large_24_p8*']
else:
    EXCLUDE_FILTERS = []


def _get_input_size(model=None, model_name='', target=None):
    if model is None:
        assert model_name, "One of model or model_name must be provided"
        input_size = get_model_default_value(model_name, 'input_size')
        fixed_input_size = get_model_default_value(model_name, 'fixed_input_size')
        min_input_size = get_model_default_value(model_name, 'min_input_size')
    else:
        default_cfg = model.default_cfg
        input_size = default_cfg['input_size']
        fixed_input_size = default_cfg.get('fixed_input_size', None)
        min_input_size = default_cfg.get('min_input_size', None)
    assert input_size is not None

    if fixed_input_size:
        return input_size

    if min_input_size:
        if target and max(input_size) > target:
            input_size = min_input_size
    else:
        if target and max(input_size) > target:
            input_size = tuple([min(x, target) for x in input_size])
    return input_size


def _create_fx_model(model, train=False):
    # This block of code does a bit of juggling to handle any case where there are multiple outputs in train mode
    # So we trace once and look at the graph, and get the indices of the nodes that lead into the original fx output
    # node. Then we use those indices to select from train_nodes returned by torchvision get_graph_node_names
    train_nodes, eval_nodes = get_graph_node_names(
        model, tracer_kwargs={'leaf_modules': list(_leaf_modules), 'autowrap_functions': list(_autowrap_functions)})

    eval_return_nodes = [eval_nodes[-1]]
    train_return_nodes = [train_nodes[-1]]
    if train:
        tracer = NodePathTracer(leaf_modules=list(_leaf_modules), autowrap_functions=list(_autowrap_functions))
        graph = tracer.trace(model)
        graph_nodes = list(reversed(graph.nodes))
        output_node_names = [n.name for n in graph_nodes[0]._input_nodes.keys()]
        graph_node_names = [n.name for n in graph_nodes]
        output_node_indices = [-graph_node_names.index(node_name) for node_name in output_node_names]
        train_return_nodes = [train_nodes[ix] for ix in output_node_indices]

    fx_model = create_feature_extractor(
        model, train_return_nodes=train_return_nodes, eval_return_nodes=eval_return_nodes,
        tracer_kwargs={'leaf_modules': list(_leaf_modules), 'autowrap_functions': list(_autowrap_functions)})
    return fx_model


@pytest.mark.timeout(120)
@pytest.mark.parametrize('model_name', list_models(exclude_filters=EXCLUDE_FILTERS))
@pytest.mark.parametrize('batch_size', [1])
def test_model_forward_fx(model_name, batch_size):
    """
    Symbolically trace each model and run single forward pass through the resulting GraphModule
    Also check that the output of a forward pass through the GraphModule is the same as that from the original Module
    """
    if not has_fx_feature_extraction:
        pytest.skip("Can't test FX. Torch >= 1.10 and Torchvision >= 0.11 are required.")

    model = create_model(model_name, pretrained=False)
    model.eval()

    input_size = _get_input_size(model=model, target=TARGET_FWD_FX_SIZE)
    if max(input_size) > MAX_FWD_FX_SIZE:
        pytest.skip("Fixed input size model > limit.")
    with torch.no_grad():
        inputs = torch.randn((batch_size, *input_size))
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = torch.cat(outputs)

        model = _create_fx_model(model)
        fx_outputs = tuple(model(inputs).values())
        if isinstance(fx_outputs, tuple):
            fx_outputs = torch.cat(fx_outputs)

    assert torch.all(fx_outputs == outputs)
    assert outputs.shape[0] == batch_size
    assert not torch.isnan(outputs).any(), 'Output included NaNs'


@pytest.mark.timeout(120)
@pytest.mark.parametrize('model_name', list_models(
    exclude_filters=EXCLUDE_FILTERS, name_matches_cfg=True))
@pytest.mark.parametrize('batch_size', [2])
def test_model_backward_fx(model_name, batch_size):
    """Symbolically trace each model and run single backward pass through the resulting GraphModule"""
    if not has_fx_feature_extraction:
        pytest.skip("Can't test FX. Torch >= 1.10 and Torchvision >= 0.11 are required.")

    input_size = _get_input_size(model_name=model_name, target=TARGET_BWD_FX_SIZE)
    if max(input_size) > MAX_BWD_FX_SIZE:
        pytest.skip("Fixed input size model > limit.")

    model = create_model(model_name, pretrained=False, num_classes=42)
    model.train()
    num_params = sum([x.numel() for x in model.parameters()])
    if 'GITHUB_ACTIONS' in os.environ and num_params > 100e6:
        pytest.skip("Skipping FX backward test on model with more than 100M params.")

    model = _create_fx_model(model, train=True)
    outputs = tuple(model(torch.randn((batch_size, *input_size))).values())
    if isinstance(outputs, tuple):
        outputs = torch.cat(outputs)
    outputs.mean().backward()
    for n, x in model.named_parameters():
        assert x.grad is not None, f'No gradient for {n}'
    num_grad = sum([x.grad.numel() for x in model.parameters() if x.grad is not None])

    assert outputs.shape[-1] == 42
    assert num_params == num_grad, 'Some parameters are missing gradients'
    assert not torch.isnan(outputs).any(), 'Output included NaNs'


EXCLUDE_JIT_FILTERS = [
    '*iabn*', 'tresnet*',  # models using inplace abn unlikely to ever be scriptable
    'dla*', 'hrnet*', 'ghostnet*',  # hopefully fix at some point
    'vit_large_*', 'vit_huge_*',
]

# reason: model is scripted after fx tracing, but beit has torch.jit.is_scripting() control flow
EXCLUDE_FX_JIT_FILTERS = [
    'deit_*_distilled_patch16_224',
    'levit*',
    'pit_*_distilled_224',
]

@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    'model_name', list_models(
        exclude_filters=EXCLUDE_FILTERS + EXCLUDE_JIT_FILTERS + EXCLUDE_FX_JIT_FILTERS, name_matches_cfg=True))
@pytest.mark.parametrize('batch_size', [1])
def test_model_forward_fx_torchscript(model_name, batch_size):
    """Symbolically trace each model, script it, and run single forward pass"""
    if not has_fx_feature_extraction:
        pytest.skip("Can't test FX. Torch >= 1.10 and Torchvision >= 0.11 are required.")

    input_size = _get_input_size(model_name=model_name, target=TARGET_JIT_SIZE)
    if max(input_size) > MAX_JIT_SIZE:
        pytest.skip("Fixed input size model > limit.")

    with set_scriptable(True):
        model = create_model(model_name, pretrained=False)
    model.eval()

    model = torch.jit.script(_create_fx_model(model))
    with torch.no_grad():
        outputs = tuple(model(torch.randn((batch_size, *input_size))).values())
        if isinstance(outputs, tuple):
            outputs = torch.cat(outputs)

    assert outputs.shape[0] == batch_size
    assert not torch.isnan(outputs).any(), 'Output included NaNs'
    