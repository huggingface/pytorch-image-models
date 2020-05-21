import pytest
import torch

from timm import list_models, create_model


@pytest.mark.timeout(120)
@pytest.mark.parametrize('model_name', list_models())
@pytest.mark.parametrize('batch_size', [1])
def test_model_forward(model_name, batch_size):
    """Run a single forward pass with each model"""
    model = create_model(model_name, pretrained=False)
    model.eval()

    input_size = model.default_cfg['input_size']
    if any([x > 448 for x in input_size]):
        # cap forward test at max res 448 * 448 to keep resource down
        input_size = tuple([min(x, 448) for x in input_size])
    inputs = torch.randn((batch_size, *input_size))
    outputs = model(inputs)

    assert outputs.shape[0] == batch_size
    assert not torch.isnan(outputs).any(), 'Output included NaNs'


@pytest.mark.timeout(120)
@pytest.mark.parametrize('model_name', list_models(exclude_filters='dla*'))  # DLA models have an issue TBD
@pytest.mark.parametrize('batch_size', [2])
def test_model_backward(model_name, batch_size):
    """Run a single forward pass with each model"""
    model = create_model(model_name, pretrained=False, num_classes=42)
    num_params = sum([x.numel() for x in model.parameters()])
    model.eval()

    input_size = model.default_cfg['input_size']
    if any([x > 128 for x in input_size]):
        # cap backward test at 128 * 128 to keep resource usage down
        input_size = tuple([min(x, 128) for x in input_size])
    inputs = torch.randn((batch_size, *input_size))
    outputs = model(inputs)
    outputs.mean().backward()
    num_grad = sum([x.grad.numel() for x in model.parameters() if x.grad is not None])

    assert outputs.shape[-1] == 42
    assert num_params == num_grad, 'Some parameters are missing gradients'
    assert not torch.isnan(outputs).any(), 'Output included NaNs'


@pytest.mark.timeout(120)
@pytest.mark.parametrize('model_name', list_models())
@pytest.mark.parametrize('batch_size', [1])
def test_model_default_cfgs(model_name, batch_size):
    """Run a single forward pass with each model"""
    model = create_model(model_name, pretrained=False)
    model.eval()
    state_dict = model.state_dict()
    cfg = model.default_cfg

    classifier = cfg['classifier']
    first_conv = cfg['first_conv']
    pool_size = cfg['pool_size']
    input_size = model.default_cfg['input_size']

    if all([x <= 448 for x in input_size]):
        # pool size only checked if default res <= 448 * 448 to keep resource down
        input_size = tuple([min(x, 448) for x in input_size])
        outputs = model.forward_features(torch.randn((batch_size, *input_size)))
        assert outputs.shape[-1] == pool_size[-1] and outputs.shape[-2] == pool_size[-2]
    assert any([k.startswith(cfg['classifier']) for k in state_dict.keys()]), f'{classifier} not in model params'
    assert any([k.startswith(cfg['first_conv']) for k in state_dict.keys()]), f'{first_conv} not in model params'
