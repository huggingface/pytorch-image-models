import pytest
import torch

from timm import list_models, create_model


@pytest.mark.timeout(60)
@pytest.mark.parametrize('model_name', list_models())
@pytest.mark.parametrize('batch_size', [1])
def test_model_forward(model_name, batch_size):
  """Run a single forward pass with each model"""
  model = create_model(model_name, pretrained=False)
  model.eval()

  inputs = torch.randn((batch_size, *model.default_cfg['input_size']))
  outputs = model(inputs)

  assert outputs.shape[0] == batch_size
  assert not torch.isnan(outputs).any(), 'Output included NaNs'
