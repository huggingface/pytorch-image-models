"""Tests for pruned model adaptation (timm.models._prune)."""
import pytest
import torch

import timm
from timm.models import list_models


@pytest.mark.parametrize('model_name', list_models('*pruned*'))
def test_pruned_feature_info_matches_forward(model_name):
    """Pruning rebuilds layers with new widths; feature_info must report the pruned channel counts (#2370)."""
    model = timm.create_model(model_name, pretrained=False, features_only=True).eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, *model.pretrained_cfg['input_size']))
    assert [o.shape[1] for o in outputs] == model.feature_info.channels()
