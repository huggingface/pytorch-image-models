from copy import deepcopy

import pytest
import torch

from timm.optim import create_optimizer_v2


@pytest.mark.parametrize('no_prox', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('saved_foreach', ['legacy', False, True])
def test_adan_resumes_legacy_and_current_checkpoint(dtype, saved_foreach, no_prox):
    parameter = torch.nn.Parameter(torch.tensor([0.4, -0.8, 1.2], dtype=dtype))
    reference = create_optimizer_v2([parameter], opt='adan', lr=0.01, foreach=False, weight_decay=0.1, no_prox=no_prox)
    for step in range(3):
        reference.zero_grad()
        (parameter.square().sum() * (step + 1)).backward()
        reference.step()

    saved = deepcopy(reference.state_dict())
    group = saved['param_groups'][0]
    if saved_foreach == 'legacy':
        # timm v1.0.11 predates both the foreach and caution parameter-group keys.
        group.pop('foreach')
        group.pop('caution')
        for state in saved['state'].values():
            state['pre_grad'] = -state.pop('neg_pre_grad')
            # The old single-tensor implementation stored these moments under reversed names.
            state['exp_avg_sq'], state['exp_avg_diff'] = state['exp_avg_diff'], state['exp_avg_sq']
    else:
        group['foreach'] = saved_foreach
    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = create_optimizer_v2(
        [resumed_parameter], opt='adan', lr=0.01, foreach=False, weight_decay=0.1, no_prox=no_prox,
    )
    resumed.load_state_dict(saved)

    for step in range(3, 8):
        for optimizer, p in ((reference, parameter), (resumed, resumed_parameter)):
            optimizer.zero_grad()
            (p.square().sum() * (step + 1)).backward()
            optimizer.step()
        torch.testing.assert_close(resumed_parameter, parameter)
        for name, value in reference.state[parameter].items():
            torch.testing.assert_close(resumed.state[resumed_parameter][name], value)
    assert resumed.param_groups[0]['foreach'] == (None if saved_foreach == 'legacy' else saved_foreach)
