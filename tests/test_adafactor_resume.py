from copy import deepcopy

import pytest
import torch

from timm.optim import Adafactor


class LegacyFactoredAdafactor(Adafactor):
    # v1.0.11 factored the last two dimensions of every matrix / higher-rank parameter.
    @staticmethod
    def _get_options(group, shape, min_size_to_factor=16):
        return ((len(shape) - 2, len(shape) - 1) if len(shape) >= 2 else None, group['beta1'] is not None)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('shape', [(3,), (2, 3), (32, 3), (32, 32), (3, 4, 5, 7)])
@pytest.mark.parametrize('beta1', [None, 0.9])
def test_adafactor_legacy_factorization_resume(dtype, shape, beta1):
    generator = torch.Generator().manual_seed(42)
    p = torch.nn.Parameter(torch.randn(shape, dtype=dtype, generator=generator))
    kwargs = dict(lr=0.001, betas=(beta1, 0.999) if beta1 is not None else None)
    reference = LegacyFactoredAdafactor([p], **kwargs)
    for step in range(3):
        reference.zero_grad()
        (p.square().sum() * (step + 1)).backward()
        reference.step()
    checkpoint = deepcopy(reference.state_dict())
    checkpoint['param_groups'][0].pop('min_dim_size_to_factor')
    checkpoint['param_groups'][0].pop('caution')
    q = torch.nn.Parameter(p.detach().clone())
    resumed = Adafactor([q], **kwargs)
    resumed.load_state_dict(checkpoint)
    for step in range(3, 6):
        for optimizer, parameter in ((reference, p), (resumed, q)):
            optimizer.zero_grad()
            (parameter.square().sum() * (step + 1)).backward()
            optimizer.step()
        torch.testing.assert_close(q, p)
        for key, value in reference.state[p].items():
            torch.testing.assert_close(resumed.state[q][key], value)
        # A checkpoint written after migration must retain the legacy factorization on the next resume.
        resumed.load_state_dict(deepcopy(resumed.state_dict()))
