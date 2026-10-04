""" Optimzier Tests

These tests were adapted from PyTorch' optimizer tests.

"""
import functools
import importlib
import inspect
import os
from copy import deepcopy

import pytest
import torch
from torch.nn import Parameter
from torch.testing._internal.common_utils import TestCase

from timm.optim import create_optimizer_v2, list_optimizers, get_optimizer_class, get_optimizer_info, OptimInfo
from timm.optim import param_groups_layer_decay, param_groups_weight_decay
from timm.scheduler import PlateauLRScheduler

torch_backend = os.environ.get('TORCH_BACKEND')
if torch_backend is not None:
    importlib.import_module(torch_backend)
torch_device = os.environ.get('TORCH_DEVICE', 'cuda')
torch_version = tuple(int(v) for v in torch.__version__.split('.')[:2])

# HACK relying on internal PyTorch test functionality for comparisons that I don't want to write
torch_tc = TestCase()


def _test_basic_cases_template(weight, bias, input, constructor, scheduler_constructors):
    weight = Parameter(weight)
    bias = Parameter(bias)
    input = Parameter(input)
    optimizer = constructor(weight, bias)
    schedulers = []
    for scheduler_constructor in scheduler_constructors:
        schedulers.append(scheduler_constructor(optimizer))

    # to check if the optimizer can be printed as a string
    optimizer.__repr__()

    def fn():
        optimizer.zero_grad()
        y = weight.mv(input)
        if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():
            y = y.cuda(bias.get_device())
        loss = (y + bias).pow(2).sum()
        loss.backward()
        return loss

    initial_value = fn().item()
    for _i in range(200):
        for scheduler in schedulers:
            if isinstance(scheduler, PlateauLRScheduler):
                val_loss = fn()
                scheduler.step(val_loss)
            else:
                scheduler.step()
        optimizer.step(fn)

    assert fn().item() < initial_value


def _test_state_dict(weight, bias, input, constructor):
    weight = Parameter(weight)
    bias = Parameter(bias)
    input = Parameter(input)

    def fn_base(optimizer, weight, bias):
        optimizer.zero_grad()
        i = input if (weight.device, weight.dtype) == (input.device, input.dtype) else input_device
        loss = (weight.mv(i) + bias).pow(2).sum()
        loss.backward()
        return loss

    optimizer = constructor(weight, bias)
    fn = functools.partial(fn_base, optimizer, weight, bias)

    # Prime the optimizer
    for _i in range(20):
        optimizer.step(fn)
    # Clone the weights and construct new optimizer for them
    with torch.no_grad():
        weight_c = Parameter(weight.clone().detach())
        bias_c = Parameter(bias.clone().detach())
    optimizer_c = constructor(weight_c, bias_c)
    fn_c = functools.partial(fn_base, optimizer_c, weight_c, bias_c)
    # Load state dict
    state_dict = deepcopy(optimizer.state_dict())
    state_dict_c = deepcopy(optimizer.state_dict())
    optimizer_c.load_state_dict(state_dict_c)

    # Run both optimizations in parallel
    for _i in range(20):
        optimizer.step(fn)
        optimizer_c.step(fn_c)
        torch_tc.assertEqual(weight, weight_c)
        torch_tc.assertEqual(bias, bias_c)
    # Make sure state dict is deterministic with equal but not identical parameters
    torch_tc.assertEqual(optimizer.state_dict(), optimizer_c.state_dict())
    # Make sure repeated parameters have identical representation in state dict
    optimizer_c.param_groups.extend(optimizer_c.param_groups)
    torch_tc.assertEqual(optimizer.state_dict()['param_groups'][-1], optimizer_c.state_dict()['param_groups'][-1])

    # validate deepcopy() copies all public attributes
    def getPublicAttr(obj):
        return set(k for k in obj.__dict__ if not k.startswith('_'))

    assert getPublicAttr(optimizer) == getPublicAttr(deepcopy(optimizer))

    # Caution masks are sensitive to rounding across devices / dtypes. Keep the same-device
    # state-dict checks above, but don't require matching trajectories after conversion.
    if any(group.get('caution', False) for group in optimizer.param_groups):
        return

    # Check that state dict can be loaded even when we cast parameters to a different type and move to a
    # different device (if available). Numerics diverge across devices / dtypes,
    # so verify the loaded state and that a few steps move in the same direction instead of exact results.
    device = torch_device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    dtype = torch.float64
    with torch.no_grad():
        input_device = Parameter(input.clone().detach().to(device, dtype))
        weight_device = Parameter(weight.clone().detach().to(device, dtype))
        bias_device = Parameter(bias.clone().detach().to(device, dtype))
    optimizer_device = constructor(weight_device, bias_device)
    fn_device = functools.partial(fn_base, optimizer_device, weight_device, bias_device)

    state_dict = deepcopy(optimizer.state_dict())
    state_dict_c = deepcopy(optimizer.state_dict())
    optimizer_device.load_state_dict(state_dict_c)

    # Make sure state dict wasn't modified
    torch_tc.assertEqual(state_dict, state_dict_c)

    # Loaded state should follow its param to the new device / dtype (unless kept in a specific dtype, e.g. bf16
    # momentum), with values intact. Scalar tensors (step counts, etc.) are bookkeeping and may stay on CPU.
    param_idx = {p: i for i, p in enumerate(p for g in optimizer_device.param_groups for p in g['params'])}
    for param, param_state in optimizer_device.state.items():
        for key, value in param_state.items():
            if not isinstance(value, torch.Tensor):
                continue
            source = state_dict['state'][param_idx[param]][key]
            if value.dim() and value.is_floating_point():
                assert value.device == param.device, key
                assert value.dtype in (param.dtype, source.dtype), key
            assert torch.equal(value.to('cpu', source.dtype), source), key

    params_device = [weight_device, bias_device]
    with torch.no_grad():
        start = [p.clone() for p in (weight, bias)]
        start_device = [p.clone() for p in params_device]
    for _i in range(5):
        optimizer.step(fn)
        optimizer_device.step(fn_device)
    with torch.no_grad():
        delta = torch.cat([(p - s).flatten() for p, s in zip((weight, bias), start)])
        delta_device = torch.cat([(p - s).flatten() for p, s in zip(params_device, start_device)])
        delta_device = delta_device.to('cpu', delta.dtype)
    assert torch.isfinite(delta_device).all()
    if delta.norm() < 1e-8:
        # reference is stalled, device copy should be too
        assert delta_device.norm() < 1e-6, f'device update {delta_device.norm():.2e} while reference stalled'
    else:
        cos_sim = torch.nn.functional.cosine_similarity(delta, delta_device, dim=0)
        assert cos_sim > 0.9, f'update direction diverged (cosine similarity {cos_sim:.3f})'


def _test_basic_cases(constructor, scheduler_constructors=None):
    if scheduler_constructors is None:
        scheduler_constructors = []
    _test_state_dict(
        torch.randn(10, 5),
        torch.randn(10),
        torch.randn(5),
        constructor
    )
    _test_basic_cases_template(
        torch.randn(10, 5),
        torch.randn(10),
        torch.randn(5),
        constructor,
        scheduler_constructors
    )
    # non-contiguous parameters
    _test_basic_cases_template(
        torch.randn(10, 5, 2)[..., 0],
        torch.randn(10, 2)[..., 0],
        torch.randn(5),
        constructor,
        scheduler_constructors
    )
    # CUDA
    if torch_device == 'cpu':
        return
    elif torch_device == 'cuda' and not torch.cuda.is_available():
        return

    _test_basic_cases_template(
        torch.randn(10, 5).to(torch_device),
        torch.randn(10).to(torch_device),
        torch.randn(5).to(torch_device),
        constructor,
        scheduler_constructors
    )


def _test_model(optimizer, params, device=torch.device('cpu'), after_step=0):
    weight = torch.tensor(
        [[-0.2109, -0.4976], [-0.1413, -0.3420], [-0.2524, 0.6976]],
        device=device, requires_grad=True)
    bias = torch.tensor([-0.1085, -0.2979, 0.6892], device=device, requires_grad=True)
    weight2 = torch.tensor([[-0.0508, -0.3941, -0.2843]], device=device, requires_grad=True)
    bias2 = torch.tensor([-0.0711], device=device, requires_grad=True)
    input = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], device=device).reshape(3, 2)

    model = torch.nn.Sequential(torch.nn.Linear(2, 3),
                                torch.nn.Sigmoid(),
                                torch.nn.Linear(3, 1),
                                torch.nn.Sigmoid())
    model.to(device)

    pretrained_dict = model.state_dict()
    pretrained_dict['0.weight'] = weight
    pretrained_dict['0.bias'] = bias
    pretrained_dict['2.weight'] = weight2
    pretrained_dict['2.bias'] = bias2
    model.load_state_dict(pretrained_dict)

    optimizer = create_optimizer_v2(model, opt=optimizer, **params)

    prev_loss = float('inf')
    for i in range(20):
        optimizer.zero_grad()
        output = model(input)
        loss = output.sum()
        loss.backward()
        loss = loss.item()
        if i > after_step:
            assert loss < prev_loss
        prev_loss = loss
        optimizer.step()


def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def drosenbrock(tensor):
    x, y = tensor
    return torch.tensor((-400 * x * (y - x ** 2) - 2 * (1 - x), 200 * (y - x ** 2)))


def _test_rosenbrock(constructor, scheduler_constructors=None):
    if scheduler_constructors is None:
        scheduler_constructors = []
    params_t = torch.tensor([1.5, 1.5])

    params = Parameter(params_t)
    optimizer = constructor([params])
    schedulers = []
    for scheduler_constructor in scheduler_constructors:
        schedulers.append(scheduler_constructor(optimizer))

    solution = torch.tensor([1, 1])
    initial_dist = params.clone().detach().dist(solution)


    def get_grad(_param, _sparse_grad, _w):
        grad = drosenbrock(params.clone().detach())
        # Depending on w, provide only the x or y gradient
        if _sparse_grad:
            if _w:
                i = torch.tensor([[0, 0]], dtype=torch.int64)
                x = grad[0]
                v = torch.tensor([x / 4.0, x - x / 4.0])
            else:
                i = torch.tensor([[1, 1]], dtype=torch.int64)
                y = grad[1]
                v = torch.tensor([y - y / 4.0, y / 4.0])
            grad_out = torch.sparse_coo_tensor(i, v, (2,), dtype=v.dtype)
        else:
            if _w:
                grad_out = torch.tensor([grad[0], 0], dtype=_param.dtype)
            else:
                grad_out = torch.tensor([0, grad[1]], dtype=_param.dtype)
        return grad_out


    def eval(_param, _sparse_grad, _w):
        # Depending on w, provide only the x or y gradient
        optimizer.zero_grad()
        loss = rosenbrock(_param)
        loss.backward()

        grad_out = get_grad(_param, _sparse_grad, _w)
        with torch.no_grad():
            _param.grad = grad_out.to_dense()

        return loss

    for i in range(2000):
        # Do cyclic coordinate descent
        w = i % 2
        optimizer.step(functools.partial(eval, params, True, w))
        for scheduler in schedulers:
            if isinstance(scheduler, PlateauLRScheduler):
                scheduler.step(rosenbrock(params))
            else:
                scheduler.step()

    torch_tc.assertLessEqual(params.clone().detach().dist(solution), initial_dist)


def _build_params_dict(weight, bias, **kwargs):
    return [{'params': [weight]}, dict(params=[bias], **kwargs)]


def _build_params_dict_single(weight, bias, **kwargs):
    return [dict(params=bias, **kwargs)]


@pytest.mark.parametrize('optimizer', list_optimizers(exclude_filters=('fused*', 'bnb*', 'kron*')))
def test_optim_factory(optimizer):
    assert issubclass(get_optimizer_class(optimizer, bind_defaults=False), torch.optim.Optimizer)

    opt_info = get_optimizer_info(optimizer)
    assert isinstance(opt_info, OptimInfo)

    lr = (1e-2,) * 4
    if optimizer in ('mars', 'nadam', 'claprop', 'crmsproptf', 'cadafactorbv', 'csgdw', 'csgdc', 'csgdp', 'clamb'):
        lr = (1e-3,) * 4
    elif optimizer in ('cmars',):
        lr = (1e-4,) * 4

    try:
        if not opt_info.second_order:  # basic tests don't support second order right now
            # test basic cases that don't need specific tuning via factory test
            _test_basic_cases(
                lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=lr[0])
            )
            _test_basic_cases(
                lambda weight, bias: create_optimizer_v2(
                    _build_params_dict(weight, bias, lr=lr[1]),
                    optimizer,
                    lr=lr[1] / 10)
            )
            _test_basic_cases(
                lambda weight, bias: create_optimizer_v2(
                    _build_params_dict_single(weight, bias, lr=lr[2]),
                    optimizer,
                    lr=lr[2] / 10)
            )
            _test_basic_cases(
                lambda weight, bias: create_optimizer_v2(
                    _build_params_dict_single(weight, bias, lr=lr[3]),
                    optimizer)
            )
    except TypeError as e:
        if 'radamw' in optimizer:
            pytest.skip("Expected for 'radamw' (decoupled decay) to fail in older PyTorch versions.")
        else:
            raise e



#@pytest.mark.parametrize('optimizer', ['sgd', 'momentum'])
# FIXME momentum variant frequently fails in GitHub runner, but never local after many attempts
@pytest.mark.parametrize('optimizer', ['sgd'])
def test_sgd(optimizer):
    # _test_basic_cases(
    #     lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3),
    #     [lambda opt: StepLR(opt, gamma=0.9, step_size=10)]
    # )
    # _test_basic_cases(
    #     lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3),
    #     [lambda opt: WarmUpLR(opt, warmup_factor=0.4, warmup_iters=4, warmup_method="linear")]
    # )
    # _test_basic_cases(
    #     lambda weight, bias: optimizer([weight, bias], lr=1e-3),
    #     [lambda opt: WarmUpLR(opt, warmup_factor=0.4, warmup_iters=4, warmup_method="constant")]
    # )
    # _test_basic_cases(
    #     lambda weight, bias: optimizer([weight, bias], lr=1e-3),
    #     [lambda opt: StepLR(opt, gamma=0.9, step_size=10),
    #      lambda opt: WarmUpLR(opt, warmup_factor=0.4, warmup_iters=4)]
    # )
    # _test_basic_cases(
    #     lambda weight, bias: optimizer([weight, bias], lr=1e-3),
    #     [lambda opt: StepLR(opt, gamma=0.9, step_size=10),
    #      lambda opt: ReduceLROnPlateau(opt)]
    # )
    # _test_basic_cases(
    #     lambda weight, bias: optimizer([weight, bias], lr=1e-3),
    #     [lambda opt: StepLR(opt, gamma=0.99, step_size=10),
    #      lambda opt: ExponentialLR(opt, gamma=0.99),
    #      lambda opt: ReduceLROnPlateau(opt)]
    # )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=3e-3, momentum=1)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=3e-3, momentum=1, weight_decay=.1)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )
    _test_model(optimizer, dict(lr=1e-3))


@pytest.mark.parametrize('optimizer',  ['adamw', 'adam', 'nadam', 'adamax', 'nadamw', 'adamwlegacy', 'adamc'])
def test_adam(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=5e-2)
    )
    _test_model(optimizer, dict(lr=5e-2))


@pytest.mark.skipif(not hasattr(torch, 'compile'), reason='requires torch.compile')
@pytest.mark.parametrize('optimizer_name', ['adamwlegacy', 'nadamw'])
@pytest.mark.parametrize('foreach', [None, False, True])
@pytest.mark.parametrize('dtype,lr,eps', [(torch.float32, 1e-3, 1e-8), (torch.float16, 1e-5, 1e-4)])
@pytest.mark.parametrize('tensor_lr', [False, True])
def test_compiled_foreach_adam_optimizers(optimizer_name, foreach, dtype, lr, eps, tensor_lr):
    from timm.optim.adamw import AdamWLegacy
    from timm.optim.nadamw import NAdamW

    optimizer_cls = AdamWLegacy if optimizer_name == 'adamwlegacy' else NAdamW
    lr = torch.tensor(lr) if tensor_lr else lr
    param = Parameter(torch.full((4,), 0.01, dtype=dtype))
    eager_param = Parameter(param.detach().clone())
    reference_param = Parameter(param.detach().clone())
    optimizer = optimizer_cls([param], lr=lr, eps=eps, foreach=foreach)
    eager_optimizer = optimizer_cls([eager_param], lr=lr, eps=eps, foreach=foreach)
    reference_optimizer = optimizer_cls([reference_param], lr=lr, eps=eps, foreach=False)
    compiled_step = torch.compile(optimizer.step, backend='eager')

    before = param.detach().clone()
    # Small positive LR must keep updating FP16 parameters as bias correction approaches one.
    for _ in range(30):
        param.grad = torch.ones_like(param)
        eager_param.grad = torch.ones_like(eager_param)
        reference_param.grad = torch.ones_like(reference_param)
        compiled_step()
        eager_optimizer.step()
        reference_optimizer.step()

    assert torch.isfinite(param).all()
    assert not torch.equal(param, before)
    torch.testing.assert_close(param, reference_param)
    torch.testing.assert_close(eager_param, reference_param)
    assert optimizer.state[param]['step'] == 30


@pytest.mark.parametrize('optimizer',  ['kron'])
def test_kron(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )
    _test_model(optimizer, dict(lr=1e-3))


@pytest.mark.parametrize('saved_corrected_weight_decay', [None, False, True])
def test_kron_load_state_dict_corrected_weight_decay(saved_corrected_weight_decay):
    # Loading must clear cached expressions and back-fill missing defaults without overwriting saved values.
    from timm.optim.kron import Kron

    def make():
        weight = Parameter(torch.ones(4, 3))
        bias = Parameter(torch.ones(4))
        return Kron(
            [{'params': [weight]}, {'params': [bias]}], lr=1e-3, weight_decay=0.1,
            decoupled_decay=True, corrected_weight_decay=True, deterministic=True)

    def step(optimizer):
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = torch.ones_like(p)
        optimizer.step()

    optimizer = make()
    step(optimizer)
    assert optimizer._param_exprs

    state_dict = deepcopy(optimizer.state_dict())
    for group in state_dict['param_groups']:
        if saved_corrected_weight_decay is None:
            del group['corrected_weight_decay']
        else:
            group['corrected_weight_decay'] = saved_corrected_weight_decay

    optimizer.load_state_dict(deepcopy(state_dict))
    assert optimizer._param_exprs == {}

    resumed = make()
    resumed.load_state_dict(deepcopy(state_dict))
    for source_group, resumed_group in zip(optimizer.param_groups, resumed.param_groups):
        for source, dest in zip(source_group['params'], resumed_group['params']):
            with torch.no_grad():
                dest.copy_(source)

    expected = saved_corrected_weight_decay if saved_corrected_weight_decay is not None else False
    for opt in (optimizer, resumed):
        for group in opt.param_groups:
            assert group['corrected_weight_decay'] is expected
            group['lr'] *= 0.5  # Exercise corrected decay at an LR different from the initial value.
        step(opt)
        assert opt._param_exprs
    for source_group, resumed_group in zip(optimizer.param_groups, resumed.param_groups):
        for source, dest in zip(source_group['params'], resumed_group['params']):
            assert torch.isfinite(dest).all()
            torch.testing.assert_close(source, dest)


@pytest.mark.parametrize('optimizer',  ['muon', 'nmuon'])
def test_muon(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-2)
    )
    _test_model(optimizer, dict(lr=1e-2))


@pytest.mark.skipif(not hasattr(torch, 'compile'), reason='requires torch.compile')
@pytest.mark.parametrize('nesterov', [False, True])
def test_compiled_muon_fallback(nesterov):
    from timm.optim.muon import Muon

    # A 1-D parameter is routed through Muon's AdamW/NAdamW fallback.
    param = Parameter(torch.ones(4))
    reference_param = Parameter(param.detach().clone())
    optimizer = Muon([param], lr=1e-3, nesterov=nesterov)
    reference_optimizer = Muon([reference_param], lr=1e-3, nesterov=nesterov)
    compiled_step = torch.compile(optimizer.step, backend='eager')

    before = param.detach().clone()
    for _ in range(2):
        param.grad = torch.ones_like(param)
        reference_param.grad = torch.ones_like(reference_param)
        compiled_step()
        reference_optimizer.step()

    assert torch.isfinite(param).all()
    assert not torch.equal(param, before)
    torch.testing.assert_close(param, reference_param)
    assert optimizer.state[param]['step'] == 2


@pytest.mark.parametrize('optimizer',  ['adamuon', 'nadamuon'])
def test_adamuon(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-2)
    )
    _test_model(optimizer, dict(lr=1e-2))


@pytest.mark.parametrize('optimizer',  ['adopt', 'adoptw'])
def test_adopt(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=3e-3)
    )
    _test_model(optimizer, dict(lr=5e-2), after_step=1)  # note no convergence in first step for ADOPT


@pytest.mark.parametrize('optimizer',  ['adan', 'adanw'])
def test_adan(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )
    _test_model(optimizer, dict(lr=5e-2), after_step=1)  # note no convergence in first step for ADOPT


@pytest.mark.parametrize('optimizer',  ['adabelief'])
def test_adabelief(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3, weight_decay=1)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=5e-2)
    )
    _test_model(optimizer, dict(lr=5e-2))


@pytest.mark.parametrize('optimizer',  ['radam', 'radabelief'])
def test_rectified(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )
    _test_model(optimizer, dict(lr=1e-3))


@pytest.mark.parametrize('optimizer',   ['adadelta', 'adagrad'])
def test_adaother(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3, weight_decay=1)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-1)
    )
    _test_model(optimizer, dict(lr=5e-2))


@pytest.mark.parametrize('optimizer',   ['adafactor', 'adafactorbv'])
def test_adafactor(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3, weight_decay=1)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=5e-2)
    )
    _test_model(optimizer, dict(lr=5e-2))


@pytest.mark.parametrize('optimizer',  ['lamb', 'lambc'])
def test_lamb(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )
    _test_model(optimizer, dict(lr=1e-3))


@pytest.mark.parametrize('optimizer', ['laprop'])
def test_laprop(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-2)
    )
    _test_model(optimizer, dict(lr=1e-2))


@pytest.mark.parametrize('optimizer',  ['lars', 'larc', 'nlars', 'nlarc'])
def test_lars(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )
    _test_model(optimizer, dict(lr=1e-3))


@pytest.mark.parametrize('optimizer',  ['madgrad', 'madgradw'])
def test_madgrad(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-2)
    )
    _test_model(optimizer, dict(lr=1e-2))


@pytest.mark.parametrize('optimizer',  ['mars'])
def test_mars(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )
    _test_model(optimizer, dict(lr=5e-2), after_step=1)  # note no convergence in first step for ADOPT


@pytest.mark.parametrize('optimizer',  ['novograd'])
def test_novograd(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )
    _test_model(optimizer, dict(lr=1e-3))


@pytest.mark.parametrize('optimizer', ['rmsprop', 'rmsproptf'])
def test_rmsprop(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-2)
    )
    _test_model(optimizer, dict(lr=1e-2))


@pytest.mark.parametrize('optimizer', ['adamp'])
def test_adamp(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=5e-2)
    )
    _test_model(optimizer, dict(lr=5e-2))


@pytest.mark.parametrize('optimizer', ['sgdp'])
def test_sgdp(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )
    _test_model(optimizer, dict(lr=1e-3))


@pytest.mark.parametrize('optimizer', ['lookahead_sgd', 'lookahead_momentum'])
def test_lookahead_sgd(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )


@pytest.mark.parametrize('optimizer', ['lookahead_adamw', 'lookahead_adam'])
def test_lookahead_adam(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=5e-2)
    )


@pytest.mark.parametrize('optimizer', ['lookahead_radam'])
def test_lookahead_radam(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-4)
    )


def test_param_groups_layer_decay_with_min():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2)
    )
    
    param_groups = param_groups_layer_decay(
        model,
        weight_decay=0.05,
        layer_decay=0.75,
        min_scale=0.5,
        verbose=True
    )
    
    assert len(param_groups) > 0
    # Verify layer scaling is applied with a min scale
    for group in param_groups:
        assert 'lr_scale' in group
        assert group['lr_scale'] <= 1.0
        assert group['lr_scale'] >= 0.5


def test_registry_create_optimizer_layer_decay_default_min_scale():
    # The registry create_optimizer must handle layer_decay when the min scale is left at its
    # default: it used to default layer_decay_min_scale to None, which crashed in
    # param_groups_layer_decay's max(min_scale, ...). create_optimizer_v2 passed 0.0 and was fine.
    from timm.optim._optim_factory import default_registry

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )
    optimizer = default_registry.create_optimizer(
        model, 'adamw', lr=1e-3, weight_decay=0.05, layer_decay=0.75,
    )
    assert len(optimizer.param_groups) > 0
    for group in optimizer.param_groups:
        assert 'lr_scale' in group
        assert 0.0 <= group['lr_scale'] <= 1.0


def test_param_groups_layer_decay_with_matcher():
    class ModelWithMatcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(10, 5)
            self.layer2 = torch.nn.Linear(5, 2)
            
        def group_matcher(self, coarse=False):
            return lambda name: int(name.split('.')[0][-1])
            
    model = ModelWithMatcher()
    param_groups = param_groups_layer_decay(
        model,
        weight_decay=0.05,
        layer_decay=0.75,
        verbose=True
    )
    
    assert len(param_groups) > 0
    # Verify layer scaling is applied
    for group in param_groups:
        assert 'lr_scale' in group
        assert 'weight_decay' in group
        assert len(group['params']) > 0


@pytest.mark.parametrize('model_name', ['eva', 'vit'])
def test_param_groups_layer_decay_reg_token_stem(model_name):
    from timm.models import group_parameters
    from timm.models.eva import Eva
    from timm.models.vision_transformer import VisionTransformer

    if model_name == 'eva':
        model = Eva(
            img_size=16,
            patch_size=8,
            embed_dim=8,
            depth=1,
            num_heads=2,
            num_reg_tokens=1,
            num_classes=0,
        )
    else:
        model = VisionTransformer(
            img_size=16,
            patch_size=8,
            embed_dim=8,
            depth=1,
            num_heads=2,
            reg_tokens=1,
            num_classes=0,
        )

    no_weight_decay = model.no_weight_decay()
    assert 'reg_token' in no_weight_decay

    layer_map = group_parameters(model, model.group_matcher(coarse=False), reverse=True)
    assert layer_map['reg_token'] == layer_map['cls_token'] == layer_map['pos_embed']

    param_groups = param_groups_layer_decay(
        model,
        weight_decay=0.05,
        no_weight_decay_list=no_weight_decay,
        layer_decay=0.75,
    )
    param_to_group = {
        id(param): group
        for group in param_groups
        for param in group['params']
    }

    cls_group = param_to_group[id(model.cls_token)]
    reg_group = param_to_group[id(model.reg_token)]
    assert reg_group['lr_scale'] == cls_group['lr_scale']
    assert reg_group['weight_decay'] == 0.


def test_param_groups_weight_decay():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2)
    )
    weight_decay = 0.01
    no_weight_decay_list = ['1.weight']
    
    param_groups = param_groups_weight_decay(
        model, 
        weight_decay=weight_decay,
        no_weight_decay_list=no_weight_decay_list
    )
    
    assert len(param_groups) == 2
    assert param_groups[0]['weight_decay'] == 0.0
    assert param_groups[1]['weight_decay'] == weight_decay
    
    # Verify parameters are correctly grouped
    no_decay_params = set(param_groups[0]['params'])
    decay_params = set(param_groups[1]['params'])
    
    for name, param in model.named_parameters():
        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            assert param in no_decay_params
        else:
            assert param in decay_params

@pytest.mark.parametrize('optimizer', ['cadamp'])
def test_cadamp(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=5e-2)
    )
    _test_model(optimizer, dict(lr=5e-2))

@pytest.mark.parametrize('optimizer', ['csgdp'])
def test_csgdp(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=5e-4)
    )
    _test_model(optimizer, dict(lr=5e-4))

@pytest.mark.parametrize('optimizer', ['csgdw'])
def test_csgdw(optimizer):
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=5e-4)
    )
    _test_model(optimizer, dict(lr=5e-4))


@pytest.mark.parametrize('shape', [(30, 20), (20, 30)])
def test_adafactor_bv_factored_row_normalization(shape):
    # AdafactorBigVision factorizes the second moment for 2D params, so the update must be
    # transpose-equivariant: stepping W and W.T with transposed grads must give transposed updates.
    # A wrong axis in the row-factor reduction silently collapsed row_factor to 1 (dropping the row
    # normalization) for matrices whose larger dim comes first, breaking that invariance.
    from timm.optim.adafactor_bv import AdafactorBigVision

    generator = torch.Generator().manual_seed(0)
    grad = torch.randn(shape, dtype=torch.double, generator=generator)

    def one_step(g):
        param = Parameter(torch.zeros(g.shape, dtype=torch.double))
        opt = AdafactorBigVision([param], lr=1.0, momentum=None, eps=1e-30, weight_decay=0.0)
        param.grad = g.clone()
        opt.step()
        return param.detach().clone()

    param = one_step(grad)
    param_t = one_step(grad.t().contiguous())
    torch.testing.assert_close(param_t, param.t())


@pytest.mark.parametrize('clipping_threshold', [0.5, 2.0])
@pytest.mark.parametrize('shape', [(4,), (16, 32), (32, 16)])
def test_adafactor_bv_clipping_threshold(clipping_threshold, shape):
    # Compare to unclipped updates above/below the threshold, including zero gradients and both
    # factored axis orders. Small updates must not grow, and zero updates must stay finite.
    from timm.optim.adafactor_bv import AdafactorBigVision

    generator = torch.Generator().manual_seed(0)
    grad = torch.randn(shape, dtype=torch.double, generator=generator)
    grads = [torch.zeros_like(grad), grad, 10 * grad, 0.1 * grad, torch.zeros_like(grad)]

    def make(threshold):
        param = Parameter(torch.zeros(shape, dtype=torch.double))
        opt = AdafactorBigVision(
            [param], lr=1.0, momentum=None, eps=1e-30, weight_decay=0.0, clipping_threshold=threshold)
        return param, opt

    param_ref, opt_ref = make(None)
    param_clip, opt_clip = make(clipping_threshold)
    for grad in grads:
        before_ref = param_ref.detach().clone()
        before_clip = param_clip.detach().clone()
        param_ref.grad = grad.clone()
        param_clip.grad = grad.clone()
        opt_ref.step()
        opt_clip.step()
        update_ref = before_ref - param_ref.detach()
        update_clip = before_clip - param_clip.detach()
        rms_ref = (update_ref.norm(2) / update_ref.numel() ** 0.5).item()
        torch.testing.assert_close(update_clip, update_ref / max(1.0, rms_ref / clipping_threshold))
        rms_clip = (update_clip.norm(2) / update_clip.numel() ** 0.5).item()
        assert rms_clip <= min(rms_ref, clipping_threshold) + 1e-9


def test_sgdw_multi_tensor_weight_decay_matches_single_tensor():
    # The foreach path groups params by (device, dtype) and iterates the groups. Decoupled weight
    # decay must be applied to each group's params, otherwise params spanning more than one partition
    # (e.g. mixed dtypes) are decayed once per partition instead of once. Compare foreach against the
    # single-tensor reference with two dtypes so there are two partitions.
    from timm.optim.sgdw import SGDW

    def run(foreach):
        p32 = Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
        p64 = Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
        for p in (p32, p64):
            p.grad = torch.ones_like(p)
        SGDW([p32, p64], lr=0.1, momentum=0.0, weight_decay=0.5, foreach=foreach).step()
        return p32.detach().clone(), p64.detach().clone()

    multi32, multi64 = run(True)
    single32, single64 = run(False)
    torch.testing.assert_close(multi32, single32)
    torch.testing.assert_close(multi64, single64)


@pytest.mark.parametrize('lag_index', [0, 1])
@pytest.mark.parametrize('clip_exp', [None, 0.333])
def test_adopt_multi_tensor_lagging_step_matches_single_tensor(lag_index, clip_exp):
    # A param without a gradient in some steps (frozen for a while, an expert that got no tokens) lags behind
    # the other params of its group in state['step']. The foreach path took the first-step initialization and
    # the clip value from device_state_steps[0] for the whole group, so a lagging param never got its
    # exp_avg_sq initialized (or the whole group was re-initialized and skipped an update when the lagging
    # param came first) and was clipped with the wrong step. Compare against the single-tensor reference.
    from timm.optim.adopt import Adopt

    grads = [torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(step)) for step in range(12)]

    def run(foreach):
        params = [Parameter(torch.full((5, 4), 1.0 + i)) for i in range(2)]
        optimizer = Adopt(params, lr=1e-3, clip_exp=clip_exp, foreach=foreach)
        for step, step_grads in enumerate(grads):
            for i, p in enumerate(params):
                p.grad = None if (i == lag_index and step < 4) else step_grads[i].clone()
            optimizer.step()
        return params, [optimizer.state[p] for p in params]

    multi_params, multi_state = run(True)
    single_params, single_state = run(False)
    for multi, single in zip(multi_params, single_params):
        torch.testing.assert_close(multi, single)
    for multi, single in zip(multi_state, single_state):
        assert multi['step'] == single['step']
        torch.testing.assert_close(multi['exp_avg'], single['exp_avg'])
        torch.testing.assert_close(multi['exp_avg_sq'], single['exp_avg_sq'])


def test_mars_last_grad_is_copied_not_aliased():
    # Mars keeps the previous gradient in state['last_grad'] for its variance reduction term. It used
    # to store a reference to p.grad instead of a copy. With zero_grad(set_to_none=False) the gradient
    # is zeroed in place and the next backward accumulates into the same tensor, so the stored
    # 'previous' gradient was always equal to the current one and the correction term was silently
    # zero. set_to_none=True allocates a fresh p.grad every step and was never affected, so the two
    # must give the same trajectory for the same sequence of gradients.
    from timm.optim.mars import Mars

    def run(set_to_none):
        generator = torch.Generator().manual_seed(0)
        param = Parameter(torch.ones(4, 3))
        optimizer = Mars([param], lr=1e-2)
        for _ in range(4):
            direction = torch.randn(param.shape, generator=generator)
            optimizer.zero_grad(set_to_none=set_to_none)
            (param * direction).sum().backward()
            optimizer.step()
        return param.detach().clone(), optimizer.state[param]['last_grad'], param.grad

    param_none, _, _ = run(True)
    param_zero, last_grad, grad = run(False)
    torch.testing.assert_close(param_zero, param_none)
    assert last_grad is not grad
    torch.testing.assert_close(last_grad, grad)


@pytest.mark.parametrize('adjust_lr_fn', ['match_rms_adamw', 'rms_to_rms', 'rsqrt_in', None])
@pytest.mark.parametrize('shape', [(8, 4, 3), (8, 4, 3, 3), (8, 4, 2, 3, 3)])
@pytest.mark.parametrize('nesterov', [False, True])
def test_adamuon_conv_flatten_matches_2d(adjust_lr_fn, shape, nesterov):
    # Flattened convolutions must follow the same trajectory as the equivalent matrix, including
    # second moments and decay. LR scaling must use matrix dimensions instead of kernel dimensions.
    from timm.optim.muon import Muon

    generator = torch.Generator().manual_seed(0)
    param_conv = Parameter(torch.randn(shape, generator=generator))
    param_2d = Parameter(param_conv.detach().reshape(shape[0], -1).clone())
    kwargs = dict(lr=0.1, weight_decay=0.1, algo='adamuon', adjust_lr_fn=adjust_lr_fn, nesterov=nesterov)
    opt_conv = Muon([param_conv], **kwargs)
    opt_2d = Muon([param_2d], **kwargs)
    for _ in range(3):
        grad = torch.randn(shape, generator=generator)
        param_conv.grad = grad.clone()
        param_2d.grad = grad.reshape_as(param_2d).clone()
        opt_conv.step()
        opt_2d.step()
        assert opt_conv.state[param_conv]['use_muon']
        torch.testing.assert_close(param_conv.reshape_as(param_2d), param_2d)
        torch.testing.assert_close(
            opt_conv.state[param_conv]['exp_avg_sq'].reshape_as(param_2d),
            opt_2d.state[param_2d]['exp_avg_sq'])


@pytest.mark.parametrize('adjust_lr_fn', ['match_rms_adamw', 'rms_to_rms', 'rsqrt_in', None])
@pytest.mark.parametrize('normalize_spatial', [False, True])
def test_adamuon_conv_batched_matches_repeated_2d(adjust_lr_fn, normalize_spatial):
    # Repeating a matrix across spatial positions must preserve its update, apart from the
    # optional spatial normalization. This also checks the non-RMS scaling modes in batched mode.
    from timm.optim.muon import Muon

    if torch_version < (2, 1):
        pytest.skip('Older CPU batched bfloat16 kernels have different rounding and a beta=0 NaN bug (pytorch#96086)')

    generator = torch.Generator().manual_seed(0)
    param_2d = Parameter(torch.zeros(8, 4))
    param_conv = Parameter(torch.zeros(8, 4, 2, 3))
    kwargs = dict(weight_decay=0.0, algo='adamuon', adjust_lr_fn=adjust_lr_fn)
    spatial_scale = 6 ** -0.5 if normalize_spatial else 1.0
    opt_2d = Muon([param_2d], lr=0.1 * spatial_scale, **kwargs)
    opt_conv = Muon([param_conv], lr=0.1, conv_mode='batched', normalize_spatial=normalize_spatial, **kwargs)
    for _ in range(3):
        grad = torch.randn(8, 4, generator=generator)
        param_2d.grad = grad.clone()
        param_conv.grad = grad[:, :, None, None].expand_as(param_conv).clone()
        opt_2d.step()
        opt_conv.step()
        assert opt_conv.state[param_conv]['use_muon']
        expected = param_2d[:, :, None, None].expand_as(param_conv)
        torch.testing.assert_close(param_conv, expected)


@pytest.mark.parametrize('shape', [(8, 36), (8, 4, 3), (8, 4, 3, 3), (8, 4, 2, 3, 3)])
@pytest.mark.parametrize('conv_mode,normalize_spatial', [('flatten', True), ('batched', False), ('batched', True)])
def test_adamuon_update_rms(shape, conv_mode, normalize_spatial):
    # RMS alignment uses the whole tensor, with optional 1/sqrt(spatial_size) scaling in batched mode.
    from timm.optim.muon import Muon

    if conv_mode == 'batched' and len(shape) > 2 and torch_version < (2, 1):
        pytest.skip('Older CPU baddbmm can propagate uninitialized NaNs with beta=0 (pytorch#96086)')

    generator = torch.Generator().manual_seed(0)
    param = Parameter(torch.zeros(shape))
    opt = Muon([param], lr=0.1, weight_decay=0.0, algo='adamuon',
               conv_mode=conv_mode, normalize_spatial=normalize_spatial)
    expected_rms = 0.2 * 0.1
    if conv_mode == 'batched' and normalize_spatial:
        spatial_size = param.numel() // (shape[0] * shape[1])
        expected_rms /= spatial_size ** 0.5
    for _ in range(3):
        before = param.detach().clone()
        param.grad = torch.randn(shape, generator=generator)
        opt.step()
        assert opt.state[param]['use_muon']
        rms = (before - param.detach()).square().mean().sqrt()
        torch.testing.assert_close(rms, torch.tensor(expected_rms), rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='capturable requires CUDA')
@pytest.mark.parametrize('optimizer', ['adamwlegacy', 'nadamw'])
@pytest.mark.parametrize('foreach', [False, True])
@pytest.mark.parametrize('dtype,eps', [(torch.float32, 1e-8), (torch.bfloat16, 1e-8), (torch.float16, 1e-4)])
def test_capturable_zero_lr(optimizer, foreach, dtype, eps):
    # The capturable paths divided eps by the step size before adding it, 0 / 0 = NaN when lr == 0 (e.g. warmup
    # from 0 or cooldown to 0 w/ an in-place updated tensor lr). Adding eps first, as the non-capturable paths do,
    # gives a zero update at lr == 0 and matches the non-capturable result otherwise.
    def run(lrs, capturable):
        param = Parameter(torch.ones(8, 8, device='cuda', dtype=dtype))
        lr = torch.tensor(lrs[0], device='cuda') if capturable else lrs[0]
        opt = create_optimizer_v2(
            [param], optimizer, lr=lr, eps=eps, weight_decay=0.05, capturable=capturable, foreach=foreach)
        for i, step_lr in enumerate(lrs):
            if capturable:
                lr.fill_(step_lr)
            else:
                opt.param_groups[0]['lr'] = step_lr
            grad = torch.randn(8, 8, device='cuda', dtype=dtype, generator=torch.Generator('cuda').manual_seed(i))
            grad[0] = 0
            param.grad = grad
            before = param.detach().clone()
            opt.step()
            if step_lr == 0:
                torch.testing.assert_close(param.detach(), before, rtol=0, atol=0)
        return param.detach()

    run([0., 0., 1e-3, 1e-3, 0., 0.], capturable=True)
    torch.testing.assert_close(run([1e-3] * 4, capturable=True), run([1e-3] * 4, capturable=False))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='capturable requires CUDA')
@pytest.mark.parametrize('optimizer', ['adamwlegacy', 'nadamw'])
@pytest.mark.parametrize('foreach', [False, True])
def test_capturable_fp16_small_lr(optimizer, foreach):
    # Folding a small step size into an FP16 denominator overflowed to inf, so the capturable paths stopped updating
    # FP16 params as bias correction approached one. Compare against the non-capturable path.
    def run(capturable):
        param = Parameter(torch.full((4,), 0.01, device='cuda', dtype=torch.float16))
        lr = torch.tensor(1e-5, device='cuda') if capturable else 1e-5
        opt = create_optimizer_v2([param], optimizer, lr=lr, eps=1e-4, capturable=capturable, foreach=foreach)
        for _ in range(30):
            param.grad = torch.ones_like(param)
            before = param.detach().clone()
            opt.step()
        assert not torch.equal(param.detach(), before)
        return param.detach()

    torch.testing.assert_close(run(True), run(False))


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('precond_dtype', [None, torch.float32])
@pytest.mark.parametrize('momentum_into_precond_update', [True, False])
def test_kron_low_precision_params(dtype, precond_dtype, momentum_into_precond_update):
    # The random probe for the preconditioner update used precond_dtype directly, None gave a float32
    # probe that failed to matmul with low precision Q.
    from timm.optim.kron import Kron
    generator = torch.Generator().manual_seed(0)
    param = Parameter(torch.randn(16, 8, generator=generator).to(dtype))
    opt = Kron(
        [param],
        lr=1e-3,
        precond_dtype=precond_dtype,
        momentum_into_precond_update=momentum_into_precond_update,
    )
    for _ in range(3):
        param.grad = torch.randn(16, 8, generator=generator).to(dtype)
        opt.step()
    assert param.dtype == dtype
    assert torch.isfinite(param).all()


@pytest.mark.parametrize('optimizer', list_optimizers(exclude_filters=('fused*', 'bnb*')))
def test_optim_factory_common_kwargs(optimizer):
    # eps / betas / momentum passed to the factory must be forwarded to optimizers that accept them,
    # and dropped (not crash) for those that don't.
    info = get_optimizer_info(optimizer)
    opt_args = inspect.signature(get_optimizer_class(optimizer, bind_defaults=False).__init__).parameters
    assert info.has_eps == ('eps' in opt_args)
    assert info.has_betas == ('betas' in opt_args)
    assert info.has_momentum == ('momentum' in opt_args)
    betas = (0.5, 0.6, 0.7)[:info.num_betas]
    opt = create_optimizer_v2([Parameter(torch.ones(4, 4))], optimizer, lr=1e-3, eps=1e-6, betas=betas, momentum=0.5)
    group = opt.param_groups[0]
    if info.has_betas:
        if 'betas' in group:
            assert tuple(group['betas']) == betas
        else:
            assert group['beta1'] == betas[0]  # Adafactor
    if info.has_momentum and 'momentum' in group:
        assert group['momentum'] == 0.5


def test_cadafactor_has_first_moment():
    # Caution is applied to the first moment in Adafactor, cadafactor enables it by default.
    def run(optimizer):
        generator = torch.Generator().manual_seed(0)
        param = Parameter(torch.ones(32, 32))
        opt = create_optimizer_v2([param], optimizer, lr=1e-2)
        for _ in range(3):
            param.grad = torch.randn(32, 32, generator=generator)
            opt.step()
        return opt.param_groups[0]['beta1'], param.detach()

    beta1, cautious = run('cadafactor')
    assert beta1 == 0.9
    _, plain = run('adafactor')
    assert not torch.allclose(cautious, plain)


def test_radam_legacy_param_group_lr():
    # The step size was cached in a buffer shared across param groups, with the lr baked in, so groups
    # used the lr of whichever group computed the step size first.
    from timm.optim.radam import RAdamLegacy

    def run(lrs):
        params = [Parameter(torch.ones(4)) for _ in lrs]
        opt = RAdamLegacy([{'params': [p], 'lr': lr} for p, lr in zip(params, lrs)])
        for _ in range(10):
            for p in params:
                p.grad = torch.ones(4)
            opt.step()
        return [p.detach() for p in params]

    both = run([1.0, 1e-3])
    torch.testing.assert_close(both[0], run([1.0])[0])
    torch.testing.assert_close(both[1], run([1e-3])[0])


def _adahessian_steps(opt, param, num_steps):
    for _ in range(num_steps):
        opt.zero_grad()
        (param ** 3).sum().backward(create_graph=True)
        opt.step()


def test_lookahead_second_order():
    opt = create_optimizer_v2([Parameter(torch.ones(4))], 'lookahead_adahessian', lr=1e-2)
    assert opt.is_second_order
    assert not create_optimizer_v2([Parameter(torch.ones(4))], 'lookahead_adamw', lr=1e-2).is_second_order


def test_lookahead_load_base_state_dict():
    # Resuming a checkpoint saved without lookahead must not drop the lookahead group keys.
    from timm.optim import Lookahead
    param = Parameter(torch.ones(4))
    base = torch.optim.AdamW([param], lr=1e-2)
    param.grad = torch.ones(4)
    base.step()

    opt = Lookahead(torch.optim.AdamW([param], lr=1e-2), k=2)
    opt.load_state_dict(base.state_dict())
    for _ in range(2):
        param.grad = torch.ones(4)
        opt.step()
    assert opt.param_groups[0]['lookahead_step'] == 2


def test_adahessian_resume_update_each():
    # p.hess is not part of the state_dict, resuming at a step where the hessian is not recomputed
    # (update_each > 1) failed as p.hess was still the initial float.
    from timm.optim.adahessian import Adahessian
    param = Parameter(torch.linspace(-1, 1, 8))
    opt = Adahessian([param], lr=1e-2, update_each=2)
    _adahessian_steps(opt, param, 3)

    param2 = Parameter(param.detach().clone())
    opt2 = Adahessian([param2], lr=1e-2, update_each=2)
    opt2.load_state_dict(opt.state_dict())
    _adahessian_steps(opt2, param2, 2)
    assert torch.isfinite(param2).all()


def test_adahessian_closure():
    from timm.optim.adahessian import Adahessian
    param = Parameter(torch.linspace(-1, 1, 8))
    opt = Adahessian([param], lr=1e-2)

    def closure():
        opt.zero_grad()
        loss = (param ** 3).sum()
        loss.backward(create_graph=True)
        return loss

    before = param.detach().clone()
    opt.step(closure)
    assert not torch.equal(param.detach(), before)


def _laprop_run(param, opt, grads):
    for grad in grads:
        param.grad = grad.clone()
        opt.step()


def test_laprop_low_precision_resume():
    # Optimizer.load_state_dict casts state to the param dtype, the lr EMA scalar was left in bf16
    # where it stops increasing well short of 1.0, permanently shrinking the update.
    from timm.optim.laprop import LaProp
    generator = torch.Generator().manual_seed(0)
    grads = [torch.randn(8, generator=generator).bfloat16() for _ in range(20)]

    param = Parameter(torch.ones(8, dtype=torch.bfloat16))
    opt = LaProp([param], lr=1e-2)
    _laprop_run(param, opt, grads)

    param2 = Parameter(torch.ones(8, dtype=torch.bfloat16))
    opt2 = LaProp([param2], lr=1e-2)
    _laprop_run(param2, opt2, grads[:10])
    opt3 = LaProp([param2], lr=1e-2)
    opt3.load_state_dict(opt2.state_dict())
    assert opt3.state[param2]['exp_avg_lr_2'].dtype == torch.float32
    _laprop_run(param2, opt3, grads[10:])
    torch.testing.assert_close(param2, param)


@pytest.mark.parametrize('tensor_lr', [False, True])
def test_laprop_zero_lr(tensor_lr):
    # lr is folded into the momentum, with lr == 0 the update must be zero.
    from timm.optim.laprop import LaProp
    generator = torch.Generator().manual_seed(0)
    lr = torch.tensor(1e-2) if tensor_lr else 1e-2
    param = Parameter(torch.ones(8))
    opt = LaProp([param], lr=lr)
    _laprop_run(param, opt, [torch.randn(8, generator=generator) for _ in range(5)])
    opt.param_groups[0]['lr'] = torch.tensor(0.) if tensor_lr else 0.
    before = param.detach().clone()
    _laprop_run(param, opt, [torch.randn(8, generator=generator) for _ in range(5)])
    torch.testing.assert_close(param.detach(), before)


@pytest.mark.parametrize('optimizer', ['nmuon', 'nadamuon'])
def test_muon_nesterov_grad_unchanged(optimizer):
    generator = torch.Generator().manual_seed(0)
    param = Parameter(torch.randn(16, 8, generator=generator))
    opt = create_optimizer_v2([param], optimizer, lr=1e-2)
    for _ in range(2):
        param.grad = torch.randn(16, 8, generator=generator)
        grad = param.grad.clone()
        opt.step()
        torch.testing.assert_close(param.grad, grad)


def test_muon_load_adamw_lr_state_dict():
    # Checkpoints from before fallback_lr_scale stored an absolute adamw_lr, the scale must be relative to
    # the un-scheduled lr, not the lr at the time of saving (e.g. mid warmup).
    from timm.optim.muon import Muon
    param = Parameter(torch.ones(4))
    opt = Muon([param], lr=0.02)
    state_dict = opt.state_dict()
    for adamw_lr, expected in ((0.02, 1.0), (0.01, 0.5), (None, 1.0)):
        group = state_dict['param_groups'][0]
        group.pop('fallback_lr_scale', None)
        group['lr'] = 2e-4
        group['initial_lr'] = 0.02
        group.pop('adamw_lr', None)
        if adamw_lr is not None:
            group['adamw_lr'] = adamw_lr
        opt.load_state_dict(state_dict)
        assert opt.param_groups[0]['fallback_lr_scale'] == pytest.approx(expected)


def test_lamb_no_grads():
    opt = create_optimizer_v2([Parameter(torch.ones(4))], 'lamb', lr=1e-2)
    opt.step()


def test_adafactor_bv_param_group_options():
    # Per param group options must be used for state init, not the optimizer defaults.
    from timm.optim.adafactor_bv import AdafactorBigVision
    p_factor = Parameter(torch.ones(32, 64))
    p_full = Parameter(torch.ones(32, 64))
    opt = AdafactorBigVision([
        {'params': [p_factor]},
        {'params': [p_full], 'min_dim_size_to_factor': 128, 'momentum': None},
    ], lr=1e-2)
    for p in (p_factor, p_full):
        p.grad = torch.ones(32, 64)
    opt.step()
    assert 'exp_avg_sq_r' in opt.state[p_factor] and 'exp_avg' in opt.state[p_factor]
    assert 'exp_avg_sq' in opt.state[p_full] and 'exp_avg' not in opt.state[p_full]


def test_adafactor_bv_decay_offset():
    # beta2 schedule is offset by decay_offset steps, before that the second moment is replaced each step.
    from timm.optim.adafactor_bv import AdafactorBigVision
    generator = torch.Generator().manual_seed(0)
    param = Parameter(torch.ones(8))
    opt = AdafactorBigVision([param], lr=1e-2, decay_offset=3, eps=1e-30)
    for _ in range(3):
        grad = torch.randn(8, generator=generator)
        param.grad = grad
        opt.step()
    torch.testing.assert_close(opt.state[param]['exp_avg_sq'], grad.square() + 1e-30)


def test_kron_update_prob_per_param():
    # The update probability schedule is evaluated per param, params in a group can be at different steps.
    from timm.optim.kron import Kron
    generator = torch.Generator().manual_seed(0)
    p_old, p_new = Parameter(torch.ones(4, 4)), Parameter(torch.ones(4, 4))
    opt = Kron([p_old, p_new], lr=1e-3, preconditioner_update_probability=lambda n: 1.0 if n < 3 else 1e-6)
    for _ in range(5):
        p_old.grad = torch.randn(4, 4, generator=generator)
        opt.step()
    p_old.grad = torch.randn(4, 4, generator=generator)
    p_new.grad = torch.randn(4, 4, generator=generator)
    opt.step()
    assert opt.state[p_new]['update_counter'] == 0  # updated on first step (prob 1.0)
    assert opt.state[p_old]['update_counter'] > 0


def test_mars_first_step_clipped():
    # c_t is clipped to unit norm on every step, including the first.
    from timm.optim.mars import Mars
    param = Parameter(torch.ones(4, 4))
    opt = Mars([param], lr=1e-3, betas=(0.9, 0.99))
    grad = torch.full((4, 4), 2.)  # norm 8
    param.grad = grad.clone()
    opt.step()
    torch.testing.assert_close(opt.state[param]['exp_avg'], 0.1 * grad / grad.norm())
    torch.testing.assert_close(param.grad, grad)


@pytest.mark.parametrize('optimizer', ['adamwlegacy', 'nadamw', 'laprop'])
@pytest.mark.parametrize('foreach', [False, True])
def test_complex_param_matches_real_view(optimizer, foreach):
    # Complex params are optimized as their real view, |g|^2 not g^2 for the second moment.
    if optimizer == 'laprop' and foreach:
        pytest.skip('LaProp has no foreach impl')
    generator = torch.Generator().manual_seed(0)
    grads = [torch.randn(4, 4, dtype=torch.complex64, generator=generator) for _ in range(3)]
    kwargs = dict(lr=1e-2) if optimizer == 'laprop' else dict(lr=1e-2, foreach=foreach)

    p_complex = Parameter(torch.ones(4, 4, dtype=torch.complex64))
    p_real = Parameter(torch.view_as_real(p_complex.detach().clone()).clone())
    opt_complex = create_optimizer_v2([p_complex], optimizer, **kwargs)
    opt_real = create_optimizer_v2([p_real], optimizer, **kwargs)
    for grad in grads:
        p_complex.grad = grad.clone()
        p_real.grad = torch.view_as_real(grad).clone()
        opt_complex.step()
        opt_real.step()
    torch.testing.assert_close(torch.view_as_real(p_complex.detach()), p_real.detach())


_CORRECTED_WD_OPTIMIZERS = [
    n for n in list_optimizers(exclude_filters=('fused*', 'bnb*'))
    if (get_optimizer_info(n).defaults or {}).get('corrected_weight_decay')
]


def _corrected_wd_run(optimizer, lr, lrs, tensor_lr):
    torch.manual_seed(0)
    param = Parameter(torch.ones(8, 8))
    lr = torch.tensor(lr) if tensor_lr else lr
    opt = create_optimizer_v2([param], optimizer, lr=lr, weight_decay=0.1)
    for i, step_lr in enumerate(lrs):
        if tensor_lr:
            lr.fill_(step_lr)
        else:
            opt.param_groups[0]['lr'] = step_lr
        param.grad = torch.randn(8, 8, generator=torch.Generator().manual_seed(i))
        opt.step()
    return param.detach(), opt


@pytest.mark.parametrize('optimizer', _CORRECTED_WD_OPTIMIZERS)
def test_corrected_weight_decay_tensor_lr(optimizer):
    # The max lr for corrected weight decay (lr ** 2 / max_lr) is snapshot at construction. It used to alias a tensor
    # lr, so in-place lr updates silently disabled the correction and lr == 0 gave 0 / 0 = NaN.
    lrs = [1e-3, 5e-4, 0., 0., 2e-4]
    param_float, opt_float = _corrected_wd_run(optimizer, 1e-3, lrs, tensor_lr=False)
    param_tensor, opt_tensor = _corrected_wd_run(optimizer, 1e-3, lrs, tensor_lr=True)
    assert torch.isfinite(param_tensor).all()
    torch.testing.assert_close(param_tensor, param_float)
    for opt in (opt_float, opt_tensor):
        assert opt.param_groups[0]['corrected_weight_decay'] is True
        assert opt.param_groups[0]['max_lr_snapshot'] == pytest.approx(1e-3)


@pytest.mark.parametrize('optimizer', _CORRECTED_WD_OPTIMIZERS)
@pytest.mark.parametrize('saved', ['old_enabled', 'old_disabled', 'new'])
def test_corrected_weight_decay_state_dict(optimizer, saved):
    # Older checkpoints have no max_lr_snapshot, it is backfilled w/ the constructor lr (the max lr used before).
    # Newer checkpoints restore the snapshot as saved.
    _, opt = _corrected_wd_run(optimizer, 2e-3, [2e-3] * 2, tensor_lr=False)
    state_dict = deepcopy(opt.state_dict())
    if saved != 'new':
        for group in state_dict['param_groups']:
            del group['max_lr_snapshot']
            group['corrected_weight_decay'] = saved == 'old_enabled'
    resumed = create_optimizer_v2([Parameter(torch.ones(8, 8))], optimizer, lr=1e-3, weight_decay=0.1)
    resumed.load_state_dict(state_dict)
    for group in resumed.param_groups:
        assert group['corrected_weight_decay'] is (saved != 'old_disabled')
        assert group['max_lr_snapshot'] == pytest.approx(2e-3 if saved == 'new' else 1e-3)


@pytest.mark.parametrize('optimizer', _CORRECTED_WD_OPTIMIZERS)
def test_corrected_weight_decay_zero_lr_raises(optimizer):
    with pytest.raises(ValueError):
        create_optimizer_v2([Parameter(torch.ones(4))], optimizer, lr=0.)


@pytest.mark.parametrize('copy_fn', ['pickle', 'deepcopy'])
def test_kron_pickle_deepcopy(copy_fn):
    # Optimizer.__getstate__ only keeps defaults / state / param_groups, Kron's other attributes (deterministic,
    # compiled fns) must survive or be rebuilt so a copy can keep stepping.
    import pickle
    from timm.optim.kron import Kron

    def step(param, opt, start, num):
        for i in range(start, start + num):
            param.grad = torch.randn(16, 8, generator=torch.Generator().manual_seed(i))
            opt.step()
        return param.detach().clone()

    param = Parameter(torch.ones(16, 8))
    opt = Kron([param], lr=1e-3, weight_decay=0.1, decoupled_decay=True, deterministic=True)
    step(param, opt, 0, 3)
    if copy_fn == 'pickle':
        param_copy, opt_copy = pickle.loads(pickle.dumps((param, opt)))
    else:
        param_copy, opt_copy = deepcopy((param, opt))
    assert opt_copy.deterministic
    torch.testing.assert_close(step(param_copy, opt_copy, 3, 3), step(param, opt, 3, 3), rtol=0, atol=0)


@pytest.mark.parametrize('optimizer,kwargs', [
    ('adabelief', {}),
    ('adabelief', dict(amsgrad=True)),
    ('radabelief', {}),
    ('novograd', {}),
    ('novograd', dict(amsgrad=True)),
    ('kron', dict(precond_dtype=torch.float32, deterministic=True)),
    ('kron', dict(mu_dtype=torch.float32, precond_dtype=torch.float32, deterministic=True)),
    ('kron', dict(mu_dtype=torch.float32, deterministic=True)),
])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_state_dtype_resume(optimizer, kwargs, dtype):
    # Higher precision state must survive loading exactly; casting back after a downcast loses values.
    def step(param, opt, start, num):
        for i in range(start, start + num):
            param.grad = torch.randn(16, 8, generator=torch.Generator().manual_seed(i)).to(dtype)
            opt.step()

    param = Parameter(torch.ones(16, 8, dtype=dtype))
    opt = create_optimizer_v2([param], optimizer, lr=1e-3, **kwargs)
    step(param, opt, 0, 3)
    saved = deepcopy(opt.state_dict())

    param_copy = Parameter(param.detach().clone())
    resumed = create_optimizer_v2([param_copy], optimizer, lr=1e-3, **kwargs)

    def check_loaded(loaded):
        torch.testing.assert_close(loaded.state[param_copy], opt.state[param], rtol=0, atol=0)

    if hasattr(resumed, 'register_load_state_dict_pre_hook'):
        # Honor user pre-hook remapping, and restore full precision before user post-hooks inspect state.
        def remap_ids(loaded, state_dict):
            state_dict = deepcopy(state_dict)
            state_dict['state'][100] = state_dict['state'].pop(0)
            state_dict['param_groups'][0]['params'] = [100]
            return state_dict

        resumed.register_load_state_dict_pre_hook(remap_ids)
        resumed.register_load_state_dict_post_hook(check_loaded)

    resumed.load_state_dict(saved)
    check_loaded(resumed)
    torch.testing.assert_close(saved['state'][0], opt.state[param], rtol=0, atol=0)
    step(param, opt, 3, 3)
    step(param_copy, resumed, 3, 3)
    torch.testing.assert_close(param_copy, param, rtol=0, atol=0)
    torch.testing.assert_close(resumed.state[param_copy], opt.state[param], rtol=0, atol=0)
    assert torch.isfinite(param_copy).all()


def test_adafactor_bv_default_eps_per_param():
    # The dtype dependent default eps was resolved once from the first param, a FP16 param after a FP32 one got
    # an eps that underflows in FP16 and NaN on zero gradients.
    from timm.optim.adafactor_bv import AdafactorBigVision
    params = [Parameter(torch.ones(32, 32)), Parameter(torch.ones(32, 32, dtype=torch.float16))]
    opt = AdafactorBigVision(params, lr=1e-2)
    for p in params:
        p.grad = torch.zeros_like(p)
    opt.step()
    assert all(torch.isfinite(p).all() for p in params)


def test_adafactor_zero_lr():
    # lr=0 is a manual lr, only lr=None enables the relative step lr schedule.
    from timm.optim.adafactor import Adafactor
    param = Parameter(torch.ones(8, 8))
    opt = Adafactor([param], lr=0.)
    param.grad = torch.ones(8, 8)
    opt.step()
    assert not opt.param_groups[0]['relative_step']
    torch.testing.assert_close(param.detach(), torch.ones(8, 8), rtol=0, atol=0)
    assert Adafactor([Parameter(torch.ones(2))]).param_groups[0]['relative_step']


_GRAD_UNCHANGED_SKIP = ('fused*', 'bnb*', 'adahessian')


@pytest.mark.parametrize('optimizer', list_optimizers(exclude_filters=_GRAD_UNCHANGED_SKIP))
def test_optimizer_grad_unchanged(optimizer):
    # Optimizers must not modify p.grad, code reading gradients after step() (e.g. grad norm logging) would see
    # the modified values. The foreach SGD nesterov paths (torch.optim.SGD, SGDW) intentionally update grads
    # in place to avoid allocating a full set of temporaries, matching PyTorch, they're excluded via foreach=False.
    opt_args = inspect.signature(get_optimizer_class(optimizer, bind_defaults=False).__init__).parameters
    configs = [{}]
    if 'decoupled_decay' in opt_args:
        configs = [dict(decoupled_decay=False), dict(decoupled_decay=True)]
    for config in configs:
        if 'foreach' in opt_args:
            config['foreach'] = False
        weight = Parameter(torch.randn(16, 8, generator=torch.Generator().manual_seed(0)))
        bias = Parameter(torch.randn(8, generator=torch.Generator().manual_seed(1)))
        opt = create_optimizer_v2([weight, bias], optimizer, lr=1e-2, weight_decay=0.1, **config)
        for i in range(2):
            # scaled so the global grad norm exceeds LAMB's default max_grad_norm clipping threshold
            grads = [3 * torch.randn(p.shape, generator=torch.Generator().manual_seed(10 * i + j))
                     for j, p in enumerate((weight, bias))]
            weight.grad, bias.grad = grads[0].clone(), grads[1].clone()
            opt.step()
            torch.testing.assert_close(weight.grad, grads[0], rtol=0, atol=0)
            torch.testing.assert_close(bias.grad, grads[1], rtol=0, atol=0)


@pytest.mark.parametrize('grad_averaging', [False, True])
@pytest.mark.parametrize('corrected', [False, True])
def test_novograd_decoupled_weight_decay(grad_averaging, corrected):
    # Decoupled decay shrinks the weights by lr * weight_decay (lr ** 2 / max_lr * weight_decay if corrected) every
    # step. It does not go through the momentum, where the paper form ramps up to ~lr * weight_decay / (1 - beta1),
    # and it does not depend on grad_averaging.
    from timm.optim.nvnovograd import NvNovoGrad
    lr, max_lr, wd = 1e-2, 2e-2, 0.1
    param = Parameter(torch.ones(4))
    opt = NvNovoGrad(
        [param], lr=max_lr, weight_decay=wd, grad_averaging=grad_averaging,
        decoupled_decay=True, corrected_weight_decay=corrected,
    )
    opt.param_groups[0]['lr'] = lr
    wd_scale = lr ** 2 / max_lr if corrected else lr
    for _ in range(5):
        before = param.detach().clone()
        param.grad = torch.zeros(4)
        opt.step()
        torch.testing.assert_close(param.detach(), before * (1 - wd_scale * wd))
