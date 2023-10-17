""" Optimzier Tests

These tests were adapted from PyTorch' optimizer tests.

"""
import math
import pytest
import functools
from copy import deepcopy

import torch
from torch.testing._internal.common_utils import TestCase
from torch.nn import Parameter
from timm.scheduler import PlateauLRScheduler

from timm.optim import create_optimizer_v2


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
        i = input_cuda if weight.is_cuda else input
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

    # Check that state dict can be loaded even when we cast parameters
    # to a different type and move to a different device.
    if not torch.cuda.is_available():
        return

    with torch.no_grad():
        input_cuda = Parameter(input.clone().detach().float().cuda())
        weight_cuda = Parameter(weight.clone().detach().cuda())
        bias_cuda = Parameter(bias.clone().detach().cuda())
    optimizer_cuda = constructor(weight_cuda, bias_cuda)
    fn_cuda = functools.partial(fn_base, optimizer_cuda, weight_cuda, bias_cuda)

    state_dict = deepcopy(optimizer.state_dict())
    state_dict_c = deepcopy(optimizer.state_dict())
    optimizer_cuda.load_state_dict(state_dict_c)

    # Make sure state dict wasn't modified
    torch_tc.assertEqual(state_dict, state_dict_c)

    for _i in range(20):
        optimizer.step(fn)
        optimizer_cuda.step(fn_cuda)
        torch_tc.assertEqual(weight, weight_cuda)
        torch_tc.assertEqual(bias, bias_cuda)

    # validate deepcopy() copies all public attributes
    def getPublicAttr(obj):
        return set(k for k in obj.__dict__ if not k.startswith('_'))

    assert getPublicAttr(optimizer) == getPublicAttr(deepcopy(optimizer))


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
    if not torch.cuda.is_available():
        return
    _test_basic_cases_template(
        torch.randn(10, 5).cuda(),
        torch.randn(10).cuda(),
        torch.randn(5).cuda(),
        constructor,
        scheduler_constructors
    )


def _test_model(optimizer, params, device=torch.device('cpu')):
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

    def eval(params, w):
        # Depending on w, provide only the x or y gradient
        optimizer.zero_grad()
        loss = rosenbrock(params)
        loss.backward()
        grad = drosenbrock(params.clone().detach())
        # NB: We torture test the optimizer by returning an
        # uncoalesced sparse tensor
        if w:
            i = torch.LongTensor([[0, 0]])
            x = grad[0]
            v = torch.tensor([x / 4., x - x / 4.])
        else:
            i = torch.LongTensor([[1, 1]])
            y = grad[1]
            v = torch.tensor([y - y / 4., y / 4.])
        x = torch.sparse.DoubleTensor(i, v, torch.Size([2])).to(dtype=v.dtype)
        with torch.no_grad():
            params.grad = x.to_dense()
        return loss

    for i in range(2000):
        # Do cyclic coordinate descent
        w = i % 2
        optimizer.step(functools.partial(eval, params, w))
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


#@pytest.mark.parametrize('optimizer', ['sgd', 'momentum'])
# FIXME momentum variant frequently fails in GitHub runner, but never local after many attempts
@pytest.mark.parametrize('optimizer', ['sgd'])
def test_sgd(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=1e-2),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=1e-2),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=1e-2), optimizer)
    )
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


@pytest.mark.parametrize('optimizer',  ['adamw', 'adam', 'nadam', 'adamax'])
def test_adam(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=5e-2)
    )
    _test_model(optimizer, dict(lr=5e-2))


@pytest.mark.parametrize('optimizer',  ['adabelief'])
def test_adabelief(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3), optimizer)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3, weight_decay=1)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=5e-2)
    )
    _test_model(optimizer, dict(lr=5e-2))


@pytest.mark.parametrize('optimizer',  ['radam', 'radabelief'])
def test_rectified(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )
    _test_model(optimizer, dict(lr=1e-3))


@pytest.mark.parametrize('optimizer',   ['adadelta', 'adagrad'])
def test_adaother(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3), optimizer)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3, weight_decay=1)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-1)
    )
    _test_model(optimizer, dict(lr=5e-2))


@pytest.mark.parametrize('optimizer',   ['adafactor'])
def test_adafactor(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(_build_params_dict_single(weight, bias), optimizer)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3, weight_decay=1)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=5e-2)
    )
    _test_model(optimizer, dict(lr=5e-2))


@pytest.mark.parametrize('optimizer',  ['lamb', 'lambc'])
def test_lamb(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=1e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=1e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=1e-3), optimizer)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )
    _test_model(optimizer, dict(lr=1e-3))


@pytest.mark.parametrize('optimizer',  ['lars', 'larc', 'nlars', 'nlarc'])
def test_lars(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=1e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=1e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=1e-3), optimizer)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )
    _test_model(optimizer, dict(lr=1e-3))


@pytest.mark.parametrize('optimizer',  ['madgrad', 'madgradw'])
def test_madgrad(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3), optimizer)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-2)
    )
    _test_model(optimizer, dict(lr=1e-2))


@pytest.mark.parametrize('optimizer',  ['novograd'])
def test_novograd(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3), optimizer)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )
    _test_model(optimizer, dict(lr=1e-3))


@pytest.mark.parametrize('optimizer', ['rmsprop', 'rmsproptf'])
def test_rmsprop(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3), optimizer)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-2)
    )
    _test_model(optimizer, dict(lr=1e-2))


@pytest.mark.parametrize('optimizer', ['adamp'])
def test_adamp(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3), optimizer)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=5e-2)
    )
    _test_model(optimizer, dict(lr=5e-2))


@pytest.mark.parametrize('optimizer', ['sgdp'])
def test_sgdp(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3), optimizer)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )
    _test_model(optimizer, dict(lr=1e-3))


@pytest.mark.parametrize('optimizer', ['lookahead_sgd', 'lookahead_momentum'])
def test_lookahead_sgd(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3), optimizer)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-3)
    )


@pytest.mark.parametrize('optimizer', ['lookahead_adamw', 'lookahead_adam'])
def test_lookahead_adam(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3), optimizer)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=5e-2)
    )


@pytest.mark.parametrize('optimizer', ['lookahead_radam'])
def test_lookahead_radam(optimizer):
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3),
            optimizer,
            lr=1e-3)
    )
    _test_basic_cases(
        lambda weight, bias: create_optimizer_v2(
            _build_params_dict_single(weight, bias, lr=3e-3), optimizer)
    )
    _test_rosenbrock(
        lambda params: create_optimizer_v2(params, optimizer, lr=1e-4)
    )

