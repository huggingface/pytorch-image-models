""" Optimzier Tests

These tests were adapted from PyTorch' optimizer tests.

"""
import fnmatch
import functools
import importlib
import inspect
import os
import pickle
from copy import deepcopy
from typing import Optional, Tuple

import pytest
import torch
from torch.nn import Parameter
from torch.testing._internal.common_utils import TestCase

from timm.optim import create_optimizer_v2, list_optimizers, get_optimizer_class, get_optimizer_info, OptimInfo
from timm.optim import param_groups_layer_decay, param_groups_weight_decay
import timm.optim._helpers as optim_helpers

torch_backend = os.environ.get('TORCH_BACKEND')
if torch_backend is not None:
    importlib.import_module(torch_backend)
torch_device = os.environ.get('TORCH_DEVICE', 'cuda')
torch_version = tuple(int(v) for v in torch.__version__.split('.')[:2])

# HACK relying on internal PyTorch test functionality for comparisons that I don't want to write
torch_tc = TestCase()

# Older PyTorch is missing CPU kernels for many FP16 / BF16 ops (eye, baddbmm, sqrt, lerp, ...)
_old_cpu_low_precision = torch_version < (2, 1)
# Older PyTorch optimizers / ops have limited support for a tensor lr (e.g. as a Tensor alpha)
_old_tensor_lr = torch_version < (2, 1)


def _opt_args(optimizer):
    return inspect.signature(get_optimizer_class(optimizer, bind_defaults=False).__init__).parameters


def _skip_unsupported_registry_defaults(optimizer):
    # Registry defaults can rely on args of newer PyTorch optimizers (e.g. radamw -> RAdam(decoupled_weight_decay))
    opt_args = _opt_args(optimizer)
    if any(arg.kind == arg.VAR_KEYWORD for arg in opt_args.values()):
        return
    unsupported = [k for k in (get_optimizer_info(optimizer).defaults or {}) if k not in opt_args]
    if unsupported:
        pytest.skip(f'{optimizer} defaults {unsupported} not supported by this PyTorch version')


def _test_basic_cases_template(weight, bias, input, constructor):
    weight = Parameter(weight)
    bias = Parameter(bias)
    input = Parameter(input)
    optimizer = constructor(weight, bias)

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


def _test_basic_cases(constructor):
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
        constructor
    )
    # non-contiguous parameters
    _test_basic_cases_template(
        torch.randn(10, 5, 2)[..., 0],
        torch.randn(10, 2)[..., 0],
        torch.randn(5),
        constructor
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
        constructor
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


def _test_rosenbrock(constructor):
    params_t = torch.tensor([1.5, 1.5])

    params = Parameter(params_t)
    optimizer = constructor([params])

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

    torch_tc.assertLessEqual(params.clone().detach().dist(solution), initial_dist)


def _build_params_dict(weight, bias, **kwargs):
    return [{'params': [weight]}, dict(params=[bias], **kwargs)]


def _build_params_dict_single(weight, bias, **kwargs):
    return [dict(params=bias, **kwargs)]


@pytest.mark.parametrize('optimizer', list_optimizers(exclude_filters=('fused*', 'bnb*', 'kron*')))
def test_optim_factory(optimizer):
    _skip_unsupported_registry_defaults(optimizer)
    assert issubclass(get_optimizer_class(optimizer, bind_defaults=False), torch.optim.Optimizer)

    opt_info = get_optimizer_info(optimizer)
    assert isinstance(opt_info, OptimInfo)

    lr = (1e-2,) * 4
    if optimizer in ('mars', 'nadam', 'claprop', 'crmsproptf', 'cadafactorbv', 'csgdw', 'csgdc', 'csgdp', 'clamb'):
        lr = (1e-3,) * 4
    elif optimizer in ('cmars',):
        lr = (1e-4,) * 4

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


def _conv(rosenbrock_lr, model=None, after_step=0, basic=()):
    return dict(rosenbrock_lr=rosenbrock_lr, model=model, after_step=after_step, basic=basic)


# Convergence smoke tests, registry name -> rosenbrock lr, model test kwargs (None to skip), steps before the model
# loss must decrease (some optimizers don't improve on the first step), extra basic case kwargs.
# NOTE the 'momentum' SGD variant frequently fails in the GitHub runner, but never locally.
_CONVERGENCE_CFGS = {
    'sgd': _conv(1e-3, dict(lr=1e-3), basic=(dict(lr=3e-3, momentum=1), dict(lr=3e-3, momentum=1, weight_decay=.1))),
    **dict.fromkeys(
        ['adamw', 'adam', 'nadam', 'adamax', 'nadamw', 'adamwlegacy', 'adamc', 'adamp', 'cadamp'],
        _conv(5e-2, dict(lr=5e-2)),
    ),
    'kron': _conv(1e-3, dict(lr=1e-3)),
    **dict.fromkeys(
        ['muon', 'nmuon', 'adamuon', 'nadamuon', 'laprop', 'madgrad', 'madgradw'],
        _conv(1e-2, dict(lr=1e-2)),
    ),
    **dict.fromkeys(['adopt', 'adoptw'], _conv(3e-3, dict(lr=5e-2), after_step=1)),
    **dict.fromkeys(['adan', 'adanw', 'mars'], _conv(1e-3, dict(lr=5e-2), after_step=1)),
    'adabelief': _conv(5e-2, dict(lr=5e-2), basic=(dict(lr=1e-3, weight_decay=1),)),
    **dict.fromkeys(
        ['radam', 'radabelief', 'lamb', 'lambc', 'lars', 'larc', 'nlars', 'nlarc', 'novograd', 'sgdp'],
        _conv(1e-3, dict(lr=1e-3)),
    ),
    **dict.fromkeys(['adadelta', 'adagrad'], _conv(1e-1, dict(lr=5e-2), basic=(dict(lr=1e-3, weight_decay=1),))),
    **dict.fromkeys(['adafactor', 'adafactorbv'], _conv(5e-2, dict(lr=5e-2), basic=(dict(lr=1e-3, weight_decay=1),))),
    **dict.fromkeys(['rmsprop', 'rmsproptf'], _conv(1e-2, dict(lr=1e-2))),
    **dict.fromkeys(['csgdp', 'csgdw'], _conv(5e-4, dict(lr=5e-4))),
    **dict.fromkeys(['lookahead_sgd', 'lookahead_momentum'], _conv(1e-3)),
    **dict.fromkeys(['lookahead_adamw', 'lookahead_adam'], _conv(5e-2)),
    'lookahead_radam': _conv(1e-4),
}


@pytest.mark.parametrize('optimizer', list(_CONVERGENCE_CFGS))
def test_optimizer_convergence(optimizer):
    cfg = _CONVERGENCE_CFGS[optimizer]
    for kwargs in cfg['basic']:
        _test_basic_cases(lambda weight, bias: create_optimizer_v2([weight, bias], optimizer, **kwargs))
    _test_rosenbrock(lambda params: create_optimizer_v2(params, optimizer, lr=cfg['rosenbrock_lr']))
    if cfg['model'] is not None:
        _test_model(optimizer, cfg['model'], after_step=cfg['after_step'])




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


@pytest.mark.parametrize('use_matcher,min_scale,expected_scales', [
    (False, None, (0.5, 0.5, 1.0)),  # registry default, automatic trunk / head grouping
    (True, 0.0, (0.25, 0.5, 1.0)),
    (True, 0.375, (0.375, 0.5, 1.0)),  # clamp only the earliest layer
])
def test_param_groups_layer_decay(
        use_matcher: bool,
        min_scale: Optional[float],
        expected_scales: Tuple[float, ...],
) -> None:
    from timm.optim._optim_factory import default_registry

    model = torch.nn.Sequential(*(torch.nn.Linear(4, 4) for _ in range(3)))
    if use_matcher:
        model.group_matcher = lambda coarse=False: lambda name: int(name.split('.')[0])
    else:
        model.pretrained_cfg = {'classifier': '2'}
    if min_scale is None:
        # Exercise the registry's omitted min scale: it used to pass None instead of 0.0.
        groups = default_registry.create_optimizer(
            model, 'adamw', lr=1e-3, weight_decay=0.05, layer_decay=0.5,
        ).param_groups
    else:
        groups = param_groups_layer_decay(model, weight_decay=0.05, layer_decay=0.5, min_scale=min_scale)
    param_to_group = {p: group for group in groups for p in group['params']}
    assert set(param_to_group) == set(model.parameters())
    for name, param in model.named_parameters():
        group = param_to_group[param]
        assert group['lr_scale'] == expected_scales[int(name.split('.')[0])]
        assert group['weight_decay'] == (0.05 if param.ndim > 1 else 0.)


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


def test_adafactor_bv_factored_row_normalization():
    # AdafactorBigVision factorizes the second moment for 2D params, so the update must be
    # transpose-equivariant: stepping W and W.T with transposed grads must give transposed updates.
    # A wrong axis in the row-factor reduction silently collapsed row_factor to 1 (dropping the row
    # normalization) for matrices whose larger dim comes first, breaking that invariance.
    from timm.optim.adafactor_bv import AdafactorBigVision

    generator = torch.Generator().manual_seed(0)
    grad = torch.randn(30, 20, dtype=torch.double, generator=generator)

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


@pytest.mark.parametrize('shape,conv_mode,normalize_spatial', [((8, 36), 'flatten', True)] + [
    (shape, mode, normalize)
    for shape in [(8, 4, 3), (8, 4, 3, 3), (8, 4, 2, 3, 3)]
    for mode, normalize in [('flatten', True), ('batched', False), ('batched', True)]
])
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


# timm optimizers w/ a capturable (CUDA graph safe) impl, torch.optim capturable impls are tested upstream
_CAPTURABLE_OPTIMIZERS = [
    n for n in list_optimizers(exclude_filters=('fused*', 'bnb*'))
    if get_optimizer_class(n, bind_defaults=False).__module__.startswith('timm') and 'capturable' in _opt_args(n)
]


def _capturable_kwargs(optimizer, foreach, **kwargs):
    opt_args = _opt_args(optimizer)
    if 'foreach' in opt_args:
        kwargs['foreach'] = foreach
    elif foreach:
        pytest.skip(f'{optimizer} has no foreach impl')
    if 'eps' not in opt_args:
        kwargs.pop('eps', None)
    return kwargs


@pytest.mark.skipif(not torch.cuda.is_available(), reason='capturable requires CUDA')
@pytest.mark.parametrize('optimizer', _CAPTURABLE_OPTIMIZERS)
@pytest.mark.parametrize('foreach', [False, True])
@pytest.mark.parametrize('dtype,eps', [(torch.float32, 1e-8), (torch.bfloat16, 1e-8), (torch.float16, 1e-4)])
def test_capturable_zero_lr(optimizer, foreach, dtype, eps):
    # Capturable paths must give a zero update at lr == 0 (e.g. warmup from 0 or cooldown to 0 w/ an in-place
    # updated tensor lr) and match the non-capturable result otherwise. AdamW / NAdamW used to divide eps by the
    # step size before adding it, 0 / 0 = NaN when lr == 0.
    kwargs = _capturable_kwargs(optimizer, foreach, eps=eps)

    def run(lrs, capturable):
        param = Parameter(torch.ones(8, 8, device='cuda', dtype=dtype))
        # construct w/ the peak lr (required for corrected weight decay), each step's lr is set below
        lr = torch.tensor(max(lrs), device='cuda') if capturable else max(lrs)
        opt = create_optimizer_v2([param], optimizer, lr=lr, weight_decay=0.05, capturable=capturable, **kwargs)
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
@pytest.mark.parametrize('optimizer', _CAPTURABLE_OPTIMIZERS)
@pytest.mark.parametrize('foreach', [False, True])
def test_capturable_fp16_small_lr(optimizer, foreach):
    # Capturable paths must keep updating FP16 params w/ a small lr, and match the non-capturable path. AdamW /
    # NAdamW folded the step size into an FP16 denominator that overflowed to inf as bias correction approached one.
    kwargs = _capturable_kwargs(optimizer, foreach, eps=1e-4)

    def run(capturable):
        param = Parameter(torch.full((4,), 0.01, device='cuda', dtype=torch.float16))
        lr = torch.tensor(1e-5, device='cuda') if capturable else 1e-5
        opt = create_optimizer_v2([param], optimizer, lr=lr, capturable=capturable, **kwargs)
        for _ in range(30):
            param.grad = torch.ones_like(param)
            before = param.detach().clone()
            opt.step()
        assert not torch.equal(param.detach(), before)
        return param.detach()

    torch.testing.assert_close(run(True), run(False))


@pytest.mark.parametrize('optimizer', list_optimizers(exclude_filters=('fused*', 'bnb*')))
def test_optim_factory_common_kwargs(optimizer):
    _skip_unsupported_registry_defaults(optimizer)
    # eps / betas / momentum passed to the factory must be forwarded to optimizers that accept them,
    # and dropped (not crash) for those that don't.
    info = get_optimizer_info(optimizer)
    opt_args = _opt_args(optimizer)
    assert info.has_eps == ('eps' in opt_args)
    assert info.has_betas == ('betas' in opt_args)
    assert info.has_momentum == ('momentum' in opt_args)
    if optimizer == 'cadafactor':
        # Caution needs a first moment; check the registry default before overriding betas below.
        opt = create_optimizer_v2([Parameter(torch.ones(4, 4))], optimizer, lr=1e-3)
        assert opt.param_groups[0]['beta1'] == 0.9
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
    generator = torch.Generator().manual_seed(0)
    grads = [torch.randn(4, 4, dtype=torch.complex64, generator=generator) for _ in range(3)]
    kwargs = dict(lr=1e-2, foreach=foreach)

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


@pytest.mark.parametrize('optimizer', _CORRECTED_WD_OPTIMIZERS)
@pytest.mark.parametrize('saved', ['old_enabled', 'old_disabled', 'new'])
def test_corrected_weight_decay_state_dict(optimizer, saved):
    # Older checkpoints have no max_lr_snapshot, it is backfilled w/ the constructor lr (the max lr used before).
    # Newer checkpoints restore the snapshot as saved.
    params, opt = _make_optimizer(optimizer, lr=2e-3, weight_decay=0.1)
    _run_steps(params, opt, range(2))
    state_dict = deepcopy(opt.state_dict())
    if saved != 'new':
        for group in state_dict['param_groups']:
            del group['max_lr_snapshot']
            group['corrected_weight_decay'] = saved == 'old_enabled'
    _, resumed = _make_optimizer(optimizer, lr=1e-3, weight_decay=0.1)
    resumed.load_state_dict(state_dict)
    for group in resumed.param_groups:
        assert group['corrected_weight_decay'] is (saved != 'old_disabled')
        assert group['max_lr_snapshot'] == pytest.approx(2e-3 if saved == 'new' else 1e-3)


@pytest.mark.parametrize('optimizer', _CORRECTED_WD_OPTIMIZERS)
def test_corrected_weight_decay_zero_lr_raises(optimizer):
    with pytest.raises(ValueError):
        create_optimizer_v2([Parameter(torch.ones(4))], optimizer, lr=0.)


@pytest.mark.skipif(_old_cpu_low_precision, reason='Older PyTorch lacks FP16 CPU kernels used by AdafactorBigVision')
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


@pytest.mark.parametrize('optimizer', ['adamwlegacy', 'nadamw'])
@pytest.mark.parametrize('foreach', [False, True])
@pytest.mark.parametrize('capturable', [False, True])
def test_adam_clipping_threshold(optimizer, foreach, capturable):
    # RMS update clipping (StableAdamW), the update is scaled by 1 / max(1, rms / threshold) w/ rms the RMS of
    # grad / (sqrt(v_hat) + eps). A gradient spike relative to the second moment estimate is damped, a normal step
    # (rms ~ 1 on the first step) is unchanged.
    if capturable and not torch.cuda.is_available():
        pytest.skip('capturable requires CUDA')
    device = 'cuda' if capturable else 'cpu'
    threshold, eps, beta2 = 1.0, 1e-8, 0.999

    def make(clipping_threshold):
        params = [
            Parameter(torch.randn(16, 8, generator=torch.Generator().manual_seed(0)).to(device)),
            Parameter(torch.randn(8, generator=torch.Generator().manual_seed(1)).to(device)),
        ]
        opt = create_optimizer_v2(
            params, optimizer, lr=1e-2, weight_decay=0., betas=(0.9, beta2), eps=eps, foreach=foreach,
            capturable=capturable, clipping_threshold=clipping_threshold,
        )
        return params, opt

    params, opt = make(threshold)
    params_ref, opt_ref = make(None)
    num_steps = 10
    for step in range(num_steps):
        # a 50x gradient spike on the last step, once the second moment estimate has settled
        scale = 50. if step == num_steps - 1 else 1.
        grads = [(g * scale).to(device) for g in _step_grads(params, step)]
        for p, p_ref, g in zip(params, params_ref, grads):
            p.grad, p_ref.grad = g, g.clone()
        before = [p.detach().clone() for p in params]
        before_ref = [p.detach().clone() for p in params_ref]
        opt.step()
        opt_ref.step()
        for p, p_ref, b, b_ref in zip(params, params_ref, before, before_ref):
            # clipping only scales the update (and w/o weight decay the update does not depend on the param value),
            # the moments match the unclipped run so the expected clip factor can be computed from its state
            v_hat = opt_ref.state[p_ref]['exp_avg_sq'] / (1 - beta2 ** (step + 1))
            rms = (p_ref.grad / (v_hat.sqrt() + eps)).norm() / p_ref.numel() ** 0.5
            clip = (rms / threshold).clamp(min=1.)
            if step == num_steps - 1:
                assert clip > 2.
            torch.testing.assert_close(p.detach() - b, (p_ref.detach() - b_ref) / clip)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph capture requires CUDA')
@pytest.mark.parametrize('optimizer', ['adamwlegacy', 'nadamw'])
@pytest.mark.parametrize('foreach', [False, True])
def test_adam_clipping_threshold_cuda_graph(optimizer, foreach):
    # A capturable step w/ RMS update clipping can be captured in a CUDA graph (no host syncs or host to device
    # copies) and replays match eager steps.
    def make():
        params = [
            Parameter(torch.randn(16, 8, generator=torch.Generator().manual_seed(0)).cuda()),
            Parameter(torch.randn(8, generator=torch.Generator().manual_seed(1)).cuda()),
        ]
        opt = create_optimizer_v2(
            params, optimizer, lr=1e-2, foreach=foreach, capturable=True, clipping_threshold=1.0)
        return params, opt

    grads = [[(g * (50. if step == 5 else 1.)).cuda() for g in _step_grads([torch.empty(16, 8), torch.empty(8)], step)]
             for step in range(6)]
    params, opt = make()
    params_ref, opt_ref = make()
    for p, g in zip(params, grads[0]):
        p.grad = g.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for step in range(3):
            for p, g in zip(params, grads[step]):
                p.grad.copy_(g)
            opt.step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        opt.step()
    for step in range(6):
        for p, g in zip(params_ref, grads[step]):
            p.grad = g.clone()
        opt_ref.step()
        if step >= 3:
            for p, g in zip(params, grads[step]):
                p.grad.copy_(g)
            graph.replay()
    torch.cuda.synchronize()
    for p, p_ref in zip(params, params_ref):
        torch.testing.assert_close(p.detach(), p_ref.detach())


# Registry-wide optimizer property tests. Each property should hold for every optimizer in the registry, intended
# deviations are listed (w/ reason) in _PROPERTY_EXCEPTIONS.
_REGISTRY_OPTIMIZERS = list_optimizers(exclude_filters=('fused*', 'bnb*', 'adahessian'))  # adahessian is 2nd order

# Non-default configs the low precision and resume property tests also run, (registry name, kwargs).
_REGISTRY_CONFIGS = [(name, {}) for name in _REGISTRY_OPTIMIZERS] + [
    ('adabelief', dict(amsgrad=True)),
    ('novograd', dict(amsgrad=True)),
    ('kron', dict(precond_dtype=torch.float32)),
    ('kron', dict(mu_dtype=torch.float32)),
    ('kron', dict(mu_dtype=torch.float32, precond_dtype=torch.float32)),
    ('kron', dict(precond_dtype=torch.float32, momentum_into_precond_update=False)),
]

_PROPERTY_EXCEPTIONS = {
    'param_group_independent': {
        '*lamb*': 'global grad norm clipping spans all param groups',
        'kron*': 'the preconditioner update / balance RNG is shared across params',
        'adagrad': 'torch.optim.Adagrad < 2.12 inits state in __init__, groups from add_param_group have none',
    },
    'zero_lr_no_update': {
        'madgrad*': 'lr is folded into the dual averaged update (as per reference)',
        '*rmsproptf*': 'lr is folded into the momentum (lr_in_momentum, as per TF)',
        'cadafactor': 'momentum accumulates lr scaled updates (as per fairseq / HF Adafactor)',
    },
    'resume_bfloat16': {
        'nadam': 'torch.optim.NAdam keeps mu_product in fp32, cast to the param dtype by Optimizer.load_state_dict',
    },
}

def _skip_property_exceptions(prop, optimizer):
    for pattern, reason in _PROPERTY_EXCEPTIONS.get(prop, {}).items():
        if fnmatch.fnmatch(optimizer, pattern):
            pytest.skip(reason)


def _config_id(config):
    name, kwargs = config
    return name + ''.join(f'-{k}={str(v).replace("torch.", "")}' for k, v in kwargs.items())


def _create_optimizer(optimizer, params, kwargs=None, lr=1e-2, weight_decay=0.01, **extra):
    kwargs = {**(kwargs or {}), **extra}
    if optimizer.startswith('kron'):
        kwargs.setdefault('deterministic', True)  # bitwise comparable across copies / resume
    return create_optimizer_v2(params, optimizer, lr=lr, weight_decay=weight_decay, **kwargs)


def _make_optimizer(optimizer, kwargs=None, dtype=torch.float32, **opt_kwargs):
    weight = Parameter(torch.randn(16, 8, generator=torch.Generator().manual_seed(0)).to(dtype))
    bias = Parameter(torch.randn(8, generator=torch.Generator().manual_seed(1)).to(dtype))
    params = [weight, bias]
    return params, _create_optimizer(optimizer, params, kwargs, **opt_kwargs)


def _step_grads(params, step):
    return [
        torch.randn(p.shape, generator=torch.Generator().manual_seed(100 * step + i)).to(p.dtype)
        for i, p in enumerate(params)
    ]


def _run_steps(params, opt, steps, lr_fn=None):
    for step in steps:
        if lr_fn is not None:
            lr_fn(step)
        for p, g in zip(params, _step_grads(params, step)):
            p.grad = g
        opt.step()
    return [p.detach().clone() for p in params]


def _assert_equal(actual, expected):
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


def _state_dtypes(state):
    dtypes = {}
    for k, v in state.items():
        if torch.is_tensor(v) and v.is_floating_point():
            dtypes[k] = v.dtype
        elif isinstance(v, (list, tuple)) and v and all(torch.is_tensor(t) for t in v):
            dtypes[k] = tuple(t.dtype for t in v)
    return dtypes


@pytest.mark.parametrize('optimizer', _REGISTRY_OPTIMIZERS)
def test_optimizer_no_grad_step(optimizer):
    # A step without any grads is a no-op.
    _skip_unsupported_registry_defaults(optimizer)
    params, opt = _make_optimizer(optimizer)
    before = [p.detach().clone() for p in params]
    opt.step()
    _assert_equal(params, before)


@pytest.mark.parametrize('copy_fn', ['pickle', 'deepcopy'])
@pytest.mark.parametrize('optimizer', _REGISTRY_OPTIMIZERS)
def test_optimizer_copy(optimizer, copy_fn):
    # A pickled / deep copied optimizer (w/ its params) continues exactly as the original.
    _skip_unsupported_registry_defaults(optimizer)
    params, opt = _make_optimizer(optimizer)
    _run_steps(params, opt, range(2))
    copy_fn = (lambda x: pickle.loads(pickle.dumps(x))) if copy_fn == 'pickle' else deepcopy
    params_copy, opt_copy = copy_fn((params, opt))
    _assert_equal(_run_steps(params_copy, opt_copy, range(2, 4)), _run_steps(params, opt, range(2, 4)))


@pytest.mark.parametrize('optimizer', _REGISTRY_OPTIMIZERS)
def test_optimizer_param_group_independent(optimizer):
    # A param group's update does not depend on other param groups (e.g. via state or lr shared across groups).
    _skip_unsupported_registry_defaults(optimizer)
    _skip_property_exceptions('param_group_independent', optimizer)
    other = Parameter(torch.randn(16, 8, generator=torch.Generator().manual_seed(0)))
    weight = Parameter(torch.randn(16, 8, generator=torch.Generator().manual_seed(1)))
    weight_alone = Parameter(weight.detach().clone())
    opt = create_optimizer_v2([other], optimizer, lr=1e-2)
    group_kwargs = dict(lr=1e-4)
    if opt.defaults.get('betas') is not None:
        # different betas too, e.g. step size caches shared across groups (added via add_param_group) bake in betas
        group_kwargs['betas'] = tuple(0.99 * b for b in opt.defaults['betas'])
    opt.add_param_group(dict(params=[weight], **group_kwargs))
    opt_alone = create_optimizer_v2([weight_alone], optimizer, **group_kwargs)
    for step in range(8):
        other.grad, weight.grad = _step_grads([other, weight], step)
        weight_alone.grad = weight.grad.clone()
        opt.step()
        opt_alone.step()
    torch.testing.assert_close(weight.detach(), weight_alone.detach())


@pytest.mark.parametrize('tensor_lr', [False, True])
@pytest.mark.parametrize('optimizer', _REGISTRY_OPTIMIZERS)
def test_optimizer_zero_lr(optimizer, tensor_lr):
    # Params must not change w/ lr == 0 (e.g. warmup from or cooldown to 0).
    _skip_unsupported_registry_defaults(optimizer)
    _skip_property_exceptions('zero_lr_no_update', optimizer)
    if tensor_lr and _old_tensor_lr:
        pytest.skip('Older PyTorch has limited tensor lr support')
    lr = torch.tensor(1e-2) if tensor_lr else 1e-2
    params, opt = _make_optimizer(optimizer, lr=lr)
    _run_steps(params, opt, range(3))
    if tensor_lr:
        lr.fill_(0.)
    else:
        opt.param_groups[0]['lr'] = 0.
    before = [p.detach().clone() for p in params]
    _assert_equal(_run_steps(params, opt, range(3, 6)), before)


@pytest.mark.parametrize('optimizer', _REGISTRY_OPTIMIZERS)
def test_optimizer_tensor_lr(optimizer):
    # An in-place updated tensor lr (as used w/ torch.compile / CUDA graphs) matches a float lr, incl lr == 0.
    _skip_unsupported_registry_defaults(optimizer)
    if _old_tensor_lr:
        pytest.skip('Older PyTorch has limited tensor lr support')
    lrs = [1e-2, 5e-3, 0., 0., 2e-3]
    params, opt = _make_optimizer(optimizer, weight_decay=0.1)
    expected = _run_steps(params, opt, range(5), lambda step: opt.param_groups[0].__setitem__('lr', lrs[step]))
    lr = torch.tensor(lrs[0])
    params, opt = _make_optimizer(optimizer, lr=lr, weight_decay=0.1)
    actual = _run_steps(params, opt, range(5), lambda step: lr.fill_(lrs[step]))
    for a, e in zip(actual, expected):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('config', _REGISTRY_CONFIGS, ids=_config_id)
def test_optimizer_resume(config, dtype):
    # Resuming from a state_dict continues exactly as an uninterrupted run, w/ state dtypes preserved (e.g. the
    # higher precision state some optimizers keep for low precision params).
    optimizer, kwargs = config
    _skip_unsupported_registry_defaults(optimizer)
    if dtype != torch.float32:
        if _old_cpu_low_precision:
            pytest.skip('Older PyTorch lacks low precision CPU kernels')
        _skip_property_exceptions(f'resume_{str(dtype)[6:]}', optimizer)
    params, opt = _make_optimizer(optimizer, kwargs, dtype=dtype)
    _run_steps(params, opt, range(3))
    state_dict = deepcopy(opt.state_dict())
    resumed_params = [Parameter(p.detach().clone()) for p in params]
    expected_dtypes = [_state_dtypes(opt.state[p]) for p in params]
    expected = _run_steps(params, opt, range(3, 6))

    resumed = _create_optimizer(optimizer, resumed_params, kwargs)
    resumed.load_state_dict(state_dict)
    _assert_equal(_run_steps(resumed_params, resumed, range(3, 6)), expected)
    assert [_state_dtypes(resumed.state[p]) for p in resumed_params] == expected_dtypes


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('config', _REGISTRY_CONFIGS, ids=_config_id)
def test_optimizer_low_precision(config, dtype):
    # Optimizing low precision params runs, stays finite and keeps the param dtype.
    optimizer, kwargs = config
    _skip_unsupported_registry_defaults(optimizer)
    if _old_cpu_low_precision:
        pytest.skip('Older PyTorch lacks low precision CPU kernels')
    if dtype == torch.float16 and 'eps' in _opt_args(optimizer):
        kwargs = dict(kwargs, eps=1e-4)  # the default eps of many optimizers (e.g. 1e-8) underflows in FP16
    params, opt = _make_optimizer(optimizer, kwargs, dtype=dtype)
    for p in _run_steps(params, opt, range(3)):
        assert p.dtype == dtype
        assert torch.isfinite(p).all()
    if hasattr(opt, 'reset'):
        # state re-created by reset() must work w/ low precision params as lazily initialized state does
        opt.reset()
        for p in _run_steps(params, opt, range(3, 5)):
            assert p.dtype == dtype
            assert torch.isfinite(p).all()


_FOREACH_OPTIMIZERS = [
    n for n in _REGISTRY_OPTIMIZERS
    if get_optimizer_class(n, bind_defaults=False).__module__.startswith('timm') and 'foreach' in _opt_args(n)
]
_FOREACH_CONFIGS = [(n, {}) for n in _FOREACH_OPTIMIZERS] + [
    ('adamwlegacy', dict(clipping_threshold=1.0)),
    ('nadamw', dict(clipping_threshold=1.0)),
    ('rmsproptf', dict(centered=True)),
    ('rmsproptf', dict(lr_in_momentum=False)),
    ('rmsproptf', dict(momentum=0.)),
    ('rmsproptf', dict(centered=True, lr_in_momentum=False)),
    ('lamb', dict(always_adapt=True)),
    ('lamb', dict(bias_correction=False, grad_averaging=False)),
    ('lars', dict(momentum=0.)),
    ('lars', dict(dampening=0.1, always_adapt=True)),
    ('adabelief', dict(amsgrad=True)),
    ('adabelief', dict(decoupled_decay=False)),
    ('adabelief', dict(fixed_decay=True)),
    ('radabelief', dict(amsgrad=True)),
    ('radabelief', dict(degenerated_to_sgd=False)),
]


def _skip_unsupported_foreach(optimizer):
    # timm foreach (multi-tensor) impls only, torch.optim impls are tested upstream
    if optimizer not in _FOREACH_OPTIMIZERS:
        pytest.skip('no timm foreach impl')
    _skip_property_exceptions('foreach_parity', optimizer)
    if (get_optimizer_info(optimizer).defaults or {}).get('caution') and \
            'Scalar' not in torch.ops.aten._foreach_maximum_.overloads():
        pytest.skip('Cautious foreach impls require a newer PyTorch (Scalar _foreach_maximum_)')
    if 'adopt' in optimizer and not hasattr(torch.optim.Optimizer, '_group_tensors_by_device_and_dtype'):
        pytest.skip('Adopt foreach (multi-tensor) impl requires a newer PyTorch')


@pytest.mark.parametrize('config', _FOREACH_CONFIGS, ids=_config_id)
def test_optimizer_foreach_parity(config):
    # The foreach (multi-tensor) impl matches the single tensor impl, incl a param lagging behind in steps (no grad
    # for a while) and params spanning more than one (device, dtype) group. torch.optim impls are tested upstream.
    # Both impls compiled (dynamo) match too, state derived Python values (e.g. Adopt's first step flags for the
    # lagging param) must not be baked into the graph.
    optimizer, kwargs = config
    _skip_unsupported_registry_defaults(optimizer)
    _skip_unsupported_foreach(optimizer)
    variants = [(False, False), (True, False)]
    if hasattr(torch, 'compile'):
        variants += [(False, True), (True, True)]
    results = []
    for foreach, compiled in variants:
        if compiled:
            torch._dynamo.reset()
        params = [
            Parameter(torch.randn(16, 8, generator=torch.Generator().manual_seed(0))),
            Parameter(torch.randn(8, generator=torch.Generator().manual_seed(1))),
            Parameter(torch.randn(16, 8, generator=torch.Generator().manual_seed(2), dtype=torch.float64)),
        ]
        opt = _create_optimizer(optimizer, params, kwargs, foreach=foreach)
        opt_step = torch.compile(opt.step, backend='eager') if compiled else opt.step
        for step in range(6):
            for i, (p, g) in enumerate(zip(params, _step_grads(params, step))):
                p.grad = None if (i == 1 and step < 3) else g
            opt_step()
        results.append([p.detach().clone() for p in params])
    for (_, compiled), result in zip(variants[1:], results[1:]):
        # compiled step scalars (e.g. bias corrections) are computed in the step tensor dtype (float32), not as
        # Python floats, so the float64 param only matches to ~float32 precision
        tol = dict(rtol=1e-5, atol=1e-6) if compiled else {}
        for a, e in zip(result, results[0]):
            torch.testing.assert_close(a, e, **tol)


@pytest.mark.skipif(not optim_helpers._HAS_L2_CACHE_SIZE, reason='foreach chunking requires PyTorch >= 2.5')
@pytest.mark.parametrize('config', _FOREACH_CONFIGS, ids=_config_id)
def test_optimizer_foreach_chunked(config, monkeypatch):
    # Running the foreach impl in chunks of params is bitwise identical to running all params at once.
    optimizer, kwargs = config
    _skip_unsupported_registry_defaults(optimizer)
    _skip_property_exceptions('foreach_parity', optimizer)
    results = []
    for chunk_size in (0, 136):  # 0 = don't chunk, 136 elements = chunks of [128, 8], [128], [64, 64]
        monkeypatch.setattr(optim_helpers, '_FOREACH_CHUNK_OVERRIDE', chunk_size)
        params = [
            Parameter(torch.randn(16, 8, generator=torch.Generator().manual_seed(0))),
            Parameter(torch.randn(8, generator=torch.Generator().manual_seed(1))),
            Parameter(torch.randn(16, 8, generator=torch.Generator().manual_seed(2))),
            Parameter(torch.randn(8, 8, generator=torch.Generator().manual_seed(3))),
            Parameter(torch.randn(64, generator=torch.Generator().manual_seed(4))),
        ]
        opt = _create_optimizer(optimizer, params, kwargs, foreach=True)
        for step in range(6):
            for i, (p, g) in enumerate(zip(params, _step_grads(params, step))):
                p.grad = None if (i == 1 and step < 3) else g
            opt.step()
        results.append([p.detach().clone() for p in params])
    # LaProp and RAdaBelief foreach apply a shared step size as a single scalar when all params in the list have
    # the same one, chunks w/o the lagging param do, rounding differently than per param scalars
    bitwise = not any(fnmatch.fnmatch(optimizer, pat) for pat in ('*laprop*', 'radabelief'))
    for a, e in zip(*results):
        if bitwise:
            assert torch.equal(a, e)
        else:
            torch.testing.assert_close(a, e)


@pytest.mark.parametrize('foreach', [False, True])
@pytest.mark.parametrize('optimizer', _REGISTRY_OPTIMIZERS)
def test_optimizer_grad_unchanged(optimizer, foreach):
    # Optimizers must not modify p.grad, code reading gradients after step() (e.g. grad norm logging) would see
    # the modified values. The foreach SGD nesterov path of torch.optim.SGD updates grads in place (not tested
    # here, the timm foreach impls are, see _skip_unsupported_foreach).
    _skip_unsupported_registry_defaults(optimizer)
    if foreach:
        _skip_unsupported_foreach(optimizer)
    opt_args = _opt_args(optimizer)
    configs = [{}]
    if 'decoupled_decay' in opt_args:
        configs = [dict(decoupled_decay=False), dict(decoupled_decay=True)]
    for config in configs:
        if 'foreach' in opt_args:
            config['foreach'] = foreach
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


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_laprop_load_legacy_float_state(dtype):
    # Older checkpoints store LaProp's bias correction EMA as a Python float, it must load (to a full precision
    # scalar tensor) w/o the dtype preserving loader trying to restore it as a tensor. The checkpoint is created w/
    # FP32 params (low precision CPU math is unsupported in older PyTorch), only loading uses the param dtype.
    from timm.optim.laprop import LaProp
    param = Parameter(torch.ones(8, 4))
    opt = LaProp([param], lr=1e-2)
    for i in range(2):
        param.grad = torch.randn(8, 4, generator=torch.Generator().manual_seed(i))
        opt.step()
    state_dict = deepcopy(opt.state_dict())
    exp_avg_lr_2 = float(state_dict['state'][0]['exp_avg_lr_2'])
    state_dict['state'][0]['exp_avg_lr_2'] = exp_avg_lr_2
    resumed_param = Parameter(torch.ones(8, 4, dtype=dtype))
    resumed = LaProp([resumed_param], lr=1e-2)
    resumed.load_state_dict(state_dict)
    loaded = resumed.state[resumed_param]['exp_avg_lr_2']
    assert torch.is_tensor(loaded) and loaded.dtype == torch.float32 and loaded.device.type == 'cpu'
    assert loaded.item() == pytest.approx(exp_avg_lr_2, rel=1e-6)
