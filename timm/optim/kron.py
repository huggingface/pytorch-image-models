""" PyTorch Implementation of the Kron (PSGD) optimizer

This is a PSGD optimizer using a Kronecker-factored preconditioner.

This impl was adapted from https://github.com/evanatyourservice/kron_torch
by Evan Walters, licensed CC-BY-4.0.

Contributions to above also made by
* Lucas Nestler, added to his https://github.com/ClashLuke/HeavyBall implementation.
* Omead Pooladzandi https://github.com/opooladz

The above work drew from https://github.com/lixilinx/psgd_torch by Xi-Lin Li

References for added functionality:
    Cautious Optimizers: https://arxiv.org/abs/2411.16085
    Why Gradients Rapidly Increase Near the End of Training: https://arxiv.org/abs/2506.02285

This `timm` impl
* works with a wider variety of torch versions
* fixes some checkpoint save/restore (resume issues)
* adds decoupled weight-decay option
* has some refactoring, cleanup of args, default/group items
* warning about not having opt_einsum (unusable without)

"""
import logging
import string
import random
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
try:
    # NOTE opt_einsum needed to avoid blowing up memory with einsum ops
    import opt_einsum
    import torch.backends.opt_einsum
    torch.backends.opt_einsum.enabled = True
    torch.backends.opt_einsum.strategy = "auto-hq"
    has_opt_einsum = True
except ImportError:
    has_opt_einsum = False

try:
    torch._dynamo.config.cache_size_limit = 1_000_000
    has_dynamo = True
except AttributeError:
    has_dynamo = False

from ._types import ParamsT

_logger = logging.getLogger(__name__)


def precond_update_prob_schedule(
        n: float,
        max_prob: float = 1.0,
        min_prob: float = 0.03,
        decay: float = 0.001,
        flat_start: float = 500,
) -> torch.Tensor:
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 200 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work very well for most models and
    training regimes.
    """

    """Exponential anneal with flat start."""
    n = torch.tensor(n, dtype=torch.float32)
    prob = max_prob * torch.exp(-decay * (n - flat_start))
    prob.clamp_(min=min_prob, max=max_prob)

    return prob


class Kron(torch.optim.Optimizer):
    """Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        momentum: Momentum parameter.
        weight_decay: Weight decay.
        preconditioner_update_probability: Probability of updating the preconditioner.
            If None, defaults to a schedule that anneals from 1.0 to 0.03 by 4000 steps.
        max_size_triangular: Max size for dim's preconditioner to be triangular.
        min_ndim_triangular: Minimum number of dimensions a layer needs to have triangular preconditioners.
        memory_save_mode: 'one_diag', 'smart_one_diag', or 'all_diag', None is default
            to set all preconditioners to be triangular, 'one_diag' sets the largest
            or last dim to be diagonal per layer, and 'all_diag' sets all preconditioners to be diagonal.
        momentum_into_precond_update: whether to send momentum into preconditioner
            update instead of raw gradients.
        mu_dtype: Dtype of the momentum accumulator.
        precond_dtype: Dtype of the preconditioner.
        decoupled_decay: AdamW style decoupled weight decay
        corrected_weight_decay: apply corrected weight decay when using decoupled_decay (lr**2 / max_lr)
        flatten: Flatten dimensions instead of fully relying on expressions for higher rank params
        flatten_start_dim: Start of flatten range, defaults to 2. Seems good tradeoff for ConvNets.
        flatten_end_dim: End of flatten range, defaults to -1.
        stochastic_weight_decay: Enable random modulation of weight decay
        deterministic: Deterministic behaviour across save / load (resume). FIXME slow, needs work
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        preconditioner_update_probability: Optional[Union[Callable, float]] = None,
        max_size_triangular: int = 2048,
        min_ndim_triangular: int = 2,
        memory_save_mode: Optional[str] = None,
        momentum_into_precond_update: bool = True,
        precond_lr: float = 0.1,
        precond_init_scale: float = 1.0,
        mu_dtype: Optional[torch.dtype] = None,
        precond_dtype: Optional[torch.dtype] = None,
        decoupled_decay: bool = False,
        corrected_weight_decay: bool = False,
        flatten: bool = False,
        flatten_start_dim: int = 2,
        flatten_end_dim: int = -1,
        stochastic_weight_decay: bool = False,
        deterministic: bool = False,
    ):
        if not has_opt_einsum:
            warnings.warn("It is highly recommended to have 'opt_einsum' installed for this optimizer.")

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid beta parameter: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            momentum_into_precond_update=momentum_into_precond_update,
            precond_lr=precond_lr,
            precond_init_scale=precond_init_scale,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            decoupled_decay=decoupled_decay,
            corrected_weight_decay=corrected_weight_decay,
            flatten=flatten,
            flatten_start_dim=flatten_start_dim,
            flatten_end_dim=flatten_end_dim,
            stochastic_weight_decay=stochastic_weight_decay,
        )
        super(Kron, self).__init__(params, defaults)

        self._param_exprs = {}  # cache for einsum expr
        self._tiny = torch.finfo(torch.bfloat16).tiny
        self.rng = random.Random(1337)
        self.deterministic = deterministic

        # make compile optional (for bwd compat)
        if has_dynamo:
            self._calc_A_and_conjB = torch.compile(_calc_A_and_conjB, fullgraph=True, dynamic=False)
            self._q_terms = torch.compile(_q_terms, fullgraph=True, dynamic=False)
            self._precond_grad = torch.compile(_precond_grad, fullgraph=True, dynamic=False)
            self._balance_Q = torch.compile(_balance_Q, fullgraph=True, dynamic=False)
        else:
            self._calc_A_and_conjB = _calc_A_and_conjB
            self._q_terms = _q_terms
            self._precond_grad = _precond_grad
            self._balance_Q = _balance_Q

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('corrected_weight_decay', False)

    def __getstate__(self):
        _dict = super().__getstate__()
        _dict["rng"] = self.rng
        return _dict

    def state_dict(self) -> Dict[str, Any]:
        # Get the optimizer's state dict
        optimizer_state = super().state_dict()

        # Add the generator state
        optimizer_state['rng_state'] = self.rng.getstate()
        return optimizer_state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Extract and remove the RNG state from the state dict
        rng_states = {}
        if 'rng_state' in state_dict:
            rng_states['rng_state'] = state_dict.pop('rng_state')
            
        # Load the optimizer state
        super().load_state_dict(state_dict)
        state_dict.update(rng_states)  # add back

        # Restore the RNG state if it exists
        if 'rng_state' in rng_states:
            self.rng.setstate(rng_states['rng_state'])

    def __setstate__(self, state):
        super().__setstate__(state)
        self._param_exprs = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_momentum_size = 0
        total_momentum_mb = 0
        total_precond_size = 0
        total_precond_mb = 0

        for group in self.param_groups:
            mu_dtype = group.get("mu_dtype")
            precond_dtype = group.get("precond_dtype", torch.float32)
            momentum_into_precond_update = group.get("momentum_into_precond_update", True)
            update_prob = group.get("preconditioner_update_probability", None)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                flattened = False
                if group['flatten']:
                    grad = safe_flatten(grad, group["flatten_start_dim"], group["flatten_end_dim"])
                    flattened = True

                if len(state) == 0:
                    state["step"] = 0
                    state["update_counter"] = 0
                    state["momentum_buffer"] = torch.zeros_like(grad, dtype=mu_dtype or grad.dtype)
                    # init Q and einsum expressions on first step
                    state["Q"], exprs = _init_Q_exprs(
                        grad,
                        group["precond_init_scale"],
                        group["max_size_triangular"],
                        group["min_ndim_triangular"],
                        group["memory_save_mode"],
                        dtype=precond_dtype,
                    )
                    self._param_exprs[p] = exprs

                    # Accumulate sizes for log
                    momentum_size = state["momentum_buffer"].numel()
                    momentum_mb = momentum_size * state["momentum_buffer"].element_size() / 2**20
                    total_momentum_size += momentum_size
                    total_momentum_mb += momentum_mb

                    precond_size = sum(q.numel() for q in state["Q"])
                    precond_mb = sum(q.numel() * q.element_size() for q in state["Q"]) / 2**20
                    total_precond_size += precond_size
                    total_precond_mb += precond_mb
                elif p not in self._param_exprs:
                    # init only the einsum expressions, called after state load, Q are loaded from state_dict
                    exprs = _init_Q_exprs(
                        grad,
                        group["precond_init_scale"],
                        group["max_size_triangular"],
                        group["min_ndim_triangular"],
                        group["memory_save_mode"],
                        dtype=precond_dtype,
                        init_q=False,
                    )
                    self._param_exprs[p] = exprs
                else:
                    # retrieve cached expressions
                    exprs = self._param_exprs[p]

                # update preconditioners all together deterministically
                if update_prob is None:
                    update_prob = precond_update_prob_schedule
                if callable(update_prob):
                    update_prob = update_prob(state["step"])
                state["update_counter"] += 1
                do_update = state["update_counter"] >= 1 / update_prob
                if do_update:
                    state["update_counter"] = 0

                state["step"] += 1

                # Update momentum buffer
                beta = group["momentum"]
                bias_correction = 1 - beta ** state["step"]
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(group["momentum"]).add_(grad, alpha=1 - group["momentum"])

                # Restore momentum dtype
                if mu_dtype is not None:
                    momentum_buffer.copy_(momentum_buffer.to(dtype=mu_dtype))
                debiased_momentum = (momentum_buffer / bias_correction).to(dtype=precond_dtype)

                # Balance preconditioners roughly every 100 updates
                balance = self.rng.random() < 0.01 and do_update
                if grad.dim() > 1 and balance:
                    self._balance_Q(state["Q"])

                # Update preconditioner
                if do_update:
                    exprA, exprGs, _ = exprs
                    Q = state["Q"]
                    if self.deterministic:
                        torch_rng = torch.Generator(device=debiased_momentum.device)
                        torch_rng.manual_seed(self.rng.randint(0, 2 ** 31))
                    else:
                        torch_rng = None
                    V = torch.randn(
                        debiased_momentum.shape,
                        generator=torch_rng,
                        dtype=precond_dtype,
                        device=debiased_momentum.device,
                    )
                    G = debiased_momentum if momentum_into_precond_update else grad

                    A, conjB = self._calc_A_and_conjB(exprA, G, Q, V)

                    terms = self._q_terms(exprGs, A, conjB)

                    for q, (term1, term2) in zip(Q, terms):
                        tmp = term1 - term2
                        tmp *= group["precond_lr"]
                        if q.dim() < 2:
                            tmp *= q
                            tmp /= (term1 + term2).norm(float("inf")) + self._tiny
                        else:
                            tmp = torch.triu(tmp)
                            tmp /= _norm_lower_bound(term1 + term2) + self._tiny
                            tmp @= q
                        q.sub_(tmp)

                # Precondition gradients
                pre_grad = self._precond_grad(
                    state["Q"],
                    exprs,
                    debiased_momentum,
                ).to(dtype=p.dtype)

                # RMS of pre_grad should be 1.0, so let's cap at 1.1
                pre_grad.mul_(torch.clamp(1.1 / (pre_grad.square().mean().sqrt_() + 1e-8), max=1.0))
                if flattened:
                    pre_grad = pre_grad.view(p.shape)

                # Apply weight decay
                weight_decay = group["weight_decay"]
                if weight_decay != 0:
                    if group["stochastic_weight_decay"]:
                        weight_decay = 2 * self.rng.random() * weight_decay

                    if group["decoupled_decay"]:
                        if group['corrected_weight_decay']:
                            wd_scale = group["lr"] ** 2 / self.defaults['lr']
                        else:
                            wd_scale = group["lr"]
                        p.mul_(1. - wd_scale * weight_decay)
                    else:
                        pre_grad.add_(p, alpha=weight_decay)

                # Update parameters
                p.add_(pre_grad, alpha=-group["lr"])

        if total_momentum_size > 0:
            _logger.info(f"PSGD Momentum buffer size: {total_momentum_size} elements, {total_momentum_mb:.2f} MB")
            _logger.info(f"PSGD Preconditioners size: {total_precond_size} elements, {total_precond_mb:.2f} MB")

        return loss


def safe_flatten(tensor, start_dim=0, end_dim=-1):
    ndim = tensor.ndim

    # Convert negative end_dim to positive and clip to end
    end_dim = min(end_dim if end_dim >= 0 else ndim + end_dim, ndim - 1)

    # If tensor has fewer dims than start_dim or start > end, return tensor as is
    if ndim <= start_dim or start_dim > end_dim:
        return tensor

    # Now safe to flatten
    return tensor.flatten(start_dim, end_dim)


def _init_Q_exprs(
        t,
        scale,
        max_size,
        min_ndim_triangular,
        memory_save_mode,
        dtype=None,
        init_q=True,
):
    """For a scalar or tensor t, we initialize its preconditioner Q and
    reusable einsum expressions for updating Q and preconditioning gradient.
    """
    letters = string.ascii_lowercase + string.ascii_uppercase

    dtype = dtype if dtype is not None else t.dtype
    shape = t.shape
    Q = []
    if len(shape) == 0:  # scalar
        if init_q:
            Q.append(scale * torch.ones_like(t, dtype=dtype))
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
    else:  # tensor
        if len(shape) > 13:
            raise ValueError(f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!")

        scale = scale ** (1 / len(shape))

        if memory_save_mode is None:
            dim_diag = [False for _ in shape]
        elif memory_save_mode == "one_diag":
            rev_sorted_dims = np.argsort(shape)[::-1]
            dim_diag = [False for _ in shape]
            dim_diag[rev_sorted_dims[0]] = True
        elif memory_save_mode == "smart_one_diag":
            # addition proposed by Lucas Nestler
            rev_sorted_dims = np.argsort(shape)[::-1]
            sorted_shape = sorted(shape)
            dim_diag = [False for _ in shape]
            if len(shape) >= 2 and sorted_shape[-1] > sorted_shape[-2]:
                dim_diag[rev_sorted_dims[0]] = True
        elif memory_save_mode == "all_diag":
            dim_diag = [True for _ in shape]
        else:
            raise ValueError(
                f"Invalid memory_save_mode: {memory_save_mode}, must be one of [None, 'one_diag', 'all_diag']")

        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
            if (
                size == 1
                or size > max_size
                or len(shape) < min_ndim_triangular
                or dim_d
            ):
                # use diagonal matrix as preconditioner for this dim
                if init_q:
                    Q.append(scale * torch.ones(size, dtype=dtype, device=t.device))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join([letters[i + 13] if j == i else letters[j] for j in range(len(shape))])
                subscripts = piece1 + "," + piece1 + "->" + letters[i + 13]
                exprGs.append(subscripts)

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                # use triangular matrix as preconditioner for this dim
                if init_q:
                    Q.append(scale * torch.eye(size, dtype=dtype, device=t.device))

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join([letters[i + 13] if j == i else letters[j] for j in range(len(shape))])
                piece2 = "".join([letters[i + 26] if j == i else letters[j] for j in range(len(shape))])
                subscripts = piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                exprGs.append(subscripts)

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P

    exprGs = tuple(exprGs)
    if init_q:
        return [Q, (exprA, exprGs, exprP)]
    else:
        return exprA, exprGs, exprP


def _lb(A, max_abs):
    A = A / max_abs
    aa = torch.real(A * A.conj())
    value0, i = torch.max(torch.sum(aa, dim=0), 0)
    value1, j = torch.max(torch.sum(aa, dim=1), 0)
    if value0 > value1:
        x = A[:, i].conj() @ A
        return max_abs * torch.linalg.vector_norm((x / torch.linalg.vector_norm(x)) @ A.H)
    else:
        x = A @ A[j].conj()
        return max_abs * torch.linalg.vector_norm(A.H @ (x / torch.linalg.vector_norm(x)))


def _norm_lower_bound(A):
    """Cheap lower bound for the spectral norm of A."""
    max_abs = A.norm(float("inf"))
    return torch.where(max_abs > 0, _lb(A, max_abs), max_abs)


def _solve_triangular_right(X, A):
    """X @ inv(A)"""
    orig_dtype = X.dtype
    X = X.to(dtype=torch.float32)
    A = A.to(dtype=torch.float32)
    out = torch.linalg.solve_triangular(A, X.reshape(-1, X.size(-1)), upper=True, left=False).reshape_as(X)
    return out.to(dtype=orig_dtype)


def _balance_Q(Q_in):
    norms = torch.stack([q.norm(float("inf")) for q in Q_in])
    geometric_mean = norms.prod() ** (1 / len(Q_in))
    norms = geometric_mean / norms
    for i, q in enumerate(Q_in):
        q.mul_(norms[i])


def _precond_grad(Q, exprs, G):
    """Precondition gradient G with preconditioner Q."""
    return torch.einsum(exprs[-1], *[q.conj() for q in Q], *Q, G)


def _calc_A_and_conjB(exprA, G, Q, V):
    A = torch.einsum(exprA, *Q, G)
    order = G.dim()
    p = tuple(range(order))
    conjB = torch.permute(V.conj(), p[1:] + p[:1])
    for i, q in enumerate(Q):
        conjB = conjB / q if q.dim() < 2 else _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)
    return A, conjB


def _q_terms(exprGs, A, conjB):
    terms = []
    for exprG in exprGs:
        term1 = torch.einsum(exprG, A, A.conj())
        term2 = torch.einsum(exprG, conjB.conj(), conjB)
        terms.append((term1, term2))
    return terms
