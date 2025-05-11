""" Adafactor (Big Vision variant) for PyTorch

Adapted from the implementation in big vision: https://github.com/google-research/big_vision

Described in 'Scaling Vision Transformers': https://arxiv.org/abs/2106.04560

Adaptation and PyTorch modifications by Ross Wightman
"""
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

from ._types import ParamsT


def _get_scalar_dtype():
    """Get the scalar dtype that the optimizer uses for state"""
    return torch.float64


def _factored_dims(
        shape: Tuple[int, ...],
        factored: bool,
        min_dim_size_to_factor: int
) -> Optional[tuple[int, int]]:
    """Whether to use a factored second moment estimator.

    This function returns a tuple with the two largest axes to reduce over.
    If no two dimensions have size >= min_dim_size_to_factor, return None.

    Args:
      shape: an input shape
      factored: whether to use factored second-moment estimator for > 2d vars.
      min_dim_size_to_factor: only factor accumulator if two array dimensions have at least this size.

    Returns:
      None or a tuple of ints
    """
    if not factored or len(shape) < 2:
        return None
    sorted_dims = sorted(((x, i) for i, x in enumerate(shape)))
    if shape[sorted_dims[-2][1]] < min_dim_size_to_factor:
        return None
    return int(sorted_dims[-2][1]), int(sorted_dims[-1][1])


class AdafactorBigVision(Optimizer):
    """
    PyTorch implementation of BigVision's Adafactor variant with both single and multi tensor implementations.

    Adapted from https://github.com/google-research/big_vision by Ross Wightman
    """

    def __init__(
            self,
            params: ParamsT,
            lr: float = 1.0,
            min_dim_size_to_factor: int = 16,
            decay_rate: float = 0.8,
            decay_offset: int = 0,
            beta2_cap: float = 0.999,
            momentum: Optional[float] = 0.9,
            momentum_dtype: Union[str, torch.dtype] = torch.bfloat16,
            eps: Optional[float] = None,
            weight_decay: float = 0.0,
            clipping_threshold: Optional[float] = None,
            unscaled_wd: bool = False,
            caution: bool = False,
            *,
            foreach: Optional[bool] = False,
    ):
        if isinstance(momentum_dtype, str):
            if momentum_dtype == 'float16':
                momentum_dtype = torch.float16
            elif momentum_dtype == 'bfloat16':
                momentum_dtype = torch.bfloat16
            else:
                assert momentum_dtype == 'float32', f'{momentum_dtype} dtype not supported'
                momentum_dtype = torch.float32
        # FIXME try to check if momentum dtype is appropriate for device? Torch API not great for this.

        defaults = dict(
            lr=lr,
            min_dim_size_to_factor=min_dim_size_to_factor,
            decay_rate=decay_rate,
            decay_offset=decay_offset,
            beta2_cap=beta2_cap,
            momentum=momentum,
            momentum_dtype=momentum_dtype,
            eps=eps,
            weight_decay=weight_decay,
            clipping_threshold=clipping_threshold,
            unscaled_wd=unscaled_wd,
            caution=caution,
            foreach=foreach,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('caution', False)
            group.setdefault('foreach', None)
            for p in group['params']:
                p_state = self.state.get(p, {})
                if len(p_state) != 0 and not torch.is_tensor(p_state['step']):
                    p_state['step'] = torch.tensor(float(p_state['step']), dtype=_get_scalar_dtype())

                if 'exp_avg' in p_state and torch.is_tensor(p_state['exp_avg']):
                    # FIXME this is a bit of a hack, optimizer.load_state_dict appears to upcast
                    # the momentum to float32 (it's half precision in the state_dict), need to
                    # look into this further. Better to override _process_value_according_to_param_policy?
                    p_state['exp_avg'] = p_state['exp_avg'].to(dtype=self.defaults['momentum_dtype'])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avg_sq_rs = []
            exp_avg_sq_cs = []
            exp_avg_sqs = []
            state_steps = []
            exp_avgs = []  # For momentum

            for p in group['params']:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("Sparse gradients not supported")

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                if len(state) == 0:
                    # NOTE step on CPU, probably need some more though to make capturable
                    state['step'] = torch.tensor(0.0, dtype=_get_scalar_dtype())

                    shape = p.grad.shape
                    factored_dims = _factored_dims(
                        shape,
                        factored=True,
                        min_dim_size_to_factor=self.defaults['min_dim_size_to_factor']
                    )

                    if factored_dims is not None:
                        dc, dr = factored_dims
                        row_shape = list(p.grad.shape)
                        row_shape[dr] = 1
                        col_shape = list(p.grad.shape)
                        col_shape[dc] = 1
                        state['exp_avg_sq_r'] = p.grad.new_zeros(row_shape)
                        state['exp_avg_sq_c'] = p.grad.new_zeros(col_shape)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)

                    if self.defaults['momentum'] is not None:
                        state['exp_avg'] = torch.zeros_like(p.grad, dtype=self.defaults['momentum_dtype'])

                state_steps.append(state['step'])
                exp_avg_sq_rs.append(state.get('exp_avg_sq_r', None))
                exp_avg_sq_cs.append(state.get('exp_avg_sq_c', None))
                exp_avg_sqs.append(state.get('exp_avg_sq', None))
                exp_avgs.append(state.get('exp_avg', None))

            if group['foreach']:
                func = _multi_tensor_adafactor
            else:
                func = _single_tensor_adafactor

            func(
                params=params_with_grad,
                grads=grads,
                exp_avg_sq_rs=exp_avg_sq_rs,
                exp_avg_sq_cs=exp_avg_sq_cs,
                exp_avg_sqs=exp_avg_sqs,
                exp_avgs=exp_avgs,
                state_steps=state_steps,
                beta2_decay=group['decay_rate'],
                beta2_cap=group['beta2_cap'],
                min_dim_size_to_factor=group['min_dim_size_to_factor'],
                eps=group['eps'],
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                momentum_dtype=group['momentum_dtype'],
                clipping_threshold=group['clipping_threshold'],
                unscaled_wd=group['unscaled_wd'],
                caution=group['caution'],
            )

        return loss


def _single_tensor_adafactor(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avg_sq_rs: List[Optional[Tensor]],
        exp_avg_sq_cs: List[Optional[Tensor]],
        exp_avg_sqs: List[Optional[Tensor]],
        exp_avgs: List[Optional[Tensor]],
        state_steps: List[Tensor],
        *,
        beta2_decay: float,
        beta2_cap: float,
        min_dim_size_to_factor: int,
        eps: float,
        lr: float,
        weight_decay: float,
        momentum: Optional[float],
        momentum_dtype: Union[str, torch.dtype],
        clipping_threshold: Optional[float],
        unscaled_wd: bool,
        caution: bool,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg_sq_r = exp_avg_sq_rs[i]
        exp_avg_sq_c = exp_avg_sq_cs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg = exp_avgs[i]
        step_t = state_steps[i]
        if eps is None:
            # default eps for avoiding div by zero, diff from float type eps
            eps = 1e-7 if grad.dtype == torch.float16 else 1e-30

        # Update step
        step_t += 1
        beta2_t = min(beta2_cap, 1.0 - float(step_t) ** (-beta2_decay))
        one_minus_beta2_t = 1 - beta2_t

        grad_sqr = torch.square(grad) + eps
        # NOTE application of eps (epsilon1) mirrors the optax/big vision/t5x approach
        if exp_avg_sq is None:
            # factorized second moment
            dc, dr = _factored_dims(grad.shape, True, min_dim_size_to_factor=min_dim_size_to_factor)
            exp_avg_sq_r.lerp_(grad_sqr.mean(dim=dr, keepdim=True), one_minus_beta2_t)
            exp_avg_sq_c.lerp_(grad_sqr.mean(dim=dc, keepdim=True), one_minus_beta2_t)

            reduce_dc = dc - 1 if dc > dr else dc
            row_col_mean = exp_avg_sq_r.mean(dim=reduce_dc, keepdim=True)
            row_factor = (exp_avg_sq_r / row_col_mean).rsqrt()
            col_factor = exp_avg_sq_c.rsqrt()

            update = grad * row_factor * col_factor
        else:
            # non-factorized second moment
            assert exp_avg_sq_r is None and exp_avg_sq_c is None
            exp_avg_sq.lerp_(grad_sqr, one_minus_beta2_t)
            update = grad * exp_avg_sq.rsqrt()

        # Clip by RMS value
        if clipping_threshold is not None:
            denom = (update.norm(2) / ((update.numel() ** 0.5) / clipping_threshold)).clamp_(max=1.0)
            update.div_(denom)

        # Apply momentum (in different dtype)
        if momentum is not None and exp_avg is not None:
            if momentum_dtype != grad.dtype:
                exp_avg.lerp_(update.to(momentum_dtype), 1 - momentum)  # ema
                update = exp_avg.to(grad.dtype)
            else:
                exp_avg.lerp_(update, 1 - momentum)  # ema
                update = exp_avg.clone()

            if caution:
                # apply caution as per 'Cautious Optimizers': https://arxiv.org/abs/2411.16085
                mask = (update * grad > 0).to(grad.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                update.mul_(mask)

        # Scale by learning rate
        update.mul_(lr)

        # Perform weight decay
        if weight_decay != 0:
            if unscaled_wd:
                # match big vision impl, 'fully decoupled' decay w/o LR scaling
                param.mul_(1. - weight_decay)
            else:
                # match typical pytorch behaviour for decoupled decay, eg adamw where wd is scaled by LR
                param.mul_(1. - lr * weight_decay)

        # Update parameters
        param.add_(update, alpha=-1.0)


def _multi_tensor_adafactor(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avg_sq_rs: List[Optional[Tensor]],
        exp_avg_sq_cs: List[Optional[Tensor]],
        exp_avg_sqs: List[Optional[Tensor]],
        exp_avgs: List[Optional[Tensor]],
        state_steps: List[Tensor],
        *,
        beta2_decay: float,
        beta2_cap: float,
        min_dim_size_to_factor: int,
        eps: float,
        lr: float,
        weight_decay: float,
        momentum: Optional[float],
        momentum_dtype: Union[str, torch.dtype],
        clipping_threshold: Optional[float],
        unscaled_wd: bool,
        caution: bool,
):
    # FIXME TODO
    assert False, 'multi-tensor fn (foreach=True) not implemented yet'
