"""RAdam Optimizer.
Implementation lifted from: https://github.com/LiyuanLucasLiu/RAdam
Paper: `On the Variance of the Adaptive Learning Rate and Beyond` - https://arxiv.org/abs/1908.03265

NOTE: This impl has been deprecated in favour of torch.optim.RAdam and remains as a reference
"""
import math
from typing import Any, Dict

import torch
from torch.optim.optimizer import Optimizer

from ._helpers import _load_state_dict_preserving_dtypes


class RAdamLegacy(Optimizer):
    """ PyTorch RAdam optimizer

    NOTE: This impl has been deprecated in favour of torch.optim.AdamW and remains as a reference
    """
    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super(RAdamLegacy, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdamLegacy, self).__setstate__(state)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # state is always kept in fp32
        _load_state_dict_preserving_dtypes(
            self, state_dict, super().load_state_dict,
            lambda group, param: {key: torch.float32 for key in ('exp_avg', 'exp_avg_sq')},
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_fp32 = p.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                # NOTE the original impl cached num_sma / step_size in a buffer shared across all param groups,
                # with lr and betas baked in, so groups w/ a different lr or betas used the wrong step size.
                beta2_t = beta2 ** state['step']
                num_sma_max = 2 / (1 - beta2) - 1
                num_sma = num_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                # more conservative since it's an approximated value
                if num_sma >= 5:
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) *
                        (num_sma - 4) / (num_sma_max - 4) *
                        (num_sma - 2) / num_sma *
                        num_sma_max / (num_sma_max - 2)) / (1 - beta1 ** state['step'])
                else:
                    step_size = group['lr'] / (1 - beta1 ** state['step'])

                if group['weight_decay'] != 0:
                    p_fp32.add_(p_fp32, alpha=-group['weight_decay'] * group['lr'])

                # more conservative since it's an approximated value
                if num_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    p_fp32.add_(exp_avg, alpha=-step_size)

                p.copy_(p_fp32)

        return loss
