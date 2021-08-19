"""
AdamP Optimizer Implementation copied from https://github.com/clovaai/AdamP/blob/master/adamp/adamp.py

Paper: `Slowing Down the Weight Norm Increase in Momentum-based Optimizers` - https://arxiv.org/abs/2006.08217
Code: https://github.com/clovaai/AdamP

Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import math


def _channel_view(x) -> torch.Tensor:
    return x.reshape(x.size(0), -1)


def _layer_view(x) -> torch.Tensor:
    return x.reshape(1, -1)


def projection(p, grad, perturb, delta: float, wd_ratio: float, eps: float):
    wd = 1.
    expand_size = (-1,) + (1,) * (len(p.shape) - 1)
    for view_func in [_channel_view, _layer_view]:
        param_view = view_func(p)
        grad_view = view_func(grad)
        cosine_sim = F.cosine_similarity(grad_view, param_view, dim=1, eps=eps).abs_()

        # FIXME this is a problem for PyTorch XLA
        if cosine_sim.max() < delta / math.sqrt(param_view.size(1)):
            p_n = p / param_view.norm(p=2, dim=1).add_(eps).reshape(expand_size)
            perturb -= p_n * view_func(p_n * perturb).sum(dim=1).reshape(expand_size)
            wd = wd_ratio
            return perturb, wd

    return perturb, wd


class AdamP(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        super(AdamP, self).__init__(params, defaults)

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

                grad = p.grad
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                # Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1.
                if len(p.shape) > 1:
                    perturb, wd_ratio = projection(p, grad, perturb, group['delta'], group['wd_ratio'], group['eps'])

                # Weight decay
                if group['weight_decay'] > 0:
                    p.mul_(1. - group['lr'] * group['weight_decay'] * wd_ratio)

                # Step
                p.add_(perturb, alpha=-step_size)

        return loss
