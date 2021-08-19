"""RAdam Optimizer.
Implementation lifted from: https://github.com/LiyuanLucasLiu/RAdam
Paper: `On the Variance of the Adaptive Learning Rate and Beyond` - https://arxiv.org/abs/1908.03265
"""
import math
import torch
from torch.optim.optimizer import Optimizer


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

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
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    num_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    num_sma_max = 2 / (1 - beta2) - 1
                    num_sma = num_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = num_sma

                    # more conservative since it's an approximated value
                    if num_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) *
                            (num_sma - 4) / (num_sma_max - 4) *
                            (num_sma - 2) / num_sma *
                            num_sma_max / (num_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

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
