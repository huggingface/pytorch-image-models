""" PyTorch MARS Optimizer

Code simplified from https://github.com/AGI-Arena/MARS

Paper: MARS: Unleashing the Power of Variance Reduction for Training Large Models - https://arxiv.org/abs/2411.10438

@article{yuan2024mars,
  title={MARS: Unleashing the Power of Variance Reduction for Training Large Models},
  author={Yuan, Huizhuo and Liu, Yifeng and Wu, Shuang and Zhou, Xun and Gu, Quanquan},
  journal={arXiv preprint arXiv:2411.10438},
  year={2024}
}
"""
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import math

import torch
from torch.optim.optimizer import Optimizer


def mars_single_tensor(
        p,
        grad,
        exp_avg,
        exp_avg_sq,
        lr,
        weight_decay,
        beta1,
        beta2,
        last_grad,
        eps,
        step,
        gamma,
        mars_type,
        is_grad_2d,
        optimize_1d,
        lr_1d_factor,
        betas_1d,
):
    # optimize_1d: use MARS for 1d para, not: use AdamW for 1d para
    if optimize_1d or is_grad_2d:
        one_minus_beta1 = 1. - beta1
        c_t = (grad - last_grad).mul_(gamma * (beta1 / one_minus_beta1)).add_(grad)
        c_t_norm = torch.norm(c_t)
        if c_t_norm > 1.:
            c_t = c_t / c_t_norm
        exp_avg.mul_(beta1).add_(c_t, alpha=one_minus_beta1)
        if mars_type == "adamw":
            exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t, value=1. - beta2)
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            update = p * weight_decay + (exp_avg / bias_correction1).div_(denom)
        elif mars_type == "lion":
            update = p * weight_decay + exp_avg.sign()
        else:
            assert False
        p.add_(update, alpha=-lr)
    else:
        beta1_1d, beta2_1d = betas_1d
        exp_avg.mul_(beta1_1d).add_(grad, alpha=1. - beta1_1d)
        exp_avg_sq.mul_(beta2_1d).addcmul_(grad, grad, value=1. - beta2_1d)
        bias_correction1 = 1.0 - beta1_1d ** step
        bias_correction2 = 1.0 - beta2_1d ** step
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        update = p * weight_decay + (exp_avg / bias_correction1).div_(denom)
        p.add_(update, alpha=-(lr * lr_1d_factor))
    return exp_avg, exp_avg_sq


class Mars(Optimizer):
    """ MARS Optimizer

    Paper: MARS: Unleashing the Power of Variance Reduction for Training Large Models
        https://arxiv.org/abs/2411.10438

    """
    def __init__(
            self,
            params,
            lr=3e-3,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=0.,
            gamma=0.025,
            mars_type="adamw",
            optimize_1d=False,
            lr_1d_factor=1.0,
            betas_1d=None,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        assert mars_type in ["adamw", "lion"], "MARS type not supported"

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            mars_type=mars_type,
            gamma=gamma,
            optimize_1d=optimize_1d,
            lr_1d_factor=lr_1d_factor,
            betas_1d=betas_1d or betas,
        )
        super(Mars, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                # ('----- starting a parameter state', state.keys(), 'Length of state', len(state))
                # State initialization
                if len(state) <= 1:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Last Gradient
                    state['last_grad'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                last_grad = state['last_grad']
                lr = group['lr']
                wd = group['weight_decay']
                beta1, beta2 = group['betas']
                is_grad_2d = grad.ndim >= 2

                mars_single_tensor(
                    p,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    lr,
                    wd,
                    beta1,
                    beta2,
                    last_grad,
                    group['eps'],
                    step,
                    group['gamma'],
                    mars_type=group['mars_type'],
                    is_grad_2d=is_grad_2d,
                    optimize_1d=group['optimize_1d'],
                    lr_1d_factor=group['lr_1d_factor'],
                    betas_1d=group['betas_1d'],
                )

                state['last_grad'] = grad

        return loss
