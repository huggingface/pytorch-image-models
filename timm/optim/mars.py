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
from typing import Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer

from ._types import ParamsT


def _mars_single_tensor_step(
        p: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        lr: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        last_grad: torch.Tensor,
        eps: float,
        step: int,
        gamma: float,
        mars_type: str,
        is_grad_2d: bool,
        optimize_1d: bool,
        lr_1d_factor: bool,
        betas_1d: Tuple[float, float],
        caution: bool,
):
    # optimize_1d ==> use MARS for 1d param, else use AdamW
    if optimize_1d or is_grad_2d:
        one_minus_beta1 = 1. - beta1
        if step == 1:
            # this is a timm addition, making first step more consistent when no grad history, otherwise tests fail
            c_t = grad
        else:
            c_t = (grad - last_grad).mul_(gamma * (beta1 / one_minus_beta1)).add_(grad)
            c_t_norm = torch.norm(c_t)
            if c_t_norm > 1.:
                c_t = c_t / c_t_norm
        exp_avg.mul_(beta1).add_(c_t, alpha=one_minus_beta1)
        if caution:
            mask = (exp_avg * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            exp_avg = exp_avg * mask
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
        if caution:
            mask = (exp_avg * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            exp_avg = exp_avg * mask
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
            params: ParamsT,
            lr: float = 3e-3,
            betas: Tuple[float, float] = (0.9, 0.99),
            eps: float = 1e-8,
            weight_decay: float = 0.,
            gamma: float = 0.025,
            mars_type: str = "adamw",
            optimize_1d: bool = False,
            lr_1d_factor: float = 1.0,
            betas_1d: Optional[Tuple[float, float]] = None,
            caution: bool = False
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
            caution=caution,
        )
        super(Mars, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Mars, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('caution', False)

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

                # FIXME add multi-tensor (if usage warrants), make more standard
                _mars_single_tensor_step(
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
                    caution=group['caution'],
                )

                state['last_grad'] = grad

        return loss
