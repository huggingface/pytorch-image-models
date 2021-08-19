"""
SGDP Optimizer Implementation copied from https://github.com/clovaai/AdamP/blob/master/adamp/sgdp.py

Paper: `Slowing Down the Weight Norm Increase in Momentum-based Optimizers` - https://arxiv.org/abs/2006.08217
Code: https://github.com/clovaai/AdamP

Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
import math

from .adamp import projection


class SGDP(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, eps=1e-8, delta=0.1, wd_ratio=0.1):
        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
            nesterov=nesterov, eps=eps, delta=delta, wd_ratio=wd_ratio)
        super(SGDP, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)

                # SGD
                buf = state['momentum']
                buf.mul_(momentum).add_(grad, alpha=1. - dampening)
                if nesterov:
                    d_p = grad + momentum * buf
                else:
                    d_p = buf

                # Projection
                wd_ratio = 1.
                if len(p.shape) > 1:
                    d_p, wd_ratio = projection(p, grad, d_p, group['delta'], group['wd_ratio'], group['eps'])

                # Weight decay
                if weight_decay != 0:
                    p.mul_(1. - group['lr'] * group['weight_decay'] * wd_ratio / (1-momentum))

                # Step
                p.add_(d_p, alpha=-group['lr'])

        return loss
