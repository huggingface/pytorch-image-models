""" Lion Optimizer
Paper: `Symbolic Discovery of Optimization Algorithms` - https://arxiv.org/abs/2302.06675
Original Impl: https://github.com/google/automl/tree/master/lion
"""
# Copyright 2023 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import List

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(
            self,
            params,
            lr=1e-4,
            betas=(0.9, 0.99),
            weight_decay=0.0,
            maximize=False,
            foreach=None,
    ):
        """Initialize the hyperparameters.

        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.

        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])

            lion(
                params_with_grad,
                grads,
                exp_avgs,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                maximize=group['maximize'],
                foreach=group['foreach'],
            )

        return loss


def lion(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        maximize: bool = False,
        foreach: bool = None,
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
):
    r"""Functional API that performs Lion algorithm computation.
    """
    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_lion
    else:
        func = _single_tensor_lion

    func(
        params,
        grads,
        exp_avgs,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        maximize=maximize,
    )


def _single_tensor_lion(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        maximize: bool,
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            param = torch.view_as_real(param)

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Weight update
        update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)
        param.add_(torch.sign(update), alpha=-lr)

        # Decay the momentum running average coefficient
        exp_avg.lerp_(grad, 1 - beta2)


def _multi_tensor_lion(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        maximize: bool,
):
    if len(params) == 0:
        return

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

    grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grads]
    exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avgs]
    params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in params]

    # Perform stepweight decay
    torch._foreach_mul_(params, 1 - lr * weight_decay)

    # Weight update
    updates = torch._foreach_mul(exp_avgs, beta1)
    torch._foreach_add_(updates, grads, alpha=1 - beta1)

    updates = [u.sign() for u in updates]
    torch._foreach_add_(params, updates, alpha=-lr)

    # Decay the momentum running average coefficient
    torch._foreach_mul_(exp_avgs, beta2)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta2)
