""" Adan Optimizer

Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models[J]. arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677

Implementation adapted from https://github.com/sail-sg/Adan
"""
# Copyright 2022 Garena Online Private Limited
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

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class MultiTensorApply(object):
    available = False
    warned = False

    def __init__(self, chunk_size):
        try:
            MultiTensorApply.available = True
            self.chunk_size = chunk_size
        except ImportError as err:
            MultiTensorApply.available = False
            MultiTensorApply.import_err = err

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)


class Adan(Optimizer):
    """ Implements a pytorch variant of Adan.

    Adan was proposed in Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models
    https://arxiv.org/abs/2208.06677

    Arguments:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        betas: Coefficients used for first- and second-order moments.
        eps: Term added to the denominator to improve numerical stability.
        weight_decay: Decoupled weight decay (L2 penalty)
        no_prox: How to perform the weight decay
        caution: Enable caution from 'Cautious Optimizers'
        foreach: If True would use torch._foreach implementation. Faster but uses slightly more memory.
    """

    def __init__(self,
            params,
            lr: float = 1e-3,
            betas: Tuple[float, float, float] = (0.98, 0.92, 0.99),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            no_prox: bool = False,
            caution: bool = False,
            foreach: Optional[bool] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError('Invalid beta parameter at index 2: {}'.format(betas[2]))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            no_prox=no_prox,
            caution=caution,
            foreach=foreach,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(Adan, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('no_prox', False)
            group.setdefault('caution', False)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    # Exponential moving average of gradient difference
                    state['exp_avg_diff'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        try:
            has_scalar_maximum = 'Scalar' in torch.ops.aten._foreach_maximum_.overloads()
        except:
            has_scalar_maximum = False

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_diffs = []
            neg_pre_grads = []

            beta1, beta2, beta3 = group['betas']
            # assume same step across group now to simplify things
            # per parameter step can be easily supported by making it a tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2 = 1.0 - beta2 ** group['step']
            bias_correction3 = 1.0 - beta3 ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)

                if 'neg_pre_grad' not in state or group['step'] == 1:
                    state['neg_pre_grad'] = -p.grad.clone()

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                exp_avg_diffs.append(state['exp_avg_diff'])
                neg_pre_grads.append(state['neg_pre_grad'])

            if not params_with_grad:
                continue

            if group['foreach'] is None:
                use_foreach = not group['caution'] or has_scalar_maximum
            else:
                use_foreach = group['foreach']

            if use_foreach:
                func = _multi_tensor_adan
            else:
                func = _single_tensor_adan

            func(
                params_with_grad,
                grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                exp_avg_diffs=exp_avg_diffs,
                neg_pre_grads=neg_pre_grads,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                bias_correction3_sqrt=math.sqrt(bias_correction3),
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                no_prox=group['no_prox'],
                caution=group['caution'],
            )

        return loss


def _single_tensor_adan(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        exp_avg_diffs: List[Tensor],
        neg_pre_grads: List[Tensor],
        *,
        beta1: float,
        beta2: float,
        beta3: float,
        bias_correction1: float,
        bias_correction2: float,
        bias_correction3_sqrt: float,
        lr: float,
        weight_decay: float,
        eps: float,
        no_prox: bool,
        caution: bool,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_diff = exp_avg_diffs[i]
        neg_grad_or_diff = neg_pre_grads[i]

        # for memory saving, we use `neg_grad_or_diff` to get some temp variable in an inplace way
        neg_grad_or_diff.add_(grad)

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
        exp_avg_diff.mul_(beta2).add_(neg_grad_or_diff, alpha=1 - beta2)  # diff_t

        neg_grad_or_diff.mul_(beta2).add_(grad)
        exp_avg_sq.mul_(beta3).addcmul_(neg_grad_or_diff, neg_grad_or_diff, value=1 - beta3)  # n_t

        denom = (exp_avg_sq.sqrt() / bias_correction3_sqrt).add_(eps)
        step_size_diff = lr * beta2 / bias_correction2
        step_size = lr / bias_correction1

        if caution:
            # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
            mask = (exp_avg * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            exp_avg = exp_avg * mask

        if no_prox:
            param.mul_(1 - lr * weight_decay)
            param.addcdiv_(exp_avg, denom, value=-step_size)
            param.addcdiv_(exp_avg_diff, denom, value=-step_size_diff)
        else:
            param.addcdiv_(exp_avg, denom, value=-step_size)
            param.addcdiv_(exp_avg_diff, denom, value=-step_size_diff)
            param.div_(1 + lr * weight_decay)

        neg_grad_or_diff.zero_().add_(grad, alpha=-1.0)


def _multi_tensor_adan(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        exp_avg_diffs: List[Tensor],
        neg_pre_grads: List[Tensor],
        *,
        beta1: float,
        beta2: float,
        beta3: float,
        bias_correction1: float,
        bias_correction2: float,
        bias_correction3_sqrt: float,
        lr: float,
        weight_decay: float,
        eps: float,
        no_prox: bool,
        caution: bool,
):
    if len(params) == 0:
        return

    # for memory saving, we use `neg_pre_grads` to get some temp variable in a inplace way
    torch._foreach_add_(neg_pre_grads, grads)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)  # m_t

    torch._foreach_mul_(exp_avg_diffs, beta2)
    torch._foreach_add_(exp_avg_diffs, neg_pre_grads, alpha=1 - beta2)  # diff_t

    torch._foreach_mul_(neg_pre_grads, beta2)
    torch._foreach_add_(neg_pre_grads, grads)
    torch._foreach_mul_(exp_avg_sqs, beta3)
    torch._foreach_addcmul_(exp_avg_sqs, neg_pre_grads, neg_pre_grads, value=1 - beta3)  # n_t

    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, bias_correction3_sqrt)
    torch._foreach_add_(denom, eps)

    step_size_diff = lr * beta2 / bias_correction2
    step_size = lr / bias_correction1

    if caution:
        # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
        masks = torch._foreach_mul(exp_avgs, grads)
        masks = [(m > 0).to(g.dtype) for m, g in zip(masks, grads)]
        mask_scale = [m.mean() for m in masks]
        torch._foreach_maximum_(mask_scale, 1e-3)
        torch._foreach_div_(masks, mask_scale)
        exp_avgs = torch._foreach_mul(exp_avgs, masks)

    if no_prox:
        torch._foreach_mul_(params, 1 - lr * weight_decay)
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)
        torch._foreach_addcdiv_(params, exp_avg_diffs, denom, value=-step_size_diff)
    else:
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)
        torch._foreach_addcdiv_(params, exp_avg_diffs, denom, value=-step_size_diff)
        torch._foreach_div_(params, 1 + lr * weight_decay)

    torch._foreach_zero_(neg_pre_grads)
    torch._foreach_add_(neg_pre_grads, grads, alpha=-1.0)
