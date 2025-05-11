""" Adafactor Optimizer

Lifted from https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

Modified by Ross Wightman to fix some issues with factorization dims for non nn.Linear layers

Original header/copyright below.
"""
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Optional, Tuple

import torch

from ._types import ParamsT


class Adafactor(torch.optim.Optimizer):
    """Implements Adafactor algorithm.

    This implementation is based on: `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)

    Note that this optimizer internally adjusts the learning rate depending on the
    *scale_parameter*, *relative_step* and *warmup_init* options.

    To use a manual (external) learning rate schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Ags:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: external learning rate
        eps: regularization constants for square gradient and parameter scale respectively
        eps_scale: regularization constants for parameter scale respectively
        clip_threshold: threshold of root-mean-square of final gradient update
        decay_rate: coefficient used to compute running averages of square gradient
        beta1: coefficient used for computing running averages of gradient
        weight_decay: weight decay
        scale_parameter: if True, learning rate is scaled by root-mean-square of parameter
        warmup_init: time-dependent learning rate computation depends on whether warm-up initialization is being used
    """

    def __init__(
            self,
            params: ParamsT,
            lr: Optional[float] = None,
            eps: float = 1e-30,
            eps_scale: float = 1e-3,
            clip_threshold: float = 1.0,
            decay_rate: float = -0.8,
            betas: Optional[Tuple[float, float]] = None,
            weight_decay: float = 0.0,
            scale_parameter: bool = True,
            warmup_init: bool = False,
            min_dim_size_to_factor: int = 16,
            caution: bool = False,
    ):
        relative_step = not lr
        if warmup_init and not relative_step:
            raise ValueError('warmup_init requires relative_step=True')

        beta1 = None if betas is None else betas[0]   # make it compat with standard betas arg
        defaults = dict(
            lr=lr,
            eps=eps,
            eps_scale=eps_scale,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
            min_dim_size_to_factor=min_dim_size_to_factor,
            caution=caution,
        )
        super(Adafactor, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('caution', False)
            group.setdefault('min_dim_size_to_factor', 16)

    @staticmethod
    def _get_lr(param_group, param_state):
        if param_group['relative_step']:
            min_step = 1e-6 * param_state['step'] if param_group['warmup_init'] else 1e-2
            lr_t = min(min_step, 1.0 / math.sqrt(param_state['step']))
            param_scale = 1.0
            if param_group['scale_parameter']:
                param_scale = max(param_group['eps_scale'], param_state['RMS'])
            param_group['lr'] = lr_t * param_scale
        return param_group['lr']

    @staticmethod
    def _get_options(param_group, param_shape, min_size_to_factor=16):
        use_first_moment = param_group['beta1'] is not None
        factored = None
        ndim = len(param_shape)
        # Use a simple heuristic to pick factorization row & col, note other PyTorch impl tend to
        # always use -2, -1 BUT this will not pick correct dims for convolutions. This is a simple
        # approach that should work in most cases, compare to the slightly more involved approach
        # in AdafactorBigVision that sorts dims by size, please report if wrong dims chosen.
        if ndim > 2 and param_shape[0] > min_size_to_factor and param_shape[1] > min_size_to_factor:
            # nD convs in torch are ND + 2 dim weights with leading in/out chs
            factored = 0, 1
        elif ndim >= 2 and param_shape[-2] > min_size_to_factor and param_shape[-1] > min_size_to_factor:
            # if the criteria above didn't match, test trailing dims for eligibility as per original impl
            factored = ndim - 2, ndim - 1

        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col, dim_col, dim_row):
        # from our dim heuristic, always dim_col < dim_row, so col reduction dim for factored row = dim_col
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=dim_col, keepdim=True)).rsqrt_().unsqueeze(dim_row)
        c_factor = exp_avg_sq_col.unsqueeze(dim_col).rsqrt()
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
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
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError('Adafactor does not support sparse gradients.')

                state = self.state[p]

                factored_dims, use_first_moment = self._get_options(
                    group,
                    grad.shape,
                    min_size_to_factor=group['min_dim_size_to_factor'],
                )
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(grad)
                    if factored_dims is not None:
                        dim_col, dim_row = factored_dims
                        def _remove_dim(shape, dim):
                            return shape[:dim] + shape[dim + 1:]
                        state['exp_avg_sq_row'] = torch.zeros(_remove_dim(grad.shape, dim_row)).to(grad)
                        state['exp_avg_sq_col'] = torch.zeros(_remove_dim(grad.shape, dim_col)).to(grad)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    state['RMS'] = 0
                else:
                    if use_first_moment:
                        state['exp_avg'] = state['exp_avg'].to(grad)
                    if factored_dims is not None:
                        state['exp_avg_sq_row'] = state['exp_avg_sq_row'].to(grad)
                        state['exp_avg_sq_col'] = state['exp_avg_sq_col'].to(grad)
                    else:
                        state['exp_avg_sq'] = state['exp_avg_sq'].to(grad)

                p_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p_fp32.float()

                state['step'] += 1
                state['RMS'] = self._rms(p_fp32)
                lr_t = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = grad ** 2 + group['eps']
                if factored_dims is not None:
                    dim_col, dim_row = factored_dims
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=dim_row), alpha=1.0 - beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=dim_col), alpha=1.0 - beta2t)

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col, dim_col, dim_row)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group['clip_threshold']).clamp_(min=1.0))
                update.mul_(lr_t)

                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(update, alpha=1 - group['beta1'])
                    if group['caution']:
                        # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                        mask = (exp_avg * grad > 0).to(grad.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        update = exp_avg * mask
                    else:
                        update = exp_avg

                if group['weight_decay'] != 0:
                    p_fp32.add_(p_fp32, alpha=-group['weight_decay'] * lr_t)

                p_fp32.add_(-update)
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_fp32)

        return loss
