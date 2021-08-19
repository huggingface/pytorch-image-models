""" Adafactor Optimizer

Lifted from https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

Original header/copyright below.

"""
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import math


class Adafactor(torch.optim.Optimizer):
    """Implements Adafactor algorithm.
    This implementation is based on: `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)

    Note that this optimizer internally adjusts the learning rate depending on the
    *scale_parameter*, *relative_step* and *warmup_init* options.

    To use a manual (external) learning rate schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constants for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3))
        clip_threshold (float): threshold of root mean square of final gradient update (default: 1.0)
        decay_rate (float): coefficient used to compute running averages of square gradient (default: -0.8)
        beta1 (float): coefficient used for computing running averages of gradient (default: None)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_parameter (bool): if True, learning rate is scaled by root mean square of parameter (default: True)
        warmup_init (bool): time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False)
    """

    def __init__(self, params, lr=None, eps=1e-30, eps_scale=1e-3, clip_threshold=1.0,
                 decay_rate=-0.8, betas=None, weight_decay=0.0, scale_parameter=True, warmup_init=False):
        relative_step = not lr
        if warmup_init and not relative_step:
            raise ValueError('warmup_init requires relative_step=True')

        beta1 = None if betas is None else betas[0]   # make it compat with standard betas arg
        defaults = dict(lr=lr, eps=eps, eps_scale=eps_scale, clip_threshold=clip_threshold, decay_rate=decay_rate,
                        beta1=beta1, weight_decay=weight_decay, scale_parameter=scale_parameter,
                        relative_step=relative_step, warmup_init=warmup_init)
        super(Adafactor, self).__init__(params, defaults)

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
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
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

                factored, use_first_moment = self._get_options(group, grad.shape)
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(grad)
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad.shape[:-1]).to(grad)
                        state['exp_avg_sq_col'] = torch.zeros(grad.shape[:-2] + grad.shape[-1:]).to(grad)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    state['RMS'] = 0
                else:
                    if use_first_moment:
                        state['exp_avg'] = state['exp_avg'].to(grad)
                    if factored:
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
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0 - beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0 - beta2t)

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
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
                    update = exp_avg

                if group['weight_decay'] != 0:
                    p_fp32.add_(p_fp32, alpha=-group['weight_decay'] * lr_t)

                p_fp32.add_(-update)
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_fp32)

        return loss
