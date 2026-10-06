""" PyTorch Lamb optimizer w/ behaviour similar to NVIDIA FusedLamb

This optimizer code was adapted from the following (starting with latest)
* https://github.com/HabanaAI/Model-References/blob/2b435114fe8e31f159b1d3063b8280ae37af7423/PyTorch/nlp/bert/pretraining/lamb.py
* https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py
* https://github.com/cybertronai/pytorch-lamb

Use FusedLamb if you can (GPU). The reason for including this variant of Lamb is to have a version that is
similar in behaviour to APEX FusedLamb if you aren't using NVIDIA GPUs or cannot install/use APEX.

In addition to some cleanup, this Lamb impl has been modified to support PyTorch XLA and has been tested on TPU.

References for added functionality:
    Cautious Optimizers: https://arxiv.org/abs/2411.16085
    Why Gradients Rapidly Increase Near the End of Training: https://arxiv.org/abs/2506.02285

Original copyrights for above sources are below.

Modifications Copyright 2021 Ross Wightman
"""
# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License
#
# Copyright (c) 2019 cybertronai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

from ._helpers import (
    _add_scaled_, _foreach_chunked, _foreach_lerp_, _foreach_trust_ratios, _get_value, _init_scalar, _max_lr_snapshot,
    _resolve_foreach, _validate_scalar,
)
from ._types import ParamsT


class Lamb(Optimizer):
    """Implements a pure pytorch variant of FuseLAMB (NvLamb variant) optimizer from apex.optimizers.FusedLAMB
    reference: https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py

    LAMB was proposed in:
    - Large Batch Optimization for Deep Learning - Training BERT in 76 minutes:  https://arxiv.org/abs/1904.00962
    - On the Convergence of Adam and Beyond: https://openreview.net/forum?id=ryQu7f-RZ

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate
        betas: Coefficients used for computing running averages of gradient and its norm.
        eps: Term added to the denominator to improve numerical stability.
        weight_decay: Weight decay
        grad_averaging: Whether apply (1-beta2) to grad when calculating running averages of gradient.
        max_grad_norm: Value used to clip global grad norm.
        trust_clip: Enable LAMBC trust ratio clipping.
        always_adapt: Apply adaptive learning rate to 0.0 weight decay parameter.
        caution: Apply caution.
        decoupled: apply decoupled weight decay
        corrected_weight_decay: apply corrected weight decay (lr**2 / max_lr) when using decoupled_decay
        foreach: use the foreach (multi-tensor) impl, by default (None) used unless unsupported
    """

    def __init__(
            self,
            params: ParamsT,
            lr: float = 1e-3,
            bias_correction: bool = True,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.01,
            grad_averaging: bool = True,
            max_grad_norm: Optional[float] = 1.0,
            trust_clip: bool = False,
            always_adapt: bool = False,
            caution: bool = False,
            decoupled_decay: bool = False,
            corrected_weight_decay: bool = False,
            foreach: Optional[bool] = None,
    ):
        _validate_scalar("learning rate", lr)
        _validate_scalar("epsilon", eps)
        _validate_scalar("weight_decay", weight_decay)
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_averaging=grad_averaging,
            max_grad_norm=max_grad_norm,
            trust_clip=trust_clip,
            always_adapt=always_adapt,
            caution=caution,
            decoupled_decay=decoupled_decay,
            corrected_weight_decay=corrected_weight_decay,
            max_lr_snapshot=_max_lr_snapshot(lr, corrected_weight_decay),
            foreach=foreach,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.defaults.setdefault('max_lr_snapshot', _max_lr_snapshot(self.defaults['lr']))  # pickled pre-snapshot
        for group in self.param_groups:
            group.setdefault('caution', False)
            group.setdefault('decoupled_decay', False)
            group.setdefault('corrected_weight_decay', False)
            group.setdefault('max_lr_snapshot', self.defaults['max_lr_snapshot'])
            group.setdefault('foreach', None)
            if 'step' in group:
                group['step'] = _init_scalar(group['step'], device='cpu')

    def _get_clip_grad_norm(self):
        max_grad_norm = self.defaults['max_grad_norm']
        if max_grad_norm is None:
            return None

        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instead.')
                grads.append(grad)
        if not grads:
            return None
        if hasattr(torch, '_foreach_norm'):
            norms = torch._foreach_norm(grads)
        else:
            norms = [torch.linalg.vector_norm(grad) for grad in grads]
        global_norm = torch.linalg.vector_norm(torch.stack(norms))
        clip_global_norm = (global_norm / max_grad_norm).clamp_(min=1.0)
        return clip_global_norm

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

        clip_grad_norm = self._get_clip_grad_norm() # None if disabled

        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0
            beta3 = 1 - beta1 if grad_averaging else 1.0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' not in group:
                group['step'] = _init_scalar(device='cpu')
            group['step'].add_(1)

            if not any(p.grad is not None for p in group['params']):
                continue

            step = _get_value(group['step'])

            if bias_correction:
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
            else:
                bias_correction1, bias_correction2 = 1.0, 1.0

            params = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient valuesa
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                params.append(p)
                grads.append(p.grad)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

            weight_decay = group['weight_decay']
            decoupled_decay = group.get('decoupled_decay', False)
            lamb(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                clip_grad_norm=clip_grad_norm,
                foreach=group['foreach'],
                lr=group['lr'],
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                eps=group['eps'],
                weight_decay=weight_decay,
                decoupled_decay=decoupled_decay,
                max_lr=group['max_lr_snapshot'] if decoupled_decay and group['corrected_weight_decay'] else None,
                trust_clip=group['trust_clip'],
                always_adapt=group['always_adapt'],
                caution=group['caution'],
            )

        return loss


def lamb(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        clip_grad_norm: Optional[Tensor] = None,
        foreach: Optional[bool] = None,
        *,
        lr: Union[float, Tensor],
        beta1: float,
        beta2: float,
        beta3: float,
        bias_correction1: float,
        bias_correction2: float,
        eps: float,
        weight_decay: float,
        decoupled_decay: bool,
        max_lr: Optional[float],
        trust_clip: bool,
        always_adapt: bool,
        caution: bool,
) -> None:
    """Functional API that performs the LAMB per group update, see Lamb class for details."""
    if _resolve_foreach(foreach, caution, lr, params) and not torch.jit.is_scripting():
        func = _multi_tensor_lamb
    else:
        func = _single_tensor_lamb
    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        clip_grad_norm,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        bias_correction1=bias_correction1,
        bias_correction2=bias_correction2,
        eps=eps,
        weight_decay=weight_decay,
        decoupled_decay=decoupled_decay,
        max_lr=max_lr,
        trust_clip=trust_clip,
        always_adapt=always_adapt,
        caution=caution,
    )


def _lamb_trust_ratio(w_norm: Tensor, g_norm: Tensor, trust_clip: bool) -> Tensor:
    trust_ratio = w_norm / g_norm
    # FIXME nested where required since logical and/or not working in PT XLA
    # Set the ratio to 1.0 (no change) if either weight norm or grad norm is zero
    trust_ratio = torch.where(
        w_norm > 0,
        torch.where(g_norm > 0, trust_ratio, 1.0),
        1.0,
    )
    if trust_clip:
        # LAMBC trust clipping, upper bound fixed at one
        trust_ratio = torch.clamp(trust_ratio, max=1.0)
    return trust_ratio


def _single_tensor_lamb(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        clip_grad_norm: Optional[Tensor],
        *,
        lr: Union[float, Tensor],
        beta1: float,
        beta2: float,
        beta3: float,
        bias_correction1: float,
        bias_correction2: float,
        eps: float,
        weight_decay: float,
        decoupled_decay: bool,
        max_lr: Optional[float],
        trust_clip: bool,
        always_adapt: bool,
        caution: bool,
) -> None:
    for i, p in enumerate(params):
        grad = grads[i]
        if clip_grad_norm is not None:
            grad = grad / clip_grad_norm  # not in-place, leave p.grad unmodified
        exp_avg, exp_avg_sq = exp_avgs[i], exp_avg_sqs[i]

        # Decay the first and second moment running average coefficient
        if beta3 == 1 - beta1:
            exp_avg.lerp_(grad, beta3)  # m_t, grad averaging
        else:
            exp_avg.mul_(beta1).add_(grad, alpha=beta3)  # m_t
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t

        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
        update = (exp_avg / bias_correction1).div_(denom)

        if caution:
            # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
            mask = (update * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            update.mul_(mask)

        if weight_decay != 0:
            if decoupled_decay:
                wd_scale = lr if max_lr is None else lr ** 2 / max_lr
                _add_scaled_(p, p, -wd_scale * weight_decay)
            else:
                update.add_(p, alpha=weight_decay)

        if weight_decay != 0 or always_adapt:
            # Layer-wise LR adaptation. By default, skip adaptation on parameters that are
            # excluded from weight decay, unless always_adapt == True, then always enabled.
            update.mul_(_lamb_trust_ratio(p.norm(2.0), update.norm(2.0), trust_clip))

        _add_scaled_(p, update, -lr)


# _foreach_div(TensorList, Tensor) overload, PyTorch >= 2.1
try:
    _HAS_FOREACH_DIV_TENSOR = 'Tensor' in torch.ops.aten._foreach_div.overloads()
except Exception:
    _HAS_FOREACH_DIV_TENSOR = False


@_foreach_chunked(4)
def _multi_tensor_lamb(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        clip_grad_norm: Optional[Tensor],
        *,
        lr: Union[float, Tensor],
        beta1: float,
        beta2: float,
        beta3: float,
        bias_correction1: float,
        bias_correction2: float,
        eps: float,
        weight_decay: float,
        decoupled_decay: bool,
        max_lr: Optional[float],
        trust_clip: bool,
        always_adapt: bool,
        caution: bool,
) -> None:
    if len(params) == 0:
        return

    if clip_grad_norm is not None:
        # not in-place, leave p.grad unmodified
        if _HAS_FOREACH_DIV_TENSOR:
            grads = torch._foreach_div(grads, clip_grad_norm)
        else:
            grads = [grad / clip_grad_norm for grad in grads]

    # Decay the first and second moment running average coefficient
    if beta3 == 1 - beta1:
        _foreach_lerp_(exp_avgs, grads, beta3)  # m_t, grad averaging
    else:
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=beta3)  # m_t
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)  # v_t

    bc2_sqrt = bias_correction2 ** 0.5
    denom = torch._foreach_sqrt(exp_avg_sqs)
    if torch.is_tensor(bias_correction1) or any(p.dtype == torch.float16 for p in params):
        # Bias correct the tensors, tensor bias corrections (compiling) can't be folded into Python scalars,
        # Inductor fuses these ops. In float16, eps * sqrt(bc2) can underflow at early steps.
        torch._foreach_div_(denom, bc2_sqrt)
        torch._foreach_add_(denom, eps)
        updates = torch._foreach_div(exp_avgs, bias_correction1)
        torch._foreach_div_(updates, denom)
        scale = 1.
    else:
        # Bias corrections folded into scalars to save full passes over the tensors,
        # m_hat / (sqrt(v_hat) + eps) == scale * m / (sqrt(v) + eps * sqrt(bc2)) w/ scale = sqrt(bc2) / bc1.
        # The unscaled update is used below, scale is applied to the weight decay, update norm, and step size.
        torch._foreach_add_(denom, eps * bc2_sqrt)
        updates = torch._foreach_div(exp_avgs, denom)
        scale = bc2_sqrt / bias_correction1

    if caution:
        # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
        masks = torch._foreach_mul(updates, grads)
        masks = [(m > 0).to(g.dtype) for m, g in zip(masks, grads)]
        mask_scale = [m.mean() for m in masks]
        torch._foreach_maximum_(mask_scale, 1e-3)
        torch._foreach_div_(masks, mask_scale)
        torch._foreach_mul_(updates, masks)

    if weight_decay != 0:
        if decoupled_decay:
            wd_scale = lr if max_lr is None else lr ** 2 / max_lr
            if torch.is_tensor(wd_scale):
                torch._foreach_add_(params, torch._foreach_mul(params, -wd_scale * weight_decay))
            else:
                torch._foreach_add_(params, params, alpha=-wd_scale * weight_decay)
        else:
            torch._foreach_add_(updates, params, alpha=weight_decay / scale)

    if weight_decay != 0 or always_adapt:
        # Layer-wise LR adaptation, see single tensor impl. The trust ratios are combined w/ the step size so the
        # update is applied in one op, foreach has no fast path for per tensor (0-dim) scalars either way.
        step_sizes = _foreach_trust_ratios(
            torch._foreach_norm(params),
            torch._foreach_norm(updates),
            lambda w, g: _lamb_trust_ratio(w, g * scale, trust_clip) * (-lr * scale),
        )
        torch._foreach_addcmul_(params, updates, step_sizes)
    elif torch.is_tensor(lr):
        torch._foreach_mul_(updates, -lr * scale)
        torch._foreach_add_(params, updates)
    else:
        torch._foreach_add_(params, updates, alpha=-lr * scale)
