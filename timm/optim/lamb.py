""" PyTorch Lamb optimizer w/ behaviour similar to NVIDIA FusedLamb

This optimizer code was adapted from the following (starting with latest)
* https://github.com/HabanaAI/Model-References/blob/2b435114fe8e31f159b1d3063b8280ae37af7423/PyTorch/nlp/bert/pretraining/lamb.py
* https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py
* https://github.com/cybertronai/pytorch-lamb

Use FusedLamb if you can (GPU). The reason for including this variant of Lamb is to have a version that is
similar in behaviour to APEX FusedLamb if you aren't using NVIDIA GPUs or cannot install/use APEX.

In addition to some cleanup, this Lamb impl has been modified to support PyTorch XLA and has been tested on TPU.

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
import math
from typing import Optional, Tuple

import torch
from torch.optim import Optimizer

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
    ):
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
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('caution', False)
            group.setdefault('decoupled_decay', False)

    def _get_clip_grad_norm(self):
        max_grad_norm = self.defaults['max_grad_norm']
        if max_grad_norm is None:
            return None

        norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instead.')
                norms.append(torch.linalg.vector_norm(grad))
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
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            if bias_correction:
                bias_correction1 = 1 - beta1 ** group['step']
                bias_correction2 = 1 - beta2 ** group['step']
            else:
                bias_correction1, bias_correction2 = 1.0, 1.0

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if clip_grad_norm is not None:
                    grad.div_(clip_grad_norm)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient valuesa
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=beta3)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                update = (exp_avg / bias_correction1).div_(denom)

                if group['caution']:
                    # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                    mask = (update * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    update.mul_(mask)

                weight_decay = group['weight_decay']
                if weight_decay != 0:
                    if group.get('decoupled_decay', False):
                        p.add_(p, alpha=-group['lr'] * weight_decay)
                    else:
                        update.add_(p, alpha=weight_decay)

                if weight_decay != 0 or group['always_adapt']:
                    # Layer-wise LR adaptation. By default, skip adaptation on parameters that are
                    # excluded from weight decay, unless always_adapt == True, then always enabled.
                    w_norm = p.norm(2.0)
                    g_norm = update.norm(2.0)
                    trust_ratio = w_norm / g_norm
                    # FIXME nested where required since logical and/or not working in PT XLA
                    # Set the ratio to 1.0 (no change) if either weight norm or grad norm is zero
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, trust_ratio, 1.0),
                        1.0,
                    )
                    if group['trust_clip']:
                        # LAMBC trust clipping, upper bound fixed at one
                        trust_ratio = torch.clamp(trust_ratio, max=1.0)
                    update.mul_(trust_ratio)

                p.add_(update, alpha=-group['lr'])

        return loss
