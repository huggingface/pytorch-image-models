""" RMSProp modified to behave like Tensorflow impl

Originally cut & paste from PyTorch RMSProp
https://github.com/pytorch/pytorch/blob/063946d2b3f3f1e953a2a3b54e0b34f1393de295/torch/optim/rmsprop.py
Licensed under BSD-Clause 3 (ish), https://github.com/pytorch/pytorch/blob/master/LICENSE

References for added functionality:
    Cautious Optimizers: https://arxiv.org/abs/2411.16085
    Why Gradients Rapidly Increase Near the End of Training: https://arxiv.org/abs/2506.02285

Modifications Copyright 2021 Ross Wightman
"""

import torch
from torch.optim import Optimizer

from ._types import ParamsT


class RMSpropTF(Optimizer):
    """Implements RMSprop algorithm (TensorFlow style epsilon)

    NOTE: This is a direct cut-and-paste of PyTorch RMSprop with eps applied before sqrt
    and a few other modifications to closer match Tensorflow for matching hyper-params.

    Noteworthy changes include:
    1. Epsilon applied inside square-root
    2. square_avg initialized to ones
    3. LR scaling of update accumulated in momentum buffer

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate
        momentum: momentum factor
        alpha: smoothing (decay) constant
        eps: term added to the denominator to improve numerical stability
        centered: if ``True``, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay: weight decay (L2 penalty) (default: 0)
        decoupled_decay: decoupled weight decay as per https://arxiv.org/abs/1711.05101
        corrected_weight_decay: apply corrected weight decay (lr**2 / max_lr) when decoupled_decay is True
        lr_in_momentum: learning rate scaling is included in the momentum buffer update as per defaults in Tensorflow
        caution: apply caution
    """

    def __init__(
            self,
            params: ParamsT,
            lr: float = 1e-2,
            alpha: float = 0.9,
            eps: float = 1e-10,
            weight_decay: float = 0,
            momentum: float = 0.,
            centered: bool = False,
            decoupled_decay: bool = False,
            corrected_weight_decay: bool = False,
            lr_in_momentum: bool = True,
            caution: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
            decoupled_decay=decoupled_decay,
            corrected_weight_decay=corrected_weight_decay,
            lr_in_momentum=lr_in_momentum,
            caution=caution,
        )
        super(RMSpropTF, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSpropTF, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)
            group.setdefault('caution', False)
            group.setdefault('corrected_weight_decay', False)

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
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(p)  # PyTorch inits to zero
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if group['decoupled_decay']:
                        if group['corrected_weight_decay']:
                            wd_scale = group['lr'] ** 2 / self.defaults['lr']
                        else:
                            wd_scale = group['lr']
                        p.mul_(1. - wd_scale * group['weight_decay'])
                    else:
                        grad = grad.add(p, alpha=group['weight_decay'])

                # Tensorflow order of ops for updating squared avg
                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)
                # square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)  # PyTorch original

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(group['eps']).sqrt_()  # eps in sqrt
                    # grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)  # PyTorch original
                else:
                    avg = square_avg.add(group['eps']).sqrt_()  # eps moved in sqrt

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum'])

                    def _apply_caution(_m, _g):
                        # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                        mask = (_m * _g > 0).to(_g.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        return _m * mask

                    if group['lr_in_momentum']:
                        # Tensorflow accumulates the LR scaling in the momentum buffer
                        buf.addcdiv_(grad, avg, value=group['lr'])
                        if group['caution']:
                            buf = _apply_caution(buf, grad)
                        p.add_(-buf)
                    else:
                        # PyTorch scales the param update by LR
                        buf.addcdiv_(grad, avg)
                        if group['caution']:
                            buf = _apply_caution(buf, grad)
                        p.add_(buf, alpha=-group['lr'])
                else:
                    p.addcdiv_(grad, avg, value=-group['lr'])

        return loss
