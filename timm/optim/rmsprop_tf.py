""" RMSProp modified to behave like Tensorflow impl

Originally cut & paste from PyTorch RMSProp
https://github.com/pytorch/pytorch/blob/063946d2b3f3f1e953a2a3b54e0b34f1393de295/torch/optim/rmsprop.py
Licensed under BSD-Clause 3 (ish), https://github.com/pytorch/pytorch/blob/master/LICENSE

References for added functionality:
    Cautious Optimizers: https://arxiv.org/abs/2411.16085
    Why Gradients Rapidly Increase Near the End of Training: https://arxiv.org/abs/2506.02285

Modifications Copyright 2021 Ross Wightman
"""

from typing import List, Optional, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

from ._helpers import (
    _add_scaled_, _addcdiv_scaled_, _foreach_chunked, _foreach_increment_steps, _init_scalar, _max_lr_snapshot,
    _resolve_foreach, _validate_scalar,
)
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
        foreach: use the foreach (multi-tensor) impl, by default (None) used unless unsupported (see rmsprop_tf())
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
            foreach: Optional[bool] = None,
    ):
        _validate_scalar("learning rate", lr)
        _validate_scalar("epsilon", eps)
        _validate_scalar("momentum", momentum)
        _validate_scalar("weight_decay", weight_decay)
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
            max_lr_snapshot=_max_lr_snapshot(lr, corrected_weight_decay),
            lr_in_momentum=lr_in_momentum,
            caution=caution,
            foreach=foreach,
        )
        super(RMSpropTF, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSpropTF, self).__setstate__(state)
        self.defaults.setdefault('max_lr_snapshot', _max_lr_snapshot(self.defaults['lr']))  # pickled pre-snapshot
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)
            group.setdefault('caution', False)
            group.setdefault('corrected_weight_decay', False)
            group.setdefault('max_lr_snapshot', self.defaults['max_lr_snapshot'])
            group.setdefault('foreach', None)
            for p in group['params']:
                p_state = self.state.get(p, {})
                if p_state and 'step' in p_state:
                    p_state['step'] = _init_scalar(p_state['step'], device='cpu')

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
            params = []
            grads = []
            square_avgs = []
            grad_avgs = []
            momentum_buffers = []
            state_steps = []

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = _init_scalar(device='cpu')
                    state['square_avg'] = torch.ones_like(p)  # PyTorch inits to zero
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p)

                params.append(p)
                grads.append(p.grad)
                square_avgs.append(state['square_avg'])
                if group['centered']:
                    grad_avgs.append(state['grad_avg'])
                if group['momentum'] > 0:
                    momentum_buffers.append(state['momentum_buffer'])
                state_steps.append(state['step'])

            rmsprop_tf(
                params,
                grads,
                square_avgs,
                grad_avgs,
                momentum_buffers,
                state_steps,
                foreach=group['foreach'],
                lr=group['lr'],
                alpha=group['alpha'],
                eps=group['eps'],
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                centered=group['centered'],
                decoupled_decay=group['decoupled_decay'],
                max_lr=group['max_lr_snapshot'] if group['corrected_weight_decay'] else None,
                lr_in_momentum=group['lr_in_momentum'],
                caution=group['caution'],
            )

        return loss


def rmsprop_tf(
        params: List[Tensor],
        grads: List[Tensor],
        square_avgs: List[Tensor],
        grad_avgs: List[Tensor],
        momentum_buffers: List[Tensor],
        state_steps: List[Tensor],
        foreach: Optional[bool] = None,
        *,
        lr: Union[float, Tensor],
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
        centered: bool,
        decoupled_decay: bool,
        max_lr: Optional[float],
        lr_in_momentum: bool,
        caution: bool,
) -> None:
    """Functional API that performs the RMSpropTF algorithm computation. See RMSpropTF class for details."""
    if _resolve_foreach(foreach, caution, lr, params) and not torch.jit.is_scripting():
        func = _multi_tensor_rmsprop_tf
    else:
        func = _single_tensor_rmsprop_tf

    func(
        params,
        grads,
        square_avgs,
        grad_avgs,
        momentum_buffers,
        state_steps,
        lr=lr,
        alpha=alpha,
        eps=eps,
        weight_decay=weight_decay,
        momentum=momentum,
        centered=centered,
        decoupled_decay=decoupled_decay,
        max_lr=max_lr,
        lr_in_momentum=lr_in_momentum,
        caution=caution,
    )


def _single_tensor_rmsprop_tf(
        params: List[Tensor],
        grads: List[Tensor],
        square_avgs: List[Tensor],
        grad_avgs: List[Tensor],
        momentum_buffers: List[Tensor],
        state_steps: List[Tensor],
        *,
        lr: Union[float, Tensor],
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
        centered: bool,
        decoupled_decay: bool,
        max_lr: Optional[float],
        lr_in_momentum: bool,
        caution: bool,
) -> None:
    one_minus_alpha = 1. - alpha

    def _apply_caution(_m, _g):
        # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
        mask = (_m * _g > 0).to(_g.dtype)
        mask.div_(mask.mean().clamp_(min=1e-3))
        return _m * mask

    for i, param in enumerate(params):
        grad = grads[i]
        square_avg = square_avgs[i]

        state_steps[i].add_(1)

        if weight_decay != 0:
            if decoupled_decay:
                wd_scale = lr if max_lr is None else lr ** 2 / max_lr
                param.mul_(1. - wd_scale * weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)

        # Tensorflow order of ops for updating squared avg
        square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)
        # square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)  # PyTorch original

        if centered:
            grad_avg = grad_avgs[i]
            grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
            avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(eps).sqrt_()  # eps in sqrt
            # grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)  # PyTorch original
        else:
            avg = square_avg.add(eps).sqrt_()  # eps moved in sqrt

        if momentum > 0:
            buf = momentum_buffers[i]
            buf.mul_(momentum)
            if lr_in_momentum:
                # Tensorflow accumulates the LR scaling in the momentum buffer
                _addcdiv_scaled_(buf, grad, avg, lr)
                if caution:
                    buf = _apply_caution(buf, grad)
                param.add_(-buf)
            else:
                # PyTorch scales the param update by LR
                buf.addcdiv_(grad, avg)
                if caution:
                    buf = _apply_caution(buf, grad)
                _add_scaled_(param, buf, -lr)
        else:
            _addcdiv_scaled_(param, grad, avg, -lr)


def _foreach_apply_caution(exp_avgs: List[Tensor], grads: List[Tensor]) -> List[Tensor]:
    # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
    masks = torch._foreach_mul(exp_avgs, grads)
    masks = [(m > 0).to(g.dtype) for m, g in zip(masks, grads)]
    mask_scale = [m.mean() for m in masks]
    torch._foreach_maximum_(mask_scale, 1e-3)
    torch._foreach_div_(masks, mask_scale)
    return torch._foreach_mul(exp_avgs, masks)


@_foreach_chunked(6)
def _multi_tensor_rmsprop_tf(
        params: List[Tensor],
        grads: List[Tensor],
        square_avgs: List[Tensor],
        grad_avgs: List[Tensor],
        momentum_buffers: List[Tensor],
        state_steps: List[Tensor],
        *,
        lr: Union[float, Tensor],
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
        centered: bool,
        decoupled_decay: bool,
        max_lr: Optional[float],
        lr_in_momentum: bool,
        caution: bool,
) -> None:
    if len(params) == 0:
        return

    one_minus_alpha = 1. - alpha
    _foreach_increment_steps(state_steps)

    if weight_decay != 0:
        if decoupled_decay:
            wd_scale = lr if max_lr is None else lr ** 2 / max_lr
            torch._foreach_mul_(params, 1. - wd_scale * weight_decay)
        else:
            grads = torch._foreach_add(grads, params, alpha=weight_decay)

    # Tensorflow order of ops for updating squared avg
    grad_sq = torch._foreach_mul(grads, grads)
    torch._foreach_sub_(grad_sq, square_avgs)
    torch._foreach_add_(square_avgs, grad_sq, alpha=one_minus_alpha)

    if centered:
        grad_diff = torch._foreach_sub(grads, grad_avgs)
        torch._foreach_add_(grad_avgs, grad_diff, alpha=one_minus_alpha)
        avg = torch._foreach_addcmul(square_avgs, grad_avgs, grad_avgs, value=-1)
        torch._foreach_add_(avg, eps)
    else:
        avg = torch._foreach_add(square_avgs, eps)
    torch._foreach_sqrt_(avg)  # eps in sqrt

    def _foreach_addcdiv_scaled_(dst, tensor1, tensor2, scale):
        if torch.is_tensor(scale):
            update = torch._foreach_div(tensor1, tensor2)
            torch._foreach_mul_(update, scale)
            torch._foreach_add_(dst, update)
        else:
            torch._foreach_addcdiv_(dst, tensor1, tensor2, value=scale)

    if momentum > 0:
        torch._foreach_mul_(momentum_buffers, momentum)
        if lr_in_momentum:
            # Tensorflow accumulates the LR scaling in the momentum buffer
            _foreach_addcdiv_scaled_(momentum_buffers, grads, avg, lr)
            bufs = _foreach_apply_caution(momentum_buffers, grads) if caution else momentum_buffers
            torch._foreach_sub_(params, bufs)
        else:
            # PyTorch scales the param update by LR
            torch._foreach_addcdiv_(momentum_buffers, grads, avg)
            bufs = _foreach_apply_caution(momentum_buffers, grads) if caution else momentum_buffers
            if torch.is_tensor(lr):
                torch._foreach_add_(params, torch._foreach_mul(bufs, -lr))
            else:
                torch._foreach_add_(params, bufs, alpha=-lr)
    else:
        _foreach_addcdiv_scaled_(params, grads, avg, -lr)
