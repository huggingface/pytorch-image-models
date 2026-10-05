""" PyTorch LARS / LARC Optimizer

An implementation of LARS (SGD) + LARC in PyTorch

Based on:
  * PyTorch SGD: https://github.com/pytorch/pytorch/blob/1.7/torch/optim/sgd.py#L100
  * NVIDIA APEX LARC: https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py

Additional cleanup and modifications to properly support PyTorch XLA.

Copyright 2021 Ross Wightman
"""
from typing import List, Optional, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ._helpers import _foreach_trust_ratios, _resolve_foreach


class Lars(Optimizer):
    """ LARS for PyTorch
    
    Paper: `Large batch training of Convolutional Networks` - https://arxiv.org/pdf/1708.03888.pdf

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate (default: 1.0).
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        trust_coeff (float): trust coefficient for computing adaptive lr / trust_ratio (default: 0.001)
        eps (float): eps for division denominator (default: 1e-8)
        trust_clip (bool): enable LARC trust ratio clipping (default: False)
        always_adapt (bool): always apply LARS LR adapt, otherwise only when group weight_decay != 0 (default: False)
        foreach (bool, optional): use the foreach (multi-tensor) impl, by default (None) used unless unsupported
    """

    def __init__(
        self,
        params,
        lr=1.0,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        trust_coeff=0.001,
        eps=1e-8,
        trust_clip=False,
        always_adapt=False,
        foreach: Optional[bool] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            trust_coeff=trust_coeff,
            eps=eps,
            trust_clip=trust_clip,
            always_adapt=always_adapt,
            foreach=foreach,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("foreach", None)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = []
            grads = []
            momentum_buffers = []
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)
                grads.append(p.grad)
                if group['momentum'] != 0:
                    momentum_buffers.append(self.state[p].get('momentum_buffer'))

            lars(
                params,
                grads,
                momentum_buffers,
                foreach=group['foreach'],
                lr=group['lr'],
                momentum=group['momentum'],
                dampening=group['dampening'],
                weight_decay=group['weight_decay'],
                nesterov=group['nesterov'],
                trust_coeff=group['trust_coeff'],
                eps=group['eps'],
                trust_clip=group['trust_clip'],
                always_adapt=group['always_adapt'],
            )

            # momentum buffers are created on the first step
            if group['momentum'] != 0:
                for p, buf in zip(params, momentum_buffers):
                    self.state[p]['momentum_buffer'] = buf

        return loss


def lars(
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffers: List[Optional[Tensor]],
        foreach: Optional[bool] = None,
        *,
        lr: Union[float, Tensor],
        momentum: float,
        dampening: float,
        weight_decay: float,
        nesterov: bool,
        trust_coeff: float,
        eps: float,
        trust_clip: bool,
        always_adapt: bool,
) -> None:
    """Functional API that performs the LARS / LARC update, see Lars class for details. New momentum buffers are
    created in place in the momentum_buffers list.
    """
    if _resolve_foreach(foreach, lr=lr, params=params) and not torch.jit.is_scripting():
        func = _multi_tensor_lars
    else:
        func = _single_tensor_lars
    func(
        params,
        grads,
        momentum_buffers,
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
        trust_coeff=trust_coeff,
        eps=eps,
        trust_clip=trust_clip,
        always_adapt=always_adapt,
    )


def _lars_trust_ratio(
        w_norm: Tensor,
        g_norm: Tensor,
        trust_coeff: float,
        weight_decay: float,
        eps: float,
        trust_clip: bool,
        lr: Union[float, Tensor],
) -> Tensor:
    trust_ratio = trust_coeff * w_norm / (g_norm + w_norm * weight_decay + eps)
    # FIXME nested where required since logical and/or not working in PT XLA
    # Set the ratio to 1.0 (no change) if either weight norm or grad norm is zero
    trust_ratio = torch.where(
        w_norm > 0,
        torch.where(g_norm > 0, trust_ratio, 1.0),
        1.0,
    )
    if trust_clip:
        trust_ratio = torch.clamp(trust_ratio / lr, max=1.0)
    return trust_ratio


def _single_tensor_lars(
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffers: List[Optional[Tensor]],
        *,
        lr: Union[float, Tensor],
        momentum: float,
        dampening: float,
        weight_decay: float,
        nesterov: bool,
        trust_coeff: float,
        eps: float,
        trust_clip: bool,
        always_adapt: bool,
) -> None:
    for i, p in enumerate(params):
        grad = grads[i]

        # apply LARS LR adaptation, LARC clipping, weight decay
        # ref: https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
        if weight_decay != 0 or always_adapt:
            trust_ratio = _lars_trust_ratio(
                p.norm(2.0), grad.norm(2.0), trust_coeff, weight_decay, eps, trust_clip, lr)
            grad = grad.add(p, alpha=weight_decay).mul_(trust_ratio)  # not in-place, leave p.grad unmodified

        # apply SGD update https://github.com/pytorch/pytorch/blob/1.7/torch/optim/sgd.py#L100
        if momentum != 0:
            buf = momentum_buffers[i]
            if buf is None:
                buf = momentum_buffers[i] = torch.clone(grad).detach()
            else:
                buf.mul_(momentum).add_(grad, alpha=1. - dampening)
            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        p.add_(grad, alpha=-lr)


def _multi_tensor_lars(
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffers: List[Optional[Tensor]],
        *,
        lr: Union[float, Tensor],
        momentum: float,
        dampening: float,
        weight_decay: float,
        nesterov: bool,
        trust_coeff: float,
        eps: float,
        trust_clip: bool,
        always_adapt: bool,
) -> None:
    if len(params) == 0:
        return

    # apply LARS LR adaptation, LARC clipping, weight decay
    # ref: https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
    if weight_decay != 0 or always_adapt:
        trust_ratios = _foreach_trust_ratios(
            torch._foreach_norm(params),
            torch._foreach_norm(grads),
            lambda w, g: _lars_trust_ratio(w, g, trust_coeff, weight_decay, eps, trust_clip, lr),
        )
        grads = torch._foreach_add(grads, params, alpha=weight_decay)  # not in-place, leave p.grad unmodified
        torch._foreach_mul_(grads, trust_ratios)

    # apply SGD update https://github.com/pytorch/pytorch/blob/1.7/torch/optim/sgd.py#L100
    if momentum != 0:
        existing = [i for i, buf in enumerate(momentum_buffers) if buf is not None]
        if existing:
            bufs = [momentum_buffers[i] for i in existing]
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, [grads[i] for i in existing], alpha=1. - dampening)
        for i, buf in enumerate(momentum_buffers):
            if buf is None:
                momentum_buffers[i] = torch.clone(grads[i]).detach()
        if nesterov:
            grads = torch._foreach_add(grads, momentum_buffers, alpha=momentum)
        else:
            grads = momentum_buffers

    if torch.is_tensor(lr):
        torch._foreach_add_(params, torch._foreach_mul(grads, -lr))
    else:
        torch._foreach_add_(params, grads, alpha=-lr)
