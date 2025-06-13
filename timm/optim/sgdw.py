""" SGD with decoupled weight-decay.

References for added functionality:
    Cautious Optimizers: https://arxiv.org/abs/2411.16085
    Why Gradients Rapidly Increase Near the End of Training: https://arxiv.org/abs/2506.02285

Hacked together by Ross Wightman
"""
from typing import List, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
try:
    from torch.optim.optimizer import _use_grad_for_differentiable, _default_to_fused_or_foreach
    has_recent_pt = True
except ImportError:
    has_recent_pt = False

from ._types import ParamsT

__all__ = ['SGDW', 'sgdw']


class SGDW(Optimizer):
    def __init__(
            self,
            params: ParamsT,
            lr: float = 1e-3,
            momentum: float = 0.,
            dampening: float = 0.,
            weight_decay: float = 0.,
            nesterov: bool = False,
            *,
            caution: bool = False,
            corrected_weight_decay: bool = False,
            maximize: bool = False,
            foreach: Optional[bool] = None,
            differentiable: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            caution=caution,
            corrected_weight_decay=corrected_weight_decay,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('caution', False)
            group.setdefault('corrected_weight_decay', False)
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    def _init_group(self, group, params_with_grad, grads, momentum_buffer_list):
        has_sparse_grad = False

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

        return has_sparse_grad

    # FIXME figure out how to make _use_grad_for_differentiable interchangeable with no_grad decorator
    #   without args, for backwards compatibility with old pytorch
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            momentum_buffer_list = []

            has_sparse_grad = self._init_group(group, params_with_grad, grads, momentum_buffer_list)

            sgdw(
                params_with_grad,
                grads,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                caution=group['caution'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'],
                max_lr=self.defaults['lr'] if group['corrected_weight_decay'] else None,
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def sgdw(
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: Optional[bool] = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        caution: bool,
        maximize: bool,
        max_lr: Optional[float] = None
):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """
    if has_recent_pt and hasattr(Optimizer, '_group_tensors_by_device_and_dtype'):
        if foreach is None:
            # why must we be explicit about an if statement for torch.jit.is_scripting here?
            # because JIT can't handle Optionals nor fancy conditionals when scripting
            if not torch.jit.is_scripting():
                _, foreach = _default_to_fused_or_foreach(params, differentiable=False, use_fused=False)
            else:
                foreach = False

        if foreach and torch.jit.is_scripting():
            raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    else:
        foreach = False  # disabling altogether for older pytorch, as using _group_tensors_by_device_and_dtype

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgdw
    else:
        func = _single_tensor_sgdw

    func(
        params,
        grads,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        caution=caution,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        max_lr=max_lr,
    )


def _single_tensor_sgdw(
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        caution: bool,
        maximize: bool,
        has_sparse_grad: bool,
        max_lr: Optional[float]
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]

        wd_scale = lr if max_lr is None else lr ** 2 / max_lr
        param.mul_(1. - wd_scale * weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if caution:
                if nesterov:
                    buf = grad.add(buf, alpha=momentum)
                # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                mask = (buf * grad > 0).to(grad.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                grad = buf * mask
            else:
                if nesterov:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf

        param.add_(grad, alpha=-lr)


def _multi_tensor_sgdw(
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        caution: bool,
        maximize: bool,
        has_sparse_grad: bool,
        max_lr: Optional[float]
):
    if len(params) == 0:
        return

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list], with_indices=True)
    for ((device_params, device_grads, device_momentum_buffer_list), indices) in grouped_tensors.values():
        device_has_sparse_grad = has_sparse_grad and any(grad.is_sparse for grad in device_grads)

        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        wd_scale = lr if max_lr is None else lr ** 2 / max_lr
        torch._foreach_mul_(params, 1. - wd_scale * weight_decay)

        if momentum != 0:
            bufs = []

            all_states_with_momentum_buffer = True
            for i in range(len(device_momentum_buffer_list)):
                if device_momentum_buffer_list[i] is None:
                    all_states_with_momentum_buffer = False
                    break
                else:
                    bufs.append(device_momentum_buffer_list[i])

            if all_states_with_momentum_buffer:
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
            else:
                bufs = []
                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        buf = device_momentum_buffer_list[i] = momentum_buffer_list[indices[i]] = \
                            torch.clone(device_grads[i]).detach()
                    else:
                        buf = device_momentum_buffer_list[i]
                        buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)

                    bufs.append(buf)

            if caution:
                if nesterov:
                    # Can't do nesterov in-place if we want to compare against orig grad for caution
                    bufs = torch._foreach_add(device_grads, bufs, alpha=momentum)
                # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                masks = torch._foreach_mul(bufs, device_grads)
                masks = [(m > 0).to(g.dtype) for m, g in zip(masks, device_grads)]
                mask_scale = [m.mean() for m in masks]
                torch._foreach_maximum_(mask_scale, 1e-3)
                torch._foreach_div_(masks, mask_scale)
                device_grads = torch._foreach_mul(bufs, masks)
            else:
                if nesterov:
                    torch._foreach_add_(device_grads, bufs, alpha=momentum)
                else:
                    device_grads = bufs

        if not device_has_sparse_grad:
            torch._foreach_add_(device_params, device_grads, alpha=-lr)
        else:
            # foreach APIs don't support sparse
            for i in range(len(device_params)):
                device_params[i].add_(device_grads[i], alpha=-lr)
