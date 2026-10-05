""" PyTorch impl of LaProp optimizer

Code simplified from https://github.com/Z-T-WANG/LaProp-Optimizer, MIT License

Paper: LaProp: Separating Momentum and Adaptivity in Adam, https://arxiv.org/abs/2002.04839

@article{ziyin2020laprop,
  title={LaProp: a Better Way to Combine Momentum with Adaptive Gradient},
  author={Ziyin, Liu and Wang, Zhikang T and Ueda, Masahito},
  journal={arXiv preprint arXiv:2002.04839},
  year={2020}
}

References for added functionality:
    Cautious Optimizers: https://arxiv.org/abs/2411.16085
    Why Gradients Rapidly Increase Near the End of Training: https://arxiv.org/abs/2506.02285

"""
from typing import Any, Dict, List, Optional, Tuple, Union

from torch.optim import Optimizer
import torch
from torch import Tensor

from ._helpers import (
    _add_scaled_, _foreach_chunked, _get_scalar_dtype, _get_value, _init_scalar, _load_state_dict_preserving_dtypes,
    _max_lr_snapshot, _resolve_foreach, _validate_scalar,
)
from ._types import ParamsT


class LaProp(Optimizer):
    """ LaProp Optimizer

    Paper: LaProp: Separating Momentum and Adaptivity in Adam, https://arxiv.org/abs/2002.04839

    Args:
        foreach: use the foreach (multi-tensor) impl, by default (None) used unless unsupported (see laprop())
    """
    def __init__(
            self,
            params: ParamsT,
            lr: float = 4e-4,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-15,
            weight_decay: float = 0.,
            caution: bool = False,
            corrected_weight_decay: bool = False,
            foreach: Optional[bool] = None,
    ):
        _validate_scalar("learning rate", lr)
        _validate_scalar("epsilon", eps)
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            caution=caution,
            corrected_weight_decay=corrected_weight_decay,
            max_lr_snapshot=_max_lr_snapshot(lr, corrected_weight_decay),
            foreach=foreach,
        )
        super(LaProp, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.defaults.setdefault('max_lr_snapshot', _max_lr_snapshot(self.defaults['lr']))  # pickled pre-snapshot
        for group in self.param_groups:
            group.setdefault('caution', False)
            group.setdefault('corrected_weight_decay', False)
            group.setdefault('max_lr_snapshot', self.defaults['max_lr_snapshot'])
            group.setdefault('foreach', None)
            for p in group['params']:
                p_state = self.state.get(p, {})
                if not p_state:
                    continue
                if 'step' in p_state:
                    p_state['step'] = _init_scalar(p_state['step'], device='cpu')
                if 'exp_avg_lr_1' in p_state and torch.is_tensor(group['lr']):
                    p_state['exp_avg_lr_1'] = _init_scalar(
                        p_state['exp_avg_lr_1'],
                        dtype=group['lr'].dtype,
                        device=group['lr'].device,
                    )
                if 'exp_avg_lr_2' in p_state:
                    # Optimizer.load_state_dict casts state to the param dtype, restore full precision
                    p_state['exp_avg_lr_2'] = _init_scalar(
                        p_state['exp_avg_lr_2'],
                        device='cpu',
                        dtype=_get_scalar_dtype(),
                    )

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        _load_state_dict_preserving_dtypes(
            self, state_dict, super().load_state_dict,
            lambda group, param: {
                'exp_avg_lr_1': (group['lr'].dtype, group['lr'].device) if torch.is_tensor(group['lr']) else None,
                'exp_avg_lr_2': (_get_scalar_dtype(), torch.device('cpu')),
            },
        )

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
            exp_avgs = []
            exp_avg_sqs = []
            step_sizes = []
            bias_corrections2 = []
            beta1, beta2 = group['betas']
            one_minus_beta1 = 1 - beta1
            one_minus_beta2 = 1 - beta2

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('LaProp does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = _init_scalar(device='cpu')
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of learning rates
                    state['exp_avg_lr_1'] = torch.zeros_like(group['lr']) if torch.is_tensor(group['lr']) else 0.
                    state['exp_avg_lr_2'] = _init_scalar(device='cpu')
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                # Per param scalar state, params of a group can be at different steps
                state['step'].add_(1)
                state['exp_avg_lr_1'] = state['exp_avg_lr_1'] * beta1 + one_minus_beta1 * group['lr']
                state['exp_avg_lr_2'] = state['exp_avg_lr_2'] * beta2 + one_minus_beta2

                # step_size = 1 / bias_correction1, w/ bias_correction1 = exp_avg_lr_1 / lr (1 - beta1 ** step
                # for a constant lr). Computed as lr / exp_avg_lr_1 so that step_size is 0 when lr == 0.
                if torch.is_tensor(state['exp_avg_lr_1']):
                    exp_avg_lr_1 = state['exp_avg_lr_1']
                    exp_avg_lr_1_safe = torch.where(exp_avg_lr_1 != 0., exp_avg_lr_1, torch.ones_like(exp_avg_lr_1))
                    step_size = torch.where(
                        exp_avg_lr_1 != 0.,
                        group['lr'] / exp_avg_lr_1_safe,
                        torch.zeros_like(exp_avg_lr_1),
                    )
                else:
                    step_size = group['lr'] / state['exp_avg_lr_1'] if state['exp_avg_lr_1'] != 0. else 0.

                params.append(p)
                grads.append(p.grad)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                step_sizes.append(step_size)
                bias_corrections2.append(state['exp_avg_lr_2'])

            laprop(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                step_sizes,
                bias_corrections2,
                foreach=group['foreach'],
                lr=group['lr'],
                beta1=beta1,
                beta2=beta2,
                eps=group['eps'],
                weight_decay=group['weight_decay'],
                max_lr=group['max_lr_snapshot'] if group['corrected_weight_decay'] else None,
                caution=group['caution'],
            )

        return loss


def laprop(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        step_sizes: List[Union[float, Tensor]],
        bias_corrections2: List[Tensor],
        foreach: Optional[bool] = None,
        *,
        lr: Union[float, Tensor],
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        max_lr: Optional[float],
        caution: bool,
) -> None:
    """Functional API that performs the LaProp tensor updates, per param step sizes and second moment bias
    corrections are computed by the LaProp class (see for details).
    """
    if _resolve_foreach(foreach, caution, lr, params) and not torch.jit.is_scripting():
        func = _multi_tensor_laprop
    else:
        func = _single_tensor_laprop

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        step_sizes,
        bias_corrections2,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        max_lr=max_lr,
        caution=caution,
    )


def _single_tensor_laprop(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        step_sizes: List[Union[float, Tensor]],
        bias_corrections2: List[Tensor],
        *,
        lr: Union[float, Tensor],
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        max_lr: Optional[float],
        caution: bool,
) -> None:
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)

        # Decay the first and second moment running average coefficient
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        denom = exp_avg_sq.div(bias_corrections2[i]).sqrt_().add_(eps)
        step_of_this_grad = grad / denom
        exp_avg.mul_(beta1)
        _add_scaled_(exp_avg, step_of_this_grad, lr * (1 - beta1))

        if caution:
            # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
            mask = (exp_avg * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            exp_avg = exp_avg * mask

        _add_scaled_(param, exp_avg, -step_sizes[i])

        if weight_decay != 0:
            wd_scale = lr if max_lr is None else lr ** 2 / max_lr
            _add_scaled_(param, param, -wd_scale * weight_decay)


@_foreach_chunked(6)
def _multi_tensor_laprop(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        step_sizes: List[Union[float, Tensor]],
        bias_corrections2: List[Tensor],
        *,
        lr: Union[float, Tensor],
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        max_lr: Optional[float],
        caution: bool,
) -> None:
    if len(params) == 0:
        return

    grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grads]
    exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avgs]
    exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avg_sqs]
    params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in params]

    def _foreach_add_scaled_(dst, update, scale, inplace_update=False):
        if torch.is_tensor(scale):
            if inplace_update:
                torch._foreach_mul_(update, scale)
            else:
                update = torch._foreach_mul(update, scale)
            torch._foreach_add_(dst, update)
        else:
            torch._foreach_add_(dst, update, alpha=scale)

    # Decay the first and second moment running average coefficient
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

    # second moment bias corrections are CPU scalars, apply as python scalars (no device mismatch / sync)
    denom = torch._foreach_div(exp_avg_sqs, [_get_value(bc) for bc in bias_corrections2])
    torch._foreach_sqrt_(denom)
    torch._foreach_add_(denom, eps)
    step_of_this_grad = torch._foreach_div(grads, denom)
    torch._foreach_mul_(exp_avgs, beta1)
    _foreach_add_scaled_(exp_avgs, step_of_this_grad, lr * (1 - beta1), inplace_update=True)

    if caution:
        # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
        masks = torch._foreach_mul(exp_avgs, grads)
        masks = [(m > 0).to(g.dtype) for m, g in zip(masks, grads)]
        mask_scale = [m.mean() for m in masks]
        torch._foreach_maximum_(mask_scale, 1e-3)
        torch._foreach_div_(masks, mask_scale)
        updates = torch._foreach_mul(exp_avgs, masks)
    else:
        updates = exp_avgs

    # Params in a group usually share the step size (same steps / lr), apply it as a single scalar if so
    if all(not torch.is_tensor(s) for s in step_sizes) and len(set(step_sizes)) == 1:
        torch._foreach_add_(params, updates, alpha=-step_sizes[0])
    else:
        torch._foreach_add_(params, torch._foreach_mul(updates, [-s for s in step_sizes]))

    if weight_decay != 0:
        wd_scale = lr if max_lr is None else lr ** 2 / max_lr
        _foreach_add_scaled_(params, params, -wd_scale * weight_decay)
