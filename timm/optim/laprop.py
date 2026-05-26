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
from typing import Tuple

from torch.optim import Optimizer
import torch

from ._helpers import _add_scaled_, _init_scalar, _validate_scalar
from ._types import ParamsT


class LaProp(Optimizer):
    """ LaProp Optimizer

    Paper: LaProp: Separating Momentum and Adaptivity in Adam, https://arxiv.org/abs/2002.04839
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
        )
        super(LaProp, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('caution', False)
            group.setdefault('corrected_weight_decay', False)
            for p in group['params']:
                p_state = self.state.get(p, {})
                if not p_state:
                    continue
                if 'step' in p_state:
                    p_state['step'] = _init_scalar(float(p_state['step']), device='cpu')
                if (
                        'exp_avg_lr_1' in p_state
                        and torch.is_tensor(group['lr'])
                        and not torch.is_tensor(p_state['exp_avg_lr_1'])
                ):
                    p_state['exp_avg_lr_1'] = torch.tensor(
                        float(p_state['exp_avg_lr_1']),
                        dtype=group['lr'].dtype,
                        device=group['lr'].device,
                    )
                if 'exp_avg_lr_2' in p_state:
                    p_state['exp_avg_lr_2'] = _init_scalar(float(p_state['exp_avg_lr_2']), device='cpu')

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

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'].add_(1)
                one_minus_beta2 = 1 - beta2
                one_minus_beta1 = 1 - beta1

                # Decay the first and second moment running average coefficient
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=one_minus_beta2)

                state['exp_avg_lr_1'] = state['exp_avg_lr_1'] * beta1 + one_minus_beta1 * group['lr']
                state['exp_avg_lr_2'] = state['exp_avg_lr_2'] * beta2 + one_minus_beta2

                # 1 - beta1 ** state['step']
                if torch.is_tensor(group['lr']):
                    bias_correction1 = torch.where(
                        group['lr'] != 0.,
                        state['exp_avg_lr_1'] / group['lr'],
                        torch.ones_like(group['lr']),
                    )
                else:
                    bias_correction1 = state['exp_avg_lr_1'] / group['lr'] if group['lr'] != 0. else 1.
                bias_correction2 = state['exp_avg_lr_2']
                step_size = 1 / bias_correction1

                denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(group['eps'])
                step_of_this_grad = grad / denom
                exp_avg.mul_(beta1)
                _add_scaled_(exp_avg, step_of_this_grad, group['lr'] * one_minus_beta1)

                if group['caution']:
                    # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                    mask = (exp_avg * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    exp_avg = exp_avg * mask

                _add_scaled_(p, exp_avg, -step_size)

                if group['weight_decay'] != 0:
                    if group['corrected_weight_decay']:
                        wd_scale = group['lr'] ** 2 / self.defaults['lr']
                    else:
                        wd_scale = group['lr']
                    _add_scaled_(p, p, -wd_scale * group['weight_decay'])

        return loss
