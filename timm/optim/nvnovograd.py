""" Nvidia NovoGrad Optimizer.
Original impl by Nvidia from Jasper example:
    - https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper
Paper: `Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks`
    - https://arxiv.org/abs/1905.11286
"""

from typing import Any, Dict

import torch
from torch.optim.optimizer import Optimizer
import math

from ._helpers import _load_state_dict_preserving_dtypes, _max_lr_snapshot


class NvNovoGrad(Optimizer):
    """
    Implements Novograd algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0.98))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging: gradient averaging
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        decoupled_decay: apply weight decay directly to the weights (lr * weight_decay per step) instead of adding
            it to the normalized gradient before the momentum update (as per the paper)
        corrected_weight_decay: apply corrected weight decay (lr**2 / max_lr) when decoupled_decay is True
    """

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.95, 0.98),
            eps=1e-8,
            weight_decay=0,
            grad_averaging=False,
            amsgrad=False,
            decoupled_decay=False,
            corrected_weight_decay=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_averaging=grad_averaging,
            amsgrad=amsgrad,
            decoupled_decay=decoupled_decay,
            corrected_weight_decay=corrected_weight_decay,
            max_lr_snapshot=_max_lr_snapshot(lr, corrected_weight_decay),
        )

        super(NvNovoGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NvNovoGrad, self).__setstate__(state)
        self.defaults.setdefault('max_lr_snapshot', _max_lr_snapshot(self.defaults['lr']))  # pickled pre-snapshot
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('decoupled_decay', False)
            group.setdefault('corrected_weight_decay', False)
            group.setdefault('max_lr_snapshot', self.defaults['max_lr_snapshot'])

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        _load_state_dict_preserving_dtypes(
            self, state_dict, super().load_state_dict,
            lambda group, param: {
                key: torch.get_default_dtype() for key in ('exp_avg_sq', 'max_exp_avg_sq')
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
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Sparse gradients are not supported.')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros([]).to(state['exp_avg'].device)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros([]).to(state['exp_avg'].device)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                norm = torch.sum(torch.pow(grad, 2))

                if exp_avg_sq == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(norm, alpha=1 - beta2)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                # NOTE p.grad is left unmodified, the normalized gradient is a new tensor
                update = grad / denom
                if group['weight_decay'] != 0:
                    if group['decoupled_decay']:
                        if group['corrected_weight_decay']:
                            wd_scale = group['lr'] ** 2 / group['max_lr_snapshot']
                        else:
                            wd_scale = group['lr']
                        p.mul_(1. - wd_scale * group['weight_decay'])
                    else:
                        update.add_(p, alpha=group['weight_decay'])
                if group['grad_averaging']:
                    update.mul_(1 - beta1)
                exp_avg.mul_(beta1).add_(update)

                p.add_(exp_avg, alpha=-group['lr'])

        return loss
