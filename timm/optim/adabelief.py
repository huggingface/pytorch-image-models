import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ._helpers import _foreach_chunked, _foreach_lerp_, _load_state_dict_preserving_dtypes, _resolve_foreach

_LOW_PRECISION = (torch.float16, torch.bfloat16)


class AdaBelief(Optimizer):
    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-16)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        decoupled_decay (boolean, optional): (default: True) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: True) If set as True, then perform the rectified
            update similar to RAdam
        degenerated_to_sgd (boolean, optional) (default:True) If set as True, then perform SGD update
            when variance of gradient is high
        foreach (boolean, optional): use the foreach (multi-tensor) impl, by default (None) used unless unsupported
    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients, NeurIPS 2020

    For a complete table of recommended hyperparameters, see https://github.com/juntang-zhuang/Adabelief-Optimizer'
    For example train/args for EfficientNet see these gists
      - link to train_script: https://gist.github.com/juntang-zhuang/0a501dd51c02278d952cf159bc233037
      - link to args.yaml: https://gist.github.com/juntang-zhuang/517ce3c27022b908bb93f78e4f786dc3
    """

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-16,
            weight_decay=0,
            amsgrad=False,
            decoupled_decay=True,
            fixed_decay=False,
            rectify=True,
            degenerated_to_sgd=True,
            foreach: Optional[bool] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            degenerated_to_sgd=degenerated_to_sgd,
            decoupled_decay=decoupled_decay,
            rectify=rectify,
            fixed_decay=fixed_decay,
            buffer=[[None, None, None] for _ in range(10)],
            foreach=foreach,
        )
        super(AdaBelief, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('foreach', None)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        _load_state_dict_preserving_dtypes(
            self, state_dict, super().load_state_dict,
            lambda group, param: {
                key: torch.float32 if param.dtype in {torch.float16, torch.bfloat16} else None
                for key in ('exp_avg', 'exp_avg_var', 'max_exp_avg_var')
            },
        )

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization, state is kept in fp32 for low precision params (as in step())
                state_dtype = torch.float32 if p.dtype in _LOW_PRECISION else p.dtype
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p, dtype=state_dtype)

                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(p, dtype=state_dtype)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = torch.zeros_like(p, dtype=state_dtype)

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
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            params = []
            grads = []
            exp_avgs = []
            exp_avg_vars = []
            max_exp_avg_vars = []
            bias_corrections1 = []
            bias_corrections2 = []
            rectified = []  # (num_sma, step_size) per param if rectify

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(
                        'AdaBelief does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                # State initialization, state is kept in fp32 for low precision params
                if len(state) == 0:
                    state_dtype = torch.float32 if p.dtype in _LOW_PRECISION else p.dtype
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, dtype=state_dtype)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p, dtype=state_dtype)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(p, dtype=state_dtype)

                # per param scalar state, params of a group can be at different steps
                state['step'] += 1
                bias_corrections1.append(1 - beta1 ** state['step'])
                bias_corrections2.append(1 - beta2 ** state['step'])
                if group['rectify']:
                    # Rectified update, forked from RAdam, the (lr free) step size is cached per step
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        num_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        num_sma_max = 2 / (1 - beta2) - 1
                        num_sma = num_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = num_sma

                        # more conservative since it's an approximated value
                        if num_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t) *
                                (num_sma - 4) / (num_sma_max - 4) *
                                (num_sma - 2) / num_sma *
                                num_sma_max / (num_sma_max - 2)) / (1 - beta1 ** state['step'])
                        elif group['degenerated_to_sgd']:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        else:
                            step_size = -1
                        buffered[2] = step_size
                    rectified.append((num_sma, step_size))

                params.append(p)
                grads.append(p.grad)
                exp_avgs.append(state['exp_avg'])
                exp_avg_vars.append(state['exp_avg_var'])
                if amsgrad:
                    max_exp_avg_vars.append(state['max_exp_avg_var'])

            adabelief(
                params,
                grads,
                exp_avgs,
                exp_avg_vars,
                max_exp_avg_vars,
                bias_corrections1,
                bias_corrections2,
                rectified if group['rectify'] else None,
                foreach=group['foreach'],
                lr=group['lr'],
                beta1=beta1,
                beta2=beta2,
                eps=group['eps'],
                weight_decay=group['weight_decay'],
                amsgrad=amsgrad,
                decoupled_decay=group['decoupled_decay'],
                fixed_decay=group['fixed_decay'],
            )

        return loss


def adabelief(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_vars: List[Tensor],
        max_exp_avg_vars: List[Tensor],
        bias_corrections1: List[float],
        bias_corrections2: List[float],
        rectified: Optional[List[Tuple[float, float]]],
        foreach: Optional[bool] = None,
        *,
        lr: Union[float, Tensor],
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        amsgrad: bool,
        decoupled_decay: bool,
        fixed_decay: bool,
) -> None:
    """Functional API that performs the AdaBelief tensor updates. Per param bias corrections and rectified
    (num_sma, step_size) are computed by the AdaBelief class (see for details).
    """
    # the foreach impl applies per param step sizes as python scalars, a tensor lr uses the single tensor impl
    if _resolve_foreach(foreach, lr=lr, params=params) and not torch.is_tensor(lr) and not torch.jit.is_scripting():
        func = _multi_tensor_adabelief
    else:
        func = _single_tensor_adabelief
    func(
        params,
        grads,
        exp_avgs,
        exp_avg_vars,
        max_exp_avg_vars,
        bias_corrections1,
        bias_corrections2,
        rectified,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        decoupled_decay=decoupled_decay,
        fixed_decay=fixed_decay,
    )


def _single_tensor_adabelief(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_vars: List[Tensor],
        max_exp_avg_vars: List[Tensor],
        bias_corrections1: List[float],
        bias_corrections2: List[float],
        rectified: Optional[List[Tuple[float, float]]],
        *,
        lr: Union[float, Tensor],
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        amsgrad: bool,
        decoupled_decay: bool,
        fixed_decay: bool,
) -> None:
    for i, p in enumerate(params):
        grad = grads[i]
        if grad.dtype in _LOW_PRECISION:
            grad = grad.float()
        p_fp32 = p.float() if p.dtype in _LOW_PRECISION else p

        # perform weight decay, check if decoupled weight decay
        if decoupled_decay:
            if not fixed_decay:
                p_fp32.mul_(1.0 - lr * weight_decay)
            else:
                p_fp32.mul_(1.0 - weight_decay)
        else:
            if weight_decay != 0:
                grad = grad.add(p_fp32, alpha=weight_decay)  # not in-place, leave p.grad unmodified

        exp_avg, exp_avg_var = exp_avgs[i], exp_avg_vars[i]
        bias_correction1, bias_correction2 = bias_corrections1[i], bias_corrections2[i]

        # Update first and second moment running average
        exp_avg.lerp_(grad, 1 - beta1)
        grad_residual = grad - exp_avg
        exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

        if amsgrad:
            max_exp_avg_var = max_exp_avg_vars[i]
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_var, exp_avg_var.add_(eps), out=max_exp_avg_var)

            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_var.add_(eps).sqrt() / math.sqrt(bias_correction2)).add_(eps)

        # update
        if rectified is None:
            # Default update
            step_size = lr / bias_correction1
            p_fp32.addcdiv_(exp_avg, denom, value=-step_size)
        else:
            # Rectified update, forked from RAdam
            num_sma, step_size = rectified[i]
            if num_sma >= 5:
                denom = exp_avg_var.sqrt().add_(eps)
                p_fp32.addcdiv_(exp_avg, denom, value=-step_size * lr)
            elif step_size > 0:
                p_fp32.add_(exp_avg, alpha=-step_size * lr)

        if p.dtype in _LOW_PRECISION:
            p.copy_(p_fp32)


@_foreach_chunked(8)
def _multi_tensor_adabelief(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_vars: List[Tensor],
        max_exp_avg_vars: List[Tensor],
        bias_corrections1: List[float],
        bias_corrections2: List[float],
        rectified: Optional[List[Tuple[float, float]]],
        *,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        amsgrad: bool,
        decoupled_decay: bool,
        fixed_decay: bool,
) -> None:
    if len(params) == 0:
        return

    # updates are computed in fp32 for low precision params, copied back at the end
    low_precision = [i for i, p in enumerate(params) if p.dtype in _LOW_PRECISION]
    params_fp32 = [p.float() if p.dtype in _LOW_PRECISION else p for p in params]
    grads = [g.float() if g.dtype in _LOW_PRECISION else g for g in grads]

    # perform weight decay, check if decoupled weight decay
    if decoupled_decay:
        decay = weight_decay if fixed_decay else lr * weight_decay
        if decay != 0:
            torch._foreach_mul_(params_fp32, 1.0 - decay)
    elif weight_decay != 0:
        grads = torch._foreach_add(grads, params_fp32, alpha=weight_decay)  # not in-place, leave p.grad unmodified

    # Update first and second moment running average
    _foreach_lerp_(exp_avgs, grads, 1 - beta1)
    grad_residuals = torch._foreach_sub(grads, exp_avgs)
    torch._foreach_mul_(exp_avg_vars, beta2)
    torch._foreach_addcmul_(exp_avg_vars, grad_residuals, grad_residuals, value=1 - beta2)
    torch._foreach_add_(exp_avg_vars, eps)
    if amsgrad:
        # Maintains the maximum of all 2nd moment running avg. till now
        torch._foreach_maximum_(max_exp_avg_vars, exp_avg_vars)

    if rectified is None:
        # Default update, normalized by the (max) bias corrected 2nd moment
        denom = torch._foreach_sqrt(max_exp_avg_vars if amsgrad else exp_avg_vars)
        torch._foreach_div_(denom, [math.sqrt(bc) for bc in bias_corrections2])
        torch._foreach_add_(denom, eps)
        torch._foreach_addcdiv_(params_fp32, exp_avgs, denom, [-lr / bc for bc in bias_corrections1])
    else:
        # Rectified update, forked from RAdam
        adaptive = [i for i, (num_sma, _) in enumerate(rectified) if num_sma >= 5]
        sgd = [i for i, (num_sma, step_size) in enumerate(rectified) if num_sma < 5 and step_size > 0]
        if adaptive:
            denom = torch._foreach_sqrt([exp_avg_vars[i] for i in adaptive])
            torch._foreach_add_(denom, eps)
            torch._foreach_addcdiv_(
                [params_fp32[i] for i in adaptive],
                [exp_avgs[i] for i in adaptive],
                denom,
                [-rectified[i][1] * lr for i in adaptive],
            )
        if sgd:
            alphas = [-rectified[i][1] * lr for i in sgd]
            sgd_params = [params_fp32[i] for i in sgd]
            sgd_exp_avgs = [exp_avgs[i] for i in sgd]
            if len(set(alphas)) == 1:
                torch._foreach_add_(sgd_params, sgd_exp_avgs, alpha=alphas[0])
            else:
                torch._foreach_add_(sgd_params, torch._foreach_mul(sgd_exp_avgs, alphas))

    if low_precision:
        dst, src = [params[i] for i in low_precision], [params_fp32[i] for i in low_precision]
        if hasattr(torch, '_foreach_copy_'):
            torch._foreach_copy_(dst, src)
        else:
            for d, s in zip(dst, src):
                d.copy_(s)
