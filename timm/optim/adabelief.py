import math
import torch
from torch.optim.optimizer import Optimizer


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
    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients, NeurIPS 2020

    For a complete table of recommended hyperparameters, see https://github.com/juntang-zhuang/Adabelief-Optimizer'
    For example train/args for EfficientNet see these gists
      - link to train_scipt: https://gist.github.com/juntang-zhuang/0a501dd51c02278d952cf159bc233037
      - link to args.yaml: https://gist.github.com/juntang-zhuang/517ce3c27022b908bb93f78e4f786dc3
    """

    def __init__(
            self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16, weight_decay=0, amsgrad=False,
            decoupled_decay=True, fixed_decay=False, rectify=True, degenerated_to_sgd=True):

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
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
            degenerated_to_sgd=degenerated_to_sgd, decoupled_decay=decoupled_decay, rectify=rectify,
            fixed_decay=fixed_decay, buffer=[[None, None, None] for _ in range(10)])
        super(AdaBelief, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p)

                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(p)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = torch.zeros_like(p)

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
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBelief does not support sparse gradients, please consider SparseAdam instead')

                p_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p_fp32.float()

                amsgrad = group['amsgrad']
                beta1, beta2 = group['betas']
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(p_fp32)
                
                # perform weight decay, check if decoupled weight decay
                if group['decoupled_decay']:
                    if not group['fixed_decay']:
                        p_fp32.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p_fp32.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p_fp32, alpha=group['weight_decay'])

                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var.add_(group['eps']), out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # update
                if not group['rectify']:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    # Rectified update, forked from RAdam
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

                    if num_sma >= 5:
                        denom = exp_avg_var.sqrt().add_(group['eps'])
                        p_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    elif step_size > 0:
                        p_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_fp32)

        return loss
