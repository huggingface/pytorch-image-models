import torch
from torch.optim import Optimizer


class RMSpropTF(Optimizer):
    """Implements RMSprop algorithm (TensorFlow style epsilon)

    NOTE: This is a direct cut-and-paste of PyTorch RMSprop with eps applied before sqrt
    to closer match Tensorflow for matching hyper-params.

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing (decay) constant (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_decay (bool, optional): decoupled weight decay as per https://arxiv.org/abs/1711.05101
        lr_in_momentum (bool, optional): learning rate scaling is included in the momentum buffer
            update as per defaults in Tensorflow

    """

    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-10, weight_decay=0, momentum=0., centered=False,
                 decoupled_decay=False, lr_in_momentum=True):
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

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay,
                        decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        super(RMSpropTF, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSpropTF, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(p.data)  # PyTorch inits to zero
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if 'decoupled_decay' in group and group['decoupled_decay']:
                        p.data.add_(-group['weight_decay'], p.data)
                    else:
                        grad = grad.add(group['weight_decay'], p.data)

                # Tensorflow order of ops for updating squared avg
                square_avg.add_(one_minus_alpha, grad.pow(2) - square_avg)
                # square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)  # PyTorch original

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(one_minus_alpha, grad - grad_avg)
                    # grad_avg.mul_(alpha).add_(1 - alpha, grad)  # PyTorch original
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).add(group['eps']).sqrt_()  # eps moved in sqrt
                else:
                    avg = square_avg.add(group['eps']).sqrt_()  # eps moved in sqrt

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    # Tensorflow accumulates the LR scaling in the momentum buffer
                    if 'lr_in_momentum' in group and group['lr_in_momentum']:
                        buf.mul_(group['momentum']).addcdiv_(group['lr'], grad, avg)
                        p.data.add_(-buf)
                    else:
                        # PyTorch scales the param update by LR
                        buf.mul_(group['momentum']).addcdiv_(grad, avg)
                        p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)

        return loss
