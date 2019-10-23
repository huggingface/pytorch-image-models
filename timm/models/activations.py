import torch
from torch import nn as nn
from torch.nn import functional as F


_USE_MEM_EFFICIENT_ISH = True
if _USE_MEM_EFFICIENT_ISH:
    # This version reduces memory overhead of Swish during training by
    # recomputing torch.sigmoid(x) in backward instead of saving it.
    class SwishAutoFn(torch.autograd.Function):
        """Swish - Described in: https://arxiv.org/abs/1710.05941
        Memory efficient variant from:
         https://medium.com/the-artificial-impostor/more-memory-efficient-swish-activation-function-e07c22c12a76
        """
        @staticmethod
        def forward(ctx, x):
            result = x.mul(torch.sigmoid(x))
            ctx.save_for_backward(x)
            return result

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_variables[0]
            sigmoid_x = torch.sigmoid(x)
            return grad_output.mul(sigmoid_x * (1 + x * (1 - sigmoid_x)))

    def swish(x, inplace=False):
        # inplace ignored
        return SwishAutoFn.apply(x)


    class MishAutoFn(torch.autograd.Function):
        """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
        Experimental memory-efficient variant
        """

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            y = x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))
            return y

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_variables[0]
            x_sigmoid = torch.sigmoid(x)
            x_tanh_sp = F.softplus(x).tanh()
            return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))

    def mish(x, inplace=False):
        # inplace ignored
        return MishAutoFn.apply(x)


    class WishAutoFn(torch.autograd.Function):
        """Wish: My own mistaken creation while fiddling with Mish. Did well in some experiments.
        Experimental memory-efficient variant
        """

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            y = x.mul(torch.tanh(torch.exp(x)))
            return y

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_variables[0]
            x_exp = x.exp()
            x_tanh_exp = x_exp.tanh()
            return grad_output.mul(x_tanh_exp + x * x_exp * (1 - x_tanh_exp * x_tanh_exp))

    def wish(x, inplace=False):
        # inplace ignored
        return WishAutoFn.apply(x)
else:
    def swish(x, inplace=False):
        """Swish - Described in: https://arxiv.org/abs/1710.05941
        """
        return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


    def mish(x, inplace=False):
        """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
        """
        inner = F.softplus(x).tanh()
        return x.mul_(inner) if inplace else x.mul(inner)


    def wish(x, inplace=False):
        """Wish: My own mistaken creation while fiddling with Mish. Did well in some experiments.
        """
        inner = x.exp().tanh()
        return x.mul_(inner) if inplace else x.mul(inner)


class Swish(nn.Module):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


class Mish(nn.Module):
    def __init__(self, inplace=False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return mish(x, self.inplace)


class Wish(nn.Module):
    def __init__(self, inplace=False):
        super(Wish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return wish(x, self.inplace)


def sigmoid(x, inplace=False):
    return x.sigmoid_() if inplace else x.sigmoid()


# PyTorch has this, but not with a consistent inplace argmument interface
class Sigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()


def tanh(x, inplace=False):
    return x.tanh_() if inplace else x.tanh()


# PyTorch has this, but not with a consistent inplace argmument interface
class Tanh(nn.Module):
    def __init__(self, inplace=False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.tanh_() if self.inplace else x.tanh()


def hard_swish(x, inplace=False):
    inner = F.relu6(x + 3.).div_(6.)
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, self.inplace)


def hard_sigmoid(x, inplace=False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, self.inplace)

