""" Activations

A collection of activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
from torch import nn as nn
from torch.nn import functional as F


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        nn.sig = nn.Sigmoid()
        self.skip = nn.quantized.FloatFunctional()
    def forward(self, x):
        out = nn.sig(x)

        return self.skip.mul(x,out)

def sigmoid(x, inplace: bool = False):
    return x.sigmoid_() if inplace else x.sigmoid()


# PyTorch has this, but not with a consistent inplace argmument interface
class Sigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()
        nn.sig = nn.Sigmoid()
    def forward(self, x):
        out = nn.sig(x)
        return out


class HardSwish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)
        self.quant_mul1 = nn.quantized.FloatFunctional()
        self.quant_mul2 = nn.quantized.FloatFunctional()
        self.quant_add = nn.quantized.FloatFunctional()
        
    def forward(self, x):
        out = self.quant_add.add_scalar(x, 3.0)
        out = self.relu6(out)
        out = self.quant_mul1.mul(x,out)
        out = self.quant_mul2.mul_scalar(out, 1/6)
        return out



class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace)
        self.quant_add = nn.quantized.FloatFunctional()
        self.quant_mul = nn.quantized.FloatFunctional()
    def forward(self, x):
        out = self.quant_add.add_scalar(x, 3.0)
        out = self.relu6(out)
        out = self.quant_mul.mul_scalar(out,1/6)
        return out

