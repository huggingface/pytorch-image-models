import torch
import torch.nn as nn
import torch.nn.functional as F

class LSEPool(nn.Module):
    """
    Implements LogSumExp (LSE) pooling, an advanced pooling technique that provides 
    a smooth approximation to the max pooling operation. This pooling method is useful 
    for capturing the global distribution of features across spatial dimensions (height and width) 
    of the input tensor, while still maintaining differentiability.

    The class supports learnable pooling behavior with an optional learnable parameter 'r'. 
    When 'r' is large, LSE pooling closely approximates max pooling, and when 'r' is small, 
    it behaves more like average pooling. The 'r' parameter can either be a fixed value or 
    learned during training.

    Parameters:
    r (float, optional): The initial value of the pooling parameter. Default is 10.
    learnable (bool, optional): If True, 'r' is a learnable parameter. Default is True.
    """

    def __init__(self, r=10, learnable=True):
        super(LSEPool, self).__init__()
        if learnable:
            self.r = nn.Parameter(torch.ones(1) * r)
        else:
            self.r = r

    def forward(self, x):
        s = (x.size(2) * x.size(3))
        x_max = F.adaptive_max_pool2d(x, 1)
        exp = torch.exp(self.r * (x - x_max))
        sumexp = 1 / s * torch.sum(exp, dim=(2, 3))
        sumexp = sumexp.view(sumexp.size(0), -1, 1, 1)
        logsumexp = x_max + 1 / self.r * torch.log(sumexp)
        return logsumexp