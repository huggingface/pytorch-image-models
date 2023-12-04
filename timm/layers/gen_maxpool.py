import torch
import torch.nn as nn

class GeneralizedMP(nn.Module):
    """
    Implements Generalized Max Pooling (GMP), a global pooling operation that 
    generalizes the concept of max pooling to capture more complex and discriminative 
    features from the input tensor. 

    The class operates by computing a linear kernel based on the input tensor, 
    then solving a linear system to obtain the pooling coefficients. These coefficients 
    are used to weigh and aggregate the input features, resulting in a pooled feature vector.

    Parameters:
    lamb (float, optional): A regularization parameter used in the linear system 
    to ensure numerical stability. Default value is 1e3.

    Note:
    - The input tensor is expected to be in the format (B, D, H, W), where B is batch size, 
    D is depth (channels), H is height, and W is width.
    - The implementation uses PyTorch's linear algebra functions to solve the linear system. 
    """
    def __init__(self, lamb = 1e3):
        super().__init__()
        self.lamb = nn.Parameter(lamb * torch.ones(1))
        #self.inv_lamb = nn.Parameter((1./lamb) * torch.ones(1))

    def forward(self, x):
        B, D, H, W = x.shape
        N = H * W
        identity = torch.eye(N).cuda()
        # reshape x, s.t. we can use the gmp formulation as a global pooling operation
        x = x.view(B, D, N)
        x = x.permute(0, 2, 1)
        # compute the linear kernel
        K = torch.bmm(x, x.permute(0, 2, 1))
        # solve the linear system (K + lambda * I) * alpha = ones
        A = K + self.lamb * identity
        o = torch.ones(B, N, 1).cuda()
        #alphas, _ = torch.gesv(o, A) # tested using pytorch 1.0.1
        alphas = torch.linalg.solve(A,o) # TODO check it again
        alphas = alphas.view(B, 1, -1)        
        xi = torch.bmm(alphas, x)
        xi = xi.view(B, -1)
        return xi