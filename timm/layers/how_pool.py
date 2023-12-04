import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class HOWPooling(nn.Module):
    """
    Implements HOW, as described in the paper 
    'Learning and Aggregating Deep Local Descriptors for Instance-Level Recognition'. 
    This pooling method focuses on aggregating deep local descriptors 
    for enhanced instance-level recognition.

    The class includes functions for L2-based attention, smoothing average pooling, 
    L2 normalization (l2n), and a forward method that integrates these components. 
    It applies dimensionality reduction to the input features before the pooling operation.

    Parameters:
    input_dim (int): Dimension of the input features.
    dim_reduction (int): Target dimension after reduction.
    kernel_size (int): Size of the kernel used in smoothing average pooling.
    """
    def __init__(self, input_dim = 512, dim_reduction = 128, kernel_size = 3):
        super(HOWPooling, self).__init__()
        self.kernel_size = kernel_size
        self.dimreduction = ConvDimReduction(input_dim, dim_reduction)

    def L2Attention(self, x):
        return (x.pow(2.0).sum(1) + 1e-10).sqrt().squeeze(0)

    def smoothing_avg_pooling(self, feats):
        """Smoothing average pooling
        :param torch.Tensor feats: Feature map
        :param int kernel_size: kernel size of pooling
        :return torch.Tensor: Smoothend feature map
        """
        pad = self.kernel_size // 2
        return F.avg_pool2d(feats, (self.kernel_size, self.kernel_size), stride=1, padding=pad,
                            count_include_pad=False)        

    def l2n(self, x, eps=1e-6):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

    def forward(self, x):

        weights = self.L2Attention(x)
        x = self.smoothing_avg_pooling(x)
        x = self.dimreduction(x)
        x = (x * weights.unsqueeze(1)).sum((-2, -1))
        return self.l2n(x)

class ConvDimReduction(nn.Conv2d):
    """
    Implements dimensionality reduction using a convolutional layer. This layer is
    designed for reducing the dimensions of input features, particularly for use in
    aggregation and pooling operations like in the HOWPooling class.

    The class also includes methods for learning and applying PCA whitening with shrinkage,
    which is a technique to reduce dimensionality while preserving important feature variations.

    Parameters:
    input_dim (int): The input dimension (number of channels) of the network.
    dim (int): The target output dimension for the whitening process.
    """
    def __init__(self, input_dim, dim):
        super().__init__(input_dim, dim, (1, 1), padding=0, bias=True)

    def pcawhitenlearn_shrinkage(X, s=1.0):
        """Learn PCA whitening with shrinkage from given descriptors"""
        N = X.shape[0]

        # Learning PCA w/o annotations
        m = X.mean(axis=0, keepdims=True)
        Xc = X - m
        Xcov = np.dot(Xc.T, Xc)
        Xcov = (Xcov + Xcov.T) / (2*N)
        eigval, eigvec = np.linalg.eig(Xcov)
        order = eigval.argsort()[::-1]
        eigval = eigval[order]
        eigvec = eigvec[:, order]

        eigval = np.clip(eigval, a_min=1e-14, a_max=None)
        P = np.dot(np.linalg.inv(np.diag(np.power(eigval, 0.5*s))), eigvec.T)

        return m, P.T

    def initialize_pca_whitening(self, des):
        """Initialize PCA whitening from given descriptors. Return tuple of shift and projection."""
        m, P = self.pcawhitenlearn_shrinkage(des)
        m, P = m.T, P.T

        projection = torch.Tensor(P[:self.weight.shape[0], :]).unsqueeze(-1).unsqueeze(-1)
        self.weight.data = projection.to(self.weight.device)

        projected_shift = -torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()
        self.bias.data = projected_shift[:self.weight.shape[0]].to(self.bias.device)
        return m.T, P.T