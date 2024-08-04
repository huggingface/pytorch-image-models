import torch
import torch.nn as nn
from torch.nn import init

def prepare_input(self, x):
    """
    Prepares the input tensor for different neural network architectures (Transformers and CNNs).
    
    This function adjusts the shape of the input tensor based on its dimensionality.
    It supports input tensors for Transformers (3D) and CNNs (4D), ensuring they are
    correctly formatted for these architectures.

    For a Transformer, it expects a tensor of shape (B, N, d), where B is the batch size,
    N are patch tokens, and d is the depth (channels). The tensor is returned as is.

    For a CNN, it expects a tensor of shape (B, d, H, W), where B is the batch size,
    d is the depth (channels), H is the height, and W is the width. The tensor is reshaped
    and permuted to the shape (B, H*W, d) to match CNN input requirements.

    Parameters:
    x (torch.Tensor): The input tensor to be preprocessed.
    """
    if len(x.shape) == 3: # Transformer
        # Input tensor dimensions:
        # x: (B, N, d), where B is batch size, N are patch tokens, d is depth (channels)
        B, N, d = x.shape
        return x
    if len(x.shape) == 4: # CNN
        # Input tensor dimensions:
        # x: (B, d, H, W), where B is batch size, d is depth (channels), H is height, and W is width
        B, d, H, W = x.shape
        x = x.reshape(B, d, H*W).permute(0, 2, 1) # (B, d, H, W) -> (B, d, H*W) -> (B, H*W, d)
        return x
    else:
        raise ValueError(f"Unsupported number of dimensions in input tensor: {len(x.shape)}")

class SlotPooling(nn.Module):
    """
    This class implements the Slot Attention module as described in the paper
    "Object-Centric Learning with Slot Attention". 
    
    The module is designed for object-centric learning tasks and utilizes the concept of 
    'slots' to represent distinct object features within an input. 
    It iteratively refines these slots through a pooling mechanism to capture
    complex object representations.
    
    Parameters:
    num_slots (int): Number of slots to be used.
    dim (int): Dimensionality of the input features.
    iters (int, optional): Number of iterations for slot refinement. Default is 3.
    eps (float, optional): A small epsilon value to avoid division by zero. Default is 1e-8.
    hidden_dim (int, optional): Dimensionality of the hidden layer within the module. Default is 128.
    """
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        inputs = prepare_input(inputs)
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps

            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        slots = slots.max(dim=1)[0]

        return slots