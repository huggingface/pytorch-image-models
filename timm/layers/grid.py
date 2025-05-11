from typing import Tuple

import torch


def ndgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    This naming is based on MATLAB, the purpose is to avoid confusion due to torch's change to make
    torch.meshgrid behaviour move from matching ndgrid ('ij') indexing to numpy meshgrid defaults of ('xy').

    """
    try:
        return torch.meshgrid(*tensors, indexing='ij')
    except TypeError:
        # old PyTorch < 1.10 will follow this path as it does not have indexing arg,
        # the old behaviour of meshgrid was 'ij'
        return torch.meshgrid(*tensors)


def meshgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in spatial dim order.

    The meshgrid function is similar to ndgrid except that the order of the
    first two input and output arguments is switched.

    That is, the statement

    [X,Y,Z] = meshgrid(x,y,z)
    produces the same result as

    [Y,X,Z] = ndgrid(y,x,z)
    Because of this, meshgrid is better suited to problems in two- or three-dimensional Cartesian space,
    while ndgrid is better suited to multidimensional problems that aren't spatially based.
    """

    # NOTE: this will throw in PyTorch < 1.10 as meshgrid did not support indexing arg or have
    # capability of generating grid in xy order before then.
    return torch.meshgrid(*tensors, indexing='xy')

