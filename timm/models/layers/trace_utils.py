import torch
try:
    from torch.overrides import has_torch_function, handle_torch_function
except ImportError:
    from torch._overrides import has_torch_function, handle_torch_function


def _assert(condition, message):
    r"""A wrapper around Python's assert which is symbolically traceable.
    This is based on _assert method in torch.__init__.py but brought here to avoid reliance
    on internal torch fn and allow compatibility with PyTorch < 1.8.
    """
    if type(condition) is not torch.Tensor and has_torch_function((condition,)):
        return handle_torch_function(_assert, (condition,), condition, message)
    assert condition, message


def _float_to_int(x: float) -> int:
    """
    Symbolic tracing helper to substitute for inbuilt `int`.
    Hint: Inbuilt `int` can't accept an argument of type `Proxy`
    """
    return int(x)
