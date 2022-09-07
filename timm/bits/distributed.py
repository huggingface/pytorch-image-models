from typing import Dict, Tuple, List, Union, Any, Callable

import torch
from torch.distributed import ReduceOp

from timm.utils import unwrap_model

from .device_env import DeviceEnv


TensorSeq = Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor], Dict[Any, torch.Tensor]]


def _validate_type(tensor: TensorSeq):
    if isinstance(tensor, (dict, list, tuple)):
        if not tensor:
            return
    else:
        assert isinstance(tensor, torch.Tensor)


def distribute_bn(model: torch.nn.Module, reduce: bool = False, dev_env: DeviceEnv = None):
    if dev_env is None:
        dev_env = DeviceEnv.instance()
    # ensure every node has the same running bn stats
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                # average bn stats across whole group
                dev_env.all_reduce_(bn_buf, average=True)
            else:
                # broadcast bn stats from rank 0 to whole group
                dev_env.broadcast_(bn_buf, 0)


def all_gather_recursive(tensor: TensorSeq, cat_dim=0, dev_env: DeviceEnv = None):
    """ Recursive all gather via DeviceEnv distributed primitives
    FIXME add group support
    """
    _validate_type(tensor)
    if dev_env is None:
        dev_env = DeviceEnv.instance()
    if isinstance(tensor, torch.Tensor):
        return dev_env.all_gather(tensor, cat_dim=cat_dim)
    elif isinstance(tensor, dict):
        return {k: all_gather_recursive(v, dev_env=dev_env) for k, v in tensor.items()}
    elif isinstance(tensor, (tuple, list)):
        return type(tensor)(all_gather_recursive(v, dev_env=dev_env) for v in tensor)


def all_reduce_recursive(tensor: TensorSeq, op=ReduceOp.SUM, average=False, dev_env: DeviceEnv = None):
    """ Recursive all reduce via DeviceEnv distributed primitives
    FIXME add group support
    """
    _validate_type(tensor)
    if dev_env is None:
        dev_env = DeviceEnv.instance()
    if isinstance(tensor, torch.Tensor):
        return dev_env.all_reduce_(tensor, op=op, average=average)
    elif isinstance(tensor, dict):
        return {k: all_reduce_recursive(v, op=op, average=average, dev_env=dev_env) for k, v in tensor.items()}
    elif isinstance(tensor, (tuple, list)):
        return type(tensor)(all_reduce_recursive(v, op=op, average=average, dev_env=dev_env) for v in tensor)


def broadcast_recursive(tensor: TensorSeq, src_rank: int, dev_env: DeviceEnv = None):
    """ Recursive broadcast via DeviceEnv distributed primitives
    FIXME add group support
    """
    _validate_type(tensor)
    if dev_env is None:
        dev_env = DeviceEnv.instance()
    if isinstance(tensor, torch.Tensor):
        return dev_env.broadcast_(tensor, src_rank=src_rank)
    elif isinstance(tensor, dict):
        return {k: broadcast_recursive(v, src_rank=src_rank, dev_env=dev_env) for k, v in tensor.items()}
    elif isinstance(tensor, (tuple, list)):
        return type(tensor)(broadcast_recursive(v, src_rank=src_rank, dev_env=dev_env) for v in tensor)


def all_gather_sequence(tensor: TensorSeq, cat_dim: int = 0, dev_env: DeviceEnv = None):
    """ All gather a flat Tensor sequence (dict, list, tuple) of same shape

    """
    _validate_type(tensor)
    if dev_env is None:
        dev_env = DeviceEnv.instance()

    with torch.no_grad():
        names = None
        # merge values into one tensor for reduction
        if isinstance(tensor, dict):
            names = tensor.keys()
            gather_values = tuple(tensor.values())
        elif isinstance(tensor, (tuple, list)):
            gather_values = tensor
        else:
            gather_values = (tensor,)

        gather_values = torch.stack(gather_values, dim=0)
        gather_values = dev_env.all_gather(gather_values, cat_dim=cat_dim + 1).unbind(dim=0)

        # separate reduced values into original structure
        if isinstance(tensor, dict):
            gather_values = {k: v for k, v in zip(names, gather_values)}
        elif isinstance(tensor, (tuple, list)):
            gather_values = type(tensor)(v for v in gather_values)
        else:
            gather_values = gather_values[0]

    return gather_values


def all_reduce_sequence(tensor: TensorSeq, op=ReduceOp.SUM, average=False, dev_env: DeviceEnv = None):
    """
    All reduce the tensors in a flat Tensor sequence (dict, list, tuple) of same tensor shape

    Args:
        tensor (dict): inputs to be reduced. All the values must be scalar Tensor.
        average (bool): whether to do average or sum
    Returns:
        a sequence with the same type as input (dict, list, tuple)
    """
    _validate_type(tensor)
    if dev_env is None:
        dev_env = DeviceEnv.instance()

    with torch.no_grad():
        names = None
        # merge values into one tensor for reduction
        if isinstance(tensor, dict):
            names = tensor.keys()
            reduce_values = tuple(tensor.values())
        elif isinstance(tensor, (tuple, list)):
            reduce_values = tensor
        else:
            reduce_values = (tensor,)

        reduce_values = torch.stack(reduce_values, dim=0)
        dev_env.all_reduce_(reduce_values, op=op, average=average)
        reduce_values = reduce_values.unbind(dim=0)
        # separate reduced values into original structure
        if isinstance(tensor, dict):
            reduce_values = {k: v for k, v in zip(names, reduce_values)}
        elif isinstance(tensor, (tuple, list)):
            reduce_values = type(tensor)(v for v in reduce_values)
        else:
            reduce_values = reduce_values[0]

    return reduce_values