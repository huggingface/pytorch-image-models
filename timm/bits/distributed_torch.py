""" PyTorch distributed helpers

Some of this lifted from Detectron2 with other fns added by myself.

FIXME many functions remain unfinished/untested
"""
from typing import Dict, Tuple, List, Union, Any, Callable

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

TensorSeq = Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor], Dict[Any, torch.Tensor]]


def synchronize_torch():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def all_reduce_sequence_torch(values: TensorSeq, op=ReduceOp.SUM, average=False, group=None):
    """
    All reduce the tensors in a sequence (dict, list, tuple)

    Args:
        values (dict): inputs to be reduced. All the values must be scalar Tensor.
        average (bool): whether to do average or sum
    Returns:
        a sequence with the same type as input (dict, list, tuple)
    """
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        return values

    with torch.no_grad():
        names = None
        if isinstance(values, dict):
            names = values.keys()
            reduce_values = torch.stack(tuple(values.values()), dim=0)
        elif isinstance(values, (tuple, list)):
            reduce_values = torch.stack(values, dim=0)
        else:
            reduce_values = values
        dist.all_reduce(reduce_values, op=op, group=group)
        if average:
            reduce_values /= world_size
        if isinstance(values, dict):
            reduce_values = {k: v for k, v in zip(names, reduce_values)}
        elif isinstance(values, (tuple, list)):
            reduce_values = type(values)(v for v in reduce_values)
    return reduce_values


def reduce_sequence_torch(values: TensorSeq, dst_rank=0, op=ReduceOp.SUM, average=False, group=None):
    """
    All reduce the tensors in a sequence (dict, list, tuple)

    Args:
        values (dict): inputs to be reduced. All the values must be scalar Tensor.
        average (bool): whether to do average or sum
    Returns:
        a sequence with the same type as input (dict, list, tuple)
    """
    world_size = dist.get_world_size(group)
    this_rank = dist.get_rank()
    if world_size <= 1:
        return values

    with torch.no_grad():
        names = None
        if isinstance(values, dict):
            names = values.keys()
            reduce_values = torch.stack(tuple(values.values()), dim=0)
        elif isinstance(values, (tuple, list)):
            reduce_values = torch.stack(values, dim=0)
        else:
            reduce_values = values
        reduce_values = torch.stack(reduce_values, dim=0)
        dist.reduce(reduce_values, dst=dst_rank, op=op, group=group)
        if average and this_rank == dst_rank:
            reduce_values /= world_size
        if isinstance(values, dict):
            reduce_values = {k: v for k, v in zip(names, reduce_values)}
        elif isinstance(values, (tuple, list)):
            reduce_values = type(values)(v for v in reduce_values)
    return reduce_values


def all_gather_sequence_torch(values: TensorSeq, group=None, join_fn=torch.cat, join_dim=0):
    world_size = dist.get_world_size(group)

    def _do_gather(tensor):
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor, group=group)
        return join_fn(tensor_list, dim=join_dim)

    if isinstance(values, dict):
        gathered = {k: _do_gather(v) for k, v in values.items()}
        return gathered
    elif isinstance(values, (list, tuple)):
        gathered = type(values)(_do_gather(v) for v in values)
        return gathered
    else:
        # if not a dict, list, tuple, expect a singular tensor
        assert isinstance(values, torch.Tensor)
        return _do_gather(values)


def gather_sequence_torch(values: TensorSeq, dst_rank, group=None, join_fn=torch.cat, join_dim=0):
    world_size = dist.get_world_size(group)
    this_rank = dist.get_rank(group)

    def _do_gather(tensor):
        tensor_list = None
        if this_rank == dst_rank:
            tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.gather(tensor, tensor_list, dst=dst_rank, group=group)
        return join_fn(tensor_list, dim=join_dim)

    if isinstance(values, dict):
        gathered = {k: _do_gather(v) for k, v in values.items()}
        return gathered
    elif isinstance(values, (list, tuple)):
        gathered = type(values)(_do_gather(v) for v in values)
        return gathered
    else:
        # if not a dict, list, tuple, expect a singular tensor
        assert isinstance(values, torch.Tensor)
        return _do_gather(values)


def all_gather_torch(value: TensorSeq, group=None, join_fn: Callable = None, join_dim=0):
    if isinstance(value, torch.Tensor):
        world_size = dist.get_world_size(group)
        out_tensors = [torch.empty_like(value) for _ in range(world_size)]
        dist.all_gather(out_tensors, value, group=group)
        if join_fn is not None:
            out_tensors = join_fn(out_tensors, dim=join_dim)
        return out_tensors
    elif isinstance(value, dict):
        return {k: all_gather_torch(v, group, join_fn, join_dim) for k, v in value.items()}
    elif isinstance(value, (tuple, list)):
        return type(value)(all_gather_torch(v, group, join_fn, join_dim) for v in value)


def gather_torch(value: TensorSeq, dst_rank=0, group=None, join_fn: Callable = None, join_dim=0):
    if isinstance(value, torch.Tensor):
        world_size = dist.get_world_size(group)
        this_rank = dist.get_rank()
        out_tensors = None
        if this_rank == dst_rank:
            out_tensors = [torch.empty_like(value) for _ in range(world_size)]
        dist.gather(value, out_tensors, dst=dst_rank, group=group)
        if join_fn is not None:
            out_tensors = join_fn(out_tensors, dim=join_dim)
        return out_tensors
    elif isinstance(value, dict):
        return {k: gather_torch(v, dst_rank, group, join_fn, join_dim) for k, v in value.items()}
    elif isinstance(value, (tuple, list)):
        return type(value)(gather_torch(v, dst_rank, group, join_fn, join_dim) for v in value)


def all_reduce_torch(value: TensorSeq, op=ReduceOp.SUM, average=False, group=None):
    if isinstance(value, torch.Tensor):
        dist.all_reduce(value, op=op, group=group)
        if average:
            value /= dist.get_world_size(group)
    elif isinstance(value, dict):
        return {k: all_reduce_torch(v, op=op, average=average, group=group) for k, v in value.items()}
    elif isinstance(value, (tuple, list)):
        return type(value)(all_reduce_torch(v, op=op, average=average, group=group) for v in value)


def broadcast_torch(value: TensorSeq, src_rank: int = 0, group=None):
    if isinstance(value, torch.Tensor):
        return dist.broadcast(value, src=src_rank, group=group)
    elif isinstance(value, dict):
        return {k: broadcast_torch(v, src_rank=src_rank, group=group) for k, v in value.items()}
    elif isinstance(value, (tuple, list)):
        return type(value)(broadcast_torch(v, src_rank=src_rank, group=group) for v in value)