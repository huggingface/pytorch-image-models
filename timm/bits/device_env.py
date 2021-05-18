import abc
from contextlib import suppress
from enum import Enum
from typing import Callable, Union, Optional, List, Tuple
from dataclasses import dataclass, field, InitVar

import torch
import torch.distributed as dist

TensorList = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]


class DeviceEnvType(Enum):
    """ Device Environment Types
    """
    CPU = "cpu"
    CUDA = "cuda"
    XLA = "xla"


@dataclass
class DeviceEnv:
    device_type: InitVar[Optional[str]] = None
    device_index: InitVar[Optional[int]] = None

    device: torch.device = field(init=False)  # set from device_type + device_index or post_init logic
    world_size: Optional[int] = None  # set by post_init from env when None
    local_rank: Optional[int] = None  # set by post_init from env when None
    global_rank: Optional[int] = None  # set by post_init from env when None
    amp: bool = False
    autocast: Optional[Callable] = None  # set by post_init from env when None
    memory_format: Optional[torch.memory_format] = None
    dtype: Optional[torch.dtype] = None

    def __post_init__(self, device_type: Optional[str], device_index: Optional[int]):
        device_type = device_type or 'cpu'
        self.device = torch.device(device_type) if device_index is None \
            else torch.device(device_type, device_index)
        self.world_size = 1 if self.world_size is None else self.world_size
        self.local_rank = 0 if self.local_rank is None else self.local_rank
        self.global_rank = 0 if self.global_rank is None else self.global_rank
        if self.autocast is None:
            self.autocast = suppress

    @property
    def type(self) -> DeviceEnvType:
        if self.device.type == 'cpu':
            return DeviceEnvType.CPU
        elif self.device.type == 'cuda':
            return DeviceEnvType.CUDA
        elif self.device.type == 'xla':
            return DeviceEnvType.XLA
        else:
            assert False, "Unexpected device type for base DevEnv impl."

    @property
    def type_cuda(self):
        # shortcut for common cuda device type
        return self.type == DeviceEnvType.CUDA

    @property
    def type_xla(self):
        # shortcut for common xla device type
        return self.type == DeviceEnvType.XLA

    @property
    def distributed(self):
        return self.world_size > 1

    @property
    def primary(self):
        return self.local_rank == 0

    @property
    def global_primary(self):
        return self.global_rank == 0

    def wrap_distributed(self, *modules):
        pass

    def wrap_parallel(self, *modules):
        pass

    def to_device(self, *modules: torch.nn.Module):
        # FIXME handling dtype / memformat... disable flags, enable flags, diff fn?
        moved = [m.to(device=self.device, memory_format=self.memory_format) for m in modules]
        return moved[0] if len(moved) == 1 else moved

    def mark_step(self):
        pass  # NO-OP for non-XLA devices

    def all_reduce_(self, tensor: TensorList, op=dist.ReduceOp.SUM, average=False):
        dist.all_reduce(tensor, op=op)
        if average:
            tensor.div_(self.world_size)
        return tensor

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM, average=False):
        reduce_tensor = tensor.clone()
        dist.all_reduce(reduce_tensor, op=op)
        if average:
            reduce_tensor = reduce_tensor / self.world_size
        return reduce_tensor

    def all_gather(self, tensor: torch.Tensor, cat_dim=0):
        output_tensors = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(output_tensors, tensor)
        return torch.cat(output_tensors, cat_dim)

    def all_to_all(self, tensor: torch.Tensor, num_splits, split_dim, cat_dim=0):
        input_tensors = torch.chunk(tensor, num_splits, split_dim)
        output_tensors = [torch.empty_like(input_tensors[0]) for _ in range(self.world_size)]
        dist.all_to_all(output_tensors, input_tensors)
        return torch.cat(output_tensors, cat_dim)

    def broadcast_(self, tensor: torch.Tensor, src_rank=0):
        dist.broadcast(tensor, src=src_rank)
        return tensor

    def broadcast(self, tensor: Optional[torch.Tensor] = None, src_rank=0):
        if self.global_rank != src_rank:
            tensor = torch.empty_like(tensor)
        assert tensor is not None
        dist.broadcast(tensor, src=src_rank)
        return tensor

    def barrier(self):
        dist.barrier()
