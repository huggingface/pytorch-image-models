import os
from contextlib import suppress
from dataclasses import dataclass, field, InitVar
from typing import Optional, Dict

import torch
from torch.distributed import ReduceOp

try:
    import torch_xla.core.xla_model as xm
    import torch_xla
    _HAS_XLA = True
except ImportError as e:
    xm = None
    torch_xla = None
    _HAS_XLA = False

try:
    # only the very latest XLA builds have AMP
    import torch_xla.amp as xa
except ImportError as e:
    xa = None

from .device_env import DeviceEnv, DeviceEnvType, TensorList


_PT_TO_XM_OP = {
    ReduceOp.SUM: 'sum',
    ReduceOp.PRODUCT: 'mul',
    ReduceOp.MIN: 'min',
    ReduceOp.MAX: 'max',
    ReduceOp.BAND: 'and',
    ReduceOp.BOR: 'or',
}


def is_xla_available(xla_device_type=None):
    if not _HAS_XLA:
        return False
    supported_devs = xm.get_xla_supported_devices(devkind=xla_device_type)
    return len(supported_devs) >= 1


@dataclass
class DeviceEnvXla(DeviceEnv):

    def __post_init__(
            self,
            device_type: Optional[str],
            device_idx: Optional[int],
            channels_last: bool,
    ):
        if device_type is not None:
            device_type = device_type.upper()
            assert device_type in ('TPU', 'GPU', 'CPU'), "XLA device type must be one of ('TPU', 'GPU', 'CPU')"
        self.device = xm.xla_device(n=device_idx, devkind=device_type)
        self.world_size = xm.xrt_world_size()
        if self.distributed:
            assert device_idx is None, "device_index is based on local rank for distributed XLA mode"
            self.local_rank = xm.get_local_ordinal()
            self.global_rank = xm.get_ordinal()
        else:
            self.local_rank = 0
            self.global_rank = 0
        if self.amp:
            assert xa is not None, 'XLA AMP is not present on this build'
        if self.autocast is None:
            self.autocast = xa.autocast if self.amp else suppress
        if channels_last:
            self.memory_format = torch.channels_last

    @property
    def type(self) -> DeviceEnvType:
        return DeviceEnvType.XLA

    def wrap_distributed(self, *modules):
        wrapped = [m for m in modules]  # NO-OP
        return wrapped[0] if len(wrapped) == 1 else wrapped

    def wrap_parallel(self, *modules):
        assert False, "Not implemented"

    def mark_step(self):
        xm.mark_step()

    def synchronize(self, tensors: Optional[TensorList] = None):
        torch_xla._XLAC._xla_sync_multi(tensors, devices=[], wait=True, sync_xla_data=True)

    def all_reduce(self, tensor: torch.Tensor, op=ReduceOp.SUM, average=False):
        assert isinstance(tensor, torch.Tensor)  # unlike in-place variant, lists/tuples not allowed
        op = _PT_TO_XM_OP[op]
        scale = 1.0 / self.world_size if average else 1.0
        return xm.all_reduce(op, tensor, scale=scale)

    def all_reduce_(self, tensor: TensorList, op=ReduceOp.SUM, average=False):
        op = _PT_TO_XM_OP[op]
        scale = 1.0 / self.world_size if average else 1.0
        wrapped = False
        if isinstance(tensor, torch.Tensor):
            tensor = [tensor]  # bare tensors are not operated on in-place
            wrapped = True
        xm.all_reduce(op, tensor, scale=scale)
        if wrapped:
            tensor = tensor[0]
        return tensor

    def all_gather(self, tensor: torch.Tensor, cat_dim=0):
        output = xm.all_gather(tensor, cat_dim)
        return output

    def all_to_all(self, tensor, num_splits, split_dim, cat_dim=0):
        output = xm.all_to_all(tensor, split_dim, cat_dim, num_splits)
        return output

    def broadcast(self, tensor: torch.Tensor, src_rank=0):
        if self.global_rank != src_rank:
            reduce_tensor = torch.zeros_like(tensor)
            xm.all_reduce('sum', reduce_tensor)
        else:
            xm.all_reduce('sum', tensor)
        return tensor

    def broadcast_(self, tensor: torch.Tensor, src_rank=0):
        out_tensor = self.broadcast(tensor, src_rank)
        return tensor.copy_(out_tensor)

    def barrier(self):
        xm.rendezvous('timm.bits.dist_barrier')

    def state_dict_to_cpu(self, state: Dict[str, torch.Tensor]):
        cpu_state = xm._maybe_convert_to_cpu(state, convert=True)
        return cpu_state

    def state_dict_to_device(self, state: Dict[str, torch.Tensor]):
        device_state = xm.send_cpu_data_to_device(state, device=self.device)
        return device_state
