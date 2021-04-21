import os
from contextlib import suppress
import torch

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.amp as xa
    _HAS_XLA = True
except ImportError as e:
    xm = None
    _HAS_XLA = False

from .device_env import DeviceEnv


def is_xla_available(xla_device_type=None):
    if not _HAS_XLA:
        return False
    supported_devs = xm.get_xla_supported_devices(devkind=xla_device_type)
    print(supported_devs)
    return len(supported_devs) >= 1


class DeviceEnvXla(DeviceEnv):

    def __init__(self, xla_device_type=None, device_idx=None, local_rank=0, amp=False):
        self._device = xm.xla_device(n=device_idx, devkind=xla_device_type)
        print(self._device)
        self._local_rank = xm.get_local_ordinal(local_rank)
        self._world_size = xm.xrt_world_size()
        self._distributed = self._world_size > 1
        self._global_rank = 0
        if self._distributed:
            self._global_rank = xm.get_ordinal()
        if amp:
            self._autocast = xa.autocast
        else:
            self._autocast = suppress
        self._memory_format = None

    @property
    def device(self):
        return self._device

    @property
    def local_rank(self):
        return self._local_rank

    @property
    def global_rank(self):
        return self._global_rank

    @property
    def is_distributed(self):
        return self._distributed

    @property
    def world_size(self):
        return self._world_size

    @property
    def is_master(self):
        return self._global_rank == 0

    @property
    def type(self) -> str:
        return 'xla'

    @property
    def amp(self) -> bool:
        return False

    @property
    def autocast(self):
        return self._autocast

    def wrap_distributed(self, *modules):
        # NO-OP
        return tuple([m for m in modules])

    def to_device(self, *modules: torch.nn.Module):
        return [m.to(device=self._device, memory_format=self._memory_format) for m in modules]

    def mark_step(self):
        xm.mark_step()
