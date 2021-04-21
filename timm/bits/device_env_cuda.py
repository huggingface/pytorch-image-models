import os
from contextlib import suppress

import torch
from torch.nn.parallel import DistributedDataParallel

from .device_env import DeviceEnv


def is_cuda_available():
    return torch.cuda.is_available()


class DeviceEnvCuda(DeviceEnv):

    def __init__(self, device_idx=None, local_rank=None, amp=False, memory_format=None):
        assert torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True
        self._local_rank = 0
        self._distributed = False
        self._world_size = 1
        self._global_rank = 0
        if 'WORLD_SIZE' in os.environ:
            self._distributed = int(os.environ['WORLD_SIZE']) > 1
        if self._distributed:
            if local_rank is None:
                lr = os.environ.get('LOCAL_RANK', None)
                if lr is None:
                    raise RuntimeError(
                        'At least one of LOCAL_RANK env variable or local_rank arg must be set to valid integer.')
                self._local_rank = lr
            else:
                self._local_rank = int(local_rank)
            self._device = torch.device('cuda:%d' % self._local_rank)
            torch.cuda.set_device(self._local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self._world_size = torch.distributed.get_world_size()
            self._global_rank = torch.distributed.get_rank()
        else:
            self._device = torch.device('cuda' if device_idx is None else f'cuda:{device_idx}')
        self._memory_format = memory_format
        if amp:
            self._amp = amp
            self._autocast = torch.cuda.amp.autocast
        else:
            self._amp = amp
            self._autocast = suppress

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
        return self._local_rank == 0

    @property
    def type(self) -> str:
        return 'cuda'

    @property
    def amp(self) -> bool:
        return self._amp

    @property
    def autocast(self):
        return self._autocast

    def wrap_distributed(self, *modules, **kwargs):
        wrapped = [DistributedDataParallel(m, device_ids=[self._local_rank], **kwargs) for m in modules]
        return wrapped[0] if len(wrapped) == 1 else wrapped

    def to_device(self, *modules: torch.nn.Module):
        # FIXME handling dtype / memformat... disable flags, enable flags, diff fn?
        moved = [m.to(device=self._device, memory_format=self._memory_format) for m in modules]
        return moved[0] if len(moved) == 1 else moved
