import os
from contextlib import suppress
from dataclasses import dataclass, field, InitVar
from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel, DataParallel

from .device_env import DeviceEnv, DeviceEnvType, TensorList


def is_cuda_available():
    return torch.cuda.is_available()


@dataclass
class DeviceEnvCuda(DeviceEnv):

    def __post_init__(
            self,
            device_type: Optional[str],
            device_index: Optional[int],
            channels_last: bool,
    ):
        assert torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True
        setup_world_size = self.world_size or int(os.environ.get('WORLD_SIZE', 1))
        assert setup_world_size
        if setup_world_size > 1:
            # setup distributed
            assert device_index is None
            if self.local_rank is None:
                lr = os.environ.get('LOCAL_RANK', None)
                if lr is None:
                    raise RuntimeError(
                        'At least one of LOCAL_RANK env variable or local_rank arg must be set to valid integer.')
                self.local_rank = int(lr)
            self.device = torch.device('cuda:%d' % self.local_rank)
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.world_size = torch.distributed.get_world_size()
            assert self.world_size == setup_world_size
            self.global_rank = torch.distributed.get_rank()
        else:
            self.device = torch.device('cuda' if device_index is None else f'cuda:{device_index}')
            self.local_rank = 0
            self.world_size = 1
            self.global_rank = 0
        if self.autocast is None:
            self.autocast = torch.cuda.amp.autocast if self.amp else suppress
        if channels_last:
            self.memory_format = torch.channels_last

    @property
    def type(self) -> DeviceEnvType:
        return DeviceEnvType.CUDA

    def wrap_distributed(self, *modules, **kwargs):
        wrapped = [DistributedDataParallel(m, device_ids=[self.local_rank], **kwargs) for m in modules]
        return wrapped[0] if len(wrapped) == 1 else wrapped

    def wrap_parallel(self, *modules, **kwargs):
        assert not self.distributed
        wrapped = [DataParallel(m, **kwargs) for m in modules]
        return wrapped[0] if len(wrapped) == 1 else wrapped

    def synchronize(self, tensors: Optional[TensorList] = None):
        torch.cuda.synchronize(self.device)
