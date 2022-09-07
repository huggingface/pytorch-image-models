from typing import Callable, Optional, Union, Any

import torch

from .device_env import DeviceEnv, DeviceEnvType
from .updater import Updater
from .updater_cuda import UpdaterCudaWithScaler
from .updater_deepspeed import UpdaterDeepSpeed
from .updater_xla import UpdaterXla, UpdaterXlaWithScaler


def create_updater(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        clip_fn: Optional[Union[Callable, str]] = None,
        clip_value: Optional[float] = None,
        scaler_kwargs: Any = None,
        dev_env: Optional[DeviceEnv] = None,
        deepspeed: bool = False,
) -> Updater:

    if not dev_env:
        dev_env = DeviceEnv.instance()

    updater_kwargs = dict(model=model, optimizer=optimizer, clip_fn=clip_fn, clip_value=clip_value)
    use_scaler = dev_env.amp
    if use_scaler:
        updater_kwargs['scaler_kwargs'] = scaler_kwargs
    updater_cls = Updater
    if dev_env.type == DeviceEnvType.XLA:
        updater_cls = UpdaterXlaWithScaler if use_scaler else UpdaterXla
    elif dev_env.type == DeviceEnvType.CUDA and use_scaler:
        updater_cls = UpdaterCudaWithScaler
    elif deepspeed:
        del updater_kwargs['scaler_kwargs']
        updater_cls = UpdaterDeepSpeed

    return updater_cls(**updater_kwargs)
