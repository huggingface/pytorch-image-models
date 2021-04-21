from typing import Callable, Optional, Union, Any

import torch

from .device_env import DeviceEnv
from .device_env_factory import get_device
from .updater import Updater
from .updater_cuda import UpdaterCuda
from .updater_xla import UpdaterXla


def create_updater(
        optimizer: torch.optim.Optimizer,
        dev_env: Optional[DeviceEnv] = None,
        clip_value: Optional[Union[Callable, float]] = None,
        clip_mode: str = 'norm',
        scaler_kwargs: Any = None) -> Updater:

    if not dev_env:
        dev_env = get_device()

    updater_kwargs = dict(
        optimizer=optimizer, clip_value=clip_value, clip_mode=clip_mode, scaler_kwargs=scaler_kwargs)
    if dev_env.type == 'xla':
        return UpdaterXla(**updater_kwargs, use_scaler=dev_env.amp)
    elif dev_env.type == 'cuda':
        return UpdaterCuda(**updater_kwargs, use_scaler=dev_env.amp)
    else:
        updater_kwargs.pop('scaler_kwargs', None)
        return Updater(**updater_kwargs)
