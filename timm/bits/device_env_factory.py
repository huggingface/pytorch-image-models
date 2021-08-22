import logging

from .device_env import DeviceEnv, is_global_device, get_global_device, set_global_device
from .device_env_cuda import DeviceEnvCuda, is_cuda_available
from .device_env_xla import DeviceEnvXla, is_xla_available

_logger = logging.getLogger(__name__)


def initialize_device(force_cpu: bool = False, **kwargs) -> DeviceEnv:
    if is_global_device():
        return get_global_device()

    denv = None
    if not force_cpu:
        xla_device_type = kwargs.get('xla_device_type', None)
        if is_xla_available(xla_device_type):
            # XLA supports more than just TPU, will search in order TPU, GPU, CPU
            denv = DeviceEnvXla(**kwargs)
        elif is_cuda_available():
            denv = DeviceEnvCuda(**kwargs)

    # CPU fallback
    if denv is None:
        if is_xla_available('CPU'):
            denv = DeviceEnvXla(device_type='CPU', **kwargs)
        else:
            denv = DeviceEnv()

    _logger.info(f'Initialized device {denv.device}. '
                 f'Rank: {denv.global_rank} ({denv.local_rank}) of {denv.world_size}.')
    print(denv)  # FIXME temporary print for debugging

    set_global_device(denv)
    return denv

