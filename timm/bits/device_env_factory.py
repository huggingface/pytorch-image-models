from .device_env import DeviceEnv
from .device_env_cuda import DeviceEnvCuda, is_cuda_available
from .device_env_xla import DeviceEnvXla, is_xla_available

_device_env = None


def initialize_device(force_cpu: bool = False, **kwargs) -> DeviceEnv:
    global _device_env
    if _device_env is not None:
        # warning
        return _device_env

    denv = None
    if not force_cpu:
        xla_device_type = kwargs.get('xla_device_type', None)
        if is_xla_available(xla_device_type):
            # XLA supports more than just TPU, will search in order TPU, GPU, CPU
            denv = DeviceEnvXla(**kwargs)
        elif is_cuda_available():
            denv = DeviceEnvCuda(**kwargs)

    if denv is None:
        denv = DeviceEnv()

    print(denv)  # FIXME DEBUG
    _device_env = denv
    return denv


def get_device() -> DeviceEnv:
    if _device_env is None:
        raise RuntimeError('Please initialize device environment by calling initialize_device first.')
    return _device_env


