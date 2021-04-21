from .device_env_cuda import DeviceEnvCuda, is_cuda_available
from .device_env_xla import DeviceEnvXla, is_xla_available

_device_env = None


def initialize_device(force_cpu: bool = False, xla_device_type=None, **kwargs):
    global _device_env
    if _device_env is not None:
        # warning
        return _device_env

    denv = None
    if not force_cpu:
        if is_xla_available(xla_device_type):
            # XLA supports more than just TPU, but by default will only look at TPU
            denv = DeviceEnvXla(**kwargs, xla_device_type=xla_device_type)
        elif is_cuda_available():
            denv = DeviceEnvCuda(**kwargs)

    if denv is None:
        # FIXME implement CPU support
        raise NotImplementedError()

    _device_env = denv
    return denv


def get_device():
    if _device_env is None:
        raise RuntimeError('Please initialize device environment by calling initialize_device first.')
    return _device_env


