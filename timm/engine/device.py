"""Device and distributed training setup for timm engine.

Provides utilities for initializing devices, distributed training,
and automatic mixed precision (AMP).
"""
import importlib
import logging
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import torch

from timm.utils import NativeScaler, init_distributed_device, random_seed, set_jit_fuser

from .config import DeviceConfig

_logger = logging.getLogger(__name__)

try:
    from timm.layers import set_fast_norm
    HAS_FAST_NORM = True
except ImportError:
    HAS_FAST_NORM = False


@dataclass
class DeviceEnv:
    """Container for device and distributed environment state.

    This dataclass holds all device-related state needed throughout training,
    including device, distributed training info, AMP settings, and loss scaler.

    Attributes:
        device: The torch device to use for training.
        world_size: Total number of processes in distributed training.
        rank: Global rank of this process.
        local_rank: Local rank within the node.
        distributed: Whether distributed training is enabled.
        amp_autocast: Context manager for automatic mixed precision.
        loss_scaler: Loss scaler for float16 AMP training, None otherwise.
        model_dtype: Optional dtype override for model parameters.
        amp_dtype: The dtype used for AMP autocast.
    """
    device: torch.device
    world_size: int
    rank: int
    local_rank: int
    distributed: bool
    amp_autocast: Callable[..., Any]
    loss_scaler: Optional[NativeScaler]
    model_dtype: Optional[torch.dtype]
    amp_dtype: Optional[torch.dtype] = None


def setup_device(cfg: DeviceConfig, seed: int = 42) -> DeviceEnv:
    """Initialize device, distributed training, and AMP settings.

    This function handles:
    - Loading custom device backend modules
    - Enabling CUDA optimizations (TF32, cuDNN benchmark)
    - Initializing distributed training
    - Setting up AMP autocast and loss scaler
    - Setting random seeds

    Args:
        cfg: Device configuration dataclass.
        seed: Random seed for reproducibility.

    Returns:
        DeviceEnv containing all device-related state.

    Example::

        from timm.engine import DeviceConfig, setup_device

        cfg = DeviceConfig(device='cuda', amp=True, amp_dtype='bfloat16')
        device_env = setup_device(cfg, seed=42)
        model = model.to(device_env.device, dtype=device_env.model_dtype)
    """
    # Import custom device modules if specified
    if cfg.device_modules:
        for module in cfg.device_modules:
            importlib.import_module(module)

    # Enable CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Create a namespace-like object for init_distributed_device
    class Args:
        pass

    args = Args()
    args.device = cfg.device
    args.world_size = 1
    args.rank = 0
    args.local_rank = cfg.local_rank
    args.dist_backend = 'nccl'
    args.dist_url = 'env://'
    args.distributed = False

    device = init_distributed_device(args)

    # Extract distributed info from args (init_distributed_device modifies it)
    distributed = getattr(args, 'distributed', False)
    world_size = getattr(args, 'world_size', 1)
    rank = getattr(args, 'rank', 0)
    local_rank = getattr(args, 'local_rank', 0)

    # Set random seed
    random_seed(seed, rank)

    # Set JIT fuser if specified
    if cfg.fuser:
        set_jit_fuser(cfg.fuser)

    # Enable fast norm if specified
    if cfg.fast_norm and HAS_FAST_NORM:
        set_fast_norm()

    # Resolve model dtype
    model_dtype = None
    if cfg.model_dtype:
        assert cfg.model_dtype in ('float32', 'float16', 'bfloat16'), \
            f"Invalid model_dtype: {cfg.model_dtype}"
        model_dtype = getattr(torch, cfg.model_dtype)
        if model_dtype == torch.float16:
            _logger.warning(
                'float16 is not recommended for training, '
                'for half precision bfloat16 is recommended.'
            )

    # Setup AMP
    amp_autocast = suppress  # no-op context manager
    amp_dtype = None
    loss_scaler = None

    if cfg.amp:
        assert model_dtype is None or model_dtype == torch.float32, \
            'float32 model dtype must be used with AMP'
        assert cfg.amp_dtype in ('float16', 'bfloat16'), \
            f"Invalid amp_dtype: {cfg.amp_dtype}"

        amp_dtype = torch.float16 if cfg.amp_dtype == 'float16' else torch.bfloat16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)

        # Loss scaler only needed for float16, not bfloat16
        if device.type in ('cuda',) and amp_dtype == torch.float16:
            loss_scaler = NativeScaler(device=device.type)

        if is_primary_from_rank(rank):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if is_primary_from_rank(rank):
            _logger.info(f'AMP not enabled. Training in {model_dtype or torch.float32}.')

    return DeviceEnv(
        device=device,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed=distributed,
        amp_autocast=amp_autocast,
        loss_scaler=loss_scaler,
        model_dtype=model_dtype,
        amp_dtype=amp_dtype,
    )


def is_primary(device_env: DeviceEnv, local: bool = False) -> bool:
    """Check if this is the primary process.

    In distributed training, only the primary process should perform
    certain operations like logging, checkpointing, and evaluation.

    Args:
        device_env: Device environment containing rank info.
        local: If True, check if primary within the local node (local_rank == 0).
               If False, check if globally primary (rank == 0).

    Returns:
        True if this is the primary process.
    """
    if local:
        return device_env.local_rank == 0
    return device_env.rank == 0


def is_primary_from_rank(rank: int) -> bool:
    """Check if primary process from rank value.

    Utility function for use during setup before DeviceEnv is created.

    Args:
        rank: Global rank value.

    Returns:
        True if rank == 0.
    """
    return rank == 0


def synchronize(device_env: DeviceEnv) -> None:
    """Synchronize all processes.

    Args:
        device_env: Device environment.
    """
    if device_env.distributed:
        torch.distributed.barrier()


def device_synchronize(device_env: DeviceEnv) -> None:
    """Synchronize device operations (e.g., CUDA synchronize).

    Args:
        device_env: Device environment.
    """
    if device_env.device.type == 'cuda':
        torch.cuda.synchronize(device_env.device)
