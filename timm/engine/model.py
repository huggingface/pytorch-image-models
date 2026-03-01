"""Model creation and setup utilities for timm engine.

Provides functions for creating models, configuring them for training,
and setting up model EMA (Exponential Moving Average).
"""
import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from timm import create_model
from timm.data import resolve_data_config
from timm.layers import convert_splitbn_model
from timm.models import load_checkpoint, safe_model_name
from timm.utils import ModelEmaV3

from .config import DeviceConfig, EMAConfig, ModelConfig
from .device import DeviceEnv, is_primary

_logger = logging.getLogger(__name__)

try:
    from torch.nn import SyncBatchNorm
    from torch.nn.modules.batchnorm import _BatchNorm as TorchBatchNorm
    HAS_SYNC_BN = True
except ImportError:
    HAS_SYNC_BN = False


def convert_sync_batchnorm(model: nn.Module) -> nn.Module:
    """Convert BatchNorm layers to SyncBatchNorm.

    Args:
        model: Model with BatchNorm layers.

    Returns:
        Model with SyncBatchNorm layers.
    """
    if HAS_SYNC_BN:
        return SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def create_train_model(
    cfg: ModelConfig,
    device_cfg: DeviceConfig,
    device_env: DeviceEnv,
    num_classes: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Create and configure a model for training.

    This function handles:
    - Model instantiation with appropriate settings
    - Head initialization (scale/bias)
    - Gradient checkpointing
    - BatchNorm configuration (split BN, sync BN)
    - Device and dtype placement
    - TorchScript compilation (if requested)

    Args:
        cfg: Model configuration.
        device_cfg: Device configuration (for sync BN, channels last, etc.).
        device_env: Device environment.
        num_classes: Override number of classes (uses model default if None).

    Returns:
        Tuple of (model, data_config) where data_config contains:
        - input_size: Tuple of (channels, height, width)
        - mean: Normalization mean
        - std: Normalization std
        - interpolation: Resize interpolation method
        - crop_pct: Center crop percentage

    Example::

        from timm.engine import ModelConfig, DeviceConfig, setup_device, create_train_model

        model_cfg = ModelConfig(model='resnet50', pretrained=True)
        device_cfg = DeviceConfig(amp=True)
        device_env = setup_device(device_cfg)
        model, data_config = create_train_model(model_cfg, device_cfg, device_env)
    """
    # Resolve input channels
    in_chans = 3
    if cfg.in_chans is not None:
        in_chans = cfg.in_chans
    elif cfg.input_size is not None:
        in_chans = cfg.input_size[0]

    # Build factory kwargs for pretrained loading
    factory_kwargs = {}
    if cfg.pretrained_path:
        factory_kwargs['pretrained_cfg_overlay'] = dict(
            file=cfg.pretrained_path,
            num_classes=-1,  # force head adaptation
        )

    # Resolve num_classes
    effective_num_classes = num_classes if num_classes is not None else cfg.num_classes

    # Create model
    model = create_model(
        cfg.model,
        pretrained=cfg.pretrained,
        in_chans=in_chans,
        num_classes=effective_num_classes,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
        drop_block_rate=cfg.drop_block_rate,
        global_pool=cfg.global_pool,
        bn_momentum=cfg.bn_momentum,
        bn_eps=cfg.bn_eps,
        scriptable=cfg.torchscript,
        checkpoint_path=cfg.initial_checkpoint,
        **factory_kwargs,
        **cfg.model_kwargs,
    )

    # Head initialization
    if cfg.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(cfg.head_init_scale)
            model.get_classifier().bias.mul_(cfg.head_init_scale)

    if cfg.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, cfg.head_init_bias)

    # Gradient checkpointing
    if cfg.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if is_primary(device_env):
        param_count = sum(p.numel() for p in model.parameters())
        _logger.info(
            f'Model {safe_model_name(cfg.model)} created, param count: {param_count:,}'
        )

    # Resolve data config from model
    data_config = resolve_data_config(
        {
            'input_size': cfg.input_size,
            'img_size': cfg.img_size,
            'crop_pct': cfg.crop_pct,
            'mean': cfg.mean,
            'std': cfg.std,
            'interpolation': cfg.interpolation,
        },
        model=model,
        verbose=is_primary(device_env),
    )

    # Split BatchNorm if configured
    if device_cfg.split_bn:
        model = convert_splitbn_model(model, 2)

    # Move model to device
    model.to(device=device_env.device, dtype=device_env.model_dtype)

    # Channels last memory format
    if device_cfg.channels_last:
        model.to(memory_format=torch.channels_last)

    # Synchronized BatchNorm for distributed training
    if device_env.distributed and device_cfg.sync_bn:
        model = convert_sync_batchnorm(model)
        if is_primary(device_env):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have '
                'issues if using zero initialized BN layers (enabled by default for '
                'ResNets) while sync-bn enabled.'
            )

    # TorchScript compilation
    if cfg.torchscript:
        assert not cfg.torchcompile, 'Cannot use both torchscript and torchcompile'
        assert not device_cfg.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    return model, data_config


def setup_model_ema(
    model: nn.Module,
    cfg: EMAConfig,
    device_env: DeviceEnv,
    resume_path: str = '',
) -> Optional[ModelEmaV3]:
    """Setup model EMA (Exponential Moving Average).

    EMA maintains a shadow copy of model weights that are updated as an
    exponential moving average of the training weights. This often improves
    validation performance.

    Args:
        model: The training model.
        cfg: EMA configuration.
        device_env: Device environment.
        resume_path: Path to checkpoint for resuming EMA weights.

    Returns:
        ModelEmaV3 instance if EMA is enabled, None otherwise.

    Example::

        model_ema = setup_model_ema(model, ema_cfg, device_env, resume_path='checkpoint.pth')
        if model_ema is not None:
            # Use model_ema.module for evaluation
            eval_model = model_ema.module
    """
    if not cfg.model_ema:
        return None

    # Create EMA model
    # Important: Create after moving to device and AMP setup, but before DDP
    model_ema = ModelEmaV3(
        model,
        decay=cfg.model_ema_decay,
        use_warmup=cfg.model_ema_warmup,
        device='cpu' if cfg.model_ema_force_cpu else None,
    )

    # Load EMA weights from checkpoint if resuming
    if resume_path:
        load_checkpoint(model_ema.module, resume_path, use_ema=True)

    if is_primary(device_env):
        _logger.info(
            f'Model EMA enabled with decay={cfg.model_ema_decay}, '
            f'warmup={cfg.model_ema_warmup}, '
            f'force_cpu={cfg.model_ema_force_cpu}'
        )

    return model_ema


def get_model_num_classes(model: nn.Module) -> int:
    """Get number of classes from model.

    Args:
        model: Model instance.

    Returns:
        Number of output classes.

    Raises:
        AssertionError: If model doesn't have num_classes attribute.
    """
    assert hasattr(model, 'num_classes'), \
        'Model must have `num_classes` attr if not set on cmd line/config.'
    return model.num_classes


def get_naflex_patch_size(model: nn.Module) -> Optional[int]:
    """Get patch size from NaFlex-compatible model.

    Args:
        model: Model instance.

    Returns:
        Patch size if model has embeds.patch_size attribute, None otherwise.
    """
    return getattr(getattr(model, 'embeds', None), 'patch_size', None)
