"""timm.engine - Training infrastructure for PyTorch Image Models.

This module provides modular, reusable components for training image models,
including configuration dataclasses, device setup, data loading, optimizer
creation, and training/validation loops.

Example usage::

    from simple_parsing import ArgumentParser
    from timm.engine import (
        TrainConfig,
        setup_device,
        create_train_model,
        create_train_loader,
        create_train_optimizer,
        train_one_epoch,
        validate,
    )

    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(TrainConfig, dest='cfg')
    args = parser.parse_args()
    cfg = args.cfg

    # Setup
    device_env = setup_device(cfg.device)
    model, data_config = create_train_model(cfg.model, cfg.device, device_env)
    loader_train, mixup_fn, _ = create_train_loader(cfg, data_config, device_env, num_classes)
    optimizer = create_train_optimizer(model, cfg.optimizer, device_env, cfg.loader.batch_size)

    # Train
    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(epoch, task, loader_train, optimizer, device_env, cfg)
"""

# Configuration dataclasses
from .config import (
    AugmentConfig,
    DataConfig,
    DataSourceConfig,
    DeviceConfig,
    DistillationConfig,
    EMAConfig,
    LoaderConfig,
    LossConfig,
    MiscConfig,
    MixupConfig,
    ModelConfig,
    NaFlexConfig,
    OptimizerConfig,
    ProbeDataConfig,
    SchedulerConfig,
    SSLConfig,
    TrainConfig,
    TrainDataConfig,
)

# Device and distributed setup
from .device import (
    DeviceEnv,
    device_synchronize,
    is_primary,
    setup_device,
    synchronize,
)

# Model creation and EMA
from .model import (
    create_train_model,
    get_model_num_classes,
    get_naflex_patch_size,
    setup_model_ema,
)

# Data loading
from .data import (
    create_eval_dataset,
    create_eval_loader,
    create_multiview_eval_loader,
    create_multiview_train_loader,
    create_train_dataset,
    create_train_loader,
    get_num_classes_from_dataset,
)

# Optimizer and scheduler
from .optim import (
    create_train_optimizer,
    create_train_scheduler,
    get_optimizer_lr,
)

# Training loop
from .train import train_one_epoch

# Validation loop
from .validate import validate, validate_with_task, validate_knn

# Checkpoint management
from .checkpoint import (
    get_output_dir,
    load_pretrained,
    resume_training,
    save_config,
    setup_checkpoint_saver,
)

__all__ = [
    # Configs
    'AugmentConfig',
    'DataConfig',
    'DataSourceConfig',
    'DeviceConfig',
    'DistillationConfig',
    'EMAConfig',
    'LoaderConfig',
    'LossConfig',
    'MiscConfig',
    'MixupConfig',
    'ModelConfig',
    'NaFlexConfig',
    'OptimizerConfig',
    'ProbeDataConfig',
    'SchedulerConfig',
    'SSLConfig',
    'TrainConfig',
    'TrainDataConfig',
    # Device
    'DeviceEnv',
    'setup_device',
    'is_primary',
    'synchronize',
    'device_synchronize',
    # Model
    'create_train_model',
    'setup_model_ema',
    'get_model_num_classes',
    'get_naflex_patch_size',
    # Data
    'create_train_dataset',
    'create_eval_dataset',
    'create_train_loader',
    'create_multiview_train_loader',
    'create_eval_loader',
    'create_multiview_eval_loader',
    'get_num_classes_from_dataset',
    # Optimizer
    'create_train_optimizer',
    'create_train_scheduler',
    'get_optimizer_lr',
    # Training
    'train_one_epoch',
    # Validation
    'validate',
    'validate_with_task',
    'validate_knn',
    # Checkpoint
    'setup_checkpoint_saver',
    'resume_training',
    'load_pretrained',
    'save_config',
    'get_output_dir',
]
