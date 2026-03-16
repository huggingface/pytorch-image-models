#!/usr/bin/env python3
"""Classification training script using timm.engine infrastructure.

A modern, modular training script for image classification with support for:
- Classification with various loss functions
- Knowledge distillation (logit, feature, token)
- Automatic mixed precision (AMP)
- Distributed data parallel (DDP)
- Model EMA
- Mixup/CutMix augmentation
- Learning rate scheduling with warmup

CLI usage::

    # As a module
    python -m timm.apps.train_cls --model.model resnet50 --model.pretrained true \\
        --data.path /path/to/imagenet --scheduler.epochs 100

    # As installed script
    timm-train-cls --model.model resnet50 --data.path /path/to/imagenet

    # With config file
    python -m timm.apps.train_cls -c configs/resnet50.yaml --data.path /path/to/data

    # Knowledge distillation
    python -m timm.apps.train_cls --model.model resnet18 \\
        --distillation.kd_model_name resnet50 \\
        --distillation.kd_distill_type logit \\
        --data.path /path/to/imagenet

Programmatic usage (for testing)::

    from timm.apps.train_cls import train_cls
    from timm.engine import TrainConfig, ModelConfig, TrainDataConfig, SchedulerConfig

    cfg = TrainConfig(
        model=ModelConfig(model='resnet18', pretrained=True),
        data=TrainDataConfig(path='/path/to/data'),
        scheduler=SchedulerConfig(epochs=5),
    )
    best_metric = train_cls(cfg)
"""
import logging
import os
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import yaml

from timm import utils
from timm.loss import (
    BinaryCrossEntropy,
    JsdCrossEntropy,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)
from timm.models import safe_model_name
from timm.task import (
    ClassificationTask,
    ClassificationEvalTask,
    FeatureDistillationTask,
    LogitDistillationTask,
    TokenDistillationTask,
)

from timm.engine import (
    TrainConfig,
    DeviceEnv,
    setup_device,
    is_primary,
    create_train_model,
    create_train_loader,
    create_eval_loader,
    create_train_optimizer,
    create_train_scheduler,
    train_one_epoch,
    validate,
    validate_with_task,
    setup_checkpoint_saver,
    resume_training,
    save_config,
    get_output_dir,
    get_model_num_classes,
    get_naflex_patch_size,
)

_logger = logging.getLogger('train_cls')


def parse_args() -> TrainConfig:
    """Parse command line arguments into TrainConfig dataclass."""
    try:
        from simple_parsing import ArgumentParser
    except ImportError:
        raise ImportError(
            "simple-parsing is required for train_cls. "
            "Install with: pip install simple-parsing"
        )

    parser = ArgumentParser(
        description='Classification Training with timm.engine',
        add_help=True,
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='',
        help='YAML config file with default values',
    )
    parser.add_arguments(TrainConfig, dest='cfg')

    args = parser.parse_args()

    # Load YAML config if specified and merge
    if args.config:
        with open(args.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        # simple-parsing handles this via config file support
        # For manual merge, we'd recursively update the dataclass
        _logger.info(f'Loaded config from: {args.config}')

    return args.cfg


def create_loss_fn(
    cfg: TrainConfig,
    mixup_active: bool,
    num_aug_splits: int,
    device: torch.device,
) -> nn.Module:
    """Create training loss function based on configuration."""
    if cfg.loss.jsd_loss:
        assert num_aug_splits > 1, 'JSD loss requires aug_splits > 1'
        train_loss_fn = JsdCrossEntropy(
            num_splits=num_aug_splits,
            smoothing=cfg.mixup.smoothing,
        )
    elif mixup_active:
        if cfg.loss.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                target_threshold=cfg.loss.bce_target_thresh,
                sum_classes=cfg.loss.bce_sum,
                pos_weight=cfg.loss.bce_pos_weight,
            )
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif cfg.mixup.smoothing:
        if cfg.loss.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=cfg.mixup.smoothing,
                target_threshold=cfg.loss.bce_target_thresh,
                sum_classes=cfg.loss.bce_sum,
                pos_weight=cfg.loss.bce_pos_weight,
            )
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=cfg.mixup.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()

    return train_loss_fn.to(device=device)


def create_task(
    model: nn.Module,
    cfg: TrainConfig,
    train_loss_fn: nn.Module,
    device_env: DeviceEnv,
) -> nn.Module:
    """Create training task based on configuration."""
    if cfg.distillation.kd_model_name is not None:
        # Knowledge distillation
        common_kwargs = dict(
            student_model=model,
            teacher_model=cfg.distillation.kd_model_name,
            criterion=train_loss_fn,
            distill_loss_weight=cfg.distillation.distill_loss_weight,
            task_loss_weight=cfg.distillation.task_loss_weight,
            device=device_env.device,
            dtype=device_env.model_dtype,
            verbose=is_primary(device_env),
        )

        distill_type = cfg.distillation.kd_distill_type

        if distill_type == 'logit':
            task = LogitDistillationTask(
                **common_kwargs,
                loss_type=cfg.distillation.kd_loss_type,
                temperature=cfg.distillation.kd_temperature,
            )
        elif distill_type == 'feature':
            task = FeatureDistillationTask(
                **common_kwargs,
                student_feature_dim=cfg.distillation.kd_student_feature_dim,
                teacher_feature_dim=cfg.distillation.kd_teacher_feature_dim,
            )
        elif distill_type == 'token':
            task = TokenDistillationTask(
                **common_kwargs,
                distill_type=cfg.distillation.kd_token_distill_type,
                temperature=cfg.distillation.kd_temperature,
            )
        else:
            raise ValueError(f"Unknown distillation type: {distill_type}")
    else:
        # Standard classification
        task = ClassificationTask(
            model=model,
            criterion=train_loss_fn,
            device=device_env.device,
            dtype=device_env.model_dtype,
            verbose=is_primary(device_env),
        )

    return task


def setup_wandb(cfg: TrainConfig, exp_name: str, device_env: DeviceEnv) -> None:
    """Initialize Weights & Biases logging if configured."""
    if not cfg.misc.log_wandb or not is_primary(device_env):
        return

    try:
        import wandb

        # Convert config to dict for wandb
        import dataclasses
        config_dict = dataclasses.asdict(cfg)

        wandb.init(
            project=cfg.misc.wandb_project,
            name=exp_name,
            config=config_dict,
            tags=cfg.misc.wandb_tags,
            resume='must' if cfg.misc.wandb_resume_id else None,
            id=cfg.misc.wandb_resume_id if cfg.misc.wandb_resume_id else None,
        )
    except ImportError:
        _logger.warning('wandb not installed, skipping')


def train_cls(cfg: TrainConfig) -> Optional[float]:
    """Run classification training with the given configuration.

    This is the main training entrypoint that can be called directly with a
    TrainConfig for testing or programmatic use, bypassing CLI argument parsing.

    Args:
        cfg: Complete training configuration.

    Returns:
        Best metric value achieved during training, or None if no validation.

    Example::

        from timm.engine import TrainConfig, ModelConfig, TrainDataConfig

        cfg = TrainConfig(
            model=ModelConfig(model='resnet18', pretrained=True),
            data=TrainDataConfig(path='/path/to/data'),
        )
        best_metric = train_cls(cfg)
    """
    # Setup device and distributed training
    device_env = setup_device(cfg.device, seed=cfg.misc.seed)

    if device_env.distributed:
        _logger.info(
            f'Training in distributed mode. '
            f'Process {device_env.rank}/{device_env.world_size}, '
            f'device {device_env.device}'
        )
    else:
        _logger.info(f'Training on single device: {device_env.device}')

    # Create model
    model, data_config = create_train_model(
        cfg.model,
        cfg.device,
        device_env,
    )

    # Get number of classes
    num_classes = cfg.model.num_classes
    if num_classes is None:
        num_classes = get_model_num_classes(model)

    # Get patch size for NaFlex
    model_patch_size = get_naflex_patch_size(model) if cfg.naflex.naflex_loader else None

    # Create optimizer
    optimizer = create_train_optimizer(
        model,
        cfg.optimizer,
        device_env,
        batch_size=cfg.loader.batch_size,
        grad_accum_steps=cfg.model.grad_accum_steps,
    )

    # Resume from checkpoint
    resume_epoch = None
    if cfg.model.resume:
        resume_epoch = resume_training(
            model,
            optimizer,
            cfg.model.resume,
            device_env,
            no_resume_opt=cfg.model.no_resume_opt,
        )

    # Create data loaders
    loader_train, mixup_fn, naflex_mode = create_train_loader(
        cfg,
        data_config,
        device_env,
        num_classes=num_classes,
        model_patch_size=model_patch_size,
    )

    loader_eval = create_eval_loader(
        cfg,
        data_config,
        device_env,
        model_patch_size=model_patch_size,
    )

    # Determine mixup state
    mixup_active = (
        cfg.mixup.mixup > 0 or
        cfg.mixup.cutmix > 0 or
        cfg.mixup.cutmix_minmax is not None
    )
    num_aug_splits = cfg.augment.aug_splits if cfg.augment.aug_splits > 1 else 0

    # Create loss functions
    train_loss_fn = create_loss_fn(cfg, mixup_active, num_aug_splits, device_env.device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device_env.device)

    # Create training task
    task = create_task(model, cfg, train_loss_fn, device_env)

    # Setup task-internal EMA (must be before DDP)
    if cfg.ema.model_ema:
        ema_device = 'cpu' if cfg.ema.model_ema_force_cpu else None
        task.setup_ema(
            decay=cfg.ema.model_ema_decay,
            warmup=cfg.ema.model_ema_warmup,
            device=ema_device,
        )

    # Prepare for distributed training
    if device_env.distributed:
        task.prepare_distributed(device_ids=[device_env.device])

    # Compile task if requested
    if cfg.model.torchcompile:
        task = torch.compile(
            task,
            backend=cfg.model.torchcompile,
            mode=cfg.model.torchcompile_mode,
        )

    # Setup output directory and checkpoint saver
    output_dir = None
    saver = None
    eval_metric = cfg.misc.eval_metric
    decreasing_metric = eval_metric == 'loss'

    if is_primary(device_env):
        exp_name = cfg.misc.experiment or '-'.join([
            datetime.now().strftime('%Y%m%d-%H%M%S'),
            safe_model_name(cfg.model.model),
            str(data_config['input_size'][-1]),
        ])
        output_dir = get_output_dir(cfg.misc.output, exp_name)

        saver = setup_checkpoint_saver(
            task,
            optimizer,
            cfg,
            output_dir,
            device_env,
            decreasing_metric=decreasing_metric,
        )

        # Save config
        save_config(cfg, output_dir)

        # Setup wandb
        setup_wandb(cfg, exp_name, device_env)

    # Create scheduler
    updates_per_epoch = (
        (len(loader_train) + cfg.model.grad_accum_steps - 1) //
        cfg.model.grad_accum_steps
    )
    lr_scheduler, num_epochs = create_train_scheduler(
        optimizer,
        cfg.scheduler,
        updates_per_epoch,
        device_env,
    )

    # Determine start epoch
    start_epoch = cfg.scheduler.start_epoch
    if start_epoch is None:
        start_epoch = resume_epoch if resume_epoch is not None else 0

    if lr_scheduler is not None and start_epoch > 0:
        if cfg.scheduler.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    # Training loop
    best_metric = None
    best_epoch = None

    try:
        for epoch in range(start_epoch, num_epochs):
            # Set epoch for distributed sampler
            if hasattr(loader_train.dataset, 'set_epoch'):
                loader_train.dataset.set_epoch(epoch)
            elif device_env.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            # Train one epoch
            train_metrics = train_one_epoch(
                epoch=epoch,
                task=task,
                loader=loader_train,
                optimizer=optimizer,
                device_env=device_env,
                cfg=cfg,
                lr_scheduler=lr_scheduler,
                mixup_fn=mixup_fn,
                saver=saver,
                output_dir=output_dir,
                num_updates_total=num_epochs * updates_per_epoch,
                naflex_mode=naflex_mode,
            )

            # Distribute batch norm stats
            if device_env.distributed and cfg.device.dist_bn in ('broadcast', 'reduce'):
                utils.distribute_bn(
                    model,
                    device_env.world_size,
                    cfg.device.dist_bn == 'reduce',
                )

            # Validation
            epoch_p_1 = epoch + 1
            if epoch_p_1 % cfg.misc.val_interval != 0 and epoch_p_1 != num_epochs:
                if lr_scheduler is not None:
                    lr_scheduler.step(epoch_p_1, metric=None)
                continue

            eval_metrics = None
            if loader_eval is not None:
                # Use EvalTask-based validation
                eval_task = task.get_eval_task(use_ema=False)
                eval_metrics = validate_with_task(eval_task, loader_eval, device_env, cfg)

                # EMA evaluation (if task has EMA enabled)
                if task.has_ema and not cfg.ema.model_ema_force_cpu:
                    if device_env.distributed and cfg.device.dist_bn in ('broadcast', 'reduce'):
                        utils.distribute_bn(
                            task.trainable_module_ema,
                            device_env.world_size,
                            cfg.device.dist_bn == 'reduce',
                        )
                    # Get eval task with EMA weights
                    ema_eval_task = task.get_eval_task(use_ema=True)
                    ema_eval_metrics = validate_with_task(
                        ema_eval_task, loader_eval, device_env, cfg, log_suffix=' (EMA)'
                    )
                    eval_metrics = ema_eval_metrics

            # Update summary
            if output_dir is not None:
                lrs = [pg['lr'] for pg in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=cfg.misc.log_wandb,
                )

            # Get metric for checkpointing
            latest_metric = (
                eval_metrics[eval_metric] if eval_metrics and eval_metric in eval_metrics
                else train_metrics.get(eval_metric, train_metrics['loss'])
            )

            # Save checkpoint
            if saver is not None:
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)

            # Step scheduler
            if lr_scheduler is not None:
                lr_scheduler.step(epoch_p_1, latest_metric)

    except KeyboardInterrupt:
        _logger.info('Training interrupted by user')

    # Cleanup
    if device_env.distributed:
        torch.distributed.destroy_process_group()

    if best_metric is not None:
        _logger.info(f'*** Best metric: {best_metric} (epoch {best_epoch})')

    return best_metric


def main():
    """CLI entrypoint for classification training.

    Parses command line arguments and calls train_cls().
    """
    utils.setup_default_logging()
    cfg = parse_args()
    train_cls(cfg)


if __name__ == '__main__':
    main()
