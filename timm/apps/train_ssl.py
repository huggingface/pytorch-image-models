#!/usr/bin/env python3
"""Self-supervised learning training script using timm.engine infrastructure.

A modern, modular training script for self-supervised learning methods including:
- NEPA (Next Embedding Prediction Architecture)
- LeJEPA (Lean Joint-Embedding Predictive Architecture)
- Future: JEPA, AIM, MAE-style methods

CLI usage::

    # As a module
    python -m timm.apps.train_ssl --model.model vit_tiny_patch16_224 \\
        --ssl.ssl_method nepa \\
        --data.path /path/to/imagenet \\
        --scheduler.epochs 100

    # As installed script
    timm-train-ssl --model.model vit_tiny_patch16_224 --ssl.ssl_method nepa \\
        --data.path /path/to/imagenet

    # NEPA with pixel decoder
    python -m timm.apps.train_ssl --model.model vit_tiny_patch16_224 \\
        --ssl.ssl_method nepa \\
        --ssl.nepa_pixel_decoder true \\
        --ssl.nepa_pixel_mlp_layer swiglu \\
        --data.path /path/to/imagenet

    # LeJEPA training
    python -m timm.apps.train_ssl --model.model vit_small_patch16_224 \\
        --ssl.ssl_method lejepa \\
        --ssl.lejepa_lamb 0.02 \\
        --data.path /path/to/imagenet \\
        --scheduler.epochs 100

    # With config file
    python -m timm.apps.train_ssl -c configs/nepa_vit.yaml --data.path /path/to/data

Programmatic usage (for testing)::

    from timm.apps.train_ssl import train_ssl
    from timm.engine import TrainConfig, ModelConfig, TrainDataConfig, SSLConfig, SchedulerConfig

    cfg = TrainConfig(
        model=ModelConfig(model='vit_tiny_patch16_224'),
        data=TrainDataConfig(path='/path/to/data'),
        ssl=SSLConfig(ssl_method='nepa'),
        scheduler=SchedulerConfig(epochs=5),
    )
    best_loss = train_ssl(cfg)
"""
import logging
import os
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import yaml

from timm import utils
from timm.models import safe_model_name
from timm.task import NEPATask, LeJEPATask

from timm.engine import (
    TrainConfig,
    DeviceEnv,
    setup_device,
    is_primary,
    create_train_model,
    create_train_loader,
    create_multiview_train_loader,
    create_eval_loader,
    create_train_optimizer,
    create_train_scheduler,
    train_one_epoch,
    validate_knn,
    setup_checkpoint_saver,
    resume_training,
    save_config,
    get_output_dir,
    get_model_num_classes,
    get_naflex_patch_size,
)

_logger = logging.getLogger('train_ssl')


def parse_args() -> TrainConfig:
    """Parse command line arguments into TrainConfig dataclass."""
    try:
        from simple_parsing import ArgumentParser
    except ImportError:
        raise ImportError(
            "simple-parsing is required for train_ssl. "
            "Install with: pip install simple-parsing"
        )

    parser = ArgumentParser(
        description='Self-Supervised Learning Training with timm.engine',
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

    # Load YAML config if specified
    if args.config:
        with open(args.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        _logger.info(f'Loaded config from: {args.config}')

    return args.cfg


def create_ssl_task(
    model: nn.Module,
    cfg: TrainConfig,
    device_env: DeviceEnv,
) -> nn.Module:
    """Create SSL training task based on configuration.

    Args:
        model: Base model for SSL training.
        cfg: Training configuration.
        device_env: Device environment.

    Returns:
        SSL training task instance.

    Raises:
        ValueError: If ssl_method is not specified or unknown.
    """
    ssl_method = cfg.ssl.ssl_method

    if ssl_method is None:
        raise ValueError(
            "SSL method must be specified with --ssl.ssl_method. "
            "Available methods: nepa, lejepa"
        )

    if ssl_method == 'nepa':
        # Build pixel decoder config if enabled
        pixel_decoder_cfg = None
        if cfg.ssl.nepa_pixel_decoder:
            pixel_decoder_cfg = dict(
                hidden_dim=cfg.ssl.nepa_pixel_hidden_dim,
                depth=cfg.ssl.nepa_pixel_depth,
                mlp_layer=cfg.ssl.nepa_pixel_mlp_layer,
                drop_path_rate=cfg.ssl.nepa_pixel_drop_path,
                init_values=cfg.ssl.nepa_pixel_init_values,
            )

        task = NEPATask(
            model=model,
            shift=not cfg.ssl.nepa_no_shift,
            pixel_decoder_cfg=pixel_decoder_cfg,
            pixel_loss_weight=cfg.ssl.nepa_pixel_loss_weight,
            pixel_loss_type=cfg.ssl.nepa_pixel_loss_type,
            norm_pixel_target=not cfg.ssl.nepa_no_norm_pixel_target,
            prefix_bidirectional=cfg.ssl.nepa_prefix_bidirectional,
            device=device_env.device,
            dtype=device_env.model_dtype,
            verbose=is_primary(device_env),
        )

        if is_primary(device_env):
            _logger.info(
                f'Created NEPATask: shift={not cfg.ssl.nepa_no_shift}, '
                f'prefix_bidirectional={cfg.ssl.nepa_prefix_bidirectional}, '
                f'pixel_decoder={cfg.ssl.nepa_pixel_decoder}'
            )

    elif ssl_method == 'lejepa':
        task = LeJEPATask(
            model=model,
            proj_dim=cfg.ssl.lejepa_proj_dim,
            proj_hidden=cfg.ssl.lejepa_proj_hidden,
            proj_layers=cfg.ssl.lejepa_proj_layers,
            lamb=cfg.ssl.lejepa_lamb,
            num_slices=cfg.ssl.lejepa_num_slices,
            num_knots=cfg.ssl.lejepa_num_knots,
            device=device_env.device,
            dtype=device_env.model_dtype,
            verbose=is_primary(device_env),
        )

        if is_primary(device_env):
            _logger.info(
                f'Created LeJEPATask: proj_dim={cfg.ssl.lejepa_proj_dim}, '
                f'lambda={cfg.ssl.lejepa_lamb}'
            )

    # Future SSL methods can be added here:
    # elif ssl_method == 'jepa':
    #     task = JEPATask(...)
    # elif ssl_method == 'aim':
    #     task = AIMTask(...)
    else:
        raise ValueError(
            f"Unknown SSL method: {ssl_method}. "
            "Available methods: nepa, lejepa"
        )

    return task


def setup_wandb(cfg: TrainConfig, exp_name: str, device_env: DeviceEnv) -> None:
    """Initialize Weights & Biases logging if configured."""
    if not cfg.misc.log_wandb or not is_primary(device_env):
        return

    try:
        import wandb
        import dataclasses

        config_dict = dataclasses.asdict(cfg)

        wandb.init(
            project=cfg.misc.wandb_project or 'timm-ssl',
            name=exp_name,
            config=config_dict,
            tags=cfg.misc.wandb_tags + [cfg.ssl.ssl_method or 'ssl'],
            resume='must' if cfg.misc.wandb_resume_id else None,
            id=cfg.misc.wandb_resume_id if cfg.misc.wandb_resume_id else None,
        )
    except ImportError:
        _logger.warning('wandb not installed, skipping')


def train_ssl(cfg: TrainConfig) -> Optional[float]:
    """Run self-supervised learning training with the given configuration.

    This is the main training entrypoint that can be called directly with a
    TrainConfig for testing or programmatic use, bypassing CLI argument parsing.

    Args:
        cfg: Complete training configuration. Must have cfg.ssl.ssl_method set.

    Returns:
        Best loss value achieved during training, or None if no validation.

    Raises:
        ValueError: If cfg.ssl.ssl_method is not specified.

    Example::

        from timm.engine import TrainConfig, ModelConfig, TrainDataConfig, SSLConfig

        cfg = TrainConfig(
            model=ModelConfig(model='vit_small_patch16_224'),
            data=TrainDataConfig(path='/path/to/data'),
            ssl=SSLConfig(ssl_method='nepa'),
        )
        best_loss = train_ssl(cfg)
    """
    # Validate SSL method is specified
    if cfg.ssl.ssl_method is None:
        raise ValueError(
            "SSL method must be specified in cfg.ssl.ssl_method. "
            "Available methods: nepa, lejepa"
        )

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

    _logger.info(f'SSL method: {cfg.ssl.ssl_method}')

    # Create model
    model, data_config = create_train_model(
        cfg.model,
        cfg.device,
        device_env,
    )

    # For SSL, we don't need num_classes from model but we need it for loader
    # (targets are still loaded but ignored by SSL tasks)
    num_classes = cfg.model.num_classes or 1000  # Default for ImageNet

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
    # Note: SSL doesn't use mixup/cutmix, set to disabled
    ssl_cfg = cfg
    # We could modify cfg here to disable mixup for SSL, but the task ignores targets anyway

    ssl_method = cfg.ssl.ssl_method
    mixup_fn = None
    naflex_mode = False

    if ssl_method == 'lejepa':
        # Multi-view methods need special loader that returns [B, V, C, H, W]
        # Disable prefetcher for multi-view (not compatible)
        _logger.info(f'Creating multi-view loader for {ssl_method} with {cfg.ssl.num_views} views')
        cfg.loader.prefetcher = False
        loader_train = create_multiview_train_loader(
            ssl_cfg,
            data_config,
            device_env,
            num_views=cfg.ssl.num_views,
        )
    else:
        # Single-view methods (NEPA, etc.) use standard loader
        _logger.info(f'Creating standard loader for {ssl_method}')
        loader_train, mixup_fn, naflex_mode = create_train_loader(
            ssl_cfg,
            data_config,
            device_env,
            num_classes=num_classes,
            model_patch_size=model_patch_size,
        )

    # Create evaluation loaders for feature-based metrics (knn, retrieval, etc.)
    # Both use eval transforms (no augmentation) - single view only
    # ref = reference set (typically train), probe = query set (typically val)
    loader_eval_ref = None
    loader_eval_probe = None
    if cfg.ssl.ssl_eval_metric != 'loss':
        ref_split = cfg.ssl.eval_data.ref.split or 'train'
        probe_split = cfg.ssl.eval_data.probe.split or 'validation'
        _logger.info(
            f'Creating {cfg.ssl.ssl_eval_metric} evaluation loaders '
            f'(ref: {ref_split}, probe: {probe_split})'
        )
        loader_eval_ref = create_eval_loader(
            ssl_cfg,
            data_config,
            device_env,
            model_patch_size=model_patch_size,
            source=cfg.ssl.eval_data.ref,
        )
        loader_eval_probe = create_eval_loader(
            ssl_cfg,
            data_config,
            device_env,
            model_patch_size=model_patch_size,
            source=cfg.ssl.eval_data.probe,
        )

    # Create SSL training task
    task = create_ssl_task(model, cfg, device_env)

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
    # For SSL, we track loss (lower is better for most SSL objectives)
    eval_metric = 'loss'
    decreasing_metric = True

    if is_primary(device_env):
        exp_name = cfg.misc.experiment or '-'.join([
            datetime.now().strftime('%Y%m%d-%H%M%S'),
            cfg.ssl.ssl_method or 'ssl',
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
            # Note: SSL tasks ignore targets
            train_metrics = train_one_epoch(
                epoch=epoch,
                task=task,
                loader=loader_train,
                optimizer=optimizer,
                device_env=device_env,
                cfg=cfg,
                lr_scheduler=lr_scheduler,
                mixup_fn=None,  # SSL doesn't use mixup
                saver=saver,
                output_dir=output_dir,
                num_updates_total=num_epochs * updates_per_epoch,
                naflex_mode=naflex_mode,
            )

            # Distribute batch norm stats
            if device_env.distributed and cfg.device.dist_bn in ('broadcast', 'reduce'):
                utils.distribute_bn(
                    task.trainable_module,
                    device_env.world_size,
                    cfg.device.dist_bn == 'reduce',
                )

            # Validation / Checkpointing
            epoch_p_1 = epoch + 1
            if epoch_p_1 % cfg.misc.val_interval != 0 and epoch_p_1 != num_epochs:
                if lr_scheduler is not None:
                    lr_scheduler.step(epoch_p_1, metric=None)
                continue

            # For SSL, use training loss as the epoch metric (eval loss isn't useful)
            # Only run feature-based eval (knn, retrieval, etc.) if configured
            eval_metrics = None
            latest_metric = train_metrics['loss']

            # Check if this is a feature-based eval epoch
            run_feature_eval = (
                cfg.ssl.ssl_eval_metric != 'loss' and
                loader_eval_ref is not None and
                loader_eval_probe is not None and
                epoch_p_1 % cfg.ssl.ssl_eval_interval == 0
            )

            if run_feature_eval:
                # Distribute EMA batch norm stats before evaluation
                if task.has_ema and not cfg.ema.model_ema_force_cpu:
                    if device_env.distributed and cfg.device.dist_bn in ('broadcast', 'reduce'):
                        utils.distribute_bn(
                            task.trainable_module_ema,
                            device_env.world_size,
                            cfg.device.dist_bn == 'reduce',
                        )

                eval_task = task.get_eval_task(use_ema=True)

                # Apply KNN config params if the eval_task supports them
                if hasattr(eval_task, 'knn_k'):
                    eval_task.knn_k = cfg.ssl.knn_k
                    eval_task.knn_temperature = cfg.ssl.knn_temperature

                if cfg.ssl.ssl_eval_metric == 'knn':
                    # KNN evaluation
                    eval_metrics = validate_knn(
                        eval_task, loader_eval_ref, loader_eval_probe, device_env, cfg
                    )
                    latest_metric = eval_metrics.get('knn_acc', 0.0)
                    if saver is not None:
                        saver.decreasing = False  # Higher KNN accuracy is better

                    if is_primary(device_env):
                        _logger.info(f'KNN accuracy: {latest_metric:.2f}%')
                else:
                    # Future: retrieval, prototype, etc.
                    _logger.warning(f'Eval metric {cfg.ssl.ssl_eval_metric} not yet implemented')
            else:
                # Use training loss for checkpointing
                if saver is not None:
                    saver.decreasing = True  # Lower loss is better

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
        _logger.info(f'*** Best loss: {best_metric} (epoch {best_epoch})')

    # Reminder about evaluation
    if is_primary(device_env):
        _logger.info(
            '\nSSL training complete. To evaluate representations, use:\n'
            '  - k-NN evaluation: python validate_knn.py --checkpoint <path>\n'
            '  - Linear probe: (coming soon)\n'
        )

    return best_metric


def main():
    """CLI entrypoint for SSL training.

    Parses command line arguments and calls train_ssl().
    """
    utils.setup_default_logging()
    cfg = parse_args()

    # Handle the error case gracefully for CLI
    if cfg.ssl.ssl_method is None:
        _logger.error(
            "SSL method must be specified. Use --ssl.ssl_method nepa or --ssl.ssl_method lejepa"
        )
        return

    train_ssl(cfg)


if __name__ == '__main__':
    main()
