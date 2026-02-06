#!/usr/bin/env python3
"""Self-supervised learning training script using jsonargparse for config management.

This is an experimental version using jsonargparse instead of simple-parsing.
Compare with train_ssl.py to evaluate the differences.

Key differences from simple-parsing:
- Native YAML/JSON config file support with proper merging
- --print_config to dump full config (useful for creating config files)
- --config can be used multiple times for layered configs
- Better handling of Optional types and defaults
- Subcommand support (not used here but available)

CLI usage::

    # Basic usage (same as simple-parsing version)
    python -m timm.apps.train_ssl_jsonargparse --model.model vit_tiny_patch16_224 \\
        --ssl.ssl_method nepa \\
        --data.path /path/to/imagenet

    # Print full config (great for creating config files)
    python -m timm.apps.train_ssl_jsonargparse --model.model vit_tiny_patch16_224 \\
        --ssl.ssl_method nepa --print_config > my_config.yaml

    # Use config file
    python -m timm.apps.train_ssl_jsonargparse --config my_config.yaml

    # Override config file values
    python -m timm.apps.train_ssl_jsonargparse --config my_config.yaml \\
        --scheduler.epochs 200

    # Layer multiple configs (later ones override earlier)
    python -m timm.apps.train_ssl_jsonargparse \\
        --config base.yaml \\
        --config nepa_specific.yaml \\
        --data.path /my/data
"""
import logging
import os
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn

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
    """Parse command line arguments using jsonargparse.

    jsonargparse key features:
    - Automatic --config support for YAML/JSON files
    - --print_config dumps the full resolved config
    - Multiple --config flags layer configs (later overrides earlier)
    - Clean handling of nested dataclasses
    - Type coercion and validation
    """
    try:
        from jsonargparse import ArgumentParser
    except ImportError:
        raise ImportError(
            "jsonargparse is required for this script. "
            "Install with: pip install jsonargparse[signatures]"
        )

    parser = ArgumentParser(
        description='Self-Supervised Learning Training (jsonargparse version)',
        default_config_files=['~/.config/timm/train_ssl.yaml'],  # Optional default config locations
        print_config='--dump_config',  # Use dump_config to avoid conflicts
    )

    # Add config file support
    parser.add_argument(
        '--config',
        action='config',
        help='Path to YAML/JSON config file(s). Can be specified multiple times.',
    )

    # Add the dataclass as arguments - jsonargparse automatically handles nested dataclasses
    # Using nested_key=None makes all fields top-level (like simple-parsing)
    parser.add_class_arguments(TrainConfig, nested_key=None, fail_untyped=False)

    # Parse arguments
    cfg = parser.parse_args()

    # Instantiate the dataclass from the parsed namespace
    return parser.instantiate_classes(cfg)


def create_ssl_task(
    model: nn.Module,
    cfg: TrainConfig,
    device_env: DeviceEnv,
) -> nn.Module:
    """Create SSL training task based on configuration."""
    ssl_method = cfg.ssl.ssl_method

    if ssl_method is None:
        raise ValueError(
            "SSL method must be specified with --ssl.ssl_method. "
            "Available methods: nepa, lejepa"
        )

    if ssl_method == 'nepa':
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

    This is identical to the simple-parsing version - only parse_args() differs.
    """
    if cfg.ssl.ssl_method is None:
        raise ValueError(
            "SSL method must be specified in cfg.ssl.ssl_method. "
            "Available methods: nepa, lejepa"
        )

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

    model, data_config = create_train_model(
        cfg.model,
        cfg.device,
        device_env,
    )

    num_classes = cfg.model.num_classes or 1000

    model_patch_size = get_naflex_patch_size(model) if cfg.naflex.naflex_loader else None

    optimizer = create_train_optimizer(
        model,
        cfg.optimizer,
        device_env,
        batch_size=cfg.loader.batch_size,
        grad_accum_steps=cfg.model.grad_accum_steps,
    )

    resume_epoch = None
    if cfg.model.resume:
        resume_epoch = resume_training(
            model,
            optimizer,
            cfg.model.resume,
            device_env,
            no_resume_opt=cfg.model.no_resume_opt,
        )

    ssl_cfg = cfg
    ssl_method = cfg.ssl.ssl_method
    mixup_fn = None
    naflex_mode = False

    if ssl_method == 'lejepa':
        _logger.info(f'Creating multi-view loader for {ssl_method} with {cfg.ssl.num_views} views')
        cfg.loader.prefetcher = False
        loader_train = create_multiview_train_loader(
            ssl_cfg,
            data_config,
            device_env,
            num_views=cfg.ssl.num_views,
        )
    else:
        _logger.info(f'Creating standard loader for {ssl_method}')
        loader_train, mixup_fn, naflex_mode = create_train_loader(
            ssl_cfg,
            data_config,
            device_env,
            num_classes=num_classes,
            model_patch_size=model_patch_size,
        )

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

    task = create_ssl_task(model, cfg, device_env)

    # Setup task-internal EMA (must be before DDP)
    if cfg.ema.model_ema:
        ema_device = 'cpu' if cfg.ema.model_ema_force_cpu else None
        task.setup_ema(
            decay=cfg.ema.model_ema_decay,
            warmup=cfg.ema.model_ema_warmup,
            device=ema_device,
        )

    if device_env.distributed:
        task.prepare_distributed(device_ids=[device_env.device])

    if cfg.model.torchcompile:
        task = torch.compile(
            task,
            backend=cfg.model.torchcompile,
            mode=cfg.model.torchcompile_mode,
        )

    output_dir = None
    saver = None
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

        save_config(cfg, output_dir)
        setup_wandb(cfg, exp_name, device_env)

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

    start_epoch = cfg.scheduler.start_epoch
    if start_epoch is None:
        start_epoch = resume_epoch if resume_epoch is not None else 0

    if lr_scheduler is not None and start_epoch > 0:
        if cfg.scheduler.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    best_metric = None
    best_epoch = None

    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(loader_train.dataset, 'set_epoch'):
                loader_train.dataset.set_epoch(epoch)
            elif device_env.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch=epoch,
                task=task,
                loader=loader_train,
                optimizer=optimizer,
                device_env=device_env,
                cfg=cfg,
                lr_scheduler=lr_scheduler,
                mixup_fn=None,
                saver=saver,
                output_dir=output_dir,
                num_updates_total=num_epochs * updates_per_epoch,
                naflex_mode=naflex_mode,
            )

            if device_env.distributed and cfg.device.dist_bn in ('broadcast', 'reduce'):
                utils.distribute_bn(
                    task.trainable_module,
                    device_env.world_size,
                    cfg.device.dist_bn == 'reduce',
                )

            epoch_p_1 = epoch + 1
            if epoch_p_1 % cfg.misc.val_interval != 0 and epoch_p_1 != num_epochs:
                if lr_scheduler is not None:
                    lr_scheduler.step(epoch_p_1, metric=None)
                continue

            eval_metrics = None
            latest_metric = train_metrics['loss']

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

                if hasattr(eval_task, 'knn_k'):
                    eval_task.knn_k = cfg.ssl.knn_k
                    eval_task.knn_temperature = cfg.ssl.knn_temperature

                if cfg.ssl.ssl_eval_metric == 'knn':
                    eval_metrics = validate_knn(
                        eval_task, loader_eval_ref, loader_eval_probe, device_env, cfg
                    )
                    latest_metric = eval_metrics.get('knn_acc', 0.0)
                    if saver is not None:
                        saver.decreasing = False

                    if is_primary(device_env):
                        _logger.info(f'KNN accuracy: {latest_metric:.2f}%')
                else:
                    _logger.warning(f'Eval metric {cfg.ssl.ssl_eval_metric} not yet implemented')
            else:
                if saver is not None:
                    saver.decreasing = True

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

            if saver is not None:
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)

            if lr_scheduler is not None:
                lr_scheduler.step(epoch_p_1, latest_metric)

    except KeyboardInterrupt:
        _logger.info('Training interrupted by user')

    if device_env.distributed:
        torch.distributed.destroy_process_group()

    if best_metric is not None:
        _logger.info(f'*** Best loss: {best_metric} (epoch {best_epoch})')

    if is_primary(device_env):
        _logger.info(
            '\nSSL training complete. To evaluate representations, use:\n'
            '  - k-NN evaluation: python validate_knn.py --checkpoint <path>\n'
            '  - Linear probe: (coming soon)\n'
        )

    return best_metric


def main():
    """CLI entrypoint for SSL training using jsonargparse."""
    utils.setup_default_logging()
    cfg = parse_args()

    if cfg.ssl.ssl_method is None:
        _logger.error(
            "SSL method must be specified. Use --ssl.ssl_method nepa or --ssl.ssl_method lejepa"
        )
        return

    train_ssl(cfg)


if __name__ == '__main__':
    main()
