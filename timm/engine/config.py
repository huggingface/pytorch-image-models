"""Configuration dataclasses for timm training infrastructure.

These are pure dataclasses with no external dependencies (except standard library).
The config library (simple-parsing) is only used at the script boundary.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DeviceConfig:
    """Device and distributed training configuration."""
    device: str = 'cuda'
    amp: bool = False
    amp_dtype: str = 'float16'
    model_dtype: Optional[str] = None
    channels_last: bool = False
    no_ddp_bb: bool = False
    synchronize_step: bool = False
    local_rank: int = 0
    device_modules: Optional[List[str]] = None
    fuser: str = ''
    fast_norm: bool = False
    sync_bn: bool = False
    dist_bn: str = 'reduce'
    split_bn: bool = False


@dataclass
class ModelConfig:
    """Model creation and architecture configuration."""
    model: str = 'resnet50'
    pretrained: bool = False
    pretrained_path: Optional[str] = None
    initial_checkpoint: str = ''
    resume: str = ''
    no_resume_opt: bool = False
    num_classes: Optional[int] = None
    in_chans: Optional[int] = None
    global_pool: Optional[str] = None
    img_size: Optional[int] = None
    input_size: Optional[Tuple[int, int, int]] = None
    crop_pct: Optional[float] = None
    mean: Optional[Tuple[float, ...]] = None
    std: Optional[Tuple[float, ...]] = None
    interpolation: str = ''
    grad_accum_steps: int = 1
    grad_checkpointing: bool = False
    head_init_scale: Optional[float] = None
    head_init_bias: Optional[float] = None
    bn_momentum: Optional[float] = None
    bn_eps: Optional[float] = None
    drop_rate: float = 0.0
    drop_path_rate: Optional[float] = None
    drop_block_rate: Optional[float] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Compilation
    torchscript: bool = False
    torchcompile: Optional[str] = None
    torchcompile_mode: Optional[str] = None


@dataclass
class DataSourceConfig:
    """Specifies a single dataset source.

    Used for individual splits (train, val) and eval sets (ref, probe).
    Fields set to None inherit from parent config defaults.
    """
    type: Optional[str] = None        # None = inherit; folder, hfds, wds, tfds
    path: Optional[str] = None        # None = inherit; meaning varies by type
    split: Optional[str] = None       # Split name, shard spec, or slice
    num_samples: Optional[int] = None # Limit samples (useful for wds/tfds subsets)


@dataclass
class TrainDataConfig:
    """Training data configuration with train + val splits.

    Shared defaults (type, path) are used when train/val don't specify their own.
    """
    # Shared defaults
    type: str = 'folder'              # Dataset format: folder, hfds, wds, tfds
    path: Optional[str] = None        # Data path (meaning varies by type)

    # Per-split configs (inherit type/path from above if not set)
    train: DataSourceConfig = field(default_factory=lambda: DataSourceConfig(split='train'))
    val: DataSourceConfig = field(default_factory=lambda: DataSourceConfig(split='validation'))

    # Common settings (apply to all sources)
    download: bool = False
    class_map: str = ''
    input_img_mode: Optional[str] = None
    input_key: Optional[str] = None
    target_key: Optional[str] = None
    trust_remote_code: bool = False


@dataclass
class ProbeDataConfig:
    """Feature-based evaluation data: reference + probe sets.

    Used for KNN, retrieval, prototype-based evaluation.
    Inherits from TrainDataConfig if type/path not specified.
    """
    # Shared defaults (empty = inherit from TrainDataConfig)
    type: str = ''
    path: Optional[str] = None

    # Per-set configs
    ref: DataSourceConfig = field(default_factory=lambda: DataSourceConfig(split='train'))
    probe: DataSourceConfig = field(default_factory=lambda: DataSourceConfig(split='validation'))


# Backward compatibility alias
DataConfig = TrainDataConfig


@dataclass
class LoaderConfig:
    """Data loader configuration."""
    batch_size: int = 128
    validation_batch_size: Optional[int] = None
    workers: int = 4
    pin_mem: bool = False
    prefetcher: bool = True
    use_multi_epochs_loader: bool = False
    worker_seeding: str = 'all'


@dataclass
class AugmentConfig:
    """Data augmentation configuration."""
    no_aug: bool = False
    train_crop_mode: Optional[str] = None
    scale: Tuple[float, float] = (0.08, 1.0)
    ratio: Tuple[float, float] = (0.75, 1.3333333333333333)
    hflip: float = 0.5
    vflip: float = 0.0
    color_jitter: float = 0.4
    color_jitter_prob: Optional[float] = None
    grayscale_prob: Optional[float] = None
    gaussian_blur_prob: Optional[float] = None
    auto_augment: Optional[str] = None
    aug_repeats: float = 0
    aug_splits: int = 0
    train_interpolation: str = 'random'
    # Random erasing
    reprob: float = 0.0
    remode: str = 'pixel'
    recount: int = 1
    resplit: bool = False


@dataclass
class MixupConfig:
    """Mixup/CutMix configuration."""
    mixup: float = 0.0
    cutmix: float = 0.0
    cutmix_minmax: Optional[Tuple[float, float]] = None
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = 'batch'
    mixup_off_epoch: int = 0
    smoothing: float = 0.1


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    opt: str = 'sgd'
    lr: Optional[float] = None
    lr_base: float = 0.1
    lr_base_size: int = 256
    lr_base_scale: str = ''
    weight_decay: float = 2e-5
    momentum: float = 0.9
    opt_eps: Optional[float] = None
    opt_betas: Optional[Tuple[float, ...]] = None
    clip_grad: Optional[float] = None
    clip_mode: str = 'norm'
    layer_decay: Optional[float] = None
    layer_decay_min_scale: float = 0.0
    layer_decay_no_opt_scale: Optional[float] = None
    opt_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    sched: str = 'cosine'
    sched_on_updates: bool = False
    epochs: int = 300
    start_epoch: Optional[int] = None
    decay_epochs: float = 90
    decay_milestones: Tuple[int, ...] = (90, 180, 270)
    decay_rate: float = 0.1
    warmup_epochs: int = 5
    warmup_lr: float = 1e-5
    warmup_prefix: bool = False
    cooldown_epochs: int = 0
    patience_epochs: int = 10
    min_lr: float = 0.0
    lr_noise: Optional[Tuple[float, ...]] = None
    lr_noise_pct: float = 0.67
    lr_noise_std: float = 1.0
    lr_cycle_mul: float = 1.0
    lr_cycle_decay: float = 0.5
    lr_cycle_limit: int = 1
    lr_k_decay: float = 1.0
    epoch_repeats: float = 0.0


@dataclass
class EMAConfig:
    """Model EMA (Exponential Moving Average) configuration."""
    model_ema: bool = False
    model_ema_decay: float = 0.9998
    model_ema_force_cpu: bool = False
    model_ema_warmup: bool = False


@dataclass
class LossConfig:
    """Loss function configuration."""
    jsd_loss: bool = False
    bce_loss: bool = False
    bce_sum: bool = False
    bce_target_thresh: Optional[float] = None
    bce_pos_weight: Optional[float] = None


@dataclass
class DistillationConfig:
    """Knowledge distillation configuration."""
    kd_model_name: Optional[str] = None
    kd_distill_type: str = 'logit'
    kd_loss_type: str = 'kl'
    distill_loss_weight: Optional[float] = None
    task_loss_weight: Optional[float] = None
    kd_temperature: float = 4.0
    kd_student_feature_dim: Optional[int] = None
    kd_teacher_feature_dim: Optional[int] = None
    kd_token_distill_type: str = 'soft'


@dataclass
class SSLConfig:
    """Self-supervised learning configuration."""
    ssl_method: Optional[str] = None  # 'nepa', 'lejepa'
    num_views: int = 2  # Number of augmented views for multi-view SSL methods (LeJEPA, DINO, etc.)
    # NEPA specific
    nepa_no_shift: bool = False
    nepa_pixel_decoder: bool = False
    nepa_pixel_hidden_dim: Optional[int] = None
    nepa_pixel_depth: int = 2
    nepa_pixel_mlp_layer: str = 'mlp'
    nepa_pixel_drop_path: float = 0.0
    nepa_pixel_init_values: Optional[float] = None
    nepa_pixel_loss_weight: float = 1.0
    nepa_pixel_loss_type: str = 'mse'
    nepa_no_norm_pixel_target: bool = False
    nepa_prefix_bidirectional: bool = False  # If True, prefix tokens (CLS, reg) attend bidirectionally (slower custom mask)
    # LeJEPA specific
    lejepa_proj_dim: int = 128
    lejepa_proj_hidden: int = 2048
    lejepa_proj_layers: int = 2
    lejepa_lamb: float = 0.02
    lejepa_num_slices: int = 256
    lejepa_num_knots: int = 17
    # SSL evaluation (feature-based: knn, retrieval, prototype, etc.)
    ssl_eval_metric: str = 'loss'  # 'loss', 'knn', 'retrieval', 'prototype'
    ssl_eval_interval: int = 5  # Run feature-based eval every N epochs
    # Evaluation data (ref/probe sets for knn, retrieval, etc.)
    eval_data: ProbeDataConfig = field(default_factory=ProbeDataConfig)
    # KNN-specific params
    knn_k: int = 20  # Number of neighbors
    knn_temperature: float = 0.07  # Temperature for weighted KNN


@dataclass
class NaFlexConfig:
    """NaFlex variable-resolution loader configuration."""
    naflex_loader: bool = False
    naflex_train_seq_lens: Tuple[int, ...] = (128, 256, 576, 784, 1024)
    naflex_max_seq_len: int = 576
    naflex_patch_sizes: Optional[Tuple[int, ...]] = None
    naflex_patch_size_probs: Optional[Tuple[float, ...]] = None
    naflex_loss_scale: str = 'linear'


@dataclass
class MiscConfig:
    """Miscellaneous training configuration."""
    seed: int = 42
    log_interval: int = 50
    val_interval: int = 1
    recovery_interval: int = 0
    checkpoint_hist: int = 10
    output: str = ''
    experiment: str = ''
    eval_metric: str = 'top1'
    tta: int = 0
    save_images: bool = False
    # WandB
    log_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_resume_id: str = ''


@dataclass
class TrainConfig:
    """Complete training configuration aggregating all sub-configs.

    This is the top-level config used by training scripts. Each sub-config
    groups related parameters for clarity and modularity.

    Example usage with simple-parsing::

        from simple_parsing import ArgumentParser
        from timm.engine import TrainConfig

        parser = ArgumentParser()
        parser.add_arguments(TrainConfig, dest='cfg')
        args = parser.parse_args()
        cfg = args.cfg

    Example usage with YAML::

        device:
          amp: true
          channels_last: true
        model:
          model: resnet50
          pretrained: true
        data:
          data_dir: /path/to/imagenet
        optimizer:
          opt: adamw
          lr: 0.001
        scheduler:
          epochs: 100
    """
    device: DeviceConfig = field(default_factory=DeviceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    mixup: MixupConfig = field(default_factory=MixupConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    naflex: NaFlexConfig = field(default_factory=NaFlexConfig)
    misc: MiscConfig = field(default_factory=MiscConfig)
