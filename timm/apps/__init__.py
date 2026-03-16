"""Training and utility applications for timm.

This module contains runnable applications for training, evaluation, and utilities.
Each app can be run as a module or via installed console scripts.

Available apps:
    - train_cls: Classification training (includes distillation)
    - train_ssl: Self-supervised learning training (NEPA, LeJEPA)
    - sweep: Hyperparameter sweep runner

Example usage::

    # As a module
    python -m timm.apps.train_cls --model.model resnet50 --data.data_dir /path/to/data
    python -m timm.apps.train_ssl --model.model vit_tiny_patch16_224 --ssl.ssl_method nepa ...
    python -m timm.apps.sweep sweeps/config.yaml

    # As installed console scripts (after pip install)
    timm-train-cls --model.model resnet50 --data.data_dir /path/to/data
    timm-train-ssl --model.model vit_tiny_patch16_224 --ssl.ssl_method nepa ...
    timm-sweep sweeps/config.yaml

    # Programmatic usage
    from timm.apps.train_cls import train_cls
    from timm.apps.train_ssl import train_ssl
    from timm.apps.sweep import run_sweep
    from timm.engine import TrainConfig, ModelConfig, DataConfig

    cfg = TrainConfig(
        model=ModelConfig(model='resnet50'),
        data=DataConfig(data_dir='/path/to/data'),
    )
    train_cls(cfg)
"""
