from dataclasses import dataclass


@dataclass
class TrainCfg:
    """ Train Loop Configuration
    Dataclass to propagate training configuration values
    """
    num_epochs: int = 0
    log_interval: int = 50
    recovery_interval: int = 0
    accumulate_steps: int = 0
