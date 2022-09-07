from dataclasses import dataclass

from .monitor import Monitor
from .checkpoint_manager import CheckpointManager


@dataclass
class TrainServices:
    """ Train Loop Services
    """
    monitor: Monitor = None
    checkpoint: CheckpointManager = None

