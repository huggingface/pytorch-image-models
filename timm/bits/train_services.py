from dataclasses import dataclass

from .logger import Logger
from timm.utils.checkpoint_saver import CheckpointSaver


@dataclass
class TrainServices:
    """ Train Loop Services
    """
    logger: Logger = None
    saver: CheckpointSaver = None

