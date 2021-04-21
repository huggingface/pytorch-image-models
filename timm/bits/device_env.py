import torch
import abc


class DeviceEnv(abc.ABC):

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abc.abstractmethod
    def local_rank(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def global_rank(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def is_distributed(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def world_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def is_master(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def type(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def autocast(self):
        pass

    @abc.abstractmethod
    def wrap_distributed(self, *modules):
        pass

    @abc.abstractmethod
    def to_device(self, *modules: torch.nn.Module):
        pass

    #@abc.abstractmethod
    def mark_step(self):
        # FIXME this is for XLA only, make it common to all devices w/ appropriate no-ops?
        pass