import abc
from typing import Callable, Union, Optional, List, Tuple, Dict
from dataclasses import dataclass

import torch
from torch.distributed import ReduceOp

from .device_env import DeviceEnv
from .distributed import all_gather_sequence, all_reduce_sequence

MetricValueT = Union[float, torch.Tensor, List[float], List[torch.Tensor]]


@dataclass
class ValueInfo:
    initial: Optional[MetricValueT] = 0.
    dtype: torch.dtype = torch.float32
    dist_reduce: str = 'sum'
    dist_average: bool = False


class Metric(abc.ABC):

    def __init__(
            self,
            dev_env: DeviceEnv = None
    ):
        self._infos: Dict[str, ValueInfo] = {}
        self._values: Dict[str, Optional[MetricValueT]] = {}
        self._values_dist: Dict[str, Optional[MetricValueT]] = {}
        if dev_env is None:
            dev_env = DeviceEnv.instance()
        self._dev_env = dev_env

    def _register_value(self, name: str, info: Optional[ValueInfo] = None):
        info = info or ValueInfo()
        self._infos[name] = info

    # def get_value(self, name: str, use_dist=True):
    #     if use_dist:
    #         return self._values_dist.get(name, self._values.get(name))
    #     else:
    #         return self._values.get(name)

    def __getattr__(self, item):
        if item not in self._infos:
            raise AttributeError
        value = self._values_dist.get(item, self._values.get(item, None))
        return value

    def __setattr__(self, key, value):
        if '_infos' in self.__dict__ and key in self._infos:
            self._values[key] = value
        else:
            super().__setattr__(key, value)

    def update(
            self,
            predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
            target: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        self._update(predictions, target)

    def _update(
            self,
            predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
            target: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        pass

    def reset(self):
        self._values = {}
        self._values_dist = {}
        for name, info in self._infos.items():
            # if info specifies an initial value, we reset here, otherwise set to None and leave it to child class
            if info.initial is not None:
                if isinstance(info.initial, torch.Tensor):
                    tensor = info.initial.detach().clone()
                else:
                    tensor = torch.ones([], dtype=info.dtype) * info.initial  # scalar
                self._values[name] = tensor.to(device=self._dev_env.device, dtype=info.dtype)
            else:
                self._values[name] = None
        self._reset()

    def _reset(self):
        pass

    def compute(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        if self._dev_env.distributed:
            self._distribute_values()
        results = self._compute()
        self._values_dist = {}
        return results

    @abc.abstractmethod
    def _compute(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        pass

    def _distribute_values(self):
        if not self._infos or not self._values:
            return

        def _args(op: str):
            if op == 'cat':
                return True, dict(cat_dim=0)
            else:
                return False, dict(op=ReduceOp.SUM)

        prev_dsr = None
        same_dsr = True
        names = []
        values = []
        reductions = []
        for name, value in self._values.items():
            if value is not None:
                info = self._infos[name]
                dsr = (value.dtype, value.shape, info.dist_reduce)
                if prev_dsr is not None and prev_dsr != dsr:
                    same_dsr = False
                prev_dsr = dsr
                names.append(name)
                values.append(value)
                reductions.append(_args(info.dist_reduce))

        if same_dsr:
            do_gather, reduce_kwargs = reductions[0]
            if do_gather:
                reduced_values = all_gather_sequence(values, dev_env=self._dev_env, **reduce_kwargs)
            else:
                reduced_values = all_reduce_sequence(values, dev_env=self._dev_env, **reduce_kwargs)
            for name, reduced_value in zip(names, reduced_values):
                info = self._infos[name]
                if info.dist_average:
                    reduced_value /= self._dev_env.world_size
                self._values_dist[name] = reduced_value
        else:
            for n, v, r in zip(names, values, reductions):
                info = self._infos[n]
                do_gather, reduce_kwargs = r
                if do_gather:
                    reduced_value = self._dev_env.all_gather(v, **reduce_kwargs)
                else:
                    reduced_value = self._dev_env.all_reduce(v, **reduce_kwargs)
                if info.dist_average:
                    reduced_value /= self._dev_env.world_size
                self._values_dist[n] = reduced_value
