import torch
from typing import Optional, Tuple, Dict

from .device_env import DeviceEnv
from .metric import Metric, ValueInfo


class Accuracy(Metric):

    def __init__(
            self,
            threshold=0.5,
            multi_label=False,
            accumulate_dtype=torch.float32,
            dev_env=None,
    ):
        super().__init__(dev_env=dev_env)
        self.accumulate_dtype = accumulate_dtype
        self.threshold = threshold
        self.eps = 1e-8
        self.multi_label = multi_label

        # statistics / counts
        self._register_value('correct', ValueInfo(dtype=accumulate_dtype))
        self._register_value('total', ValueInfo(dtype=accumulate_dtype))

    def _update(self, predictions, target):
        raise NotImplemented()

    def _compute(self):
        raise NotImplemented()


class AccuracyTopK(Metric):

    def __init__(
            self,
            topk=(1, 5),
            accumulate_dtype=torch.float32,
            dev_env: DeviceEnv = None
    ):
        super().__init__(dev_env=dev_env)
        self.accumulate_dtype = accumulate_dtype
        self.eps = 1e-8
        self.topk = topk
        self.maxk = max(topk)

        # statistics / counts
        for k in self.topk:
            self._register_value(f'top{k}', ValueInfo(dtype=accumulate_dtype))
        self._register_value('total', ValueInfo(dtype=accumulate_dtype))
        self.reset()

    def _update(self, predictions: torch.Tensor, target: torch.Tensor):
        batch_size = predictions.shape[0]
        sorted_indices = predictions.topk(self.maxk, dim=1)[1]
        target_reshape = target.reshape(-1, 1).expand_as(sorted_indices)
        correct = sorted_indices.eq(target_reshape).to(dtype=self.accumulate_dtype).sum(0)
        for k in self.topk:
            attr_name = f'top{k}'
            correct_at_k = correct[:k].sum()
            setattr(self, attr_name, getattr(self, attr_name) + correct_at_k)
        self.total += batch_size

    def _compute(self) -> Dict[str, torch.Tensor]:
        assert self.total is not None
        output = {}
        for k in self.topk:
            attr_name = f'top{k}'
            output[attr_name] = 100 * getattr(self, attr_name) / self.total
        return output
