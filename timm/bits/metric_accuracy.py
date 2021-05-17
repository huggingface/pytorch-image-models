import torch
from typing import Optional, Tuple, Dict

from .device_env import DeviceEnv
from .metric import Metric, ValueInfo


class Accuracy(Metric):

    def __init__(self, threshold=0.5, multi_label=False, dev_env=None):
        super().__init__(dev_env=dev_env)
        self.threshold = threshold
        self.eps = 1e-8
        self.multi_label = multi_label

        # statistics / counts
        self._register_value('correct')
        self._register_value('total')

    def _update(self, predictions, target):
        raise NotImplemented()

    def _compute(self):
        raise NotImplemented()


# class AccuracyTopK(torch.nn.Module):
#
#     def __init__(self, topk=(1, 5), device=None):
#         super().__init__()
#         self.eps = 1e-8
#         self.device = device
#         self.topk = topk
#         self.maxk = max(topk)
#         # FIXME handle distributed operation
#
#         # statistics / counts
#         self.reset()
#
#     def update(self, predictions: torch.Tensor, target: torch.Tensor):
#         sorted_indices = predictions.topk(self.maxk, dim=1)[1]
#         sorted_indices.t_()
#         correct = sorted_indices.eq(target.reshape(1, -1).expand_as(sorted_indices))
#
#         batch_size = target.shape[0]
#         correct_k = {k: correct[:k].reshape(-1).float().sum(0) for k in self.topk}
#         for k, v in correct_k.items():
#             attr = f'_correct_top{k}'
#             old_v = getattr(self, attr)
#             setattr(self, attr, old_v + v)
#         self._total_sum += batch_size
#
#     def reset(self):
#         for k in self.topk:
#             setattr(self, f'_correct_top{k}', torch.tensor(0, dtype=torch.float32))
#         self._total_sum = torch.tensor(0, dtype=torch.float32)
#
#     @property
#     def counts(self):
#         pass
#
#     def compute(self) -> Dict[str, torch.Tensor]:
#         # FIXME handle distributed reduction
#         return {f'top{k}': 100 * getattr(self, f'_correct_top{k}') / self._total_sum for k in self.topk}


class AccuracyTopK(Metric):

    def __init__(self, topk=(1, 5), dev_env: DeviceEnv = None):
        super().__init__(dev_env=dev_env)
        self.eps = 1e-8
        self.topk = topk
        self.maxk = max(topk)

        # statistics / counts
        for k in self.topk:
            self._register_value(f'top{k}')
        self._register_value('total')
        self.reset()

    def _update(self, predictions: torch.Tensor, target: torch.Tensor):
        batch_size = predictions.shape[0]
        sorted_indices = predictions.topk(self.maxk, dim=1)[1]
        target_reshape = target.reshape(-1, 1).expand_as(sorted_indices)
        correct = sorted_indices.eq(target_reshape).float().sum(0)
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
