import torch
from typing import Optional, Tuple, Dict


class Accuracy(torch.nn.Module):

    def __init__(self, threshold=0.5, multi_label=False):
        self.threshold = threshold
        self.eps = 1e-8
        self.multi_label = multi_label

        # statistics / counts
        self._correct_sum = torch.tensor(0, dtype=torch.long)
        self._total_sum = torch.tensor(0, dtype=torch.long)

    def update(self, predictions, target):
        raise NotImplemented()

    def reset(self):
        self._correct_sum = 0
        self._total_sum = 0

    @property
    def counts(self):
        pass

    def compute(self):
        raise NotImplemented()


class AccuracyTopK(torch.nn.Module):

    def __init__(self, topk=(1, 5), device=None):
        super().__init__()
        self.eps = 1e-8
        self.device = device
        self.topk = topk
        self.maxk = max(topk)

        # statistics / counts
        self.reset()

    def update(self, predictions: torch.Tensor, target: torch.Tensor):
        sorted_indices = predictions.topk(self.maxk, dim=1)[1]
        sorted_indices.t_()
        correct = sorted_indices.eq(target.reshape(1, -1).expand_as(sorted_indices))

        batch_size = target.shape[0]
        correct_k = {k: correct[:k].reshape(-1).float().sum(0) for k in self.topk}
        for k, v in correct_k.items():
            attr = f'_correct_top{k}'
            old_v = getattr(self, attr)
            setattr(self, attr, old_v + v)
        self._total_sum += batch_size

    def reset(self):
        for k in self.topk:
            setattr(self, f'_correct_top{k}', torch.tensor(0, dtype=torch.float32))
        self._total_sum = torch.tensor(0, dtype=torch.float32)

    @property
    def counts(self):
        pass

    def compute(self) -> Dict[str, torch.Tensor]:
        return {f'top{k}': 100 * getattr(self, f'_correct_top{k}') / self._total_sum for k in self.topk}


#
# class AccuracyTopK:
#
#     def __init__(self, topk=(1, 5), device=None):
#         self.eps = 1e-8
#         self.device = device
#         self.topk = topk
#         self.maxk = max(topk)
#
#         # statistics / counts
#         self._correct_sum = None
#         self._total_sum = None
#
#     def _check_init(self, device):
#         to_device = self.device if self.device else device
#         if self._correct_sum is None:
#             self._correct_sum = {f'top{k}': torch.tensor(0., device=to_device) for k in self.topk}
#         if self._total_sum is None:
#             self._total_sum = torch.tensor(0, dtype=torch.long, device=to_device)
#
#     def update(self, predictions: torch.Tensor, target: torch.Tensor):
#         sorted_indices = predictions.topk(self.maxk, dim=1)[1]
#         sorted_indices.t_()
#         correct = sorted_indices.eq(target.reshape(1, -1).expand_as(sorted_indices))
#
#         batch_size = target.shape[0]
#         correct_k = {f'top{k}': correct[:k].reshape(-1).float().sum(0) for k in self.topk}
#         self._check_init(device=predictions.device)
#         for k, v in correct_k.items():
#             old_v = self._correct_sum[k]
#             self._correct_sum[k] = old_v + v
#         self._total_sum += batch_size
#
#     def reset(self):
#         self._correct_sum = None
#         self._total_sum = None
#
#     @property
#     def counts(self):
#         pass
#
#     def compute(self) -> Dict[str, torch.Tensor]:
#         assert self._correct_sum is not None and self._total_sum is not None
#         return {k: 100 * v / self._total_sum for k, v in self._correct_sum.items()}
