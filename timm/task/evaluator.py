"""Dataset-level classification evaluation shared by training and validation."""
import logging
from collections import OrderedDict
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

_logger = logging.getLogger(__name__)


def evaluation_sample_limit(loader) -> Optional[int]:
    """Number of real examples on this rank, excluding distributed sampler padding.

    Forward passes still run on padded batches so distributed model collectives
    remain balanced. Evaluators discard only the extra trailing predictions.
    Samplers without padding and iterable readers need no limit.
    """
    sampler = getattr(loader, 'sampler', None)
    return getattr(sampler, 'num_valid_samples', None)


class ClassificationEvaluator:
    """Accumulate sample-weighted loss and top-k accuracy, expressed as percentages.

    Counts accumulate in int64 and the loss sum in float64 (float32 on MPS, which has
    no float64), both on the given device so no per-batch host sync is needed.

    Args:
        device: Device for the running totals, normally the model output device.
        max_samples: Ignore predictions beyond this many examples (distributed sampler padding).
        criterion: Loss to report, defaults to cross-entropy on class indices.
    """

    default_metric = 'top1'
    metric_names = ('loss', 'top1', 'top5')

    def __init__(self, device=None, *, max_samples: Optional[int] = None, criterion=None):
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.device = torch.device(device or 'cpu')
        self.max_samples = max_samples
        self.reset()

    def reset(self):
        self.num_samples = 0
        loss_dtype = torch.float32 if self.device.type == 'mps' else torch.float64
        self.loss_sum = torch.zeros((), dtype=loss_dtype, device=self.device)
        self.counts = torch.zeros(3, dtype=torch.int64, device=self.device)  # examples, top-1 correct, top-5 correct

    def _prepare(self, output, target):
        if self.max_samples is not None:
            count = max(0, self.max_samples - self.num_samples)
            output, target = output[:count], target[:count]
        self.num_samples += output.shape[0]
        return output.detach().float(), target.detach()

    def _accumulate_loss(self, loss, num_examples):
        self.loss_sum += loss.to(self.loss_sum.dtype) * num_examples
        self.counts[0] += num_examples

    @torch.no_grad()
    def update(self, output, target):
        if output.ndim != 2 or target.ndim != 1 or target.shape[0] != output.shape[0]:
            raise ValueError('Single-label evaluation expects (batch, num_classes) logits and (batch,) class indices.')
        output, target = self._prepare(output, target)
        if not output.shape[0]:
            return
        self._accumulate_loss(self.criterion(output, target), output.shape[0])
        predictions = output.topk(min(5, output.shape[-1]), dim=-1).indices
        correct = predictions.eq(target[:, None])
        self.counts[1] += correct[:, 0].sum()
        self.counts[2] += correct.any(dim=-1).sum()

    def _summary(self, loss_sum, counts):
        count, top1, top5 = counts.tolist()
        count = max(count, 1)
        return OrderedDict(loss=loss_sum.item() / count, top1=100 * top1 / count, top5=100 * top5 / count)

    def summary(self):
        """Return local running metrics for progress logging, without collectives."""
        return self._summary(self.loss_sum, self.counts)

    def compute(self, distributed=False):
        loss_sum, counts = self.loss_sum.clone(), self.counts.clone()
        if distributed:
            dist.all_reduce(loss_sum)
            dist.all_reduce(counts)
        return self._summary(loss_sum, counts)


def _average_precision_columns(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num_examples = output.shape[0]
    order = output.argsort(dim=0, descending=True)
    scores = output.gather(0, order)
    positives = target.gather(0, order).to(torch.float64)
    true_positives = positives.cumsum(0)
    # A threshold includes every example tied at its score, so precision is taken at the end of each tie group.
    ends = torch.ones_like(scores, dtype=torch.bool)
    ends[:-1] = scores[:-1] != scores[1:]
    index = torch.arange(num_examples, device=output.device).unsqueeze(1).expand_as(ends)
    group_end = torch.where(ends, index, torch.full_like(index, num_examples)).flip(0).cummin(0).values.flip(0)
    precision = true_positives.gather(0, group_end) / (group_end + 1).to(torch.float64)
    total = true_positives[-1]
    return torch.where(total > 0, (positives * precision).sum(0) / total.clamp(min=1), torch.zeros_like(total))


def _average_precision(output: torch.Tensor, target: torch.Tensor, chunk_elements: int = 1 << 24) -> torch.Tensor:
    """Per-class non-interpolated AP for (N, C) scores and binary targets.

    Tied scores form a single threshold, matching sklearn's average_precision_score
    per column. Classes without positives get zero. Columns are processed in chunks
    of roughly chunk_elements values to bound the memory used by the sort.
    """
    num_examples, num_classes = output.shape
    if not num_examples or not num_classes:
        return torch.zeros(num_classes, dtype=torch.float64, device=output.device)
    chunk = max(1, chunk_elements // num_examples)
    return torch.cat([
        _average_precision_columns(output[:, i:i + chunk], target[:, i:i + chunk])
        for i in range(0, num_classes, chunk)
    ])


def _gather_rows(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Concatenate the rows of a 2-D tensor from every rank; row counts may differ, including zero."""
    world_size = dist.get_world_size()
    shape = torch.tensor(tensor.shape, dtype=torch.int64, device=device)
    shapes = [torch.zeros_like(shape) for _ in range(world_size)]
    dist.all_gather(shapes, shape)
    max_rows, num_cols = torch.stack(shapes).max(0).values.tolist()
    padded = torch.zeros((max_rows, num_cols), dtype=tensor.dtype, device=device)
    padded[:tensor.shape[0], :tensor.shape[1]] = tensor.to(device)
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return torch.cat([rows[:int(rank_shape[0])].cpu() for rows, rank_shape in zip(gathered, shapes)])


class MultiLabelClassificationEvaluator(ClassificationEvaluator):
    """BCE loss, exact macro AP, and micro/macro/sample F1 for multi-label targets.

    Metrics are percentages. F1 uses sigmoid probabilities >= threshold; zero
    denominators contribute zero. AP uses logits to retain their ranking and
    includes classes with no positive targets as zero. The loss uses the targets
    as given; soft targets are binarized at 0.5 for the ranking / F1 metrics.
    Predictions are stored on CPU and gathered across ranks once at compute(),
    never averaged per batch.

    Args:
        device: Device for the running loss totals and distributed collectives.
        max_samples: Ignore predictions beyond this many examples (distributed sampler padding).
        threshold: Sigmoid probability threshold for the F1 metrics.
    """

    default_metric = 'map'
    metric_names = ('loss', 'map', 'micro_f1', 'macro_f1', 'sample_f1')

    def __init__(self, device=None, *, max_samples: Optional[int] = None, threshold: float = 0.5):
        if not 0. < threshold < 1.:
            raise ValueError('Multi-label prediction threshold must be between 0 and 1.')
        self.threshold = threshold
        self._warned_soft_targets = False
        super().__init__(device, max_samples=max_samples, criterion=nn.BCEWithLogitsLoss())

    def reset(self):
        super().reset()
        self.outputs = []
        self.targets = []

    @torch.no_grad()
    def update(self, output, target):
        if output.ndim != 2 or target.shape != output.shape:
            raise ValueError('Multi-label evaluation expects matching (batch, num_classes) logits and targets.')
        output, target = self._prepare(output, target)
        if not output.shape[0]:
            return
        target = target.float()
        self._accumulate_loss(self.criterion(output, target), output.shape[0])
        if not torch.all((target == 0) | (target == 1)):
            if not self._warned_soft_targets:
                _logger.warning('Multi-label evaluation binarizes soft targets at 0.5 for the mAP and F1 metrics.')
                self._warned_soft_targets = True
            target = target >= 0.5
        self.outputs.append(output.cpu())
        self.targets.append(target.bool().cpu())

    def _summary(self, loss_sum, counts):
        return OrderedDict(loss=loss_sum.item() / max(int(counts[0]), 1))

    def compute(self, distributed=False):
        metrics = super().compute(distributed=distributed)
        if self.outputs:
            output, target = torch.cat(self.outputs), torch.cat(self.targets)
        else:
            output, target = torch.empty(0, 0), torch.empty(0, 0, dtype=torch.bool)
        if distributed:
            output = _gather_rows(output, self.device)
            target = _gather_rows(target.to(torch.uint8), self.device).bool()
        if not output.shape[0]:
            metrics.update(map=0., micro_f1=0., macro_f1=0., sample_f1=0.)
            return metrics
        prediction = output.sigmoid() >= self.threshold
        true_positive = (prediction & target).double()
        denominator = prediction.double() + target.double()
        ap = _average_precision(output, target)
        metrics.update(
            map=100 * ap.mean().item(),
            micro_f1=200 * (true_positive.sum() / denominator.sum().clamp(min=1)).item(),
            macro_f1=200 * (true_positive.sum(0) / denominator.sum(0).clamp(min=1)).mean().item(),
            sample_f1=200 * (true_positive.sum(1) / denominator.sum(1).clamp(min=1)).mean().item(),
        )
        return metrics
