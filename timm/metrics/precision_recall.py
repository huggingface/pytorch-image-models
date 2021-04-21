import torch
import torch.nn.functional as F


class PrecisionRecall:

    def __init__(self, threshold=0.5, multi_label=False, device=None):
        self.threshold = threshold
        self.device = device
        self.multi_label = multi_label

        # statistics

        # the total number of true positive instances under each class
        # Shape: (num_classes, )
        self._tp_sum = None

        # the total number of instances
        # Shape: (num_classes, )
        self._total_sum = None

        # the total number of instances under each _predicted_ class,
        # including true positives and false positives
        # Shape: (num_classes, )
        self._pred_sum = None

        # the total number of instances under each _true_ class,
        # including true positives and false negatives
        # Shape: (num_classes, )
        self._true_sum = None

        self.reset()

    def reset(self):
        self._tp_sum = None
        self._total_sum = None
        self._pred_sum = None
        self._true_sum = None

    def update(self, predictions, target):
        output_type = predictions.type()
        num_classes = predictions.size(-1)
        if self.multi_label:
            if self.threshold is not None:
                predictions = (predictions > self.threshold).type(output_type)
            predictions = predictions.t().reshape(num_classes, -1)
            target = target.t().reshape(num_classes, -1)
        else:
            target = F.one_hot(target.view(-1), num_classes=num_classes)
            indices = torch.argmax(predictions, dim=1).view(-1)
            predictions = F.one_hot(indices, num_classes=num_classes)
        # FIXME make sure binary case works

        target = target.type(output_type)
        correct = (target * predictions > 0).type(output_type)
        pred_positives = predictions.sum(dim=0)
        target_positives = target.sum(dim=0)
        if correct.sum() == 0:
            true_positives = torch.zeros_like(pred_positives)
        else:
            true_positives = correct.sum(dim=0)

        if self._tp_sum is None:
            self._tp_sum = torch.zeros(num_classes, device=self.device)
            self._true_sum = torch.zeros(num_classes, device=self.device)
            self._pred_sum = torch.zeros(num_classes, device=self.device)
            self._total_sum = torch.tensor(0, device=self.device)

        self._tp_sum += true_positives
        self._pred_sum += pred_positives
        self._true_sum += target_positives
        self._total_sum += target.shape[0]

    def counts_as_tuple(self, reduce=False):
        tp_sum = self._tp_sum
        pred_sum = self._pred_sum
        true_sum = self._true_sum
        total_sum = self._total_sum
        if reduce:
            tp_sum = reduce_tensor_sum(tp_sum)
            pred_sum = reduce_tensor_sum(pred_sum)
            true_sum = reduce_tensor_sum(true_sum)
            total_sum = reduce_tensor_sum(total_sum)
        return tp_sum, pred_sum, true_sum, total_sum

    def counts(self, reduce=False):
        tp_sum, pred_sum, true_sum, total_sum = self.counts_as_tuple(reduce=reduce)
        return dict(tp_sum=tp_sum, pred_sum=pred_sum, true_sum=true_sum, total_sum=total_sum)

    def confusion(self, reduce=False):
        tp_sum, pred_sum, true_sum, total_sum = self.counts_as_tuple(reduce=reduce)
        fp = pred_sum - tp_sum
        fn = true_sum - tp_sum
        tp = tp_sum
        tn = total_sum - tp - fp - fn
        return dict(tp=tp, fp=fp, fn=fn, tn=tn)

    def compute(self, fscore_beta=1, average='micro', no_reduce=False, distributed=False):
        tp_sum, pred_sum, true_sum, total_sum = self.counts_as_tuple(reduce=distributed)
        if average == 'micro':
            tp_sum = tp_sum.sum()
            pred_sum = pred_sum.sum()
            true_sum = true_sum.sum()

        precision = tp_sum / pred_sum
        recall = tp_sum / true_sum
        beta_sq = fscore_beta ** 2
        f1_denom = beta_sq * precision + recall
        fscore = (1 + beta_sq) * precision * recall / f1_denom

        if average == 'macro' and not no_reduce:
            precision = precision.mean()
            recall = recall.mean()
            fscore = fscore.mean()
            return dict(fscore=fscore, precision=precision, recall=recall)

        return dict(fscore=fscore, precision=precision, recall=recall)
