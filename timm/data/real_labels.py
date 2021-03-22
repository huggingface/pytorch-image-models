""" Real labels evaluator for ImageNet
Paper: `Are we done with ImageNet?` - https://arxiv.org/abs/2006.07159
Based on Numpy example at https://github.com/google-research/reassessed-imagenet

Hacked together by / Copyright 2020 Ross Wightman
"""
import os
import json
import numpy as np


class RealLabelsImagenet:

    def __init__(self, filenames, real_json='real.json', topk=(1, 5)):
        with open(real_json) as real_labels:
            real_labels = json.load(real_labels)
            real_labels = {f'ILSVRC2012_val_{i + 1:08d}.JPEG': labels for i, labels in enumerate(real_labels)}
        self.real_labels = real_labels
        self.filenames = filenames
        assert len(self.filenames) == len(self.real_labels)
        self.topk = topk
        self.is_correct = {k: [] for k in topk}
        self.sample_idx = 0

    def add_result(self, output):
        maxk = max(self.topk)
        _, pred_batch = output.topk(maxk, 1, True, True)
        pred_batch = pred_batch.cpu().numpy()
        for pred in pred_batch:
            filename = self.filenames[self.sample_idx]
            filename = os.path.basename(filename)
            if self.real_labels[filename]:
                for k in self.topk:
                    self.is_correct[k].append(
                        any([p in self.real_labels[filename] for p in pred[:k]]))
            self.sample_idx += 1

    def get_accuracy(self, k=None):
        if k is None:
            return {k: float(np.mean(self.is_correct[k])) * 100 for k in self.topk}
        else:
            return float(np.mean(self.is_correct[k])) * 100
