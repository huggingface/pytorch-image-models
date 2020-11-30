""" Classifier head and layer factory

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn
from torch.nn import functional as F

from .adaptive_avgmax_pool import SelectAdaptivePool2d
from .linear import Linear


def create_classifier(num_features, num_classes, pool_type='avg', use_conv=False):
    flatten = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv,\
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=flatten)
    num_pooled_features = num_features * global_pool.feat_mult()
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_pooled_features, num_classes, 1, bias=True)
    else:
        # NOTE: using my Linear wrapper that fixes AMP + torchscript casting issue
        fc = Linear(num_pooled_features, num_classes, bias=True)
    return global_pool, fc


class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0.):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool, self.fc = create_classifier(in_chs, num_classes, pool_type=pool_type)

    def forward(self, x):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x
