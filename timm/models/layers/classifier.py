from torch import nn as nn
from torch.nn import functional as F

from .adaptive_avgmax_pool import SelectAdaptivePool2d


class ClassifierHead(nn.Module):
    """Classifier Head w/ configurable global pooling and dropout."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0.):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        if num_classes > 0:
            self.fc = nn.Linear(in_chs * self.global_pool.feat_mult(), num_classes, bias=True)
        else:
            self.fc = nn.Identity()

    def forward(self, x):
        x = self.global_pool(x).flatten(1)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x
