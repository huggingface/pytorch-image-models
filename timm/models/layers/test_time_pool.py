""" Test Time Pooling (Average-Max Pool)

Hacked together by Ross Wightman
"""

import logging
from torch import nn
import torch.nn.functional as F

from .adaptive_avgmax_pool import adaptive_avgmax_pool2d


class TestTimePoolHead(nn.Module):
    def __init__(self, base, original_pool=7):
        super(TestTimePoolHead, self).__init__()
        self.base = base
        self.original_pool = original_pool
        base_fc = self.base.get_classifier()
        if isinstance(base_fc, nn.Conv2d):
            self.fc = base_fc
        else:
            self.fc = nn.Conv2d(
                self.base.num_features, self.base.num_classes, kernel_size=1, bias=True)
            self.fc.weight.data.copy_(base_fc.weight.data.view(self.fc.weight.size()))
            self.fc.bias.data.copy_(base_fc.bias.data.view(self.fc.bias.size()))
        self.base.reset_classifier(0)  # delete original fc layer

    def forward(self, x):
        x = self.base.forward_features(x)
        x = F.avg_pool2d(x, kernel_size=self.original_pool, stride=1)
        x = self.fc(x)
        x = adaptive_avgmax_pool2d(x, 1)
        return x.view(x.size(0), -1)


def apply_test_time_pool(model, config, args):
    test_time_pool = False
    if not hasattr(model, 'default_cfg') or not model.default_cfg:
        return model, False
    if not args.no_test_pool and \
            config['input_size'][-1] > model.default_cfg['input_size'][-1] and \
            config['input_size'][-2] > model.default_cfg['input_size'][-2]:
        logging.info('Target input size %s > pretrained default %s, using test time pooling' %
                     (str(config['input_size'][-2:]), str(model.default_cfg['input_size'][-2:])))
        model = TestTimePoolHead(model, original_pool=model.default_cfg['pool_size'])
        test_time_pool = True
    return model, test_time_pool
