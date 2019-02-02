""" Pytorch Wide-Resnet-50-2
Sourced by running https://github.com/clcarwin/convert_torch_to_pytorch (MIT) on
https://github.com/szagoruyko/wide-residual-networks/blob/master/pretrained/README.md
License of above is, as of yet, unclear.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from functools import reduce
from collections import OrderedDict
from .adaptive_avgmax_pool import *

model_urls = {
    'wrn50_2': 'https://www.dropbox.com/s/fe7rj3okz9rctn0/wrn50_2-d98ded61.pth?dl=1',
}


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


def wrn_50_2_features(activation_fn=nn.ReLU()):
    features = nn.Sequential(  # Sequential,
        nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias=False),
        nn.BatchNorm2d(64),
        activation_fn,
        nn.MaxPool2d((3, 3), (2, 2), (1, 1)),
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(64, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(128),
                              activation_fn,
                              nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(128),
                              activation_fn,
                              nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                          ),
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                          ),
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(128),
                              activation_fn,
                              nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(128),
                              activation_fn,
                              nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(128),
                              activation_fn,
                              nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(128),
                              activation_fn,
                              nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
        ),
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              activation_fn,
                              nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              activation_fn,
                              nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                          ),
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(256, 512, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                          ),
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              activation_fn,
                              nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              activation_fn,
                              nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              activation_fn,
                              nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              activation_fn,
                              nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              activation_fn,
                              nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(256),
                              activation_fn,
                              nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
        ),
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              activation_fn,
                              nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              activation_fn,
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 1024, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              activation_fn,
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              activation_fn,
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              activation_fn,
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              activation_fn,
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              activation_fn,
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              activation_fn,
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              activation_fn,
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              activation_fn,
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              activation_fn,
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                              activation_fn,
                              nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
        ),
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                              activation_fn,
                              nn.Conv2d(1024, 1024, (3, 3), (2, 2), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                              activation_fn,
                              nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(2048),
                          ),
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(2048),
                          ),
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                              activation_fn,
                              nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                              activation_fn,
                              nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(2048),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                              activation_fn,
                              nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                              activation_fn,
                              nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(2048),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                activation_fn,
            ),
        ),
    )
    return features


class Wrn50_2(nn.Module):
    def __init__(self, num_classes=1000, activation_fn=nn.ReLU(), drop_rate=0., global_pool='avg'):
        super(Wrn50_2, self).__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = 2048
        self.global_pool = global_pool
        self.features = wrn_50_2_features(activation_fn=activation_fn)
        self.fc = nn.Linear(2048, num_classes)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.fc = nn.Linear(2048, num_classes)

    def forward_features(self, x, pool=True):
        x = self.features(x)
        if pool:
            x = adaptive_avgmax_pool2d(x, self.global_pool)
            x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x, pool=True)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


def wrn50_2(pretrained=False, num_classes=1000, **kwargs):
    model = Wrn50_2(num_classes=num_classes, **kwargs)
    if pretrained:
        # Remap pretrained weights to match our class module with features + fc
        pretrained_weights = model_zoo.load_url(model_urls['wrn50_2'])
        feature_keys = filter(lambda k: '10.1.' not in k, pretrained_weights.keys())
        remapped_weights = OrderedDict()
        for k in feature_keys:
            remapped_weights['features.' + k] = pretrained_weights[k]
        remapped_weights['fc.weight'] = pretrained_weights['10.1.weight']
        remapped_weights['fc.bias'] = pretrained_weights['10.1.bias']
        model.load_state_dict(remapped_weights)
    return model