from torch import nn as nn


class SEModule(nn.Module):

    def __init__(self, channels, reduction=16, act_layer=nn.ReLU, min_channels=8, reduction_channels=None):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduction_channels = reduction_channels or max(channels // reduction, min_channels)
        self.fc1 = nn.Conv2d(
            channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(
            reduction_channels, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()
