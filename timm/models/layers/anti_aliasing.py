import torch
import torch.nn.parallel
import torch.nn as nn
import torch.nn.functional as F


class AntiAliasDownsampleLayer(nn.Module):
    def __init__(self, channels: int = 0, filt_size: int = 3, stride: int = 2, no_jit: bool = False):
        super(AntiAliasDownsampleLayer, self).__init__()
        if no_jit:
            self.op = Downsample(channels, filt_size, stride)
        else:
            self.op = DownsampleJIT(channels, filt_size, stride)

        # FIXME I should probably override _apply and clear DownsampleJIT filter cache for .cuda(), .half(), etc calls

    def forward(self, x):
        return self.op(x)


@torch.jit.script
class DownsampleJIT(object):
    def __init__(self, channels: int = 0, filt_size: int = 3, stride: int = 2):
        self.channels = channels
        self.stride = stride
        self.filt_size = filt_size
        assert self.filt_size == 3
        assert stride == 2
        self.filt = {}  # lazy init by device for DataParallel compat

    def _create_filter(self, like: torch.Tensor):
        filt = torch.tensor([1., 2., 1.], dtype=like.dtype, device=like.device)
        filt = filt[:, None] * filt[None, :]
        filt = filt / torch.sum(filt)
        return filt[None, None, :, :].repeat((self.channels, 1, 1, 1))

    def __call__(self, input: torch.Tensor):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        filt = self.filt.get(str(input.device), self._create_filter(input))
        return F.conv2d(input_pad, filt, stride=2, padding=0, groups=input.shape[1])


class Downsample(nn.Module):
    def __init__(self, channels=None, filt_size=3, stride=2):
        super(Downsample, self).__init__()
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride

        assert self.filt_size == 3
        filt = torch.tensor([1., 2., 1.])
        filt = filt[:, None] * filt[None, :]
        filt = filt / torch.sum(filt)

        # self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])
