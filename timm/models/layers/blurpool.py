'''independent attempt to implement

MaxBlurPool2d in a more general fashion(separate maxpooling from BlurPool)
which was again inspired by
Making Convolutional Networks Shift-Invariant Again :cite:`zhang2019shiftinvar`

'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurPool2d(nn.Module):
    r"""Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details.
    Corresponds to the Downsample class, which does blurring and subsampling
    Args:
        channels = Number of input channels
        blur_filter_size (int): filter size for blurring. currently supports either 3 or 5 (most common)
                                defaults to 3.
        stride (int): downsampling filter stride
    Shape:
    Returns:
        torch.Tensor: the transformed tensor.
    Examples:
    """

    def __init__(self, channels=None, blur_filter_size=3, stride=2) -> None:
        super(BlurPool2d, self).__init__()
        assert blur_filter_size in [3, 5]
        self.channels = channels
        self.blur_filter_size = blur_filter_size
        self.stride = stride

        if blur_filter_size == 3:
            pad_size = [1] * 4
            blur_matrix = torch.Tensor([[1., 2., 1]]) / 4 # binomial kernel b2
        else:
            pad_size = [2] * 4
            blur_matrix = torch.Tensor([[1., 4., 6., 4., 1.]]) / 16 # binomial filter kernel b4

        self.padding = nn.ReflectionPad2d(pad_size)
        blur_filter = blur_matrix * blur_matrix.T
        self.register_buffer('blur_filter', blur_filter[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor: # type: ignore
        if not torch.is_tensor(input_tensor):
            raise TypeError("Input input type is not a torch.Tensor. Got {}"
                            .format(type(input_tensor)))
        if not len(input_tensor.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input_tensor.shape))
        # apply blur_filter on input
        return F.conv2d(self.padding(input_tensor), self.blur_filter, stride=self.stride, groups=input_tensor.shape[1])


######################
# functional interface
######################


'''def blur_pool2d() -> torch.Tensor:
    r"""Creates a module that computes pools and blurs and downsample a given
    feature map.
    See :class:`~kornia.contrib.MaxBlurPool2d` for details.
    """
    return BlurPool2d(kernel_size, ceil_mode)(input)'''