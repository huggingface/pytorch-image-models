# FastViT for PyTorch
#
# Original implementation and weights from https://github.com/apple/ml-fastvit
#
# For licensing see accompanying LICENSE file at https://github.com/apple/ml-fastvit/tree/main
# Original work is copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import os
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_, create_conv2d, ConvNormAct, SqueezeExcite, use_fused_attn, \
    ClassifierHead
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['FastVit']

def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class MobileOneBlock(nn.Module):
    """MobileOne building block.

    This block has a multi-branched architecture at train-time
    and plain-CNN style architecture at inference time
    For more details, please refer to our paper:
    `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 0,
            inference_mode: bool = False,
            use_se: bool = False,
            use_act: bool = True,
            use_scale_branch: bool = True,
            num_conv_branches: int = 1,
            act_layer: nn.Module = nn.GELU,
    ) -> None:
        """Construct a MobileOneBlock module.

        Args:
            in_chs: Number of channels in the input.
            out_chs: Number of channels produced by the block.
            kernel_size: Size of the convolution kernel.
            stride: Stride size.
            dilation: Kernel dilation factor.
            group_size: Convolution group size.
            inference_mode: If True, instantiates model in inference mode.
            use_se: Whether to use SE-ReLU activations.
            use_act: Whether to use activation. Default: ``True``
            use_scale_branch: Whether to use scale branch. Default: ``True``
            num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = num_groups(group_size, in_chs)
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        self.se = SqueezeExcite(out_chs, rd_divisor=1) if use_se else nn.Identity()

        if inference_mode:
            self.reparam_conv = create_conv2d(
                in_chs,
                out_chs,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=self.groups,
                bias=True,
            )
        else:
            # Re-parameterizable skip connection
            self.reparam_conv = None

            self.identity = (
                nn.BatchNorm2d(num_features=in_chs)
                if out_chs == in_chs and stride == 1
                else None
            )

            # Re-parameterizable conv branches
            if num_conv_branches > 0:
                self.conv_kxk = nn.ModuleList([
                    ConvNormAct(
                        self.in_chs,
                        self.out_chs,
                        kernel_size=kernel_size,
                        stride=self.stride,
                        groups=self.groups,
                        apply_act=False,
                    ) for _ in range(self.num_conv_branches)
                ])
            else:
                self.conv_kxk = None

            # Re-parameterizable scale branch
            self.conv_scale = None
            if kernel_size > 1 and use_scale_branch:
                self.conv_scale = ConvNormAct(
                    self.in_chs,
                    self.out_chs,
                    kernel_size=1,
                    stride=self.stride,
                    groups=self.groups,
                    apply_act=False
                )

        self.act = act_layer() if use_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        # Inference mode forward pass.
        if self.reparam_conv is not None:
            return self.act(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Identity branch output
        identity_out = 0
        if self.identity is not None:
            identity_out = self.identity(x)

        # Scale branch output
        scale_out = 0
        if self.conv_scale is not None:
            scale_out = self.conv_scale(x)

        # Other kxk conv branches
        out = scale_out + identity_out
        if self.conv_kxk is not None:
            for rc in self.conv_kxk:
                out += rc(x)

        return self.act(self.se(out))

    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.reparam_conv is not None:
            return

        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = create_conv2d(
            in_channels=self.in_chs,
            out_channels=self.out_chs,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for name, para in self.named_parameters():
            if 'reparam_conv' in name:
                continue
            para.detach_()

        self.__delattr__("conv_kxk")
        self.__delattr__("conv_scale")
        if hasattr(self, "identity"):
            self.__delattr__("identity")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.conv_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.conv_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.identity is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.identity)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        if self.conv_kxk is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_bn_tensor(self.conv_kxk[ix])
                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(
        self, branch: Union[nn.Sequential, nn.BatchNorm2d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        Args:
            branch: Sequence of ops to be fused.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, ConvNormAct):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_chs // self.groups
                kernel_value = torch.zeros(
                    (self.in_chs, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_chs):
                    kernel_value[
                        i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                    ] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):
    """Building Block of RepLKNet

    This class defines overparameterized large kernel conv block
    introduced in `RepLKNet <https://arxiv.org/abs/2203.06717>`_

    Reference: https://github.com/DingXiaoH/RepLKNet-pytorch
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int,
            stride: int,
            group_size: int,
            small_kernel: Optional[int] = None,
            use_se: bool = False,
            act_layer: Optional[nn.Module] = None,
            inference_mode: bool = False,
    ) -> None:
        """Construct a ReparamLargeKernelConv module.

        Args:
            in_chs: Number of input channels.
            out_chs: Number of output channels.
            kernel_size: Kernel size of the large kernel conv branch.
            stride: Stride size. Default: 1
            group_size: Group size. Default: 1
            small_kernel: Kernel size of small kernel conv branch.
            act_layer: Activation module. Default: ``nn.GELU``
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
        """
        super(ReparamLargeKernelConv, self).__init__()
        self.stride = stride
        self.groups = num_groups(group_size, in_chs)
        self.in_chs = in_chs
        self.out_chs = out_chs

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        if inference_mode:
            self.reparam_conv = create_conv2d(
                in_chs,
                out_chs,
                kernel_size=kernel_size,
                stride=stride,
                dilation=1,
                groups=self.groups,
                bias=True,
            )
        else:
            self.reparam_conv = None
            self.large_conv = ConvNormAct(
                in_chs,
                out_chs,
                kernel_size=kernel_size,
                stride=self.stride,
                groups=self.groups,
                apply_act=False,
            )
            if small_kernel is not None:
                assert (
                    small_kernel <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = ConvNormAct(
                    in_chs,
                    out_chs,
                    kernel_size=small_kernel,
                    stride=self.stride,
                    groups=self.groups,
                    apply_act=False,
                )
        self.se = SqueezeExcite(out_chs, rd_ratio=0.25) if use_se else nn.Identity()
        # FIXME output of this act was not used in original impl, likely due to bug
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is not None:
            out = self.reparam_conv(x)
        else:
            out = self.large_conv(x)
            if self.small_conv is not None:
                out = out + self.small_conv(x)
        out = self.se(out)
        out = self.act(out)
        return out

    def get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepLKNet-pytorch

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        eq_k, eq_b = self._fuse_bn(self.large_conv.conv, self.large_conv.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = self._fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += nn.functional.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4
            )
        return eq_k, eq_b

    def reparameterize(self) -> None:
        """
        Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        eq_k, eq_b = self.get_kernel_bias()
        self.reparam_conv = create_conv2d(
            self.in_chs,
            self.out_chs,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
            bias=True,
        )

        self.reparam_conv.weight.data = eq_k
        self.reparam_conv.bias.data = eq_b
        self.__delattr__("large_conv")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")

    @staticmethod
    def _fuse_bn(
        conv: nn.Conv2d, bn: nn.BatchNorm2d
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to fuse batchnorm layer with conv layer.

        Args:
            conv: Convolutional kernel weights.
            bn: Batchnorm 2d layer.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


def convolutional_stem(
        in_chs: int,
        out_chs: int,
        act_layer: nn.Module = nn.GELU,
        inference_mode: bool = False
) -> nn.Sequential:
    """Build convolutional stem with MobileOne blocks.

    Args:
        in_chs: Number of input channels.
        out_chs: Number of output channels.
        inference_mode: Flag to instantiate model in inference mode. Default: ``False``

    Returns:
        nn.Sequential object with stem elements.
    """
    return nn.Sequential(
        MobileOneBlock(
            in_chs=in_chs,
            out_chs=out_chs,
            kernel_size=3,
            stride=2,
            act_layer=act_layer,
            inference_mode=inference_mode,
        ),
        MobileOneBlock(
            in_chs=out_chs,
            out_chs=out_chs,
            kernel_size=3,
            stride=2,
            group_size=1,
            act_layer=act_layer,
            inference_mode=inference_mode,
        ),
        MobileOneBlock(
            in_chs=out_chs,
            out_chs=out_chs,
            kernel_size=1,
            stride=1,
            act_layer=act_layer,
            inference_mode=inference_mode,
        ),
    )


class Attention(nn.Module):
    """Multi-headed Self Attention module.

    Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            head_dim: int = 32,
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ) -> None:
        """Build MHSA module that can handle 3D or 4D input tensors.

        Args:
            dim: Number of embedding dimensions.
            head_dim: Number of hidden dimensions per head. Default: ``32``
            qkv_bias: Use bias or not. Default: ``False``
            attn_drop: Dropout rate for attention tensor.
            proj_drop: Dropout rate for projection tensor.
        """
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2).transpose(-2, -1)  # (B, N, C)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


class PatchEmbed(nn.Module):
    """Convolutional patch embedding layer."""

    def __init__(
            self,
            patch_size: int,
            stride: int,
            in_chs: int,
            embed_dim: int,
            act_layer: nn.Module = nn.GELU,
            lkc_use_act: bool = False,
            use_se: bool = False,
            inference_mode: bool = False,
    ) -> None:
        """Build patch embedding layer.

        Args:
            patch_size: Patch size for embedding computation.
            stride: Stride for convolutional embedding layer.
            in_chs: Number of channels of input tensor.
            embed_dim: Number of embedding dimensions.
            inference_mode: Flag to instantiate model in inference mode. Default: ``False``
        """
        super().__init__()
        self.proj = nn.Sequential(
            ReparamLargeKernelConv(
                in_chs=in_chs,
                out_chs=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                group_size=1,
                small_kernel=3,
                use_se=use_se,
                act_layer=act_layer if lkc_use_act else None,  # NOTE original weights didn't use this act
                inference_mode=inference_mode,
            ),
            MobileOneBlock(
                in_chs=embed_dim,
                out_chs=embed_dim,
                kernel_size=1,
                stride=1,
                use_se=False,
                act_layer=act_layer,
                inference_mode=inference_mode,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


class LayerScale2d(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim, 1, 1))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class RepMixer(nn.Module):
    """Reparameterizable token mixer.

    For more details, please refer to our paper:
    `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
            self,
            dim,
            kernel_size=3,
            layer_scale_init_value=1e-5,
            inference_mode: bool = False,
    ):
        """Build RepMixer Module.

        Args:
            dim: Input feature map dimension. :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, H, W)`.
            kernel_size: Kernel size for spatial mixing. Default: 3
            layer_scale_init_value: Initial value for layer scale. Default: 1e-5
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                self.dim,
                self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True,
            )
        else:
            self.reparam_conv = None
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                group_size=1,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                group_size=1,
                use_act=False,
            )
            if layer_scale_init_value is not None:
                self.layer_scale = LayerScale2d(dim, layer_scale_init_value)
            else:
                self.layer_scale = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is not None:
            x = self.reparam_conv(x)
        else:
            x = x + self.layer_scale(self.mixer(x) - self.norm(x))
        return x

    def reparameterize(self) -> None:
        """Reparameterize mixer and norm into a single
        convolutional layer for efficient inference.
        """
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if isinstance(self.layer_scale, LayerScale2d):
            w = self.mixer.id_tensor + self.layer_scale.gamma.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale.gamma) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = (
                self.mixer.id_tensor
                + self.mixer.reparam_conv.weight
                - self.norm.reparam_conv.weight
            )
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = create_conv2d(
            self.dim,
            self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        for name, para in self.named_parameters():
            if 'reparam_conv' in name:
                continue
            para.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        self.__delattr__("layer_scale")


class ConvMlp(nn.Module):
    """Convolutional FFN Module."""

    def __init__(
            self,
            in_chs: int,
            hidden_channels: Optional[int] = None,
            out_chs: Optional[int] = None,
            act_layer: nn.Module = nn.GELU,
            drop: float = 0.0,
    ) -> None:
        """Build convolutional FFN module.

        Args:
            in_chs: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_chs: Number of output channels. Default: None
            act_layer: Activation layer. Default: ``GELU``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__()
        out_chs = out_chs or in_chs
        hidden_channels = hidden_channels or in_chs
        self.conv = ConvNormAct(
            in_chs,
            out_chs,
            kernel_size=7,
            groups=in_chs,
            apply_act=False,
        )
        self.fc1 = nn.Conv2d(in_chs, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_chs, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepConditionalPosEnc(nn.Module):
    """Implementation of conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_

    In our implementation, we can reparameterize this module to eliminate a skip connection.
    """

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
            inference_mode=False,
    ) -> None:
        """Build reparameterizable conditional positional encoding

        Args:
            dim: Number of input channels.
            dim_out: Number of embedding dimensions. Default: 768
            spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """
        super(RepConditionalPosEnc, self).__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(spatial_shape, Tuple), (
            f'"spatial_shape" must by a sequence or int, '
            f"get {type(spatial_shape)} instead."
        )
        assert len(spatial_shape) == 2, (
            f'Length of "spatial_shape" should be 2, '
            f"got {len(spatial_shape)} instead."
        )

        self.spatial_shape = spatial_shape
        self.dim = dim
        self.dim_out = dim_out or dim
        self.groups = dim

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                self.dim,
                self.dim_out,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=spatial_shape[0] // 2,
                groups=self.groups,
                bias=True,
            )
        else:
            self.reparam_conv = None
            self.pos_enc = nn.Conv2d(
                self.dim,
                self.dim_out,
                spatial_shape,
                1,
                int(spatial_shape[0] // 2),
                groups=self.groups,
                bias=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is not None:
            x = self.reparam_conv(x)
        else:
            x = self.pos_enc(x) + x
        return x

    def reparameterize(self) -> None:
        # Build equivalent Id tensor
        input_dim = self.dim // self.groups
        kernel_value = torch.zeros(
            (
                self.dim,
                input_dim,
                self.spatial_shape[0],
                self.spatial_shape[1],
            ),
            dtype=self.pos_enc.weight.dtype,
            device=self.pos_enc.weight.device,
        )
        for i in range(self.dim):
            kernel_value[
                i,
                i % input_dim,
                self.spatial_shape[0] // 2,
                self.spatial_shape[1] // 2,
            ] = 1
        id_tensor = kernel_value

        # Reparameterize Id tensor and conv
        w_final = id_tensor + self.pos_enc.weight
        b_final = self.pos_enc.bias

        # Introduce reparam conv
        self.reparam_conv = nn.Conv2d(
            self.dim,
            self.dim_out,
            kernel_size=self.spatial_shape,
            stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        for name, para in self.named_parameters():
            if 'reparam_conv' in name:
                continue
            para.detach_()
        self.__delattr__("pos_enc")


class RepMixerBlock(nn.Module):
    """Implementation of Metaformer block with RepMixer as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
            self,
            dim: int,
            kernel_size: int = 3,
            mlp_ratio: float = 4.0,
            act_layer: nn.Module = nn.GELU,
            proj_drop: float = 0.0,
            drop_path: float = 0.0,
            layer_scale_init_value: float = 1e-5,
            inference_mode: bool = False,
    ):
        """Build RepMixer Block.

        Args:
            dim: Number of embedding dimensions.
            kernel_size: Kernel size for repmixer. Default: 3
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            proj_drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """

        super().__init__()

        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )

        self.mlp = ConvMlp(
            in_chs=dim,
            hidden_channels=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(dim, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.token_mixer(x)
        x = x + self.drop_path(self.layer_scale(self.mlp(x)))
        return x


class AttentionBlock(nn.Module):
    """Implementation of metaformer block with MHSA as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 4.0,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.BatchNorm2d,
            proj_drop: float = 0.0,
            drop_path: float = 0.0,
            layer_scale_init_value: float = 1e-5,
    ):
        """Build Attention Block.

        Args:
            dim: Number of embedding dimensions.
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            norm_layer: Normalization layer. Default: ``nn.BatchNorm2d``
            proj_drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
        """

        super().__init__()

        self.norm = norm_layer(dim)
        self.token_mixer = Attention(dim=dim)
        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale2d(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = ConvMlp(
            in_chs=dim,
            hidden_channels=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        if layer_scale_init_value is not None:
            self.layer_scale_2 = LayerScale2d(dim, layer_scale_init_value)
        else:
            self.layer_scale_2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.layer_scale_1(self.token_mixer(self.norm(x))))
        x = x + self.drop_path2(self.layer_scale_2(self.mlp(x)))
        return x


class FastVitStage(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            depth: int,
            token_mixer_type: str,
            downsample: bool = True,
            se_downsample: bool = False,
            down_patch_size: int = 7,
            down_stride: int = 2,
            pos_emb_layer: Optional[nn.Module] = None,
            kernel_size: int = 3,
            mlp_ratio: float = 4.0,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.BatchNorm2d,
            proj_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            layer_scale_init_value: Optional[float] = 1e-5,
            lkc_use_act=False,
            inference_mode=False,
    ):
        """FastViT stage.

        Args:
            dim: Number of embedding dimensions.
            depth: Number of blocks in stage
            token_mixer_type: Token mixer type.
            kernel_size: Kernel size for repmixer.
            mlp_ratio: MLP expansion ratio.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            proj_drop_rate: Dropout rate.
            drop_path_rate: Drop path rate.
            layer_scale_init_value: Layer scale value at initialization.
            inference_mode: Flag to instantiate block in inference mode.
        """
        super().__init__()
        self.grad_checkpointing = False

        if downsample:
            self.downsample = PatchEmbed(
                patch_size=down_patch_size,
                stride=down_stride,
                in_chs=dim,
                embed_dim=dim_out,
                use_se=se_downsample,
                act_layer=act_layer,
                lkc_use_act=lkc_use_act,
                inference_mode=inference_mode,
            )
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        if pos_emb_layer is not None:
            self.pos_emb = pos_emb_layer(dim_out, inference_mode=inference_mode)
        else:
            self.pos_emb = nn.Identity()

        blocks = []
        for block_idx in range(depth):
            if token_mixer_type == "repmixer":
                blocks.append(RepMixerBlock(
                    dim_out,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    proj_drop=proj_drop_rate,
                    drop_path=drop_path_rate[block_idx],
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=inference_mode,
                ))
            elif token_mixer_type == "attention":
                blocks.append(AttentionBlock(
                    dim_out,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    proj_drop=proj_drop_rate,
                    drop_path=drop_path_rate[block_idx],
                    layer_scale_init_value=layer_scale_init_value,
                ))
            else:
                raise ValueError(
                    "Token mixer type: {} not supported".format(token_mixer_type)
                )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.pos_emb(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class FastVit(nn.Module):
    fork_feat: torch.jit.Final[bool]

    """
    This class implements `FastViT architecture <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
            self,
            in_chans: int = 3,
            layers: Tuple[int, ...] = (2, 2, 6, 2),
            token_mixers: Tuple[str, ...] = ("repmixer", "repmixer", "repmixer", "repmixer"),
            embed_dims: Tuple[int, ...] = (64, 128, 256, 512),
            mlp_ratios: Tuple[float, ...] = (4,) * 4,
            downsamples: Tuple[bool, ...] = (False, True, True, True),
            se_downsamples: Tuple[bool, ...] = (False, False, False, False),
            repmixer_kernel_size: int = 3,
            num_classes: int = 1000,
            pos_embs: Tuple[Optional[nn.Module], ...] = (None,) * 4,
            down_patch_size: int = 7,
            down_stride: int = 2,
            drop_rate: float = 0.0,
            proj_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            layer_scale_init_value: float = 1e-5,
            lkc_use_act: bool = False,
            fork_feat: bool = False,
            cls_ratio: float = 2.0,
            global_pool: str = 'avg',
            norm_layer: nn.Module = nn.BatchNorm2d,
            act_layer: nn.Module = nn.GELU,
            inference_mode: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = 0 if fork_feat else num_classes
        self.fork_feat = fork_feat
        self.global_pool = global_pool
        self.feature_info = []

        # Convolutional stem
        self.stem = convolutional_stem(
            in_chans,
            embed_dims[0],
            act_layer,
            inference_mode,
        )

        # Build the main stages of the network architecture
        prev_dim = embed_dims[0]
        scale = 1
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(layers)).split(layers)]
        stages = []
        for i in range(len(layers)):
            downsample = downsamples[i] or prev_dim != embed_dims[i]
            stage = FastVitStage(
                dim=prev_dim,
                dim_out=embed_dims[i],
                depth=layers[i],
                downsample=downsample,
                se_downsample=se_downsamples[i],
                down_patch_size=down_patch_size,
                down_stride=down_stride,
                pos_emb_layer=pos_embs[i],
                token_mixer_type=token_mixers[i],
                kernel_size=repmixer_kernel_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                proj_drop_rate=proj_drop_rate,
                drop_path_rate=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
                lkc_use_act=lkc_use_act,
                inference_mode=inference_mode,
            )
            stages.append(stage)
            prev_dim = embed_dims[i]
            if downsample:
                scale *= 2
            self.feature_info += [dict(num_chs=prev_dim, reduction=4 * scale, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)
        self.num_stages = len(self.stages)
        self.num_features = self.head_hidden_size = prev_dim

        # For segmentation and detection, extract intermediate output
        if self.fork_feat:
            # Add a norm layer for each output. self.stages is slightly different than self.network
            # in the original code, the PatchEmbed layer is part of self.stages in this code where
            # it was part of self.network in the original code. So we do not need to skip out indices.
            self.out_indices = [0, 1, 2, 3]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get("FORK_LAST3", None):
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f"norm{i_layer}"
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.num_features = self.head_hidden_size = final_features = int(embed_dims[-1] * cls_ratio)
            self.final_conv = MobileOneBlock(
                in_chs=embed_dims[-1],
                out_chs=final_features,
                kernel_size=3,
                stride=1,
                group_size=1,
                inference_mode=inference_mode,
                use_se=True,
                act_layer=act_layer,
                num_conv_branches=1,
            )
            self.head = ClassifierHead(
                final_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=drop_rate,
            )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Init. for classification"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',  # stem and embed
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+).downsample', (0,)),
                (r'^stages\.(\d+).pos_emb', (0,)),
                (r'^stages\.(\d+)\.\w+\.(\d+)', None),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages), indices)

        # forward pass
        x = self.stem(x)
        last_idx = self.num_stages - 1
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]
        feat_idx = 0
        for feat_idx, stage in enumerate(stages):
            x = stage(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        if feat_idx == last_idx:
            x = self.final_conv(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        self.stages = self.stages[:max_index + 1]  # truncate blocks w/ stem as idx 0
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # input embedding
        x = self.stem(x)
        outs = []
        for idx, block in enumerate(self.stages):
            x = block(x)
            if self.fork_feat:
                if idx in self.out_indices:
                    norm_layer = getattr(self, f"norm{idx}")
                    x_out = norm_layer(x)
                    outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        x = self.final_conv(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        if self.fork_feat:
            return x
        x = self.forward_head(x)
        return x


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 256, 256),
        "pool_size": (8, 8),
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        'first_conv': ('stem.0.conv_kxk.0.conv', 'stem.0.conv_scale.conv'),
        "classifier": "head.fc",
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    "fastvit_t8.apple_in1k": _cfg(
        hf_hub_id='timm/'),
    "fastvit_t12.apple_in1k": _cfg(
        hf_hub_id='timm/'),

    "fastvit_s12.apple_in1k": _cfg(
        hf_hub_id='timm/'),
    "fastvit_sa12.apple_in1k": _cfg(
        hf_hub_id='timm/'),
    "fastvit_sa24.apple_in1k": _cfg(
        hf_hub_id='timm/'),
    "fastvit_sa36.apple_in1k": _cfg(
        hf_hub_id='timm/'),

    "fastvit_ma36.apple_in1k": _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95),

    "fastvit_t8.apple_dist_in1k": _cfg(
        hf_hub_id='timm/'),
    "fastvit_t12.apple_dist_in1k": _cfg(
        hf_hub_id='timm/'),

    "fastvit_s12.apple_dist_in1k": _cfg(
        hf_hub_id='timm/',),
    "fastvit_sa12.apple_dist_in1k": _cfg(
        hf_hub_id='timm/',),
    "fastvit_sa24.apple_dist_in1k": _cfg(
        hf_hub_id='timm/',),
    "fastvit_sa36.apple_dist_in1k": _cfg(
        hf_hub_id='timm/',),

    "fastvit_ma36.apple_dist_in1k": _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95
    ),

    "fastvit_mci0.apple_mclip": _cfg(
        hf_hub_id='apple/mobileclip_s0_timm',
        url='https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt',
        crop_pct=0.95,
        num_classes=512,  # CLIP proj dim
        mean=(0., 0., 0.), std=(1., 1., 1.)
    ),
    "fastvit_mci1.apple_mclip": _cfg(
        hf_hub_id='apple/mobileclip_s1_timm',
        url='https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s1.pt',
        crop_pct=0.95,
        num_classes=512,  # CLIP proj dim
        mean=(0., 0., 0.), std=(1., 1., 1.)
    ),
    "fastvit_mci2.apple_mclip": _cfg(
        hf_hub_id='apple/mobileclip_s2_timm',
        url='https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt',
        crop_pct=0.95,
        num_classes=512,  # CLIP proj dim
        mean=(0., 0., 0.), std=(1., 1., 1.)
    ),
})


def checkpoint_filter_fn(state_dict, model):
    """ Remap original checkpoints -> timm """
    if 'stem.0.conv_kxk.0.conv.weight' in state_dict:
        return state_dict  # non-original checkpoint, no remapping needed

    state_dict = state_dict.get('state_dict', state_dict)
    if 'image_encoder.model.patch_embed.0.rbr_conv.0.conv.weight' in state_dict:
        # remap MobileCLIP checkpoints
        prefix = 'image_encoder.model.'
    else:
        prefix = ''

    import re
    import bisect

    # find stage ends by locating downsample layers
    stage_ends = []
    for k, v in state_dict.items():
        match = re.match(r'^(.*?)network\.(\d+)\.proj.*', k)
        if match:
            stage_ends.append(int(match.group(2)))
    stage_ends = list(sorted(set(stage_ends)))

    out_dict = {}
    for k, v in state_dict.items():
        if prefix:
            if prefix not in k:
                continue
            k = k.replace(prefix, '')

        # remap renamed layers
        k = k.replace('patch_embed', 'stem')
        k = k.replace('rbr_conv', 'conv_kxk')
        k = k.replace('rbr_scale', 'conv_scale')
        k = k.replace('rbr_skip', 'identity')
        k = k.replace('conv_exp', 'final_conv')  # to match byobnet, regnet, nfnet
        k = k.replace('lkb_origin', 'large_conv')
        k = k.replace('convffn', 'mlp')
        k = k.replace('se.reduce', 'se.fc1')
        k = k.replace('se.expand', 'se.fc2')
        k = re.sub(r'layer_scale_([0-9])', r'layer_scale_\1.gamma', k)
        if k.endswith('layer_scale'):
            k = k.replace('layer_scale', 'layer_scale.gamma')
        k = k.replace('dist_head', 'head_dist')
        if k.startswith('head.'):
            if k == 'head.proj' and hasattr(model.head, 'fc') and isinstance(model.head.fc, nn.Linear):
                # if CLIP projection, map to head.fc w/ bias = zeros
                k = k.replace('head.proj', 'head.fc.weight')
                v = v.T
                out_dict['head.fc.bias'] = torch.zeros(v.shape[0])
            else:
                k = k.replace('head.', 'head.fc.')

        # remap flat sequential network to stages
        match = re.match(r'^network\.(\d+)', k)
        stage_idx, net_idx = None, None
        if match:
            net_idx = int(match.group(1))
            stage_idx = bisect.bisect_right(stage_ends, net_idx)
        if stage_idx is not None:
            net_prefix = f'network.{net_idx}'
            stage_prefix = f'stages.{stage_idx}'
            if net_prefix + '.proj' in k:
                k = k.replace(net_prefix + '.proj', stage_prefix + '.downsample.proj')
            elif net_prefix + '.pe' in k:
                k = k.replace(net_prefix + '.pe', stage_prefix + '.pos_emb.pos_enc')
            else:
                k = k.replace(net_prefix, stage_prefix + '.blocks')

        out_dict[k] = v
    return out_dict


def _create_fastvit(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', (0, 1, 2, 3))
    model = build_model_with_cfg(
        FastVit,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs
    )
    return model


@register_model
def fastvit_t8(pretrained=False, **kwargs):
    """Instantiate FastViT-T8 model variant."""
    model_args = dict(
        layers=(2, 2, 4, 2),
        embed_dims=(48, 96, 192, 384),
        mlp_ratios=(3, 3, 3, 3),
        token_mixers=("repmixer", "repmixer", "repmixer", "repmixer")
    )
    return _create_fastvit('fastvit_t8', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def fastvit_t12(pretrained=False, **kwargs):
    """Instantiate FastViT-T12 model variant."""
    model_args = dict(
        layers=(2, 2, 6, 2),
        embed_dims=(64, 128, 256, 512),
        mlp_ratios=(3, 3, 3, 3),
        token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
    )
    return _create_fastvit('fastvit_t12', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def fastvit_s12(pretrained=False, **kwargs):
    """Instantiate FastViT-S12 model variant."""
    model_args = dict(
        layers=(2, 2, 6, 2),
        embed_dims=(64, 128, 256, 512),
        mlp_ratios=(4, 4, 4, 4),
        token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
    )
    return _create_fastvit('fastvit_s12', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def fastvit_sa12(pretrained=False, **kwargs):
    """Instantiate FastViT-SA12 model variant."""
    model_args = dict(
        layers=(2, 2, 6, 2),
        embed_dims=(64, 128, 256, 512),
        mlp_ratios=(4, 4, 4, 4),
        pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
        token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
    )
    return _create_fastvit('fastvit_sa12', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def fastvit_sa24(pretrained=False, **kwargs):
    """Instantiate FastViT-SA24 model variant."""
    model_args = dict(
        layers=(4, 4, 12, 4),
        embed_dims=(64, 128, 256, 512),
        mlp_ratios=(4, 4, 4, 4),
        pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
        token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
    )
    return _create_fastvit('fastvit_sa24', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def fastvit_sa36(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant."""
    model_args = dict(
        layers=(6, 6, 18, 6),
        embed_dims=(64, 128, 256, 512),
        mlp_ratios=(4, 4, 4, 4),
        pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
        token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
    )
    return _create_fastvit('fastvit_sa36', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def fastvit_ma36(pretrained=False, **kwargs):
    """Instantiate FastViT-MA36 model variant."""
    model_args = dict(
        layers=(6, 6, 18, 6),
        embed_dims=(76, 152, 304, 608),
        mlp_ratios=(4, 4, 4, 4),
        pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
        token_mixers=("repmixer", "repmixer", "repmixer", "attention")
    )
    return _create_fastvit('fastvit_ma36', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def fastvit_mci0(pretrained=False, **kwargs):
    """Instantiate MCi0 model variant."""
    model_args = dict(
        layers=(2, 6, 10, 2),
        embed_dims=(64, 128, 256, 512),
        mlp_ratios=(3, 3, 3, 3),
        se_downsamples=(False, False, True, True),
        pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
        token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
        lkc_use_act=True,
    )
    return _create_fastvit('fastvit_mci0', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def fastvit_mci1(pretrained=False, **kwargs):
    """Instantiate MCi1 model variant."""
    model_args = dict(
        layers=(4, 12, 20, 4),
        embed_dims=(64, 128, 256, 512),
        mlp_ratios=(3, 3, 3, 3),
        se_downsamples=(False, False, True, True),
        pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
        token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
        lkc_use_act=True,
    )
    return _create_fastvit('fastvit_mci1', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def fastvit_mci2(pretrained=False, **kwargs):
    """Instantiate MCi2 model variant."""
    model_args = dict(
        layers=(4, 12, 24, 4),
        embed_dims=(80, 160, 320, 640),
        mlp_ratios=(3, 3, 3, 3),
        se_downsamples=(False, False, True, True),
        pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
        token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
        lkc_use_act=True,
    )
    return _create_fastvit('fastvit_mci2', pretrained=pretrained, **dict(model_args, **kwargs))
