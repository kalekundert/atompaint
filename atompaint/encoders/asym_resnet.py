import torch.nn as nn

from atompaint.encoders.resnet import ResBlock
from atompaint.utils import identity

from typing import Optional, Callable

def conv1x1x1(in_channels, out_channels):
    return nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
    )

class AsymResBlock(ResBlock):

    def __init__(
            self,
            in_channels,
            out_channels,
            *,
            in_stride: int = 1,
            in_padding: int = 1,
            in_activation: nn.Module,
            out_stride: int = 1,
            out_padding: int = 1,
            out_activation: nn.Module,
            resize: Optional[nn.Module] = None,
            resize_before_conv: bool = False,
            activation_before_skip: bool = False,
            batch_norm: bool = True,
            skip_factory: Callable[[int, int], nn.Module] = conv1x1x1,
            bottleneck_factor: int = 1,
    ):
        super().__init__()

        assert resize_before_conv or in_stride or out_stride

        mid_channels = in_channels // bottleneck_factor
        assert mid_channels > 0

        self.conv1 = nn.Conv3d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=in_stride,
                padding=in_padding,
                bias=not batch_norm,
        )
        self.bn1 = nn.BatchNorm3d(mid_channels) if batch_norm else identity
        self.act1 = in_activation
        self.conv2 = nn.Conv3d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=out_stride,
                padding=out_padding,
                bias=not batch_norm,
        )
        self.bn2 = nn.BatchNorm3d(out_channels) if batch_norm else identity
        self.act2 = out_activation
        self.resize = resize if resize is not None else identity
        self.resize_before_conv = resize_before_conv
        self.activation_before_skip = activation_before_skip

        if in_channels == out_channels:
            self.skip = identity
        else:
            self.skip = skip_factory(in_channels, out_channels)

