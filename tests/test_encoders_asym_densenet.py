import torch
import torchyield as ty

from torch import nn
from atompaint.encoders import Encoder
from atompaint.encoders.asym_densenet import AsymDenseBlock
from multipartial import multipartial, rows
from functools import partial
from torchtest import assert_vars_change

def test_asym_densenet():

    def concat_layer(in_channels, out_channels):
        mid_channels = 4 * out_channels

        yield from ty.conv3_bn_relu_layer(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                padding=0,
        )
        yield from ty.conv3_bn_relu_layer(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
        )

    def gather_layer(in_channels, out_channels):
        yield nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=2,
        )

    densenet = Encoder(
            channels=[1, 2, 4, 8],
            head_factory=partial(nn.Conv3d, kernel_size=3),
            block_factories=multipartial[2,1](
                AsymDenseBlock,
                concat_channels=rows(2, 4),
                concat_factories=3 * [concat_layer],
                gather_factory=gather_layer,
            ),
    )

    x = torch.randn(1, 1, 11, 11, 11)
    y = torch.randn(1, 8, 3, 3, 3)

    assert_vars_change(
            model=densenet,
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(densenet.parameters()),
            batch=(x, y),
            device='cpu',
    )


