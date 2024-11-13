import torch

from torch import nn
from atompaint.encoders import Encoder
from atompaint.encoders.asym_resnet import AsymResBlock
from multipartial import multipartial, cols
from functools import partial
from torchtest import assert_vars_change

def test_asym_resnet():

    def res_block(in_channels, out_channels, *, pool):
        return AsymResBlock(
                in_channels,
                out_channels,
                in_activation=nn.ReLU(),
                out_activation=nn.ReLU(),
                resize=(
                    nn.MaxPool3d(2, ceil_mode=True) if pool else
                    nn.Identity()
                ),
                resize_before_conv=True,
        )

    resnet = Encoder(
            channels=[1, 2, 4, 8],
            head_factory=partial(nn.Conv3d, kernel_size=3),
            block_factories=multipartial[2,2](
                res_block,
                pool=cols(True, False),
            ),
    )

    x = torch.randn(1, 1, 11, 11, 11)
    y = torch.randn(1, 8, 3, 3, 3)

    assert_vars_change(
            model=resnet,
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(resnet.parameters()),
            batch=(x, y),
            device='cpu',
    )

