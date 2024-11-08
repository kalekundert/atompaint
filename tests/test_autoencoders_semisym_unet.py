import torch
import torch.nn as nn
import pytest

from test_time_embedding import ModuleWrapper, InputWrapper
from atompaint.autoencoders.semisym_unet import SemiSymUNet
from atompaint.autoencoders.sym_unet import SymUNetBlock
from atompaint.autoencoders.asym_unet import AsymConditionedConvBlock, AsymAttentionBlock
from atompaint.layers import UnwrapTensor
from atompaint.field_types import make_fourier_field_types
from atompaint.upsampling import Upsample3d
from atompaint.time_embedding import SinusoidalEmbedding, LinearTimeActivation
from escnn.nn import R3Conv, IIDBatchNorm3d, FourierPointwise, PointwiseAvgPoolAntialiased3D
from escnn.gspaces import rot3dOnR3
from multipartial import multipartial, rows
from torchtest import assert_vars_change

@pytest.mark.parametrize('skip_algorithm', ['cat', 'add'])
def test_semisym_unet(skip_algorithm):
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    grid = so3.grid(type='thomson_cube', N=4)

    def head_factory(in_type, out_type):
        yield R3Conv(in_type, out_type, kernel_size=3, bias=False)
        yield IIDBatchNorm3d(out_type)
        yield FourierPointwise(out_type, grid=grid)

    def tail_factory(in_channels, out_channels):
        yield nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=0,
        )

    def encoder_factory(in_type, out_type, time_dim):
        return SymUNetBlock(
                in_type,
                time_activation=LinearTimeActivation(
                    time_dim=time_dim,
                    activation=FourierPointwise(
                        out_type,
                        grid=grid,
                    )
                ),
                out_activation=FourierPointwise(
                    out_type,
                    grid=grid,
                ),
        )

    def decoder_factory(in_channels, out_channels, time_dim, attention):
        yield AsymConditionedConvBlock(
                in_channels,
                out_channels,
                time_dim=time_dim,
        )

        if attention:
            yield AsymAttentionBlock(
                    out_channels,
                    num_heads=2,
                    channels_per_head=out_channels // 2,
            )

    def latent_factory(in_type, time_dim):
        yield encoder_factory(in_type, in_type, time_dim)
        yield UnwrapTensor()

    def downsample_factory(in_type):
        return PointwiseAvgPoolAntialiased3D(
                in_type,
                sigma=0.6,
                stride=2,
        )

    def upsample_factory(channels):
        return Upsample3d(
                size_expr=lambda x: 2*x - 1,
                mode='trilinear',
        )

    def time_factory(time_dim):
        yield SinusoidalEmbedding(
                out_dim=time_dim,
                min_wavelength=0.1,
                max_wavelength=100,
        )
        yield nn.Linear(time_dim, time_dim)
        yield nn.ReLU()

    unet = SemiSymUNet(
            img_channels=3,
            encoder_types=make_fourier_field_types(gspace, [1, 2, 3], 2),

            head_factory=head_factory,
            tail_factory=tail_factory,

            encoder_factories=multipartial[:,2](encoder_factory),
            decoder_factories=multipartial[:,2](
                decoder_factory,
                attention=rows(False, True),
            ),
            latent_factory=latent_factory,

            downsample_factory=downsample_factory,
            upsample_factory=upsample_factory,

            time_factory=time_factory,
            time_dim=16,

            skip_algorithm=skip_algorithm,
    )

    x = torch.randn(2, 3, 7, 7, 7)
    t = torch.randn(2)
    y = torch.randn(2, 3, 7, 7, 7)

    assert_vars_change(
            model=ModuleWrapper(unet),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(unet.parameters()),
            batch=(InputWrapper(x, t), y),
            device='cpu',
    )
