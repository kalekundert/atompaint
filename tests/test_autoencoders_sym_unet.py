import torch
import torch.nn as nn
import pytest
import parametrize_from_file as pff

from atompaint.autoencoders.sym_unet import SymUNet, SymUNetBlock
from atompaint.layers import conv_bn_fourier_layer
from atompaint.field_types import make_fourier_field_types
from atompaint.upsampling import R3Upsampling
from atompaint.time_embedding import SinusoidalEmbedding, LinearTimeActivation
from atompaint.vendored.escnn_nn_testing import (
        check_equivariance, get_exact_3d_rotations,
)
from escnn.nn import (
        FourierFieldType, FourierPointwise, R3ConvTransposed,
        PointwiseAvgPoolAntialiased3D,
)
from escnn.gspaces import rot3dOnR3
from torchtest import assert_vars_change

from test_time_embedding import ModuleWrapper, InputWrapper

# I didn't include an equivariance test here because the model is big enough 
# that its equivariance is too imperfect to test automatically.  However, in 
# experiment #87, I manually confirmed that many different configurations of 
# the U-Net are acceptably equivariant.

@pytest.mark.parametrize('skip_algorithm', ['cat', 'add'])
def test_sym_unet(skip_algorithm):
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    grid = so3.grid(type='thomson_cube', N=4)

    def head_factory(in_type, out_type):
        yield from conv_bn_fourier_layer(in_type, out_type, ift_grid=grid)

    def tail_factory(in_type, out_type):
        yield R3ConvTransposed(in_type, out_type, kernel_size=3)

    def block_factory(in_type, out_type, time_dim):
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

    def latent_factory(in_type, time_dim):
        return block_factory(in_type, in_type, time_dim)

    def downsample_factory(in_type):
        return PointwiseAvgPoolAntialiased3D(
                in_type,
                sigma=0.6,
                stride=2,
        )

    def upsample_factory(in_type):
        return R3Upsampling(
                in_type,
                size_expr=lambda x: 2*x - 1,
        )

    def time_factory(time_dim):
        yield SinusoidalEmbedding(
                out_dim=time_dim,
                min_wavelength=0.1,
                max_wavelength=100,
        )
        yield nn.Linear(time_dim, time_dim)
        yield nn.ReLU()
    
    unet = SymUNet(
            img_channels=3,
            field_types=make_fourier_field_types(
                gspace, 
                channels=[1, 2, 3],
                max_frequencies=1,
            ),
            head_factory=head_factory,
            tail_factory=tail_factory,
            block_factories=[block_factory] * 2,
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

    check_equivariance(
            lambda x: unet(x, t),
            in_tensor=x,
            in_type=unet.in_type,
            out_shape=x.shape,
            out_type=unet.out_type,
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

@pff.parametrize(
        schema=[
            pff.cast(in_channels=int, out_channels=int),
            pff.defaults(
                size_algorithm='padded-conv',
                in_channels=1,
                out_channels=1,
            ),
        ]
)
def test_sym_unet_block_equivariance(size_algorithm, in_channels, out_channels):
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    in_type = FourierFieldType(gspace, in_channels, so3.bl_irreps(1))
    out_type = FourierFieldType(gspace, out_channels, so3.bl_irreps(1))
    grid = so3.grid(type='thomson_cube', N=4)

    block = SymUNetBlock(
            in_type,
            time_activation=LinearTimeActivation(
                time_dim=16,
                activation=FourierPointwise(
                    out_type,
                    grid=grid,
                )
            ),
            out_activation=FourierPointwise(
                out_type,
                grid=grid,
            ),
            size_algorithm=size_algorithm,
    )
    t = torch.randn(2, 16)
    d = block.min_input_size

    check_equivariance(
            lambda x: block(x, t),
            in_tensor=(2, in_type.size, d, d, d),
            in_type=block.in_type,
            out_shape=(2, out_type.size, d, d, d),
            out_type=block.out_type,
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )
