import torch
import torch.nn as nn
import torchyield as ty

from atompaint.encoders import Encoder, SymEncoder
from atompaint.layers import UnwrapTensor, sym_conv_bn_fourier_layer
from atompaint.field_types import make_fourier_field_types
from atompaint.vendored.escnn_nn_testing import (
        check_equivariance, get_exact_3d_rotations,
)
from escnn.gspaces import rot3dOnR3
from torchtest import assert_vars_change
from functools import partial

# Note that there's not an `atompaint.encoders.cnn` module.  This is because 
# CNNs are simple enough that they can be created entirely with general-purpose 
# blocks.

def test_asym_cnn():
    cnn = Encoder(
            channels=[6, 8, 16],
            block_factories=partial(
                ty.conv3_bn_relu_layer,
                kernel_size=3,
            ),
            block_kwargs=('in_channels', 'out_channels'),
    )

    x = torch.randn(1, 6, 6, 6, 6)
    y = torch.randn(1, 16, 2, 2, 2)

    assert_vars_change(
            model=cnn,
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(cnn.parameters()),
            batch=(x, y),
            device='cpu',
    )

def test_sym_cnn():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    ift_grid = so3.grid('thomson_cube', N=4)

    cnn = SymEncoder(
            in_channels=6,
            field_types=make_fourier_field_types(
                gspace=gspace, 
                channels=[1, 2],
                max_frequencies=2,
            ),
            block_factories=partial(
                sym_conv_bn_fourier_layer,
                ift_grid=ift_grid,
            ),
    )

    x = torch.randn(1, 6, 6, 6, 6)
    y = torch.randn(1, 70, 2, 2, 2)

    assert_vars_change(
            model=nn.Sequential(cnn, UnwrapTensor()),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(cnn.parameters()),
            batch=(x, y),
            device='cpu',
    )

    check_equivariance(
        cnn,
        in_tensor=x,
        out_shape=y.shape,
        group_elements=get_exact_3d_rotations(so3),
        atol=1e-4,
    )
