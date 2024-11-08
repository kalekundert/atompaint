import torch
import torch.nn as nn

from atompaint.encoders import SymEncoder
from atompaint.encoders.sym_resnet import (
        make_escnn_example_block, make_alpha_block, make_beta_block,
)
from atompaint.layers import (
        sym_conv_layer, sym_conv_bn_fourier_layer, sym_conv_bn_gated_layer,
        UnwrapTensor,
)
from atompaint.field_types import (
        make_fourier_field_types,
        make_polynomial_field_types, make_exact_polynomial_field_types,
)
from atompaint.vendored.escnn_nn_testing import (
        check_equivariance, get_exact_3d_rotations,
)
from escnn.gspaces import rot3dOnR3
from torchtest import assert_vars_change
from functools import partial
from multipartial import multipartial, rows, cols

def test_escnn_example_resnet():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup

    # Use the same grid and frequency band limit as I intend to use in real 
    # applications.  These parameters can affect how well equivariance is 
    # approximated, so I want to test reasonable values.
    ift_grid = so3.grid('thomson_cube', N=4)
    L = 2

    resnet = SymEncoder(
            in_channels=6,
            field_types=make_polynomial_field_types(
                gspace=gspace, 
                channels=[1, 2],
                terms=[3, 4],
            ),
            head_factory=sym_conv_layer,
            block_factories=multipartial[:,1](
                make_escnn_example_block,
                mid_type=rows(
                    *make_fourier_field_types(
                        gspace=gspace, 
                        channels=[2],
                        max_frequencies=L,
                    ),
                ),
                pool_factor=2,
                ift_grid=ift_grid,
            ),
    )

    x = torch.randn(1, 6, 5, 5, 5)
    y = torch.randn(1, 80, 2, 2, 2)

    assert_vars_change(
            model=nn.Sequential(resnet, UnwrapTensor()),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(resnet.parameters()),
            batch=(x, y),
            device='cpu',
    )

    check_equivariance(
            resnet,
            in_tensor=x,
            out_shape=y.shape,
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

def test_alpha_resnet():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup

    # Use the same grid and frequency band limit as I intend to use in real 
    # applications.  These parameters can affect how well equivariance is 
    # approximated, so I want to test reasonably values.
    ift_grid = so3.grid('thomson_cube', N=4)

    resnet = SymEncoder(
            in_channels=6,
            field_types=make_fourier_field_types(
                gspace=gspace, 
                channels=[1, 2],
                max_frequencies=2,
            ),
            head_factory=partial(
                sym_conv_bn_fourier_layer,
                ift_grid=ift_grid,
            ),
            block_factories=partial(
                make_alpha_block,
                pool_factor=2,
                ift_grid=ift_grid,
            ),
    )

    x = torch.randn(1, 6, 6, 6, 6)
    y = torch.randn(1, 70, 2, 2, 2)

    assert_vars_change(
            model=nn.Sequential(resnet, UnwrapTensor()),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(resnet.parameters()),
            batch=(x, y),
            device='cpu',
    )

    check_equivariance(
            resnet,
            in_tensor=x,
            out_shape=y.shape,
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

def test_beta_resnet():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup

    # I could make the network smaller by using fewer than 3 terms, but I like 
    # the idea of using the same number of terms here as I plan to for real.  
    # Plus, 16 channels is already smaller than what I have in the other tests.

    resnet = SymEncoder(
            in_channels=6,
            field_types=make_exact_polynomial_field_types(
                    gspace=gspace, 
                    channels=[8, 16],
                    terms=3,
                    gated=True,
            ),
            head_factory=sym_conv_bn_gated_layer,
            block_factories=multipartial[1,2](
                make_beta_block,
                pool_factor=cols(2, 1)
            ),
    )

    x = torch.randn(1, 6, 7, 7, 7)
    y = torch.randn(1, 16, 2, 2, 2)

    assert_vars_change(
            model=nn.Sequential(resnet, UnwrapTensor()),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(resnet.parameters()),
            batch=(x, y),
            device='cpu',
    )

    check_equivariance(
            resnet,
            in_tensor=x,
            out_shape=y.shape,
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )
