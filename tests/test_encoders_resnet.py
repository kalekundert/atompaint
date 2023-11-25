import torch
from atompaint.encoders.resnet import (
        ResNet, make_alpha_block, make_escnn_example_block,
)
from atompaint.encoders.layers import (
        make_conv_layer, make_conv_fourier_layer,
        make_top_level_field_types, make_fourier_field_types,
        make_polynomial_field_types,
)
from atompaint.vendored.escnn_nn_testing import (
        check_equivariance, get_exact_3d_rotations,
)
from escnn.gspaces import rot3dOnR3
from functools import partial

def test_escnn_example_resnet_equivariance():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup

    # Use the same grid and frequency band limit as I intend to use in real 
    # applications.  These parameters can affect how well equivariance is 
    # approximated, so I want to test reasonable values.
    grid = so3.grid('thomson_cube', N=4)
    L = 2

    resnet = ResNet(
            outer_types=make_top_level_field_types(
                gspace=gspace, 
                channels=[6, 2, 2],
                make_nontrivial_field_types=partial(
                    make_polynomial_field_types,
                    terms=[3, 4],
                ),
            ),
            inner_types=make_fourier_field_types(
                gspace=gspace, 
                channels=[2],
                max_frequencies=L,
            ),
            initial_layer_factory=make_conv_layer,
            block_factory=partial(
                make_escnn_example_block,
                grid=grid,
            ),
            block_repeats=1,
            pool_factors=[2],
    )
    check_equivariance(
            resnet,
            in_tensor=(1, 6, 5, 5, 5),
            out_shape=(1, 80, 2, 2, 2),
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

def test_alpha_resnet_equivariance():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup

    # Use the same grid and frequency band limit as I intend to use in real 
    # applications.  These parameters can affect how well equivariance is 
    # approximated, so I want to test reasonably values.
    grid = so3.grid('thomson_cube', N=4)
    L = 2

    resnet = ResNet(
            outer_types = make_top_level_field_types(
                gspace=gspace, 
                channels=[6, 2, 2],
                make_nontrivial_field_types=partial(
                    make_fourier_field_types,
                    max_frequencies=L,
                ),
            ),
            inner_types = make_fourier_field_types(
                gspace=gspace, 
                channels=[2],
                max_frequencies=L,
            ),
            initial_layer_factory=partial(
                make_conv_fourier_layer,
                ift_grid=grid,
            ),
            block_factory=partial(
                make_alpha_block,
                grid=grid,
            ),
            block_repeats=1,
            pool_factors=[2],
    )

    check_equivariance(
            resnet,
            in_tensor=(1, 6, 6, 6, 6),
            out_shape=(1, 70, 2, 2, 2),
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

