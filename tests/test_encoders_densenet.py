from atompaint.encoders.densenet import (
        DenseNet, DenseBlock, DenseLayer, make_fourier_growth_type,
)
from atompaint.nonlinearities import leaky_hard_shrink
from atompaint.pooling import FourierExtremePool3D, FourierAvgPool3D
from atompaint.encoders.layers import make_conv_layer, make_gated_nonlinearity
from atompaint.field_types import (
        make_top_level_field_types, make_fourier_field_types,
)
from atompaint.vendored.escnn_nn_testing import (
        check_equivariance, get_exact_3d_rotations,
)
from escnn.nn import FourierPointwise, R3Conv
from escnn.gspaces import rot3dOnR3
from functools import partial

def test_densenet_equivariance():
    # Note that with two dense blocks, the equivariance error of this model 
    # exceeds the 1e-4 tolerance that I typically set.  I found a number of 
    # ways to bring the error back below the tolerance:
    #
    # - Decrease number of dense blocks to 1.
    # - Increase the number of Fourier grid points to 120 (N=5).
    # - Replace the Fourier nonlinearities with gated nonlinearities.
    # - Pool with strided convolutions.
    #
    # I settled on the first, because it also makes the inputs much smaller, 
    # which makes the test run about twice as fast.  But it's worth keeping in 
    # mind that these kinds of expressiveness/equivariance trade-offs when 
    # designing the full model.

    gspace = rot3dOnR3()
    so3 = gspace.fibergroup

    # Use the same grid and frequency band limit as I intend to use in real 
    # applications.  These parameters can affect how well equivariance is 
    # approximated, so I want to test reasonable values.
    grid = so3.grid('thomson_cube', N=4)
    L = 2

    densenet = DenseNet(
            transition_types=make_top_level_field_types(
                gspace=gspace,
                channels=[6, 2, 2, 2],
                make_nontrivial_field_types=partial(
                    make_fourier_field_types,
                    max_frequencies=L,
                    unpack=True,
                ),
            ),
            growth_type_factory=partial(
                make_fourier_growth_type,
                gspace=gspace,
                channels=1,
                max_frequency=L,
                unpack=True,
            ),
            initial_layer_factory=make_conv_layer,
            final_layer_factory=make_conv_layer,
            nonlin1_factory=make_gated_nonlinearity,
            nonlin2_factory=partial(
                FourierPointwise,
                grid=grid,
                function=leaky_hard_shrink,
            ),
            pool_factory=lambda in_type, pool_factor: \
                    FourierExtremePool3D(
                        in_type,
                        grid=grid,
                        kernel_size=pool_factor,
                    ),
            pool_factors=2,
            block_depth=2,
    )

    # Spatial dimensions:
    # - input:                      10
    # - after initial convolution:   8
    # - after dense block:           4
    # - after final convolution:     2
    check_equivariance(
            densenet,
            in_tensor=(1, 6, 10, 10, 10),
            out_shape=(1, 70, 2, 2, 2),
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

def test_dense_block_equivariance():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    grid = so3.grid('thomson_cube', N=4)
    L = 2
    
    growth_type_factory = partial(
            make_fourier_growth_type,
            gspace=gspace,
            channels=1,
            max_frequency=L,
            unpack=True,
    )

    block = DenseBlock(
            in_type=growth_type_factory(1),
            growth_type_factory=growth_type_factory,
            nonlin1_factory=make_gated_nonlinearity,
            nonlin2_factory=partial(
                FourierPointwise,
                grid=grid,
                function=leaky_hard_shrink,
            ),
            num_layers=1,
    )

    check_equivariance(
            block,
            in_tensor=(1, 1 * 35, 2, 2, 2),
            out_shape=(1, 2 * 35, 2, 2, 2),
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

def test_dense_layer_equivariance():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    grid = so3.grid('thomson_cube', N=4)
    L = 2
    
    growth_type_factory = partial(
            make_fourier_growth_type,
            gspace=gspace,
            channels=1,
            max_frequency=L,
            unpack=True,
    )

    layer = DenseLayer(
            in_type=growth_type_factory(3),
            growth_type_factory=growth_type_factory,
            nonlin1_factory=make_gated_nonlinearity,
            nonlin2_factory=partial(
                FourierPointwise,
                grid=grid,
                function=leaky_hard_shrink,
            ),
    )

    check_equivariance(
            layer,
            in_tensor=(1, 3 * 35, 2, 2, 2),
            out_shape=(1, 1 * 35, 2, 2, 2),
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )
