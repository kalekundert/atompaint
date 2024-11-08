import torch
import torch.nn as nn

from atompaint.encoders import SymEncoder
from atompaint.encoders.sym_densenet import (
        SymDenseBlock, sym_concat_conv_bn_fourier_layer, sym_gather_conv_layer,
)
from atompaint.layers import UnwrapTensor, sym_conv_layer
from atompaint.field_types import make_fourier_field_types
from atompaint.vendored.escnn_nn_testing import (
        check_equivariance, get_exact_3d_rotations,
)
from escnn.gspaces import rot3dOnR3
from torchtest import assert_vars_change
from functools import partial

def test_densenet():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    ift_grid = so3.grid('thomson_cube', N=4)
    L = 2

    def concat_layer(in_type):
        yield from sym_concat_conv_bn_fourier_layer(
                in_type=in_type,
                out_types=make_fourier_field_types(
                    gspace=gspace,
                    channels=[2, 2],
                    max_frequencies=L,
                ),
                ift_grid=ift_grid,
        )

    densenet = SymEncoder(
            in_channels=6,
            field_types=make_fourier_field_types(
                gspace=gspace,
                channels=[2, 2, 2],
                max_frequencies=L,
                unpack=True,
            ),
            head_factory=sym_conv_layer,
            block_factories=partial(
                SymDenseBlock,
                concat_factories=2 * [concat_layer],
                gather_factory=partial(
                    sym_gather_conv_layer,
                    pool_factor=2,
                )
            )
    )

    x = torch.randn(1, 6, 7, 7, 7)
    y = torch.randn(1, 70, 2, 2, 2)

    assert_vars_change(
            model=nn.Sequential(densenet, UnwrapTensor()),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(densenet.parameters()),
            batch=(x, y),
            device='cpu',
    )

    # Spatial dimensions:
    # - input:                       7
    # - after initial convolution:   5
    # - after 1st dense block:       3
    # - after 2nd dense block:       2
    check_equivariance(
            densenet,
            in_tensor=x,
            out_shape=y.shape,
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

def test_dense_block_equivariance():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    ift_grid = so3.grid('thomson_cube', N=4)
    L = 2
    
    in_type, out_type = make_fourier_field_types(
            gspace=gspace,
            channels=[1, 2],
            max_frequencies=L,
    )

    def concat_layer(in_type):
        yield from sym_concat_conv_bn_fourier_layer(
                in_type=in_type,
                out_types=make_fourier_field_types(
                    gspace=gspace,
                    channels=[2, 2],
                    max_frequencies=L,
                ),
                ift_grid=ift_grid,
        )

    block = SymDenseBlock(
            in_type=in_type,
            out_type=out_type,
            concat_factories=2 * [concat_layer],
            gather_factory=partial(
                sym_gather_conv_layer,
                pool_factor=1,
            )
    )

    check_equivariance(
            block,
            in_tensor=(1, 1 * 35, 3, 3, 3),
            out_shape=(1, 2 * 35, 3, 3, 3),
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

