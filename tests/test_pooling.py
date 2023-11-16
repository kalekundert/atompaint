import pytest
import torch

from atompaint.pooling import (
        FourierExtremePool3D, FourierAvgPool3D, ExtremePool3D,
)
from atompaint.vendored.escnn_nn_testing import (
        check_equivariance, get_exact_3d_rotations,
)
from escnn.nn import FourierFieldType
from escnn.gspaces import rot3dOnR3

def test_fourier_extreme_pool_3d_equivariance():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup

    pool = FourierExtremePool3D(
            in_type=FourierFieldType(gspace, 1, so3.bl_irreps(2)),
            grid=so3.grid('thomson_cube', N=4),
            kernel_size=2,
    )

    check_equivariance(
            pool,
            in_tensor=(1, 35, 6, 6, 6),
            out_shape=(1, 35, 3, 3, 3),
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

def test_fourier_avg_pool_3d_equivariance():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup

    pool = FourierAvgPool3D(
            in_type=FourierFieldType(gspace, 1, so3.bl_irreps(2)),
            grid=so3.grid('thomson_cube', N=4),
            stride=2,
    )

    check_equivariance(
            pool,
            in_tensor=(1, 35, 5, 5, 5),
            out_shape=(1, 35, 3, 3, 3),
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

@pytest.mark.xfail
def test_extreme_pool_2d():
    f = ExtremePool2D(2)
    x = torch.Tensor([
        [ 1,  4,  2,  2],
        [-1,  0, -2,  1],
        [ 3,  1, -2,  0],
        [ 0,  0, -4,  1],
    ])
    x = x.view(1, 1, 4, 4)

    y = f(x)

    y_expected = torch.Tensor([
        [4, 2],
        [3, -4],
    ])

    torch.testing.assert_close(y, y_expected.view(1, 1, 2, 2))

def test_extreme_pool_3d():
    f = ExtremePool3D(2)

    # Make sure batches and channels are handled correctly.
    x = torch.Tensor([
       [[[[-4,  0,  2, -2],
          [ 1,  1,  2, -1],
          [-3, -2, -3, -3],
          [-1,  0,  0,  2]],

         [[ 0,  0, -2,  0],
          [ 2,  3, -1,  0],
          [ 0, -2,  0, -1],
          [ 1,  0,  0, -1]],

         [[-1, -2, -2, -2],
          [ 3,  0,  0,  0],
          [-1,  4, -1, -1],
          [-2,  1, -2, -1]],

         [[ 0,  0,  0,  0],
          [-3,  2,  1,  0],
          [ 0, -1,  0,  1],
          [-1, -1,  1,  0]]],


        [[[ 2,  1, -1,  3],
          [ 0,  0,  3, -1],
          [-1,  4,  2,  3],
          [ 2, -1,  0, -2]],

         [[-2,  0, -1,  0],
          [ 0, -1, -1, -1],
          [ 3,  1,  0,  1],
          [ 1,  0,  1, -1]],

         [[ 1,  3,  3,  1],
          [-2,  3,  1,  0],
          [ 4,  0,  0, -1],
          [-1,  0,  1,  1]],

         [[-1, -1, -2, -1],
          [-3,  0, -1,  0],
          [-4,  0,  1, -1],
          [ 0, -1,  0,  0]]]],



       [[[[ 2,  0,  0,  1],
          [ 0,  2,  1, -1],
          [ 1,  0,  0, -1],
          [-2,  1,  0,  1]],

         [[ 1, -1,  0,  2],
          [-1, -1, -1,  0],
          [-1,  0, -2, -1],
          [ 2,  3,  0, -1]],

         [[ 0,  0,  1,  2],
          [ 2,  1, -2, -1],
          [-1,  0,  0,  0],
          [ 0, -1,  1,  0]],

         [[ 0,  1, -1, -2],
          [ 2,  0, -1,  0],
          [-1,  0,  3, -3],
          [ 2,  4, -3, -4]]],


        [[[ 1,  3, -1,  1],
          [ 3,  2,  0,  1],
          [-2,  0, -1, -1],
          [-1,  0,  0,  0]],

         [[ 1, -1, -1,  2],
          [ 0, -1,  0,  1],
          [ 0,  1,  1,  0],
          [ 0, -3,  2, -1]],

         [[ 1, -1,  0,  0],
          [ 3, -1,  0,  0],
          [ 0, -3, -1,  1],
          [ 0,  0,  1,  0]],

         [[-2,  0,  1, -2],
          [-1, -1,  0, -1],
          [ 0,  3, -1,  0],
          [ 0,  1, -1,  3]]]],
    ])

    y = f(x)

    y_expected = torch.Tensor([
       [[[[-4,  2],
          [-3, -3]],
          
         [[ 3, -2],
          [ 4, -2]]],
          
        [[[ 2,  3],
          [ 4,  3]],
          
         [[ 3,  3],
          [ 4, -1]]]],
          
       [[[[ 2,  2],
          [ 3, -2]],
          
         [[ 2,  2],
          [ 4, -4]]],
          
        [[[ 3,  2],
          [-3,  2]],
          
         [[ 3, -2],
          [-3,  3]]]],
    ])

    torch.testing.assert_close(y, y_expected)

