from atompaint.upsampling import R3Upsampling
from escnn.nn import FourierFieldType
from escnn.gspaces import rot3dOnR3
from atompaint.vendored.escnn_nn_testing import (
        check_equivariance, get_exact_3d_rotations,
)

def test_upsampling_equivariance():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    in_type = FourierFieldType(gspace, 2, so3.bl_irreps(2))

    up = R3Upsampling(
            in_type,
            size_expr=lambda x: 2*x + 1,
    )

    check_equivariance(
            up,
            in_tensor=(2, 70, 2, 2, 2),
            out_shape=(2, 70, 5, 5, 5),
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

    
