
from atompaint.encoders.cnn import IcosahedralCnn, FourierCnn
from atompaint.vendored.escnn_nn_testing import check_equivariance
from utils import *

# For these tests, we construct the inputs very carefully to ensure that 
# equivariance only reflects on the algorithms themselves, not artifacts.

# Important for output to be exactly right size, otherwise rotations will not 
# be equivariant.

def test_icosahedral_equivariance():
    # Just use a single channel and the smallest possible fields of view.  
    # That's enough to test equivariance, and keeps the test running fast by 
    # not requiring too much memory.
    module = IcosahedralCnn(
            channels=[1,1,1],
            conv_field_of_view=3,
            pool_field_of_view=2,
    )

    c = module.fields[0].size
    rots = get_exact_rotations(module.gspace.fibergroup)

    # The input needs to be big enough that the input for each layer will be 
    # bigger than the convolutional filter.
    check_equivariance(
        module,
        in_tensor=(1, c, 30, 30, 30),
        group_elements=rots,
        atol=1e-4,
    )

def test_fourier_equivariance():
    module = FourierCnn(
            channels=[1,1,1],
            conv_field_of_view=3,
    )

    c = module.fields[0].size
    rots = get_exact_rotations(module.gspace.fibergroup)

    check_equivariance(
        module,
        in_tensor=(1, c, 23, 23, 23),
        group_elements=rots,
        atol=1e-4,
    )
