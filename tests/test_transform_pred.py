import atompaint as ap
import torch
import pytest
import parametrize_from_file as pff

from escnn.nn import FieldType
from escnn.gspaces import no_base_space
from escnn.group import SO3
from pytorch3d.transforms import axis_angle_to_matrix
from math import radians
from utils import *

with_math = pff.Namespace('from math import *')

def se3_matrix(params):
    """
    Return the transformation matrix specified by the given test parameters.

    The matrix is returned as a tensor with dimensions (1, 4, 4).  The first 
    axis is the "mini-batch", which is expected by most APIs that interact with 
    pytorch.
    """
    if params == 'identity':
        return torch.eye(4).reshape(1,4,4)

    if 'axis' in params:
        rot_axis = vector_from_str(params.pop('axis'))
        rot_angle = float(params.pop('degrees_ccw'))
        rot_vector = radians(rot_angle) * rot_axis / torch.norm(rot_axis)
        rot_matrix = axis_angle_to_matrix(rot_vector)
    else:
        rot_matrix = torch.eye(3)

    if 'origin' in params:
        trans_vector = vector_from_str(params.pop('origin'))
    else:
        trans_vector = torch.zeros(3)

    if params:
        raise pff.ConfigError(
                lambda e: f"SE(3) matrix has unexpected parameter(s): {', '.join(repr(x) for x in e.unused_params)}",
                unused_params=params,
        )

    se3_matrix = torch.zeros((1,4,4))
    se3_matrix[:, :3, :3] = rot_matrix
    se3_matrix[:, :3, 3] = trans_vector
    se3_matrix[:, 3, 3] = 1

    return se3_matrix

def vector_from_str(vector_str):
    return torch.tensor([float(with_math.eval(x)) for x in vector_str.split()])


@pff.parametrize(
        schema=pff.cast(
            network=with_ap.eval(defer=True),
            input_size=int,
        ),
)
def test_equivariance(network, input_size):
    network = network()

    # The input has an extra "region" dimension that it's incompatible with the 
    # transformation functions provided by escnn.  We work around this by 
    # putting the regions in the batch dimension and reshaping after the 
    # transformation.
    in_shape = network.in_type.size, input_size, input_size, input_size
    x0 = torch.randn(2, *in_shape)
    x = x0.reshape(1, 2, *in_shape)
    f_x = network(x)

    assert torch.det(f_x).detach() == pytest.approx(1)

    group = network.encoder.gspace.fibergroup
    rots = get_exact_rotations(group)

    if isinstance(group, SO3):
        standard_repr = group.standard_representation()
    else:
        standard_repr = group.standard_representation

    std_type = FieldType(no_base_space(group), [standard_repr])

    for rot in rots:
        gx = network.in_type.transform(x0, rot).reshape(1, 2, *in_shape)
        f_gx = network(gx)

        assert torch.det(f_gx).detach() == pytest.approx(1)

        # Need to manually deconstruct the output matrices to check that the 
        # rotation component is invariant while the translation component is 
        # equivariant.

        r_f_x = f_x[:, 0:3, 0:3]
        r_f_gx = f_gx[:, 0:3, 0:3]

        assert torch.allclose(r_f_x, r_f_gx, atol=1e-4)
        
        t_f_x = f_x[:, 0:3, 3].reshape(-1, 3)
        t_f_gx = f_gx[:, 0:3, 3].reshape(-1, 3)
        gt_f_x = std_type.transform(t_f_x, rot)

        assert torch.allclose(gt_f_x, t_f_gx, atol=1e-4)

@pff.parametrize(
        schema=[
            pff.cast(ref=se3_matrix, query=se3_matrix, radius=float, loss=with_math.eval),
            pff.defaults(radius=1.0),
        ],
)
def test_coord_frame_mse_loss(ref, query, radius, loss):
    loss_fn = ap.CoordFrameMseLoss(radius)
    assert loss_fn(query, ref) == pytest.approx(loss)
