import atompaint.transform_pred.datasets.classification as ap
import numpy as np
import parametrize_from_file as pff

from atompaint.transform_pred.datasets.origins import select_origin_filtering_atoms
from atompaint.datasets.coords import transform_coords, get_origin
from scipy.spatial.transform import Rotation
from pytest import approx
from pytest_unordered import unordered
from math import pi
from utils import *

from test_datasets_coords import coords, frame
from test_datasets_atoms import atoms
from test_transform_pred_datasets_origins import origin_params

np.set_printoptions(suppress=True)

def view_slots(params):
    view_slots = ap.make_view_slots(
            pattern=params['pattern'],
            radius=float(params.get('radius', 1)),
    )
    check_origins(view_slots, coords(params['origins']))
    return view_slots

def check_origins(view_slots, origins_a):

    # Check that the slot origins are in the given order, so that future tests 
    # can make assertions using the slot indices.  There's no way to work out 
    # what the indices "should be" a priori, so the order is obtained by 
    # running the code and seeing what it is.  This function then makes sure 
    # that this order doesn't change unexpectedly.

    np.testing.assert_allclose(
            [get_origin(x) for x in view_slots.frames_ab],
            origins_a,
            atol=1e-7,
    )


def test_view_slots_equivariance():
    view_slots = ap.make_view_slots('cube-faces', 1)

    # I worked out the expected results for this function by hand, and they 
    # depend on the indices assigned to each origin.  Here I check that these 
    # indices haven't changed, because if they have, the rest of the test will 
    # be meaningless.

    origins = np.array([
        [ 0,  0,  1],
        [ 0,  0, -1],
        [ 1,  0,  0],
        [ 0,  1,  0],
        [-1,  0,  0],
        [ 0, -1,  0],
    ])
    check_origins(view_slots, origins)

    # Prepare a 90° rotation around the z-axis.

    group = view_slots.representation.group
    g = group.element([0, 0, pi/2], 'EV')

    rotated_slots = [
            (0,0),  # +z is unaffected
            (1,1),  # -z is unaffected
            (2,3),  # +x becomes +y
            (3,4),  # +y becomes -x
            (4,5),  # etc.
            (5,2),
    ]

    # Double-check that the origins rotate as they should.  (This doesn't check 
    # the code, it just checks that I didn't make a mistake writing the test.)

    R = group.standard_representation(g)

    for i_pre, i_post in rotated_slots:
        np.testing.assert_allclose(
                R @ origins[i_pre].reshape(-1, 1),
                origins[i_post].reshape(-1, 1),
                atol=1e-7,
        )

    # Check that the rotation has the intended effect on the slot indices.

    x = np.arange(6)

    # I'm not 100% sure if the representation should be on the left or the 
    # right.  This worries me because I have to make the same decision in the 
    # code under test, so either way I'm not really testing this behavior.
    gx = view_slots.representation(g) @ x

    for i_pre, i_post in rotated_slots:
        assert x[i_pre] == approx(gx[i_post])

@pff.parametrize(
        schema=pff.cast(
            frame_ia=frame,
            view_slots=view_slots,
            origin_params=origin_params,
            atoms_i=atoms,
            expected=with_py.eval,
        )
)
def test_filter_view_slots(frame_ia, view_slots, origin_params, atoms_i, expected):
    relevant_atoms_i = select_origin_filtering_atoms(atoms_i)
    hits = ap._filter_view_slots(frame_ia, view_slots, origin_params, relevant_atoms_i)
    assert hits == unordered(expected)


