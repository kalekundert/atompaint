import numpy as np
import parametrize_from_file as pff
import atompaint.datasets.coords as apdc

from io import StringIO
from pytest import approx
from hypothesis import given, settings, example
from hypothesis.extra.numpy import arrays
from scipy.spatial.transform import Rotation
from math import pi
from utils import *

def matrix(params):
    io = StringIO(params)
    return np.loadtxt(io, dtype=float)

def vector(params):
    return np.array([with_math.eval(x) for x in params.split()])

def frame(params):
    origin = coord(params['origin'])
    rot_vec_rad = vector(params['rot_vec_rad'])
    return apdc.make_coord_frame(origin, rot_vec_rad)

def frames(params):
    return [frame(x) for x in params]

def coord(params):
    return matrix(params)

def coords(params):
    coords = matrix(params)
    coords.shape = (1, *coords.shape)[-2:]
    return coords

def float_bounds(max=100):
    return dict(
            min_value=-max,
            max_value=max,
            allow_nan=False,
            allow_infinity=False,
    )


def test_homogenize_coords_1d():
    coords = np.array([1, 2, 3])
    expected = np.array([1, 2, 3, 1])
    np.testing.assert_array_equal(
            apdc.homogenize_coords(coords),
            expected,
    )

def test_homogenize_coords_2d():
    coords = np.array([
        [1, 2, 3],
        [2, 3, 4],
    ])
    expected = np.array([
        [1, 2, 3, 1],
        [2, 3, 4, 1],
    ])
    np.testing.assert_array_equal(
            apdc.homogenize_coords(coords),
            expected,
    )

@pff.parametrize(
        schema=pff.cast(frame_xy=matrix, coords_x=coords, expected_y=coords),
)
def test_transform_coords(frame_xy, coords_x, expected_y):
    coords_x = apdc.homogenize_coords(coords_x)
    coords_y = apdc.transform_coords(coords_x, frame_xy)
    expected_y = apdc.homogenize_coords(expected_y)
    assert coords_y == approx(expected_y)

@pff.parametrize(
        schema=pff.cast(
            origin=coord,
            rot_vec_rad=vector,
            coords_x=coords,
            expected_y=coords,
        ),
)
def test_make_coord_frame(origin, rot_vec_rad, coords_x, expected_y):
    # It's not enough to test the 16 numbers making up the matrix, I need to 
    # test that it transforms things in the way it should.
    frame_xy = apdc.make_coord_frame(origin, rot_vec_rad)
    test_transform_coords(frame_xy, coords_x, expected_y)

    np.testing.assert_allclose(
            apdc.get_origin(frame_xy),
            origin,
    )
    np.testing.assert_allclose(
            apdc.get_rotation_matrix(frame_xy),
            Rotation.from_rotvec(rot_vec_rad).as_matrix(),
    )

@settings(deadline=None)
@given(
        arrays(float, 3, elements=float_bounds()),
        arrays(float, 3, elements=float_bounds(2*pi)),
        arrays(float, 3, elements=float_bounds()),
)
def test_invert_coord_frame(origin, rot_vec_rad, coords_x):
    frame_xy = apdc.make_coord_frame(origin, rot_vec_rad)
    frame_yx = apdc.invert_coord_frame(frame_xy)

    coords_x = apdc.homogenize_coords(coords_x)
    coords_y = apdc.transform_coords(coords_x, frame_xy)
    coords_x2 = apdc.transform_coords(coords_y, frame_yx)

    assert coords_x2 == approx(coords_x)


