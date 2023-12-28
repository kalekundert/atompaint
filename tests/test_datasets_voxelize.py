import atompaint.datasets.voxelize as apdv
import atompaint.datasets._voxelize as apdv_c
import numpy as np
import pandas as pd
import pytest
import pickle

from atompaint.datasets.voxelize import Sphere, Atom, Grid
from io import StringIO
from itertools import product
from collections import namedtuple
from test_datasets_coords import coord, coords
from test_datasets_atoms import atoms
from utils import *

with_apdv = pff.Namespace('from atompaint.datasets.voxelize import *')

def grid(params):
    if isinstance(params, str):
        length_voxels = int(params)
        resolution_A = 1.0
        center_A = np.zeros(3)

    else:
        params = params.copy()
        length_voxels = int(params.pop('length_voxels'))
        resolution_A = float(params.pop('resolution_A', 1.0))
        center_A = coord(params.pop('center_A', '0 0 0'))

        if params:
            raise ValueError(f"unexpected grid parameter(s): {list(params)}")

    return Grid(length_voxels, resolution_A, center_A)

def sphere(params):
    return Sphere(
            center_A=coord(params['center_A']),
            radius_A=with_math.eval(params['radius_A']),
    )

def cube(params):
    return apdv.Cube(
            center_A=coord(params['center_A']),
            length_A=with_math.eval(params['length_A']),
    )

def atom(params):
    return Atom(
            sphere=sphere(params),
            channel=int(params['channel']),
            occupancy=float(params.get('occupancy', 1)),
    )

def index(params):
    return np.array([int(x) for x in params.split()])

def indices(params):
    io = StringIO(params)
    indices = np.loadtxt(io, dtype=int)
    indices.shape = (1, *indices.shape)[-2:]
    return indices

def image_params(params):
    grid_ = grid(params['grid'])
    channels = params.get('channels', '.*').split(' ')

    try:
        radii = params['element_radii_A']
    except KeyError:
        radii = grid_.resolution_A / 2
    else:
        if isinstance(radii, dict):
            radii = {k: float(v) for k, v in radii.items()}
        else:
            radii = float(radii)

    return apdv.ImageParams(
            grid=grid_,
            channels=channels,
            element_radii_A=radii,
    )

def image(params):
    return {
            tuple(index(k)): with_math.eval(v)
            for k, v in params.items()
    }

def assert_images_match(actual, expected):
    axes = [range(x) for x in actual.shape]
    for i in product(*axes):
        assert actual[i] == approx(expected.get(i, 0))


@pff.parametrize(
        schema=pff.cast(atoms=atoms, img_params=image_params, expected=image)
)
def test_image_from_atoms(atoms, img_params, expected):
    img = apdv.image_from_atoms(atoms, img_params)
    assert_images_match(img, expected)

def test_make_empty_image():
    img_params = apdv.ImageParams(
            grid=Grid(
                length_voxels=3,
                resolution_A=1,         # not relevant
                center_A=np.zeros(3),   # not relevant
            ),
            channels=['C', '*'],
            element_radii_A=1,          # not relevant
    )
    np.testing.assert_array_equal(
            apdv._make_empty_image(img_params),
            np.zeros((2, 3, 3, 3)),
            verbose=True,
    )

@pff.parametrize(
        schema=pff.cast(atoms=atoms, img_params=image_params, expected=atoms),
)
def test_discard_atoms_outside_image(atoms, img_params, expected):
    actual = apdv._discard_atoms_outside_image(atoms, img_params)
    pd.testing.assert_frame_equal(actual, expected)

def test_make_atom():
    Row = namedtuple('Row', ['element', 'x', 'y', 'z', 'occupancy'])
    row = Row('C', 1, 2, 3, 0.8)
    img_params = apdv.ImageParams(
            grid=None,
            channels=['C', '*'],
            element_radii_A=1,
    )
    atom = apdv._make_atom(row, img_params, {})

    assert atom.sphere.center_A == approx([1, 2, 3])
    assert atom.sphere.radius_A == approx(1)
    assert atom.channel == 0
    assert atom.occupancy == approx(0.8)

@pff.parametrize(
        schema=pff.cast(radii=with_py.eval, expected=float),
)
def test_get_element_radius(radii, element, expected):
    assert apdv._get_element_radius(radii, element) == expected

@pff.parametrize(
        schema=pff.cast(radii=with_py.eval, expected=float),
)
def test_get_element_channel(channels, element, expected):
    assert apdv.get_element_channel(channels, element, {}) == expected


@pff.parametrize(
        schema=pff.cast(grid=grid, atom=atom, expected=image)
)
def test_add_atom_to_image(grid, atom, expected):
    img = np.zeros((atom.channel + 1, *grid.shape), dtype=np.float32)
    apdv_c._add_atom_to_image(img, grid, atom)
    assert_images_match(img, expected)

def test_add_atom_to_image_no_copy():
    grid = Grid(
            length_voxels=3,
            resolution_A=1,
    )
    atom = Atom(
            sphere=Sphere(
                center_A=np.zeros(3),
                radius_A=1,
            ),
            channel=0,
            occupancy=1,
    )

    # `float64` is the wrong data type; `float32` is required.  The binding 
    # code should notice the discrepancy and complain.
    img = np.zeros((2, 3, 3, 3), dtype=np.float64)

    with pytest.raises(TypeError):
        apdv_c._add_atom_to_image(img, grid, atom)

@pff.parametrize(
        schema=pff.cast(
            grid=grid,
            sphere=sphere,
            expected=pff.cast(
                min_index=index,
                max_index=index,
            ),
        ),
)
def test_find_voxels_possibly_contacting_sphere(grid, sphere, expected):
    voxels = apdv_c._find_voxels_possibly_contacting_sphere(grid, sphere)
    voxel_tuples = {
            tuple(x)
            for x in voxels.T
    }

    if expected == 'empty':
        expected_tuples = set()
    else:
        axes = [
                range(expected['min_index'][i], expected['max_index'][i] + 1)
                for i in range(3)
        ]
        expected_tuples = {
                (i, j, k)
                for i, j, k in product(*axes)
        }

    assert voxel_tuples >= expected_tuples

@pff.parametrize(
        key=['test_get_voxel_center_coords', 'test_find_voxels_containing_coords'],
        schema=pff.cast(grid=grid, coords=coords, voxels=indices),
)
def test_find_voxels_containing_coords(grid, coords, voxels):
    np.testing.assert_array_equal(
            apdv_c._find_voxels_containing_coords(grid, coords.T),
            voxels.T,
            verbose=True,
    )

@pff.parametrize(
        schema=pff.cast(grid=grid, voxels=indices, expected=indices),
)
def test_discard_voxels_outside_image(grid, voxels, expected):
    np.testing.assert_array_equal(
            apdv_c._discard_voxels_outside_image(grid, voxels.T),
            expected.reshape(-1, 3).T,
    )

@pff.parametrize(
        schema=pff.cast(grid=grid, voxels=indices, coords=coords),
)
def test_get_voxel_center_coords(grid, voxels, coords):
    actual = apdv.get_voxel_center_coords(grid, voxels)
    assert actual == approx(coords)


def test_sphere_attrs():
    s = Sphere(
            center_A=np.array([1,2,3]),
            radius_A=4,
    )
    assert s.center_A == approx([1,2,3])
    assert s.radius_A == 4

    # https://www.omnicalculator.com/math/sphere-volume
    assert s.volume_A3 == approx(268.1, abs=0.1)

def test_sphere_repr():
    s = Sphere(
            center_A=np.array([1,2,3]),
            radius_A=4,
    )
    s_repr = eval(repr(s))

    np.testing.assert_array_equal(s_repr.center_A, [1,2,3])
    assert s_repr.radius_A == 4

def test_sphere_pickle():
    s = Sphere(
            center_A=np.array([1,2,3]),
            radius_A=4,
    )
    s_pickle = pickle.loads(pickle.dumps(s))

    np.testing.assert_array_equal(s_pickle.center_A, [1,2,3])
    assert s_pickle.radius_A == 4


def test_grid_attrs():
    g = Grid(
            center_A=np.array([1,2,3]),
            length_voxels=4,
            resolution_A=0.5,
    )
    assert g.center_A == approx([1,2,3])
    assert g.length_voxels == 4
    assert g.resolution_A == 0.5

def test_grid_repr():
    g = Grid(
            center_A=np.array([1,2,3]),
            length_voxels=4,
            resolution_A=0.5,
    )
    g_repr = eval(repr(g))

    np.testing.assert_array_equal(g_repr.center_A, [1,2,3])
    assert g_repr.length_voxels == 4
    assert g_repr.resolution_A == 0.5

def test_grid_pickle():
    g = Grid(
            center_A=np.array([1,2,3]),
            length_voxels=4,
            resolution_A=0.5,
    )
    g_pickle = pickle.loads(pickle.dumps(g))

    np.testing.assert_array_equal(g_pickle.center_A, [1,2,3])
    assert g_pickle.length_voxels == 4
    assert g_pickle.resolution_A == 0.5


def test_atom_attrs():
    a = Atom(
            sphere=Sphere(
                center_A=np.array([1,2,3]),
                radius_A=4,
            ),
            channel=0,
            occupancy=0.5,
    )
    assert a.sphere.center_A == approx([1,2,3])
    assert a.sphere.radius_A == 4
    assert a.channel == 0
    assert a.occupancy == 0.5

def test_atom_repr():
    a = Atom(
            sphere=Sphere(
                center_A=np.array([1,2,3]),
                radius_A=4,
            ),
            channel=0,
            occupancy=0.5,
    )
    a_repr = eval(repr(a))

    np.testing.assert_array_equal(a_repr.sphere.center_A, [1,2,3])
    assert a_repr.sphere.radius_A == 4
    assert a_repr.channel == 0
    assert a_repr.occupancy == 0.5

def test_atom_pickle():
    a = Atom(
            sphere=Sphere(
                center_A=np.array([1,2,3]),
                radius_A=4,
            ),
            channel=0,
            occupancy=0.5,
    )
    a_pickle = pickle.loads(pickle.dumps(a))

    np.testing.assert_array_equal(a_pickle.sphere.center_A, [1,2,3])
    assert a_pickle.sphere.radius_A == 4
    assert a_pickle.channel == 0
    assert a_pickle.occupancy == 0.5
