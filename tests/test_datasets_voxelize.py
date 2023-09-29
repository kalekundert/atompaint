import atompaint.datasets.voxelize as apdv
import numpy as np
import pandas as pd
import pytest

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

    return apdv.Grid(length_voxels, resolution_A, center_A)

def sphere(params):
    return apdv.Sphere(
            center_A=coord(params['center_A']),
            radius_A=with_math.eval(params['radius_A']),
    )

def cube(params):
    return apdv.Cube(
            center_A=coord(params['center_A']),
            length_A=with_math.eval(params['length_A']),
    )

def atom(params):
    return apdv.Atom(
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
            grid=apdv.Grid(
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
    assert atom.occupancy == 0.8

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
    img = np.zeros((atom.channel + 1, *grid.shape))
    apdv._add_atom_to_image(img, grid, atom)
    assert_images_match(img, expected)

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
    voxels = apdv._find_voxels_possibly_contacting_sphere_jit(
            grid.length_voxels,
            grid.resolution_A,
            grid.center_A,
            sphere.center_A,
            sphere.radius_A,
    )
    voxel_tuples = {
            tuple(x)
            for x in voxels
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
            apdv._find_voxels_containing_coords_jit(
                grid.length_voxels,
                grid.resolution_A,
                grid.center_A,
                coords,
            ),
            voxels,
            verbose=True,
    )

@pff.parametrize(
        schema=pff.cast(grid=grid, voxels=indices, expected=indices),
)
def test_discard_voxels_outside_image(grid, voxels, expected):
    np.testing.assert_array_equal(
            apdv._discard_voxels_outside_image_jit(grid.length_voxels, voxels),
            expected.reshape(-1, 3),
    )

@pff.parametrize(
        schema=pff.cast(grid=grid, voxels=indices, coords=coords),
)
def test_get_voxel_center_coords(grid, voxels, coords):
    actual = apdv._get_voxel_center_coords_jit(
            grid.length_voxels,
            grid.resolution_A,
            grid.center_A,
            voxels,
    )
    assert actual == approx(coords)

def test_get_cube_verts():
    # Be careful to use the "right" types, so we don't unnecessarily compile 
    # another version of this function.
    cube_center_A = coord('1 2 3')
    cube_length_A = 1.0

    verts = apdv._get_cube_verts_jit(cube_center_A, cube_length_A)

    # The specific order of the vertices is important.
    expected = np.array([
        [0.5, 1.5, 2.5],
        [1.5, 1.5, 2.5],
        [1.5, 2.5, 2.5],
        [0.5, 2.5, 2.5],
        [0.5, 1.5, 3.5],
        [1.5, 1.5, 3.5],
        [1.5, 2.5, 3.5],
        [0.5, 2.5, 3.5],
    ])

    assert verts == approx(expected)

@pff.parametrize(
        schema=pff.cast(sphere=sphere, cube=cube, expected=with_math.eval)
)
def test_calc_sphere_cube_overlap_volume_A3(sphere, cube, expected):
    verts = apdv._get_cube_verts_jit(cube.center_A, cube.length_A)
    assert apdv._calc_sphere_cube_overlap_volume_A3(sphere, verts) == \
            approx(expected * sphere.volume_A3)

@pff.parametrize(
        schema=pff.cast(
            overlap_A3=with_math.eval,
            radius_A=with_math.eval,
            occupancy=with_math.eval,
            expected=with_math.eval,
        ),
)
def test_calc_fraction_atom_in_voxel(overlap_A3, radius_A, occupancy, expected):
    voxel = apdv._calc_fraction_atom_in_voxel_jit(
            overlap_A3,
            radius_A,
            occupancy,
    )
    assert voxel == approx(expected)

def test_calc_sphere_volume_A3():
    # https://www.omnicalculator.com/math/sphere-volume
    assert apdv._calc_sphere_volume_A3_jit(1) == approx(4.189, abs=0.001)
    assert apdv._calc_sphere_volume_A3_jit(2) == approx(33.51, abs=0.01)

