import atompaint.datasets.voxelize as apdv
import numpy as np
import pandas as pd

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
    return indices(params)

def indices(params):
    io = StringIO(params)
    return np.loadtxt(io, dtype=int)

def image_params(params):
    grid_ = grid(params['grid'])
    channels = params['channels'].split(' ')

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
        schema=pff.cast(radii=with_py.eval, expected=float),
)
def test_get_element_radius(radii, element, expected):
    assert apdv._get_element_radius(radii, element) == expected

@pff.parametrize(
        schema=pff.cast(radii=with_py.eval, expected=float),
)
def test_get_element_channel(channels, element, expected):
    assert apdv._get_element_channel(channels, element, {}) == expected

@pff.parametrize(
        schema=pff.cast(grid=grid, voxels=indices, coords=coords),
)
def test_get_voxel_center_coords(grid, voxels, coords):
    assert apdv._get_voxel_center_coords(grid, voxels) == approx(coords)

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

def test_make_cube():
    # The meat of this function is implemented by `_get_voxel_center_coords()`, 
    # which is tested above.  The test here is just a sanity check.
    grid = apdv.Grid(
            length_voxels=2,
            resolution_A=0.5,
            center_A=np.array([1, 2, 3]),
    )
    cube = apdv._make_cube(grid, np.array([0, 0, 1]))

    assert cube.center_A == approx([0.75, 1.75, 3.25])
    assert cube.length_A == approx(0.5)

@pff.parametrize(
        schema=pff.cast(grid=grid, voxels=indices, expected=indices),
)
def test_discard_voxels_outside_image(grid, voxels, expected):
    assert (apdv._discard_voxels_outside_image(grid, voxels) == expected).all()

@pff.parametrize(
        key=['test_get_voxel_center_coords', 'test_find_voxels_containing_coords'],
        schema=pff.cast(grid=grid, coords=coords, voxels=indices),
)
def test_find_voxels_containing_coords(grid, coords, voxels):
    np.testing.assert_array_equal(
            apdv._find_voxels_containing_coords(grid, coords),
            voxels,
            verbose=True,
    )

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
    voxels = apdv._find_voxels_possibly_contacting_sphere(grid, sphere)
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
        schema=pff.cast(sphere=sphere, cube=cube, expected=with_math.eval)
)
def test_calc_sphere_cube_overlap_volume_A3(sphere, cube, expected):
    assert apdv._calc_sphere_cube_overlap_volume_A3(sphere, cube) == \
            approx(expected * sphere.volume_A3)

@pff.parametrize(
        schema=pff.cast(grid=grid, atom=atom, expected=image)
)
def test_add_atom_to_image(grid, atom, expected):
    img = np.zeros((atom.channel + 1, *grid.shape))
    apdv._add_atom_to_image(img, grid, atom)
    assert_images_match(img, expected)

@pff.parametrize(
        schema=pff.cast(atoms=atoms, img_params=image_params, expected=image)
)
def test_image_from_atoms(atoms, img_params, expected):
    img = apdv.image_from_atoms(atoms, img_params)
    assert_images_match(img, expected)
