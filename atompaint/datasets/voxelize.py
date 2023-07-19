import numpy as np
import pandera as pa
import overlap
import re

from itertools import product
from functools import cached_property
from dataclasses import dataclass, field
from numbers import Real
from math import pi

from typing import TypeAlias, Optional
from numpy.typing import NDArray
from atompaint.datasets.atoms import Atoms

"""\
Data structures and naming conventions
======================================
This list only includes data types that don't have their own classes.

`img`:
  A `np.ndarray` that contains the voxelized scene.

`atoms`:
  A `pandas.DataFrame` with the following columns: x, y, z, element.  This is 
  the way atoms are expected to be represented externally.  Internally, the 
  Atom data class is used to represent individual atoms.

`voxel`:
  A 3D `numpy.ndarray` containing the index for one of the cells in the image.  
  Generally, these indices are constrained to actually fall within the image in 
  question (i.e. no indices greater than the size of the image or less than 0).  
  Note that a `grid` object is needed to determine the physical location of the 
  voxel.  When multiple voxels are involved, the array has dimensions of (N,3).

`coords`:
  A 3D `numpy.ndarray` containing the location of the center of a voxel in 
  real-world coordinates, in units of Angstroms.  When multiple coordinates 
  that involved, the array has dimensions of (N,3).
"""

@dataclass(frozen=True)
class Sphere:
    center_A: NDArray
    radius_A: float

    @cached_property
    def volume_A3(self):
        return 4 / 3 * pi * self.radius_A**3

@dataclass(frozen=True)
class Cube:
    center_A: NDArray
    length_A: float

@dataclass(frozen=True)
class Atom:
    sphere: Sphere
    channel: str
    occupancy: float

@dataclass(frozen=True)
class Grid:
    length_voxels: int
    resolution_A: float
    center_A: NDArray = field(default_factory=lambda: np.zeros(3))

    @cached_property
    def length_A(self):
        return self.length_voxels * self.resolution_A

    @cached_property
    def shape(self):
        return 3 * (self.length_voxels,)

@dataclass
class ImageParams:
    grid: Grid
    channels: list[str]
    element_radii_A: dict[str: float] | float

Image: TypeAlias = NDArray

def image_from_atoms(
        atoms: Atoms,
        img_params: ImageParams,
        channel_cache: Optional[dict]=None,
) -> Image:

    img = _make_empty_image(img_params)
    channel_cache = {} if channel_cache is None else channel_cache

    # Without this filter, `find_voxels_possibly_contacting_sphere()` becomes a
    # performance bottleneck.
    atoms = _discard_atoms_outside_image(atoms, img_params)

    for row in atoms.itertuples(index=False):
        atom = _make_atom(row, img_params, channel_cache)
        _add_atom_to_image(img, img_params.grid, atom)

    return img
        

def _add_atom_to_image(img, grid, atom):
    sphere = atom.sphere
    for voxel in _find_voxels_possibly_contacting_sphere(grid, sphere):
        i = atom.channel, *voxel
        cube = _make_cube(grid, voxel)
        overlap_A3 = _calc_sphere_cube_overlap_volume_A3(sphere, cube)
        img[i] += atom.occupancy * overlap_A3 / sphere.volume_A3

def _calc_sphere_cube_overlap_volume_A3(sphere, cube):
    # Coordinates based on CGNS conventions, but really just copied from the 
    # example provided by the `overlap` library:
    # https://github.com/severinstrobl/overlap
    # https://cgns.github.io/CGNS_docs_current/sids/conv.html#unst_hexa
    x = cube.length_A / 2
    origin_cube = np.array([
        [-x, -x, -x],
        [ x, -x, -x],
        [ x,  x, -x],
        [-x,  x, -x],
        [-x, -x,  x],
        [ x, -x,  x],
        [ x,  x,  x],
        [-x,  x,  x],
    ])

    cube_ = overlap.Hexahedron(origin_cube + cube.center_A)
    sphere_ = overlap.Sphere(sphere.center_A, sphere.radius_A)

    overlap_A3 = overlap.overlap(sphere_, cube_)

    # I got this check from the source code of the `voxelize` package, which 
    # also uses `overlap` to calculate sphere/cube intersection volumes.  The 
    # claim is that, although `overlap` puts an emphasis on numerical 
    # stability, it's still possible to get inaccurate results.  I haven't 
    # experienced these errors myself yet, but I thought it would be prudent to 
    # at least check for impossible values.
    fudge_factor = 1 + 1e-6
    if not (0 <= overlap_A3 <= sphere.volume_A3 * fudge_factor):
        raise RuntimeError(f"numerical instability in overlap: overlap volume ({overlap_A3} Å³) exceeds sphere volume ({sphere.volume_A3} Å³)")

    return overlap_A3
    
def _find_voxels_possibly_contacting_sphere(grid, sphere):
    """
    Return the indices of all voxels that could possibly contact the given 
    sphere.

    The voxels yielded by this function are not guaranteed to actually contact 
    the sphere, so the user is still responsible for checking with a more 
    accurate algorithm.  Voxels outside the borders of the image won't be 
    returned.
    """
    probes = np.array([
        [ 1,  0,  0],
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0, -1,  0],
        [ 0,  0,  1],
        [ 0,  0, -1],
    ])
    probes = sphere.center_A + (probes * sphere.radius_A)
    probe_hits = _find_voxels_containing_coords(grid, probes)

    min_index = np.min(probe_hits, axis=0)
    max_index = np.max(probe_hits, axis=0)

    axes = [
            np.arange(min_index[i], max_index[i] + 1)
            for i in range(3)
    ]
    mesh = np.meshgrid(*axes)
    voxels = np.vstack([x.flat for x in mesh]).T

    return _discard_voxels_outside_image(grid, voxels)

def _find_voxels_containing_coords(grid, coords_A):
    # Consider each grid coordinate to be the center of the cell.
    center_to_coords_A = coords_A - grid.center_A
    origin_to_center_A = grid.resolution_A * (grid.length_voxels - 1) / 2
    origin_to_coords_A = origin_to_center_A + center_to_coords_A

    ijk = origin_to_coords_A / grid.resolution_A
    return np.rint(ijk).astype(int)

def _discard_atoms_outside_image(atoms, img_params):
    grid = img_params.grid
    max_r = _get_max_element_radius(img_params.element_radii_A)

    min_corner = grid.center_A - (grid.length_A / 2 + max_r)
    max_corner = grid.center_A + (grid.length_A / 2 + max_r)

    return atoms[
            (atoms['x'] > min_corner[0]) &
            (atoms['x'] < max_corner[0]) &
            (atoms['y'] > min_corner[1]) &
            (atoms['y'] < max_corner[1]) &
            (atoms['z'] > min_corner[2]) &
            (atoms['z'] < max_corner[2])
    ]

def _discard_voxels_outside_image(grid, voxels):
    not_too_low = voxels.min(axis=1) >= 0
    not_too_high = voxels.max(axis=1) < grid.length_voxels
    return voxels[not_too_low & not_too_high]

def _get_voxel_center_coords(grid, voxels):
    center_offset = (grid.length_voxels - 1) / 2
    return grid.center_A + (voxels - center_offset) * grid.resolution_A

def _make_empty_image(img_params):
    shape = len(img_params.channels), *img_params.grid.shape
    return np.zeros(shape)

def _make_cube(grid, voxel):
    return Cube(_get_voxel_center_coords(grid, voxel), grid.resolution_A)

def _make_atom(row, img_params, channel_cache):
    return Atom(
            sphere=Sphere(
                center_A=np.array([row.x, row.y, row.z]),
                radius_A=_get_element_radius(
                    img_params.element_radii_A,
                    row.element,
                ),
            ),
            channel=_get_element_channel(
                img_params.channels,
                row.element,
                channel_cache,
            ),
            occupancy=row.occupancy,
    )

def _get_element_radius(radii, element):
    if isinstance(radii, Real):
        return radii
    try:
        return radii[element]
    except KeyError:
        return radii['*']

def _get_max_element_radius(radii):
    if isinstance(radii, Real):
        return radii
    else:
        return max(radii.values())

def _get_element_channel(channels, element, cache):
    if element in cache:
        return cache[element]

    for i, channel in enumerate(channels):
        if re.fullmatch(channel, element):
            cache[element] = i
            return i

    raise RuntimeError(f"element {element} didn't match any channels")

