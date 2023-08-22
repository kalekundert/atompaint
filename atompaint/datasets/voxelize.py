import numpy as np
import pandera as pa
import overlap
import numba
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
        return _calc_sphere_volume_A3_jit(self.radius_A)

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
        

def _make_empty_image(img_params):
    shape = len(img_params.channels), *img_params.grid.shape
    return np.zeros(shape)

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

# The following functions are performance-critical:

jit = numba.njit(cache=True)

def _add_atom_to_image(img, grid, atom):
    sphere = atom.sphere

    for voxel in _find_voxels_possibly_contacting_sphere_jit(
            grid.length_voxels,
            grid.resolution_A,
            grid.center_A,
            sphere.center_A,
            sphere.radius_A,
    ):
        verts = _get_voxel_verts_jit(
                grid.length_voxels,
                grid.resolution_A,
                grid.center_A,
                voxel,
        )
        overlap_A3 = _calc_sphere_cube_overlap_volume_A3(sphere, verts)

        i = atom.channel, *voxel
        img[i] += _calc_fraction_atom_in_voxel_jit(
                overlap_A3,
                sphere.radius_A,
                atom.occupancy,
        )

@jit
def _find_voxels_possibly_contacting_sphere_jit(
        grid_length_voxels,
        grid_resolution_A,
        grid_center_A,
        sphere_center_A,
        sphere_radius_A,
):
    """
    Return the indices of all voxels that could possibly contact the given 
    sphere.

    The voxels yielded by this function are not guaranteed to actually contact 
    the sphere, so the user is still responsible for checking with a more 
    accurate algorithm.  Voxels outside the borders of the image won't be 
    returned.
    """
    r = sphere_radius_A
    probe_rel_coords_A = np.array([
        [ r,  0,  0],
        [-r,  0,  0],
        [ 0,  r,  0],
        [ 0, -r,  0],
        [ 0,  0,  r],
        [ 0,  0, -r],
    ])
    probe_coords_A = sphere_center_A + probe_rel_coords_A
    probe_voxels = _find_voxels_containing_coords_jit(
        grid_length_voxels,
        grid_resolution_A,
        grid_center_A,
        probe_coords_A,
    )

    min_index = _min_jit(probe_voxels, 0)
    max_index = _max_jit(probe_voxels, 0)

    mesh = _meshgrid_jit(
            np.arange(min_index[0], max_index[0] + 1),
            np.arange(min_index[1], max_index[1] + 1),
            np.arange(min_index[2], max_index[2] + 1),
    )
    layers = mesh[0].flatten(), mesh[1].flatten(), mesh[2].flatten()
    voxels = np.vstack(layers).T

    return _discard_voxels_outside_image_jit(grid_length_voxels, voxels)

@jit
def _find_voxels_containing_coords_jit(
        grid_length_voxels,
        grid_resolution_A,
        grid_center_A,
        coords_A,
):
    center_to_coords_A = coords_A - grid_center_A
    origin_to_center_A = grid_resolution_A * (grid_length_voxels - 1) / 2
    origin_to_coords_A = origin_to_center_A + center_to_coords_A

    ijk = origin_to_coords_A / grid_resolution_A
    return np.rint(ijk).astype(np.int8)

@jit
def _discard_voxels_outside_image_jit(
        grid_length_voxels,
        voxels,
):
    not_too_low = _min_jit(voxels, 1) >= 0
    not_too_high = _max_jit(voxels, 1) < grid_length_voxels
    return voxels[not_too_low & not_too_high]

@jit
def _get_voxel_verts_jit(
        grid_length_voxels,
        grid_resolution_A,
        grid_center_A,
        voxel,
):
    center_A = _get_voxel_center_coords_jit(
        grid_length_voxels,
        grid_resolution_A,
        grid_center_A,
        voxel,
    )
    return _get_cube_verts_jit(center_A, grid_resolution_A)

def _get_voxel_center_coords(grid, voxels):
    return _get_voxel_center_coords_jit(
            grid.length_voxels,
            grid.resolution_A,
            grid.center_A,
            voxels,
    )

@jit
def _get_voxel_center_coords_jit(
        grid_length_voxels,
        grid_resolution_A,
        grid_center_A,
        voxels,
):
    center_offset = (grid_length_voxels - 1) / 2
    return grid_center_A + (voxels - center_offset) * grid_resolution_A

@jit
def _get_cube_verts_jit(cube_center_A, cube_length_A):
    # Coordinates based on CGNS conventions, but really just copied from the 
    # example provided by the `overlap` library:
    # https://github.com/severinstrobl/overlap
    # https://cgns.github.io/CGNS_docs_current/sids/conv.html#unst_hexa
    x = cube_length_A / 2
    origin_verts = np.array([
        [-x, -x, -x],
        [ x, -x, -x],
        [ x,  x, -x],
        [-x,  x, -x],
        [-x, -x,  x],
        [ x, -x,  x],
        [ x,  x,  x],
        [-x,  x,  x],
    ])
    return origin_verts + cube_center_A
    
def _calc_sphere_cube_overlap_volume_A3(sphere, cube_verts):
    cube = overlap.Hexahedron(cube_verts)
    sphere_ = overlap.Sphere(sphere.center_A, sphere.radius_A)
    overlap_A3 = overlap.overlap(sphere_, cube)

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

@jit
def _calc_fraction_atom_in_voxel_jit(overlap_A3, radius_A, occupancy):
    return occupancy * overlap_A3 / _calc_sphere_volume_A3_jit(radius_A)

@jit
def _calc_sphere_volume_A3_jit(radius_A):
    return 4/3 * pi * radius_A**3

# Reimplemented versions of numpy functions that numba doesn't support.

@jit
def _apply_along_axis_jit(func1d, axis, arr):
  # https://github.com/numba/numba/issues/1269
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1], dtype=arr.dtype)
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0], dtype=arr.dtype)
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result

@jit
def _min_jit(array, axis):
  return _apply_along_axis_jit(np.min, axis, array)

@jit
def _max_jit(array, axis):
  return _apply_along_axis_jit(np.max, axis, array)

@jit
def _meshgrid_jit(x, y, z):
    # https://stackoverflow.com/questions/70613681/numba-compatible-numpy-meshgrid
    xx = np.empty(shape=(z.size, y.size, x.size), dtype=x.dtype)
    yy = np.empty(shape=(z.size, y.size, x.size), dtype=y.dtype)
    zz = np.empty(shape=(z.size, y.size, x.size), dtype=z.dtype)
    for i in range(z.size):
        for j in range(y.size):
            for k in range(x.size):
                xx[i,j,k] = x[k]
                yy[i,j,k] = y[j]
                zz[i,j,k] = z[i]
    return xx, yy, zz

