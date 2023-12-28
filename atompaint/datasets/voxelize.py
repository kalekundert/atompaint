import numpy as np
import re

from ._voxelize import (
        Sphere, Atom, Grid,
        _add_atom_to_image, _get_voxel_center_coords,
)
from dataclasses import dataclass
from numbers import Real

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

    # Without this filter, `_find_voxels_possibly_contacting_sphere()` becomes 
    # a performance bottleneck.
    atoms = _discard_atoms_outside_image(atoms, img_params)

    for row in atoms.itertuples(index=False):
        atom = _make_atom(row, img_params, channel_cache)
        _add_atom_to_image(img, img_params.grid, atom)

    return img
        
def get_element_channel(channels, element, cache):
    if element in cache:
        return cache[element]

    for i, channel in enumerate(channels):
        if re.fullmatch(channel, element):
            cache[element] = i
            return i

    raise RuntimeError(f"element {element} didn't match any channels")

def get_max_element_radius(radii):
    if isinstance(radii, Real):
        return radii
    else:
        return max(radii.values())

def get_voxel_center_coords(grid, voxels):
    # There are two things to keep in mind when passing arrays between 
    # python/numpy and C++/Eigen:
    #
    # - Coordinates and voxel indices are represented as row vectors by 
    #   python/numpy, and as column vectors by C++/Eigen.  This means that 
    #   arrays have to be transposed when moving from one language to the 
    #   other.  In principle, it would be possible to use the same row/column 
    #   vector convention in both languages.  But this would make it harder to 
    #   interact with third-party libraries like `overlap`.
    #
    # - Eigen doesn't have 1D arrays.  Instead it has vectors, which are just 
    #   2D matrices with either 1 row or 1 column.  When converting a vector 
    #   from C++/Eigen back to python/numpy, it's not clear whether the 
    #   resulting array should be 1D or 2D.  This ambiguity can be resolved by 
    #   looking at the shape of the original numpy input.
    #
    # I decided against accounting for either of these issues in the binding 
    # code itself.  The main reason for exposing most of the C++ functions to 
    # python is testing, and for that it's not helpful to be changing the 
    # inputs and outputs.  But this specific function is useful in other 
    # contexts, so I wrote this wrapper function to enforce the python 
    # conventions.

    coords_A = _get_voxel_center_coords(grid, voxels.T).T
    return coords_A.reshape(voxels.shape)


def _make_empty_image(img_params):
    shape = len(img_params.channels), *img_params.grid.shape
    return np.zeros(shape, dtype=np.float32)

def _discard_atoms_outside_image(atoms, img_params):
    grid = img_params.grid
    max_r = get_max_element_radius(img_params.element_radii_A)

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
            channel=get_element_channel(
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

