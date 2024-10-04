import parametrize_from_file as pff
import macromol_dataframe as mmdf
import numpy as np

from escnn.group import octa_group
from pathlib import Path
from io import StringIO

TEST_DIR = Path(__file__).parent
IMAGE_DIR = TEST_DIR / 'images'

with_py = pff.Namespace()
with_math = pff.Namespace('from math import *')
with_ap = pff.Namespace('import atompaint as ap')

def get_exact_rotations(group):
    """
    Return all the rotations that (i) all belong to the given group and (ii) do 
    not require interpolation.  These rotations are good for checking 
    equivariance, because interpolation can be a significant source of error.
    """
    octa = octa_group()
    exact_rots = []

    for octa_element in octa.elements:
        value = octa_element.value
        param = octa_element.param

        try:
            exact_rot = group.element(value, param)
        except ValueError:
            continue

        exact_rots.append(exact_rot)

    assert len(exact_rots) > 1
    return exact_rots

def matrix(params):
    io = StringIO(params)
    return np.loadtxt(io, dtype=float)

def vector(params):
    return np.array([with_math.eval(x) for x in params.split()])

def frame(params):
    origin = coord(params['origin'])
    rot_vec_rad = vector(params['rot_vec_rad'])
    return mmdf.make_coord_frame_from_rotation_vector(origin, rot_vec_rad)

def frames(params):
    return [frame(x) for x in params]

def coord(params):
    return matrix(params)

def coords(params):
    coords = matrix(params)
    coords.shape = (1, *coords.shape)[-2:]
    return coords

def integers(params):
    return [int(x) for x in params.split()]

