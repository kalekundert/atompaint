import numpy as np
import pandas as pd

from .origins import Origins, get_origin_coord, get_origin_tag
from atompaint.datasets.coords import Coord, make_coord_frame
from math import pi

def sample_origin(rng: np.random.Generator, origins: Origins):
    if origins.empty:
        raise NoOriginsToSample()

    i = _sample_uniform_iloc(rng, origins)
    return get_origin_coord(origins, i), get_origin_tag(origins, i)

def sample_coord_frame(rng: np.random.Generator, origin: Coord):
    """
    Return a matrix that will perform a uniformly random rotation, then move 
    the given point to the origin of the new frame.
    """
    u = _sample_uniform_unit_vector(rng)
    th = rng.uniform(0, 2 * pi)
    return make_coord_frame(origin, u * th)


def _sample_uniform_unit_vector(rng: np.random.Generator):
    # https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe

    # I chose the rejection sampling approach rather than the Gaussian approach 
    # because (i) I'd need the while loop either way to check for a null vector 
    # and (ii) I understand why it works.  The Gaussian approach would be ≈2x 
    # faster, though.

    while True:
        v = rng.uniform(-1, 1, size=3)
        m = np.linalg.norm(v)
        if 0 < m < 1:
            return v / m

def _sample_uniform_iloc(rng: np.random.Generator, df: pd.DataFrame):
    return rng.integers(0, df.shape[0])

class NoOriginsToSample(Exception):
    pass

