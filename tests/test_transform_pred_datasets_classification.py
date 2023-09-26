import atompaint.transform_pred.datasets.classification as ap
import numpy as np
import parametrize_from_file as pff

from atompaint.transform_pred.datasets.origins import (
        select_origin_filtering_atoms,
)
from atompaint.datasets.coords import (
        make_coord_frame, transform_coords, get_origin,
)
from scipy.spatial.transform import Rotation
from pytest import approx
from pytest_unordered import unordered
from math import pi
from dataclasses import dataclass
from utils import *

from test_datasets_coords import coords, frame
from test_datasets_atoms import atoms
from test_transform_pred_datasets_origins import origin_params

np.set_printoptions(suppress=True)

@dataclass
class MockWorkerInfo:
    id: int
    num_workers: int

def frames_from_origins(origins):
    return np.array([
        make_coord_frame(x, np.zeros(3))
        for x in origins
    ])

def worker_info(params):
    return MockWorkerInfo(
            id=int(params['id']),
            num_workers=int(params['num_workers']),
    )

def check_origins(frames_ab, origins_a):

    # Check that the slot origins are in the given order, so that future tests 
    # can make assertions using the slot indices.  There's no way to work out 
    # what the indices "should be" a priori, so the order is obtained by 
    # running the code and seeing what it is.  This function then makes sure 
    # that this order doesn't change unexpectedly.

    np.testing.assert_allclose(
            [get_origin(x) for x in frames_ab],
            origins_a,
            atol=1e-7,
    )


def test_cube_face_frames_ab():
    check_origins(
            ap.make_cube_face_frames_ab(2, 1), [
                [ 0, -3,  0],
                [ 0,  3,  0],
                [-3,  0,  0],
                [ 3,  0,  0],
                [ 0,  0,  3],
                [ 0,  0, -3],
            ]
    )

@pff.parametrize(
        schema=pff.cast(
            frame_ia=frame,
            frame_b_origins_a=coords,
            origin_params=origin_params,
            atoms_i=atoms,
            expected=integers,
        )
)
def test_filter_views(frame_ia, frame_b_origins_a, origin_params, atoms_i, expected):
    frames_ab = frames_from_origins(frame_b_origins_a)
    relevant_atoms_i = select_origin_filtering_atoms(atoms_i)
    hits = ap._filter_views(frame_ia, frames_ab, origin_params, relevant_atoms_i)
    assert hits == unordered(expected)

@pff.parametrize(
        schema=[
            pff.cast(
                seed_offset=int,
                epoch_size=int,
                worker_info=worker_info,
                expected=integers,
            ),
            pff.defaults(
                worker_info=None,
            ),
        ],
)
def test_get_seeds(seed_offset, epoch_size, worker_info, expected):
    seeds = ap._get_seeds(seed_offset, epoch_size, worker_info)
    assert list(seeds) == expected
