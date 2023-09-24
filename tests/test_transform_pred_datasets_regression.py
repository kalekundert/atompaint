import atompaint.transform_pred.datasets.regression as ap
import numpy as np
import pandas as pd
import parametrize_from_file as pff

from test_datasets_atoms import atoms
from test_datasets_coords import coord, frame
from test_transform_pred_datasets_origins import origins
from atompaint.datasets.atoms import transform_atom_coords
from atompaint.datasets.voxelize import ImageParams, Grid

@pff.parametrize(
        schema=pff.cast(
            atoms_i=atoms,
            frame_ia=frame,
            frame_ib=frame,
            atoms_a=atoms,
            atoms_b=atoms,
        ),
)
def test_view_pair(atoms_i, frame_ia, frame_ib, atoms_a, atoms_b):
    view_pair = ap.ViewPair(atoms_i, frame_ia, frame_ib)
    pd.testing.assert_frame_equal(view_pair.atoms_a, atoms_a)
    pd.testing.assert_frame_equal(view_pair.atoms_b, atoms_b)

    atoms_ai = transform_atom_coords(atoms_a, view_pair.frame_ai)
    pd.testing.assert_frame_equal(atoms_ai, atoms_i)

    atoms_bi = transform_atom_coords(atoms_b, view_pair.frame_bi)
    pd.testing.assert_frame_equal(atoms_bi, atoms_i)

    atoms_ab = transform_atom_coords(atoms_a, view_pair.frame_ab)
    pd.testing.assert_frame_equal(atoms_ab, atoms_b)

@pff.parametrize(
        schema=pff.cast(
            origins=origins,
            center=coord,
            min_dist_A=float,
            max_dist_A=float,
            expected=origins,
        ),
)
def test_filter_by_distance(origins, center, min_dist_A, max_dist_A, expected):
    actual = ap.filter_by_distance(
            origins,
            center,
            min_dist_A=min_dist_A,
            max_dist_A=max_dist_A,
    )
    print(actual.to_string())
    print(expected.to_string())
    assert actual.to_dict('records') == expected.to_dict('records')

def test_calc_min_distance():
    grid = Grid(
            length_voxels=10,
            resolution_A=1,
            center_A=np.zeros(3),
    )
    img_params = ImageParams(
            grid=grid,
            channels=[],
            element_radii_A=1,
    )
    dist_A = ap.calc_min_distance_between_origins(img_params)

    # The goal is to test that two grids separated by the calculated distance 
    # can't overlap.  I know that the overlap will be closest in the corners, 
    # when the grids have the same alignment, so that's the only scenario I 
    # test here.  I don't account for the atom radius, though.

    direction = np.ones(3)
    direction /= np.linalg.norm(direction)
    offset = direction * dist_A
    corner = np.ones(3) * grid.length_A / 2

    assert np.all(corner < offset - corner)
    assert np.all(corner + corner < offset)

