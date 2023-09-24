import atompaint.transform_pred.datasets.origins as ap
import numpy as np
import pandas as pd
import parametrize_from_file as pff

from test_datasets_atoms import atoms
from test_datasets_coords import coord, coords, frame, matrix
from atompaint.datasets.atoms import get_atom_coords
from io import StringIO

def origins(params):
    io = StringIO(params)

    head = io.readline()
    n_cols = len(head.split())
    if n_cols != 4:
        raise IOError(f"expected origins data frame to have 4 columns, not {n_cols}\nfirst row: {head!r}")
    io.seek(0)

    return pd.read_fwf(
            io,
            sep=' ',
            names=['tag', 'x', 'y', 'z'],
            dtype={
                'tag': 'category',
                'x': float,
                'y': float,
                'z': float,
            },
    )

def origin_params(params):
    return ap.OriginParams(
            radius_A=float(params['radius_A']),
            min_nearby_atoms=int(params['min_nearby_atoms']),
    )


def test_get_origin_tags_coords():
    rows = [
            dict(tag='a', x=1, y=2, z=3),
            dict(tag='b', x=5, y=6, z=7),
    ]
    origins = pd.DataFrame(rows, index=[0, 2])

    # Note that `strict=True` checks that the data types match.  This is 
    # important, because it's easy to get the 'object' data type (instead of 
    # 'float') by selecting columns/rows from the data frame in the wrong 
    # order, and having the wrong data type can mess up downstream 
    # calculations.  Note that the `strict=True` argument requires 
    # numpy>=1.24.0.

    np.testing.assert_array_equal(
            ap.get_origin_coord(origins, 0),
            np.array([1, 2, 3]),
            strict=True,
    )
    assert ap.get_origin_tag(origins, 0) == 'a'

    # Origins are often filtered, so the index numbers may be greater than the 
    # actual number of elements in the data frame.  Make sure this case works.
    np.testing.assert_array_equal(
            ap.get_origin_coord(origins, 1),
            np.array([5, 6, 7]),
            strict=True,
    )
    assert ap.get_origin_tag(origins, 1) == 'b'

    np.testing.assert_array_equal(
            ap.get_origin_coords(origins),
            np.array([[1, 2, 3], [5, 6, 7]]),
            strict=True,
    )

@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            origin_params=origin_params,
            expected=origins,
        ),
)
def test_choose_origins_for_atoms(tag, atoms, origin_params, expected):
    origins = ap.choose_origins_for_atoms(tag, atoms, origin_params)
    # Filtering can change the indices of the data frame, so we need to compare 
    # in a way that ignores the index.
    assert origins.to_dict('records') == expected.to_dict('records')

@pff.parametrize(
        schema=pff.cast(
            coords_A=coords,
            origin_params=origin_params,
            atoms=atoms,
            expected=coords,
        ),
)
def test_filter_origin_coords(coords_A, origin_params, atoms, expected):
    filtering_atoms = ap.select_origin_filtering_atoms(atoms)
    origin_coords_A = ap.filter_origin_coords(
            coords_A,
            origin_params,
            filtering_atoms,
    )
    np.testing.assert_array_equal(origin_coords_A, expected)

@pff.parametrize(
        schema=[
            pff.cast(
                coords_A=coords,
                atoms=atoms,
                radius_A=float,
                expected=matrix,
            ),
            pff.defaults(
                coords_A=None,
            ),
        ],
)
def test_count_nearby_atoms(coords_A, atoms, radius_A, expected):
    if coords_A is None:
        coords_A = get_atom_coords(atoms)
    if len(coords_A) == 1:
        coords_A = coords_A.ravel()

    filtering_atoms = ap.select_origin_filtering_atoms(atoms)
    counts = ap._count_nearby_atoms(coords_A, filtering_atoms, radius_A)
    np.testing.assert_array_equal(counts, expected)
