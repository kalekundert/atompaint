import atompaint.transform_pred.datasets.neighbor_count as apd
import numpy as np
import pandas as pd
import pytest
import parametrize_from_file as pff

from test_datasets_atoms import atoms
from test_datasets_coords import coord, frame, matrix
from scipy.stats import multinomial, ks_1samp
from scipy.spatial.distance import mahalanobis
from scipy.spatial.transform import Rotation
from atompaint.datasets.atoms import transform_atom_coords
from atompaint.datasets.coords import transform_coords
from atompaint.datasets.voxelize import ImageParams, Grid
from itertools import combinations
from collections import Counter
from pytest import approx
from math import sqrt
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
    return apd.OriginParams(
            radius_A=float(params['radius_A']),
            min_nearby_atoms=int(params['min_nearby_atoms']),
    )


def test_origin_coord():
    rows = [
            dict(tag='a', x=1, y=2, z=3),
            dict(tag='b', x=5, y=6, z=7),
    ]
    origins = pd.DataFrame(rows, index=[0, 2])

    # Note that `strict=True` checks that the data types match.  This is 
    # important, because it's easy to get the 'object' data type (instead of 
    # 'float') by selecting columns/rows from the data frame in the wrong 
    # order, and having the wrong data type can mess up downstream 
    # calculations.

    # The `strict=True` argument requires numpy>=1.24.0, but scipy and prody 
    # (for some reason) require earlier versions.  I'm sure this will be sorted 
    # out eventually.

    np.testing.assert_array_equal(
            apd.get_origin_coord(origins, 0),
            np.array([1, 2, 3]),
            #strict=True,
    )

    # Origins are often filtered, so the index numbers may be greater than the 
    # actual number of elements in the data frame.  Make sure this case works.
    np.testing.assert_array_equal(
            apd.get_origin_coord(origins, 1),
            np.array([5, 6, 7]),
            #strict=True,
    )

def test_sample_uniform_unit_vector():
    # The following references give the distribution for the distance between 
    # two random points on a unit sphere of any dimension:
    #
    # https://math.stackexchange.com/questions/4654438/distribution-of-distances-between-random-points-on-spheres
    # https://johncarlosbaez.wordpress.com/2018/07/10/random-points-on-a-sphere-part-1/
    #
    # For the 3D case, the PDF is remarkably simple:
    #
    #   p(d) = d/2
    #
    # Here, we use the 1-sample KS test to check that our sampled distribution 
    # is consistent with this expected theoretical distribution.
    
    n = 1000
    rng = np.random.default_rng(0)

    d = np.zeros(n)
    x = np.array([1, 0, 0])  # arbitrary reference point

    for i in range(n):
        y = apd._sample_uniform_unit_vector(rng)
        d[i] = np.linalg.norm(y - x)

    cdf = lambda d: d**2 / 4
    ks = ks_1samp(d, cdf)

    # This test should fail for 5% of random seeds, but 0 is one that passes.
    assert ks.pvalue > 0.05

def test_sample_coord_frame():
    # Don't test that the sampling is actually uniform; I think this would be 
    # hard to show, and the two underlying sampling functions are both 
    # well-tested already.  Instead, just make sure that the resulting 
    # coordinate frame doesn't distort distances.

    def calc_pairwise_distances(x):
        return np.array([
            np.linalg.norm(x[i] - x[j])
            for i, j in combinations(range(len(x)), 2)
        ])

    rng = np.random.default_rng(0)
    x = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ])
    expected_dists = calc_pairwise_distances(x)

    for i in range(1000):
        origin = rng.uniform(-10, 10, 3)
        frame_xy = apd.sample_coord_frame(rng, origin)
        y = transform_coords(x, frame_xy)
        actual_dists = calc_pairwise_distances(y)

        assert actual_dists == approx(expected_dists)

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
    dist_A = apd.calc_min_distance_between_origins(img_params)

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
    actual = apd.filter_by_distance(
            origins,
            center,
            min_dist_A=min_dist_A,
            max_dist_A=max_dist_A,
    )
    print(actual.to_string())
    print(expected.to_string())
    assert actual.to_dict('records') == expected.to_dict('records')

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
    view_pair = apd.ViewPair(atoms_i, frame_ia, frame_ib)
    pd.testing.assert_frame_equal(view_pair.atoms_a, atoms_a)
    pd.testing.assert_frame_equal(view_pair.atoms_b, atoms_b)

    atoms_ai = transform_atom_coords(atoms_a, view_pair.frame_ai)
    pd.testing.assert_frame_equal(atoms_ai, atoms_i)

    atoms_bi = transform_atom_coords(atoms_b, view_pair.frame_bi)
    pd.testing.assert_frame_equal(atoms_bi, atoms_i)

    atoms_ab = transform_atom_coords(atoms_a, view_pair.frame_ab)
    pd.testing.assert_frame_equal(atoms_ab, atoms_b)


@pff.parametrize(
        schema=pff.cast(atoms=atoms, radius_A=float, expected=matrix),
)
def test_count_nearby_atoms(atoms, radius_A, expected):
    counts = apd._count_nearby_atoms(atoms, radius_A)
    np.testing.assert_array_equal(counts.values, expected)

@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            origin_params=origin_params,
            expected=origins,
        ),
)
def test_choose_origins_for_atoms(tag, atoms, origin_params, expected):
    origins = apd.choose_origins_for_atoms(tag, atoms, origin_params)
    # Filtering can change the indices of the data frame, so we need to compare 
    # in a way that ignores the index.
    assert origins.to_dict('records') == expected.to_dict('records')

