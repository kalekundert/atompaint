import numpy as np
import pandas as pd
import pandera as pa
import nestedtext as nt

from atompaint.datasets.atoms import (
        get_atom_coords, transform_atom_coords, atoms_from_tag,
)
from atompaint.datasets.coords import make_coord_frame, invert_coord_frame
from atompaint.datasets.voxelize import _get_max_element_radius
from scipy.spatial import KDTree
from torch.utils.data import IterableDataset
from dataclasses import dataclass
from functools import cached_property, partial
from multiprocessing import Pool
from math import pi, sqrt

from pandera.typing import DataFrame, Series
from numpy.typing import NDArray
from typing import TypeAlias

"""\
Consider each atom a possible origin, and weight those origins by the number of
neighboring atoms within some radius.

This is a relatively simple way to pick training examples.  The purpose of the 
weighting is to discourage training on surface regions, since we don't want the 
network to get in the habit of guessing based on the orientation of the 
surface.

In the future, I'd also like to weight based on more factors, e.g.:

- The specific atoms in each view, to avoid redundancy.
- The quality of the structure.
"""

class NeighborCountDataStream(IterableDataset):

    def __init__(
            self,
            rng,
            *,
            origins,
            input_from_atoms,
            view_pair_params,
            batch_size,
    ):
        self.rng = rng
        self.origins = origins
        self.input_from_atoms = input_from_atoms
        self.view_pair_params = view_pair_params
        self.batch_size = batch_size

    def __iter__(self):
        inputs, outputs = unzip(
                self.sample()
                for i in range(self.batch_size)
        )
        inputs = np.stack(inputs)
        outputs = np.stack(outputs)

        return torch.from_numpy(inputs), torch.from_numpy(outputs)

    def sample(self):
        origin_a, tag = sample_origin(self.origins)
        origins_b = filter_by_tag(self.origins, tag)
        atoms = atoms_from_tag(tag)

        view_pair = sample_view_pair(
                self.rng,
                atoms,
                origin_a,
                origins_b,
                self.view_pair_params,
        )
        input_a = self.input_from_atoms(view_pair.atoms_a)
        input_b = self.input_from_atoms(view_pair.atoms_b)

        inputs = np.stack([input_a, input_b])
        return inputs, view_pair.frame_ba

class NeighborCountDataStreamForCnn(NeighborCountDataStream):

    def __init__(
            self,
            rng,
            *,
            atom_db,
            origin_db,
            batch_size,
            img_params,
            max_dist,
    ):
        view_pair_params = ViewPairParams(
                min_dist=calc_min_distance_between_origins(img_params),
                max_dist=max_dist,
        )
        input_from_atoms = partial(image_from_atoms, img_params=img_params)

        super().__init__(
                rng=rng,
                atom_db=atom_db,
                origin_db=origin_db,
                input_from_atoms=input_from_atoms,
                view_pair_params=view_pair_params,
                batch_size=batch_size,
        )



@dataclass
class ViewPair:
    atoms_i: pd.DataFrame
    frame_ia: NDArray
    frame_ib: NDArray

    @cached_property
    def atoms_a(self):
        return transform_atom_coords(self.atoms_i, self.frame_ia)

    @cached_property
    def atoms_b(self):
        return transform_atom_coords(self.atoms_i, self.frame_ib)

    @cached_property
    def frame_ai(self):
        return invert_coord_frame(self.frame_ia)

    @cached_property
    def frame_bi(self):
        return invert_coord_frame(self.frame_ib)

    @cached_property
    def frame_ab(self):
        # Keep in mind that the coordinates transformed by this matrix will be 
        # multiplied by `frame_ai` first, then that product will be multiplied 
        # by `frame_ib`.  So the order is right, even if it seems backwards at 
        # first glance.
        return self.frame_ib @ self.frame_ai

@dataclass
class ViewPairParams:
    min_dist_A: float
    max_dist_A: float

@dataclass
class OriginParams:
    radius_A: float
    min_neighbors: int

class OriginsSchema(pa.DataFrameModel):
    tag: Series[str]
    x: Series[float]
    y: Series[float]
    z: Series[float]
    weight: Series[float] = pa.Field(coerce=True)

Origins: TypeAlias = DataFrame[OriginsSchema]

# Pre-calculate origins

def choose_origins(
        tags,
        origin_params,
        *,
        progress_bar=lambda x: x,
        meta=None,
):
    worker = partial(_choose_origins_worker, origin_params)

    with Pool() as pool:
        results = list(progress_bar(pool.imap(worker, tags)))

    if meta is not None:
        meta['tags_skipped'] = [x[1] for x in results if x[0] == 'skip']
        meta['tags_loaded'] = [x[1] for x in results if x[0] == 'load']

    dfs = [x[2] for x in results if x[0] == 'load']
    return pd.concat(dfs, ignore_index=True)

def _choose_origins_worker(origin_params, tag):
    try:
        atoms = atoms_from_tag(tag)
    except FileNotFoundError:
        return 'skip', tag

    df = choose_origins_for_atoms(tag, atoms, origin_params)
    return 'load', tag, df

def choose_origins_for_atoms(tag, atoms, origin_params):
    df = atoms[['x', 'y', 'z']].copy()
    df['tag'] = tag
    df['weight'] = n = count_neighbors(atoms, origin_params.radius_A)
    return df[n >= origin_params.min_neighbors]

def load_origins(path):
    return pd.read_parquet(path / 'origins.parquet')

def save_origins(path, df, params):
    path.mkdir()
    nt.dump(params, path / 'params.nt')
    df.to_parquet(path / 'origins.parquet')

def count_neighbors(atoms, radius_A):
    xyz = get_atom_coords(atoms)
    kd_tree = KDTree(xyz)
    weights = np.zeros(kd_tree.n)

    # This will count an atom as being a "neighbor" to other conformers of 
    # itself, which isn't strictly correct, but I think the error will be small 
    # enough (and hard enough to fix) that I'm willing to just overlook it.

    for i,j in kd_tree.query_pairs(radius_A):
        weights[i] += atoms.iloc[j]['occupancy']
        weights[j] += atoms.iloc[i]['occupancy']

    return pd.Series(weights, index=atoms.index)

# Make view pairs

def sample_view_pair(rng, atoms, origin_a, origin_b_candidates, view_pair_params):
    # This is a bit of a janky function.  It only exists because there are a 
    # couple of ways to arrive at the first origin, but all the steps after 
    # that are the same.
    origin_b_candidates = filter_by_distance(
            origin_b_candidates,
            origin_a,
            min_dist_A=view_pair_params.min_dist_A,
            max_dist_A=view_pair_params.max_dist_A,
    )
    origin_b, _ = sample_origin(rng, origin_b_candidates)

    frame_a = sample_coord_frame(rng, origin_a)
    frame_b = sample_coord_frame(rng, origin_b)

    return ViewPair(atoms, frame_a, frame_b)

@pa.check_types
def sample_origin(rng, origins: Origins):
    i = sample_weighted_index(rng, origins['weight'])
    return get_origin_coord(origins, i), origins['tag'].loc[i]

def sample_coord_frame(rng, origin):
    """
    Return a matrix that will perform a uniformly random rotation, then move 
    the given point to the origin of the new frame.

    This function uses the torch RNG.
    """
    u = sample_uniform_unit_vector(rng)
    th = rng.uniform(0, 2 * pi)
    return make_coord_frame(origin, u * th)

def filter_by_distance(origins, coord, *, min_dist_A, max_dist_A):
    origin_coords = get_origin_coords(origins)
    dist = np.linalg.norm(origin_coords - coord, axis=1)
    return origins[(min_dist_A <= dist) & (dist <= max_dist_A)]

def filter_by_tag(origins, tag):
    mask = origins['tag'] == tag
    return origins[mask]

def calc_min_distance_between_origins(img_params):
    # Calculate the radius of the sphere that inscribes in grid.  This ensures 
    # that the grids won't overlap, no matter how they're rotated.
    grid_radius_A = sqrt(3) * img_params.grid.length_A / 2

    # Add the radius of the largest possible atom, so that no atom can possibly 
    # appear in both views.  This degree of overlap probably wouldn't matter 
    # anyways, but might as well be as correct as possible.
    atom_radius_A = _get_max_element_radius(img_params.element_radii_A)

    return 2 * (grid_radius_A + atom_radius_A)

def sample_uniform_unit_vector(rng):
    # https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe

    # I chose the rejection sampling approach rather than the Gaussian approach 
    # because (i) I'd need the while loop either way to check for a null vector 
    # and (ii) it makes more sense to me.  The Gaussian approach would be ≈2x 
    # faster, though.

    while True:
        v = rng.uniform(-1, 1, size=3)
        m = np.linalg.norm(v)
        if 0 < m < 1:
            return v / m

def sample_weighted_index(rng, weights: pd.Series):
    return rng.choice(weights.index, p=weights / weights.sum())

def get_origin_coord(origins, i):
    # Important to select columns before `loc`: This ensures that the resulting 
    # array is of dtype float rather than object, because all of the selected 
    # rows are float.
    return origins[['x', 'y', 'z']].loc[i].values

def get_origin_coords(origins):
    return origins[['x', 'y', 'z']].values
