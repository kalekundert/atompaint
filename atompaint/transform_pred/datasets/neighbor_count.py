import torch
import numpy as np
import pandas as pd
import pandera as pa
import json

from atompaint.datasets.atoms import (
        get_atom_coords, transform_atom_coords, atoms_from_tag,
        filter_nonbiological_atoms,
)
from atompaint.datasets.coords import make_coord_frame, invert_coord_frame
from atompaint.datasets.voxelize import image_from_atoms, _get_max_element_radius
from scipy.spatial import KDTree
from torch.utils.data import Dataset
from dataclasses import dataclass, asdict
from functools import cached_property, partial
from pathlib import Path
from shutil import rmtree
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

class NeighborCountDataset(Dataset):

    def __init__(
            self,
            *,
            origins,
            input_from_atoms,
            view_pair_params,
            epoch_size,
    ):
        self.origins = origins
        self.input_from_atoms = input_from_atoms
        self.view_pair_params = view_pair_params
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, i):
        rng = np.random.default_rng(i)

        while True:
            origin_a, tag = sample_origin(rng, self.origins)
            origins_b = filter_by_tag(self.origins, tag)
            atoms = atoms_from_tag(tag)

            try:
                view_pair = sample_view_pair(
                        rng,
                        atoms,
                        origin_a,
                        origins_b,
                        self.view_pair_params,
                )
            except NoOriginsToSample:
                continue

            input_a = self.input_from_atoms(view_pair.atoms_a)
            input_b = self.input_from_atoms(view_pair.atoms_b)
            input_ab = np.stack([input_a, input_b])

            return (
                    torch.from_numpy(input_ab).float(),
                    torch.from_numpy(view_pair.frame_ab).float(),
            )

class NeighborCountDatasetForCnn(NeighborCountDataset):

    def __init__(
            self,
            *,
            origins,
            img_params,
            max_dist_A,
            epoch_size,
    ):
        min_dist_A = calc_min_distance_between_origins(img_params)
        view_pair_params = ViewPairParams(
                min_dist_A=min_dist_A,
                max_dist_A=min_dist_A + max_dist_A,
        )
        input_from_atoms = partial(image_from_atoms, img_params=img_params)

        super().__init__(
                origins=origins,
                input_from_atoms=input_from_atoms,
                view_pair_params=view_pair_params,
                epoch_size=epoch_size,
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
    min_nearby_atoms: int

class OriginsSchema(pa.DataFrameModel):
    tag: Series[str]
    x: Series[float]
    y: Series[float]
    z: Series[float]
    weight: Series[float] = pa.Field(coerce=True)

class NoOriginsToSample(Exception):
    pass

Origins: TypeAlias = DataFrame[OriginsSchema]

# Pre-calculate origins

def choose_origins_for_tags(tags, origin_params):
    dfs = []
    status = {
            'tags_skipped': [],
            'tags_loaded': [],
    }

    for tag in tags:
        try:
            atoms = atoms_from_tag(tag)
        except FileNotFoundError:
            status['tags_skipped'].append(tag)
            continue

        df = choose_origins_for_atoms(tag, atoms, origin_params)
        dfs.append(df)

        status['tags_loaded'].append(tag)

    return pd.concat(dfs, ignore_index=True), status

def choose_origins_for_atoms(tag, atoms, origin_params):
    atoms = filter_nonbiological_atoms(atoms)

    # Count atoms after filtering out the nonbiological ones, so that we only 
    # get origins centered on and mostly surrounded by biological atoms.
    n = _count_nearby_atoms(atoms, origin_params.radius_A)

    df = atoms[['x', 'y', 'z']].copy()
    df['tag'] = tag
    df['weight'] = 1
    return df[n >= origin_params.min_nearby_atoms]

def load_origins(path: Path):
    dfs = [
            pd.read_parquet(p)
            for p in sorted(path.glob('origins.parquet*'))
    ]
    return pd.concat(dfs, ignore_index=True)

def load_origin_params(path: Path):
    with open(path / 'params.json') as f:
        params = json.load(f)

    tags = params.pop('tags')
    origin_params = OriginParams(**params)

    return tags, origin_params

def save_origins(path: Path, df, status, suffix=None):
    if suffix:
        worker_id, num_workers = suffix
        suffix = f'.{worker_id:0{len(str(num_workers - 1))}}'
    else:
        suffix = ''

    df.to_parquet(path / f'origins.parquet{suffix}')
    
    with open(path / f'status.json{suffix}', 'w') as f:
        json.dump(status, f)

def save_origin_params(path: Path, tags, origin_params, force=False):
    if path.exists():
        if force or not any(path.glob('origins.parquet*')):
            rmtree(path)
        else:
            raise FileExistsError(path)

    path.mkdir()

    params = {
            'tags': list(tags),
            **asdict(origin_params),
    }
    with open(path / 'params.json', 'w') as f:
        json.dump(params, f)

def consolidate_origins(path: Path, dry_run: bool=False):
    df = load_origins(path)
    status = {'tags_skipped': [], 'tags_loaded': []}

    for p in path.glob('status.json*'):
        with open(p) as f:
            status_i = json.load(f)

        status['tags_loaded'] += status_i['tags_loaded']
        status['tags_skipped'] += status_i['tags_skipped']

    if not dry_run:
        save_origins(path, df, status)

        for p in path.glob('origins.parquet.*'):
            p.unlink()
        for p in path.glob('status.json.*'):
            p.unlink()

    return df, status


def _count_nearby_atoms(atoms, radius_A):
    """
    Calculate the number of atoms within the given radius of each given atom.

    The counts will include the atom itself, and will account for occupancy.
    """
    xyz = get_atom_coords(atoms)
    kd_tree = KDTree(xyz)
    counts = atoms['occupancy'].copy()

    for i,j in kd_tree.query_pairs(radius_A):
        counts.iloc[i] += atoms.iloc[j]['occupancy']
        counts.iloc[j] += atoms.iloc[i]['occupancy']

    return counts

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

def sample_origin(rng, origins: Origins):
    if origins.empty:
        raise NoOriginsToSample()

    i = _sample_weighted_index(rng, origins['weight'])
    return get_origin_coord(origins, i), origins['tag'].loc[i]

def sample_coord_frame(rng, origin):
    """
    Return a matrix that will perform a uniformly random rotation, then move 
    the given point to the origin of the new frame.
    """
    u = _sample_uniform_unit_vector(rng)
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

def get_origin_coord(origins, i):
    # Important to select columns before `loc`: This ensures that the resulting 
    # array is of dtype float rather than object, because all of the selected 
    # rows are float.
    return origins[['x', 'y', 'z']].loc[i].values

def get_origin_coords(origins):
    return origins[['x', 'y', 'z']].values


def _sample_uniform_unit_vector(rng):
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

def _sample_weighted_index(rng, weights: pd.Series):
    return rng.choice(weights.index, p=weights / weights.sum())

