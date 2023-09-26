"""\
Create training example views that can be related by any rotation 
and translation (subject to distance cutoffs).
"""

import torch
import numpy as np
import pandas as pd

from .origins import get_origin_coords
from .utils import sample_origin, sample_coord_frame, NoOriginsToSample
from atompaint.datasets.coords import invert_coord_frame
from atompaint.datasets.atoms import transform_atom_coords
from atompaint.datasets.voxelize import image_from_atoms, _get_max_element_radius
from torch.utils.data import Dataset
from numpy.typing import NDArray
from dataclasses import dataclass
from functools import cached_property, partial
from math import sqrt

class ViewPairDataset(Dataset):

    def __init__(
            self,
            *,
            origin_sampler,
            input_from_atoms,
            view_pair_params,
            low_seed,
            high_seed,
    ):
        self.origin_sampler = origin_sampler
        self.input_from_atoms = input_from_atoms
        self.view_pair_params = view_pair_params
        self.low_seed = low_seed
        self.epoch_size = high_seed - low_seed

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, i):
        assert 0 <= i < self.epoch_size

        rng = np.random.default_rng(self.low_seed + i)

        while True:
            origin_a, origins_b, atoms = self.origin_sampler.sample(rng)

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

class CnnViewPairDataset(ViewPairDataset):

    def __init__(
            self,
            *,
            origin_sampler,
            img_params,
            max_dist_A,
            low_seed,
            high_seed,
    ):
        min_dist_A = calc_min_distance_between_origins(img_params)
        view_pair_params = ViewPairParams(
                min_dist_A=min_dist_A,
                max_dist_A=min_dist_A + max_dist_A,
        )
        input_from_atoms = partial(image_from_atoms, img_params=img_params)

        super().__init__(
                origin_sampler=origin_sampler,
                input_from_atoms=input_from_atoms,
                view_pair_params=view_pair_params,
                low_seed=low_seed,
                high_seed=high_seed,
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

def filter_by_distance(origins, coord, *, min_dist_A, max_dist_A):
    origin_coords = get_origin_coords(origins)
    dist = np.linalg.norm(origin_coords - coord, axis=1)
    return origins[(min_dist_A <= dist) & (dist <= max_dist_A)]

def calc_min_distance_between_origins(img_params):
    # Calculate the radius of the sphere that inscribes in grid.  This ensures 
    # that the grids won't overlap, no matter how they're rotated.
    grid_radius_A = sqrt(3) * img_params.grid.length_A / 2

    # Add the radius of the largest possible atom, so that no atom can possibly 
    # appear in both views.  This degree of overlap probably wouldn't matter 
    # anyways, but might as well be as correct as possible.
    atom_radius_A = _get_max_element_radius(img_params.element_radii_A)

    return 2 * (grid_radius_A + atom_radius_A)



