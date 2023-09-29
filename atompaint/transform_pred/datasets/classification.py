import numpy as np
import torch

from .origins import select_origin_filtering_atoms, filter_origin_coords
from .recording import init_recording, record_training_example, record_img_params
from .utils import sample_origin, sample_coord_frame
from atompaint.datasets.atoms import transform_atom_coords
from atompaint.datasets.coords import make_coord_frame, get_origin
from atompaint.datasets.voxelize import image_from_atoms
from torch.utils.data import IterableDataset
from escnn.group import GroupElement, so3_group
from more_itertools import take
from functools import partial

class ViewIndexDataStream(IterableDataset):

    def __init__(
            self,
            *,
            frames_ab,
            input_from_atoms,
            origin_sampler,
            low_seed,
            high_seed,
            reuse_count,
            recording_path=None,
    ):
        self.frames_ab = frames_ab
        self.input_from_atoms = input_from_atoms
        self.origin_sampler = origin_sampler
        self.low_seed = low_seed
        self.epoch_size = high_seed - low_seed
        self.reuse_count = reuse_count

        if recording_path is None:
            self.recording_db = None
        else:
            self.recording_db = init_recording(recording_path, frames_ab)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seeds = _get_seeds(self.low_seed, self.epoch_size // self.reuse_count, worker_info)

        for seed in seeds:
            rng = np.random.default_rng(seed)

            tag, _, origins, atoms_i = self.origin_sampler.sample(rng)
            view_pairs = _sample_view_pairs(
                    rng,
                    origins,
                    self.frames_ab,
                    self.origin_sampler.params,
                    atoms_i,
            )

            for frame_ia, b in take(self.reuse_count, view_pairs):
                frame_ab = self.frames_ab[b]

                atoms_a = transform_atom_coords(atoms_i, frame_ia)
                atoms_b = transform_atom_coords(atoms_a, frame_ab)

                input_a = self.input_from_atoms(atoms_a)
                input_b = self.input_from_atoms(atoms_b)
                input_ab = np.stack([input_a, input_b])

                if self.recording_db:
                    record_training_example(
                            self.recording_db,
                            seed, tag, frame_ia, b, input_ab,
                    )

                yield torch.from_numpy(input_ab).float(), b



class CnnViewIndexDataStream(ViewIndexDataStream):

    def __init__(
            self, *,
            frames_ab,
            img_params,
            origin_sampler, 
            low_seed,
            high_seed,
            reuse_count,
            recording_path=None,
    ):
        input_from_atoms = partial(image_from_atoms, img_params=img_params)

        super().__init__(
                frames_ab=frames_ab,
                origin_sampler=origin_sampler,
                input_from_atoms=input_from_atoms,
                low_seed=low_seed,
                high_seed=high_seed,
                reuse_count=reuse_count,
                recording_path=recording_path,
        )

        if self.recording_db:
            record_img_params(self.recording_db, img_params)

def make_cube_face_frames_ab(length_A, padding_A):
    """
    Calculate 6 view frames: one for each face of a cube.
    """
    so3 = so3_group()
    grid = so3.sphere_grid('cube')
    return make_view_frames_ab(grid, length_A + padding_A)

def make_view_frames_ab(grid: list[GroupElement], radius_A: float):
    """
    Use `SO3.sphere_grid()` to get the grid.  The normal `SO3.grid()` method 
    attempts to cover all of $SO(3)$, which means that when end up projecting 
    onto $S^2$, there will be a bunch of duplicate points.
    """
    frames_ab = np.zeros((len(grid), 4, 4))
    z = np.array([0, 0, radius_A])

    for i, g in enumerate(grid):
        R = g.to('MAT')
        origin = R @ z
        frames_ab[i] = make_coord_frame(origin, np.zeros(3))

    return frames_ab


def _sample_view_pairs(rng, origins, frames_ab, origin_params, atoms_i):
    filtering_atoms_i = select_origin_filtering_atoms(atoms_i)

    num_origins = 0
    num_pairs = 0

    while True:
        origin_a, _ = sample_origin(rng, origins)
        frame_ia = sample_coord_frame(rng, origin_a)
        ok_indices = _filter_views(
                frame_ia,
                frames_ab,
                origin_params,
                filtering_atoms_i,
        )
        num_origins += 1

        if ok_indices:
            yield frame_ia, rng.choice(ok_indices)
            num_pairs += 1

        # Bail out if it's too hard to find valid view pairs in this structure.  
        # I should add some sort of logging to see how often this happens.
        if (num_pairs + 1) / (num_origins + 1) < 0.1:
            return

def _filter_views(frame_ia, frames_ab, origin_params, filtering_atoms_i):
    ok_indices = []

    for b, frame_ab in enumerate(frames_ab):
        frame_ib = frame_ab @ frame_ia
        origin_i = get_origin(frame_ib)

        ok_coords = filter_origin_coords(origin_i, origin_params, filtering_atoms_i)
        if ok_coords.size:
            ok_indices.append(b)

    return ok_indices

def _get_seeds(low_seed, epoch_size, worker_info):
    if worker_info is None:
        worker_id = 0
        num_workers = 1
    else:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

    return range(
            low_seed + worker_id,
            low_seed + epoch_size,
            num_workers,
    )
            
            

