import numpy as np

from .origins import filter_origin_coords
from atompaint.datasets.coords import Frame, make_coord_frame, get_origin
from torch.utils.data import IterableDataset
from escnn.group import octa_group, ico_group
from scipy.linalg import null_space
from more_itertools import all_equal
from dataclasses import dataclass
from typing import Any

class ViewSlotDataStream(IterableDataset):

    def __init__(
            self,
            *,
            view_slots,
            input_from_atoms,
            origin_sampler,
            low_seed,
            high_seed,
            reuse_count,
    ):
        self.view_slots = view_slots
        self.input_from_atoms = input_from_atoms
        self.origin_sampler = origin_sampler
        self.low_seed = low_seed
        self.epoch_size = high_seed - low_seed
        self.reuse_count = reuse_count

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seeds = _get_seeds(self.seed_offest, self.epoch_size, worker_info)

        for seed in seeds:
            rng = np.random.default_rng(seed)

            _, origins, atoms_i = self.origin_sampler.sample(rng)
            view_pairs = _sample_view_pairs(rng, origins, self.view_slots, atoms_i)

            for frame_ia, slot_b in take(view_pairs, self.reuse_count):
                frame_ab = self.view_slots.frames_ab[slot_b]

                atoms_a = transform_atom_coords(atoms_i, frame_ia)
                atoms_b = transform_atom_coords(atoms_a, frame_ab)

                input_a = self.input_from_atoms(atoms_a)
                input_b = self.input_from_atoms(atoms_b)
                input_ab = np.stack([input_a, input_b])

                yield torch.from_numpy(input_ab).float(), slot_b,

@dataclass
class ViewSlots:
    representation: Any
    frames_ab: list[Frame]

def make_view_slots(pattern: str, radius: float):
    # This algorithm doesn't work with "cube edges", because multiple slot 
    # indices end up corresponding to the same position.  Presumably this is 
    # because the indices in question differ by a rotation that doesn't affect 
    # a single point.  That said, it is possible to create 12 pairs of group 
    # elements that each give different positions, and even though the "cube 
    # edges" representation doesn't do this, the "ico edges" representation 
    # does (with 30 edges instead of 12).  I'm suspicious that something is 
    # wrong here, but for now I'm going to assume that this algorithm still 
    # works for cube faces, vertices, etc.

    if pattern == 'cube-faces':
        group = octa_group()
        r = group.cube_faces_representation

    if pattern == 'cube-vertices':
        group = octa_group()
        r = group.cube_vertices_representation

    if pattern == 'ico-faces':
        group = ico_group()
        r = group.ico_faces_representation

    if pattern == 'ico-vertices':
        group = ico_group()
        r = group.ico_vertices_representation

    if pattern == 'ico-edges':
        group = ico_group()
        r = group.ico_edges_representation

    degenerates = {}

    for g in group.elements:
        e = np.zeros(r.size)
        e[0] = 1
        i = np.argmax(r(g) @ e)

        degenerates.setdefault(i, []).append(g)

    # In order to work out a position for each of the indices determined above, 
    # we need to find a vector that is transformed in the same way by each 
    # group element with the same index.  By transforming this vector, we get a 
    # position that represents the whole index.
    # 
    # Consider two rotation matrices with the same index, A and B.  We seek a 
    # vector x that satisfies the following equation:
    # 
    #   Ax = Bx;  (A-B)x = 0
    #
    # In other words, we want the null space of (A-B).  For the rotation 
    # matrices we're working with, the null space should always be 1D.  In 
    # general, there may be more than two matrices associated with each index, 
    # but it doesn't matter which two are used.  It also turns out that every 
    # index will have the same null space, so we just calculate it once up 
    # front (on index 0, arbitrarily).

    g1, g2 = degenerates[0][0:2]
    A = group.standard_representation(g1)
    B = group.standard_representation(g2)
    x = null_space(A - B)

    assert x.shape == (3, 1)

    # This vector is guaranteed to be normalized, but it's sign is arbitrary. 
    # Forcing the vector to point mostly in the positive direction makes the 
    # result deterministic.
    sign = 1 if np.dot(x.ravel(), np.ones(3)) >= 0 else -1
    x *= sign * radius

    frames_ab = np.zeros((r.size, 4, 4))
    for i, elements in degenerates.items():
        R = group.standard_representation(elements[0])
        frames_ab[i] = make_coord_frame(R @ x, np.zeros(3))

    return ViewSlots(
            representation=r,
            frames_ab=frames_ab,
    )

def _sample_view_pairs(rng, origins, view_slots, atoms):
    relevant_atoms = find_relevant_atoms(atoms)

    while True:
        origin_a = sample_origin(rng, origins)
        frame_ia = sample_coord_frame(rng, origin_a)
        slots_b = _filter_view_slots(frame_ia, view_slots, relevant_atoms)

        if slots_b:
            yield frame_ia, rng.choice(slots_b)

def _filter_view_slots(frame_ia, view_slots, origin_params, relevant_atoms_i):
    slots_b = []

    for slot_b, frame_ab in enumerate(view_slots.frames_ab):
        frame_ib = frame_ab @ frame_ia
        origin_i = get_origin(frame_ib)

        hits = filter_origin_coords(origin_i, origin_params, relevant_atoms_i)
        if hits.size:
            slots_b.append(slot_b)

    return slots_b

def _get_seeds(seed_offset, epoch_size, worker_info):
    if worker_info is None:
        worker_id = 0
        num_workers = 1
    else:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

    return range(
            seed_offset + worker_id,
            seed_offset + epoch_size,
            num_workers,
    )
            
            

