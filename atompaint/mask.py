"""
Functions for masking out parts of macromolecular images.

Use cases:
- Mask out all sidechains for calculating sequence recovery.
- Train mask-conditioned diffusion models by creating random masks for the 
  training data.
"""

import macromol_dataframe as mmdf
import macromol_voxelize as mmvox
import visible_residues as vizres
import polars as pl
import numpy as np

from macromol_gym_unsupervised import (
        sample_coord_from_cube, sample_uniform_unit_vector,
)
from dataclasses import replace
from typing import Callable

AMINO_ACID_COMP_IDS = [
        'ALA',
        'CYS',
        'ASP',
        'GLU',
        'GLY',
        'HIS',
        'ILE',
        'LYS',
        'LEU',
        'MET',
        'MSE',
        'ASN',
        'PHE',
        'PRO',
        'GLN',
        'ARG',
        'SER',
        'THR',
        'VAL',
        'TRP',
        'TYR',
]

def make_protein_residue_mask(
        atoms: mmdf.Atoms,
        *,
        grid: mmvox.Grid,
        mask_sphere: vizres.Sphere | None = None,
        unmask_backbone: bool,
        unmask_radii_A: dict[str, float] = {},
        filter_residues: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
):
    if mask_sphere is None:
        mask_sphere = replace(
                vizres.get_sidechain_sphere(),
                radius_A=6,
        )

    img_params = mmvox.ImageParams(
            channels=1,
            grid=grid,
            fill_algorithm=mmvox.FillAlgorithm.FractionVoxel,
            agg_algorithm=mmvox.AggAlgorithm.Max,
    )

    atoms = mmvox.discard_atoms_outside_image(
            atoms,
            img_params=replace(
                img_params,

                # We need to ensure that we include N, Cα, and C for every 
                # residue that could be in the image.  Cα is at the origin, so 
                # we need to account for the distance from the center of the 
                # sphere to the origin.  N and C are bonded to Cα, so they are 
                # ≈1.5Å away from it.  To be safe, we add 2Å to account for 
                # this.
                max_radius_A=(
                    mask_sphere.radius_A
                    + np.linalg.norm(mask_sphere.center_A)
                    + 2
                )
            ),
    )
    atoms = mmdf.assign_residue_ids(
            atoms,
            drop_null_ids=False,
    )

    mask_spheres = vizres.find_visible_residues(
            atoms,
            grid=grid,
            sidechain_sphere=mask_sphere,
            visible_rule='any',
    )
    if filter_residues is not None:
        mask_spheres = filter_residues(mask_spheres)
        mask_residue_ids = mask_spheres['residue_id']
    else:
        mask_residue_ids = None

    mask = mmvox.image_from_all_atoms(mask_spheres, img_params)

    if unmask_backbone:
        backbone_atoms = (
                atoms
                .filter(
                    pl.struct('atom_id', 'element').is_in([
                        dict(atom_id='N', element='N'),
                        dict(atom_id='CA', element='C'),
                        dict(atom_id='C', element='C'),
                        dict(atom_id='O', element='O'),
                    ]).or_(
                        ~pl.col('comp_id').is_in(AMINO_ACID_COMP_IDS)
                    )
                )
                .with_columns(
                    radius_A=pl.col('atom_id').replace_strict(
                        (x := unmask_radii_A.copy()),
                        default=x.pop('*'),
                    ),
                )
        )
        unmask = mmvox.image_from_all_atoms(backbone_atoms, img_params)
        mask = np.minimum(mask, 1 - unmask)

    unmasked_atoms = drop_sidechain_atoms(
            atoms,
            residue_ids=mask_residue_ids,
    )

    return mask, unmasked_atoms

def make_random_walk_mask(
        rng,
        *,
        grid: mmvox.Grid,
        num_steps: int,
        mask_radius_A: float,
        step_size_A: float,
        momentum: float = 0,
):
    traj = np.zeros((num_steps + 1, 3))
    traj[0] = sample_coord_from_cube(rng, grid.center_A, grid.length_A)
    step = np.zeros(3)

    for i in range(num_steps):
        step *= momentum
        step += sample_uniform_unit_vector(rng) * step_size_A
        traj[i+1] = traj[i] + step

    traj_spheres = (
            pl.DataFrame(traj, ['x', 'y', 'z'])
            .with_columns(radius_A=mask_radius_A)
    )
    img_params = mmvox.ImageParams(
            channels=1,
            grid=grid,
            fill_algorithm=mmvox.FillAlgorithm.FractionVoxel,
            agg_algorithm=mmvox.AggAlgorithm.Max,
    )
    return mmvox.image_from_all_atoms(traj_spheres, img_params)


def sample_contiguous_residues(rng, all_residues, num_residues, temperature):
    from scipy.special import softmax

    n = len(all_residues)
    xyz = all_residues[['x', 'y', 'z']].to_numpy()

    i_first_yes = int(rng.integers(n))
    mask_yes = np.zeros(n, dtype=bool)
    mask_yes[i_first_yes] = True

    min_dists = np.linalg.norm(xyz - xyz[i_first_yes], axis=1)

    for _ in range(num_residues):
        i_maybe = np.flatnonzero(~mask_yes)
        probs = softmax(-min_dists[i_maybe] / temperature)
        i_pick = int(rng.choice(i_maybe, p=probs))

        mask_yes[i_pick] = True
        min_dists[i_maybe] = np.minimum(
                min_dists[i_maybe],
                np.linalg.norm(xyz[i_maybe] - xyz[i_pick], axis=1),
        )

    return all_residues.filter(mask_yes)

def drop_sidechain_atoms(atoms, residue_ids=None):
    predicate = (
            pl.struct('atom_id', 'element').is_in([
                dict(atom_id='N', element='N'),
                dict(atom_id='CA', element='C'),
                dict(atom_id='C', element='C'),
                dict(atom_id='O', element='O'),
            ]).or_(
                ~pl.col('comp_id').is_in(AMINO_ACID_COMP_IDS)
            )
    )

    if residue_ids is not None:
        predicate = pl.col('residue_id').is_in(~residue_ids).or_(predicate)

    return atoms.filter(predicate)

def get_backbone_unmask_radii_A(atom_radius_A):
        h_bond_dist_A = 3.0
        covalent_dist_A = 1.5
        atom_radius_A += 0.5

        return {
                'N':  covalent_dist_A - atom_radius_A,
                'CA': covalent_dist_A - atom_radius_A,
                '*':  h_bond_dist_A - atom_radius_A,
        }


