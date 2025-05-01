import torch
import polars as pl
import numpy as np
import macromol_voxelize as mmvox
import macromol_dataframe as mmdf
import visible_residues as vizres

from atompaint.classifiers.amino_acid import (
        sample_targeted_crop, sample_uniform_crop,
        make_amino_acid_coords_full, find_gap_label,
)
from macromol_gym_unsupervised import (
        MakeSampleArgs, ImageParams, normalize_image_in_place,
        get_cached, select_cached_metadatum
)
from visible_residues import Sphere
from scipy.stats import Normal
from dataclasses import replace

from typing import Optional

def make_end_to_end_sample_full(
        sample: MakeSampleArgs,
        *,
        img_params: ImageParams,
        amino_acids: pl.DataFrame,
        sidechain_sphere: Optional[Sphere] = None,
        max_residues: int,
        coord_radius_A: float,
        crop_length_voxels: int,
        use_x_pred_fraction: float = 1.0,
        sequence_recovery: bool = False,

        # [Karras2022], Table 1.  This mean and standard deviation should lead 
        # to σ values in roughly the range [1.9e-3, 5.4e2].
        log_sigma_mean: float = -1.2,
        log_sigma_std: float = 1.2,
):
    rng = sample.rng
    C = crop_length_voxels

    x = make_amino_acid_coords_full(
            sample,
            img_params=img_params,
            amino_acids=amino_acids,
            sidechain_sphere=sidechain_sphere,
            max_residues=max_residues,
    )

    log_sigma_dist = Normal(mu=log_sigma_mean, sigma=log_sigma_std)
    log_sigma = log_sigma_dist.sample(rng=rng)
    log_sigma_threshold = log_sigma_dist.icdf(use_x_pred_fraction)
    sigma = np.exp(log_sigma).astype(np.float32)
    x_noise = rng.normal(loc=0, scale=sigma, size=x['image'].shape)
    x_noise = x_noise.astype(x['image'].dtype)

    coord_labels = x['coord_labels']
    pseudoatoms = (
            coord_labels
            .select(
                pl.col('Cα_coord_A')
                    .arr.to_struct(['x', 'y', 'z'])
                    .struct.unnest(),
                radius_A=coord_radius_A,
                channels=[0],
            )
    )
    n = len(pseudoatoms)

    aa_crops = []
    aa_channels = []
    aa_labels = []

    for i in range(n):
        aa_crop = sample_targeted_crop(
                rng=rng,
                grid=img_params.grid,
                crop_length_voxels=C,
                target_center_A=coord_labels[i, 'centroid_coord_A'].to_numpy(),
                target_radius_A=coord_labels[i, 'centroid_radius_A'],
        )
        aa_crops.append(aa_crop)

        aa_channel = mmvox.image_from_all_atoms(
                pseudoatoms[i],
                mmvox.ImageParams(
                    channels=1,
                    grid=img_params.grid,
                    fill_algorithm=mmvox.FillAlgorithm.FractionVoxel,
                ),
        )
        aa_channel = aa_channel[aa_crop]
        normalize_image_in_place(
                aa_channel,
                img_params.normalize_mean,
                img_params.normalize_std,
        )
        aa_channels.append(aa_channel)

        aa_labels.append(coord_labels[i, 'label'])

    gaps_ok, gap_label = find_gap_label(amino_acids)

    if gaps_ok:
        for _ in range(n, max_residues):
            aa_crop = sample_uniform_crop(
                    rng=rng,
                    grid=img_params.grid,
                    crop_length_voxels=C,
            )
            aa_crops.append(aa_crop)

            aa_channel = np.zeros(
                    (1, C, C, C),
                    dtype=x['image'].dtype,
            )
            aa_channels.append(aa_channel)

            aa_labels.append(gap_label)

    if not aa_channels:
        aa_channels = np.zeros((0, 1, C, C, C), dtype=x['image'].dtype)
    else:
        aa_channels = np.stack(aa_channels)

    mask = None

    if sequence_recovery:
        protein_label = find_L_polypeptide_label(sample.db, sample.db_cache)

        # We want a radius that is large enough to prevent the model from adding 
        # new covalent bonds to the unmasked atoms, but small enough to allow new 
        # H-bonds.  The typical distance between the heavy atoms participating in 
        # an H-bond is about 3.0Å, from which we subtract the radius used for atoms 
        # in the image and a buffer of 0.5Å.

        h_bond_dist_A = 3.0
        unmask_radius_A = h_bond_dist_A - img_params.resolve_atom_radius_A() - 0.5

        mask = make_sequence_recovery_mask(
                atoms=x['atoms_a'],
                grid=img_params.grid,
                protein_label=protein_label,
                unmask_radius_A=unmask_radius_A,
        )

    return {
            **x,
            'x_noise': x_noise,
            'sigma': sigma,
            'aa_crops': aa_crops,
            'aa_channels': aa_channels,
            'aa_labels': np.array(aa_labels, dtype=int),
            'use_x_pred': bool(log_sigma < log_sigma_threshold),
            'seq_recovery_mask': mask,
    }

def make_end_to_end_sample(*args, **kwargs):
    x = make_end_to_end_sample_full(*args, **kwargs)
    return {
            'x_clean': x['image'],
            'x_noise': x['x_noise'],
            'sigma': x['sigma'],
            'aa_crops': x['aa_crops'],
            'aa_channels': x['aa_channels'],
            'aa_labels': x['aa_labels'],
            'use_x_pred': x['use_x_pred'],
            'seq_recovery_mask': x['seq_recovery_mask'],
    }

def collate_end_to_end_samples(batch):
    def tensor(k):
        return torch.tensor([x[k] for x in batch])

    def stack(k):
        return torch.stack([torch.from_numpy(x[k]) for x in batch])

    def cat(k):
        return torch.cat([torch.from_numpy(x[k]) for x in batch])

    out = {}

    out['x_clean'] = stack('x_clean')
    out['x_noise'] = stack('x_noise')
    out['sigma'] = tensor('sigma')
    out['aa_crops'] = []
    out['aa_channels'] = cat('aa_channels')
    out['aa_labels'] = cat('aa_labels')
    out['use_x_pred'] = tensor('use_x_pred')

    for b, x in enumerate(batch):
        out['aa_crops'] += [(b, *cxyz) for cxyz in x['aa_crops']]

    if all(x['seq_recovery_mask'] is not None for x in batch):
        out['seq_recovery_mask'] = stack('seq_recovery_mask')

    return out


def make_sequence_recovery_mask(atoms, *, grid, protein_label, unmask_radius_A):
    sidechain_sphere = replace(
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
                    sidechain_sphere.radius_A
                    + np.linalg.norm(sidechain_sphere.center_A)
                    + 2
                )
            ),
    )
    atoms = mmdf.assign_residue_ids(
            atoms,
            drop_null_ids=False,
    )
    masked_sidechains = vizres.find_visible_residues(
            atoms,
            grid=grid,
            sidechain_sphere=sidechain_sphere,
            visible_rule='any',
    )
    mask = mmvox.image_from_all_atoms(masked_sidechains, img_params)

    unmasked_atoms = (
            atoms
            .filter(
                # Don't include the Cα here.  If we included it, the Cβ is 
                # close enough that it would end up unmasked too, if present.  
                # The Cα will still end up unmasked, because it's close enough 
                # to the N and C.
                pl.struct('atom_id', 'element').is_in([
                    dict(atom_id='N', element='N'),
                    dict(atom_id='C', element='C'),
                    dict(atom_id='O', element='O'),
                ]).or_(
                    pl.col('polymer_label') != protein_label
                ).or_(
                    pl.col('polymer_label').is_null()
                )

            )
            .with_columns(
                radius_A=unmask_radius_A,
            )
    )
    unmask = mmvox.image_from_all_atoms(unmasked_atoms, img_params)

    return np.minimum(mask, 1 - unmask)

def find_L_polypeptide_label(db, db_cache):
    return get_cached(
            cache=db_cache,
            key='L_polypeptide_label',
            value_factory=lambda: (
                    select_cached_metadatum(db, db_cache, 'polymer_labels')
                    .index('polypeptide(L)')
            ),
    )

def make_amino_acid_crops(
        *,
        image,
        aa_crops,
        aa_channels,
):
    x_crops = []

    for aa_crop, aa_channel in zip(aa_crops, aa_channels, strict=True):
        x_crop = image[aa_crop]
        x_crop = torch.cat([x_crop, aa_channel])
        x_crops.append(x_crop)

    return torch.stack(x_crops)

def make_amino_acid_crops_where(
        *,
        aa_crops,
        aa_channels,
        x_pred,
        x_clean,
        use_x_pred,
):
    x_crops = []
    use_x_pred_out = []

    for aa_crop, aa_channel in zip(aa_crops, aa_channels, strict=True):
        b = aa_crop[0]

        if use_x_pred[b]:
            x_crop = x_pred[aa_crop]
        else:
            x_crop = x_clean[aa_crop]

        x_crop = torch.cat([x_crop, aa_channel])
        x_crops.append(x_crop)

        use_x_pred_out.append(use_x_pred[b])

    return torch.stack(x_crops), torch.tensor(use_x_pred_out).to(x_pred.device)

