import torch
import polars as pl
import numpy as np
import macromol_voxelize as mmvox

from atompaint.classifiers.amino_acid import (
        sample_targeted_crop, sample_uniform_crop,
        make_amino_acid_coords_full, find_gap_label,
)
from macromol_gym_unsupervised import (
        MakeSampleArgs, ImageParams, normalize_image_in_place,
)
from visible_residues import Sphere
from scipy.stats import Normal

from typing import Optional

def make_end_to_end_sample_full(
        sample: MakeSampleArgs,
        *,
        img_params: ImageParams,
        amino_acids: pl.DataFrame,
        bounding_sphere: Optional[Sphere] = None,
        max_residues: int,
        coord_radius_A: float,
        crop_length_voxels: int,
        use_x_pred_fraction: float = 1.0,

        # [Karras2022], Table 1.  This mean and standard deviation should lead 
        # to σ values in roughly the range [1.9e-3, 5.4e2].
        log_sigma_mean: float = -1.2,
        log_sigma_std: float = 1.2,
):
    # - On average, a 35Å image will contain ≈25 amino acids.
    # - I typically use a batch size of 16 when training diffusion models.
    # - If I used every amino acid, that would lead to a batch size of 400 when 
    #   training the amino acid predictor.
    # - The amino acid classifiers I've trained so far have required ≈4 GB VRAM 
    #   for a batch size of 64.  That would scale up to ≈25 GB for a batch size 
    #   of 400.  That's probably too much.
    # - I think 8 amino acids per images will be a good starting point.  It's 
    #   enough to cover a large fraction of the image, while still having a 
    #   reasonably modest (8 GB) VRAM requirement.

    rng = sample.rng
    C = crop_length_voxels

    x = make_amino_acid_coords_full(
            sample,
            img_params=img_params,
            amino_acids=amino_acids,
            bounding_sphere=bounding_sphere,
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

    return {
            **x,
            'x_noise': x_noise,
            'sigma': sigma,
            'aa_crops': aa_crops,
            'aa_channels': aa_channels,
            'aa_labels': np.array(aa_labels, dtype=int),
            'use_x_pred': bool(log_sigma < log_sigma_threshold),
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

    return out

def make_amino_acid_crops(
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


