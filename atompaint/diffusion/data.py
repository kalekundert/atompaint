import torch
import polars as pl
import numpy as np

from macromol_gym_unsupervised import (
        select_cached_metadatum, make_unsupervised_image_sample,
)
from functools import partial

def make_diffusion_sample(db, db_cache, rng, zone_id, *, img_params):
    x = make_unsupervised_image_sample(
            db, db_cache, rng, zone_id,
            img_params=img_params,
    )
    noise = rng.normal(size=x['image'].shape).astype(x['image'].dtype)

    return dict(
            **x,
            noise=noise,
    )

def make_diffusion_tensors(db, db_cache, rng, zone_id, *, img_params):
    x = make_diffusion_sample(db, db_cache, rng, zone_id, img_params=img_params)
    return dict(
            rng=x['rng'],
            x_clean=x['image'],
            noise=x['noise'],
    )

def make_labeled_diffusion_tensors(db, db_cache, rng, zone_id, *, img_params):
    x = make_diffusion_sample(db, db_cache, rng, zone_id, img_params=img_params)

    label = _get_polymer_cath_label(
            x['image_atoms_a'],
            n_polymer_labels=len(
                select_cached_metadatum(db, db_cache, 'polymer_labels'),
            ),
            n_cath_labels=len(
                select_cached_metadatum(db, db_cache, 'cath_labels')
            ),
    )
    label = torch.from_numpy(label.astype(x['image'].dtype))

    return dict(
            rng=x['rng'],
            x_clean=x['image'],
            noise=x['noise'],
            label=label,
    )

def load_random_label_factory(db, db_cache):
    return partial(
            random_label_factory,
            polymer_labels=select_cached_metadatum(db, db_cache, 'polymer_labels'),
            cath_labels=select_cached_metadatum(db, db_cache, 'cath_labels'),
    )

def random_label_factory(rng, batch_size, *, polymer_labels, cath_labels):
    n_polymer_labels = len(polymer_labels)
    polymer_one_hot = np.zeros((batch_size, n_polymer_labels))

    n_cath_labels = len(cath_labels)
    cath_one_hot = np.zeros((batch_size, n_cath_labels))

    for b in range(batch_size):
        i = rng.integers(n_polymer_labels)
        polymer_one_hot[b, i] = 1

        if polymer_labels[i] == 'polypetide(L)':
            j = rng.integers(n_cath_labels + 1)
            if j < n_cath_labels:
                cath_one_hot[b, j] = 1

    labels = np.hstack([polymer_one_hot, cath_one_hot])
    return torch.from_numpy(labels).to(dtype=torch.float32)


def _get_polymer_cath_label(atoms, *, n_polymer_labels, n_cath_labels):
    polymer_label = _get_label(atoms, 'polymer_label', n_polymer_labels)
    cath_label = _get_label(atoms, 'cath_label', n_cath_labels)
    return np.concat([polymer_label, cath_label])

def _get_label(atoms, column, n_labels):
    if atoms.is_empty():
        return np.zeros(n_labels)

    label_counts = (
            pl.DataFrame({column: np.arange(n_labels)})
            .join(
                # Unfortunately, the atoms dataframes in the database use int64 
                # for polymer labels and uint32 for CATH labels.  Join columns 
                # must have the same datatype, so without remaking the 
                # database, there's no way around this cast.
                atoms.cast({column: int}).group_by(column).len(),
                on=column,
                how='left',
                maintain_order='left',
            )
            .get_column('len')
            .fill_null(0)
            .to_numpy()
    )
    return label_counts / len(atoms)

