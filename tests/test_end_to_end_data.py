import atompaint.end_to_end.data as ap
import macromol_dataframe as mmdf
import macromol_voxelize as mmvox
import polars as pl
import torch
import numpy as np

from pytest import approx, fail
from utils import CIF_DIR

def test_make_sequence_recovery_mask_3v4i():
    # Manually check that the mask looks reasonable on a real protein 
    # structure.  I chose 3V4I because it contains a protein, nucleic acid, and 
    # small molecule, all in close proximity.

    atoms = mmdf.read_biological_assembly(
            CIF_DIR / '3v4i.cif.gz',
            model_id='1',
            assembly_id='1',
    )
    atoms = mmdf.prune_hydrogen(atoms)
    atoms = mmdf.prune_water(atoms)
    atoms = atoms.with_columns(
            polymer_label=(
                pl.col('entity_id')
                .replace_strict({'1': 0}, default=None)
            )
    )

    grid = mmvox.Grid(
            length_voxels=19,
            resolution_A=1.0,
            center_A=np.array([-4.104,  24.917,  42.030]),
    )

    mask = ap.make_sequence_recovery_mask(
            atoms,
            grid=grid,
            protein_label=0,
            unmask_radius_A=2.0,
    )

    expected_path = CIF_DIR / 'mask.npz'

    if not expected_path.exists():
        candidate_path = CIF_DIR / 'mask_candidate.npz'
        mmvox.write_npz(candidate_path, mask, grid)

        fail(f"Reference image not found: {expected_path}\nTest image: {candidate_path}\nIf the test image looks right, rename it to the above reference path and rerun the test.")

    else:
        expected = np.load(expected_path)['image']
        assert mask == approx(expected)

def test_find_L_polypeptide_label():
    from macromol_gym_unsupervised import open_db, init_db, insert_metadata

    db = open_db(':memory:', mode='rwc')
    db_cache = {}

    init_db(db)
    insert_metadata(db, {
        'polymer_labels': [
            'polypeptide(L)',
            'polydeoxyribonucleotide',
            'polyribonucleotide',
        ],
    })

    assert ap.find_L_polypeptide_label(db, db_cache) == 0
    assert db_cache['L_polypeptide_label'] == 0

    db_cache['L_polypeptide_label'] = 1
    assert ap.find_L_polypeptide_label(db, db_cache) == 1

def test_make_amino_acid_crops():
    # To simplify things, this test only uses 1 spatial dimension instead of 3.

    image = torch.tensor([
        [[ 1,  2,  3]],
        [[ 4,  5,  6]],
    ])
    aa_crops = [
            (0, slice(None), slice(0, 2)),
            (0, slice(None), slice(1, 3)),
            (1, slice(None), slice(0, 2)),
            (1, slice(None), slice(1, 3)),
    ]
    aa_channels = torch.tensor([
        [[1, 0]],
        [[0, 1]],
        [[1, 0]],
        [[0, 1]],
    ])

    x_crop = ap.make_amino_acid_crops(
            image=image,
            aa_crops=aa_crops,
            aa_channels=aa_channels,
    )
    x_expected = torch.tensor([
        [[1, 2], [1, 0]],
        [[2, 3], [0, 1]],
        [[4, 5], [1, 0]],
        [[5, 6], [0, 1]],
    ])
    torch.testing.assert_close(x_crop, x_expected)

def test_make_amino_acid_crops_where():
    x_pred = torch.tensor([
        [[ 1,  2,  3]],
        [[ 4,  5,  6]],
    ])
    x_clean = torch.tensor([
        [[ 7,  8,  9]],
        [[10, 11, 12]],
    ])
    use_x_pred = torch.tensor([True, False])

    aa_crops = [
            (0, slice(None), slice(0, 2)),
            (0, slice(None), slice(1, 3)),
            (1, slice(None), slice(0, 2)),
    ]
    aa_channels = torch.tensor([
        [[1, 0]],
        [[0, 1]],
        [[0, 1]],
    ])

    x_crop, use_x_pred = ap.make_amino_acid_crops_where(
            aa_crops=aa_crops,
            aa_channels=aa_channels,
            x_pred=x_pred,
            x_clean=x_clean,
            use_x_pred=use_x_pred,
    )
    x_expected = torch.tensor([
        [[ 1,  2], [1, 0]],
        [[ 2,  3], [0, 1]],
        [[10, 11], [0, 1]],
    ])

    torch.testing.assert_close(x_crop, x_expected)
    torch.testing.assert_close(use_x_pred, torch.tensor([True, True, False]))


