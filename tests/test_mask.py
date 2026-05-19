import atompaint.mask as ap
import macromol_dataframe as mmdf
import macromol_voxelize as mmvox
import numpy as np

from pytest import approx, fail
from utils import CIF_DIR

def test_make_protein_residue_mask_3v4i():
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

    grid = mmvox.Grid(
            length_voxels=19,
            resolution_A=1.0,
            center_A=np.array([-4.104,  24.917,  42.030]),
    )

    mask, _ = ap.make_protein_residue_mask(
            atoms,
            grid=grid,
            unmask_backbone=True,
            unmask_radii_A={'N': 0.5, 'CA': 0.5, '*': 2.0},
    )

    expected_path = CIF_DIR / 'mask_3v4i.npz'

    if not expected_path.exists():
        candidate_path = CIF_DIR / 'mask_candidate_3v4i.npz'
        mmvox.write_npz(candidate_path, mask, grid)

        fail(f"Reference image not found: {expected_path}\nTest image: {candidate_path}\nIf the test image looks right, rename it to the above reference path and rerun the test.")

    else:
        expected = np.load(expected_path)['image']
        assert mask == approx(expected)

def test_make_random_walk_mask():
    grid = mmvox.Grid(
        length_voxels=70,
        resolution_A=1.0,
    )

    mask = ap.make_random_walk_mask(
            rng=np.random.default_rng(5),
            grid=grid,
            num_steps=50,
            mask_radius_A=3,
            step_size_A=2,
    )

    expected_path = CIF_DIR / 'mask_rand_walk.npz'

    if not expected_path.exists():
        candidate_path = CIF_DIR / 'mask_candidate_rand_walk.npz'
        mmvox.write_npz(candidate_path, mask, grid)

        fail(f"Reference image not found: {expected_path}\nTest image: {candidate_path}\nIf the test image looks right, rename it to the above reference path and rerun the test.")

    else:
        expected = np.load(expected_path)['image']
        assert mask == approx(expected)
