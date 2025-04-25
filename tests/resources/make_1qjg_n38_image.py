import numpy as np
import macromol_voxelize as mmvox
import macromol_dataframe as mmdf

from atompaint.classifiers.amino_acid import sample_targeted_crop
from visible_residues import find_visible_residues

rng = np.random.default_rng(0)
channels = [['C'], ['N'], ['O'], ['P'], ['S','SE'], ['*']]

atoms = mmdf.read_biological_assembly(
        '1qjg.cif.gz',
        model_id='1',
        assembly_id='2',
)
atoms = mmdf.prune_hydrogen(atoms)
atoms = mmdf.prune_water(atoms)
atoms = mmdf.assign_residue_ids(atoms, maintain_order=True, drop_null_ids=False)
atoms = mmvox.set_atom_radius_A(atoms, 0.5)
atoms = mmvox.set_atom_channels_by_element(atoms, channels)

n38 = (
        atoms
        .filter(
            chain_id='A',
            seq_id=38,
        )
)
n38_Ca = (
        n38
        .filter(
            atom_id='CA',
        )
        .with_columns(
            radius_A=1,
            channels=[0],
        )
)
center_A = (
        n38
        .select('x', 'y', 'z')
        .mean()
        .to_numpy()
)[0]

grid = mmvox.Grid(
        center_A=center_A,
        length_voxels=21,
        resolution_A=1.0,
)

visible = find_visible_residues(n38, grid)
assert len(visible) == 1

crop = sample_targeted_crop(
        rng,
        grid,
        crop_length_voxels=11,
        target_center_A=visible.select('x', 'y', 'z').to_numpy()[0],
        target_radius_A=visible.select('radius_A').item(),
)

img_params = mmvox.ImageParams(
        channels=len(channels),
        grid=grid,
)
img_atoms = mmvox.image_from_all_atoms(
        atoms,
        mmvox.ImageParams(
            channels=len(channels),
            grid=grid,
        ),
)
img_Ca = mmvox.image_from_all_atoms(
        n38_Ca,
        mmvox.ImageParams(
            channels=1,
            grid=grid,
            fill_algorithm=mmvox.FillAlgorithm.FractionVoxel,
        ),
)
img = np.concat([img_atoms, img_Ca])
img = img[crop]

crop_corners = np.array([
    [crop[1].start, crop[2].start, crop[3].start],
    [crop[1].stop - 1,  crop[2].stop - 1,  crop[3].stop - 1],
])
crop_corners_A = mmvox.get_voxel_center_coords(grid, crop_corners)
crop_center_A = np.mean(crop_corners_A, axis=0)
crop_grid = mmvox.Grid(
        center_A=crop_center_A,
        length_voxels=11,
        resolution_A=1.0,
)

mmvox.write_npz('1qjg_n38.npz', img, crop_grid)

