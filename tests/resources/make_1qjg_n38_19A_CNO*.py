"""
Create a voxelized image of the N38 residue of 1QJG, suitable for use with
the VAE models in atompaint.autoencoders.asym_vae.  The image uses 4 element
channels (C, N, O, *) on a 19x19x19 grid at 1 Å/voxel resolution.
"""

import macromol_voxelize as mmvox
import macromol_dataframe as mmdf

channels = [['C'], ['N'], ['O'], ['*']]

atoms = mmdf.read_biological_assembly(
        '1qjg.cif.gz',
        model_id='1',
        assembly_id='2',
)
atoms = mmdf.prune_hydrogen(atoms)
atoms = mmdf.prune_water(atoms)
atoms = mmvox.set_atom_radius_A(atoms, 0.5)
atoms = mmvox.set_atom_channels_by_element(atoms, channels)

n38 = atoms.filter(chain_id='A', seq_id=38)
center_A = n38.select('x', 'y', 'z').mean().to_numpy()[0]

print(f'center_A: {center_A}')

grid = mmvox.Grid(
        center_A=center_A,
        length_voxels=19,
        resolution_A=1.0,
)
img = mmvox.image_from_all_atoms(
        atoms,
        mmvox.ImageParams(
            channels=len(channels),
            grid=grid,
        ),
)

mmvox.write_npz('1qjg_n38_19A_CNO*.npz', img, grid)
