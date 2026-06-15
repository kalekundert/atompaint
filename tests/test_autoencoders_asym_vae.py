import atompaint.autoencoders.asym_vae as ap
import macromol_voxelize as mmvox
import macromol_dataframe as mmdf
import torch
import torch.nn.functional as F

from utils import require_apw, IMAGE_DIR

def _test_image():
    # Center coordinate taken from make_1qjg_n38_image.py (mean of N38 atoms).
    center_A = [11.822, 63.239375, -28.364875]
    channels = [['C'], ['N'], ['O'], ['*']]

    atoms = mmdf.read_biological_assembly(
            IMAGE_DIR / '1qjg.cif.gz',
            model_id='1',
            assembly_id='2',
    )
    atoms = mmdf.prune_hydrogen(atoms)
    atoms = mmdf.prune_water(atoms)
    atoms = mmvox.set_atom_radius_A(atoms, 0.5)
    atoms = mmvox.set_atom_channels_by_element(atoms, channels)

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
    return torch.from_numpy(img).unsqueeze(0)

@require_apw
def test_load_expt_145_vae():
    vae = ap.load_expt_145_vae()
    x = _test_image()
    latent = vae.encoder(x)

    assert latent.shape == (1, 2, 16, 5, 5, 5)

    recon = vae.decoder(latent[:, 0])

    assert recon.shape == x.shape
    assert F.mse_loss(recon, x) < F.mse_loss(torch.zeros_like(x), x)

@require_apw
def test_load_expt_157_vae():
    vae = ap.load_expt_157_vae()
    x = _test_image()
    latent = vae.encoder(x)

    assert latent.shape == (1, 2, 4, 5, 5, 5)

    recon = vae.decoder(latent[:, 0])

    assert recon.shape == x.shape
    assert F.mse_loss(recon, x) < F.mse_loss(torch.zeros_like(x), x)
