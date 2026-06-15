import atompaint.metrics.autoencoder as _ap
import numpy as np
import torch

from utils import require_apw, IMAGE_DIR


@require_apw
def test_recon_mse():
    img = np.load(IMAGE_DIR / '1qjg_n38_19A_CNO*.npz')['image']
    imgs = torch.from_numpy(img).unsqueeze(0)
    noise = torch.randn(*imgs.shape)

    metric = _ap.ReconMSE()

    mse_imgs = metric(imgs)
    metric.reset()
    mse_noise = metric(noise)

    assert mse_imgs < mse_noise


@require_apw
def test_faed():
    img = np.load(IMAGE_DIR / '1qjg_n38_19A_CNO*.npz')['image']
    # Fréchet distance requires n≥2 to compute a covariance.
    imgs = torch.from_numpy(img).unsqueeze(0).tile(2, 1, 1, 1, 1)
    noise = torch.randn(*imgs.shape)

    faed = _ap.Faed()
    dist_imgs = faed(imgs)
    faed.reset()
    dist_noise = faed(noise)

    assert dist_imgs < dist_noise
