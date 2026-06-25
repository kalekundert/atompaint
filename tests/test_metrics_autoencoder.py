import atompaint.metrics.autoencoder as _ap
import atompaint.metrics.frechet_dist as _fd
import numpy as np
import pytest
import torch
import torch.nn as nn

from utils import require_apw, IMAGE_DIR


def _make_ref_accum(*, d, rng):
    # Build a FaedAccum from a large sample drawn from N(0, I_d), so that the
    # reference statistics are accurate enough not to be the limiting factor.
    ref_feats = torch.from_numpy(rng.standard_normal((10_000, d))).float()
    mean, ncov, n = _fd._calc_batch_stats(ref_feats)
    return _ap.FaedAccum(
        latent_dim=d,
        mean=mean,
        mean_err=torch.zeros(d),
        ncov=ncov,
        ncov_err=torch.zeros(d, d),
        n=torch.tensor(n, dtype=torch.int64),
    )


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


def test_faed_unbiased_too_few_samples():
    d = 5
    rng = np.random.default_rng(0)
    ref_accum = _make_ref_accum(d=d, rng=rng)

    metric = _ap.FaedUnbiased(
        rng_factory=lambda: np.random.default_rng(0),
        encoder=nn.Identity().eval(),
        stats_ref=ref_accum,
        min_fit_n=10,
        min_samples=100,
    )
    metric.update(torch.from_numpy(rng.standard_normal((50, d))).float())

    with pytest.raises(RuntimeError, match="at least 100"):
        metric.compute()


def test_faed_unbiased_zero():
    # Same distribution as reference → extrapolated distance ≈ 0.
    d = 5
    rng = np.random.default_rng(0)
    ref_accum = _make_ref_accum(d=d, rng=rng)

    metric = _ap.FaedUnbiased(
        rng_factory=lambda: np.random.default_rng(0),
        encoder=nn.Identity().eval(),
        stats_ref=ref_accum,
        min_fit_n=10,
        min_samples=20,
    )
    metric.update(torch.from_numpy(rng.standard_normal((500, d))).float())

    assert metric.compute() == pytest.approx(0, abs=1.0)


def test_faed_unbiased_nonzero():
    # Shifted distribution → result larger than for the unperturbed case.
    d = 5
    rng = np.random.default_rng(0)
    ref_accum = _make_ref_accum(d=d, rng=rng)

    def make_metric():
        return _ap.FaedUnbiased(
            rng_factory=lambda: np.random.default_rng(0),
            encoder=nn.Identity().eval(),
            stats_ref=ref_accum,
            min_fit_n=10,
            min_samples=20,
        )

    m = make_metric()
    m.update(torch.from_numpy(rng.standard_normal((500, d))).float())
    dist_same = m.compute()

    m = make_metric()
    shift = torch.ones(d) * 2.0
    m.update(torch.from_numpy(rng.standard_normal((500, d))).float() + shift)
    dist_shifted = m.compute()

    assert dist_shifted > dist_same


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
