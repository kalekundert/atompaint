import atompaint.metrics.frechet_dist as _fd
import numpy as np
import pytest
import torch


def test_extrapolate_frechet_dist2_inf_samples_zero():
    # When test features are drawn from the same distribution as the reference,
    # the extrapolated distance should be ≈0.
    d = 5
    rng = np.random.default_rng(0)

    # Build reference statistics from a large sample so estimation error is
    # negligible compared to what we're testing.
    ref_feats = torch.from_numpy(rng.standard_normal((10_000, d))).float()
    ref_mean, ref_ncov, ref_n = _fd._calc_batch_stats(ref_feats)
    ref_cov = _fd._calc_cov(ref_ncov, ref_n)

    feats = torch.from_numpy(rng.standard_normal((500, d))).float()

    result = _fd._extrapolate_frechet_dist2_inf_samples(
        rng=rng,
        feats=feats,
        ref_mean=ref_mean,
        ref_cov=ref_cov,
        sample_sizes=[10, 20, 50, 100, 200, 500],
        num_replicates=5,
    )

    assert result == pytest.approx(0, abs=1.0)


def test_extrapolate_frechet_dist2_inf_samples_nonzero():
    # A shifted test distribution should give a result close to the true d².
    d = 5
    rng = np.random.default_rng(0)

    ref_feats = torch.from_numpy(rng.standard_normal((10_000, d))).float()
    ref_mean, ref_ncov, ref_n = _fd._calc_batch_stats(ref_feats)
    ref_cov = _fd._calc_cov(ref_ncov, ref_n)

    # Shift by delta in each dimension; true d² = d * delta².
    delta = 2.0
    feats = torch.from_numpy(rng.standard_normal((500, d))).float() + delta

    result = _fd._extrapolate_frechet_dist2_inf_samples(
        rng=rng,
        feats=feats,
        ref_mean=ref_mean,
        ref_cov=ref_cov,
        sample_sizes=[10, 20, 50, 100, 200, 500],
        num_replicates=5,
    )

    assert result == pytest.approx(d * delta ** 2, abs=2.0)


def test_extrapolate_frechet_dist2_inf_samples_rank_deficient():
    d = 5
    rng = np.random.default_rng(0)

    feats = torch.randn(100, d)
    ref_mean = torch.zeros(d)
    ref_cov = torch.eye(d)

    with pytest.raises(ValueError, match="full rank"):
        _fd._extrapolate_frechet_dist2_inf_samples(
            rng=rng,
            feats=feats,
            ref_mean=ref_mean,
            ref_cov=ref_cov,
            sample_sizes=[3, 10, 50],  # 3 <= d=5: not full rank
            num_replicates=1,
        )
