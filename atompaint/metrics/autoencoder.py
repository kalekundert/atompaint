import torch
import torch.nn as nn
import torchmetrics
import numpy as np

from torch import Tensor, int64
from typing import Protocol, runtime_checkable, Callable
from pathlib import Path

from atompaint.autoencoders.asym_vae import AsymMeanOnly
from atompaint.metrics.frechet_dist import (
    _calc_frechet_dist2,
    _calc_cov,
    _calc_batch_stats,
    _merge_batch_stats_in_place,
    _extrapolate_frechet_dist2_inf_samples,
)
from torchmetrics.utilities.data import dim_zero_cat

@runtime_checkable
class FaedStats(Protocol):
    mean: Tensor  # (d,)
    ncov: Tensor  # (d, d)
    n:    Tensor  # scalar


class FaedAccum(nn.Module):

    def __init__(
            self,
            latent_dim: int,
            mean:     Tensor,  # (d,)    running mean (Kahan primary)
            mean_err: Tensor,  # (d,)    running mean (Kahan error)
            ncov:     Tensor,  # (d, d)  running n×cov (Kahan primary)
            ncov_err: Tensor,  # (d, d)  running n×cov (Kahan error)
            n:        Tensor,  # scalar
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer('mean',     mean)
        self.register_buffer('mean_err', mean_err)
        self.register_buffer('ncov',     ncov)
        self.register_buffer('ncov_err', ncov_err)
        self.register_buffer('n',        n)


def init_faed_accum(latent_dim: int, *, device=None) -> FaedAccum:
    d = latent_dim
    return FaedAccum(
        latent_dim=d,
        mean=torch.zeros(d, device=device),
        mean_err=torch.zeros(d, device=device),
        ncov=torch.zeros(d, d, device=device),
        ncov_err=torch.zeros(d, d, device=device),
        n=torch.tensor(0, dtype=int64, device=device),
    )

def update_faed(accum: FaedAccum, encoder: nn.Module, images: Tensor):
    if encoder.training:
        raise ValueError(
            "encoder must be in eval mode; "
            "call encoder.eval() before passing it to update_faed()"
        )

    with torch.no_grad():
        z = encoder(images)

    mean, ncov, n = _calc_batch_stats(z)
    _merge_batch_stats_in_place(
        accum.mean, accum.mean_err, accum.ncov, accum.ncov_err, accum.n,
        mean, ncov, n,
    )

def calc_faed_distance(stats_ref: FaedStats, stats_test: FaedStats) -> float:
    # Returns d² (squared Fréchet distance), following FID convention.
    cov_ref = _calc_cov(stats_ref.ncov, stats_ref.n)
    cov_test = _calc_cov(stats_test.ncov, stats_test.n)
    return _calc_frechet_dist2(stats_test.mean, cov_test, stats_ref.mean, cov_ref)

def save_faed_accum(path: Path, accum: FaedAccum):
    torch.save(dict(
        latent_dim=accum.latent_dim,
        mean=accum.mean.cpu(),
        mean_err=accum.mean_err.cpu(),
        ncov=accum.ncov.cpu(),
        ncov_err=accum.ncov_err.cpu(),
        n=accum.n.cpu(),
    ), path)

def load_faed_accum(path: Path) -> FaedAccum:
    d = torch.load(path, weights_only=True)
    return FaedAccum(**d)

def load_expt_157_ref_stats() -> FaedAccum:
    from atompaint.checkpoints import get_artifact_dir
    return load_faed_accum(
        get_artifact_dir() / (
            'expt_157/frechet_ref_stats'
            ';channels=48-96-192-8;scale=8;kl-weight=2e-3'
            ';image-size=19A;job-id=hparams_06;epoch=48'
            ';dataset=expt_107;split=val.pt'
        )
    )


class ReconMSE(torchmetrics.Metric):
    higher_is_better = False
    is_differentiable = False
    full_state_update = False

    def __init__(self, model: nn.Module = None):
        super().__init__()
        if model is None:
            from atompaint.autoencoders.asym_vae import load_expt_145_vae
            model = load_expt_145_vae().eval()
        self.model = model
        self.add_state('sum_sq_err', default=torch.tensor(0.0),            dist_reduce_fx='sum')
        self.add_state('n',          default=torch.tensor(0, dtype=int64), dist_reduce_fx='sum')

    def update(self, images: Tensor):
        if self.model.training:
            raise ValueError("model must be in eval mode")

        with torch.no_grad():
            z_mean = self.model.encoder(images)[:, 0]
            recon = self.model.decoder(z_mean)

        sq_err = ((images - recon) ** 2).mean(dim=tuple(range(1, images.ndim)))
        self.sum_sq_err += sq_err.sum()
        self.n += images.shape[0]

    def compute(self) -> Tensor:
        return self.sum_sq_err / self.n


class Faed(torchmetrics.Metric):
    higher_is_better = False
    is_differentiable = False
    full_state_update = False

    def __init__(self, encoder: nn.Module = None, stats_ref: FaedAccum = None):
        if encoder is None and stats_ref is None:
            from atompaint.autoencoders.asym_vae import load_expt_157_vae
            encoder = AsymMeanOnly(load_expt_157_vae().encoder).eval()
            stats_ref = load_expt_157_ref_stats()

        super().__init__()

        self.encoder = encoder
        self.stats_ref = stats_ref

        d = stats_ref.latent_dim
        self.add_state('mean',     default=torch.zeros(d),               dist_reduce_fx='sum')
        self.add_state('mean_err', default=torch.zeros(d),               dist_reduce_fx='sum')
        self.add_state('ncov',     default=torch.zeros(d, d),            dist_reduce_fx='sum')
        self.add_state('ncov_err', default=torch.zeros(d, d),            dist_reduce_fx='sum')
        self.add_state('n',        default=torch.tensor(0, dtype=int64), dist_reduce_fx='sum')

    def update(self, images: Tensor):
        update_faed(self, self.encoder, images)

    def compute(self) -> float:
        return calc_faed_distance(self.stats_ref, self)


class FaedUnbiased(torchmetrics.Metric):
    """
    A bias-corrected variant of `Faed`.

    `Faed` reports the raw squared Fréchet distance, which is badly biased when
    the number of test images is not large relative to the latent dimension
    (d = 500 here): the value shrinks as more samples are added, even when the
    underlying distributions don't change.  `FaedUnbiased` removes this bias by
    storing all the encoded feature vectors, evaluating the distance at several
    full-rank sample sizes, and extrapolating to N → ∞ (see
    `_extrapolate_frechet_dist2_inf_samples` and [Chong2020]).

    Because it stores every feature vector (one [d] vector per image), its
    memory grows with the number of images seen; this is cheap (≈2 MB per 1000
    images at d = 500) and buys the flexibility to subsample freely at compute
    time.

    [Chong2020]: https://doi.org/10.1109/CVPR42600.2020.00611
    """
    higher_is_better = False
    is_differentiable = False
    full_state_update = False

    def __init__(
            self,
            rng_factory: Callable[[], np.random.Generator],
            encoder: nn.Module = None,
            stats_ref: FaedAccum = None,
            *,
            min_samples: int = None,
            min_fit_n: int = None,
            num_sample_sizes: int = 15,
            num_replicates: int = 1,
    ):
        """
        Args:
            rng_factory:
                A callable that returns a fresh random number generator.  A new
                generator is created on each `compute()` call and used to draw
                random subsets of the accumulated features when evaluating the
                Fréchet distance at each sample size (see
                `_extrapolate_frechet_dist2_inf_samples`).  Using a factory
                rather than a generator keeps repeated `compute()` calls
                reproducible, since each starts from the same generator state.

            encoder:
                Module that maps images to latent feature vectors.  Must be in
                eval mode.  Defaults to the expt-157 encoder.

            stats_ref:
                Pre-computed mean and covariance of the reference (unperturbed)
                distribution.  Defaults to the expt-157 reference statistics.

            min_fit_n:
                Smallest sample size N used in the 1/N extrapolation fit.
                Must exceed the latent dimension d so the test covariance is
                full rank; the default of 2×d gives a comfortable margin,
                since covariances estimated just past N = d are severely
                ill-conditioned and don't yet fall on the distance-vs-1/N line.

            min_samples:
                Minimum number of images that must be passed to `update()`
                before `compute()` can run.  Defaults to 4×d, so the fit spans
                at least [2×d, 4×d] and the largest sample size is
                meaningfully bigger than the smallest.

            num_sample_sizes:
                Number of distinct N values at which to evaluate the Fréchet
                distance, evenly spaced between `min_fit_n` and the total
                number of accumulated images.

            num_replicates:
                Number of independent random subsets drawn at each N.  More
                replicates reduce variance of the line fit at the cost of extra
                computation.
        """
        if encoder is None and stats_ref is None:
            from atompaint.autoencoders.asym_vae import load_expt_157_vae
            encoder = AsymMeanOnly(load_expt_157_vae().encoder).eval()
            stats_ref = load_expt_157_ref_stats()

        super().__init__()

        self.encoder = encoder
        self.stats_ref = stats_ref

        d = stats_ref.latent_dim

        self.min_fit_n = min_fit_n if min_fit_n is not None else 2 * d
        self.min_samples = min_samples if min_samples is not None else 4 * d

        if self.min_samples < self.min_fit_n:
            raise ValueError(
                f"min_samples ({self.min_samples}) must be at least min_fit_n "
                f"({self.min_fit_n}) to leave room for the line fit"
            )

        self.num_sample_sizes = num_sample_sizes
        self.num_replicates = num_replicates
        self.rng_factory = rng_factory

        self.add_state('features', default=[], dist_reduce_fx='cat')

    def update(self, images: Tensor):
        if self.encoder.training:
            raise ValueError(
                "encoder must be in eval mode; "
                "call encoder.eval() before passing it to FaedUnbiased"
            )

        with torch.no_grad():
            z = self.encoder(images)

        self.features.append(z)

    def compute(self) -> float:
        feats = dim_zero_cat(self.features)
        n = feats.shape[0]

        if n < self.min_samples:
            raise RuntimeError(
                f"FaedUnbiased needs at least {self.min_samples} images before "
                f"compute(); only {n} were provided.  Pass more images, or "
                f"lower `min_samples`/`min_fit_n` (keeping min_fit_n above the "
                f"latent dimension {self.stats_ref.latent_dim})."
            )

        # Note that the fit points are spaced evenly in N, not in 1/N.  The 
        # finite-sample Fréchet distance has higher variance at small N, so 
        # even spacing in N concentrates the regression points at large N 
        # (small 1/N), near the 1/N = 0 intercept we extrapolate to; this keeps 
        # the noisy small-N estimates from dominating the fit [Chong2020].
        sample_sizes = sorted(set(
            int(x)
            for x in torch.linspace(self.min_fit_n, n, self.num_sample_sizes).tolist()
        ))

        cov_ref = _calc_cov(self.stats_ref.ncov, self.stats_ref.n)

        return _extrapolate_frechet_dist2_inf_samples(
            rng=self.rng_factory(),
            feats=feats,
            ref_mean=self.stats_ref.mean,
            ref_cov=cov_ref,
            sample_sizes=sample_sizes,
            num_replicates=self.num_replicates,
        )
