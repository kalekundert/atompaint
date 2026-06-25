import torch
import torch.nn as nn
import torchmetrics

from torch import Tensor, int64
from typing import Protocol, runtime_checkable
from pathlib import Path

from atompaint.metrics.frechet_dist import (
    _calc_frechet_dist2,
    _calc_cov,
    _calc_batch_stats,
    _merge_batch_stats_in_place,
)


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


class LatentMeans(nn.Module):
    """
    Wrap a VAE encoder to return flattened latent means.

    The VAE encoder returns (B, 2, C, h, w, d) — index 0 is means, index 1
    is log-stds.  This wrapper extracts the means and flattens the spatial
    dimensions to produce a (B, C*h*w*d) feature vector.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, images: Tensor) -> Tensor:
        enc_out = self.encoder(images)  # (B, 2, C, h, w, d)
        return enc_out[:, 0].flatten(1)  # (B, d)


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
            encoder = LatentMeans(load_expt_157_vae().encoder).eval()
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
