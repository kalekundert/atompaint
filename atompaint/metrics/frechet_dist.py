import torch

from torch import Tensor


def _calc_frechet_dist2(mean, cov, ref_mean, ref_cov):
    r"""
    Compute the Fréchet distance between the two given multivariate Gaussian
    distributions.

    Consider two multivariate Gaussian distributions $\mathcal{N}(\mu_1,
    \Sigma_1)$ and $\mathcal{N}(\mu_2, \Sigma_2)$.  The Fréchet distance, also
    known as 2-Wasserstein distance, between these two distributions is given
    by the following equation:

    $$
    d^2 = \norm{\mu_1 - \mu_2}^2 + \mathrm{Tr} \left[ \Sigma_1 + \Sigma_2 - 2 \sqrt{\Sigma_1 \Sigma_2} \right]
    $$

    This implementation is mostly copied from `torchmetrics.image.fid`, but
    contains an additional check for NaN/inf values in either of the covariance
    parameters, as these can lead to segfaults in the underlying linear algebra
    libraries.

    Args:
        mean: mean of activations calculated on predicted (x) samples
        cov: covariance matrix over activations calculated on predicted (x) samples
        ref_mean: mean of activations calculated on target (y) samples
        ref_cov: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between distributions.
    """

    ΣΣ = cov @ ref_cov

    # `torch.linalg.eigvals()` can segfault when given non-finite inputs [1].
    # I ran into this issue with poorly trained models that generated images
    # with voxel values on the order of 1e20 and infinite standard deviations.
    #
    # I solved this particular problem by clamping the generated images to the
    # range [0, 1].  But it's also prudent to check for non-finite inputs and
    # avoid segfaults.  I decided to silently return NaN instead of raising an
    # exception, because this code is used in evaluating models.  Just because
    # a model is bad at one evaluation point doesn't mean it won't get better,
    # so there's no need to terminate the whole training run when this
    # condition is detected.
    #
    # [1]: https://github.com/pytorch/pytorch/issues/93124

    if not torch.isfinite(ΣΣ).all():
        return float('nan')

    a = (mean - ref_mean).square().sum(dim=-1)
    b = cov.trace() + ref_cov.trace()
    c = torch.linalg.eigvals(ΣΣ).sqrt().real.sum(dim=-1)
    d = a + b - 2 * c

    return d.item()

def _calc_cov(ncov, n):
    assert n >= 2
    return ncov / (n - 1)

def _calc_batch_stats(x: Tensor):
    mean = torch.mean(x, dim=0)
    dx = x - mean
    ncov = dx.T @ dx
    n = len(x)
    return mean, ncov, n

def _merge_batch_stats_in_place(
        mean_accum, mean_accum_err, ncov_accum, ncov_accum_err, n_accum,
        mean_batch, ncov_batch, n_batch,
):
    """
    Add the given batch to a running calculation of the mean and covariance of
    the whole dataset.

    Args:
        mean_*:
            A tensor of shape [C] where each entry contains the mean of one
            variable over either the "accum" or "batch" subsets of the data.

        ncov_*:
            A tensor of shape [C, C] where each entry contains the sum of
            deviation products for two variables, over either the "accum" or
            "batch" subsets of the data.  More simply, this tensor is a
            covariance matrix multiplied by the number of observations in the
            corresponding subset.  The name "ncov" can be thought of as a
            mnemonic for "n × covariance".

        n_*:
            A scalar tensor that contains the number of observations in the
            indicated subset of the data.

        *_accum:
            The running totals for the whole dataset, expect the new batch
            being merged in.  These tensors are modified in place.

        *_batch:
            The statistics for the new batch to merge into the running totals.

    This algorithm implements equation 21 from [Schubert2018], for the special
    case of unweighted data.  The motivation for using this algorithm is to
    avoid the loss of precision that can happen with more naive ways of
    calculating these statistics.

    [Schubert2018]: https://doi.org/10.1145/3221269.3223036
    """
    mean_diff = mean_batch - mean_accum
    n_total = n_accum + n_batch
    n_factor = n_accum * n_batch / n_total

    # Kahan summation helps improve accuracy when lots of small numbers are
    # being added to a large number, which is exactly what we're doing here.

    _kahan_sum_in_place(
        ncov_accum,
        ncov_batch + n_factor * torch.outer(mean_diff, mean_diff),
        ncov_accum_err,
    )
    _kahan_sum_in_place(
        mean_accum,
        n_batch * mean_diff / n_total,
        mean_accum_err,
    )
    n_accum += n_batch

def _kahan_sum_in_place(x, dx, err):
    dx_corr = dx - err
    x_most_sig = x + dx_corr
    err[:] = (x_most_sig - x) - dx_corr
    x[:] = x_most_sig
