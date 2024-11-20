import torch
import torch.nn as nn
import numpy as np

from torch import Tensor, float32, int64
from torch.nn.parameter import is_lazy
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy
from macromol_dataframe import Coords
from dataclasses import dataclass

class NeighborLocAccuracy(MulticlassAccuracy):

    def __init__(self):
        from atompaint.classifiers.neighbor_loc import load_expt_94_model
        from macromol_gym_pretrain.geometry import cube_faces

        super().__init__(num_classes=6)

        self.classifier = load_expt_94_model()
        self.view_params = ViewParams(
                direction_candidates=cube_faces(),
                length_voxels=15,
                padding_voxels=3,
        )

    def update(self, x):
        x, y = _extract_view_pairs(x, self.view_params)
        y_hat = self.classifier(x)

        super().update(y_hat, y)

class FrechetNeighborLocDistance(Metric):
    higher_is_better = True
    is_differentiable = False
    full_state_update = False

    def __init__(self, *, layer=-2, dtype=float32):
        """
        Arguments:
            dtype:
                The data type to use for the internal tensors that track 
                running means and covariances.  Because both values are tracked 
                using numerically stable algorithms, single-precision should be 
                fine.  You can expect 8 digits of precision for at least 1 
                million updates.  But double-precision is still an option is 
                you want to be totally sure.  The corresponding increases in 
                runtime and memory usage should be negligible.
        """
        from atompaint.classifiers.neighbor_loc import load_expt_94_model
        from macromol_gym_pretrain.geometry import cube_faces

        super().__init__()

        self.model = load_expt_94_model()
        self.view_params = ViewParams(
                direction_candidates=cube_faces(),
                length_voxels=15,
                padding_voxels=3,
        )

        if layer == -2:
            del self.model.classifier.classifier[1:]
            c = 512
        elif layer == -3:
            del self.model.classifier.classifier[:]
            c = 1120
        else:
            raise ValueError(f"layer must be -2 or -3, not {layer}")

        self.add_state(
                name='mean',
                default=torch.zeros(c, dtype=dtype),
                dist_reduce_fx='sum',
        )
        self.add_state(
                name='mean_err',
                default=torch.zeros(c, dtype=dtype),
                dist_reduce_fx='sum',
        )
        self.add_state(
                name='ncov',
                default=torch.zeros((c, c), dtype=dtype),
                dist_reduce_fx='sum',
        )
        self.add_state(
                name='ncov_err',
                default=torch.zeros((c, c), dtype=dtype),
                dist_reduce_fx='sum',
        )
        self.add_state(
                name='n',
                default=torch.tensor(0, dtype=int64),
                dist_reduce_fx='sum',
        )

        self.register_buffer(
                name='ref_mean',
                tensor=nn.UninitializedBuffer(),
                persistent=True,
        )
        self.register_buffer(
                name='ref_ncov',
                tensor=nn.UninitializedBuffer(),
                persistent=True,
        )
        self.register_buffer(
                name='ref_n',
                tensor=nn.UninitializedBuffer(),
                persistent=True,
        )

    def update(self, x: Tensor):
        x, _ = _extract_view_pairs(x, self.view_params)
        y = self.model(x).to(self.mean.dtype)

        assert y.ndim == 2

        # I went to a lot of effort to make this update step numerically 
        # stable, although in the end I'm not sure the effort was worth it.  
        #
        # In contrast, the `torchmetrics` implementation of this step is 
        # unstable in two ways.  It works (loosely) by accumulating Σ[x²] and 
        # Σ[x], then calculating covariance as (Σ[x²] - Σ[x]²) / n.  The first 
        # problem is the accumulation.  After enough iterations, this will 
        # involve adding relatively small numbers to relatively large ones, 
        # which will result in some precision from the smaller numbers being 
        # lost.  The second problem is the subtraction.  Both Σ[x²] and Σ[x]² 
        # can be very large and very similar in magnitude, in which case the 
        # difference would have very low precision.
        #
        # In practice, though, neither of these instabilities seem to cause 
        # real problems.  The `torchmetrics` implementation just uses 
        # double-precision floats for everything, and for the number of updates 
        # one would make for this kind of metric, that seems to be enough.  
        # Double-precision math is slow, but not nearly enough to make these 
        # updates slower than the model evaluations themselves.
        # 
        # I'm keeping my stable algorithm, though, because (i) I already went 
        # to the effort to write it, (ii) it probably has fewer pathological 
        # inputs, and (iii) it's probably a little faster.
        
        mean, ncov, n = _calc_batch_stats(y)
        _merge_batch_stats_in_place(
                self.mean, self.mean_err, self.ncov, self.ncov_err, self.n,
                mean, ncov, n,
        )

    def compute(self):
        if is_lazy(self.ref_mean):
            raise ValueError("must load reference statistics to calculate Fréchet distance\n• Did you remember to call `load_reference_stats()`?")

        cov = _calc_cov(self.ncov, self.n)
        ref_cov = _calc_cov(self.ref_ncov, self.ref_n)

        return _calc_fid(self.mean, cov, self.ref_mean, ref_cov)

    def load_reference_stats(self, ref_path):
        ref_stats = torch.load(
                ref_path,
                map_location=self.mean.device,
                weights_only=True,
        )

        def materialize(buffer, x):
            buffer.materialize(shape=x.shape, device=x.device, dtype=x.dtype)
            buffer.copy_(x)

        materialize(self.ref_mean, ref_stats['mean'])
        materialize(self.ref_ncov, ref_stats['ncov'])
        materialize(self.ref_n, ref_stats['n'])

    def save_reference_stats(self, ref_path):
        ref_stats = dict(
                mean=self.mean,
                mean_err=self.mean_err,
                ncov=self.ncov,
                ncov_err=self.ncov_err,
                n=self.n,
        )
        torch.save(ref_stats, ref_path)

@dataclass
class ViewParams:
    direction_candidates: Coords
    length_voxels: int
    padding_voxels: int

def _extract_view_pairs(
        imgs: torch.Tensor,
        view_params: ViewParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Sample 4 view pairs from each image.  This is enough to cover the entire 
    # image, so the generator can't get credit for for making good images 
    # unless the whole image is good.  Note that to ensure that the whole batch 
    # has an equal number of view pairs in each direction, the total size of 
    # the batch must be divisible by 12.

    L = view_params.length_voxels
    b, c, w, h, d = imgs.shape
    assert b % 12 == 0
    assert w == h == d
    assert w >= 2 * L + view_params.padding_voxels

    assert np.issubdtype(view_params.direction_candidates.dtype, np.integer)
    assert len(view_params.direction_candidates) == 6

    x = torch.empty(b * 4, 2, c, L, L, L, dtype=imgs.dtype, device=imgs.device)
    y = torch.empty(b * 4, dtype=torch.int32, device=imgs.device)

    y_map = {
            tuple(x): i
            for i, x in enumerate(view_params.direction_candidates)
    }

    for i in range(b):
        for j, (view_ai, view_ba) in enumerate(_iter_view_pair_indices(i)):
            ij = i * 4 + j

            slice_0 = slice(None),
            slices_a = _get_slices(view_ai, view_params)
            slices_b = _get_slices(view_ba + view_ai, view_params)

            # The `imgs[i][:, *slices_a]` syntax doesn't work in python 3.10, 
            # so instead we have to call `__getitem__()` manually.
            x[ij,0] = imgs[i].__getitem__(slice_0 + slices_a)
            x[ij,1] = imgs[i].__getitem__(slice_0 + slices_b)
            y[ij] = y_map[tuple(view_ba)]

    return x, y

def _iter_view_pair_indices(i):
    """
    Return a set of 4 view pairs that collectively fill a cube.

    Each view pair is returned as a tuple of two vectors.  The first gives the 
    position of the first view, and the second gives the position of the second 
    view relative to the first.

    The index argument *i* determines which of the 12 possible sets is 
    returned.  Because each set covers the whole image, every voxel of every 
    generated image will be validated.  Because every possible set is cycled 
    through sequentially, no particular voxel is validated more of less often 
    than any other, on average.
    """

    def mod(i, *divisors):
        for div in divisors:
            yield i % div
            i = i // div

    def vec(kv0, kv1, kv2):
        out = np.empty(3, dtype=int)
        for k, v in [kv0, kv1, kv2]:
            out[k] = v
        return out

    def flip(v):
        return (v + 1) % 2

    # `k` is a dimension index, `v` is a boolean value (0 or 1) indicating what 
    # side of a dimension a view is located on.
    k0, v1, v2 = mod(i, 3, 2, 2)
    k1 = (k0 + 1) % 3
    k2 = (k0 + 2) % 3

    yield (
            vec((k0,  0),       (k1, v1),       (k2, v2)),
            vec((k0,  1),       (k1,  0),       (k2,  0)),
    )
    yield (
            vec((k0,  1),       (k1, flip(v1)), (k2, v2)),
            vec((k0, -1),       (k1,  0),       (k2,  0)),
    )
    yield (
            vec((k0, v1),       (k1,  0),       (k2, flip(v2))),
            vec((k0,  0),       (k1,  1),       (k2,  0)),
    )
    yield (
            vec((k0, flip(v1)), (k1,  1),       (k2, flip(v2))),
            vec((k0,  0),       (k1, -1),       (k2,  0)),
    )

def _get_slices(bool_vec, view_params):
    L = view_params.length_voxels
    pad = view_params.padding_voxels
    slice_map = slice(0, L), slice(L + pad, 2*L + pad)
    return tuple(slice_map[x] for x in bool_vec)


def _calc_fid(mean, cov, ref_mean, ref_cov):
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

def _calc_batch_stats(x):
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



