import torch
import atompaint.metrics.neighbor_loc as _ap
import numpy as np
import pytest

from more_itertools import all_equal
from utils import IMAGE_DIR

def test_neighbor_loc_accuracy():
    # This image shows part of a ribosome structure.  I manually verified that 
    # the image truly reflects the actual coordinates from 6s0z.  I chose this 
    # image because the structure fills the entire image, so there aren't blank 
    # corners that might not be possible to classify accurately.  I also 
    # kind-of like that this image is almost entirely of RNA, which makes this 
    # something of a test that the metric isn't specific to proteins.

    img = np.load(IMAGE_DIR / '6s0z_104348_33A.npy')
    imgs = torch.from_numpy(img).tile((12, 1, 1, 1, 1))
    noise = torch.randn(*imgs.shape)

    try:
        metric = _ap.NeighborLocAccuracy()
    except KeyError:
        pytest.skip()

    acc_imgs = metric(imgs)
    acc_noise = metric(noise)

    assert acc_imgs > acc_noise

def test_extract_view_pairs():
    # 5x5x5 images are the smallest we can make with greater-than-1-voxel 
    # blocks separated by non-zero padding, and 12 is the smallest batch size 
    # the extract function will accept.
    img = torch.arange(5**3).reshape(5, 5, 5)
    imgs = torch.tile(img, (12, 1, 1, 1, 1))

    view_params = _ap.ViewParams(
            direction_candidates=np.array([
                # Copied from `macromol_gym_pretrain`.
                [ 0, -1,  0],
                [ 0,  1,  0],
                [-1,  0,  0],
                [ 1,  0,  0],
                [ 0,  0,  1],
                [ 0,  0, -1],
            ]),
            length_voxels=2,
            padding_voxels=1,
    )
    x, y = _ap._extract_view_pairs(imgs, view_params)

    def as_tuple(arr):
        if len(arr.shape) == 1:
            return tuple(arr.tolist())
        else:
            return tuple(as_tuple(x) for x in arr)

    xy_actual = {
            (as_tuple(x[b,0,0]), as_tuple(x[b,1,0]), y[b].item())
            for b in range(x.shape[0])
    }

    # Manually specify some of the view pairs that are expected to be in the 
    # output.  It's not practical to specify every expected view pair, because 
    # there are too many.
    xy_expected = [
            dict(
                a=(((15, 16), (20, 21)), ((40, 41), (45, 46))),
                b=(((0, 1), (5, 6)), ((25, 26), (30, 31))),
                y=0,  #  0 -1  0
            ),
            dict(
                a=(((0, 1), (5, 6)), ((25, 26), (30, 31))),
                b=(((15, 16), (20, 21)), ((40, 41), (45, 46))),
                y=1,  #  0  1  0
            ),
            dict(
                a=(((75, 76), (80, 81)), ((100, 101), (105, 106))),
                b=(((0, 1), (5, 6)), ((25, 26), (30, 31))),
                y=2,  # -1  0  0
            ),
            dict(
                a=(((0, 1), (5, 6)), ((25, 26), (30, 31))),
                b=(((75, 76), (80, 81)), ((100, 101), (105, 106))),
                y=3,  #  1  0  0
            ),
            dict(
                a=(((0, 1), (5, 6)), ((25, 26), (30, 31))),
                b=(((3, 4), (8, 9)), ((28, 29), (33, 34))),
                y=4,  #  0  0  1
            ),
            dict(
                a=(((3, 4), (8, 9)), ((28, 29), (33, 34))),
                b=(((0, 1), (5, 6)), ((25, 26), (30, 31))),
                y=5,  #  0  0 -1
            ),
    ]
    for xy in xy_expected:
        xy_tuple = xy['a'], xy['b'], xy['y']
        assert xy_tuple in xy_actual

def test_iter_view_pair_indices():
    # Test that:
    # - Each invocation alone covers the whole cube.
    # - Every invocation combined samples each corner of the cube uniformly.

    hits = np.zeros((2, 2, 2))

    for i in range(24):
        hits_i = np.zeros((2, 2, 2))

        for view_ai, view_ba in _ap._iter_view_pair_indices(i): 
            hits[tuple(view_ai)] += 1
            hits[tuple(view_ba + view_ai)] += 1
            hits_i[tuple(view_ai)] += 1
            hits_i[tuple(view_ba + view_ai)] += 1

        np.testing.assert_equal(hits_i, np.ones((2, 2, 2)))

    assert all_equal(hits.flat)

@pytest.mark.parametrize('n', [
    # Smallest possible input; should highlight errors related to DOFs.
    2,

    # Small number of batches and partial batches
    16, 24, 32,

    # Large inputs; 2²⁰=1M samples is enough to induce errors on the order of 
    # 0.1 in `torch.cov()` when comparing 32- and 64-bit floats.  Typically the 
    # FID metric is evaluated using 50K images, which is not as strenuous as 
    # this, but still potentially dangerous.
    2**10, 2**20,
])
def test_merge_batch_stats_in_place(n):
    rng = np.random.default_rng(0)
    d, Z = 3, 10

    # Take care to create a synthetic dataset with a non-trivial covariance 
    # matrix.  There are probably lots of ways to do this, but the approach 
    # taken here is to draw samples from a multivariate normal distribution.  
    # This distribution requires a positive semi-definite covariance matrix.  
    # We can create a positive semi-definite matrix by multiplying a random 
    # matrix by its own transpose.
    #
    # Note that, according to [Schubert2018], sorted datasets tend to 
    # exacerbate loss-of-precision issues.  But since I don't imagine my FID 
    # algorithm being used on sorted datasets, I'm not going to bother testing  
    # that case.
    #
    # [Schubert2018] https://dl.acm.org/doi/10.1145/3221269.3223036

    latent_mean = rng.uniform(-Z, Z, size=d)
    latent_cov = (A := rng.uniform(-Z, Z, size=(d, d))) @ A.T

    assert is_pos_def(latent_cov)

    x = rng.multivariate_normal(latent_mean, latent_cov, size=n)

    # Compare to the numpy implementation, which I've found to be more accurate 
    # than the native torch functions in some cases.

    mean_np = np.mean(x, 0)
    cov_np = np.cov(x.T)

    # Use single-precision floats in this test.  This allows treating the 
    # double-precision values calculated by numpy as "the truth".  Plus, since 
    # the tests pass, it's probably fine to use single-precision even for real 
    # applications.

    x = torch.from_numpy(x).float()

    mean_accum = torch.zeros(d)
    mean_accum_err = torch.zeros(d)
    ncov_accum = torch.zeros(d, d)
    ncov_accum_err = torch.zeros(d, d)
    n_accum = torch.tensor(0, dtype=int)
    accum_stats = mean_accum, mean_accum_err, ncov_accum, ncov_accum_err, n_accum

    for xi in x.split(16):
        batch_stats = _ap._calc_batch_stats(xi)
        _ap._merge_batch_stats_in_place(*accum_stats, *batch_stats)

    torch.testing.assert_close(
            mean_accum,
            torch.from_numpy(mean_np).float(),
    )
    torch.testing.assert_close(
            _ap._calc_cov(ncov_accum, n_accum),
            torch.from_numpy(cov_np).float(),
    )

def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

