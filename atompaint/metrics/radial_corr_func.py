import torch
import torchmetrics
import numpy as np

from scipy.fft import fftn, ifftn, next_fast_len
from array_api_compat import array_namespace
from math import ceil
from dataclasses import dataclass, asdict
from itertools import pairwise
from more_itertools import all_equal
from pathlib import Path

# RCF stands for "radial correlation function".  This is similar in concept to 
# the radial distribution function (RDF), except that it is a histogram over 
# volume-normalized auto/cross-correlation rather than density.

@dataclass
class RcfAccum:
    corr_histogram: np.ndarray | torch.Tensor   # (C, B)       C: channel pairs
    num_images: int                             #              B: bins               

@dataclass
class RcfParams:
    channel_pairs: list[tuple[int, int]]
    bin_widths: np.ndarray                      # (B,)         B: bins
    bin_centers: np.ndarray                     # (B,)         L: image size (voxels)
    bin_masks: np.ndarray                       # (B, L, L, L)
    bin_weights: np.ndarray                     # (B,)
    image_size: int
    padded_image_size: int

type Rcf = np.ndarray | torch.Tensor            # (C, B)

def init_rcf_params(
        *,
        channel_pairs,
        image_length_voxels,
        padded_image_length_voxels=None,
        image_resolution_A,
        min_bin_width_A,
        max_radius_A,
) -> RcfParams:

    _, bin_widths, bin_centers, bin_masks = _make_rcf_bins(
            image_length_voxels,
            image_resolution_A,
            max_radius_A,
            min_bin_width_A,
    )
    window_overlap_grid = _make_window_overlap_grid(image_length_voxels)
    bin_weights = _bin_window_overlaps(bin_masks, window_overlap_grid)

    if padded_image_length_voxels is None:
        padding_voxels = int(ceil(max_radius_A / image_resolution_A))
        padded_image_length_voxels = next_fast_len(image_length_voxels + padding_voxels)

    return RcfParams(
            channel_pairs=channel_pairs,
            bin_widths=bin_widths,
            bin_centers=bin_centers,
            bin_masks=bin_masks,
            bin_weights=bin_weights,
            image_size=image_length_voxels,
            padded_image_size=padded_image_length_voxels,
    )
    
def init_rcf_accum(rcf_params: RcfParams) -> RcfAccum:
    C = len(rcf_params.channel_pairs)
    B = len(rcf_params.bin_widths)

    return RcfAccum(
            corr_histogram=np.zeros((C, B)),
            num_images=0,
    )

def load_rcf(path: Path) -> tuple[Rcf, RcfParams]:
    kwargs = dict(np.load(path))
    rcf = kwargs.pop('rcf')
    return rcf, RcfParams(**kwargs)

def save_rcf(path: Path, rcf: Rcf, rcf_params: RcfParams):
    np.savez(path, rcf=rcf, **asdict(rcf_params))

def update_rcf(rcf_accum: RcfAccum, rcf_params: RcfParams, images: np.ndarray):
    _accum_correlations(
            images,
            corr_histogram_accum=rcf_accum.corr_histogram,
            padded_image_size=rcf_params.padded_image_size,
            channel_pairs=rcf_params.channel_pairs,
            bin_masks=rcf_params.bin_masks,
    )
    rcf_accum.num_images += len(images)

def calc_rcf(rcf_accum: RcfAccum, rcf_params: RcfParams):
    return _normalize_correlations(
            corr_histogram=rcf_accum.corr_histogram,
            num_images=rcf_accum.num_images,
            bin_weights=rcf_params.bin_weights,
    )

def calc_rcf_distance_L2(rcf_ref: Rcf, rcf_test: Rcf, rcf_params: RcfParams):
    return _calc_l2_dist(
            rcf_ref, rcf_test,
            bin_widths=rcf_params.bin_widths,
    )


def _make_rcf_bins(image_length_voxels, image_resolution_A, max_radius_A, min_bin_width_A):
    L = np.arange(image_length_voxels, dtype=float) * image_resolution_A
    x, y, z = np.meshgrid(L, L, L)

    r_grid = np.sqrt(x**2 + y**2 + z**2)
    r_near = r_grid[r_grid <= max_radius_A]
    r_uniq = np.unique(r_near)
    r_bin_edges = r_uniq - np.diff(r_uniq, prepend=0) / 2
    r_bin_widths = np.diff(r_bin_edges, prepend=-np.inf)

    bin_edges = r_bin_edges[r_bin_widths > min_bin_width_A]
    bin_edges = np.concatenate([
            bin_edges,
            np.arange(bin_edges[-1], max_radius_A, min_bin_width_A)[1:],
    ])
    bin_widths = np.diff(bin_edges)

    N = len(bin_widths)
    bin_masks = np.empty((N, *r_grid.shape), dtype=bool)
    bin_centers = np.zeros(N)

    for i, (r_min, r_max) in enumerate(pairwise(bin_edges)):
        bin_masks[i] = np.logical_and(r_grid >= r_min, r_grid < r_max)
        bin_centers[i] = np.mean(r_grid[bin_masks[i]])

    return bin_edges, bin_widths, bin_centers, bin_masks

def _make_window_overlap_grid(image_length_voxels):
    idx_x, idx_y, idx_z = np.mgrid[
            0:image_length_voxels,
            0:image_length_voxels,
            0:image_length_voxels,
    ]
    return (
            (image_length_voxels - idx_x) *
            (image_length_voxels - idx_y) *
            (image_length_voxels - idx_z)
    )

def _bin_window_overlaps(bin_masks, window_overlap_grid):
    return np.array([
            window_overlap_grid[bin_masks[i]].sum()
            for i in range(len(bin_masks))
    ])

def _accum_correlations(
        images,
        *,
        corr_histogram_accum,
        padded_image_size,
        channel_pairs,
        bin_masks,
):
    assert all_equal(bin_masks.shape[1:])

    B = bin_masks.shape[0]
    L = bin_masks.shape[1]
    P = padded_image_size

    for img in images:
        F = [
                fftn(img[c], s=(P, P, P))
                for c in range(img.shape[0])
        ]

        for i, (c1, c2) in enumerate(channel_pairs):
            corr = np.real(ifftn(np.conj(F[c1]) * F[c2]))
            corr = corr[:L,:L,:L]

            for j in range(B):
                corr_histogram_accum[i,j] += np.sum(corr[bin_masks[j]])

def _normalize_correlations(*, corr_histogram, num_images, bin_weights) -> Rcf:
    xp = array_namespace(corr_histogram)
    return corr_histogram / num_images / xp.asarray(bin_weights)

def _calc_l2_dist(rcf_ref, rcf_test, *, bin_widths):
    xp = array_namespace(rcf_test)
    rcf_ref = xp.asarray(rcf_ref)
    bin_widths = xp.asarray(bin_widths)

    l2_dists = xp.sqrt(xp.sum(bin_widths * (rcf_test - rcf_ref)**2))
    return xp.sum(l2_dists)


class RcfL2(torchmetrics.Metric):

    @classmethod
    def from_npz(cls, path):
        rcf, rcf_params = load_rcf(path)
        return cls(rcf, rcf_params)


    def __init__(self, rcf_ref: Rcf, rcf_params: RcfParams):
        super().__init__()

        self.rcf_ref = rcf_ref
        self.rcf_params = rcf_params

        self.add_state(
                'corr_histogram',
                default=torch.zeros(rcf_ref.shape),
                dist_reduce_fx='sum',
        )
        self.add_state(
                'num_images',
                default=torch.tensor(0),
                dist_reduce_fx='sum',
        )

    def update(self, images):
        update_rcf(self, self.rcf_params, images)

    def compute(self):
        rcf_test = calc_rcf(self, self.rcf_params)
        return calc_rcf_distance_L2(self.rcf_ref, rcf_test, self.rcf_params)



def test_make_rcf_bins_3():
    from math import sqrt

    bin_edges, bin_widths, _, bin_masks = _make_rcf_bins(
            image_length_voxels=3,
            image_resolution_A=1,
            max_radius_A=2.5,
            min_bin_width_A=0.3,
    )

    expected_bin_edges = [
            0,
            1 / 2,
            (sqrt(2) - 1) / 2 + 1,
            (sqrt(3) - sqrt(2)) / 2 + sqrt(2),
    ]
    expected_bin_widths = [
            expected_bin_edges[1] - expected_bin_edges[0],
            expected_bin_edges[2] - expected_bin_edges[1],
            expected_bin_edges[3] - expected_bin_edges[2],
            0.3,
            0.3,
            0.3,
    ]
    expected_bin_masks = [

            # bin 1: 0
            [[[1, 0, 0],
              [0, 0, 0],
              [0, 0, 0]],

             [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]],

             [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]],

            # bin 2: 1
            [[[0, 1, 0],
              [1, 0, 0],
              [0, 0, 0]],

             [[1, 0, 0],
              [0, 0, 0],
              [0, 0, 0]],

             [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]],

            # bin 3: sqrt(2)
            [[[0, 0, 0],
              [0, 1, 0],
              [0, 0, 0]],

             [[0, 1, 0],
              [1, 0, 0],
              [0, 0, 0]],

             [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]],

            # bin 4: sqrt(3)
            [[[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]],

             [[0, 0, 0],
              [0, 1, 0],
              [0, 0, 0]],

             [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]],

            # bin 5: 
            [[[0, 0, 1],
              [0, 0, 0],
              [1, 0, 0]],

             [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]],

             [[1, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]],

            # bin 6:
            [[[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0]],

             [[0, 0, 1],
              [0, 0, 1],
              [1, 1, 0]],

             [[0, 1, 0],
              [1, 1, 0],
              [0, 0, 0]]],

    ]

    np.testing.assert_allclose(bin_widths, expected_bin_widths)
    np.testing.assert_equal(bin_masks, expected_bin_masks)

    window_overlap_grid = _make_window_overlap_grid(3)
    bin_overlaps = _bin_window_overlaps(bin_masks, window_overlap_grid)

    expected_bin_overlaps = [
            27, 
            18 * 3,
            12 * 3,
            8,
            9 * 3,
            6 * 6 + 4 * 3,
    ]

    np.testing.assert_equal(bin_overlaps, expected_bin_overlaps)

def test_make_rcf_bins_19():
    _, bin_widths, _, bin_masks = _make_rcf_bins(
            image_length_voxels=19,
            image_resolution_A=1,
            max_radius_A=9,
            min_bin_width_A=0.2,
    )

    # Make sure that each bin has the minimum width.
    assert all(bin_widths + 1e-8 >= 0.2)

    # Make sure that no voxel is included in more than one mask.
    assert np.sum(bin_masks, axis=0).max() == 1

    # Make sure that none of the masks are empty
    assert all(np.sum(bin_masks, axis=(1,2,3)) > 0)

def test_make_window_overlap_grid_3():
    actual = _make_window_overlap_grid(3)
    expected = [
            [[27, 18,  9],
             [18, 12,  6],
             [ 9,  6,  3]],

            [[18, 12,  6],
             [12,  8,  4],
             [ 6,  4,  2]],

            [[ 9,  6,  3],
             [ 6,  4,  2],
             [ 3,  2,  1]]
    ]
    np.testing.assert_equal(actual, expected)

def test_calc_rcf_uniform_image():
    # The formula for calculating 1D correlation is:
    #
    #   (f . g)[n] = \sum_{m}^{N - n} f[m] g[m + n]
    #
    # Of course, we're working with 3D correlation, but the idea is the same.  
    # If $f$ and $g$ both have uniform density, we can ignore the $m$ and $m + 
    # n$ arguments, as the functions will have the same output for all inputs.  
    # That leaves:
    #
    #   (f . g)[n] = \sum_{m}^{N - n} f[] g[]
    #             = (N - n) f[] g[]
    #
    # The histograms we calculate should be normalized by the $N - n$ factor, 
    # so every bin should have a value of $f[] g[]$.

    rcf_params = init_rcf_params(
            channel_pairs=[(0, 0)],
            image_length_voxels=(L := 19),
            image_resolution_A=1,
            min_bin_width_A=0.2,
            max_radius_A=9,
    )
    rcf_accum = init_rcf_accum(rcf_params)

    uniform_image = np.ones((1, 1, L, L, L))
    update_rcf(rcf_accum, rcf_params, uniform_image)

    g = calc_rcf(rcf_accum, rcf_params)
    np.testing.assert_allclose(g, 1)

