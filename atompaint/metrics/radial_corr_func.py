import torch
import torchmetrics
import numpy as np

from scipy.fft import fftn, ifftn, next_fast_len
from array_api_compat import array_namespace
from math import ceil
from dataclasses import dataclass, asdict
from itertools import pairwise
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
    bin_labels: np.ndarray                      # (L, L, L)   -1 outside all bins
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

    _, bin_widths, bin_centers, bin_labels = _make_rcf_bins(
            image_length_voxels,
            image_resolution_A,
            max_radius_A,
            min_bin_width_A,
    )
    window_overlap_grid = _make_window_overlap_grid(image_length_voxels)
    bin_weights = _bin_window_overlaps(bin_labels, window_overlap_grid)

    if padded_image_length_voxels is None:
        padding_voxels = int(ceil(max_radius_A / image_resolution_A))
        padded_image_length_voxels = next_fast_len(image_length_voxels + padding_voxels)

    return RcfParams(
            channel_pairs=channel_pairs,
            bin_widths=bin_widths,
            bin_centers=bin_centers,
            bin_labels=bin_labels,
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
            bin_labels=rcf_params.bin_labels,
    )
    rcf_accum.num_images += len(images)

def calc_rcf(rcf_accum: RcfAccum, rcf_params: RcfParams):
    return _normalize_correlations(
            corr_histogram=rcf_accum.corr_histogram,
            num_images=rcf_accum.num_images,
            bin_weights=rcf_params.bin_weights,
    )

def calc_rcf_distance_L2(
        rcf_ref: Rcf,
        rcf_test: Rcf,
        rcf_params: RcfParams,
        *,
        include_autocorr_zero_bin: bool = False,
):
    C = len(rcf_params.channel_pairs)
    bin_widths = np.tile(rcf_params.bin_widths, (C, 1))

    if not include_autocorr_zero_bin:
        for i, (c1, c2) in enumerate(rcf_params.channel_pairs):
            if c1 == c2:
                bin_widths[i, 0] = 0

    return _calc_l2_dist(rcf_ref, rcf_test, bin_widths=bin_widths)


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
    bin_labels = np.full(r_grid.shape, -1, dtype=np.intp)
    bin_centers = np.zeros(N)

    for i, (r_min, r_max) in enumerate(pairwise(bin_edges)):
        mask = np.logical_and(r_grid >= r_min, r_grid < r_max)
        bin_labels[mask] = i
        bin_centers[i] = np.mean(r_grid[mask])

    return bin_edges, bin_widths, bin_centers, bin_labels

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

def _bin_window_overlaps(bin_labels, window_overlap_grid):
    B = int(bin_labels.max()) + 1
    valid = bin_labels.ravel() >= 0
    return np.bincount(
            bin_labels.ravel()[valid],
            weights=window_overlap_grid.ravel()[valid].astype(float),
            minlength=B,
    ).astype(int)

def _accum_correlations(
        images,
        *,
        corr_histogram_accum,
        padded_image_size,
        channel_pairs,
        bin_labels,
):
    B = corr_histogram_accum.shape[1]
    L = bin_labels.shape[0]
    P = padded_image_size

    valid = bin_labels.ravel() >= 0
    flat_labels = bin_labels.ravel()[valid]

    for img in images:
        F = [
                fftn(img[c], s=(P, P, P))
                for c in range(img.shape[0])
        ]

        for i, (c1, c2) in enumerate(channel_pairs):
            corr = np.real(ifftn(np.conj(F[c1]) * F[c2]))
            corr = corr[:L,:L,:L]
            corr_histogram_accum[i] += np.bincount(
                    flat_labels,
                    weights=corr.ravel()[valid],
                    minlength=B,
            )

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
    def from_npz(cls, path, *, include_autocorr_zero_bin=False):
        rcf, rcf_params = load_rcf(path)
        return cls(rcf, rcf_params, include_autocorr_zero_bin=include_autocorr_zero_bin)


    def __init__(
            self,
            rcf_ref: Rcf,
            rcf_params: RcfParams,
            *,
            include_autocorr_zero_bin: bool = False,
    ):
        super().__init__()

        self.rcf_ref = rcf_ref
        self.rcf_params = rcf_params
        self.include_autocorr_zero_bin = include_autocorr_zero_bin

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
        return calc_rcf_distance_L2(
                self.rcf_ref, rcf_test, self.rcf_params,
                include_autocorr_zero_bin=self.include_autocorr_zero_bin,
        )
