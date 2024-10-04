import torch
import numpy as np

from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy
from macromol_dataframe import Coords
from dataclasses import dataclass

class NeighborLocAccuracy(MulticlassAccuracy):

    def __init__(self):
        from atompaint.classifiers.neighbor_loc import load_default_15A_model
        from macromol_gym_pretrain.geometry import cube_faces

        super().__init__(num_classes=6)

        self.classifier = load_default_15A_model()
        self.view_params = ViewParams(
                direction_candidates=cube_faces(),
                length_voxels=15,
                padding_voxels=3,
        )

    def update(self, x):
        b, c, sx, sy, sz = x.shape
        assert sx == sy == sz == 33

        x, y = _extract_view_pairs(x, self.view_params)
        y_hat = self.classifier(x)

        super().update(y_hat, y)

class FrechetNeighborLocDistance(Metric):
    # I can use FID with almost no modification.  I just need to take a 
    # neighbor location classifier trained on 33Å inputs, and use the final 
    # layer right before the two encoder outputs are concatenated.
    pass

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
    assert w == h == d == 2 * L + view_params.padding_voxels

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

            slices_a = _get_slices(view_ai, view_params)
            slices_b = _get_slices(view_ba + view_ai, view_params)

            x[ij,0] = imgs[i][:, *slices_a]
            x[ij,1] = imgs[i][:, *slices_b]
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
    return [slice_map[x] for x in bool_vec]


