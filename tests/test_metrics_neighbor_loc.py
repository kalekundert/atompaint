import torch
import atompaint.metrics.neighbor_loc as _ap
import numpy as np

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

    metric = _ap.NeighborLocAccuracy()

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

