import numpy as np
import atompaint.metrics.radial_corr_func as _ap

def test_make_rcf_bins_3():
    from math import sqrt

    bin_edges, bin_widths, _, bin_labels = _ap._make_rcf_bins(
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
    expected_bin_labels = [
            [[ 0,  1,  4],
             [ 1,  2,  5],
             [ 4,  5, -1]],

            [[ 1,  2,  5],
             [ 2,  3,  5],
             [ 5,  5, -1]],

            [[ 4,  5, -1],
             [ 5,  5, -1],
             [-1, -1, -1]],
    ]

    np.testing.assert_allclose(bin_widths, expected_bin_widths)
    np.testing.assert_equal(bin_labels, expected_bin_labels)

    window_overlap_grid = _ap._make_window_overlap_grid(3)
    bin_overlaps = _ap._bin_window_overlaps(bin_labels, window_overlap_grid)

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
    _, bin_widths, _, bin_labels = _ap._make_rcf_bins(
            image_length_voxels=19,
            image_resolution_A=1,
            max_radius_A=9,
            min_bin_width_A=0.2,
    )

    # Make sure that each bin has the minimum width.
    assert all(bin_widths + 1e-8 >= 0.2)

    # Make sure that none of the bins are empty.
    B = len(bin_widths)
    assert all(np.bincount(bin_labels[bin_labels >= 0], minlength=B) > 0)

def test_make_window_overlap_grid_3():
    actual = _ap._make_window_overlap_grid(3)
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

    rcf_params = _ap.init_rcf_params(
            channel_pairs=[(0, 0)],
            image_length_voxels=(L := 19),
            image_resolution_A=1,
            min_bin_width_A=0.2,
            max_radius_A=9,
    )
    rcf_accum = _ap.init_rcf_accum(rcf_params)

    uniform_image = np.ones((1, 1, L, L, L))
    _ap.update_rcf(rcf_accum, rcf_params, uniform_image)

    g = _ap.calc_rcf(rcf_accum, rcf_params)
    np.testing.assert_allclose(g, 1)

