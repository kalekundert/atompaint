import atompaint.end_to_end.data as ap
import torch

def test_make_amino_acid_crops():
    # To simplify things, this test only uses 1 spatial dimension instead of 3.

    image = torch.tensor([
        [[ 1,  2,  3]],
        [[ 4,  5,  6]],
    ])
    aa_crops = [
            (0, slice(None), slice(0, 2)),
            (0, slice(None), slice(1, 3)),
            (1, slice(None), slice(0, 2)),
            (1, slice(None), slice(1, 3)),
    ]
    aa_channels = torch.tensor([
        [[1, 0]],
        [[0, 1]],
        [[1, 0]],
        [[0, 1]],
    ])

    x_crop = ap.make_amino_acid_crops(
            image=image,
            aa_crops=aa_crops,
            aa_channels=aa_channels,
    )
    x_expected = torch.tensor([
        [[1, 2], [1, 0]],
        [[2, 3], [0, 1]],
        [[4, 5], [1, 0]],
        [[5, 6], [0, 1]],
    ])
    torch.testing.assert_close(x_crop, x_expected)

def test_make_amino_acid_crops_where():
    x_pred = torch.tensor([
        [[ 1,  2,  3]],
        [[ 4,  5,  6]],
    ])
    x_clean = torch.tensor([
        [[ 7,  8,  9]],
        [[10, 11, 12]],
    ])
    use_x_pred = torch.tensor([True, False])

    aa_crops = [
            (0, slice(None), slice(0, 2)),
            (0, slice(None), slice(1, 3)),
            (1, slice(None), slice(0, 2)),
    ]
    aa_channels = torch.tensor([
        [[1, 0]],
        [[0, 1]],
        [[0, 1]],
    ])

    x_crop, use_x_pred = ap.make_amino_acid_crops_where(
            aa_crops=aa_crops,
            aa_channels=aa_channels,
            x_pred=x_pred,
            x_clean=x_clean,
            use_x_pred=use_x_pred,
    )
    x_expected = torch.tensor([
        [[ 1,  2], [1, 0]],
        [[ 2,  3], [0, 1]],
        [[10, 11], [0, 1]],
    ])

    torch.testing.assert_close(x_crop, x_expected)
    torch.testing.assert_close(use_x_pred, torch.tensor([True, True, False]))


