import torch

from atompaint.mask import MaskModule
from escnn.gspaces import rot3dOnR3
from escnn.nn import FieldType, GeometricTensor

def test_mask():
    gspace = rot3dOnR3()
    in_type = FieldType(gspace, [gspace.trivial_repr])

    # Use a very low standard deviation to make the transition between 
    # masked/unmasked more dramatic.
    mask = MaskModule(in_type, 5, margin=0, sigma=0.1)

    x = GeometricTensor(
            torch.ones(1, 1, 5, 5, 5),
            in_type,
    )
    y = mask(x)

    unmasked_indices = [
            (1, 1, 1),
            (1, 1, 3),
            (1, 3, 1),
            (1, 3, 3),
            (3, 1, 1),
            (3, 1, 3),
            (3, 3, 1),
            (3, 3, 3),
    ]
    masked_indices = [
            (0, 0, 0),
            (0, 0, 4),
            (0, 4, 0),
            (0, 4, 4),
            (4, 0, 0),
            (4, 0, 4),
            (4, 4, 0),
            (4, 4, 4),
    ]

    assert y.type == in_type

    for i,j,k in unmasked_indices:
        assert y.tensor[0, 0, i, j, k] > 1 - 1e5
    for i,j,k in masked_indices:
        assert y.tensor[0, 0, i, j, k] < 1e-5



