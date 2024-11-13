import atompaint.autoencoders.sym_vae as _ap
import torch
import numpy as np
import pytest

from escnn.nn import FieldType, FourierFieldType, GeometricTensor
from escnn.gspaces import rot3dOnR3
from math import sqrt

def test_sym_mean_std_trivial():
    gspace = rot3dOnR3()
    field_type = FieldType(gspace, 2 * [gspace.trivial_repr])

    # For trivial representations, the norm calculation won't change any of the 
    # values; it'll just make the signs positive.  Since it's easy to calculate 
    # the expected result, we can use a tensor that's bigger than the ones we 
    # work out by hand in other tests.

    x = torch.randn(1, 2, 2, 2, 2)
    x_geom = GeometricTensor(x, field_type)
    x_expected = x.clone().reshape(1, 2, 1, 2, 2, 2)
    x_expected[:, 1] = x_expected[:, 1].abs()

    f = _ap.SymMeanStd(field_type)
    y = f(x_geom)

    torch.testing.assert_close(x_expected, y)

def test_sym_mean_std_fourier():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    field_type = FourierFieldType(gspace, 2, so3.bl_irreps(1))

    x = torch.arange(20).float().reshape(1, 20, 1, 1, 1)
    x_geom = GeometricTensor(x, field_type)

    f = _ap.SymMeanStd(field_type)
    y = f(x_geom)

    torch.testing.assert_close(
            y,
            torch.tensor([
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                10 * [np.linalg.norm([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])],
            ]).float().reshape(1, 2, 10, 1, 1, 1),
    )

def test_sym_mean_std_fourier_unpack():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    field_type = FourierFieldType(gspace, 2, so3.bl_irreps(1), unpack=True)

    x = torch.arange(20).float().reshape(1, 20, 1, 1, 1)
    x_geom = GeometricTensor(x, field_type)

    f = _ap.SymMeanStd(field_type)
    y = f(x_geom)

    def norm(x, y, z):
        return 3 * [sqrt(x**2 + y**2 + z**2)]

    torch.testing.assert_close(
            y,
            torch.tensor([
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
                [10, *norm(11, 12, 13), *norm(14, 15, 16), *norm(17, 18, 19)],
            ]).float().reshape(1, 2, 10, 1, 1, 1),
    )

def test_sym_mean_std_err_odd():
    gspace = rot3dOnR3()
    field_type = FieldType(gspace, 3 * [gspace.trivial_repr])

    with pytest.raises(ValueError):
        _ap.SymMeanStd(field_type)

def test_sym_mean_std_err_diff_halves():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    field_type = FourierFieldType(gspace, 1, so3.bl_irreps(1), unpack=True)

    with pytest.raises(ValueError):
        _ap.SymMeanStd(field_type)
