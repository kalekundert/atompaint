import torch
import torch.nn as nn
import atompaint.autoencoders.unet as ap

from escnn.nn import FieldType, GeometricTensor
from escnn.gspaces import rot3dOnR3

def test_pop_cat_skip_asym():
    x1 = torch.zeros(2, 3, 4, 4, 4)
    x2 = torch.zeros(2, 3, 4, 4, 4)
    t = torch.zeros(2)

    m = ap.PopCatSkip([nn.Identity()])

    y = m(x1, t, skips=[x2])

    assert y.shape == (2, 6, 4, 4, 4)

def test_pop_cat_skip_semisym():
    gspace = rot3dOnR3()
    ft = FieldType(gspace, 3 * [gspace.trivial_repr])

    x1 = torch.zeros(2, 3, 4, 4, 4)
    x2 = GeometricTensor(torch.zeros(2, 3, 4, 4, 4), ft)
    t = torch.zeros(2)

    m = ap.PopCatSkip([nn.Identity()])

    y = m(x1, t, skips=[x2])

    assert y.shape == (2, 6, 4, 4, 4)

def test_pop_cat_skip_sym():
    gspace = rot3dOnR3()
    ft = FieldType(gspace, 3 * [gspace.trivial_repr])

    x1 = GeometricTensor(torch.zeros(2, 3, 4, 4, 4), ft)
    x2 = GeometricTensor(torch.zeros(2, 3, 4, 4, 4), ft)
    t = torch.zeros(2)

    m = ap.PopCatSkip([nn.Identity()])

    y = m(x1, t, skips=[x2])

    assert y.shape == (2, 6, 4, 4, 4)

