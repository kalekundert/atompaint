import atompaint.nonlinearities as _ap
import torch

from atompaint.field_types import make_fourier_field_type
from escnn.nn import GeometricTensor, FourierPointwise
from escnn.gspaces import rot3dOnR3
from torch.optim import SGD
from more_itertools import one

def test_first_hermite():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    ift_grid = so3.grid(type='thomson_cube', N=1)

    in_type = one(make_fourier_field_type(
        gspace,
        channels=1,
        max_frequency=1,
    ))

    a0, b0 = 1, 1

    fh = _ap.FirstHermite(a=a0, b=b0)
    f = FourierPointwise(in_type, ift_grid, function=fh)

    opt = SGD(f.parameters(), lr=1)

    x = GeometricTensor(torch.randn(1, 10, 1, 1, 1), in_type)
    y = f(x).tensor

    loss = torch.sum(y)
    loss.backward()
    opt.step()

    # Make sure that the parameters are updated by the optimizer.
    assert fh.a.item() != a0
    assert fh.b.item() != b0

def test_linear_minus_first_hermite():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    ift_grid = so3.grid(type='thomson_cube', N=1)

    in_type = one(make_fourier_field_type(
        gspace,
        channels=1,
        max_frequency=1,
    ))

    a0, b0 = 1, 1

    lmfh = _ap.LinearMinusFirstHermite(a=a0, b=b0)
    f = FourierPointwise(in_type, ift_grid, function=lmfh)

    opt = SGD(f.parameters(), lr=1)

    x = GeometricTensor(torch.randn(1, 10, 1, 1, 1), in_type)
    y = f(x).tensor

    loss = torch.sum(y)
    loss.backward()
    opt.step()

    # Make sure that the parameters are updated by the optimizer.
    assert lmfh.first_hermite.a.item() != a0
    assert lmfh.first_hermite.b.item() != b0
