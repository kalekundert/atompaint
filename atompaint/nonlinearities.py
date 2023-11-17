import torch.nn.functional as F
import numpy as np

from escnn.nn import EquivariantModule, FieldType, GeometricTensor
from math import sqrt, pi

_FIRST_HERMITE_COEFF = sqrt(2) * pi**(-1/4)

class SetFourierFieldType(EquivariantModule):
    """
    Some modules output field types that are compatible with the Fourier 
    transform modules, but that aren't `FourierFieldType` instances.  This 
    module replaces the output field with a `FourierFieldType`, after doing 
    some checks.
    """

    def __init__(self, in_type, out_type):
        super().__init__()

        self.in_type = in_type
        self.out_type = out_type

        # The user is responsible for making sure that the input type of `x` is 
        # genuinely that same as the given output type.  Here we just do a few 
        # sanity checks to catch mistakes.
        assert in_type.size == self.out_type.size
        assert in_type.irreps == self.out_type.irreps
        assert np.allclose(
            in_type.change_of_basis.toarray(), 
            np.eye(in_type.size),
        )

    def forward(self, x: GeometricTensor):
        assert x.type == self.in_type
        x.type = self.out_type
        return x

    def evaluate_output_shape(self, input_shape):
        return input_shape


def add_gates(in_type):
    gspace = in_type.gspace
    group = in_type.fibergroup
    rho = in_type.representations
    gates = len(rho) * [group.trivial_representation]
    return FieldType(gspace, [*gates, *rho])


def first_hermite(x):
    return _FIRST_HERMITE_COEFF * x * torch.exp(-x**2 / 2)

def leaky_hard_shrink(x, cutoff=2, slope=0.1):
    return F.hardshrink(x, cutoff) + slope * x
