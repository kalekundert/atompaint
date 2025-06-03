import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt, pi

_FIRST_HERMITE_COEFF_A = sqrt(2) * pi**(-1/4)
_FIRST_HERMITE_COEFF_B = 1 / sqrt(2)

class FirstHermite(nn.Module):

    def __init__(self, a=_FIRST_HERMITE_COEFF_A, b=_FIRST_HERMITE_COEFF_B):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))

    def forward(self, x):
        return self.a * x * torch.exp(-(self.b * x)**2)

class LinearMinusFirstHermite(nn.Module):

    def __init__(self, a=_FIRST_HERMITE_COEFF_A, b=_FIRST_HERMITE_COEFF_B):
        super().__init__()
        self.first_hermite = FirstHermite(a, b)

    def forward(self, x):
        return x - self.first_hermite(x)

def first_hermite(
        x,
        *,
        a=_FIRST_HERMITE_COEFF_A,
        b=_FIRST_HERMITE_COEFF_B,
):
    return a * x * torch.exp(-(b * x)**2)

def linear_minus_first_hermite(
        x,
        *,
        a=_FIRST_HERMITE_COEFF_A,
        b=_FIRST_HERMITE_COEFF_B,
):
    return x - first_hermite(x, a=a, b=b)

def leaky_hard_shrink(x, cutoff=2, slope=0.1):
    return F.hardshrink(x, cutoff) + slope * x

