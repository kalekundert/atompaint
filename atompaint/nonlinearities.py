import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt, pi

_FIRST_HERMITE_COEFF = sqrt(2) * pi**(-1/4)

class FirstHermite(nn.Module):

    def __init__(self, a=_FIRST_HERMITE_COEFF, b=1/sqrt(2)):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))

    def forward(self, x):
        return self.a * x * torch.exp(-(self.b * x)**2)

class LinearMinusFirstHermite(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.first_hermite = FirstHermite(*args, **kwargs)

    def forward(self, x):
        return x - self.first_hermite(x)

def first_hermite(x):
    return _FIRST_HERMITE_COEFF * x * torch.exp(-x**2 / 2)

def linear_minus_first_hermite(x):
    return x - first_hermite(x)

def leaky_hard_shrink(x, cutoff=2, slope=0.1):
    return F.hardshrink(x, cutoff) + slope * x

