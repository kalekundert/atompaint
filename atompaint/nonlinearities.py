import torch
import torch.nn.functional as F

from math import sqrt, pi

_FIRST_HERMITE_COEFF = sqrt(2) * pi**(-1/4)

def first_hermite(x):
    return _FIRST_HERMITE_COEFF * x * torch.exp(-x**2 / 2)

def leaky_hard_shrink(x, cutoff=2, slope=0.1):
    return F.hardshrink(x, cutoff) + slope * x

