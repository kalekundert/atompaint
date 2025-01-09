import atompaint.autoencoders.vae as _ap
import torch
import numpy as np

from hypothesis import given, assume
from hypothesis.strategies import composite, floats, one_of, just
from hypothesis.extra.numpy import arrays
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence

@composite
def mean_std_tensors(draw):
    shape = draw(one_of(just((2,)), just((2, 2)), just((2, 2, 2))))

    def finite_f32_arrays(**float_kwargs):
        return arrays(
            dtype=np.float32,
            shape=shape,
            elements=floats(
                width=32,
                allow_nan=False,
                allow_infinity=False,
                **float_kwargs,
            ),
        )

    mean = draw(finite_f32_arrays())
    std = draw(finite_f32_arrays(min_value=np.float32(1e-5)))

    return torch.from_numpy(mean), torch.from_numpy(std)

def test_kl_divergence():
    d = _ap.kl_divergence_vs_std_normal

    mean = torch.zeros(2)
    std = torch.ones(2)

    assert d(mean, std).item() == 0
    assert d(mean + 1, std).item() < d(mean + 2, std).item()
    assert d(mean, std * 2).item() < d(mean, std * 3).item()

@given(mean_std_tensors())
def test_kl_divergence_torch(mean_std):
    mean, std = mean_std
    n = mean.nelement()

    N0 = MultivariateNormal(torch.zeros(n), torch.eye(n))
    N1 = MultivariateNormal(mean.flatten(), torch.diag(std.flatten()))

    kl_ap = _ap.kl_divergence_vs_std_normal(mean, std)

    # Note that `N1` and `N0` are swapped relative to what you might expect; 
    # the standard normal is not the reference distribution.  This is how 
    # variational inference is derived, though.
    kl_torch = kl_divergence(N1, N0)

    # Hypothesis finds some extreme cases where one of the KL divergences is 
    # infinite and the other is just very large.  This isn't a problem, because 
    # we're not trying to exactly reproduce the torch implementation, but it 
    # will cause the following assertion to fail.  So we skip these cases.
    assume(kl_ap.isfinite().all() == kl_torch.isfinite().all())

    torch.testing.assert_close(kl_ap, kl_torch)
