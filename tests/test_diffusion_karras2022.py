import atompaint.diffusion.karras2022 as _ap
import torch.nn as nn
import torch.testing
import numpy as np
import parametrize_from_file as pff

from einops import repeat
from scipy.stats import kstest, norm
from dataclasses import dataclass

with_py = pff.Namespace()

def test_karras_diffusion_forward_self_cond():
    # The biggest weakness of this test is that it depends on the internal 
    # details of how the RNG is used.  I think the test is still worth having, 
    # because it exercises some complicated code, but the mock RNG will need to 
    # be updated if the implementation changes.

    class MockPrecond(nn.Module):

        def __init__(self):
            super().__init__()

            self.sigma_data = 1
            self.x_shape = (1, 2, 2, 2)
            self.self_condition = True

            self.calls = []

        def forward(self, x_noisy, sigma, *, x_self_cond=None, label=None):
            args = ForwardArgs(self.training, x_noisy, sigma, x_self_cond, label)
            self.calls.append(args)

            # Flip the sign so we can distinguish input from output.
            return -x_noisy

    class MockRng:

        def normal(self, loc, scale):
            # This is used to calculate sigma, which would normally affect the 
            # signal:noise ratio.  In the context of this test, though, the 
            # "noise" is always zero so the signal:noise ratio doesn't matter. 
            # I chose the following values to produce predictable sigma values, 
            # which I check for later.
            return torch.tensor([.1, .2, .3]).log()

        def choice(self, choices):
            return torch.tensor([True, False, True])

    @dataclass
    class ForwardArgs:
        training: bool
        x_noisy: torch.Tensor
        sigma: torch.Tensor
        x_self_cond: torch.Tensor
        label: torch.Tensor

    precond = MockPrecond()
    model = _ap.KarrasDiffusion(
            precond=precond,
            opt_factory=lambda params: None,
            gen_metrics={},
    )

    x = dict(
            x_clean=repeat(
                torch.tensor([1., 2., 3.]),
                'b -> b 1 2 2 2',
            ),

            # Use zero noise, because we want to be able to easily calculate 
            # what the expected arguments are for each forward pass.
            noise=torch.zeros(3, 1, 2, 2, 2),

            label=torch.eye(3),
            rng=MockRng(),
    )
    model.forward(x)

    assert len(precond.calls) == 2

    # The first call is in eval mode, and only includes the first and third 
    # images (the second was "randomly" excluded from self-conditioning).
    assert not precond.calls[0].training
    assert precond.calls[0].x_self_cond is None

    torch.testing.assert_close(
            precond.calls[0].x_noisy,
            repeat(torch.tensor([1., 3.]), 'b -> b 1 2 2 2'),
    )
    torch.testing.assert_close(
            precond.calls[0].sigma,
            repeat(torch.tensor([.1, .3]), 'b -> b 1 1 1 1'),
    )
    torch.testing.assert_close(
            precond.calls[0].label,
            torch.tensor([
                [1., 0., 0.],
                [0., 0., 1.],
            ]),
    )

    # The second call is in training mode, and includes all three images.
    assert precond.calls[1].training

    torch.testing.assert_close(
            precond.calls[1].x_noisy,
            repeat(torch.tensor([1., 2., 3.]), 'b -> b 1 2 2 2'),
    )
    torch.testing.assert_close(
            precond.calls[1].sigma,
            repeat(torch.tensor([.1, .2, .3]), 'b -> b 1 1 1 1'),
    )
    torch.testing.assert_close(
            precond.calls[1].x_self_cond,
            repeat(torch.tensor([-1., 0., -3.]), 'b -> b 1 2 2 2'),
    )
    torch.testing.assert_close(
            precond.calls[1].label,
            torch.tensor([
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
            ]),
    )

def test_karras_diffusion_forward_self_cond_empty_mask():
    # Check the case where self-conditioning is enabled, but the entire batch 
    # is randomly excluded from self-conditioning.  This needs to be handled 
    # with care, because the model will error if given empty input.
    #
    # This test may be fragile, because it depends on some internal 
    # implementation details of the RNG.  See above test for details.

    class MockPrecond(nn.Module):

        def __init__(self):
            super().__init__()

            self.sigma_data = 1
            self.x_shape = (1, 2, 2, 2)
            self.self_condition = True

            self.calls = []

        def forward(self, x_noisy, sigma, *, x_self_cond=None, label=None):
            assert x_noisy.shape[0] > 0
            assert x_self_cond.shape[0] > 0
            assert sigma.shape[0] > 0
            return -x_noisy

    class MockRng:

        def normal(self, loc, scale):
            return torch.tensor([.1, .2, .3]).log()

        def choice(self, choices):
            return torch.tensor([False, False, False])

    precond = MockPrecond()
    model = _ap.KarrasDiffusion(
            precond=precond,
            opt_factory=lambda params: None,
            gen_metrics={},
    )

    x = dict(
            x_clean=repeat(
                torch.tensor([1., 2., 3.]),
                'b -> b 1 2 2 2',
            ),

            # Use zero noise, because we want to be able to easily calculate 
            # what the expected arguments are for each forward pass.
            noise=torch.zeros(3, 1, 2, 2, 2),

            rng=MockRng(),
    )

    # Run a forward pass.  The actual checks made by this test are in the 
    # `MockPrecond.forward()` method.
    model.forward(x)

def test_generate_sigmas():
    # Check that every intermediate step in the diffusion process has the 
    # expected amount of noise.  See experiment #116 for a visualization of 
    # this test.

    class MockPrecond(nn.Module):

        def __init__(self):
            super().__init__()
            self.x_shape = (1, 8, 8, 8)
            self.label_dim = 0
            self.self_condition = False

        def forward(self, x_noisy, sigma, **kwargs):
            # This method is supposed to return a denoised version of 
            # `x_noisy`.  So for a model trained entirely on empty images, this 
            # would actually be the ideal implementation.
            return torch.zeros_like(x_noisy)

    precond = MockPrecond()
    precond.eval()

    params = _ap.GenerateParams(
            # Add some churn, so we're not just reusing the same noise at every 
            # iteration.
            S_churn=10,
    )
    rng = np.random.default_rng(0)

    img, traj = _ap.generate(
            precond=precond,
            params=params,
            num_images=1,
            rng=rng,
            record_trajectory=True,
    )

    # This isn't much of a test, but when we have a model that always outputs 
    # the same thing, the `generate()` function should produce that thing.
    torch.testing.assert_close(img, torch.zeros(1, 1, 8, 8, 8))

    img_sigma_pairs = [
            ('x_noisy_σ1', 'σ1'),
            ('x_noisy_σ2', 'σ2'),
            ('x_noisy_σ3_1st_order', 'σ3'),
            ('x_noisy_σ3_2nd_order', 'σ3'),
    ]

    for row in traj.iter_rows(named=True):
        for img_key, sigma_key in img_sigma_pairs:
            if row[img_key] is None:
                continue
            if row[sigma_key] == 0:
                continue

            ks = kstest(
                    row[img_key].numpy().reshape(-1),
                    norm(loc=0, scale=row[sigma_key]).cdf,
            )

            # `p > 0.05` is an arbitrary threshold.  I applied the Bonferroni 
            # correction w.r.t. the number of iterations, but not the number of 
            # images produced in each iteration.  Randomness is only added at 
            # the beginning of each iteration, so all of the images within an 
            # iteration are perfectly correlated.  Note that there's a 5% 
            # chance that this test would fail with an arbitrary random seed, 
            # but it passes with the current seed.
            assert ks.pvalue > 0.05 / traj.height

def test_generate_max_batch_size():

    class MockPrecond(nn.Module):

        def __init__(self):
            super().__init__()
            self.x_shape = (1, 8, 8, 8)
            self.label_dim = 0
            self.self_condition = False

        def forward(self, x_noisy, sigma, **kwargs):
            assert x_noisy.shape[0] <= 3
            return torch.zeros_like(x_noisy)

    precond = MockPrecond()
    precond.eval()

    params = _ap.GenerateParams(
            max_batch_size=3,
    )
    rng = np.random.default_rng(0)

    img, traj = _ap.generate(
            precond=precond,
            params=params,
            num_images=10,
            rng=rng,
            record_trajectory=True,
    )

    assert img.shape == (10, 1, 8, 8, 8)

def test_inpaint_sigmas():
    # Check that every intermediate step in the inpainting process has the 
    # expected amount of noise.  See experiment #116 for a visualization of 
    # this test.

    class MockPrecond(nn.Module):

        def __init__(self):
            super().__init__()
            self.x_shape = (1, 8, 8, 8)
            self.label_dim = 0
            self.self_condition = False

        def forward(self, x_noisy, sigma, **kwargs):
            # This method is supposed to return a denoised version of 
            # `x_noisy`.  So for a model trained entirely on empty images, this 
            # would actually be the ideal implementation.
            return torch.zeros_like(x_noisy)

    precond = MockPrecond()
    precond.eval()

    x_known = torch.zeros(1, 1, 8, 8, 8)

    # Mask out half the image.  Hopefully this will make it as easy as possible 
    # to detect problems in both the masked and unmasked regions.
    mask = torch.zeros(1, 1, 8, 8, 8)
    mask[:, :, :4] = 1

    params = _ap.InpaintParams(
            # Add some churn, so we're not just reusing the same noise at every 
            # iteration.
            S_churn=10,
    )
    rng = np.random.default_rng(0)

    img, traj = _ap.inpaint(
            precond=precond,
            x_known=x_known,
            mask=mask,
            params=params,
            rng=rng,
            record_trajectory=True,
    )

    # This isn't much of a test, but when we have a model that always outputs 
    # the same thing, the `generate()` function should produce that thing.
    torch.testing.assert_close(img, torch.zeros(1, 1, 8, 8, 8))

    img_sigma_pairs = [
            ('x_noisy_σ1_before_mask', 'σ1'),
            ('x_noisy_σ1_after_mask', 'σ1'),
            ('x_noisy_σ2', 'σ2'),
            ('x_noisy_σ3_1st_order', 'σ3'),
            ('x_noisy_σ3_2nd_order', 'σ3'),
    ]

    for row in traj.iter_rows(named=True):
        for img_key, sigma_key in img_sigma_pairs:
            if row[img_key] is None:
                continue
            if row[sigma_key] == 0:
                continue

            ks = kstest(
                    row[img_key].numpy().reshape(-1),
                    norm(loc=0, scale=row[sigma_key]).cdf,
            )

            # `p > 0.05` is an arbitrary threshold.  I applied the Bonferroni 
            # correction w.r.t. the number of iterations, but not the number of 
            # images produced in each iteration.  Randomness is only added at 
            # the beginning of each iteration, so all of the images within an 
            # iteration are perfectly correlated.  Note that there's a 5% 
            # chance that this test would fail with an arbitrary random seed, 
            # but it passes with the current seed.
            assert ks.pvalue > 0.05 / traj.height

def test_calc_sigmas():
    params = _ap.GenerateParams(
            noise_steps=5,
            sigma_max=80,
            sigma_min=1,
            rho=7,
    )
    sigmas = _ap._calc_sigma_schedule(params)
    expected = torch.tensor([
            80.00,
            33.66,
            12.53,
             3.97,
             1.00,
             0.00,
    ])

    torch.testing.assert_close(sigmas, expected, atol=0.01, rtol=0)

@pff.parametrize(
        schema=pff.cast(
            num_images=int,
            max_batch_size=with_py.eval,
            expected=with_py.eval,
        ),
)
def test_get_batch_indices(num_images, max_batch_size, expected):
    ij = list(_ap._get_batch_indices(num_images, max_batch_size))
    assert ij == expected

