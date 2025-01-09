import atompaint.diffusion.karras2022 as _ap
import torch.nn as nn
import torch.testing

from einops import repeat
from dataclasses import dataclass

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

        def forward(self, x_noisy, sigma, *, x_prev=None, label=None):
            args = ForwardArgs(self.training, x_noisy, sigma, x_prev, label)
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
        x_prev: torch.Tensor
        label: torch.Tensor

    precond = MockPrecond()
    model = _ap.KarrasDiffusion(
            model=precond,
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
    assert precond.calls[0].x_prev is None

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
            precond.calls[1].x_prev,
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

        def forward(self, x_noisy, sigma, *, x_prev=None, label=None):
            assert x_noisy.shape[0] > 0
            assert x_prev.shape[0] > 0
            assert sigma.shape[0] > 0
            return -x_noisy

    class MockRng:

        def normal(self, loc, scale):
            return torch.tensor([.1, .2, .3]).log()

        def choice(self, choices):
            return torch.tensor([False, False, False])

    precond = MockPrecond()
    model = _ap.KarrasDiffusion(
            model=precond,
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


def test_calc_sigmas():
    params = _ap.GenerateParams(
            time_steps=5,
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

