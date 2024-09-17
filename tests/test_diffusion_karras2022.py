import atompaint.diffusion.karras2022 as _ap
import torch.testing

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

