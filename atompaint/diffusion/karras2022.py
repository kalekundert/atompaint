import lightning as L
import torch
import torch.nn as nn

from einops import reduce
from dataclasses import dataclass
from itertools import pairwise
from math import sqrt

# TODO:
# - Save exponential moving averages of model weights [1].  This is apparently 
#   an important hyperparameter.
#
#   [1] https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/

class KarrasDiffusion(L.LightningModule):
    """
    A diffusion model inspired by (but not exactly identical to) the EDM 
    frameworks described in [Karras2022]_ and [Karras2023]_.
    """

    def __init__(self, model, *, opt_factory):
        super().__init__()

        self.model = model
        self.optimizer = opt_factory(model.parameters())

        # [Karras2022], Table 1
        self.P_mean = -1.2
        self.P_std = 1.2

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        x_clean, noise, t_uniform = x
        d = x_clean.ndim - 2

        # Determine how much noise to add.  The dataset provides us with a 
        # uniformly-distributed random value for this purpose.  To implement 
        # [Karras2022], however, we need a normally-distributed value.  We get 
        # this via the inverse CDF of the normal distribution.
        norm = torch.distributions.Normal(loc=self.P_mean, scale=self.P_std)
        t_norm = norm.icdf(t_uniform)
        sigma = torch.exp(t_norm).reshape(-1, 1, *([1] * d))

        x_noisy = x_clean + sigma * noise
        x_pred = self.model(x_noisy, sigma)

        sigma_data = self.model.sigma_data
        weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
        loss = weight * (x_pred - x_clean)**2

        # The reference code divides by the batch size
        #loss = loss.sum().mul(1/64)

        # This is another way to implement the same calculation as in the 
        # reference code; here the division by the batch size is implicit.
        loss = reduce(loss, 'b c ... -> c ...', 'mean').sum()

        # loss = (x_pred - x_clean)**2
        # loss = reduce(loss, 'b c ... -> b c', 'mean')
        # loss = torch.mean(weight * loss)

        # debug(loss.sum(), weight.sum())
        # raise SystemExit

        return loss

    def training_step(self, x):
        loss = self.forward(x)
        self.log('train/loss', loss, on_epoch=True)
        return loss

    def validation_step(self, x):
        loss = self.forward(x)
        self.log('val/loss', loss, on_epoch=True)
        return loss

    def test_step(self, x):
        loss = self.forward(x)
        self.log('test/loss', loss, on_epoch=True)
        return loss

class KarrasPrecond(nn.Module):
    
    def __init__(
            self,
            model: nn.Module,
            *,
            sigma_data: float,
            x_shape: list[int],
    ):
        super().__init__()

        self.model = model
        self.x_shape = x_shape

        d = len(x_shape) - 1
        self.register_buffer(
                'sigma_data',
                torch.tensor(sigma_data).reshape(-1, *([1] * d)).float(),
                persistent=False,
        )

    def forward(self, x_noisy, sigma):
        """
        Output a denoised version of $x_\textrm{noisy}$.

        The underlying model will actually predict a mixture of the noise and 
        the denoised image, in a ratio that depends on the noise level.  The 
        purpose is to avoid dramatically scaling the model outputs.  When 
        $x_\textrm{noisy}$ is dominated by noise, its easiest to predict the 
        noise directly.  When $x_\textrm{noisy}$ is nearly noise-free to begin 
        with, it easiest to directly predict $x$.
        """
        assert x_noisy.shape[1:] == self.x_shape
        sigma_data = self.sigma_data
        
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        #c_noise = sigma.log() / 4
        c_noise = sigma

        F_x = self.model(c_in * x_noisy, c_noise.flatten())
        D_x = c_skip * x_noisy + c_out * F_x

        return D_x

@dataclass
class GenerateParams:
    time_steps: int = 18
    sigma_min: float = 0.002
    sigma_max: float = 80
    rho: float = 7

    S_churn: float = 0
    S_min: float = 0
    S_max: float = float('inf')

    batch_size: int = 1

@torch.no_grad
def generate(
        rng,
        model: KarrasPrecond,
        params: GenerateParams,
        record_trajectory: bool = False,
):
    device = next(model.parameters()).device

    # See Algorithm 2 from [Karras2022].
    t = _calc_sigma_schedule(params)
    x_shape = params.batch_size, *model.x_shape
    x_cur = rng.normal(scale=t[0], size=x_shape)
    x_cur = torch.from_numpy(x_cur).float().to(device)
    t = t.to(device)

    if record_trajectory:
        traj = torch.zeros(params.time_steps + 1, *x_shape).to(device)
        traj[0] = x_cur
        i = 1

    for t_cur, t_next in pairwise(t):
        t_cur, x_cur = _add_churn(rng, t_cur, x_cur, params)

        dx_dt1 = (x_cur - model(x_cur, t_cur)) / t_cur
        x_next = x_cur + (t_next - t_cur) * dx_dt1

        # import matplotlib.pyplot as plt
        # from einops import rearrange

        # for i, x in enumerate([x_cur, model(x_cur, t1), dx_dt1, x_next]):
        #     plt.subplot(3, 4, i+1)
        #     plt.imshow(
        #             rearrange(
        #                 x, 
        #                 'b c w h -> (b c w) (h)',
        #             )
        #     )
        #     plt.colorbar()

        if t_next > 0:
            dx_dt2 = (x_next - model(x_next, t_next)) / t_next
            x_next = x_cur + (t_next - t_cur) * (dx_dt1 + dx_dt2) / 2

        # for i, x in enumerate([model(x_next, t2), dx_dt2, x_next]):
        #     plt.subplot(3, 4, i+6)
        #     plt.imshow(
        #             rearrange(
        #                 x, 
        #                 'b c w h -> (b c w) (h)',
        #             )
        #     )
        #     plt.colorbar()

        if record_trajectory:
            traj[i] = x_next

            # plt.subplot(3, 4, 12)
            # plt.imshow(
            #         rearrange(
            #             traj[i], 
            #             'b c w h -> (b c w) (h)',
            #         )
            # )
            # plt.colorbar()

            i += 1

        #plt.show()

        x_cur = x_next

    if record_trajectory:
        return traj
    else:
        return x_cur

def _calc_sigma_schedule(params):
    n = params.time_steps
    i = torch.arange(n + 1)
    inv_rho = 1 / params.rho
    σ_max = params.sigma_max
    σ_min = params.sigma_min

    # See Equation 5 from [Karras2022].
    sigmas = (
            σ_max ** inv_rho +
            (i / (n - 1)) * (σ_min ** inv_rho - σ_max ** inv_rho)
    ) ** params.rho
    sigmas[-1] = 0

    return sigmas

def _add_churn(rng, t_cur, x_cur, params):
    if params.S_min <= t_cur <= params.S_max:
        gamma = min(params.S_churn / params.time_steps, sqrt(2) - 1)
    else:
        gamma = 0

    t_churn = t_cur * (1 + gamma)
    eps_churn = rng.normal(
            scale=sqrt(t_churn**2 - t_cur**2),
            size=x_cur.shape,
    )
    eps_churn = torch.from_numpy(eps_churn).float().to(x_cur.device)

    return t_churn, x_cur + eps_churn

