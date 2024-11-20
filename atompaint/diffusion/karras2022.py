from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import numpy as np
import sys

from atompaint.metrics.neighbor_loc import (
        NeighborLocAccuracy, FrechetNeighborLocDistance,
)
from einops import reduce
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from tqdm import trange
from math import sqrt

from typing import Optional
from atompaint.type_hints import OptFactory

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

    def __init__(
            self,
            model: KarrasPrecond,
            *,
            opt_factory: OptFactory,
            gen_params: Optional[GenerateParams] = None,
            frechet_ref_path: str | Path,
    ):
        super().__init__()

        self.model = model
        self.optimizer = opt_factory(model.parameters())

        # The neighbor-location-based metrics require batch sizes that are 
        # multiples of 12.
        self.gen_params = gen_params or GenerateParams()
        self.gen_params.batch_size = 12

        self.gen_metrics = {
                'accuracy': NeighborLocAccuracy(),
                'frechet_dist': FrechetNeighborLocDistance(),
        }
        self.gen_metrics['frechet_dist'].load_reference_stats(frechet_ref_path)

        # [Karras2022], Table 1.  This mean and standard deviation should lead 
        # to σ values in roughly the range [1.9e-3, 5.4e2].
        self.P_mean = -1.2
        self.P_std = 1.2

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        x_clean, noise, rngs = x
        d = x_clean.ndim - 2

        t_norm = rngs.normal(loc=self.P_mean, scale=self.P_std)
        t_norm = t_norm.to(dtype=torch.float32, device=x_clean.device)
        sigma = torch.exp(t_norm).reshape(-1, 1, *([1] * d))

        x_noisy = x_clean + sigma * noise
        x_pred = self.model(x_noisy, sigma)

        sigma_data = self.model.sigma_data
        weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
        loss = weight * (x_pred - x_clean)**2
        loss = reduce(loss, 'b c ... -> c ...', 'mean').sum()

        return loss

    def training_step(self, x):
        loss = self.forward(x)
        self.log('train/loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, x):
        loss = self.forward(x)
        self.log('val/loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        # Lightning takes care of putting the model in eval-mode and disabling 
        # gradients before this hook is invoked [1], so we don't need to do 
        # that ourselves.
        #
        # [1]: https://pytorch-lightning.readthedocs.io/en/1.7.2/common/lightning_module.html#hooks

        # We set the batch size to 12 in the constructor, so we're generating 
        # 12 × 32 = 384 images.  Each image allows 4 updates, for a total of 
        # 384 × 4 = 1536 updates.  In Experiment 97, I showed that this is 
        # about the smallest number needed to get a stable result.

        rng = np.random.default_rng(0)

        for i in trange(32, desc="Generative metrics", file=sys.stdout):
            x = generate(rng, self.model, self.gen_params)

            for metric in self.gen_metrics.values():
                metric.to(x.device)
                metric.update(x)

        for name, metric in self.gen_metrics.items():
            self.log(f'gen/{name}', metric.compute(), sync_dist=True)
            metric.reset()

    def test_step(self, x):
        loss = self.forward(x)
        self.log('test/loss', loss, on_epoch=True, sync_dist=True)
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
        with, it's easiest to directly predict $x$.
        """
        assert x_noisy.shape[1:] == self.x_shape
        sigma_data = self.sigma_data
        
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
        c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
        # [Karras2022] includes this log-transformation, but I don't think it 
        # makes sense.  See Experiment 83 for details.
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

    # The mean and standard deviation of the underlying dataset, if the model 
    # was trained on data where these parameters were normalized.  More 
    # prescriptively, these values should match the values of `normalize_mean` 
    # and `normalize_std` that were passed to `MacromolImageDiffusionData`.
    unnormalize_mean: float = 0
    unnormalize_std: float = 1

    clamp_low: float = 0
    clamp_high: float = 1

    batch_size: int = 1

@torch.no_grad()
def generate(
        rng,
        model: KarrasPrecond,
        params: GenerateParams,
        record_trajectory: bool = False,
):
    assert not model.training
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

        if t_next > 0:
            dx_dt2 = (x_next - model(x_next, t_next)) / t_next
            x_next = x_cur + (t_next - t_cur) * (dx_dt1 + dx_dt2) / 2

        if record_trajectory:
            traj[i] = x_next
            i += 1

        x_cur = x_next

    if record_trajectory:
        x_out = traj
    else:
        x_out = x_cur

    # On tensors this big, in-place operations are slightly faster.  It's a 
    # minor effect, though.
    x_out.mul_(params.unnormalize_std)
    x_out.add_(params.unnormalize_mean)
    x_out.clamp_(params.clamp_low, params.clamp_high)

    return x_out

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

