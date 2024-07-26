import lightning as L
import torch
from einops import reduce

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

    def __init__(self, model, *, noise_embedding, opt_factory):
        super().__init__()

        self.model = model
        self.noise_embedding = noise_embedding
        self.optimizer = opt_factory(model.parameters())

        # [Karras2022], Table 1
        self.P_mean = -1.2
        self.P_std = 1.2

        # Expt 65
        self.register_buffer(
                'sigma_data',
                torch.tensor([
                    0.068791,
                    0.036372,
                    0.038710,
                    0.006235,
                    0.006190,
                    0.001196,
                ]).reshape(-1, 1, 1, 1),
        )

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        x_clean, noise, t_uniform = x
        sigma_data = self.sigma_data

        # Determine how much noise to add.  The dataset provides us with a 
        # uniformly-distributed random value for this purpose.  To implement 
        # [Karras2022], however, we need a normally-distributed value.  We get 
        # this via the inverse CDF of the normal distribution.
        norm = torch.distributions.Normal(loc=self.P_mean, scale=self.P_std)
        t_norm = norm.icdf(t_uniform)
        sigma = torch.exp(t_norm).reshape(-1, 1, 1, 1, 1)

        x_noisy = x_clean + sigma * noise

        c_in = 1 / torch.sqrt(self.sigma_data**2 + sigma**2)
        c_out = sigma * sigma_data / torch.sqrt(sigma**2 + sigma_data**2)
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_noise = self.noise_embedding(sigma.flatten())

        x_pred = (
                c_skip * x_noisy +
                c_out * self.model(c_in * x_noisy, c_noise)
        )

        weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
        loss = (x_pred - x_clean)**2
        loss = reduce(loss, 'b c ... -> b c', 'mean')
        return torch.mean(weight * loss)

    def training_step(self, x):
        loss = self.forward(x)
        self.log('train/loss', loss, on_epoch=True)
        return loss

    def validation_step(self, x):
        loss = self.forward(x)
        self.log('train/loss', loss, on_epoch=True)
        return loss

    def test_step(self, x):
        loss = self.forward(x)
        self.log('train/loss', loss, on_epoch=True)
        return loss
