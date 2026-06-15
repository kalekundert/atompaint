import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from atompaint.type_hints import OptFactory
from macromol_gym_unsupervised import MakeSampleArgs

class Autoencoder(nn.Module):

    def __init__(
            self, *,
            encoder: nn.Module,
            decoder: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


class VariationalAutoencoder(L.LightningModule):
    """
    Train a variational autoencoder.
    """

    def __init__(
            self, *,
            encoder: nn.Module,
            decoder: nn.Module,
            opt_factory: OptFactory,
            std_fn=None,
            scale_factor: float = 1.0,
            kl_weight: float = 1.0,
            free_bits_per_dim: float = 0.0,
    
    ):
        """
        Arguments:
            encoder:
                A torch module that converts input to a lower-dimensional
                latent representation.  The shape of the latent representation
                must be: (B, 2, *), where B is the batch size and * is any
                number of dimensions of any size.  The two indices in the
                second dimension will be used as the mean and raw scale
                parameter, respectively.  The raw scale parameter is converted
                to a standard deviation by `std_fn`.

            decoder:
                A torch module that reconstructs the input from the latent
                representation created by the encoder.  The input to the
                decoder will be a tensor of size (B, *), where each element is
                sampled from a normal distribution parameterized by the
                encoder.  The output of the decoder must be the same shape as
                the input to the encoder.

            opt_factory:
                A callable that will return an optimizer, given an iterable of
                model parameters.

            std_fn:
                A function that converts the raw scale output of the encoder
                into a positive standard deviation.  Defaults to
                `std_from_log_var`.  Also available: `std_from_softplus`.
                Both functions accept a `min_std` keyword argument (default
                1e-5) that can be tuned via `functools.partial`.

            scale_factor:
                The input image is multiplied by this value before being passed
                to the encoder.  The decoder output is divided by this value to
                recover the unscaled reconstruction.  This can improve training
                by putting the input into a range where the loss landscape is
                easier to optimize.

            kl_weight:
                Weight applied to the KL-divergence term of the loss.
                Equivalent to the β parameter in β-VAE.

            free_bits_per_dim:
                Minimum KL contribution allowed per latent dimension.
                Dimensions below this threshold are clamped, which prevents
                the encoder from collapsing those dimensions to the prior.
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = opt_factory(self.parameters())
        self.std_fn = std_fn if std_fn is not None else std_from_log_var
        self.scale_factor = scale_factor
        self.kl_weight = kl_weight
        self.free_bits_per_dim = free_bits_per_dim

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, x):
        return self._generic_step('train', x, sample=True)

    def validation_step(self, x):
        return self._generic_step('val', x, sample=False)

    def test_step(self, x):
        return self._generic_step('test', x, sample=False)

    def _generic_step(self, step, x, *, sample):
        img = x['image']
        rngs = x['rng']

        img_scaled = img * self.scale_factor

        mean_std = self.encoder(img_scaled)
        mean, std = mean_std[:, 0], self.std_fn(mean_std[:, 1])

        if sample:
            noise = rngs.normal(size=std.shape[1:]).float().to(std.device)
            latent = mean + noise * std
        else:
            latent = mean

        img_pred_scaled = self.decoder(latent)
        img_pred = img_pred_scaled / self.scale_factor

        recon_loss = F.mse_loss(img_scaled, img_pred_scaled)
        prior_loss = self.kl_weight * kl_divergence_vs_std_normal(
                mean, std,
                free_bits_per_dim=self.free_bits_per_dim,
        )
        loss = recon_loss + prior_loss

        self.log(f'{step}/loss',        loss)
        self.log(f'{step}/loss/recon',  recon_loss)
        self.log(f'{step}/loss/prior',  prior_loss)
        self.log(f'{step}/kl_weight',   self.kl_weight)
        self.log(f'{step}/recon_mse',   F.mse_loss(img, img_pred))
        return loss


class ApplyScaleFactor(nn.Module):

    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor

def std_from_log_var(x, *, min_std=1e-5):
    return torch.exp(x / 2) + min_std

def std_from_softplus(x, *, min_std=1e-5):
    return F.softplus(x) + min_std


def make_vae_image_tensors(sample: MakeSampleArgs, *, img_params):
    from macromol_gym_unsupervised import make_unsupervised_image_sample

    x = make_unsupervised_image_sample(sample, img_params=img_params)

    return dict(
            rng=x['rng'],
            image=x['image'],
    )

def kl_divergence_vs_std_normal(mean, std, *, free_bits_per_dim=0.0, dim=None):
    """
    Calculate the KL-divergence between the given diagonal Gaussian 
    distribution and a Gaussian distribution with a mean of 0 and a standard 
    deviation of 1.

    The *dim* argument specifies over which dimensions the KL-divergence is 
    summed.  It is averaged over all other dimensions.  The default is 
    appropriate for 3D images with a batch dimension.

    The *free_bits_per_dim* argument implements the "free bits" heuristic from 
    [Kingma2016].  The idea is to clamp the per-dimension KL to a minimum 
    value, so the encoder is never penalized for using a latent dimension to 
    encode information.  Without this, the optimizer can reduce the KL term to 
    zero by ignoring some dimensions entirely (posterior collapse).  The name 
    comes from the information-theoretic interpretation: each latent dimension 
    is allowed to carry up to *free_bits_per_dim* nats of information "for 
    free", i.e. without contributing to the loss.

    [Kingma2016]: https://arxiv.org/abs/1606.04934
    """
    if dim is None:
        dim = list(range(1, mean.dim()))

    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    var = std ** 2
    kl_per_dim = 0.5 * (var + mean ** 2 - 1 - torch.log(var))

    if free_bits_per_dim > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits_per_dim)

    return torch.mean(torch.sum(kl_per_dim, dim=dim))
