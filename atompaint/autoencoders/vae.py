import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import chain
from dataclasses import dataclass
from functools import cached_property

from atompaint.type_hints import OptFactory

class VariationalAutoencoder(L.LightningModule):
    """
    Train a variational autoencoder.
    """

    def __init__(
            self, *,
            encoder: nn.Module,
            decoder: nn.Module,
            opt_factory: OptFactory,
            min_std: float = 1e-5,
    ):
        """
        Arguments:
            encoder:
                A torch module that converts input to a lower-dimensional 
                latent representation.  The shape of the latent representation 
                must be: (B, 2, *), where B is the batch size and * is any 
                number of dimensions of any size.  The two indices in the 
                second dimension will be used as the mean and standard 
                deviation, respectively, when sampling the latent variable.

            decoder:
                A torch module that reconstructs the input from the latent 
                representation created by the encoder.  The input to the 
                decoder will be a tensor of size (B, *), where each element is 
                sampled from a normal distribution parameterized by the 
                encoder.  The output for the decoder must be the same shape as 
                the input to the encoder.

            opt_factory:
                A callable that will return an optimizer, given an iterable of 
                model parameters.  This is typically either a 
                `torch.optim.Optimizer` class, or a partial function that 
                returns such an optimizer.

            min_std:
                The smallest standard deviation that will be allowed.  
                Specifically, the standard deviations are calculated as 
                `softmax(x) + min_std`, where x is part of the encoder output. 
                The `softmax` ensures that the standard deviations are 
                grater than 0, and this argument ensures that they aren't too 
                close to zero.
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = opt_factory(
                chain(encoder.parameters(), decoder.parameters()),
        )
        self.min_std = min_std

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        img = x['image']
        rngs = x['rng']

        mean_std = self.encoder(img)

        mean, std = mean_std[:,0], mean_std[:,1]
        std = F.softplus(std) + self.min_std
        noise = rngs.normal(size=std.shape[1:]).float().to(std.device)
        latent = mean + noise * std
        
        img_pred = self.decoder(latent)

        loss = VaeLoss(
                data=F.mse_loss(img, img_pred, reduction='sum'),
                prior=kl_divergence_vs_std_normal(mean, std),
                beta=1,
        )
        return loss

    def training_step(self, x):
        return self._generic_step('train', x)

    def validation_step(self, x):
        return self._generic_step('val', x)

    def test_step(self, x):
        return self._generic_step('test', x)

    def _generic_step(self, step, x, on_epoch=True):
        loss = self.forward(x)
        self.log(f'{step}/loss', loss.total, on_epoch=on_epoch)
        self.log(f'{step}/loss/data', loss.data, on_epoch=on_epoch)
        self.log(f'{step}/loss/prior', loss.prior, on_epoch=on_epoch)
        self.log(f'{step}/loss/beta', loss.beta, on_epoch=on_epoch)
        return loss.total


@dataclass
class VaeLoss:
    data: float
    prior: float
    beta: float = 1

    @cached_property
    def total(self):
        return self.data + self.beta * self.prior

def make_vae_image_tensors(db, db_cache, rng, zone_id, *, img_params):
    from macromol_gym_unsupervised import make_unsupervised_image_sample

    x = make_unsupervised_image_sample(
            db, db_cache, rng, zone_id,
            img_params=img_params,
    )
    return dict(
            rng=x['rng'],
            image=x['image'],
    )

def kl_divergence_vs_std_normal(mean, std):
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    return 0.5 * torch.sum(std + mean**2 - 1 - torch.log(std))

