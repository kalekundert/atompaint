import torchyield as ty
import torch.nn as nn

from functools import partial
from torch import Tensor
from einops import rearrange


class AsymMeanStd(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return rearrange(x, 'b (m c) ... -> b m c ...', m=2)


class AsymMeanOnly(nn.Module):
    """
    Wrap an AsymMeanStd encoder to return flattened latent means.

    The encoder tail (via AsymMeanStd) outputs (B, 2, C, ...) where index 0
    is means and index 1 is log-stds.  This wrapper extracts the means and
    flattens the spatial dimensions to produce a (B, C*...) feature vector.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, images: Tensor) -> Tensor:
        enc_out = self.encoder(images)  # (B, 2, C, ...)
        return enc_out[:, 0].flatten(1)  # (B, C, ...)


def conv_split_mean_std(in_channels, out_channels, **kwargs):
    yield nn.Conv3d(in_channels, out_channels, **kwargs)
    yield AsymMeanStd()


def make_resnet_vae(
        *,
        encoder_channels,
        decoder_channels=None, 
        block_repeats=1,
):
    from atompaint.autoencoders.vae import Autoencoder
    from atompaint.encoders import Encoder, Decoder, early_schedule, late_schedule
    from atompaint.encoders.asym_resnet import asym_resblock
    from atompaint.utils import partial_ch
    from multipartial import multipartial, put

    if decoder_channels is None:
        decoder_channels = encoder_channels[::-1]
        decoder_channels[0] //= 2

    return Autoencoder(
            encoder=Encoder(
                channels=encoder_channels,
                channel_schedule=early_schedule,
                head_factory=partial_ch(
                    ty.conv3_bn_relu_layer,
                    kernel_size=3,
                    padding=0,
                    stride=1,
                    bias=True,
                ),
                block_factories=multipartial[:, block_repeats](
                    asym_resblock,
                    downsample=put[:, 0](True, False),
                ),
                tail_factory=partial(
                    conv_split_mean_std,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
            ),
            decoder=Decoder(
                channels=decoder_channels,
                channel_schedule=late_schedule,
                head_factory=partial_ch(
                    ty.convT3_bn_relu_layer,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=True,
                ),
                block_factories=multipartial[:, block_repeats](
                    asym_resblock,
                    upsample=put[:, -1](True, False),
                ),
                tail_factory=partial_ch(
                    ty.convT3_layer,
                    kernel_size=3,
                    padding=0,
                    stride=1,
                ),
            ),
    )

def make_expt_145_vae():
    """
    Instantiate one of the top-performing encoder and decoder model 
    architectures from experiment 145.

    The purpose of this experiment was to identify the trade-off between model 
    width/depth and reconstruction accuracy.  However, I also ended up using 
    one of the top-performing models the create a quality metric based on 
    measuring the L2 distance between a generated image and its reconstruction.

    This specific model compresses a (4, 19, 19, 19) input image to a (16, 5, 
    5, 5) latent representation (2000 dimensions, ~14x compression).  The 
    relatively mild compression leads to relatively high-fidelity 
    reconstructions.
    """
    return make_resnet_vae(
            encoder_channels=[4, 48, 96, 192, 32],
            block_repeats=1,
            )

def load_expt_145_vae():
    from atompaint.autoencoders.vae import ApplyScaleFactor
    from atompaint.checkpoints import load_model_weights

    # The checkpoint encoder was wrapped as Sequential(Encoder, AsymMeanStd),
    # giving keys like 'encoder.0.N.*'.  The refactored model stores the
    # Encoder directly, so strip the extra '0.' level.
    def fix_keys(k):
        if k.startswith('encoder.0.'):
            return 'encoder.' + k[len('encoder.0.'):]
        return k

    vae = make_expt_145_vae()
    load_model_weights(
            model=vae,
            path='expt_145/channels=48-96-192-32;repeats=1;image-size=19A;job-id=hparams_07;epoch=91.ckpt',
            fix_keys=fix_keys,
            xxh32sum='92c24561',
    )

    vae.encoder = nn.Sequential(
            ApplyScaleFactor(20.0),
            vae.encoder,
    )
    vae.decoder = nn.Sequential(
            vae.decoder,
            ApplyScaleFactor(1/20.0),
    )

    return vae

def make_expt_157_vae():
    """
    Instantiate the encoder and decoder model architectures from experiment 157, 
    which aimed to create a model specifically for use with the Fréchet 
    autoencoder distance (FAED) metric.

    This model compresses a (4, 19, 19, 19) input image to a (4, 5, 5, 5)
    latent representation (500 dimensions, ~55x compression by volume).  The
    aggressive compression produces a compact latent space well-suited for
    estimating distributional distances, but at the cost of reconstruction
    accuracy.
    """
    return make_resnet_vae(
            encoder_channels=[4, 48, 96, 192, 8],
            block_repeats=2,
    )

def load_expt_157_vae():
    from atompaint.autoencoders.vae import ApplyScaleFactor
    from atompaint.checkpoints import load_model_weights

    vae = make_expt_157_vae()
    load_model_weights(
            model=vae,
            path='expt_157/channels=48-96-192-8;scale=8;kl-weight=2e-3;image-size=19A;job-id=hparams_06;epoch=48.ckpt',
            xxh32sum='d9def09e',
    )

    # 20x: dataset normalization (MMG_VOXEL_STD = 0.05)
    # 8x: FaedVAE scale_factor for hparams_06
    scale_factor = 160.0
    vae.encoder = nn.Sequential(
            ApplyScaleFactor(scale_factor),
            vae.encoder,
    )
    vae.decoder = nn.Sequential(
            vae.decoder,
            ApplyScaleFactor(1 / scale_factor),
    )

    return vae


