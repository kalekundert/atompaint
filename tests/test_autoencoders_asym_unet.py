import torch
import torch.nn as nn
import atompaint.autoencoders.asym_unet as ap
import parametrize_from_file as pff
import pytest

from atompaint.conditioning import SinusoidalEmbedding
from multipartial import multipartial
from torchtest import assert_vars_change
from test_conditioning import ModuleWrapper, InputWrapper

with_py = pff.Namespace()
with_ap = pff.Namespace('from atompaint.autoencoders.asym_unet import *')

@pytest.mark.parametrize(
        'kwargs', [
            dict(),
            dict(skip_algorithm='add'),
            dict(allow_self_cond=True),
        ],
)
def test_asym_unet_block_channels(kwargs):

    def head_factory(in_channels, out_channels):
        yield nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                bias=False,
        )
        yield nn.BatchNorm3d(out_channels)
        yield nn.ReLU()

    def tail_factory(in_channels, out_channels):
        yield nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=0,
        )

    def block_factory(in_channels, out_channels, cond_dim):
        yield from conv_attn_factory(
                in_channels,
                out_channels,
                cond_dim=cond_dim,
        )

    def latent_factory(channels, cond_dim):
        yield from conv_attn_factory(
                in_channels=channels,
                out_channels=channels,
                cond_dim=cond_dim,
                attention=True,
        )

    def conv_attn_factory(in_channels, out_channels, cond_dim, attention=False):
        yield ap.AsymConditionedConvBlock(
                in_channels,
                out_channels,
                cond_dim=cond_dim,
        )

        if attention:
            yield ap.AsymAttentionBlock(
                    out_channels,
                    num_heads=2,
                    channels_per_head=out_channels // 2,
            )

    def downsample_factory(channels):
        return nn.MaxPool3d(kernel_size=2)

    def upsample_factory(channels):
        return nn.Upsample(scale_factor=2, mode='trilinear')

    def noise_embedding(cond_dim):
        yield SinusoidalEmbedding(
                out_dim=cond_dim,
                min_wavelength=0.1,
                max_wavelength=100,
        )
        yield nn.Linear(cond_dim, cond_dim)
        yield nn.ReLU()

    unet = ap.AsymUNet(
            channels=[1, 2, 3, 4],
            head_factory=head_factory,
            tail_factory=tail_factory,
            block_factories=multipartial[1,2](block_factory),
            latent_factory=latent_factory,
            downsample_factory=downsample_factory,
            upsample_factory=upsample_factory,
            cond_dim=6,
            noise_embedding=noise_embedding,
            **kwargs,
    )

    x = torch.randn(2, 1, 8, 8, 8)
    y = torch.randn(2)
    xy = torch.randn(2, 1, 8, 8, 8)

    assert_vars_change(
            model=ModuleWrapper(unet),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(unet.parameters()),
            batch=(InputWrapper(x, y), xy),
            device='cpu',
    )

@pytest.mark.parametrize(
        'kwargs', [
            dict(),
            dict(skip_algorithm='add'),
            dict(allow_self_cond=True),
        ],
)
def test_asym_unet_down_up_channels(kwargs):

    def head_factory(in_channels, out_channels):
        yield nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                bias=False,
        )
        yield nn.BatchNorm3d(out_channels)
        yield nn.ReLU()

    def tail_factory(in_channels, out_channels):
        yield nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=0,
        )

    def block_factory(in_channels, out_channels, cond_dim):
        yield ap.AsymConditionedConvBlock(
                in_channels,
                out_channels,
                cond_dim=cond_dim,
        )

    def latent_factory(channels, cond_dim):
        yield from block_factory(
                in_channels=channels,
                out_channels=channels,
                cond_dim=cond_dim,
        )

    def downsample_factory(in_channels, out_channels):
        yield nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
        )
        yield nn.BatchNorm3d(out_channels)
        yield nn.ReLU()

    def upsample_factory(in_channels, out_channels):
        yield nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
        )
        yield nn.BatchNorm3d(out_channels)
        yield nn.ReLU()

    def noise_embedding(cond_dim):
        yield SinusoidalEmbedding(
                out_dim=cond_dim,
                min_wavelength=0.1,
                max_wavelength=100,
        )
        yield nn.Linear(cond_dim, cond_dim)
        yield nn.ReLU()

    unet = ap.AsymUNet_DownUpChannels(
            channels=[1, 2, 3, 4],
            head_factory=head_factory,
            tail_factory=tail_factory,
            block_factories=multipartial[1,2](block_factory),
            latent_factory=latent_factory,
            downsample_factory=downsample_factory,
            upsample_factory=upsample_factory,
            cond_dim=6,
            noise_embedding=noise_embedding,
            **kwargs,
    )

    x = torch.randn(2, 1, 11, 11, 11)
    y = torch.randn(2)
    xy = torch.randn(2, 1, 11, 11, 11)

    assert_vars_change(
            model=ModuleWrapper(unet),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(unet.parameters()),
            batch=(InputWrapper(x, y), xy),
            device='cpu',
    )

@pff.parametrize(
        schema=[
            pff.cast(
                x_shape=with_py.eval,
                y_shape=with_py.eval,
                xy_shape=with_py.eval,
            ),
            pff.defaults(
                xy_shape=None,
            ),
        ],
)
def test_asym_conditioned_conv_block(block, x_shape, y_shape, xy_shape):
    block = with_ap.eval(block)

    x = torch.randn(*x_shape)
    y = torch.randn(*y_shape)
    xy = torch.randn(*(xy_shape or x_shape))

    assert_vars_change(
            model=ModuleWrapper(block),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(block.parameters()),
            batch=(InputWrapper(x, y), xy),
            device='cpu',
    )

def test_asym_attention_block():
    attn = ap.AsymAttentionBlock(
            img_channels=3, 
            num_heads=2,
            channels_per_head=2,
    )

    x = torch.randn(2, 3, 4, 4, 4)
    y = torch.randn(2, 3, 4, 4, 4)

    assert_vars_change(
            model=attn,
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(attn.parameters()),
            batch=(x, y),
            device='cpu',
    )

