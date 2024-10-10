import torch
import torch.nn as nn
import atompaint.autoencoders.asym_unet as ap
import parametrize_from_file as pff
import pytest

from atompaint.time_embedding import SinusoidalEmbedding
from atompaint.utils import partial_grid
from torchtest import assert_vars_change
from test_time_embedding import ModuleWrapper, InputWrapper

with_py = pff.Namespace()
with_ap = pff.Namespace('from atompaint.autoencoders.asym_unet import *')

@pytest.mark.parametrize('skip_algorithm', ['cat', 'add'])
def test_asym_unet(skip_algorithm):

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

    def block_factory(in_channels, out_channels, time_dim):
        yield from conv_attn_factory(
                in_channels,
                out_channels,
                time_dim=time_dim,
        )

    def latent_factory(channels, time_dim):
        yield from conv_attn_factory(
                in_channels=channels,
                out_channels=channels,
                time_dim=time_dim,
                attention=True,
        )

    def conv_attn_factory(in_channels, out_channels, time_dim, attention=False):
        yield ap.AsymConditionedConvBlock(
                in_channels,
                out_channels,
                time_dim=time_dim,
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

    def time_factory(time_dim):
        yield SinusoidalEmbedding(
                out_dim=time_dim,
                min_wavelength=0.1,
                max_wavelength=100,
        )
        yield nn.Linear(time_dim, time_dim)
        yield nn.ReLU()

    unet = ap.AsymUNet(
            channels=[1, 2, 3, 4],
            head_factory=head_factory,
            tail_factory=tail_factory,
            block_factories=partial_grid(cols=2)(block_factory),
            latent_factory=latent_factory,
            downsample_factory=downsample_factory,
            upsample_factory=upsample_factory,
            time_dim=6,
            time_factory=time_factory,
            skip_algorithm=skip_algorithm,
    )

    x = torch.randn(2, 1, 8, 8, 8)
    t = torch.randn(2)
    y = torch.randn(2, 1, 8, 8, 8)

    assert_vars_change(
            model=ModuleWrapper(unet),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(unet.parameters()),
            batch=(InputWrapper(x, t), y),
            device='cpu',
    )

@pff.parametrize(
        schema=[
            pff.cast(
                x_shape=with_py.eval,
                y_shape=with_py.eval,
                t_shape=with_py.eval,
            ),
            pff.defaults(
                y_shape=None,
            ),
        ],
)
def test_asym_conditioned_conv_block(block, x_shape, t_shape, y_shape):
    block = with_ap.eval(block)

    x = torch.randn(*x_shape)
    t = torch.randn(*t_shape)
    y = torch.randn(*(y_shape or x_shape))

    assert_vars_change(
            model=ModuleWrapper(block),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(block.parameters()),
            batch=(InputWrapper(x, t), y),
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

