from __future__ import annotations

import torch.nn as nn

from .unet import UNet, PushSkip, NoSkip, get_pop_skip_class
from atompaint.conditioning import ConditionedModel, AddConditionToImage
from atompaint.upsampling import Upsample3d
from einops import rearrange
from more_itertools import pairwise
from pipeline_func import f
from more_itertools import mark_ends
from multipartial import require_grid

from typing import Literal, Callable, Optional
from torchyield import LayerFactory

class AsymUNet(ConditionedModel):

    def __init__(
            self,
            *,
            channels: list[int],
            head_factory: LayerFactory,
            tail_factory: LayerFactory,
            block_factories: list[list[LayerFactory]],
            latent_factory: LayerFactory,
            downsample_factory: LayerFactory,
            upsample_factory: LayerFactory,
            skip_algorithm: Literal['cat', 'add'] = 'cat',
            cond_dim: int,
            noise_embedding: LayerFactory,
            label_embedding: Optional[LayerFactory] = None,
            allow_self_cond: bool = False,
    ):
        """
        Construct a non-equivariant U-Net.

        Arguments:
            channels:
                A list giving the number of channels to use for each layer of 
                the U-Net.  The first value in the list should be the number of 
                channels that the input to the U-Net will have, and the last 
                value should be the number of channels in the innermost latent 
                representation.  These same channel counts, in reverse, will be 
                used for the up-sampling side of the U-Net.

            head_factory:
                A function that can be used to instantiate one or more modules 
                that will be invoked before the U-Net, e.g. to perform an 
                initial convolution.  The function should have the following 
                signature::

                    head_factory(
                            *,
                            in_channels: int,
                            out_channels: int,
                    ) -> nn.Module | Iterable[nn.Module]

            tail_factory:
                A function that can be used to instantiate one or more modules 
                that will be invoked after the U-Net, e.g. to restore the 
                expected number of output channels.  The function should have 
                the following signature::

                    tail_factory(
                            *,
                            in_channels: int,
                            out_channels: int,
                    ) -> nn.Module | Iterable[nn.Module]

            block_factory:
                A list-of-lists-of-functions that can be used to instantiate 
                the "blocks" making up the U-Net encoder.  Each entry in the 
                outer list corresponds to a different input size.  This size of 
                this list must match the number of channel pairs, or be 1, in 
                which case the same factories will be repeated at each level.  
                The factories in the inner lists will be executed back-to-back, 
                but each will get it's own skip connection.  The functions 
                should have the following signature::

                    block_factory(
                            *,
                            in_channels: int,
                            out_channels: int,
                            cond_dim: int,
                    ) -> nn.Module | Iterable[nn.Module]

            latent_factory:
                A function that can be used to instantiate the "latent" block 
                that will be invoked between the encoder and decoder.  The 
                function should have the following signature::

                    latent_factory(
                            *,
                            channels: int,
                            cond_dim: int,
                    ) -> nn.Module | Iterable[nn.Module]

            downsample_factory:
                A function than can be used to instantiate one or more modules 
                that will shrink the spatial dimensions of the input on the 
                "encoder" side of the U-Net.  These modules should not alter 
                the number of channels.  The function should have the following 
                signature::

                    downsample_factory(
                            *,
                            channels: int,
                    ) -> nn.Module | Iterable[nn.Module]

            upsample_factory:
                A function than can be used to instantiate one or more modules 
                that will be used to expand the spatial dimensions of the input 
                on the "decoder" side of the U-Net.  These modules should not 
                alter the number of channels.  The function should have the 
                following signature::

                    upsample_factory(
                            *,
                            channels: int,
                    ) -> nn.Module | Iterable[nn.Module]

            cond_dim:
                The dimension of the condition embedding that will be passed to the 
                `forward()` method.  The purpose of this embedding is to 
                inform the model about the amount of noise present in the 
                input.

            noise_embedding:
                A function than can be used to instantiate one or more 
                non-equivariant modules that will be used to make a latent 
                embedding of the noise level of the current diffusion step.
                This embedding will be shared between each encoder/decoder 
                block of the U-Net.  Typically, this module is a shallow MLP.  
                It is also typical for each encoder/decoder block to pass this 
                embedding through another shallow MLP before incorporating it 
                into the main latent representation of the image, but how/if 
                this is done is up to the encoder/decoder factories.  This 
                factory should have the following signature:

                    noise_embedding(
                            *,
                            cond_dim: int,
                    ) -> nn.Module | Iterable[nn.Module]
        """

        PopSkip = get_pop_skip_class(skip_algorithm)
        block_factories = require_grid(
                block_factories,
                rows=len(channels) - 2,
        )

        def iter_unet_blocks():
            head = head_factory(
                    in_channels=channels[0] * (2 if allow_self_cond else 1),
                    out_channels=channels[1],
            )
            yield NoSkip.from_layers(head)

            for _, is_last_i, (in_channels, out_channels, block_factories_i) in \
                    mark_ends(iter_encoder_params()):

                for is_first_j, _, factory in mark_ends(block_factories_i):
                    block = factory(
                            in_channels=in_channels if is_first_j else out_channels,
                            out_channels=out_channels,
                            cond_dim=cond_dim,
                    )
                    yield PushSkip.from_layers(block)

                if not is_last_i:
                    yield NoSkip.from_layers(downsample_factory(out_channels))

            latent = latent_factory(
                    channels=out_channels,
                    cond_dim=cond_dim,
            )
            yield NoSkip.from_layers(latent)

            for is_first_i, _, (in_channels, out_channels, block_factories_i) in \
                    mark_ends(iter_decoder_params()):

                if not is_first_i:
                    yield NoSkip.from_layers(upsample_factory(in_channels))

                for _, is_last_j, factory in mark_ends(reversed(block_factories_i)):
                    block = factory(
                            in_channels=PopSkip.adjust_in_channels(in_channels),
                            out_channels=in_channels if not is_last_j else out_channels,
                            cond_dim=cond_dim,
                    )
                    yield PopSkip.from_layers(block)

            tail = tail_factory(
                    in_channels=channels[1],
                    out_channels=channels[0],
            )
            yield NoSkip.from_layers(tail)

        def iter_encoder_params():
            params = zip(
                    pairwise(channels[1:]),
                    block_factories,
                    strict=True,
            )
            for in_out_type, block_factories_i in params:
                yield *in_out_type, block_factories_i

        def iter_decoder_params():
            params = zip(
                    pairwise(reversed(channels[1:])),
                    reversed(block_factories),
                    strict=True,
            )
            for in_out_channels, block_factories_i in params:
                yield *in_out_channels, block_factories_i

        super().__init__(
                model=UNet(iter_unet_blocks()),
                noise_embedding=noise_embedding(cond_dim),
                label_embedding=label_embedding and label_embedding(cond_dim),
                allow_self_cond=allow_self_cond,
        )


class AsymConditionedConvBlock(nn.Module):
    """
    A conditioned block with two convolutions and a residual connection.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            *,
            cond_dim,
            size_algorithm: Literal['padded-conv', 'upsample', 'transposed-conv'] = 'padded-conv',
            activation_factory: Callable[[], nn.Module] = nn.ReLU,
    ):
        super().__init__()

        conv1_kwargs = dict(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
        )
        conv2_kwargs = dict(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    bias=False,
        )

        if size_algorithm == 'padded-conv':
            self.conv1 = nn.Conv3d(padding=1, **conv1_kwargs)
            self.conv2 = nn.Conv3d(padding=1, **conv2_kwargs)
            self.upsample = nn.Identity()
            self.min_input_size = 3

        elif size_algorithm == 'upsample':
            self.conv1 = nn.Conv3d(padding=0, **conv1_kwargs)
            self.conv2 = nn.Conv3d(padding=0, **conv2_kwargs)
            self.upsample = Upsample3d(
                    size_expr=lambda x: x+4,
                    align_corners=True,
                    mode='trilinear',
            )
            self.min_input_size = 7

        elif size_algorithm == 'transposed-conv':
            self.conv1 = nn.ConvTranspose3d(padding=0, **conv1_kwargs)
            self.conv2 = nn.Conv3d(padding=0, **conv2_kwargs)
            self.upsample = nn.Identity()
            self.min_input_size = 3

        else:
            raise ValueError(f"unknown size-maintenance algorithm: {size_algorithm!r}")

        self.cond = AddConditionToImage(
                cond_dim=cond_dim,
                channel_dim=out_channels,
                affine=True,
        )

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.act1 = activation_factory()
        self.act2 = activation_factory()

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
            )

    def forward(self, x, y):
        w, h, d = x.shape[-3:]
        assert w == h == d
        assert w >= self.min_input_size

        x_conv = (
                x
                | f(self.conv1)
                | f(self.cond, y)
                | f(self.bn1)
                | f(self.act1)
                | f(self.conv2)
                | f(self.bn2)
                | f(self.act2)
                | f(self.upsample)
        )
        x_skip = self.skip(x)
        return x_skip + x_conv

class AsymAttentionBlock(nn.Module):
    """
    A block that mimics the first part of a Transformer encoder block, i.e.  
    normalization, followed by self-attention, followed by a residual 
    connection.

    This block is specifically meant to work with 3D image inputs, which leads 
    to a few differences from the "real" transformer encoder blocks:

    - The self-attention step is bracketed by 1x1x1 convolutions.
    - Batch normalization is used instead of layer normalization.
    """

    def __init__(self, img_channels, *, num_heads, channels_per_head):
        super().__init__()

        assert num_heads % 2 == 0
        attn_channels = num_heads * channels_per_head

        self.norm = nn.BatchNorm3d(img_channels)
        self.conv1 = nn.Conv3d(
                img_channels,
                attn_channels,
                kernel_size=1,
                bias=False,
        )
        self.attn = nn.MultiheadAttention(
                attn_channels,
                num_heads,
                batch_first=True,
        )
        self.conv2 = nn.Conv3d(
                attn_channels,
                img_channels,
                kernel_size=1,
        )

    def forward(self, x):
        w, h, d = x.shape[-3:]
        assert w == h == d

        def self_attn(x):
            return self.attn(x, x, x, need_weights=False)[0]

        return x + (
                x
                | f(self.norm)
                | f(self.conv1)
                | f(rearrange, 'b c w h d -> b (w h d) c')
                | f(self_attn)
                | f(rearrange, 'b (w h d) c -> b c w h d', w=w, h=h, d=d)
                | f(self.conv2)
        )

