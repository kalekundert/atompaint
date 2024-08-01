import torch.nn as nn
import torchyield as ty

from atompaint.time_embedding import AddTimeToImage
from more_itertools import pairwise
from pipeline_func import f

from typing import Callable

class AsymUNet(nn.Module):

    def __init__(
            self,
            *,
            channels: list[int],
            time_dim: int,
            block_factory: Callable[..., nn.Module],
            downsample: nn.Module,
            upsample: nn.Module,
            latent_blocks: int = 1,
    ):
        """
        Arguments:
            channels:
                A list giving the number of channels to use for each layer of 
                the U-Net.  The first value in the list should be the number of 
                channels that the input to the U-Net will have, and the last 
                value should be the number of channels in the innermost latent 
                representation.  These same channel counts, in reverse, will be 
                used for the up-sampling side of the U-Net.

            time_dim:
                The dimension of the time embedding that will be passed to the 
                `forward()` method.  The purpose of the time embedding is to 
                inform the model about the amount of noise present in the 
                input.

            block_factory:
                A function that can be used to instantiate the "blocks" making 
                up the U-Net.  The function should have the following 
                signature:

                    block_factory(
                            *,
                            in_channels: int,
                            out_channels: int,
                            time_dim: int,
                    ) -> nn.Module

            downsample:
                A torch module that will be used to shrink the spatial 
                dimensions of the input on the "encoder" side of the U-Net.  
                This module should not alter the number of channels.

            upsample:
                A torch module that will be used to expand the spatial 
                dimensions of the input on the "decoder" side of the U-Net.  
                This module should not alter the number of channels.

            latent_blocks:
                The number of blocks that will be included between the encoder 
                and decoder halves of the U-Net.
        """
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
                block_factory(
                    in_channels=c_in,
                    out_channels=c_out,
                    time_dim=time_dim,
                )
                for c_in, c_out in pairwise(channels)
        ])
        self.latent_blocks = nn.ModuleList([
                block_factory(
                    in_channels=channels[-1],
                    out_channels=channels[-1],
                    time_dim=time_dim,
                )
                for _ in range(latent_blocks)
        ])
        self.decoder_blocks = nn.ModuleList([
                block_factory(
                    in_channels=c_in,
                    out_channels=c_out,
                    time_dim=time_dim,
                )
                for c_in, c_out in pairwise(reversed(channels))
        ])
        self.downsample = ty.Layers(downsample)
        self.upsample = ty.Layers(upsample)

    def forward(self, x, t):
        skips = []

        for block in self.encoder_blocks:
            x = block(x, t); skips.append(x)
            x = self.downsample(x)

        for block in self.latent_blocks:
            x = block(x, t)

        for block in self.decoder_blocks:
            x = self.upsample(x)
            # The DDPM example I'm following concatenates the skip connections, 
            # but addition seems more in the spirit of a "simple ResNet".
            #x = torch.cat([x, skips.pop()], dim=1)
            x = block(x + skips.pop(), t)

        assert not skips
        return x

class AsymUNetBlock(nn.Module):

    def __init__(
            self,
            *,
            in_channels,
            out_channels,
            time_dim,
            dropout_p=0.5,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
        )
        self.conv2 = nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
        )
        self.time = AddTimeToImage(
                time_dim,
                out_channels,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.skip = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
        )

    def forward(self, x, t):
        x_conv = (
                x
                | f(self.conv1)
                | f(self.time, t)
                | f(self.act)
                | f(self.dropout)
                | f(self.conv2)
                | f(self.act)
        )
        x_skip = self.skip(x)
        return x_conv + x_skip


