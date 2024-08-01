import torch.nn as nn

from atompaint.upsampling import R3Upsampling
from atompaint.field_types import make_trivial_field_type

from escnn.nn import (
        GeometricTensor,
        R3Conv, IIDBatchNorm3d, IdentityModule, SequentialModule
)
from itertools import pairwise
from more_itertools import one
from pipeline_func import f
from math import sqrt

from torch import Tensor
from escnn.nn import FieldType
from collections.abc import Iterable

class UNet(nn.Module):

    def __init__(
            self,
            *,
            field_types: Iterable[FieldType],
            block_factory,
            block_repeats: int,
            downsample_factory,
            upsample_factory,
            time_embedding,
    ):
        super().__init__()

        field_types = list(field_types)
        self.in_type = field_types[0]
        self.out_type = field_types[0]

        def outer_encoder_params():
            yield from pairwise(field_types)

        def inner_encoder_types(in_type, out_type):
            yield in_type, out_type
            for i in range(block_repeats - 1):
                yield out_type, out_type

        def outer_decoder_params():
            yield from pairwise(reversed(field_types))

        def inner_decoder_types(in_type, out_type):
            for i in range(block_repeats - 1):
                yield in_type, in_type
            yield in_type, out_type

        self.time_embedding = time_embedding

        self.encoder_blocks = nn.ModuleList([])
        self.encoder_transitions = nn.ModuleList([])
        self.decoder_blocks = nn.ModuleList([])
        self.decoder_transitions = nn.ModuleList([])

        for in_type, out_type in outer_encoder_params():
            self.encoder_blocks.append(
                    nn.ModuleList([
                        block_factory(t1, t2)
                        for t1, t2 in inner_encoder_types(in_type, out_type)
                    ])
            )
            self.encoder_transitions.append(
                    downsample_factory(out_type)
            )

        for in_type, out_type in outer_decoder_params():
            self.decoder_transitions.append(
                    upsample_factory(in_type)
            )
            self.decoder_blocks.append(
                    nn.ModuleList([
                        block_factory(t1, t2)
                        for t1, t2 in inner_decoder_types(in_type, out_type)
                    ])
            )

        self.latent_blocks = nn.ModuleList([
            block_factory(field_types[-1], field_types[-1])
            for _ in range(block_repeats)
        ])

    def forward(self, x: GeometricTensor, t: Tensor) -> GeometricTensor:
        assert x.type == self.in_type

        t = self.time_embedding(t)

        skips = []

        for blocks, downsample in zip(
                self.encoder_blocks,
                self.encoder_transitions,
        ):
            for block in blocks:
                x = block(x, t); skips.append(x)
            x = downsample(x)

        for block in self.latent_blocks:
            x = block(x, t)

        for upsample, blocks in zip(
                self.decoder_transitions,
                self.decoder_blocks,
        ):
            x = upsample(x)
            for block in blocks:
                x = block(x + skips.pop(), t)

        assert not skips
        assert x.type == self.out_type

        return x


class UNetBlock(nn.Module):

    def __init__(
            self,
            in_type,
            *,
            time_activation: nn.Module,
            out_activation: nn.Module,
            padded_conv: bool = True,
            ):
        """
        Arguments:
            time_activation:
                A module that will integrate a time embedding into the main 
                image representation, and then apply some sort of nonlinearity.  
                This module's `forward()` method should have the following 
                signature:

                    forward(x: GeometricTensor, t: Tensor)

                Where the inputs have the following dimensions:

                    x: (B, C, W, H, D)
                        B: batch size
                        C: channels
                        W, H, D: spatial dimensions (all equal)

                    t: (B, T)
                        B: batch size
                        T: time embedding size

            padded_conv:
                The output of this block must be the same shape as the input.  
                If *padded_conv* is true, this is accomplished using padded 
                convolutions.  This how most ResNet-style architectures work, 
                but one possible downside is that this allows the model to 
                detect the edges of the input, which in turns allows the model 
                to break equivariance.  Therefore, if *padded_conv* is False, 
                only unpadded convolutions are used, and a linear interpolation 
                step is added to restore the input size.
        """
        super().__init__()

        self.in_type = in_type
        mid_type_1 = time_activation.in_type
        mid_type_2 = time_activation.out_type
        mid_type_3 = out_activation.in_type
        self.out_type = out_activation.out_type

        self.conv1 = R3Conv(
                in_type,
                mid_type_1,
                kernel_size=3,
                stride=1,
                padding=1 if padded_conv else 0,

                # Batch-normalization will recenter everything on 0, so there's 
                # no point having a bias just before that.
                # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm
                bias=False,
        )
        self.bn1 = IIDBatchNorm3d(mid_type_1)
        self.act1 = time_activation
        self.conv2 = R3Conv(
                mid_type_2,
                mid_type_3,
                kernel_size=3,
                stride=1,
                padding=1 if padded_conv else 0,
                bias=False,
        )
        self.bn2 = IIDBatchNorm3d(mid_type_3)
        self.act2 = out_activation

        if padded_conv:
            self.upsample = IdentityModule(mid_type_3)
            self.min_input_size = 3
        else:
            self.upsample = R3Upsampling(
                    mid_type_3, 
                    size_expr=lambda x: x+4,
                    align_corners=True,
            )
            self.min_input_size = 7

        if in_type == mid_type_3:
            self.skip = IdentityModule(in_type)
        else:
            self.skip = SequentialModule(
                    R3Conv(
                        in_type,
                        mid_type_3,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    IIDBatchNorm3d(mid_type_3),
            )

    def forward(self, x: GeometricTensor, t: Tensor):
        *_, w, h, d = x.shape
        assert w == h == d
        assert w >= self.min_input_size

        x_conv = (
                x
                | f(self.conv1)
                | f(self.bn1)
                | f(self.act1, t)
                | f(self.conv2)
                | f(self.bn2)
                | f(self.upsample)
        )
        x_skip = self.skip(x)

        # Both `x_conv` and `x_skip` are batch normalized, so ignoring the 
        # learnable affine batch normalization parameters, they should have 
        # variance=1.  Additionally ignoring any covariance between the two 
        # summands, this means that we can loosely expect the sum to have 
        # variance=2.  We want the output from this layer to maintain 
        # variance=1, so we have to divide by the standard deviation.
        x_sum = x_conv + x_skip
        x_sum.tensor /= sqrt(2)

        return self.act2(x_sum)

class UNetWrapper(nn.Module):
    """
    Convert the input image into the field type expected by the U-Net.

    A secondary function of this layer is to control the first and last layers 
    of the network.  This is useful for a few reasons:

    - The first convolution should be unpadded, so that the padding is not 
      incorrectly interpreted as empty space.  (In subsequent layers, this is 
      not a problem, because 0 doesn't necessarily mean "empty space".)

    - The last layer of the U-Net itself is a nonlinearity, but the last layer 
      of a whole model should usually be linear.
    """

    def __init__(
            self,
            *,
            unet,
            preprocess_factory,
            postprocess_factory,
            channels,
    ):
        super().__init__()

        self.trivial_type = one(make_trivial_field_type(
                gspace=unet.in_type.gspace,
                channels=channels,
        ))

        self.unet = unet
        self.preprocess = preprocess_factory(
                self.trivial_type,
                unet.in_type,
        )
        self.postprocess = postprocess_factory(
                unet.out_type,
                self.trivial_type,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x_wrap = GeometricTensor(x, self.trivial_type)
        y_wrap = (
                x_wrap
                | f(self.preprocess)
                | f(self.unet, t)
                | f(self.postprocess)
        )
        return y_wrap.tensor


