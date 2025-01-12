import torch.nn as nn

from .unet import UNet, PushSkip, NoSkip, get_pop_skip_class
from atompaint.conditioning import ConditionedModel
from atompaint.upsampling import R3Upsampling
from atompaint.field_types import make_trivial_field_type
from escnn.nn import (
        GeometricTensor,
        R3Conv, R3ConvTransposed, IIDBatchNorm3d, SequentialModule,
)
from itertools import pairwise
from more_itertools import one, mark_ends
from multipartial import require_grid
from pipeline_func import f

from torch import Tensor
from torchyield import LayerFactory
from escnn.nn import FieldType
from collections.abc import Iterable
from typing import Literal, Optional

class SymUNet(ConditionedModel):

    def __init__(
            self,
            *,
            img_channels: int,
            field_types: Iterable[FieldType],
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
        field_types = list(field_types)
        block_factories = require_grid(
                block_factories,
                rows=len(field_types) - 1,
        )
        gspace = field_types[0].gspace
        head_channels = img_channels * (2 if allow_self_cond else 1)

        self.in_type = one(make_trivial_field_type(gspace, img_channels))
        self.head_type = one(make_trivial_field_type(gspace, head_channels))
        self.out_type = self.in_type
        self.img_channels = img_channels

        PopSkip = get_pop_skip_class(skip_algorithm)

        def iter_unet_blocks():
            head = head_factory(
                    in_type=self.head_type,
                    out_type=field_types[0],
            )
            yield NoSkip.from_layers(head)

            for _, is_last_i, (in_type, out_type, block_factories_i) in \
                    mark_ends(iter_encoder_params()):

                for is_first_j, _, factory in mark_ends(block_factories_i):
                    encoder = factory(
                            in_type=in_type if is_first_j else out_type,
                            out_type=out_type,
                            cond_dim=cond_dim,
                    )
                    yield PushSkip.from_layers(encoder)

                if not is_last_i:
                    yield NoSkip.from_layers(downsample_factory(out_type))

            latent = latent_factory(
                    in_type=out_type,
                    cond_dim=cond_dim,
            )
            yield NoSkip.from_layers(latent)

            for is_first_i, _, (in_type, out_type, block_factories_i) in \
                    mark_ends(iter_decoder_params()):

                if not is_first_i:
                    yield NoSkip.from_layers(upsample_factory(in_type))

                for _, is_last_j, factory in mark_ends(block_factories_i):
                    decoder = factory(
                            in_type=PopSkip.adjust_in_channels(in_type),
                            out_type=in_type if not is_last_j else out_type,
                            cond_dim=cond_dim,
                    )
                    yield PopSkip.from_layers(decoder)

            tail = tail_factory(
                    in_type=field_types[0],
                    out_type=self.out_type,
            )
            yield NoSkip.from_layers(tail)

        def iter_encoder_params():
            params = zip(pairwise(field_types), block_factories, strict=True)
            for in_out_type, block_factories_i in params:
                yield *in_out_type, block_factories_i

        def iter_decoder_params():
            params = zip(
                    pairwise(reversed(field_types)),
                    reversed(block_factories),
                    strict=True,
            )
            for in_out_type, block_factories_i in params:
                yield *in_out_type, block_factories_i
            
        super().__init__(
                model=UNet(iter_unet_blocks()),
                noise_embedding=noise_embedding(cond_dim),
                label_embedding=label_embedding and label_embedding(cond_dim),
                allow_self_cond=allow_self_cond,
        )

    def wrap_input(self, x: Tensor) -> GeometricTensor:
        return GeometricTensor(x, self.head_type)

    def unwrap_output(self, x: GeometricTensor) -> Tensor:
        assert x.type == self.out_type
        return x.tensor

class SymUNetBlock(nn.Module):
    """
    An equivariant implementation of a ResNet block, i.e. two convolutions 
    followed by a residual connection.
    """

    def __init__(
            self,
            in_type,
            *,
            cond_activation: nn.Module,
            out_activation: nn.Module,
            size_algorithm: Literal['padded-conv', 'upsample', 'transposed-conv'] = 'padded-conv',
    ):
        """
        Arguments:
            cond_activation:
                A module that will integrate a condition into the main image 
                representation, and then apply some sort of nonlinearity.  
                This module's `forward()` method should have the following 
                signature:

                    forward(x: GeometricTensor, y: Tensor)

                Where the inputs have the following dimensions:

                    x: (B, C, W, H, D)
                        B: batch size
                        C: channels
                        W, H, D: spatial dimensions (all equal)

                    y: (B, E)
                        B: batch size
                        E: embedding size
            
            out_activation:
                The activation to apply at the end of the block, i.e. after the 
                second convolution.

            size_algorithm:
                The means by which the output of this block is kept the same 
                shape as the input:

                `'padded-conv'`:
                    Pad all convolutions such that their output is the same 
                    shape as their input.  This how almost all ResNet-style 
                    architectures work, but one possible downside is that this 
                    allows the model to detect the edges of the input, which in 
                    turns allows the model to break equivariance.  

                `'upsample'`:
                    Use unpadded convolutions, then include a linear 
                    interpolation step at the end to restore the input shape.

                `'transposed-conv'`:
                    Use a transposed convolution followed by a regular 
                    convolution, instead of two regular convolutions.
        """
        super().__init__()

        self.in_type = in_type
        mid_type_1 = cond_activation.in_type
        mid_type_2 = cond_activation.out_type
        mid_type_3 = out_activation.in_type
        self.out_type = out_type = out_activation.out_type

        conv1_kwargs = dict(
                in_type=in_type,
                out_type=mid_type_1,
                kernel_size=3,
                stride=1,

                # Batch-normalization will recenter everything on 0, so there's 
                # no point having a bias just before that.
                # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm
                bias=False,
        )
        conv2_kwargs = dict(
                in_type=mid_type_2,
                out_type=mid_type_3,
                kernel_size=3,
                stride=1,
                bias=False,
        )

        if size_algorithm == 'padded-conv':
            self.conv1 = R3Conv(padding=1, **conv1_kwargs)
            self.conv2 = R3Conv(padding=1, **conv2_kwargs)
            self.upsample = lambda x: x
            self.min_input_size = 3

        elif size_algorithm == 'upsample':
            self.conv1 = R3Conv(padding=0, **conv1_kwargs)
            self.conv2 = R3Conv(padding=0, **conv2_kwargs)
            self.upsample = R3Upsampling(
                    out_type, 
                    size_expr=lambda x: x+4,
                    align_corners=True,
            )
            self.min_input_size = 7

        elif size_algorithm == 'transposed-conv':
            self.conv1 = R3ConvTransposed(padding=0, **conv1_kwargs)
            self.conv2 = R3Conv(padding=0, **conv2_kwargs)
            self.upsample = lambda x: x
            self.min_input_size = 3

        else:
            raise ValueError(f"unknown size-maintenance algorithm: {size_algorithm!r}")

        self.bn1 = IIDBatchNorm3d(mid_type_1)
        self.bn2 = IIDBatchNorm3d(mid_type_3)

        self.act1 = cond_activation
        self.act2 = out_activation

        if in_type == out_type:
            self.skip = lambda x: x
        else:
            self.skip = SequentialModule(
                    R3Conv(
                        in_type,
                        out_type,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    IIDBatchNorm3d(out_type),
            )

    def forward(self, x: GeometricTensor, y: Tensor):
        *_, w, h, d = x.shape
        assert w == h == d
        assert w >= self.min_input_size
        assert x.type == self.in_type

        x_conv = (
                x
                | f(self.conv1)
                | f(self.bn1)
                | f(self.act1, y)
                | f(self.conv2)
                | f(self.bn2)
                | f(self.act2)
                | f(self.upsample)
        )
        x_skip = self.skip(x)

        return x_conv + x_skip

