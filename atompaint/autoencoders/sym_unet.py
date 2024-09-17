import torch.nn as nn

from .unet import UNet, PushSkip, NoSkip, get_pop_skip_class
from atompaint.upsampling import R3Upsampling
from atompaint.field_types import make_trivial_field_type

from escnn.nn import (
        GeometricTensor,
        R3Conv, R3ConvTransposed, IIDBatchNorm3d, SequentialModule,
)
from itertools import pairwise
from more_itertools import one, mark_ends
from pipeline_func import f

from torch import Tensor
from torchyield import LayerFactory
from escnn.nn import FieldType
from collections.abc import Iterable
from typing import Literal

class SymUNet(UNet):

    def __init__(
            self,
            *,
            img_channels: int,
            field_types: Iterable[FieldType],
            head_factory: LayerFactory,
            tail_factory: LayerFactory,
            block_factories: list[LayerFactory],
            latent_factory: LayerFactory,
            downsample_factory: LayerFactory,
            upsample_factory: LayerFactory,
            time_dim: int,
            time_factory: LayerFactory,
            skip_algorithm: Literal['cat', 'add'] = 'cat',
    ):
        field_types = list(field_types)
        gspace = field_types[0].gspace
        self.in_type = one(make_trivial_field_type(gspace, img_channels))
        self.out_type = self.in_type
        self.img_channels = img_channels

        PopSkip = get_pop_skip_class(skip_algorithm)

        def iter_unet_blocks():
            t1, t2 = self.in_type, field_types[0]
            
            head = head_factory(
                    in_type=t1,
                    out_type=t2,
            )
            yield NoSkip.from_layers(head)

            for _, is_last, (in_type, out_type) in mark_ends(pairwise(field_types)):
                for is_first, _, factory in mark_ends(block_factories):
                    encoder = factory(
                            in_type=in_type if is_first else out_type,
                            out_type=out_type,
                            time_dim=time_dim,
                    )
                    yield PushSkip.from_layers(encoder)

                if not is_last:
                    yield NoSkip.from_layers(downsample_factory(out_type))

            latent = latent_factory(
                    in_type=out_type,
                    time_dim=time_dim,
            )
            yield NoSkip.from_layers(latent)

            for is_first, _, (in_type, out_type) in mark_ends(pairwise(reversed(field_types))):
                if not is_first:
                    yield NoSkip.from_layers(upsample_factory(in_type))

                for _, is_last, factory in mark_ends(block_factories):
                    decoder = factory(
                            in_type=PopSkip.adjust_in_channels(in_type),
                            out_type=in_type if not is_last else out_type,
                            time_dim=time_dim,
                    )
                    yield PopSkip.from_layers(decoder)

            tail = tail_factory(
                    in_type=t2,
                    out_type=t1,
            )
            yield NoSkip.from_layers(tail)
            
        super().__init__(
                blocks=iter_unet_blocks(),
                time_embedding=time_factory(time_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x_hat = GeometricTensor(x, self.in_type)
        y_hat = super().forward(x_hat, t)
        assert y_hat.type == self.out_type
        return y_hat.tensor

class SymUNetBlock(nn.Module):
    """
    An equivariant implementation of a ResNet block, i.e. two convolutions 
    followed by a residual connection.
    """

    def __init__(
            self,
            in_type,
            *,
            size_algorithm: Literal['padded-conv', 'upsample', 'transposed-conv'] = 'padded-conv',
            time_activation: nn.Module,
            out_activation: nn.Module,
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

        self.act1 = time_activation
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
                | f(self.act2)
                | f(self.upsample)
        )
        x_skip = self.skip(x)

        return x_conv + x_skip

