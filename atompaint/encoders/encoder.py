import torchyield as ty
import torch.nn as nn

from atompaint.field_types import make_trivial_field_type
from atompaint.utils import identity
from escnn.nn import FieldType, GeometricTensor
from torch import Tensor
from itertools import pairwise
from more_itertools import one
from multipartial import require_grid

from typing import TypeVar, TypeAlias, Optional
from collections.abc import Iterable, Callable
from torchyield import LayerFactory

T = TypeVar('T')
ChannelSchedule: TypeAlias = Callable[[T, T, int], Iterable[T]]
LayerWrapper: TypeAlias = Callable[[nn.Module], nn.Module]

def early_schedule(in_ch, out_ch, n):
    return [in_ch] + n * [out_ch]

def late_schedule(in_ch, out_ch, n):
    return n * [in_ch] + [out_ch]

class Encoder(ty.FrozenSequential):
    """
    A generic framework for building encoder or decoder networks.

    The main idea behind this framework is to pass the input through a grid of 
    blocks.  Within each row of this grid, the number of channels in the latent 
    representation advance from one entry in `channels` to the next.  
    """

    def __init__(
            self,
            *,
            channels,
            channel_schedule: ChannelSchedule = early_schedule,
            head_factory: Optional[LayerFactory] = None,
            tail_factory: Optional[LayerFactory] = None,
            block_factories: list[list[LayerFactory]],
            block_kwargs: Optional[tuple[str, str]] = None,
            layer_wrapper: LayerWrapper = identity,
            verbose: bool = False,
    ):
        """
        Arguments:
            channel_schedule:
                A function that determines how the channels vary over a single 
                row.  The function is passed the initial and final channels for 
                the row, and the number of blocks in the row.  It should return 
                the sequence of channels (of length 1 + the given number of 
                blocks) to adopt between each block.

                The most common values for this argument are `early_schedule` 
                and `late_schedule`.  The former transitions from the input 
                channels to the output as soon as possible, and the latter 
                transitions as late as possible.

            block_kwargs:
                Normally, the input and output channels are passed to the block 
                factories as positional arguments.  However, if it would be 
                more convenient to use keyword arguments, this argument can be 
                used to specify the keys.
        """
        def iter_layers():
            channel_pairs = list(pairwise(channels))

            if head_factory:
                yield from ty.modules_from_layers(
                        head_factory(*channel_pairs.pop(0))
                )

            if tail_factory:
                tail = tail_factory(*channel_pairs.pop(-1))

            block_factories_ = require_grid(
                    block_factories,
                    rows=len(channel_pairs),
            )
            for channel_pair_i, block_factories_i in zip(
                    channel_pairs, block_factories_, strict=True,
            ):
                for channel_pair_ij, block_factory_ij in zip(
                        pairwise(
                            channel_schedule(
                                *channel_pair_i,
                                len(block_factories_i),
                            ),
                        ),
                        block_factories_i,
                        strict=True,
                ):
                    if block_kwargs:
                        kwargs = dict(zip(block_kwargs, channel_pair_ij))
                        block = block_factory_ij(**kwargs)
                    else:
                        block = block_factory_ij(*channel_pair_ij)

                    yield from ty.modules_from_layers(block)

            if tail_factory:
                yield from ty.modules_from_layers(tail)

        layers = map(layer_wrapper, iter_layers())

        if verbose:
            layers = ty.verbose(layers)

        super().__init__(layers)

class Decoder(Encoder):

    def __init__(
            self,
            *,
            channels,
            channel_schedule: ChannelSchedule = late_schedule,
            head_factory: Optional[LayerFactory] = None,
            tail_factory: Optional[LayerFactory] = None,
            block_factories: list[list[LayerFactory]],
            block_kwargs: Optional[tuple[str, str]] = None,
            layer_wrapper: LayerWrapper = identity,
            verbose: bool = False,
    ):
        # This is a very thin wrapper around `Encoder`, which really is 
        # flexible enough to be both an encoder and a decoder.  The only 
        # difference is that the default channel schedule changes from early to 
        # late.
        super().__init__(
                channels=channels,
                channel_schedule=channel_schedule,
                head_factory=head_factory,
                tail_factory=tail_factory,
                block_factories=block_factories,
                block_kwargs=block_kwargs,
                layer_wrapper=layer_wrapper,
                verbose=verbose,
        )

class SymEncoder(Encoder):
    """
    A generic framework for encoders that maintain equivariance.
    """

    def __init__(
            self,
            *,
            in_channels: int,
            field_types: Iterable[FieldType],
            channel_schedule: ChannelSchedule = early_schedule,
            head_factory: Optional[LayerFactory] = None,
            tail_factory: Optional[LayerFactory] = None,
            block_factories: list[list[LayerFactory]],
            block_kwargs: Optional[tuple[str, str]] = None,
            layer_wrapper: LayerWrapper = identity,
            verbose: bool = False,
    ):
        field_types = list(field_types)
        gspace = field_types[0].gspace

        self.in_type = one(make_trivial_field_type(gspace, in_channels))
        self.out_type = field_types[-1]

        super().__init__(
                channels=[self.in_type, *field_types],
                channel_schedule=channel_schedule,
                head_factory=head_factory,
                tail_factory=tail_factory,
                block_factories=block_factories,
                block_kwargs=block_kwargs,
                layer_wrapper=layer_wrapper,
                verbose=verbose,
        )

    def forward(self, x: Tensor) -> GeometricTensor:
        x_hat = GeometricTensor(x, self.in_type)
        y_hat = super().forward(x_hat)
        assert y_hat.type == self.out_type
        return y_hat


class SymDecoder(Decoder):

    def __init__(
            self,
            *,
            field_types: Iterable[FieldType],
            out_channels: int,
            channel_schedule: ChannelSchedule = late_schedule,
            head_factory: Optional[LayerFactory] = None,
            tail_factory: Optional[LayerFactory] = None,
            block_factories: list[list[LayerFactory]],
            block_kwargs: Optional[tuple[str, str]] = None,
            layer_wrapper: LayerWrapper = identity,
            verbose: bool = False,
    ):
        field_types = list(field_types)
        gspace = field_types[0].gspace

        self.in_type = field_types[0]
        self.out_type = one(make_trivial_field_type(gspace, out_channels))

        super().__init__(
                channels=[*field_types, self.out_type],
                channel_schedule=channel_schedule,
                head_factory=head_factory,
                tail_factory=tail_factory,
                block_factories=block_factories,
                block_kwargs=block_kwargs,
                layer_wrapper=layer_wrapper,
                verbose=verbose,
        )

    def forward(self, x: GeometricTensor) -> Tensor:
        assert x.type == self.in_type
        y = super().forward(x)
        assert y.type == self.out_type
        return y.tensor
