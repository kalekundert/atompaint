from .unet import UNet, PushSkip, NoSkip, get_pop_skip_class
from atompaint.field_types import make_trivial_field_type
from torch import Tensor
from escnn.nn import GeometricTensor
from more_itertools import one, pairwise, mark_ends
from multipartial import require_grid

from typing import Iterable, Literal
from torchyield import LayerFactory
from escnn.nn import FieldType

class SemiSymUNet(UNet):

    def __init__(
            self,
            *,
            img_channels: int,
            encoder_types: Iterable[FieldType],
            head_factory: LayerFactory,
            tail_factory: LayerFactory,
            encoder_factories: list[list[LayerFactory]],
            decoder_factories: list[list[LayerFactory]],
            latent_factory: LayerFactory,
            downsample_factory: LayerFactory,
            upsample_factory: LayerFactory,
            time_dim: int,
            time_factory: LayerFactory,
            skip_algorithm: Literal['cat', 'add'] = 'cat',
    ):
        """
        Construct a U-Net with an equivariant encoder and a non-equivariant 
        decoder.

        The idea behind this architecture is to take advantage of both the 
        better inductive bias that equivariant models have, and the greater 
        expressivity that non-equivariant models have.

        Arguments:
            img_channels:
                The number of channels present in the input images.

            encoder_types:
                The field types to use in each layer of the U-Net encoder.  
                This excludes the very first layer, which is assumed to have 
                `img_channels` trivial representations.

            head_factory:
                A function that can be used to instantiate one or more 
                equivariant modules that will be invoked before the U-Net, e.g. 
                to perform an initial convolution.  The function should have 
                the following signature::

                    head_factory(
                            *,
                            in_type: escnn.nn.FieldType,
                            out_type: escnn.nn.FieldType,
                    ) -> nn.Module | Iterable[nn.Module]

            tail_factory:
                A function that can be used to instantiate one or more 
                non-equivariant modules that will be invoked after the U-Net, 
                e.g. to restore the expected number of output channels.  The 
                function should have the following signature::

                    tail_factory(
                            *,
                            in_channels: int,
                            out_channels: int,
                    ) -> nn.Module | Iterable[nn.Module]

            encoder_factories:
                A list-of-lists-of-functions that can be used to instantiate 
                the equivariant "blocks" making up the U-Net encoder.  Each 
                entry in the outer list corresponds to a different input size.  
                This size of this list must match the number of encoder 
                type pairs, or be 1, in which case the same factories will be 
                repeated at each level.  The entries in the inner lists will be 
                executed back-to-back, but each will get it's own skip 
                connection.  The functions should have the following 
                signature::

                    encoder_factory(
                            *,
                            in_type: escnn.nn.FieldType,
                            out_type: escnn.nn.FieldType,
                            time_dim: int,
                    ) -> nn.Module | Iterable[nn.Module]

            decoder_factories:
                A list-of-list-of-functions that can be used to instantiate the 
                non-equivariant "blocks" making up the U-Net decoder.  Refer to 
                *encoder_factories* for details.  Note that the factories 
                specified here will be invoked in reverse order.  In other 
                words, this means that you should specify the factories in the 
                same order that you would for the encoder argument.  The 
                functions should have the following signature::

                    decoder_factory(
                            *,
                            in_channels: int,
                            out_channels: int,
                            time_dim: int,
                    ) -> nn.Module | Iterable[nn.Module]

            latent_factory:
                A function that can be used to instantiate the "latent" block 
                that will be invoked between the encoder and decoder.  This 
                block must convert its input from a `GeometricTensor` into a 
                regular `Tensor`.  The function should have the following 
                signature::

                    latent_factory(
                            *,
                            field_type: escnn.nn.FieldType,
                    ) -> nn.Module | Iterable[nn.Module]

            downsample_factory:
                A function than can be used to instantiate one or more 
                equivariant modules that will shrink the spatial dimensions of 
                the input on the "encoder" side of the U-Net.  These modules 
                should not alter the number of channels.  The function should 
                have the following signature::

                    downsample_factory(
                            *,
                            field_type: escnn.nn.FieldType,
                    ) -> nn.Module | Iterable[nn.Module]

            upsample_factory:
                A function than can be used to instantiate one or more 
                non-equivariant modules that will be used to expand the spatial 
                dimensions of the input on the "decoder" side of the U-Net.  
                These modules should not alter the number of channels.  The 
                function should have the following signature::

                    upsample_factory(
                            *,
                            channels: int,
                    ) -> nn.Module | Iterable[nn.Module]

            time_dim:
                The dimension of the time embedding that will be passed to the 
                `forward()` method.  The purpose of the time embedding is to 
                inform the model about the amount of noise present in the 
                input.

            time_factory:
                A function than can be used to instantiate one or more 
                non-equivariant modules that will be used to make a latent 
                embedding of the time vectors that will be shared between each 
                encoder/decoder block of the U-Net.  Typically this is a 
                shallow MLP.  It is also typical for each encoder/decoder block 
                to pass this embedding through another shallow MLP before 
                incorporating it into the main latent representation of the 
                image, but how/if this is done is up to the encoder/decoder 
                factories.  This factory should have the following signature:

                    time_factory(
                            *,
                            time_dim: int,
                    ) -> nn.Module | Iterable[nn.Module]
        """
        encoder_types = list(encoder_types)
        encoder_factories = require_grid(
                encoder_factories,
                rows=len(encoder_types) - 1,
        )
        decoder_factories = require_grid(
                decoder_factories,
                rows=len(encoder_types) - 1,
        )
        gspace = encoder_types[0].gspace

        self.in_type = one(make_trivial_field_type(gspace, img_channels))
        self.img_channels = img_channels

        PopSkip = get_pop_skip_class(skip_algorithm)

        def iter_unet_blocks():
            t1, t2 = self.in_type, encoder_types[0]
            
            head = head_factory(
                    in_type=t1,
                    out_type=t2,
            )
            yield NoSkip.from_layers(head)

            for _, is_last_i, (in_type, out_type, encoder_factories_i) in \
                    mark_ends(iter_encoder_params()):

                for is_first_j, _, factory in mark_ends(encoder_factories_i):
                    encoder = factory(
                            in_type=in_type if is_first_j else out_type,
                            out_type=out_type,
                            time_dim=time_dim,
                    )
                    yield PushSkip.from_layers(encoder)

                if not is_last_i:
                    yield NoSkip.from_layers(downsample_factory(out_type))

            latent = latent_factory(
                    in_type=out_type,
                    time_dim=time_dim,
            )
            yield NoSkip.from_layers(latent)

            for is_first_i, _, (in_channels, out_channels, decoder_factories_i) in \
                    mark_ends(iter_decoder_params()):

                if not is_first_i:
                    yield NoSkip.from_layers(upsample_factory(in_channels))

                for _, is_last_j, factory in mark_ends(reversed(decoder_factories_i)):
                    decoder = factory(
                            in_channels=PopSkip.adjust_in_channels(in_channels),
                            out_channels=in_channels if not is_last_j else out_channels,
                            time_dim=time_dim,
                    )
                    yield PopSkip.from_layers(decoder)

            tail = tail_factory(
                    in_channels=t2.size,
                    out_channels=t1.size,
            )
            yield NoSkip.from_layers(tail)
            
        def iter_encoder_params():
            params = zip(
                    pairwise(encoder_types),
                    encoder_factories,
                    strict=True,
            )
            for in_out_type, encoder_factories_i in params:
                yield *in_out_type, encoder_factories_i

        def iter_decoder_params():
            params = zip(
                    pairwise(reversed(encoder_types)),
                    reversed(decoder_factories),
                    strict=True,
            )
            for (in_type, out_type), decoder_factories_i in params:
                yield in_type.size, out_type.size, decoder_factories_i

        super().__init__(
                blocks=iter_unet_blocks(),
                time_embedding=time_factory(time_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x_hat = GeometricTensor(x, self.in_type)
        return super().forward(x_hat, t)

