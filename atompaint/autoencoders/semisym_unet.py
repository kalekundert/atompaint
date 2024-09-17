from .unet import UNet, PushSkip, NoSkip, get_pop_skip_class
from atompaint.field_types import make_trivial_field_type
from torch import Tensor
from escnn.nn import GeometricTensor
from more_itertools import one, pairwise, mark_ends

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
            encoder_factories: list[LayerFactory],
            decoder_factories: list[LayerFactory],
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
                A list of functions that can be used to instantiate the 
                equivariant "blocks" making up the U-Net encoder.  In each 
                level of the U-Net, each of the blocks produces by the 
                functions will be invoked in order.  The functions should have 
                the following signature::

                    encoder_factory(
                            *,
                            in_type: escnn.nn.FieldType,
                            out_type: escnn.nn.FieldType,
                            time_dim: int,
                            depth: int,
                    ) -> nn.Module | Iterable[nn.Module]

            decoder_factories:
                A list of functions that can be used to instantiate the 
                non-equivariant "blocks" making up the U-Net decoder.  In each 
                level of the U-Net, each of the blocks produces by the 
                functions will be invoked in order.  The functions should have 
                the following signature::

                    decoder_factory(
                            *,
                            in_channels: int,
                            out_channels: int,
                            time_dim: int,
                            depth: int,
                    ) -> nn.Module | Iterable[nn.Module]

            latent_factory:
                A function that can be used to instantiate the "latent" block 
                that will be invoked between the encoder and decoder.  This 
                block must convert its input from a `GeometricTensor` into a 
                regular `Tensor`.  The function should have the following 
                signature::

                    latent_factory(
                            field_type: escnn.nn.FieldType,
                    ) -> nn.Module | Iterable[nn.Module]

            downsample_factory:
                A function than can be used to instantiate one or more 
                equivariant modules that will shrink the spatial dimensions of 
                the input on the "encoder" side of the U-Net.  These modules 
                should not alter the number of channels.  The function should 
                have the following signature::

                    downsample_factory() -> nn.Module | Iterable[nn.Module]

            upsample_factory:
                A function than can be used to instantiate one or more 
                non-equivariant modules that will be used to expand the spatial 
                dimensions of the input on the "decoder" side of the U-Net.  
                These modules should not alter the number of channels.  The 
                function should have the following signature::

                    upsample_factory() -> nn.Module | Iterable[nn.Module]

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
                        time_dim: int,
                    ) -> nn.Module | Iterable[nn.Module]
        """
        encoder_types = list(encoder_types)
        gspace = encoder_types[0].gspace
        self.in_type = one(make_trivial_field_type(gspace, img_channels))
        self.img_channels = img_channels

        PopSkip = get_pop_skip_class(skip_algorithm)

        def iter_unet_blocks():
            t1, t2 = self.in_type, encoder_types[0]
            max_depth = len(encoder_types) - 2
            
            head = head_factory(
                    in_type=t1,
                    out_type=t2,
            )
            yield NoSkip.from_layers(head)

            for i, (in_type, out_type) in enumerate(pairwise(encoder_types)):
                for is_first, _, factory in mark_ends(encoder_factories):
                    encoder = factory(
                            in_type=in_type if is_first else out_type,
                            out_type=out_type,
                            time_dim=time_dim,
                            depth=i,
                    )
                    yield PushSkip.from_layers(encoder)

                if i != max_depth:
                    yield NoSkip.from_layers(downsample_factory(out_type))

            latent = latent_factory(
                    in_type=out_type,
                    time_dim=time_dim,
            )
            yield NoSkip.from_layers(latent)

            for i, (in_type, out_type) in enumerate(pairwise(reversed(encoder_types))):
                if i != 0:
                    yield NoSkip.from_layers(upsample_factory())

                for _, is_last, factory in mark_ends(decoder_factories):
                    decoder = factory(
                            in_channels=PopSkip.adjust_in_channels(in_type.size),
                            out_channels=in_type.size if not is_last else out_type.size,
                            time_dim=time_dim,
                            depth=max_depth - i,
                    )
                    yield PopSkip.from_layers(decoder)

            tail = tail_factory(
                    in_channels=t2.size,
                    out_channels=t1.size,
            )
            yield NoSkip.from_layers(tail)
            
        super().__init__(
                blocks=iter_unet_blocks(),
                time_embedding=time_factory(time_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x_hat = GeometricTensor(x, self.in_type)
        return super().forward(x_hat, t)

