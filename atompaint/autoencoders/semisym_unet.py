import torchyield as ty

from .unet import UNet, PushSkip, NoSkip, get_pop_skip_class
from atompaint.conditioning import ConditionedModel
from atompaint.field_types import make_trivial_field_type
from torch import Tensor
from escnn.nn import GeometricTensor
from more_itertools import one, pairwise, mark_ends
from multipartial import require_grid

from typing import Iterable, Literal, Optional
from torchyield import LayerFactory
from escnn.nn import FieldType

class SemiSymUNet_EncDecChannels(ConditionedModel):

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
            skip_algorithm: Literal['cat', 'add'] = 'cat',
            cond_dim: int,
            noise_embedding: LayerFactory,
            label_embedding: Optional[LayerFactory] = None,
            allow_self_cond: bool = False,
            verbose: bool = False,
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
                            cond_dim: int,
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
                            cond_dim: int,
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

            cond_dim:
                The dimension of the condition embedding that will be passed to the 
                `forward()` method.  The purpose of this embedding is to inform 
                the model about the amount of noise present in the input.

            noise_embedding:
                A function than can be used to instantiate one or more 
                non-equivariant modules that will be used to make a latent 
                embedding of the noise level of the current diffusion step.
                This embedding will be shared between each encoder/decoder 
                block of the U-Net.  Typically, this is a shallow MLP.  It is 
                also typical for each encoder/decoder block to pass this 
                embedding through another shallow MLP before incorporating it 
                into the main latent representation of the image, but how/if 
                this is done is up to the encoder/decoder factories.  This 
                factory should have the following signature:

                    noise_embedding(
                            *,
                            cond_dim: int,
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
        head_channels = img_channels * (2 if allow_self_cond else 1)

        self.in_type = one(make_trivial_field_type(gspace, img_channels))
        self.head_type = one(make_trivial_field_type(gspace, head_channels))
        self.img_channels = img_channels

        PopSkip = get_pop_skip_class(skip_algorithm)

        def iter_unet_blocks():
            head = head_factory(
                    in_type=self.head_type,
                    out_type=encoder_types[0],
            )
            yield NoSkip.from_layers(head)

            for _, is_last_i, (in_type, out_type, encoder_factories_i) in \
                    mark_ends(iter_encoder_params()):

                for is_first_j, _, factory in mark_ends(encoder_factories_i):
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

            for is_first_i, _, (in_channels, out_channels, decoder_factories_i) in \
                    mark_ends(iter_decoder_params()):

                if not is_first_i:
                    yield NoSkip.from_layers(upsample_factory(in_channels))

                for _, is_last_j, factory in mark_ends(reversed(decoder_factories_i)):
                    decoder = factory(
                            in_channels=PopSkip.adjust_in_channels(in_channels),
                            out_channels=in_channels if not is_last_j else out_channels,
                            cond_dim=cond_dim,
                    )
                    yield PopSkip.from_layers(decoder)

            tail = tail_factory(
                    in_channels=encoder_types[0].size,
                    out_channels=img_channels,
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

        unet_blocks = iter_unet_blocks()

        if verbose:
            unet_blocks = ty.verbose(unet_blocks)

        super().__init__(
                model=UNet(unet_blocks),
                noise_embedding=noise_embedding(cond_dim),
                label_embedding=label_embedding and label_embedding(cond_dim),
                allow_self_cond=allow_self_cond,
        )

    def wrap_input(self, x: Tensor) -> GeometricTensor:
        return GeometricTensor(x, self.head_type)

class SemiSymUNet_DownUpChannels(ConditionedModel):

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
            skip_algorithm: Literal['cat', 'add'] = 'cat',
            cond_dim: int,
            noise_embedding: LayerFactory,
            label_embedding: Optional[LayerFactory] = None,
            allow_self_cond: bool = False,
            verbose: bool = False,
    ):
        """
        Construct a U-Net with an equivariant encoder and a non-equivariant 
        decoder.

        The idea behind this architecture is to take advantage of both the 
        better inductive bias that equivariant models have, and the greater 
        expressivity that non-equivariant models have.

        The difference between this module and `SemiSymUNet_EncDecChannels` is 
        in which blocks change the number of channels and the size of the 
        spatial dimensions in the latent representation.  For this module, both 
        changes are made by the upsampling/downsampling blocks.  For the other, 
        the encoder/decoder blocks change the channel size while the 
        upsampling/downsampling blocks change the spatial size.

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
                            cond_dim: int,
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
                            channels: int,
                            cond_dim: int,
                    ) -> nn.Module | Iterable[nn.Module]

            latent_factory:
                A function that can be used to instantiate the "latent" block 
                that will be invoked between the encoder and decoder.  This 
                block must convert its input from a `GeometricTensor` into a 
                regular `Tensor`.  The function should have the following 
                signature::

                    latent_factory(
                            *,
                            in_type: escnn.nn.FieldType,
                    ) -> nn.Module | Iterable[nn.Module]

            downsample_factory:
                A function than can be used to instantiate one or more 
                equivariant modules that will shrink the spatial dimensions of 
                the input on the "encoder" side of the U-Net.  These modules 
                should not alter the number of channels.  The function should 
                have the following signature::

                    downsample_factory(
                            *,
                            in_type: escnn.nn.FieldType,
                            out_type: escnn.nn.FieldType,
                    ) -> nn.Module | Iterable[nn.Module]

            upsample_factory:
                A function than can be used to instantiate one or more 
                non-equivariant modules that will be used to expand the spatial 
                dimensions of the input on the "decoder" side of the U-Net.  
                These modules should not alter the number of channels.  The 
                function should have the following signature::

                    upsample_factory(
                            *,
                            in_channels: int,
                            out_channels: int,
                    ) -> nn.Module | Iterable[nn.Module]

            cond_dim:
                The dimension of the condition embedding that will be passed to the 
                `forward()` method.  The purpose of this embedding is to inform 
                the model about the amount of noise present in the input.

            noise_embedding:
                A function than can be used to instantiate one or more 
                non-equivariant modules that will be used to make a latent 
                embedding of the noise level of the current diffusion step.
                This embedding will be shared between each encoder/decoder 
                block of the U-Net.  Typically, this is a shallow MLP.  It is 
                also typical for each encoder/decoder block to pass this 
                embedding through another shallow MLP before incorporating it 
                into the main latent representation of the image, but how/if 
                this is done is up to the encoder/decoder factories.  This 
                factory should have the following signature:

                    noise_embedding(
                            *,
                            cond_dim: int,
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
        head_channels = img_channels * (2 if allow_self_cond else 1)

        self.in_type = one(make_trivial_field_type(gspace, img_channels))
        self.head_type = one(make_trivial_field_type(gspace, head_channels))
        self.img_channels = img_channels

        PopSkip = get_pop_skip_class(skip_algorithm)

        def iter_unet_blocks():
            head = head_factory(
                    in_type=self.head_type,
                    out_type=encoder_types[0],
            )
            yield NoSkip.from_layers(head)

            for in_type, out_type, encoder_factories_i in iter_encoder_params():
                for factory in encoder_factories_i:
                    encoder = factory(
                            in_type=in_type,
                            out_type=in_type,
                            cond_dim=cond_dim,
                    )
                    yield PushSkip.from_layers(encoder)

                downsample = downsample_factory(
                        in_type=in_type,
                        out_type=out_type,
                )
                yield NoSkip.from_layers(downsample)

            latent = latent_factory(
                    in_type=out_type,
                    cond_dim=cond_dim,
            )
            yield NoSkip.from_layers(latent)

            for in_channels, out_channels, decoder_factories_i in \
                    iter_decoder_params():

                upsample = upsample_factory(
                        in_channels=in_channels,
                        out_channels=out_channels,
                )
                yield NoSkip.from_layers(upsample)

                for factory in reversed(decoder_factories_i):
                    decoder = factory(
                            in_channels=PopSkip.adjust_in_channels(out_channels),
                            out_channels=out_channels,
                            cond_dim=cond_dim,
                    )
                    yield PopSkip.from_layers(decoder)

            tail = tail_factory(
                    in_channels=out_channels,
                    out_channels=img_channels,
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

        unet_blocks = iter_unet_blocks()

        if verbose:
            unet_blocks = ty.verbose(unet_blocks)

        super().__init__(
                model=UNet(unet_blocks),
                noise_embedding=noise_embedding(cond_dim),
                label_embedding=label_embedding and label_embedding(cond_dim),
                allow_self_cond=allow_self_cond,
        )

    def wrap_input(self, x: Tensor) -> GeometricTensor:
        return GeometricTensor(x, self.head_type)

SemiSymUNet = SemiSymUNet_EncDecChannels  # for backwards compatibility
