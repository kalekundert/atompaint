import torch

from atompaint.utils import get_scalar
from atompaint.type_hints import LayerFactory
from escnn.group import Representation
from escnn.gspaces import icoOnR3, rot3dOnR3, GSpace
from escnn.nn import *
from itertools import pairwise
from more_itertools import all_equal

class EquivariantCnn(torch.nn.Module):
    """
    A base class implementing some common features for equivariant CNNs.

    This class isn't meant to be used directly; it's meant to factor out code 
    that would otherwise be duplicated between the `IcosahedralCNN` and 
    `FourierCNN` classes.
    """

    def __init__(
            self,
            gspace: GSpace,
            channels: list[int],
            latent_repr: list[Representation],
            layer_factory: LayerFactory,
    ):
        """
        Arguments:
            channels:
                How many channels should make up each layer of the model.  This includes the first layer.
        """
        super().__init__()

        def iter_fields():
            yield FieldType(gspace, channels[0] * [gspace.trivial_repr])
            for n in channels[1:]:
                yield FieldType(gspace, n * latent_repr)

        def iter_layers():
            for i, (field_in, field_out) in enumerate(pairwise(self.fields)):
                yield from layer_factory(i, field_in, field_out)

        self.gspace = gspace
        self.out_repr = latent_repr
        self.out_channels = channels[-1]

        self.fields = list(iter_fields())
        self.layers = SequentialModule(*iter_layers())

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        """
        Create a latent representation of the given voxelized atom cloud.

        Arguments:
            input:
                A voxelized representation of the atoms comprising a 
                macromolecule.  This should be a tensor with shape
                (B, C, X, X, X), where:

                B: minibatch size
                C: number of channels (i.e. atom types)
                X: number of voxels in each spatial dimension
        """
        assert all_equal(input.shape[-3:])
        return self.layers(input)

    @property
    def in_type(self):
        return self.fields[0]

    @property
    def out_type(self):
        return self.fields[-1]

class IcosahedralCnn(EquivariantCnn):
    """\
    A CNN that is equivariant to the symmetries of the icosahedral group.  

    Because this is a finite group, it's possible to use standard element-wise 
    nonlinearities and pooling functions.
    """

    def __init__(
            self, *,
            channels: list[int] = [1, 1],
            conv_field_of_view: int | list[int] = 3,
            pool_field_of_view: int | list[int] = 2,
    ):
        gspace = icoOnR3()
        super().__init__(
                gspace=gspace,
                channels=channels,
                latent_repr=[gspace.regular_repr],

                # Order of layers taken from this Stack Overflow post:
                # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
                layer_factory=lambda i, in_type, out_type: [

                    # No need for bias: subsequent batch-norm step will 
                    # recenter everything on 0 anyways.
                    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm
                    R3Conv(
                        in_type,
                        out_type,
                        kernel_size=get_scalar(conv_field_of_view, i),
                        bias=False,
                    ),
                    IIDBatchNorm3d(out_type),
                    ReLU(out_type),
                    PointwiseMaxPoolAntialiased3D(
                        out_type,
                        get_scalar(pool_field_of_view, i),
                    ),
                ],
        )

class FourierCnn(EquivariantCnn):
    """\
    A CNN that is equivariant to any 3D rotation.
    """

    def __init__(
            self, *,
            channels: list[int] = [1, 1],
            conv_field_of_view: int | list[int] = 3,
            conv_stride: int | list[int] = 2,
            conv_padding: int | list[int] = 0,
            frequencies: int = 2,
    ):
        gspace = rot3dOnR3(frequencies)
        so3 = gspace.fibergroup
        irreps = so3.bl_irreps(frequencies)
        fourier_repr = so3.spectral_regular_representation(*irreps)

        # This information makes it easier for external users of this module to 
        # construct pointwise Fourier nonlinearities.  It is actually possible 
        # to reconstruct these irreps directly from the field type argument to 
        # the *layer_factory* callback, but doing so is pretty hacky.
        self.irreps = irreps

        def layer_factory(i, in_type, out_type):
            use_batch_norm = i < len(channels) - 2

            # Order of layers taken from this Stack Overflow post:
            # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout

            yield R3Conv(
                    in_type,
                    out_type,
                    kernel_size=get_scalar(conv_field_of_view, i),
                    stride=get_scalar(conv_stride, i),
                    padding=get_scalar(conv_padding, i),

                    # Batch-normalization will recenter everything on 0, so 
                    # there's no point having a bias just before that.
                    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm
                    bias=not use_batch_norm,
            )

            # Batch normalization gets noisy when the combined number of 
            # minibatch and spatial dimensions get small.  In fact, if all of 
            # these dimensions are 1, the process fails because it becomes 
            # impossible to calculate variance.  With this in mind, I decided 
            # to skip the batch normalization step for the last layer of the 
            # CNN, where the spatial dimensions will usually be 1x1x1.
            if use_batch_norm:
                yield IIDBatchNorm3d(out_type)

            yield FourierPointwise(
                    # By setting the number of channels equal to the length of 
                    # the output field type, we're implicitly assuming that 
                    # there is only a single representation per channel.  This 
                    # happens to be true here, because we define *latent_repr* 
                    # as such just below, but this is a fragile construction in 
                    # general.
                    gspace, len(out_type), irreps,

                    # Default grid parameters from SO(3) example:
                    type='thomson_cube', N=4,
            )

        super().__init__(
                gspace=gspace,
                channels=channels,
                latent_repr=[fourier_repr],
                layer_factory=layer_factory,
        )
