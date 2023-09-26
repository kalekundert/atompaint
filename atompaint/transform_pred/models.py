import torch

from .fourier import QuotientInverseFourier
from atompaint.downsample import EquivariantCnn, FourierCnn
from atompaint.type_hints import LayerFactory

from escnn.nn import (
        SequentialModule, Linear, FourierPointwise, FieldType, GeometricTensor,
        tensor_directsum,
)
from escnn.group import GroupElement
from escnn.gspaces import no_base_space
from itertools import pairwise
from typing import Any

class TransformationPredictor(torch.nn.Module):
    """
    Predict the relative orientation of two atom clouds.

    The main purpose of this class is to being easy to configure via Lightning.  
    That means accepting a limited set of constructor arguments that are all 
    relatively simple types.
    """

    def __init__(
            self, *,
            frequencies: int = 2,
            conv_channels: list[int] = [1, 1],
            conv_field_of_view: int | list[int] = 4,
            conv_stride: int | list[int] = 2,
            mlp_channels: int | list[int] = [1],

    ):
        super().__init__()

        self.encoder = FourierCnn(
                channels=conv_channels,
                conv_field_of_view=conv_field_of_view,
                conv_stride=conv_stride,
                frequencies=frequencies,
        )
        self.mlp = ViewClassifierMlp(
                so3_fields = _parse_mlp_channels(self.encoder, mlp_channels),
                layer_factory = lambda in_type, out_type: [
                    Linear(in_type, out_type),
                    FourierPointwise(
                        out_type.gspace,
                        len(out_type) // len(self.encoder.out_repr),
                        self.encoder.irreps,
                        # Default grid parameters from SO(3) example:
                        type='thomson_cube', N=4,
                    ),
                ],
                fourier_irreps = self.encoder.irreps,

                # This has to be the same as the grid used to construct the 
                # dataset.  For now, I've just hard-coded the 'cube' grid.
                fourier_grid = self.encoder.gspace.fibergroup.sphere_grid('cube'),
        )

    def forward(self, input: torch.Tensor):
        """
        Arguments:
            input:
                A tensor of dimension (B, 2, C, W, H, D) containing two regions 
                of a macromolecule that are related by an unknown transformation.
            
                B: minibatch size
                2: region index
                C: atom-type channels
                W: region width
                H: region height
                D: region depth

                Note that the regions will always be cubical, meaning that W, 
                H, and D will always be equal.

        Returns:
            A tensor of dimension (B, V) that describes the probability that 
            each minibatch member belongs to each possible view.  The values in 
            this tensor are unnormalized logits, suitable to be passed to the 
            softmax or cross-entropy loss functions.

            B: minibatch size
            V: number of possible views
        """
        latent_0 = self.encoder(input[:,0])
        latent_1 = self.encoder(input[:,1])
        latent = _flatten_base_space(tensor_directsum([latent_0, latent_1]))
        return self.mlp(latent)

    @property
    def in_type(self):
        return self.encoder.fields[0]

class ViewClassifierMlp(torch.nn.Module):

    def __init__(
            self, *,
            so3_fields: list[FieldType],
            layer_factory: LayerFactory,
            fourier_irreps: [Any],
            fourier_grid: list[GroupElement],
    ):
        super().__init__()

        layers = []
        for in_type, out_type in pairwise(so3_fields):
            layers += list(layer_factory(in_type, out_type))

        # This quotient-space inverse Fourier transform approach requires that 
        # the group being used have SO(2) as a subgroup.  I think that 
        # effectively means that the group has to be SO(3).  It definitely 
        # can't be the icosahedral group.  That's too bad, because the 
        # icosahedral group is compatible with pointwise nonlinearities, and I 
        # was hoping to see how well those worked.  I could always get around 
        # this requirement by doing a non-quotient-space Fourier transform, but 
        # that would add a lot of complexity relating to working out which 
        # basis vectors to ignore.

        so2_z = False, -1
        self.inv_fourier_s2 = QuotientInverseFourier(
                so3_fields[-1].gspace,
                subgroup_id=so2_z,
                channels=1,
                irreps=fourier_irreps,
                grid=fourier_grid,
        )

        layers += [
                Linear(so3_fields[-1], self.inv_fourier_s2.in_type),
        ]

        # I thought about including an additional ReLU after converting to 
        # real-space, but this just throws away information and doesn't add any 
        # expressiveness:
        #
        # https://stats.stackexchange.com/questions/163695/non-linearity-before-final-softmax-layer-in-a-convolutional-neural-network

        self.mlp = SequentialModule(*layers)

    def forward(self, input: GeometricTensor):
        """
        Arguments:
            input:
                A geometric tensor containing the concatenated latent 
                representations of two regions of a macromolecular structure.  
                The tensor must have dimensions (B, F), where:

                B: minibatch size
                F: fiber size

                The fiber in question is the one defined by the representation 
                *in_repr* provided to the constructor.  Note that the tensor 
                must not have any spatial dimensions remaining.
        """
        assert input.tensor.dim() == 2
        x_fourier = self.mlp(input)
        x_real = self.inv_fourier_s2.forward(x_fourier)

        # There's only one channel, so get rid of that dimension.
        b, c, g = x_real.shape
        assert c == 1
        return x_real.reshape((b, g))

def _parse_mlp_channels(
        encoder: EquivariantCnn,
        mlp_channels: list[int] | int,
) -> list[FieldType]:
    """
    Make input and output field types for each layer of the MLP, from 
    parameters that are convenient for end-users to specify.

    Arguments:
        encoder:
            The model used to make a single latent-space encoding of the input.  
            Note that the actual input to the MLP will be the concatenated 
            input from two such encoders.

        mlp_channels:
            If a list of integers is given, it is interpreted as the number of 
            channels that should be in each hidden layer of the MLP.  

            If an integer is given instead, it is interpreted as the number of 
            hidden layers to create, and each hidden layer will be the same 
            size as the input layer. This is a rule-of-thumb based on the idea 
            that (i) the ideal hidden layer size is probably between the input 
            and output sizes and (ii) it's better to err on the side of making 
            the hidden layer too big.  A too-big hidden layer will be less 
            efficient to train, but still capable of learning the necessary 
            features.  A too-small hidden layer may not give good performance.
    
            https://stackoverflow.com/questions/10565868/multi-layer-perceptron-mlp-architecture-criteria-for-choosing-number-of-hidde

    Returns:
        A list of field types.  An input field type, which is corresponds 
        exactly to the concatenated outputs from the two encoders, is prepended 
        to the field types specified by the *mlp_channels* argument.
    """
    if isinstance(mlp_channels, int):
        mlp_channels = [2 * encoder.out_channels] * mlp_channels

    gspace = no_base_space(encoder.gspace.fibergroup)
    mlp_fields = [
            FieldType(gspace, n * encoder.out_repr)
            for n in [2 * encoder.out_channels] + mlp_channels
    ]

    return mlp_fields

def _flatten_base_space(geom_tensor):
    # TODO: I'd like to contribute this as a method of the `GeometricTensor` 
    # class.
    tensor = geom_tensor.tensor
    in_type = geom_tensor.type
    spatial_dims = in_type.gspace.dimensionality

    assert geom_tensor.coords is None
    # If you get this error; it's because your convolutional layers are not 
    # sized to your input properly.
    assert all(x == 1 for x in tensor.shape[-spatial_dims:])

    out_shape = tensor.shape[:-spatial_dims]
    out_type = FieldType(
            no_base_space(in_type.gspace.fibergroup),
            in_type.representations,
    )

    return GeometricTensor(
            tensor.reshape(out_shape),
            out_type,
    )
