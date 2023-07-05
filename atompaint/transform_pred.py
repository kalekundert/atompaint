import torch

from escnn.nn import *
from escnn.group import Group, Representation, SO3
from escnn.gspaces import no_base_space
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
from itertools import pairwise
from more_itertools import unique_everseen as unique

from .downsample import EquivariantCnn, IcosahedralCnn, FourierCnn
from .type_hints import LayerFactory

class TransformationPredictor(torch.nn.Module):
    """
    Predict the relative orientation of two atom clouds.
    """

    def __init__(
            self,
            # The model doesn't really have to be an equivariant CNN, it just 
            # needs to be a module with *out_repr* and *out_channels* 
            # attributes.  For now, though, this is the only class that meets 
            # these requirements.
            encoder: EquivariantCnn,
            mlp_layer_factory: LayerFactory,
            mlp_channels: list[int] | int = 1,
    ):
        """
        Arguments:
            model:
                A torch module that creates an embedding given a macromolecular 
                atom cloud, and defines the following attributes:

                - ``gspace``
                - ``out_repr``
                - ``out_channel``

            mlp_channels:
                How many channels should be in the hidden layers of the MLP.  
                Unlike the `CoordFrameMlp` argument of the same name, this 
                argument only specifies the hidden layers and does not include 
                the input layer.  The reason is that the input layer can be 
                inferred from the given model.

                If an integer is given (instead of a list of integers), this is 
                interpreted as the number of hidden layers to create.  Each 
                hidden layer will be the same size as the input layer.
        """
        super().__init__()

        # If the user specified the number of layers to use, but not the number 
        # of channels to use in each layer, make each hidden layer the same 
        # size as the input layer.  This is a rule-of-thumb based on the idea 
        # that the ideal hidden layer size is probably between the input and 
        # output sizes, and that it's better to err on the side of making the 
        # hidden layer too big.  A too-big hidden layer will be less efficient 
        # to train, but still capable of learning the necessary features.  A 
        # too-small hidden layer may not give good performance.
        #
        # https://stackoverflow.com/questions/10565868/multi-layer-perceptron-mlp-architecture-criteria-for-choosing-number-of-hidde

        if isinstance(mlp_channels, int):
            mlp_channels = [2 * encoder.out_channels] * mlp_channels

        self.encoder = encoder
        self.mlp = CoordFrameMlp(
                group = encoder.gspace.fibergroup,
                in_reprs = encoder.out_repr,
                channels = [2 * encoder.out_channels] + mlp_channels,
                layer_factory = mlp_layer_factory,
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
            A tensor of dimension (B, 4, 4) containing an SE(3) transformation 
            matrix for each member of the minibatch.
        """
        b = input.shape[0]

        latent_0 = self.encoder(input[:,0])
        latent_1 = self.encoder(input[:,1])
        latent = flatten_base_space(tensor_directsum([latent_0, latent_1]))

        return self.mlp(latent)

    @property
    def in_type(self):
        return self.encoder.fields[0]

class IcosahedralTransformationPredictor(TransformationPredictor):

    def __init__(
            self, *,
            conv_channels: list[int] = [1, 1],
            conv_field_of_view: int = 4,
            pool_field_of_view: int = 2,
            mlp_channels: list[int] = [1],
    ):
        super().__init__(
                IcosahedralCnn(
                    channels=conv_channels,
                    conv_field_of_view=conv_field_of_view,
                    pool_field_of_view=pool_field_of_view,
                ),
                mlp_channels=mlp_channels,
                mlp_layer_factory=lambda in_type, out_type: [
                    Linear(in_type, out_type),
                    ReLU(out_type),
                ],
        )

class FourierTransformationPredictor(TransformationPredictor):

    def __init__(
            self, *,
            conv_channels: list[int] = [1, 1],
            conv_field_of_view: int = 4,
            conv_stride: int = 2,
            mlp_channels: list[int] = [1],
            frequencies: int = 2,
    ):
        super().__init__(
                cnn := FourierCnn(
                    channels=conv_channels,
                    conv_field_of_view=conv_field_of_view,
                    conv_stride=conv_stride,
                    frequencies=frequencies,
                ),
                mlp_channels=mlp_channels,
                mlp_layer_factory=lambda in_type, out_type: [
                    Linear(in_type, out_type),
                    FourierPointwise(
                        out_type.gspace,
                        len(out_type) // len(cnn.out_repr),
                        cnn.irreps,

                        # Default grid parameters from SO(3) example:
                        type='thomson_cube', N=4,
                    ),
                ],
        )

class CoordFrameMlp(torch.nn.Module):
    """
    Predict the relative orientation of two macromolecular embeddings.

    This class is not meant to be used directly.  Rather, it's meant to be 
    constructed internally by classes like `IcosahedralTransformationPredictor` 
    and `FourierTransformationPredictor`.
    """

    def __init__(
            self,
            group: Group,
            in_reprs: list[Representation],
            channels: list[int],
            layer_factory: LayerFactory,
    ):
        """
        Arguments:
            in_reprs:
                The representations that act on the input fibers.

            channels:
                The number of channels to use for each layer in the network.  
                The number of elements in this list determines the number of 
                layers to create.  The first value in this list describes the 
                input layer, and so its value (along with *in_reprs*) must 
                describe the dimensions of the geometric tensors that will be 
                fed into this module.

            layer_factory:
                A function that returns the equivariant modules that should be 
                used to move between adjacent layers.  Typically, this would 
                include `escnn.nn.Linear`, a nonlinearity, and optionally batch 
                normalization or dropout, in some order.  This function will be 
                called once for each pair of layers, except for the last.  A 
                single linear module will be used to move between the last pair 
                of layers.
        """
        super().__init__()

        gspace = no_base_space(group)

        in_fields = [
                FieldType(gspace, n * in_reprs)
                for n in channels
        ]

        # The structure of the output fiber is an internal implementation 
        # detail.  This model only guarantees to return an SE(3) transformation 
        # matrix with certain equivariance/invariance properties; it is free to 
        # parametrize that matrix however it likes.
        #
        # The predicted translation should be 3D and equivariant w.r.t.  
        # rotation of the input.  The standard representation of SO(3)---3D 
        # rotation matrices---meets both of these requirements.  The predicted 
        # rotation should be invariant w.r.t. rotation of the input, which 
        # calls for the trivial representation.  Only 3 dimensions are 
        # necessary to specify a rotation, but [Zhou2020] shows that rotations 
        # are easier to learn when encoded using 6 dimensions, so that's the 
        # approach I'm taking.

        if isinstance(group, SO3):
            standard_repr = group.standard_representation()
        else:
            standard_repr = group.standard_representation

        out_field = FieldType(
                gspace,
                [standard_repr] + 6 * [gspace.trivial_repr],
        )

        layers = []

        for in_type, out_type in pairwise(in_fields):
            layers += list(layer_factory(in_type, out_type))

        layers += [
                Linear(in_fields[-1], out_field),
        ]

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
        b, _ = input.shape

        xyz_rot = self.mlp(input).tensor

        frame = torch.zeros((b,4,4))
        frame[:, 0:3, 0:3] = rotation_6d_to_matrix(xyz_rot[:, 3:])
        frame[:, 0:3,   3] = xyz_rot[:, :3]
        frame[:,   3,   3] = 1

        return frame

class CoordFrameMseLoss(torch.nn.Module):
    """
    Calculate the distance between two coordinate frames.
    """

    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def forward(self, predicted_frame, expected_frame):
        """
        Arguments:
            predicted_frame:
                The coordinate frames predicted by the context encoder, as 
                tensors of dimension (B, 4, 4). 

                B: minibatch size
                4,4: 3D roto-translation matrix

            expected_frame:
                The true coordinate frames, as tensors of dimension (B, 4, 4).
        """
        xyz = torch.cat([
                torch.eye(3) * self.radius,
                torch.ones((1,3)),
        ])
        xyz_pred = predicted_frame @ xyz
        xyz_expect = expected_frame @ xyz

        # This is mean-squared distance.  No reason to bother calculating the 
        # square root; it won't change the location of the minimum.
        return torch.mean(torch.sum((xyz_pred - xyz_expect)**2, axis=1))

def flatten_base_space(geom_tensor):
    # TODO: I'd like to contribute this as a method of the `GeometricTensor` 
    # class.
    tensor = geom_tensor.tensor
    in_type = geom_tensor.type
    spatial_dims = in_type.gspace.dimensionality

    assert geom_tensor.coords is None
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
