import torch

from atompaint.encoders.layers import make_fourier_field_types
from atompaint.type_hints import Grid, LayerFactory
from escnn.nn import (
        FieldType, FourierFieldType, GeometricTensor, InverseFourierTransform,
        SequentialModule, Linear, IIDBatchNorm1d, FourierPointwise,
        tensor_directsum, 
)
from escnn.group import GroupElement
from escnn.gspaces import no_base_space
from torch.nn import Module
from itertools import pairwise
from typing import Iterable

class TransformationPredictor(Module):
    """
    Predict the relative orientation of two atom clouds.
    """

    def __init__(
            self, *,
            encoder: Module,
            classifier: Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
        latent = self.encoder(input)
        return self.classifier(latent)

    @property
    def in_type(self):
        return self.encoder.in_type

class ViewPairEncoder(Module):

    def __init__(self, encoder: Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> GeometricTensor:
        x0 = GeometricTensor(x[:,0], self.encoder.in_type)
        x1 = GeometricTensor(x[:,1], self.encoder.in_type)

        y0 = self.encoder(x0)
        y1 = self.encoder(x1)

        return _flatten_base_space(tensor_directsum([y0, y1]))

    @property
    def in_type(self):
        return self.encoder.in_type

    @property
    def out_type(self):
        out_type = self.encoder.out_type
        gspace = no_base_space(out_type.gspace.fibergroup)
        return FieldType(gspace, 2 * out_type.representations)


class ViewPairClassifier(Module):

    def __init__(
            self, *,
            layer_types: list[FieldType],
            layer_factory: LayerFactory,
            logits_max_freq: int,
            logits_grid: list[GroupElement],
    ):
        super().__init__()

        self.layer_types = list(layer_types)

        # This quotient-space inverse Fourier transform approach requires that 
        # the group being used have SO(2) as a subgroup.  I think that 
        # effectively means that the group has to be SO(3).  It definitely 
        # can't be the icosahedral group.  That's too bad, because the 
        # icosahedral group is compatible with pointwise nonlinearities, and I 
        # was hoping to see how well those worked.  I could always get around 
        # this requirement by doing a non-quotient-space Fourier transform, but 
        # that would add a lot of complexity relating to working out which 
        # basis vectors to ignore.

        gspace = self.layer_types[-1].gspace
        so2_z = False, -1
        fourier_type = FourierFieldType(
                gspace=gspace,
                channels=1,
                bl_irreps=gspace.fibergroup.bl_irreps(logits_max_freq),
                subgroup_id=so2_z,
        )

        layers = []
        for in_type, out_type in pairwise(self.layer_types):
            layers += list(layer_factory(in_type, out_type))

        # Convert the last user-specified field type into the field type needed 
        # for the inverse Fourier transform (IFT).  This is not done using the 
        # layer factory, because factories are meant to have nonlinearities as 
        # their last steps, and I don't like the idea of having a nonlinearity 
        # immediately before the IFT.  Instead, I prefer having the last step 
        # be a linear classifier based on (presumably) well-engineered latent 
        # features.

        layers += [
                Linear(self.layer_types[-1], fourier_type),
        ]
        self.layer_types.append(fourier_type)

        self.mlp = SequentialModule(*layers)
        self.inv_fourier_s2 = InverseFourierTransform(
                in_type=fourier_type,
                out_grid=logits_grid,
        )

    def forward(self, input: GeometricTensor) -> torch.Tensor:
        """
        Arguments:
            input:
                A geometric tensor containing the concatenated latent 
                representations of two regions of a macromolecular structure.  
                The tensor must have dimensions (B, F), where:

                B: minibatch size
                F: fiber size

                The fiber in question is the one defined by the first field type 
                provided to the constructor.  Note that the input tensor must 
                not have any spatial dimensions remaining.
        """
        assert input.tensor.dim() == 2
        x_fourier = self.mlp(input)
        x_logits = self.inv_fourier_s2(x_fourier).tensor

        # I thought about applying a nonlinear transformation to the logits, 
        # but this would just throw away information without adding any 
        # expressiveness [1].
        #
        # [1]: https://stats.stackexchange.com/questions/163695/non-linearity-before-final-softmax-layer-in-a-convolutional-neural-network

        # There's only one channel, so get rid of that dimension.
        b, c, g = x_logits.shape
        assert c == 1
        return x_logits.reshape(b, g)

    @property
    def in_type(self):
        return self.layer_types[0]

class NonequivariantViewPairClassifier(Module):
    """
    A simple MLP classifier.

    The motivations behind this classifier are that:

    - As it is, the equivariant classifier doesn't account for all symmetries 
      of the problem.  Specifically, if the two input views are swapped, then 
      the output should change in a deterministic way, but this is not done.

    - I expect that a regular, non-equivariant MLP will be more expressive, 
      because it can use ordinary nonlinearities.  

    - Even when not enforced, the model can still learn a degree of 
      "equivariance" via data augmentation.  This might be enough to train the 
      model.
    """

    def __init__(
            self, *,
            channels: list[int],
            layer_factory: LayerFactory,
    ):
        super().__init__()

        *channels, num_categories = channels

        layers = []
        for in_channels, out_channels in pairwise(channels):
            layers += list(layer_factory(in_channels, out_channels))

        self.mlp = torch.nn.Sequential(
                *layers,
                torch.nn.Linear(channels[-1], num_categories),
        )

    def forward(self, input: GeometricTensor) -> torch.Tensor:
        assert input.tensor.dim() == 2
        return self.mlp(input.tensor)


def make_fourier_classifier_field_types(
        in_type: FieldType,
        channels: int | list[int],
        max_frequencies: int | list[int],
) -> Iterable[FieldType]:
    """
    Make input and output field types for each layer of the classifier MLP, 
    from parameters that are convenient for end-users to specify.

    Arguments:
        in_type:
            The field type produced by the view pair encoder.  Note that this 
            field type should combine latent representations for the two views 
            in question, and should have no spatial dimensions.

        channels:
            If a list of integers is given, it is interpreted as the number of 
            replicates of the spectral regular representation to include in 
            each field type.

            If an integer is given instead, it is interpreted as the number of 
            hidden layers to create.  Each hidden layer will be roughly the 
            same size as the input layer.  This is a rule-of-thumb based on the 
            idea that (i) the ideal hidden layer size is probably between the 
            input and output sizes and (ii) it's better to err on the side of 
            making the hidden layer too big.  A too-big hidden layer will be 
            less efficient to train, but still capable of learning the 
            necessary features.  A too-small hidden layer may not give good 
            performance.
    
            https://stackoverflow.com/questions/10565868/multi-layer-perceptron-mlp-architecture-criteria-for-choosing-number-of-hidde
    """
    yield in_type

    if isinstance(channels, int):
        ft = next(make_fourier_field_types(
            gspace=in_type.gspace,
            channels=[1],
            max_frequencies=max_frequencies,
        ))
        channels = channels * [in_type.size // ft.size]

    yield from make_fourier_field_types(
        gspace=in_type.gspace,
        channels=channels,
        max_frequencies=max_frequencies,
    )

def make_linear_fourier_layer(
        in_type: FieldType,
        out_type: FourierFieldType,
        ift_grid: Grid,
        *,
        nonlinearity: str = 'p_relu',
):
    yield Linear(in_type, out_type, bias=False)
    yield IIDBatchNorm1d(out_type)
    yield FourierPointwise(
            out_type,
            ift_grid,
            function=nonlinearity
    )
    # If I were going to use drop-out, it'd come after the nonlinearity.  But 
    # I've seen some comments saying the batch norm and dropout don't work well 
    # together.

def make_nonequivariant_linear_relu_dropout_layer(
        in_channels,
        out_channels,
        *,
        drop_rate,
):
    from torch import nn

    yield nn.Linear(in_channels, out_channels)
    yield nn.ReLU()
    yield nn.Dropout(drop_rate)

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
