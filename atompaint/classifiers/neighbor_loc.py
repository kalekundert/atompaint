import lightning.pytorch as L
import torch.nn.functional as F
import torchyield as ty
import torch
import re

from atompaint.field_types import (
        make_fourier_field_types, make_polynomial_field_types,
        make_exact_polynomial_field_types,
)
from atompaint.layers import (
        sym_conv_layer, sym_conv_bn_fourier_layer, sym_conv_bn_gated_layer,
        sym_linear_fourier_layer, flatten_base_space,
)
from atompaint.nonlinearities import leaky_hard_shrink, first_hermite
from atompaint.checkpoints import (
        EvalModeCheckpointMixin, load_model_weights, strip_prefix,
)
from escnn.nn import (
        FieldType, FourierFieldType, GeometricTensor, InverseFourierTransform,
        SequentialModule, Linear, tensor_directsum, 
)
from escnn.group import GroupElement
from escnn.gspaces import rot3dOnR3, no_base_space
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Adam
from torchmetrics import Accuracy
from itertools import pairwise
from functools import partial
from multipartial import multipartial, Dim, dim, put
from pipeline_func import f, X

from typing import Iterable, Optional
from torchyield import Layer, LayerFactory

class ClassificationTask(EvalModeCheckpointMixin, L.LightningModule):

    def __init__(self, model, *, num_classes):
        super().__init__()

        self.model = model
        self.loss = CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.optimizer = Adam(model.parameters())

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, batch):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)

        return loss, acc

    def training_step(self, batch, _):
        loss, acc = self.forward(batch)
        self.log('train/loss', loss, on_epoch=True)
        self.log('train/accuracy', acc, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss, acc = self.forward(batch)
        self.log('val/loss', loss)
        self.log('val/accuracy', acc)
        return loss

    def test_step(self, batch, _):
        loss, acc = self.forward(batch)
        self.log('test/loss', loss)
        self.log('test/accuracy', acc)
        return loss

    @property
    def in_type(self):
        return self.model.in_type

class NeighborLoc(Module):
    """
    Predict the location of one group of atoms (encoded as a 3D image) relative
    to another.
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

    # The existence of this property confuses `torchlens`.  This is definitely 
    # a `torchlens` bug, but it's easiest to just comment out this method when 
    # necessary.
    @property
    def in_type(self):
        return self.encoder.in_type

class SymViewPairEncoder(Module):

    def __init__(self, encoder: Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> GeometricTensor:
        y0 = self.encoder(x[:,0])
        y1 = self.encoder(x[:,1])

        return flatten_base_space(tensor_directsum([y0, y1]))

    @property
    def in_type(self):
        return self.encoder.in_type

    @property
    def out_type(self):
        out_type = self.encoder.out_type
        gspace = no_base_space(out_type.gspace.fibergroup)
        return FieldType(gspace, 2 * out_type.representations)

class AsymViewPairEncoder(Module):

    def __init__(self, encoder: Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.encoder(x[:,0])
        y1 = self.encoder(x[:,1])

        return torch.cat([y0, y1], dim=1)

class SymViewPairClassifier(Module):

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

class AsymViewPairClassifier(Module):
    """
    A non-equivariant MLP classifier.

    The motivations behind this classifier are that:

    - As it is, the equivariant classifier doesn't account for all symmetries 
      of the problem.  Specifically, if the two input views are swapped, then 
      the output should change in a deterministic way, but this is not done.

    - I expect that a regular, non-equivariant MLP will be more expressive, 
      because it can use ordinary nonlinearities.  

    - Even when not enforced, the model can still learn a degree of 
      "equivariance" via data augmentation.  This might be enough to train the 
      model.

    It is possible to use this classifier with an equivariant encoder.  The 
    result won't be equivariant, of course, but the overall model may still 
    perform better than a fully non-equivariant model.
    """

    def __init__(self, classifier: Layer):
        super().__init__()
        self.classifier = ty.module_from_layers(classifier)

    def forward(self, x: torch.Tensor | GeometricTensor) -> torch.Tensor:
        if isinstance(x, GeometricTensor):
            x = x.tensor 

        assert x.dim() == 2
        return self.classifier(x)

def load_expt_72_model():

    def rename_old_keys(k):
        return (
                k
                | f(strip_prefix, prefix='model.')
                | f(re.sub, r'encoder\.layers\.', r'encoder.', X)
                | f(re.sub, r'(\d+)\.nonlin(\d+)\.', r'\1.act\2.', X)
                | f(re.sub, r'(\d+)\.pool\.', r'\1.resize.', X)
        )

    classifier = make_expt_72_model()
    load_model_weights(
            model=classifier,
            path='expt_72/padding=2-6A;angle=40deg;image-size=24A;job-id=40481465;epoch=49.ckpt',
            fix_keys=rename_old_keys,
            xxh32sum='e4b0330d',
    )
    return classifier

def make_expt_72_model():
    from atompaint.encoders.sym_resnet import make_expt_72_resnet

    resnet = make_expt_72_resnet()
    mlp = ty.mlp_layer(
            ty.linear_relu_dropout_layer,
            **ty.channels([1960, 1024, 6]),
            dropout_p=0.2,
    )

    return NeighborLoc(
            encoder=SymViewPairEncoder(resnet),
            classifier=AsymViewPairClassifier(mlp),
    )

def load_expt_94_model():

    def rename_old_keys(k):
        return (
                k
                | f(strip_prefix, prefix='model.')
                | f(re.sub, r'encoder\.layers\.', r'encoder.', X)
                | f(re.sub, r'(\d+)\.nonlin(\d+)\.', r'\1.act\2.', X)
                | f(re.sub, r'(\d+)\.pool\.', r'\1.resize.', X)
        )

    classifier = make_expt_94_model()
    load_model_weights(
            model=classifier,
            path='expt_94/block=gamma;channels=log-16;image-size=15A;job-id=48156726;epoch=99.ckpt',
            fix_keys=rename_old_keys,
            xxh32sum='ec925b15',
    )
    return classifier

def make_expt_94_model():
    from atompaint.encoders.sym_resnet import make_expt_94_resnet

    resnet = make_expt_94_resnet()
    mlp = ty.mlp_layer(
            ty.linear_relu_dropout_layer,
            **ty.channels([1120, 512, 6]),
            dropout_p=0.2,
    )
    return NeighborLoc(
            encoder=SymViewPairEncoder(resnet),
            classifier=AsymViewPairClassifier(mlp),
    )

# Below are a number of functions for instantiating models from simpler sets of 
# arguments.  These functions are mostly historical artifacts, although I 
# suppose they still could be useful.  Their original purpose was to allow 
# instantiating models from YAML config files.  This is why the arguments to 
# these functions are exclusively primitive data types.  Now, I find it more 
# flexible to instantiate models directly in python.

NONLINEARITIES = {
        # rectifier
        'relu': F.relu,
        'gelu': F.gelu,
        'elu': F.elu,
        'silu': F.silu,
        'mish': F.mish,

        # linear
        'hardshrink': F.hardshrink,
        'leaky_hardshrink': leaky_hard_shrink,
        'softshrink': F.softshrink,
        'tanhshrink': F.tanhshrink,

        # sigmoid
        'hardtanh': F.hardtanh,
        'softsign': F.softsign,
        'sigmoid': F.sigmoid,
        'tanh': F.tanh,
        'hermite': first_hermite,
}

def make_neighbor_loc_model(
        *,
        architecture,
        **kwargs,
):
    MODEL_FACTORIES = {
            'cnn': make_sym_cnn,
            'cnn-noneq': make_asym_cnn,
            'resnet': make_sym_resnet,
            'densenet': make_sym_densenet,
    }
    return MODEL_FACTORIES[architecture](**kwargs)

def make_sym_cnn(
        *,
        frequencies: int,
        conv_channels: list[int],
        conv_field_of_view: int | list[int],
        conv_stride: int | list[int],
        conv_padding: int | list[int],
        mlp_channels: int | list[int],
        equivariant_mlp: bool = True,
):
    from atompaint.encoders import SymEncoder
    from atompaint.layers import sym_linear_fourier_layer

    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    ift_grid = so3.grid('thomson_cube', N=4)
    n = len(conv_channels)

    def cnn_layer(
            in_type,
            out_type,
            *,
            kernel_size,
            stride,
            padding,
            batch_norm,
    ):
        # The purpose of this custom layer is mimic my original CNN 
        # implementation, which skipped batch normalization in the last layer.  
        # This is when the latent representation becomes small, typically 
        # 1x1x1, and so that statistics become less reliable.  In the extreme 
        # case where the batch size is 1, it even becomes impossible to 
        # calculate variance.
        #
        # If I were writing a new CNN from scratch, I'd probably handle this 
        # problem using `tail_factory` rather than a custom `block_factory`.  
        # But my goal here is to output the same model as before.

        conv, bn, act = sym_conv_bn_fourier_layer(
                in_type, out_type,
                kernel_size=kernel_size,
                stride=stride,
                ift_grid=ift_grid,
        )
        yield conv
        if batch_norm:
            yield bn
        yield act

    encoder = SymViewPairEncoder(
            SymEncoder(
                in_channels=conv_channels[0],
                field_types=make_fourier_field_types(
                    gspace=gspace, 
                    channels=conv_channels[1:],
                    max_frequencies=frequencies,
                ),
                block_factories=multipartial[n-1,1](
                    cnn_layer,
                    kernel_size=Dim[0](conv_field_of_view),
                    stride=Dim[0](conv_stride),
                    padding=Dim[0](conv_padding),
                    batch_norm=put[-1,-1](False, True),
                ),
            ),
    )

    if equivariant_mlp:
        classifier = SymViewPairClassifier(
                layer_types=make_fourier_mlp_field_types(
                    in_type=encoder.out_type,
                    channels=mlp_channels,
                    max_frequencies=frequencies,
                ),
                layer_factory=partial(
                    sym_linear_fourier_layer,
                    ift_grid=ift_grid,
                ),
                logits_max_freq=frequencies,

                # This has to be the same as the grid used to construct the 
                # dataset.  For now, I've just hard-coded the 'cube' grid.
                logits_grid=so3.sphere_grid('cube'),
        )
    else:
        assert mlp_channels[-1] == 6
        classifier = AsymViewPairClassifier(
                ty.mlp_layer(
                    ty.linear_relu_dropout_layer,
                    **ty.channels(mlp_channels),
                    dropout_p=0.2,
                ),
        )

    return NeighborLoc(
            encoder=encoder,
            classifier=classifier,
    )

def make_asym_cnn(
        *,
        conv_channels: list[int],
        conv_kernel_sizes: int | list[int],
        conv_strides: int | list[int],
        conv_paddings: int | list[int],
        pool_sizes: int | list[int],
        mlp_channels: int | list[int],
):
    from atompaint.encoders import Encoder

    assert mlp_channels[-1] == 6

    encoder = AsymViewPairEncoder(
            Encoder(
                channels=conv_channels,
                block_factories=multipartial[:,1](
                    ty.conv3_bn_relu_maxpool,
                    kernel_size=Dim[0](conv_kernel_sizes),
                    stride=Dim[0](conv_strides),
                    padding=Dim[0](conv_paddings),
                    pool_stride=Dim[0](pool_sizes),
                ),
            ),
    )
    classifier = AsymViewPairClassifier(
            ty.mlp_layer(
                ty.linear_relu_dropout_layer,
                **ty.channels(mlp_channels),
                dropout_p=0.2,
            ),
    )
    return NeighborLoc(
            encoder=encoder,
            classifier=classifier,
    )

def make_sym_resnet(
        *,
        block_type: str,
        resnet_outer_channels: list[int],
        resnet_inner_channels: list[int],
        polynomial_terms: int | list[int] = 0,
        max_frequency: int = 0,
        grid: Optional[str] = None,
        block_repeats: int,
        pool_factors: int | list[int],
        final_conv: int = 0,
        mlp_channels: list[int],
        equivariant_mlp: bool = True,
):
    from atompaint.encoders import SymEncoder
    from atompaint.encoders.sym_resnet import (
            make_escnn_example_block, make_alpha_block, make_beta_block,
    )
    from atompaint.encoders.utils import toggle_pool
    from atompaint.utils import parse_so3_grid

    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    ift_grid = parse_so3_grid(so3, grid) if grid else None

    if block_type == 'escnn':
        assert polynomial_terms
        assert max_frequency > 0
        assert grid

        outer_types = make_polynomial_field_types(
                gspace=gspace, 
                channels=resnet_outer_channels[1:],
                terms=polynomial_terms,
        )
        inner_types = make_fourier_field_types(
                gspace=gspace, 
                channels=resnet_inner_channels,
                max_frequencies=max_frequency,
        )
        head_factory = sym_conv_layer
        block_factories = multipartial[:,block_repeats](
                toggle_pool(make_escnn_example_block),
                mid_type=dim[0](*inner_types),
                pool=put[:,0](True, False),
                pool_factor=Dim[0](pool_factors),
                ift_grid=ift_grid,
        )

    elif block_type == 'alpha':
        assert polynomial_terms == 0
        assert max_frequency > 0
        assert grid

        outer_types = make_fourier_field_types(
                gspace=gspace, 
                channels=resnet_outer_channels[1:],
                max_frequencies=max_frequency,
        )
        inner_types = make_fourier_field_types(
                gspace=gspace, 
                channels=resnet_inner_channels,
                max_frequencies=max_frequency,
        )
        head_factory = partial(
                sym_conv_bn_fourier_layer,
                ift_grid=ift_grid,
        )
        block_factories = multipartial[:,block_repeats](
                toggle_pool(make_alpha_block),
                mid_type=dim[0](*inner_types),
                pool=put[:,0](True, False),
                pool_factor=Dim[0](pool_factors),
                ift_grid=ift_grid,
        )

    elif block_type == 'beta':
        assert polynomial_terms
        assert max_frequency == 0
        assert grid is None

        # Beta is closely modeled on the Wide ResNet architecture (WRN).

        outer_types = make_exact_polynomial_field_types(
                gspace=gspace, 
                channels=resnet_outer_channels[1:],
                terms=polynomial_terms,
                gated=True,
        )
        inner_types = make_exact_polynomial_field_types(
                gspace=gspace,
                channels=resnet_inner_channels,
                terms=polynomial_terms,
                gated=True,
        )
        head_factory = sym_conv_bn_gated_layer
        block_factories = multipartial[:,block_repeats](
                toggle_pool(make_beta_block),
                mid_type=dim[0](*inner_types),
                pool=put[:,0](True, False),
                pool_factor=Dim[0](pool_factors),
        )

    else:
        raise ValueError(f"unknown block type: {block_type}")

    if final_conv:
        tail_factory = partial(
                sym_conv_layer,
                kernel_size=final_conv,
        )
    else:
        tail_factory = None

    encoder = SymViewPairEncoder(
            SymEncoder(
                in_channels=resnet_outer_channels[0],
                field_types=outer_types,
                head_factory=head_factory,
                block_factories=block_factories,
                tail_factory=tail_factory,
            ),
    )

    if equivariant_mlp:
        classifier = SymViewPairClassifier(
                layer_types=make_fourier_mlp_field_types(
                    in_type=encoder.out_type,
                    channels=mlp_channels,
                    max_frequencies=max_frequency,
                ),
                layer_factory=partial(
                    sym_linear_fourier_layer,
                    ift_grid=ift_grid,
                ),
                logits_max_freq=max_frequency,

                # This has to be the same as the grid used to construct the 
                # dataset.  For now, I've just hard-coded the 'cube' grid.
                logits_grid=so3.sphere_grid('cube'),
        )
    else:
        assert mlp_channels[-1] == 6
        classifier = AsymViewPairClassifier(
                ty.mlp_layer(
                    ty.linear_relu_dropout_layer,
                    **ty.channels(mlp_channels),
                    dropout_p=0.2,
                ),
        )

    return NeighborLoc(
            encoder=encoder,
            classifier=classifier,
    )

def make_sym_densenet(
        *,
        transition_channels: list[int],
        growth_channels: int,
        grid: str,
        max_frequency: int,
        fourier_nonlinearity: str = 'leaky_hardshrink',
        block_depth: int | list[int],
        pool_factors: int | list[int],
        final_conv: int = 0,
        mlp_channels: list[int],
        equivariant_mlp: bool = True,
):
    from atompaint.encoders import SymEncoder
    from atompaint.encoders.sym_densenet import (
            SymDenseBlock, sym_gather_conv_pool_fourier_extreme_layer,
    )
    from atompaint.utils import parse_so3_grid

    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    ift_grid = parse_so3_grid(so3, grid)

    # Things that are hard-coded for now:
    #
    # - Using spectral regular representations at all levels.  Where 
    #   Fourier transforms are required, I could also try spectral induced 
    #   representations (i.e. quotient space Fourier transforms).  
    #   Everywhere else, I could try polynomial representations.
    #
    # - Using the same max frequency everywhere.
    #
    # - Dense layer nonlinearities: gated first, Fourier second.  It might 
    #   be better to use only one or the other, or even something else.

    def dense_block(in_type, out_type, *, block_depth, pool_factor):
        return SymDenseBlock(
                in_type,
                out_type,
                concat_factories=multipartial[block_depth](concat_layer),
                gather_factory=partial(
                    sym_gather_conv_pool_fourier_extreme_layer,
                    pool_factor=pool_factor,
                    ift_grid=ift_grid,
                    check_input_shape=False,
                ),
        )

    def concat_layer(in_type):
        mid_type, out_type = make_fourier_field_types(
                gspace=gspace,
                channels=[growth_channels * 4, growth_channels],
                max_frequencies=max_frequency,
        )
        yield from sym_conv_bn_gated_layer(
                in_type,
                mid_type,
                kernel_size=1,
        )
        yield from sym_conv_bn_fourier_layer(
                mid_type,
                out_type,
                kernel_size=3,
                padding=1,
                function=NONLINEARITIES[fourier_nonlinearity],
                ift_grid=ift_grid,
        )

    if final_conv:
        tail_factory = partial(
                sym_conv_layer,
                kernel_size=final_conv,
        )
    else:
        tail_factory = None

    encoder = SymViewPairEncoder(
            SymEncoder(
                in_channels=transition_channels[0],
                field_types=make_fourier_field_types(
                    gspace=gspace,
                    channels=transition_channels[1:],
                    max_frequencies=max_frequency,
                    unpack=True,
                ),
                head_factory=partial(
                        sym_conv_bn_fourier_layer,
                        ift_grid=ift_grid,
                ),
                block_factories=multipartial[:,1](
                    dense_block,
                    block_depth=Dim[0](block_depth),
                    pool_factor=Dim[0](pool_factors),
                ),
                tail_factory=tail_factory,
            ),
    )

    if equivariant_mlp:
        classifier = SymViewPairClassifier(
                layer_types=make_fourier_mlp_field_types(
                    in_type=encoder.out_type,
                    channels=mlp_channels,
                    max_frequencies=max_frequency,
                ),
                layer_factory=partial(
                    sym_linear_fourier_layer,
                    ift_grid=ift_grid,
                ),
                logits_max_freq=max_frequency,

                # This has to be the same as the grid used to construct the 
                # dataset.  For now, I've just hard-coded the 'cube' grid.
                logits_grid=so3.sphere_grid('cube'),
        )
    else:
        assert mlp_channels[-1] == 6
        classifier = AsymViewPairClassifier(
                ty.mlp_layer(
                    ty.linear_relu_dropout_layer,
                    **ty.channels(mlp_channels),
                    dropout_p=0.2,
                ),
        )

    return NeighborLoc(
            encoder=encoder,
            classifier=classifier,
    )

def make_fourier_mlp_field_types(
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

