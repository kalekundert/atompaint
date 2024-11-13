import torch
import torchyield as ty

from atompaint.layers import sym_conv_bn_fourier_layer
from atompaint.pooling import FourierExtremePool3D
from escnn.nn import (
        FieldType, GeometricTensor,
        R3Conv, PointwiseAvgPoolAntialiased3D,
)
from torch.nn import Module, ModuleList, Sequential
from more_itertools import pairwise

from atompaint.type_hints import Grid
from torchyield import LayerFactory

class SymDenseBlock(Module):
    """
    A block that implements the core functionality of a DenseNet, i.e. 
    successively concatenating channels to the latent representation.
    """

    def __init__(
            self,
            in_type: FieldType,
            out_type: FieldType,
            *,
            concat_factories: list[LayerFactory],
            gather_factory: LayerFactory,
    ):
        """
        Arguments:
            concat_factories:
                - list of factories
                - arguments: input type
                - output type can be anything, will be concatenated to input 
                  tensor.
                - Typical: convolution, batch norm, activation

            gather_factory:
                - Produce modules that reduce number of channels and optionally 
                  reduce spatial dimensions.
                - Argument: input type and output type
                - Typical: 1x1x1 convolution to reduce channel dimension, then 
                  pool to reduce spatial dimensions.
        """
        super().__init__()

        self.in_type = in_type
        self.out_type = out_type

        next_type = in_type
        concats = []

        for i, factory in enumerate(concat_factories):
            layer = factory(next_type)
            modules = ty.modules_from_layers(layer)
            concat = Sequential(*modules)
            concats.append(concat)

            next_type = concat_field_types(next_type, concat[-1].out_type)

        self.concats = ModuleList(concats)
        self.gather = ty.module_from_layers(
                gather_factory(next_type, out_type),
        )

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        for layer in self.concats:
            x = concat_tensors_by_channel(x, layer(x))
        return self.gather(x)

def sym_concat_conv_bn_fourier_layer(
        in_type,
        *,
        out_types,
        ift_grid: Grid,
        function='p_elu',
):
    for in_type, out_type in pairwise([in_type, *out_types]):
        yield from sym_conv_bn_fourier_layer(
                in_type,
                out_type,
                kernel_size=3,
                padding=1,
                stride=1,
                ift_grid=ift_grid,
                function=function,
        )

def sym_gather_conv_layer(
        in_type,
        out_type,
        *,
        pool_factor,
):
    yield R3Conv(
            in_type,
            out_type,
            kernel_size=3,
            stride=pool_factor,
            padding=1,
    )

def sym_gather_conv_pool_avg_layer(
        in_type,
        out_type,
        *,
        pool_factor,
        sigma=0.6,
):
    yield R3Conv(
            in_type,
            out_type,
            kernel_size=1,
            stride=1,
            padding=0,
    )

    if pool_factor == 1:
        return

    yield PointwiseAvgPoolAntialiased3D(
            in_type,
            sigma=sigma,
            stride=pool_factor,
    )

def sym_gather_conv_pool_fourier_extreme_layer(
        in_type,
        out_type,
        *,
        ift_grid,
        pool_factor,
        check_input_shape,
):
    yield R3Conv(
            in_type,
            out_type,
            kernel_size=1,
            stride=1,
            padding=0,
    )

    if pool_factor == 1:
        return

    yield FourierExtremePool3D(
            out_type,
            grid=ift_grid,
            kernel_size=pool_factor,
            check_input_shape=check_input_shape,
    )


def concat_tensors_by_channel(
        x1: GeometricTensor,
        x2: GeometricTensor,
) -> GeometricTensor:
    x_tensor = torch.cat([x1.tensor, x2.tensor], 1)
    x_type = concat_field_types(x1.type, x2.type)
    return GeometricTensor(x_tensor, x_type)

def concat_field_types(type_1, type_2):
    assert type_1.gspace == type_2.gspace
    return FieldType(
            type_1.gspace,
            type_1.representations + type_2.representations,
    )
