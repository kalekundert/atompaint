import torch

from atompaint.utils import get_scalar
from escnn.nn import (
        FieldType, FourierFieldType, GeometricTensor,
        R3Conv, IIDBatchNorm3d,
)
from torch.nn import Module, ModuleList, Sequential
from more_itertools import pairwise, all_equal

from atompaint.type_hints import LayerFactory, ModuleFactory, PoolFactory
from typing import TypeAlias, Iterable, Optional, Callable

GrowthTypeFactory: TypeAlias = Callable[[int], FieldType]

class DenseNet(Module):

    def __init__(
            self,
            *,
            transition_types: Iterable[FieldType],
            growth_type_factory: GrowthTypeFactory,
            initial_layer_factory: LayerFactory,
            final_layer_factory: Optional[LayerFactory],
            nonlin1_factory: ModuleFactory,
            nonlin2_factory: ModuleFactory,
            pool_factory: PoolFactory,
            pool_factors: int | list[int],
            block_depth: int | list[int],
    ):
        """
        Arguments:
            transition_types:

            growth_type:
                Field type that is the equivalent of growth rate.  In other 
                words, these are the representations that are added after each 
                dense layer.

            initial_layer_factory:
                Any steps before the dense blocks, e.g. an unpadded convolution 
                on the input voxels.

            final_layer_factory:
                Any steps after the dense blocks, e.g. to eliminate any 
                remaining spatial dimensions.

            nonlin1_factory:
                The nonlinearity to apply after the 1x1x1 convolution in each 
                layer of the dense block.

            nonlin2_factory:
                The nonlinearity to apply after the 3x3x3 convolution in each 
                layer of the dense block.

            pool_factory:

            block_depth:
                The number of layers in each dense block.
        """
        super().__init__()

        transition_types = list(transition_types)

        self.in_type = transition_types[0]
        self.out_type = transition_types[-1]

        def iter_layers():
            type_pairs = list(pairwise(transition_types))

            if initial_layer_factory:
                yield from initial_layer_factory(*type_pairs.pop(0))

            if final_layer_factory:
                final_types = type_pairs.pop(-1)

            for i, (in_type, out_type) in enumerate(type_pairs):
                block = DenseBlock(
                        in_type,
                        growth_type_factory=growth_type_factory, 
                        nonlin1_factory=nonlin1_factory,
                        nonlin2_factory=nonlin2_factory,
                        num_layers=get_scalar(block_depth, i),
                )
                yield block
                yield R3Conv(
                        block.out_type,
                        out_type,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                )
                yield pool_factory(
                        out_type,
                        get_scalar(pool_factors, i),
                )

            if final_layer_factory:
                yield from final_layer_factory(*final_types)

        self.layers = Sequential(*iter_layers())

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        assert all_equal(x.shape[-3:])
        return self.layers(x)


class DenseBlock(Module):

    def __init__(
            self,
            in_type: FieldType,
            *,
            growth_type_factory: GrowthTypeFactory,
            nonlin1_factory: ModuleFactory,
            nonlin2_factory: ModuleFactory,
            num_layers: int,
    ):
        super().__init__()

        layers = []
        next_type = in_type

        for i in range(num_layers):
            layer = DenseLayer(
                    next_type,
                    growth_type_factory=growth_type_factory,
                    nonlin1_factory=nonlin1_factory,
                    nonlin2_factory=nonlin2_factory,
            )
            layers.append(layer)

            next_type = concat_field_types(next_type, layer.out_type)

        self.layers = ModuleList(layers)
        self.in_type = in_type
        self.out_type = next_type

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        for layer in self.layers:
            x = concat_tensors_by_channel(x, layer(x))
        return x

class DenseLayer(Module):

    def __init__(
            self,
            in_type: FieldType,
            *,
            growth_type_factory: GrowthTypeFactory,
            nonlin1_factory: ModuleFactory,
            nonlin2_factory: ModuleFactory,
    ):
        super().__init__()

        out_type = growth_type_factory(1)
        mid_type = growth_type_factory(4)

        nonlin1 = nonlin1_factory(mid_type)
        nonlin2 = nonlin2_factory(out_type)

        conv1 = R3Conv(
                in_type=in_type,
                out_type=nonlin1.in_type,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
        )
        conv2 = R3Conv(
                in_type=mid_type,
                out_type=nonlin2.in_type,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
        )
        bn1 = IIDBatchNorm3d(nonlin1.in_type)
        bn2 = IIDBatchNorm3d(nonlin2.in_type)

        self.layers = Sequential(
                conv1,
                bn1,
                nonlin1,
                conv2,
                bn2,
                nonlin2,
        )
        self.in_type = in_type
        self.out_type = out_type

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        return self.layers(x)

def make_fourier_growth_type(n, *, gspace, channels, max_frequency, **kwargs):
    group = gspace.fibergroup

    return FourierFieldType(
            gspace=gspace,
            channels=n * channels,
            bl_irreps=group.bl_irreps(max_frequency),
            **kwargs,
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
