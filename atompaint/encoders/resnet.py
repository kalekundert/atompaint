from __future__ import annotations

from atompaint.pooling import FourierExtremePool3D
from atompaint.nonlinearities import SetFourierFieldType, add_gates
from atompaint.utils import identity
from escnn.nn import (
        FieldType, GeometricTensor,
        R3Conv, FourierPointwise, FourierELU, GatedNonLinearity1,
        IIDBatchNorm3d, PointwiseAvgPoolAntialiased3D, SequentialModule,
        IdentityModule,
)
from torch.nn import Module, Sequential
from more_itertools import pairwise, zip_broadcast, all_equal

from typing import Optional, Callable
from atompaint.type_hints import LayerFactory, Grid

# Hyperparameters of interest:
#
# [ ] First vs second nonlinearity
# [ ] Downsampling:
#     - Skip connection: Pool vs stride
#     - Backbone: Fourier ReLU+pool, vs separate nonlinearity/downsample.
# [ ] Bottleneck
# [ ] Network depth:
#     - Blocks, block repeats, pooling factors, etc.

class ResBlock(Module):

    def __init__(
            self,
            in_type,
            out_type,
            *,
            stride: int = 1,
            mid_nonlinearity: Module,
            out_nonlinearity: Module,
            pool: Optional[Module] = None,
            pool_before_conv: bool = False,
    ):
        super().__init__()

        self.in_type = in_type
        self.out_type = out_type

        hidden_type_1 = mid_nonlinearity.in_type
        hidden_type_2 = mid_nonlinearity.out_type

        self.conv1 = conv3x3x3(in_type, hidden_type_1)
        self.bn1 = IIDBatchNorm3d(hidden_type_1)
        self.nonlin1 = mid_nonlinearity
        self.conv2 = conv3x3x3(hidden_type_2, out_type, stride)
        self.bn2 = IIDBatchNorm3d(out_type)
        self.nonlin2 = out_nonlinearity
        self.pool = pool if pool is not None else identity
        self.pool_before_conv = pool_before_conv

        assert self.nonlin2.out_type == out_type

        if in_type == out_type:
            self.skip = lambda x: x
        else:
            self.skip = conv1x1x1(in_type, out_type)

    def forward(self, x: GeometricTensor):
        if self.pool_before_conv:
            x = self.pool(x)

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.nonlin1(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if not self.pool_before_conv:
            x = self.pool(x)

        if self.skip is not None:
            x = self.skip(x)

        y = self.nonlin2(x + y)

        return y


class ResNet(Module):

    def __init__(
            self,
            *,
            outer_types,
            inner_types,
            initial_layer_factory: LayerFactory,
            final_layer_factory: Optional[LayerFactory] = None,
            block_factory: Callable[[int, int, FieldType, FieldType, FieldType, int], Module], 
            block_repeats,
            pool_factors,
    ):
        super().__init__()

        outer_types = list(outer_types)
        inner_types = list(inner_types)

        assert len(outer_types) == len(inner_types) + 2 + (final_layer_factory is not None)

        self.in_type = outer_types[0]
        self.out_type = outer_types[-1]

        def iter_layer_params():
            for i, ((in_type, out_type), mid_type, n_repeats, pool_factor) in enumerate(
                    zip_broadcast(
                        pairwise(outer_types),
                        inner_types,
                        block_repeats,
                        pool_factors,
                        strict=True,
                    )
            ):
                yield i, in_type, mid_type, out_type, n_repeats, pool_factor

        def iter_blocks():
            for i, in_type, mid_type, out_type, n_repeats, pool_factor in iter_layer_params():
                yield block_factory(i, 0, in_type, mid_type, out_type, pool_factor)

                for j in range(1, n_repeats):
                    yield block_factory(i, j, in_type, mid_type, out_type, 1)

        initial_layer = initial_layer_factory(*outer_types[:2])
        outer_types = outer_types[1:]

        if final_layer_factory:
            final_layer = final_layer_factory(*outer_types[-2:])
            outer_types = outer_types[:-1]
        else:
            final_layer = []

        self.layers = Sequential(
                *initial_layer,
                *iter_blocks(),
                *final_layer,
        )

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        assert all_equal(x.shape[-3:])
        return self.layers(x)

def make_escnn_example_block(
        i: int,
        j: int,
        in_type: FieldType,
        mid_type: FieldType,
        out_type: FieldType,
        pool_factor: int,
        grid: Grid,
):
    return ResBlock(
            in_type,
            out_type,
            stride=pool_factor if j == 0 else 1,
            mid_nonlinearity=FourierELU(mid_type, grid),
            out_nonlinearity=IdentityModule(out_type),
            pool=PointwiseAvgPoolAntialiased3D(
                in_type,
                sigma=0.33,
                stride=pool_factor,
                padding=1,
            ),
    )

def make_alpha_block(
        i: int,
        j: int,
        in_type: FieldType,
        mid_type: FieldType,
        out_type: FieldType,
        pool_factor: int,
        grid: Grid,
):
    # I couldn't think of a succinct name to describe this block, and to 
    # differentiate it from other block architectures I might want to try, so I 
    # decided to just give it the symbolic name "alpha".  I think this will be 
    # easier to think about than a longer name that somehow describes what the 
    # block does.

    gate_type = add_gates(mid_type)

    if pool_factor == 1:
        pool = None
    else:
        pool = FourierExtremePool3D(
                in_type,
                grid=grid,
                kernel_size=pool_factor,
        )

    return ResBlock(
            in_type,
            out_type,
            mid_nonlinearity=SequentialModule(
                f := GatedNonLinearity1(gate_type),
                SetFourierFieldType(f.out_type, mid_type),
            ),
            out_nonlinearity=SequentialModule(
                FourierPointwise(
                    in_type=out_type,
                    grid=grid,
                    function=lambda x: x**3,
                ),
                IIDBatchNorm3d(out_type),
            ),
            pool=pool,
            pool_before_conv=True,
    )

def conv3x3x3(in_type, out_type, stride=1):
    return R3Conv(
            in_type,
            out_type,
            kernel_size=3,
            padding=1,
            stride=stride,

            # Batch-normalization will recenter everything on 0, so there's no 
            # point having a bias just before that.
            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm
            bias=False,
    )

def conv1x1x1(in_type, out_type):
    return R3Conv(
            in_type,
            out_type,
            kernel_size=1,
            padding=0,
            bias=False,
    )
