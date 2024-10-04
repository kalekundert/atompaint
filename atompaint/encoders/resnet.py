from __future__ import annotations

import torchyield as ty

from atompaint.pooling import FourierExtremePool3D
from atompaint.layers import conv_bn_fourier_layer, pool_conv_layer
from atompaint.field_types import (
        CastToFourierFieldType, add_gates,
        make_trivial_field_types, make_fourier_field_types,
)
from atompaint.nonlinearities import leaky_hard_shrink, first_hermite
from atompaint.utils import identity
from escnn.gspaces import rot3dOnR3
from escnn.nn import (
        FieldType, GeometricTensor,
        R3Conv, IIDBatchNorm3d, PointwiseAvgPoolAntialiased3D, 
        FourierPointwise, FourierELU, GatedNonLinearity1, TensorProductModule,
        SequentialModule, IdentityModule,
)
from torch.nn import Module, Sequential
from more_itertools import pairwise, zip_broadcast, all_equal
from itertools import chain
from functools import partial

from typing import Optional, Callable
from atompaint.type_hints import LayerFactory, ConvFactory, Grid

def conv3x3x3(in_type, out_type, stride=1, padding=1):
    return R3Conv(
            in_type,
            out_type,
            kernel_size=3,
            stride=stride,
            padding=padding,

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
                    yield block_factory(i, j, out_type, mid_type, out_type, 1)

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

class ResBlock(Module):

    def __init__(
            self,
            in_type,
            out_type,
            *,
            in_stride: int = 1,
            in_padding: int = 1,
            out_stride: int = 1,
            out_padding: int = 1,
            mid_nonlinearity: Module,
            out_nonlinearity: Module,
            pool: Optional[Module] = None,
            pool_before_conv: bool = False,
            skip_factory: ConvFactory = conv1x1x1,
    ):
        super().__init__()

        assert pool_before_conv or in_stride or out_stride

        self.in_type = in_type
        self.out_type = out_type

        hidden_type_1 = mid_nonlinearity.in_type
        hidden_type_2 = mid_nonlinearity.out_type

        assert out_nonlinearity.in_type == out_type
        assert out_nonlinearity.out_type == out_type

        self.conv1 = conv3x3x3(
                in_type,
                hidden_type_1,
                stride=in_stride,
                padding=in_padding,
        )
        self.bn1 = IIDBatchNorm3d(hidden_type_1)
        self.nonlin1 = mid_nonlinearity
        self.conv2 = conv3x3x3(
                hidden_type_2,
                out_type,
                stride=out_stride,
                padding=out_padding,
        )
        self.bn2 = IIDBatchNorm3d(out_type)
        self.nonlin2 = out_nonlinearity
        self.pool = pool if pool is not None else identity
        self.pool_before_conv = pool_before_conv

        if in_type == out_type:
            self.skip = lambda x: x
        else:
            self.skip = skip_factory(in_type, out_type)

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

def load_expt_72_resnet(*, device=None):
    from atompaint.checkpoints import load_model_weights

    classifier = make_expt_72_resnet()
    load_model_weights(
            model=classifier,
            path='expt_72/padding=2-6A;angle=40deg;image-size=24A;job-id=40481465;epoch=49.ckpt',
            prefix='model.encoder.encoder.',
            xxh32sum='e4b0330d',
            device=device,
    )
    return classifier

def make_expt_72_resnet():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    ift_grid = so3.grid('thomson_cube', N=96//24)

    return ResNet(
            outer_types = chain(
                make_trivial_field_types(
                    gspace=gspace, 
                    channels=[7],
                ),
                make_fourier_field_types(
                    gspace=gspace,
                    channels=[2, 4, 7, 14, 28],
                    max_frequencies=2,
                ),
            ),
            inner_types = make_fourier_field_types(
                    gspace=gspace, 
                    channels=[2, 5, 5, 10],
                    max_frequencies=2,
            ),
            initial_layer_factory = partial(
                    conv_bn_fourier_layer,
                    ift_grid=ift_grid,
            ),
            block_factory=partial(
                    make_alpha_block,
                    grid=ift_grid,
            ),
            block_repeats=1,
            pool_factors=[2, 2, 2, 2],
    )

def make_expt_94_resnet():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    ift_grid = so3.grid('thomson_cube', N=96//24)

    outer_channels = [2, 3, 5, 7, 11, 16]
    inner_channels = outer_channels[1:-1]

    return ResNet(
            outer_types=chain(
                make_trivial_field_types(
                    gspace=gspace, 
                    channels=[6],
                ),
                make_fourier_field_types(
                    gspace=gspace,
                    channels=outer_channels,
                    max_frequencies=2,
                ),
            ),
            inner_types=make_fourier_field_types(
                gspace=gspace, 
                channels=inner_channels,
                max_frequencies=2,
            ),
            initial_layer_factory=partial(
                conv_bn_fourier_layer,
                ift_grid=ift_grid,
            ),
            final_layer_factory=partial(
                conv_bn_fourier_layer,
                kernel_size=4,
                ift_grid=ift_grid,
            ),
            block_factory=partial(
                make_gamma_block,
                ift_grid=ift_grid,
            ),
            block_repeats=1,
            pool_factors=[1, 2, 1, 2],
    )


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
            out_stride=pool_factor if j == 0 else 1,
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
                check_input_shape=False,
        )

    return ResBlock(
            in_type,
            out_type,
            mid_nonlinearity=SequentialModule(
                f := GatedNonLinearity1(gate_type),
                CastToFourierFieldType(f.out_type, mid_type),
            ),
            out_nonlinearity=FourierPointwise(
                in_type=out_type,
                grid=grid,
                function=leaky_hard_shrink,
            ),
            pool=pool,
            pool_before_conv=True,
    )

def make_beta_block(
        i: int,
        j: int,
        in_type: FieldType,
        mid_type: FieldType,
        out_type: FieldType,
        pool_factor: int,
):
    # My intention is to use this block to closely reimplement the Wide ResNet 
    # (WRN) architecture.  That said, a lot of the actual WRN details come from 
    # other arguments to the `ResNet` class; this block really just provides 
    # the appropriate nonlinearities and pools.

    if j == 0:
        pool = PointwiseAvgPoolAntialiased3D(
                in_type,
                sigma=0.33,
                stride=pool_factor,
                padding=0,
        )
        in_stride = pool_factor
        in_padding = 0

    else:
        pool = None
        in_stride = 1
        in_padding = 1

    return ResBlock(
            in_type,
            out_type,
            in_stride=in_stride,
            in_padding=in_padding,
            # Assume that the field types will already include gates.
            mid_nonlinearity=GatedNonLinearity1(mid_type),
            out_nonlinearity=GatedNonLinearity1(out_type, drop_gates=False),
            pool=pool,
            #skip_factory=skip_concat,
    )

def make_gamma_block(
        i: int,
        j: int,
        in_type: FieldType,
        mid_type: FieldType,
        out_type: FieldType,
        pool_factor: int,
        ift_grid: Grid,
):
    # This block comes from Experiment #91, where I found that the combination 
    # of tensor product and first Hermite Fourier activations worked 
    # particularly well.

    if pool_factor == 1:
        pool = []
    elif pool_factor == 2:
        pool = pool_conv_layer(in_type)
    else:
        raise ValueError("`pool_factor` must be 1 or 2, not {pool_factor!r}")

    return ResBlock(
            in_type,
            out_type,
            mid_nonlinearity=TensorProductModule(mid_type, mid_type),
            out_nonlinearity=FourierPointwise(
                in_type=out_type,
                grid=ift_grid,
                function=first_hermite,
            ),
            pool=ty.module_from_layers(pool),
            pool_before_conv=True,
    )


def skip_concat(in_type, out_type):
    # Not so easy:
    # - Can't just concat tensors, because representations might not be in the 
    #   same order.  In fact, they certainly won't be if I use my 
    #   exact-channels algorithm.
    #
    # - Pseudocode:
    #   - Find matching representations.
    #     - No general way to do this, but I can go by size and that will work 
    #       for my purposes.
    #   - Make sure `out_type` has 2x each `in_type` representation.
    #   - Create empty tensor of correct size
    #   - Copy in each field from the input tensor, as appropriate.

    # Warning: this function assumes that if two representations are the same 
    # size, then they are in fact the same.  This is not true in general, so 
    # make sure that your representations have this property before using this 
    # function!

    pass

