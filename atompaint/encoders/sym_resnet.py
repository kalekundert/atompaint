from __future__ import annotations

import torchyield as ty
import re

from atompaint.encoders.encoder import SymEncoder
from atompaint.encoders.resnet import ResBlock
from atompaint.pooling import FourierExtremePool3D
from atompaint.layers import sym_conv_bn_fourier_layer, sym_pool_conv_layer
from atompaint.field_types import (
        CastToFourierFieldType, add_gates, make_fourier_field_types,
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
from torch.nn import Module
from functools import partial
from multipartial import multipartial, rows
from pipeline_func import f, X

from typing import Optional, Union, Literal
from atompaint.type_hints import ConvFactory, Grid

def conv1x1x1(in_type, out_type):
    return R3Conv(
            in_type,
            out_type,
            kernel_size=1,
            padding=0,
            bias=False,
    )

class SymResBlock(ResBlock):

    def __init__(
            self,
            in_type,
            *,
            in_stride: int = 1,
            in_padding: int = 1,
            in_activation: Module,
            out_stride: int = 1,
            out_padding: int = 1,
            out_activation: Module,
            resize: Optional[Module] = None,
            resize_before_conv: bool = False,
            activation_before_skip: bool = False,
            batch_norm: bool = True,
            skip_factory: ConvFactory = conv1x1x1,
    ):
        assert resize_before_conv or in_stride or out_stride

        self.in_type = in_type
        self.out_type = out_type = out_activation.out_type

        mid_type_1 = in_activation.in_type
        mid_type_2 = in_activation.out_type
        mid_type_3 = out_activation.in_type

        if not activation_before_skip:
            assert out_activation.in_type == out_type

        if in_type == out_type:
            skip = identity
        else:
            skip = skip_factory(in_type, out_type)
            assert skip.out_type == out_type

        super().__init__(
                conv1=R3Conv(
                    in_type,
                    mid_type_1,
                    kernel_size=3,
                    stride=in_stride,
                    padding=in_padding,
                    bias=not batch_norm,
                ),
                bn1=IIDBatchNorm3d(mid_type_1) if batch_norm else identity,
                act1=in_activation,
                conv2=R3Conv(
                    mid_type_2,
                    mid_type_3,
                    kernel_size=3,
                    stride=out_stride,
                    padding=out_padding,
                    bias=not batch_norm,
                ),
                bn2=IIDBatchNorm3d(mid_type_3) if batch_norm else identity,
                act2=out_activation,
                resize=resize if resize is not None else identity,
                resize_before_conv=resize_before_conv,
                skip=skip,
                activation_before_skip=activation_before_skip,
        )

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        assert x.type == self.in_type
        y = super().forward(x)
        assert y.type == self.out_type
        return y

def load_expt_72_resnet():
    from atompaint.checkpoints import load_model_weights, strip_prefix

    def rename_old_keys(k):
        return (
                k
                | f(strip_prefix, prefix='model.encoder.encoder.layers.')
                | f(re.sub, r'(\d+)\.nonlin(\d+)\.', r'\1.act\2.', X)
                | f(re.sub, r'(\d+)\.pool\.', r'\1.resize.', X)
        )

    classifier = make_expt_72_resnet()
    load_model_weights(
            model=classifier,
            path='expt_72/padding=2-6A;angle=40deg;image-size=24A;job-id=40481465;epoch=49.ckpt',
            fix_keys=rename_old_keys,
            xxh32sum='e4b0330d',
    )
    return classifier

def make_expt_72_resnet():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    ift_grid = so3.grid('thomson_cube', N=96//24)

    return SymEncoder(
            in_channels=7,
            field_types=make_fourier_field_types(
                gspace=gspace,
                channels=[2, 4, 7, 14, 28],
                max_frequencies=2,
            ),
            head_factory=partial(
                sym_conv_bn_fourier_layer,
                ift_grid=ift_grid,
            ),
            block_factories=multipartial[:,:](
                make_alpha_block,
                mid_type=rows(
                    *make_fourier_field_types(
                        gspace=gspace,
                        channels=[2, 5, 5, 10],
                        max_frequencies=2,
                    ),
                ),
                ift_grid=ift_grid,
                pool_factor=2,
            ),
    )

def make_expt_94_resnet():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    ift_grid = so3.grid('thomson_cube', N=96//24)

    outer_channels = [2, 3, 5, 7, 11, 16]
    inner_channels = outer_channels[1:-1]

    return SymEncoder(
            in_channels=6,
            field_types=make_fourier_field_types(
                gspace=gspace,
                channels=outer_channels,
                max_frequencies=2,
            ),
            head_factory=partial(
                sym_conv_bn_fourier_layer,
                ift_grid=ift_grid,
            ),
            tail_factory=partial(
                sym_conv_bn_fourier_layer,
                kernel_size=4,
                ift_grid=ift_grid,
            ),
            block_factories=multipartial[:,:](
                make_gamma_block,
                mid_type=rows(
                    *make_fourier_field_types(
                        gspace=gspace,
                        channels=inner_channels,
                        max_frequencies=2,
                    ),
                ),
                pool_factor=rows(1, 2, 1, 2),
                ift_grid=ift_grid,
            ),
    )


def make_escnn_example_block(
        in_type: FieldType,
        out_type: FieldType,
        *,
        mid_type: Optional[FieldType] = None,
        pool_factor: int,
        ift_grid: Grid,
):
    if mid_type is None:
        mid_type = in_type

    if pool_factor == 1:
        pool = None
    else:
        pool = PointwiseAvgPoolAntialiased3D(
                in_type,
                sigma=0.33,
                stride=pool_factor,
                padding=1,
        )

    return SymResBlock(
            in_type,
            in_activation=FourierELU(mid_type, ift_grid),
            out_stride=pool_factor,
            out_activation=IdentityModule(out_type),
            resize=pool,
    )

def make_alpha_block(
        in_type: FieldType,
        out_type: FieldType,
        *,
        mid_type: Optional[FieldType] = None,
        pool_factor: int,
        ift_grid: Grid,
):
    # I couldn't think of a succinct name to describe this block, and to 
    # differentiate it from other block architectures I might want to try, so I 
    # decided to just give it the symbolic name "alpha".  I think this will be 
    # easier to think about than a longer name that somehow describes what the 
    # block does.

    if mid_type is None:
        mid_type = in_type

    gate_type = add_gates(mid_type)

    if pool_factor == 1:
        pool = None
    else:
        pool = FourierExtremePool3D(
                in_type,
                grid=ift_grid,
                kernel_size=pool_factor,
                check_input_shape=False,
        )

    return SymResBlock(
            in_type,
            in_activation=SequentialModule(
                f := GatedNonLinearity1(gate_type),
                CastToFourierFieldType(f.out_type, mid_type),
            ),
            out_activation=FourierPointwise(
                in_type=out_type,
                grid=ift_grid,
                function=leaky_hard_shrink,
            ),
            resize=pool,
            resize_before_conv=True,
    )

def make_beta_block(
        in_type: FieldType,
        out_type: FieldType,
        *,
        mid_type: Optional[FieldType] = None,
        pool_factor: int,
):
    # My intention is to use this block to closely reimplement the Wide ResNet 
    # (WRN) architecture.  That said, a lot of the actual WRN details come from 
    # other arguments to the `ResNet` class; this block really just provides 
    # the appropriate nonlinearities and pools.

    if mid_type is None:
        mid_type = in_type

    if pool_factor == 1:
        pool = None
        in_stride = 1
        in_padding = 1
    else:
        pool = PointwiseAvgPoolAntialiased3D(
                in_type,
                sigma=0.33,
                stride=pool_factor,
                padding=0,
        )
        in_stride = pool_factor
        in_padding = 0

    return SymResBlock(
            in_type,
            in_stride=in_stride,
            in_padding=in_padding,
            # Assume that the field types will already include gates.
            in_activation=GatedNonLinearity1(mid_type),
            out_activation=GatedNonLinearity1(out_type, drop_gates=False),
            resize=pool,
    )

def make_gamma_block(
        in_type: FieldType,
        out_type: FieldType,
        *,
        mid_type: Union[FieldType, Literal['in', 'out']] = 'in',
        pool_factor: int,
        ift_grid: Grid,
):
    # This block comes from Experiment #91, where I found that the combination 
    # of tensor product and first Hermite Fourier activations worked 
    # particularly well.

    if mid_type == 'in':
        mid_type = in_type
    elif mid_type == 'out':
        mid_type = out_type

    if pool_factor == 1:
        pool = []
    elif pool_factor == 2:
        pool = sym_pool_conv_layer(in_type)
    else:
        raise ValueError("`pool_factor` must be 1 or 2, not {pool_factor!r}")

    return SymResBlock(
            in_type,
            in_activation=TensorProductModule(mid_type, mid_type),
            out_activation=FourierPointwise(
                in_type=out_type,
                grid=ift_grid,
                function=first_hermite,
            ),
            resize=ty.module_from_layers(pool),
            resize_before_conv=True,
    )

