from escnn.nn import (
        FieldType, FourierFieldType,
        R3Conv, IIDBatchNorm3d, FourierPointwise, GatedNonLinearity1,
)
from atompaint.nonlinearities import add_gates
from atompaint.utils import get_scalar
from atompaint.type_hints import Grid
from more_itertools import take

def make_conv_layer(
        in_type: FieldType,
        out_type: FieldType,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
):
    # Note that this layer has no nonlinearity.  So if it's followed 
    # immediately by another convolutional layer, it's equivalent to a single 
    # linear operation.
    yield R3Conv(
            in_type,
            out_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
    )

def make_conv_fourier_layer(
        in_type: FourierFieldType,
        out_type: FourierFieldType,
        ift_grid: Grid,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        nonlinearity: str = 'p_relu',
):
    yield R3Conv(
            in_type,
            out_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,

            # Batch-normalization will recenter everything on 0, so there's no 
            # point having a bias just before that.
            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm
            bias=False
    )
    yield IIDBatchNorm3d(out_type)
    yield FourierPointwise(
            out_type,
            ift_grid,
            function=nonlinearity,
    )

def make_gated_nonlinearity(out_type):
    in_type = add_gates(out_type)
    return GatedNonLinearity1(in_type)

def make_top_level_field_types(gspace, channels, make_nontrivial_field_types):
    group = gspace.fibergroup
    yield FieldType(gspace, channels[0] * [group.trivial_representation])
    yield from make_nontrivial_field_types(gspace, channels[1:])

def make_fourier_field_types(gspace, channels, max_frequencies, **kwargs):
    group = gspace.fibergroup

    for i, channels_i in enumerate(channels):
        max_freq = get_scalar(max_frequencies, i)
        bl_irreps = group.bl_irreps(max_freq)
        yield FourierFieldType(gspace, channels_i, bl_irreps, **kwargs)

def make_polynomial_field_types(gspace, channels, terms):
    for i, channels_i in enumerate(channels):
        terms_i = get_scalar(terms, i)
        assert terms_i > 0

        rho = take(terms_i, iter_polynomial_representations(gspace.fibergroup))
        yield FieldType(gspace, channels_i * list(rho))

def iter_polynomial_representations(group):
    rho_next = group.irrep(0)
    rho_1 = group.irrep(1)

    yield rho_next

    while True:
        rho_next = rho_next.tensor(rho_1)
        yield rho_next
