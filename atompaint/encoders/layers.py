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
    yield make_trivial_field_type(gspace, channels[0])
    yield from make_nontrivial_field_types(gspace, channels[1:])

def make_trivial_field_type(gspace, channels):
    return FieldType(gspace, channels * [gspace.trivial_repr])

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
    # Avoid generating the tensor product of the zeroth and first irrep.  The 
    # resulting representation doesn't have any "supported nonlinearities", see 
    # the if-statement at the end of `Group._tensor_product()`.  This is 
    # probably a bug in escnn, but the work-around is easy.

    rho_0 = group.irrep(0)
    rho_1 = rho_next = group.irrep(1)

    yield rho_0
    yield rho_1

    while True:
        rho_next = rho_next.tensor(rho_1)
        yield rho_next
