from escnn.nn import (
        FieldType, FourierFieldType,
        R3Conv, IIDBatchNorm3d, FourierPointwise, GatedNonLinearity1,
)
from atompaint.field_types import add_gates
from atompaint.type_hints import Grid

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

def make_conv_gated_layer(
        in_type: FieldType,
        out_type: FieldType,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
):
    gate_type = add_gates(out_type)
    yield R3Conv(
            in_type,
            gate_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
    )
    yield IIDBatchNorm3d(gate_type)
    yield GatedNonLinearity1(gate_type)

def make_gated_nonlinearity(out_type):
    in_type = add_gates(out_type)
    return GatedNonLinearity1(in_type)

