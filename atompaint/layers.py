import torch
import torch.nn as nn
import torch.nn.functional as F

from atompaint.field_types import add_gates
from atompaint.pooling import FourierExtremePool3D, FourierAvgPool3D
from atompaint.upsampling import R3Upsampling
from atompaint.type_hints import Grid
from escnn.nn import (
        FieldType, FourierFieldType, GeometricTensor,
        R3Conv, Linear, IIDBatchNorm3d, IIDBatchNorm1d,
        FourierPointwise, GatedNonLinearity1, NormNonLinearity,
        PointwiseAvgPoolAntialiased3D,
)
from escnn.gspaces import no_base_space

class WrapTensor(nn.Module):

    def __init__(self, in_type):
        super().__init__()
        self.in_type = in_type

    def forward(self, x: torch.Tensor) -> GeometricTensor:
        return GeometricTensor(x, self.in_type)

class UnwrapTensor(nn.Module):

    def forward(self, x: GeometricTensor) -> torch.Tensor:
        return x.tensor

class Require1x1x1(nn.Module):

    def forward(self, x):
        if isinstance(x, GeometricTensor):
            y = x.tensor
        else:
            y = x

        assert len(y.shape) == 5
        assert y.shape[2:] == (1, 1, 1)

        return x

def sym_conv_layer(
        in_type: FieldType,
        out_type: FieldType,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
):
    yield R3Conv(
            in_type,
            out_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
    )

def sym_conv_bn_norm_layer(
        in_type: FieldType,
        out_type: FieldType,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        function: str = 'n_relu',
        bias: bool = True,
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
            bias=False,
    )
    yield IIDBatchNorm3d(out_type)
    yield NormNonLinearity(out_type, function=function, bias=bias)

def sym_conv_bn_gated_layer(
        in_type: FieldType,
        out_type: FieldType,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        function=F.sigmoid,
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
    yield GatedNonLinearity1(gate_type, function=function)

def sym_conv_bn_fourier_layer(
        in_type: FieldType,
        out_type: FourierFieldType,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        ift_grid: Grid,
        function='p_elu',
):
    yield R3Conv(
            in_type,
            out_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
    )
    yield IIDBatchNorm3d(out_type)
    yield FourierPointwise(out_type, ift_grid, function=function)

def sym_linear_fourier_layer(
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

def sym_gated_layer(out_type, *, function=F.sigmoid):
    in_type = add_gates(out_type)
    return GatedNonLinearity1(in_type, function=function)

def sym_pool_conv_layer(in_type, *, padding=1):
    return R3Conv(
            in_type,
            in_type,
            kernel_size=3,
            stride=2,
            padding=padding,
    )

def sym_pool_avg_layer(in_type):
    return PointwiseAvgPoolAntialiased3D(
            in_type,
            sigma=0.6,
            stride=2,
    )

def sym_pool_fourier_avg_layer(in_type, ift_grid):
    return FourierAvgPool3D(
            in_type,
            grid=ift_grid,
            stride=2,
    )

def sym_pool_fourier_extreme_layer(in_type, ift_grid, *, check_input_shape=True):
    return FourierExtremePool3D(
            in_type,
            grid=ift_grid,
            kernel_size=2,
            check_input_shape=check_input_shape,
    )

def sym_up_pool_fourier_extreme_layer(in_type, ift_grid):
    # To avoid edge effects, convolution layers need odd-sized inputs and 
    # Fourier pooling layers need even-sized ones.  One way to accommodate both 
    # is to upsample the input by one before pooling.
    yield R3Upsampling(
            in_type,
            size_expr=lambda x: x+1,
            align_corners=True,
    )
    yield FourierExtremePool3D(
            in_type,
            grid=ift_grid,
            kernel_size=2,
    )

def invariant_conv_layer(in_type, out_channels, kernel_size, **kwargs):
    out_type = FieldType(
            in_type.gspace,
            out_channels * [in_type.gspace.trivial_repr],
    )
    yield R3Conv(in_type, out_type, kernel_size, **kwargs)
    yield UnwrapTensor()
    yield Require1x1x1()
    yield nn.Flatten()

def invariant_fourier_layer(
        in_type: FourierFieldType,
        *,
        ift_grid: Grid,
        function='p_elu',
):
    # This only works if the input size is 1x1x1, or if everything is 
    # average/max-pooled right afterward.  Otherwise, rotating the input will 
    # result in output that has all the same values, but in different 
    # positions.  Subsequent steps will break equivariance.
    yield Require1x1x1()

    out_type = FourierFieldType(
            in_type.gspace, 
            channels=in_type.channels,
            bl_irreps=in_type.gspace.fibergroup.bl_irreps(0),
            subgroup_id=in_type.subgroup_id,
    )
    yield FourierPointwise(
            in_type=in_type,
            out_type=out_type,
            grid=ift_grid,
            function=function,
    )
    yield UnwrapTensor()
    yield nn.Flatten()

def invariant_fourier_pool_layer(
        in_type: FourierFieldType,
        *,
        ift_grid: Grid,
        function='p_elu',
        pool: nn.Module,
):
    out_type = FourierFieldType(
            in_type.gspace, 
            channels=in_type.channels,
            bl_irreps=in_type.gspace.fibergroup.bl_irreps(0),
            subgroup_id=in_type.subgroup_id,
    )
    yield FourierPointwise(
            in_type=in_type,
            out_type=out_type,
            grid=ift_grid,
            function=function,
    )
    yield UnwrapTensor()
    yield pool
    yield Require1x1x1()
    yield nn.Flatten()


def flatten_base_space(geom_tensor):
    """
    Remove the spatial dimensions from the given geometric tensor.

    All of the spatial dimensions in the input must be of size 1, so they can 
    be removed without losing information or changing the size of any other 
    dimension.  If this condition is not met, an assertion error will be 
    raised.

    The return value is still a geometric tensor, but with a 0D base space (see 
    `no_base_space()`) instead of whatever the original base space was.  The 
    fiber representations are unchanged.
    """

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
