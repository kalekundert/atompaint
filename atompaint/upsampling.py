import torch.nn.functional as F

from torch import Tensor
from torch.nn import Module
from escnn.nn import EquivariantModule, FieldType, GeometricTensor
from typing import Callable, Literal

# The class names in this module are a bit confusing.  `R3Upsampling` is 
# equivariant, and follows the naming conventions of `escnn`.  `Upsample3d` is 
# not equivariant, and follows the naming conventions the builtin torch 
# modules.

class R3Upsampling(EquivariantModule):
    # This is a reimplementation of the `R3Upsampling` module from escnn, with 
    # the only difference being that the size is given as a callable rather 
    # than an integer.

    def __init__(
            self,
            in_type: FieldType,
            *,
            size_expr: Callable[[int], int] = lambda x: 2*x,
            align_corners: bool = False,
    ):
        super().__init__()

        self.in_type = in_type
        self.out_type = in_type
        self.size_expr = size_expr
        self.align_corners = align_corners

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        assert x.type == self.in_type
        assert len(x.shape) == 5

        *_, w, h, d = x.shape
        assert w == h == d

        y = F.interpolate(
                x.tensor,
                size=self.size_expr(w),
                align_corners=self.align_corners,

                # The only modes applicable to 3D inputs are 'nearest' and 
                # 'trilinear', and 'nearest' is not equivariant, so we always 
                # use 'trilinear'.
                mode='trilinear',
        )

        return GeometricTensor(y, self.out_type, coords=None)

    def evaluate_output_shape(self, input_shape):
        return input_shape

class Upsample3d(Module):

    def __init__(
            self,
            *,
            size_expr: Callable[[int], int] = lambda x: 2*x,
            align_corners: bool = False,
            mode: Literal['nearest', 'trilinear'] = 'nearest',
    ):
        super().__init__()
        self.size_expr = size_expr
        self.align_corners = align_corners
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 5

        w, h, d = x.shape[-3:]
        assert w == h == d

        return F.interpolate(
                x,
                size=self.size_expr(w),
                mode=self.mode,
                align_corners=self.align_corners,
        )

