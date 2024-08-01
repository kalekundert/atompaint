import torch.nn.functional as F

from escnn.nn import EquivariantModule, FieldType, GeometricTensor
from typing import Callable

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

                # The only modes applicable to 3D inputs are 'nearest' and 
                # 'trilinear', and 'nearest' is not equivariant, so we always 
                # use 'trilinear'.
                mode='trilinear',
                align_corners=self.align_corners,
        )

        return GeometricTensor(y, self.out_type, coords=None)

    def evaluate_output_shape(self, input_shape):
        return input_shape
