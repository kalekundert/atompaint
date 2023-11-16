from __future__ import annotations

import torch
import torch.nn.functional as F

from escnn.nn import (
        FourierFieldType, GeometricTensor, GridTensor,
        InverseFourierTransform, FourierTransform,
)
from escnn.nn.modules.pooling.gaussian_blur import (
        GaussianBlurND, kernel_size_from_radius,
)
from escnn.nn.modules.pooling.pointwise import check_dimensions
from escnn.group import GroupElement

from typing import Optional, Union, List

class FourierExtremePool3D(torch.nn.Module):

    def __init__(
            self,
            in_type: FourierFieldType,
            grid: List[GroupElement],
            *,
            kernel_size: Union[int, tuple[int, int, int]],
            stride: Optional[int] = None,
            sigma: float = 0.6,
            normalize: bool = True,
            extra_irreps: List = [],
    ):
        super().__init__()

        check_dimensions(in_type, d := 3)

        self.d = d
        self.in_type = in_type
        self.out_type = in_type

        self.ift = InverseFourierTransform(
                in_type, grid,
                normalize=normalize,
        )
        self.ft = FourierTransform(
                grid, self.out_type,
                extra_irreps=in_type.bl_irreps + extra_irreps,
                normalize=normalize,
        )
        self.pool = ExtremePool3D(
                kernel_size=kernel_size,
                stride=1,
        )
        self.blur = GaussianBlurND(
                sigma=sigma,
                kernel_size=kernel_size_from_radius(sigma * 4),
                stride=stride if stride is not None else kernel_size,
                d=d,
                edge_correction=True,
        )

    def forward(self, x_hat_wrap: GeometricTensor) -> GeometricTensor:
        # check that all spatial dimensions are odd.  Otherwise: edge effects.
        x_wrap = self.ift(x_hat_wrap)

        b, c, g, *xyz = x_wrap.tensor.shape
        x = x_wrap.tensor.reshape(b, c*g, *xyz)

        y = self.pool(x)
        y = self.blur(y)

        b, _, *xyz = y.shape
        y = y.reshape(b, c, g, *xyz)
        y_wrap = GridTensor(y, x_wrap.grid, x_wrap.coords)

        return self.ft(y_wrap)


class FourierAvgPool3D(torch.nn.Module):

    def __init__(
            self,
            in_type: FourierFieldType,
            grid: List[GroupElement],
            *,
            stride: int,
            sigma: float = 0.6,
            normalize: bool = True,
            extra_irreps: List = [],
    ):
        super().__init__()

        check_dimensions(in_type, d := 3)

        self.d = d
        self.in_type = in_type
        self.out_type = in_type

        self.ift = InverseFourierTransform(
                in_type, grid,
                normalize=normalize,
        )
        self.ft = FourierTransform(
                grid, self.out_type,
                extra_irreps=in_type.bl_irreps + extra_irreps,
                normalize=normalize,
        )
        self.blur = GaussianBlurND(
                sigma=sigma,
                kernel_size=kernel_size_from_radius(sigma * 4),
                stride=stride,
                d=d,
                edge_correction=True,
        )

    def forward(self, x_hat_wrap: GeometricTensor) -> GeometricTensor:
        x_wrap = self.ift(x_hat_wrap)

        b, c, g, *xyz = x_wrap.tensor.shape
        x = x_wrap.tensor.view(b, c*g, *xyz)

        y = self.blur(x)

        b, _, *xyz = y.shape
        y = y.view(b, c, g, *xyz)
        y_wrap = GridTensor(y, x_wrap.grid, x_wrap.coords)

        return self.ft(y_wrap)


class ExtremePool3D(torch.nn.Module):

    def __init__(
            self,
            kernel_size,
            stride=None,
            padding=0,
            dilation=1,
            ceil_mode=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x):
        _, i = F.max_pool3d(
                x.abs(),
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.ceil_mode,
                return_indices=True,
        )
        b, c, *d = x.shape

        # I feel like making all these views can't be the simplest way to do 
        # the necessary indexing, but at least it works.
        x_flat = x.view(b, c, -1)
        i_flat = i.view(b, c, -1)

        y = torch.gather(x_flat, 2, i_flat)

        y = y.view(*i.shape)
        return y
