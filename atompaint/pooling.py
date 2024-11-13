from __future__ import annotations

import torch
import torch.nn as nn
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
from pipeline_func import f

from typing import Optional, Union, List
from torch import Tensor

class FourierExtremePool3D(torch.nn.Module):
    """
    Downsample Fourier-domain input by extreme-pooling in the spatial domain.

    This operation is equivariant with respect to both translation ans 
    rotation.  Translational equivariance is maintained by performing the 
    pooling with stride=1, then downsampling via a Gaussian blur filter.  
    Rotational equivariance is maintained by performing the pooling in the 
    spatial domain, where any operation that treats every spatial location in 
    the same way will be equivariant.

    Extreme-pooling is similar to max-pooling, except that the values with the 
    greatest magnitudes---positive or negative---are the ones that are kept.  
    This works better in the context of spatial domain values that will be 
    subsequently returned to the Fourier domain.  Max-pooling would introduce a 
    positive bias, which would result in the frequency=0 component of the 
    Fourier vector being much greater in magnitude than the others.

    After applying this layer, an input of size $N$ would be downsampled to 
    $\frac{N - k}{s} + 1$, where $k$ and $s$ are the kernel size and stride 
    parameters, respectively.  An error will be raised if the above expression 
    doesn't result in a whole number, because such inputs would break 
    rotational equivariance.
    """

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
            check_input_shape: bool = True,
    ):
        """
        Arguments:
            check_input_shape:
                This layer is only equivariant if the internal filters align 
                perfectly with the edges of the inputs.  If this argument is 
                True, then an error will be triggered if this is not the case.  
                Otherwise, the check will be skipped.  Practically, the model 
                may still be acceptably equivariant even with this condition 
                violated.
        """
        super().__init__()

        check_dimensions(in_type, d := 3)

        self.d = d
        self.in_type = in_type
        self.out_type = in_type
        self.check_input_shape = check_input_shape

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
        w, h, d = x_hat_wrap.shape[-3:]
        assert w == h == d
        if self.check_input_shape:
            assert (w - self.pool.kernel_size) % self.blur.stride == 0

        x_wrap = self.ift(x_hat_wrap)

        b, c, g = x_wrap.tensor.shape[:3]
        x = x_wrap.tensor.reshape(b, c*g, w, h, d)

        y = self.pool(x)
        y = self.blur(y)

        w2, h2, d2 = y.shape[-3:]
        y = y.reshape(b, c, g, w2, h2, d2)
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
        x = x_wrap.tensor.reshape(b, c*g, *xyz)

        y = self.blur(x)

        b, _, *xyz = y.shape
        y = y.reshape(b, c, g, *xyz)
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

class AntialiasedPool3D(torch.nn.Module):

    def __init__(
            self,
            *,
            pool_factory,
            kernel_size: Union[int, tuple[int, int, int]],
            stride: Optional[int] = None,
            padding: int = 0,
            sigma: float = 0.6,
            check_input_shape: bool = True,
    ):
        super().__init__()

        self.pool = pool_factory(
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
        )
        self.blur = GaussianBlurND(
                sigma=sigma,
                kernel_size=kernel_size_from_radius(sigma * 4),
                stride=stride if stride is not None else kernel_size,
                rel_padding=padding,
                edge_correction=True,
                d=3,
        )
        self.check_input_shape = check_input_shape

    def forward(self, x: Tensor) -> Tensor:
        w, h, d = x.shape[-3:]
        assert w == h == d
        if self.check_input_shape:
            assert (w - self.pool.kernel_size) % self.blur.stride == 0

        return (
                x
                | f(self.pool)
                | f(self.blur)
        )

class AntialiasedMaxPool3D(AntialiasedPool3D):

    def __init__(
            self,
            *,
            kernel_size: Union[int, tuple[int, int, int]],
            stride: Optional[int] = None,
            padding: int = 0,
            sigma: float = 0.6,
            check_input_shape: bool = True,
    ):
        super().__init__(
                self,
                pool_factory=nn.MaxPool3d,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                sigma=sigma,
                check_input_shape=check_input_shape,
        )

class AntialiasedAvgPool3D(AntialiasedPool3D):

    def __init__(
            self,
            *,
            kernel_size: Union[int, tuple[int, int, int]],
            stride: Optional[int] = None,
            padding: int = 0,
            sigma: float = 0.6,
            check_input_shape: bool = True,
    ):
        super().__init__(
                self,
                pool_factory=nn.AvgPool3d,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                sigma=sigma,
                check_input_shape=check_input_shape,
        )

