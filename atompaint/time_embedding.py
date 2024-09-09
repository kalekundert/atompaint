import torch
import torch.nn as nn

from torch import Tensor
from escnn.nn import (
        GeometricTensor, GridTensor, FieldType, FourierFieldType, Linear,
        InverseFourierTransform, FourierTransform, GatedNonLinearity2
)
from escnn.gspaces import no_base_space
from escnn.group import GroupElement
from einops.layers.torch import Rearrange
from einops import repeat
from math import tau, log2

class AddTimeToImage(nn.Module):

    def __init__(self, time_dim, channel_dim, latent_dim=None):
        super().__init__()

        if not latent_dim:
            latent_dim = min(time_dim, channel_dim * 4)

        self.time_mlp = nn.Sequential(
                nn.Linear(time_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, channel_dim * 2),
                Rearrange('b c -> b c 1 1 1'),
        )

    def forward(self, x, t):
        t = self.time_mlp(t)
        m, b = t.chunk(2, dim = 1)
        return m * x + b

class SinusoidalEmbedding(nn.Module):

    def __init__(
            self,
            out_dim: int,
            *,
            min_wavelength: float = 4,
            max_wavelength: float = 1e4/tau,
    ):
        """
        Arguments:
            out_dim:
                The size of the output embedding dimension.  This number must 
                be even, and must be greater than 1.

            min_wavelength:
                The shortest wavelength to use in the embedding.  In other 
                words, the number of indices required for the fastest-changing 
                embedding dimension to make a full cycle.  The default is 4, 
                which works well for indices that increment by 1.

            max_wavelength:
                The largest wavelength to use in the embedding.  In other 
                words, the number of indices required for the slowest-changing 
                embedding dimension to make a full cycle.  Each dimension of 
                the output embedding uses a different wavelength.
        """
        super().__init__()

        if out_dim % 2 != 0:
            raise ValueError(f"output dimension must be even, not {out_dim}")
        if out_dim < 2:
            raise ValueError(f"output dimension must be greater than 1, not {out_dim}")

        self.out_dim = out_dim
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength

    def forward(self, t):
        """
        Arguments:
            t: torch.Tensor
                A 1D tensor of diffusion timepoints.  Typically, the values 
                in this tensor make up one minibatch.

        Returns:
            A 2D tensor of dimension (B, D):

            - B: minibatch size
            - D: output embedding size, i.e. the *out_dim* argument provided to 
              the constructor.

            Earlier positions in the output embedding change rapidly as a 
            function of the input index, while later positions change slowly.  
            This embedding is typically added to whatever input data is 
            associated with each index (images, in this project).
        """
        assert len(t.shape) == 1

        freq = tau / torch.logspace(
                start=log2(self.min_wavelength),
                end=log2(self.max_wavelength),
                steps=self.out_dim // 2,
                base=2,
                device=t.device,
        )
        theta = torch.outer(t, freq)
        return torch.cat((torch.sin(theta), torch.cos(theta)), dim=-1)

class FourierTimeActivation(nn.Module):
    """
    Integrate a time embedding into a geometric tensor, and then apply an 
    activation function.

    In order to maintain equivariance, both steps are done in the Fourier 
    domain.
    """

    def __init__(
            self,
            in_type: FourierFieldType,
            grid: list[GroupElement],
            *,
            time_dim: int,
            activation: nn.Module = nn.SELU(),
            normalize: bool = True,
            extra_irreps: list = [],
    ):
        """
        Arguments:
            time_dim:
                The size of the of the time embedding that will be passed to 
                the `forward()` method.

            activation:
                An elementwise nonlinear activation function.
        """
        super().__init__()

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
        self.time_mlp = nn.Linear(time_dim, in_type.channels)
        self.act = activation

    def forward(self, x_hat_wrap: GeometricTensor, t: Tensor) -> GeometricTensor:
        """
        Arguments:
            x_hat_wrap:
                Geometric tensor of shape (B, C, D, D, D):
            t: 
                Tensor of shape (B, T).
        """
        x_wrap = self.ift(x_hat_wrap)

        b, c, *_ = x_wrap.tensor.shape

        t = self.time_mlp(t)
        assert t.shape == (b, c)

        y = x_wrap.tensor + t.view(b, c, 1, 1, 1, 1)
        y = self.act(y)

        y_wrap = GridTensor(y, x_wrap.grid, x_wrap.coords)

        return self.ft(y_wrap)

class LinearTimeActivation(nn.Module):

    def __init__(
            self,
            *,
            time_dim: int,
            activation: nn.Module,
    ):
        super().__init__()

        self.in_type = activation.in_type
        self.out_type = activation.out_type
        self.time_dim = time_dim

        g = self.in_type.gspace.fibergroup

        self.time_mlp = Linear(
                in_type=FieldType(
                    no_base_space(g),
                    time_dim * [g.trivial_representation],
                ),
                out_type=FieldType(
                    no_base_space(g),
                    self.in_type.representations,
                ),
        )
        self.activation = activation

    def forward(self, x: GeometricTensor, t: Tensor):
        assert x.type == self.in_type
        b, c = t.shape
        assert c == self.time_dim

        t_in = GeometricTensor(t, self.time_mlp.in_type)
        t_out = self.time_mlp(t_in)
        t_3d = GeometricTensor(
                t_out.tensor.view(b, -1, 1, 1, 1),
                x.type,
        )

        return self.activation(x + t_3d)

class GatedTimeActivation(nn.Module):
    """
    Use the time embedding to scale the representations in the main input 
    tensor.

    This is effectively just a gated nonlinearity, where the gates come from 
    the time embedding instead of trivial representations within the main input 
    tensor.  This works because any operation which affects only the magnitude 
    of the representation vectors will maintain equivariance.

    The gate values are obtained by passing the time embedding through a single 
    linear layer, and then applying a sigmoid function.
    """
    
    def __init__(
            self,
            in_type: FieldType,
            time_dim: int,
    ):
        super().__init__()

        g = in_type.gspace.fibergroup
        self.in_type = in_type
        self.out_type = in_type
        self.gate_type = FieldType(
                in_type.gspace,
                len(in_type.representations) * [g.trivial_representation],
        )
        self.time_dim = time_dim

        self.linear = nn.Linear(time_dim, self.gate_type.size)
        self.activation = GatedNonLinearity2(
                in_type=(self.gate_type, in_type),
        )

    def forward(self, x: GeometricTensor, t: Tensor):
        t = self.linear(t)
        t_wrap = GeometricTensor(
                repeat(t, 'b t -> b t {2} {3} {4}'.format(*x.shape)),
                self.gate_type,
        )
        return self.activation(t_wrap, x)



