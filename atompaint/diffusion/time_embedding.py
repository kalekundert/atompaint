import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
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
                A 1D tensor of diffusion step indices.  Typically, the indices 
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
        freq = tau / torch.logspace(
                start=log2(self.min_wavelength),
                end=log2(self.max_wavelength),
                steps=self.out_dim // 2,
                base=2,
                device=t.device,
        )
        theta = torch.outer(t, freq)
        return torch.cat((torch.sin(theta), torch.cos(theta)), dim=-1)
