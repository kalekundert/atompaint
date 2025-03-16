import torch
import torch.nn as nn
import torchyield as ty
import inspect

from torch import Tensor
from escnn.nn import (
        GeometricTensor, GridTensor, FieldType, FourierFieldType, Linear,
        InverseFourierTransform, FourierTransform, GatedNonLinearity2
)
from escnn.gspaces import no_base_space
from escnn.group import GroupElement
from einops.layers.torch import Rearrange
from einops import repeat
from pipeline_func import f
from math import tau, log2

from typing import Optional

class ConditionedModel(nn.Module):
    """
    Base class for diffusion models that can be given extra information about 
    the image they're trying to reconstruct.

    There are three kinds of conditional information that can be provided:

    - Noise: The amount of noise that has been added to the input image.

    - Label: Some sort of supervised signal about the input image.

    - Self-conditioning: The noise-free image predicted by the 
      previous diffusion step.
    """

    def __init__(
            self,
            model: ty.Layer,
            *,
            noise_embedding: ty.Layer,
            label_embedding: Optional[ty.Layer] = None,
            concat_label: bool = False,
            allow_self_cond: bool = False,
    ):
        """
        Initialize a conditioned model.

        Arguments:
            model:
                The model that will process the conditioned input.  The forward 
                method of this model must accept two positional arguments.  The 
                first will be the input image, with dimensions (B, C, W, H, D).  
                The second will be the conditions, with dimensions (B, E).

                    B: batch size
                    C: number of channels
                    W, H, D: spatial dimensions (all equal)
                    E: embedding size

            noise_embedding:
                A layer that will encode the noise level for the model.  The 
                noise will be given as a tensor with dimensions (B,).  The 
                embedding must produce a tensor with dimensions (B, E) or
                (B, E//2), depending on if *concat_label* is false or true, 
                respectively.  A typical noise embedding would be a sinusoidal 
                embedding followed by a shallow MLP.

            label_embedding:
                A layer that will encode any supervisory labels that are 
                available to describe the input image.  The label will be given 
                as a tensor with dimensions (B, *).  The embedding must produce 
                a tensor with dimensions (B, E) or (B, E//2), depending on if 
                *concat_label* is false or true, respectively.

                This embedding is optional.  If it is specified, then the user 
                must provide a label to the `forward()` method.  If not, then
                the user must *not* provide a label to the `forward()` method.

            concat_label:
                If True, combine the noise and label embeddings by 
                concatenation rather than by addition.  Concatenation is more 
                intuitive, since it prevents the two signals from interfering 
                with each other.  It also allows the noise and label embeddings 
                to have different sizes.  But state-of-the-art models 
                exclusively use addition, perhaps for some of the reasons 
                described in the following links:
                
                https://ai.stackexchange.com/questions/35990/why-are-embeddings-added-not-concatenated
                https://datascience.stackexchange.com/questions/55901/in-a-transformer-model-why-does-one-sum-positional-encoding-to-the-embedding-ra

            allow_self_cond:
                If true, allow the user to provide the denoised output from the 
                previous diffusion step to the forward pass.  This denoised 
                image will be concatenated to the input image along the channel 
                dimension, and the result will be passed to the model.  Note 
                that the model must be aware of whether or not this happens, 
                since it changes the size of the input.
        """
        super().__init__()
        self.model = ty.module_from_layer(model)
        self.noise_embedding = ty.module_from_layer(noise_embedding)
        self.label_embedding = label_embedding and ty.module_from_layer(label_embedding)
        self.concat_label = False
        self.allow_self_cond = allow_self_cond

    def forward(
            self,
            x: Tensor,
            noise: Tensor,
            *,
            label: Tensor = None,
            x_self_cond: Tensor = None
    ):
        if not self.allow_self_cond:
            if x_self_cond is not None:
                raise ValueError("self-conditioning is not allowed for this model")
        else:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)

            x = torch.cat([x, x_self_cond], dim=1)

        y = y_noise = self.noise_embedding(noise)

        if self.label_embedding is None:
            if label is not None:
                raise ValueError("must not provide labeled inputs, because there is not label embedding")
        else:
            if label is None:
                raise ValueError("must provide a label for each input, because there is a label embedding")

            y_label = self.label_embedding(label)

            if self.concat_label:
                y = torch.cat([y_noise, y_label])
            else:
                y = y_noise + y_label

        # The `wrap_input()` and `wrap_output()` hooks are just provided to 
        # make it a little easier for subclasses to work with equivariant 
        # inputs/outputs.  The same effect could be obtained by adding pre- and 
        # post-processing layers to the model, but this is a little cleaner.

        return (
                x
                | f(self.wrap_input)
                | f(self.model, y)
                | f(self.unwrap_output)
        )

    def wrap_input(self, x):
        return x

    def unwrap_output(self, x):
        return x



class AddConditionToImage(nn.Module):

    def __init__(
            self,
            *,
            cond_dim: int,
            channel_dim: int,
            latent_dim: Optional[int] = None,
            affine: bool = False,
    ):
        super().__init__()
        self.affine = affine

        if not latent_dim:
            latent_dim = min(cond_dim, channel_dim * 4)

        self.mlp = nn.Sequential(
                nn.Linear(cond_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, channel_dim * (2 if affine else 1)),
                Rearrange('b c -> b c 1 1 1'),
        )

    def forward(self, x, y):
        y = self.mlp(y)

        if self.affine:
            m, b = y.chunk(2, dim=1)
            return m * x + b
        else:
            return x + y

class SinusoidalEmbedding(nn.Module):
    """
    Convert scalars to n-dimensional vectors, where each dimension varies 
    with a different frequency.

    This can be thought of as a generalization of a binary encoding, with 
    "bits" that are partially redundant.
    """

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
                An tensor of "timepoints" to embed, of dimension (..., T).

        Returns:
            A 2D tensor of dimension (..., T, D):

            - T: number of timepoints to embed, i.e. the last dimension of the 
              input tensor.
            - D: output embedding size, i.e. the *out_dim* argument provided to 
              the constructor.

            Earlier positions in the output embedding change rapidly as a 
            function of the input index, while later positions change slowly.  
            This embedding is typically added or concatenated to whatever input 
            data is associated with each timepoint.
        """
        freq = tau / torch.logspace(
                start=log2(self.min_wavelength),
                end=log2(self.max_wavelength),
                steps=self.out_dim // 2,
                base=2,
                device=t.device,
        )
        theta = torch.einsum('...t,w->...tw', t, freq)
        return torch.cat((torch.sin(theta), torch.cos(theta)), dim=-1)

class FourierConditionedActivation(nn.Module):
    """
    Integrate a condition embedding into a geometric tensor, and then apply an 
    activation function.

    In order to maintain equivariance, both steps are done in the Fourier 
    domain.
    """

    def __init__(
            self,
            in_type: FourierFieldType,
            grid: list[GroupElement],
            *,
            cond_dim: int,
            activation: nn.Module = nn.SELU(),
            normalize: bool = True,
            extra_irreps: list = [],
    ):
        """
        Arguments:
            cond_dim:
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
        self.cond_mlp = nn.Linear(cond_dim, in_type.channels)
        self.act = activation

    def forward(self, x_hat_wrap: GeometricTensor, y: Tensor) -> GeometricTensor:
        """
        Arguments:
            x_hat_wrap:
                Geometric tensor of shape (B, C, D, D, D):
            y: 
                Tensor of shape (B, E).
        """
        x_wrap = self.ift(x_hat_wrap)

        b, c, *_ = x_wrap.tensor.shape

        y = self.cond_mlp(y)
        assert y.shape == (b, c)

        xy = x_wrap.tensor + y.view(b, c, 1, 1, 1, 1)
        xy = self.act(xy)

        xy_wrap = GridTensor(xy, x_wrap.grid, x_wrap.coords)

        return self.ft(xy_wrap)

class LinearConditionedActivation(nn.Module):

    def __init__(
            self,
            *,
            cond_dim: int,
            activation: nn.Module,
    ):
        super().__init__()

        self.in_type = activation.in_type
        self.out_type = activation.out_type
        self.cond_dim = cond_dim

        g = self.in_type.gspace.fibergroup

        self.cond_mlp = Linear(
                in_type=FieldType(
                    no_base_space(g),
                    cond_dim * [g.trivial_representation],
                ),
                out_type=FieldType(
                    no_base_space(g),
                    self.in_type.representations,
                ),
        )
        self.act = activation

    def forward(self, x: GeometricTensor, y: Tensor):
        assert x.type == self.in_type
        b, c = y.shape
        assert c == self.cond_dim

        y_in = GeometricTensor(y, self.cond_mlp.in_type)
        y_out = self.cond_mlp(y_in)
        y_3d = GeometricTensor(
                y_out.tensor.view(b, -1, 1, 1, 1),
                x.type,
        )

        return self.act(x + y_3d)

class GatedConditionedActivation(nn.Module):
    """
    Use the condition embedding to scale the representations in the main input 
    tensor.

    This is effectively just a gated nonlinearity, where the gates come from 
    the condition embedding instead of trivial representations within the main 
    input tensor.  This works because any operation which affects only the 
    magnitude of the representation vectors will maintain equivariance.

    The gate values are obtained by passing the condition embedding through a 
    single linear layer, and then applying a sigmoid function.
    """
    
    def __init__(
            self,
            in_type: FieldType,
            cond_dim: int,
    ):
        super().__init__()

        g = in_type.gspace.fibergroup
        self.in_type = in_type
        self.out_type = in_type
        self.gate_type = FieldType(
                in_type.gspace,
                len(in_type.representations) * [g.trivial_representation],
        )
        self.cond_dim = cond_dim

        # The gate will apply a sigmoid nonlinearity to these values, so we 
        # don't need to add another nonlinearity here.
        self.linear = nn.Linear(cond_dim, self.gate_type.size)

        self.act = GatedNonLinearity2(
                in_type=(self.gate_type, in_type),
        )

    def forward(self, x: GeometricTensor, y: Tensor):
        y = self.linear(y)
        y_wrap = GeometricTensor(
                repeat(y, 'b c -> b c {2} {3} {4}'.format(*x.shape)),
                self.gate_type,
        )
        return self.act(y_wrap, x)

def forward_with_condition(module: nn.Module, x, y):
    """
    Run a forward pass of the given module.  Provide the condition only if the 
    model accepts it.
    """

    # `torch.nn.Module.__call__()` accepts `*args` and `**kwargs`, so we need 
    # to inspect `forward()` instead.
    sig = inspect.signature(module.forward)

    # Some modules, for example `ConvTranspose3d`, accept an optional 
    # second argument.  We don't want to pass the `y` input to these 
    # modules.  This is why we first check to see if we can invoke the 
    # module with only one argument, and provide the second only if 
    # necessary.
    try:
        sig.bind(x)
    except TypeError:
        args = x, y
    else:
        args = x,

    return module(*args)

