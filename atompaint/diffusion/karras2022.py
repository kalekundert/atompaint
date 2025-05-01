from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import polars as pl
import numpy as np
import sys
import re

from atompaint.metrics.neighbor_loc import (
        NeighborLocAccuracy, FrechetNeighborLocDistance,
)
from atompaint.utils import eval_mode
from einops import reduce, repeat
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from pathlib import Path
from tqdm import trange
from tquiet import tquiet, ProgressBarFactory
from math import sqrt

from typing import TypeAlias, Optional, Callable, Any
from atompaint.type_hints import OptFactory, LrFactory
from torchmetrics import Metric

RngFactory: TypeAlias = Callable[[], np.random.Generator]
LabelFactory: TypeAlias = Callable[[np.random.Generator, int], torch.Tensor]
SolverHook: TypeAlias = Callable[[dict[str, Any]], Any]

class KarrasDiffusion(L.LightningModule):
    """
    A diffusion model inspired by (but not exactly identical to) the EDM 
    frameworks described in [Karras2022]_ and [Karras2023]_.
    """

    def __init__(
            self,
            precond: KarrasPrecond,
            *,
            opt_factory: OptFactory,
            lr_factory: Optional[LrFactory] = None,
            gen_metrics: Optional[dict[str, Metric]] = None,
            gen_params: Optional[GenerateParams] = None,
            gen_rng_factory: Optional[RngFactory] = None,
            gen_label_factory: Optional[LabelFactory] = None,
            frechet_ref_path: Optional[str | Path] = None,
    ):
        super().__init__()

        self.precond = precond
        self.optimizer = opt_factory(precond.parameters())
        self.lr_scheduler = lr_factory(self.optimizer) if lr_factory else None

        # The neighbor-location-based metrics require batch sizes that are 
        # multiples of 12.
        self.gen_params = gen_params or GenerateParams()
        self.gen_rng_factory = gen_rng_factory or (lambda: np.random.default_rng(0))
        self.gen_label_factory = gen_label_factory

        if gen_metrics is not None:
            self.gen_metrics = gen_metrics
        else:
            self.gen_metrics = {
                    'accuracy': NeighborLocAccuracy(),
            }
            if frechet_ref_path:
                frechet = FrechetNeighborLocDistance()
                frechet.load_reference_stats(frechet_ref_path)
                self.gen_metrics['frechet_dist'] = frechet

        # [Karras2022], Table 1.  This mean and standard deviation should lead 
        # to σ values in roughly the range [1.9e-3, 5.4e2].
        self.P_mean = -1.2
        self.P_std = 1.2

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return self.optimizer
        else:
            return {
                'optimizer': self.optimizer,
                'lr_scheduler': {
                    'scheduler': self.lr_scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    'monitor': 'val/loss',
                },
            }

    def forward(self, x):
        x_clean = x['x_clean']
        noise = x['noise']
        rngs = x['rng']
        labels = x.get('label', None)

        assert x_clean.dtype == noise.dtype
        if labels is not None:
            assert labels.dtype == x_clean.dtype

        d = x_clean.ndim - 2

        t_norm = rngs.normal(loc=self.P_mean, scale=self.P_std)
        t_norm = t_norm.to(dtype=x_clean.dtype, device=x_clean.device)
        sigma = torch.exp(t_norm).reshape(-1, 1, *([1] * d))
        x_noisy = x_clean + sigma * noise

        if not self.precond.self_condition:
            x_self_cond = None
        else:
            x_self_cond = _make_x_self_cond(
                    precond=self.precond,
                    x_noisy=x_noisy,
                    sigma=sigma,
                    labels=labels,
                    mask=rngs.choice([True, False]),
            )

        x_pred = self.precond(
                x_noisy,
                sigma,
                label=labels,
                x_self_cond=x_self_cond,
        )

        sigma_data = self.precond.sigma_data
        weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
        loss = weight * (x_pred - x_clean)**2
        loss = reduce(loss, 'b c ... -> c ...', 'mean').sum()

        return loss

    def training_step(self, x):
        loss = self.forward(x)
        self.log('train/loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, x):
        loss = self.forward(x)
        self.log('val/loss', loss, on_epoch=True, sync_dist=True)
        return loss

    @torch.inference_mode()
    def on_validation_epoch_end(self):
        if not self.gen_metrics:
            return

        # Lightning takes care of putting the model in eval-mode and disabling 
        # gradients before this hook is invoked [1], so we don't need to do 
        # that ourselves.
        #
        # [1]: https://pytorch-lightning.readthedocs.io/en/1.7.2/common/lightning_module.html#hooks

        # We set the batch size to 12, so we're generating 12 × 32 = 384 
        # images.  Each image allows 4 updates, for a total of 384 × 4 = 1536 
        # updates.  In Experiment 97, I showed that this is about the smallest 
        # number needed to get a stable result.

        rng = self.gen_rng_factory()
        b = 12

        for i in trange(32, desc="Generative metrics", leave=True, file=sys.stdout):
            if self.gen_label_factory:
                labels = self.gen_label_factory(rng, b).to(self.device)
            else:
                labels = None

            x = generate(
                    precond=self.precond,
                    params=self.gen_params,
                    labels=labels,
                    num_images=b,
                    rng=rng,
            )

            for metric in self.gen_metrics.values():
                metric.to(self.device)
                metric.update(x)

        for name, metric in self.gen_metrics.items():
            self.log(f'gen/{name}', metric.compute(), sync_dist=True)
            metric.reset()

    def test_step(self, x):
        loss = self.forward(x)
        self.log('test/loss', loss, on_epoch=True, sync_dist=True)
        return loss

class KarrasPrecond(nn.Module):

    def __init__(
            self,
            model: nn.Module,
            *,
            sigma_data: float,
            x_shape: list[int],
            label_dim: int = 0,
            self_condition: bool = False,
            use_karras_c_noise: bool = False,
    ):
        super().__init__()

        self.model = model
        self.x_shape = x_shape
        self.label_dim = label_dim
        self.self_condition = self_condition
        self.use_karras_c_noise = use_karras_c_noise

        d = len(x_shape) - 1
        self.register_buffer(
                'sigma_data',
                torch.tensor(sigma_data).reshape(-1, *([1] * d)).float(),
                persistent=False,
        )

    def forward(self, x_noisy, sigma, *, x_self_cond=None, label=None):
        """
        Output a denoised version of $x_\textrm{noisy}$.

        The underlying model will actually predict a mixture of the noise and 
        the denoised image, in a ratio that depends on the noise level.  The 
        purpose is to avoid dramatically scaling the model outputs.  When 
        $x_\textrm{noisy}$ is dominated by noise, its easiest to predict the 
        noise directly.  When $x_\textrm{noisy}$ is nearly noise-free to begin 
        with, it's easiest to directly predict $x$.
        """
        assert x_noisy.shape[1:] == self.x_shape

        sigma = sigma.reshape(-1, *([1] * (len(x_noisy.shape) - 1)))
        sigma_data = self.sigma_data

        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
        c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()

        # [Karras2022] calls for `c_noise = sigma.log() / 4`, but I don't think 
        # this makes sense.  See experiment #83 for details.  However, I 
        # included an option to use the original formula so that I could test
        # my `KarrasPrecond` class on the pretrained models from [Karras2022].
        c_noise = sigma.log() / 4 if self.use_karras_c_noise else sigma

        F_x = self.model(
                c_in * x_noisy,
                c_noise.flatten(),
                **self._get_model_kwargs(x_self_cond, label),
        )
        D_x = c_skip * x_noisy + c_out * F_x

        return D_x

    def _get_model_kwargs(self, x_self_cond, label):
        model_kwargs = {}

        # If the caller provides a label, the model must be expecting it.
        if self.label_dim == 0:
            assert label is None
        else:
            assert label is not None
            assert label.shape[-1] == self.label_dim
            model_kwargs['label'] = label

        # The caller is allowed to provide `x_self_cond` even if the model 
        # doesn't need it.  It will just be ignored in this case.
        if self.self_condition:
            assert x_self_cond is not None
            model_kwargs['x_self_cond'] = x_self_cond

        return model_kwargs

@dataclass(kw_only=True)
class GenerateParams:
    # See Experiment #120 for details on how these parameters were chosen.
    noise_steps: int = 98
    resample_steps: int = 1

    sigma_min: float = 0.0278876278887711
    sigma_max: float = 65.7645071048362
    rho: float = 7.53983816992786

    S_churn: float = 12.3927241214305
    S_min: float = 0
    S_max: float = float('inf')

    # The mean and standard deviation of the underlying dataset, if the model 
    # was trained on data where these parameters were normalized.  More 
    # prescriptively, these values should match the values of `normalize_mean` 
    # and `normalize_std` that were passed to `MacromolImageDiffusionData`.
    unnormalize_mean: float = 0
    unnormalize_std: float = 1

    clamp_low: float = 0
    clamp_high: float = 1

    # Cap the number of images that can be fed to the model at once.  This is 
    # meant to help prevent the image-generation process from exceeding the 
    # available VRAM.  If `None`, no cap is set.
    max_batch_size: Optional[int] = None

@dataclass(kw_only=True)
class InpaintParams(GenerateParams):
    resample_steps: int = 10

@torch.inference_mode()
def generate(
        *,
        precond: KarrasPrecond,
        labels: Optional[torch.Tensor] = None,
        params: GenerateParams,
        num_images: int,
        rng: np.random.Generator,
        device: Optional[torch.device] = None,
        progress_bar: ProgressBarFactory = tquiet,
        record_trajectory: bool = False,
):
    if labels is not None:
        if labels.shape[-1] != precond.label_dim:
            raise ValueError(f"The model requires labels with {precond.label_dim} dimensions, but the given label has shape {labels.shape}")

        if len(labels.shape) == 1:
            labels = repeat(labels, '... -> b ...', b=num_images)
        if num_images != labels.shape[0]:
            raise ValueError(f"Requested {num_images} images, but provided {len(labels)} labels")

    if record_trajectory:
        traj = []

    def on_end_step(**locals):
        if record_trajectory:
            traj.append(dict(
                    i=locals['i'],
                    j=locals['j'],

                    σ1=locals['σ1'].item(),
                    σ2=locals['σ2'].item(),
                    σ3=locals['σ3'].item(),

                    x_noisy_σ1=locals['x_noisy_σ1'],
                    x_clean_σ1=locals['x_clean_σ1'],

                    x_noisy_σ2=locals['x_noisy_σ2'],
                    x_clean_σ2=locals['x_clean_σ2'],

                    x_noisy_σ3=locals['x_noisy_σ3'],
                    x_noisy_σ3_1st_order=locals['x_noisy_σ3_1st_order'],
                    x_noisy_σ3_2nd_order=locals['x_noisy_σ3'] if locals['σ3'] > 0 else None,
                    x_clean_σ3=locals['x_clean_σ3'] if locals['σ3'] > 0 else None,
            ))

    x_generate = _heun_sde_solver(
            precond=precond,
            params=params,
            num_images=num_images,
            labels=labels,
            rng=rng,
            device=device,

            on_end_step=on_end_step,
            progress_bar=progress_bar,
    )

    if record_trajectory:
        return x_generate, pl.DataFrame(traj)
    else:
        return x_generate

@torch.inference_mode()
def inpaint(
        *,
        precond: KarrasPrecond,
        x_known: torch.Tensor,
        mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        params: InpaintParams,
        num_images: Optional[int] = None,
        rng: np.random.Generator,
        device: Optional[torch.device] = None,
        progress_bar: ProgressBarFactory = tquiet,
        record_trajectory: bool = False,
):
    """
    Arguments:

        mask:
            A tensor the same shape as the image.  0 means to take the value 
            from the known image, 1 means to fill in the value using the 
            diffusion model.

            Allowed dimensions.  SPATIAL indicates a variable number of dimensions.

                [*SPATIAL]
                [C, *SPATIAL]
                [B, 1, *SPATIAL]
                [B, C, *SPATIAL]
                
    """
    d = len(precond.x_shape)

    if x_known.shape[-d:] != precond.x_shape:
        raise ValueError(f"The model requires images with shape {precond.x_shape}, but the known image has shape {x_known.shape}")

    if len(x_known.shape) == d:
        x_known = repeat(x_known, '... -> 1 ...')

    if num_images is None:
        num_images = x_known.shape[0]
    else:
        if x_known.shape[0] == 1:
            x_known = repeat(x_known, '1 ... -> b ...', b=num_images)
        if num_images != x_known.shape[0]:
            raise ValueError(f"Requested {num_images} images, but provided {len(x_known)} known images")

    if len(mask.shape) == d - 1:
        mask = repeat(mask, '... -> b 1 ...', b=num_images)

    if len(mask.shape) == d:
        mask = repeat(mask, '... -> b ...', b=num_images)

    if mask.shape[-(d-1):] != precond.x_shape[1:]:
        raise ValueError(f"The model requires images with shape {precond.x_shape}, but the mask has shape {mask.shape}")

    if labels is not None:
        if labels.shape[-1] != precond.label_dim:
            raise ValueError(f"The model requires labels with {precond.label_dim} dimensions, but the given label has shape {labels.shape}")

        if len(labels.shape) == 1:
            labels = repeat(labels, '... -> b ...', b=num_images)

    if record_trajectory:
        traj = []

    def on_begin_step(*, i, j, σ1, x_noisy_σ1, N, **locals):
        x_known_σ1 = x_known + σ1 * N()

        if record_trajectory:
            frame = dict(
                    i=i,
                    j=j,
                    σ1=σ1.item(),
                    x_known_σ1=x_known_σ1,
                    x_noisy_σ1_before_mask=x_noisy_σ1.clone(),
            )
            traj.append(frame)

        # Modify this tensor in place, so that the changes are seen by the 
        # solver.  Note that we don't need to record this tensor now; it'll get 
        # recorded at the end of the step.
        x_noisy_σ1 *= mask
        x_noisy_σ1 += x_known_σ1 * (1 - mask)

    def on_end_step(*, σ2, σ3, **locals):
        if record_trajectory:
            traj[-1].update(
                    σ2=σ2.item(),
                    σ3=σ3.item(),

                    x_noisy_σ1_after_mask=locals['x_noisy_σ1'],
                    x_clean_σ1=locals['x_clean_σ1'],

                    x_noisy_σ2=locals['x_noisy_σ2'],
                    x_clean_σ2=locals['x_clean_σ2'],

                    x_noisy_σ3=locals['x_noisy_σ3'],
                    x_noisy_σ3_1st_order=locals['x_noisy_σ3_1st_order'],
                    x_noisy_σ3_2nd_order=locals['x_noisy_σ3'] if σ3 > 0 else None,
                    x_clean_σ3=locals['x_clean_σ3'] if σ3 > 0 else None,
            )

    x_inpaint = _heun_sde_solver(
            precond=precond,
            params=params,
            num_images=num_images,
            labels=labels,
            rng=rng,
            device=device,

            on_begin_step=on_begin_step,
            on_end_step=on_end_step,
            progress_bar=progress_bar,
    )

    if record_trajectory:
        return x_inpaint, pl.DataFrame(traj)
    else:
        return x_inpaint


def load_expt_102_unet(*, lr=5, epoch=99, mode='eval'):
    from atompaint.checkpoints import load_model_weights, strip_prefix

    def fix_keys(k):
        k = strip_prefix(k, prefix='model.')
        k = re.sub(r'^model\.blocks\.(\d+)\.', r'model.model.\1.', k)
        k = re.sub(r'^model\.time_embedding', 'model.noise_embedding', k)
        k = re.sub(r'\.time\.time_mlp', '.cond.mlp', k)
        k = re.sub(r'\.time_mlp\.', '.cond_mlp.', k)
        k = re.sub(r'\.activation\.', '.act.', k)
        return k

    ckpt_paths = {
            (4, 63): 'expt_102/lr=p4-epoch=63-step=809152.ckpt',
            (4, 69): 'expt_102/lr=p4-epoch=69-step=885010.ckpt',
            (5, 99): 'expt_102/lr=p5-epoch=99-step=1264300.ckpt',
    }
    ckpt_xxh32s = {
            (4, 63): '9b89217d',
            (4, 69): '7920aea5',
            (5, 99): 'b7304d0d',
    }

    try:
        ckpt_path = ckpt_paths[lr, epoch]
        ckpt_xxh32 = ckpt_xxh32s[lr, epoch]
    except KeyError:
        raise KeyError(f"no checkpoint available for lr={lr}, epoch={epoch}") from None

    unet = make_expt_102_unet()
    load_model_weights(
            model=unet,
            path=ckpt_path,
            xxh32sum=ckpt_xxh32,
            fix_keys=fix_keys,
            mode=mode,
    )
    return unet

def make_expt_102_unet():
    from atompaint.autoencoders.semisym_unet import SemiSymUNet
    from atompaint.field_types import make_fourier_field_types
    from escnn.gspaces import rot3dOnR3
    from functools import partial
    from multipartial import multipartial, rows

    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    ift_grid = so3.grid(type='thomson_cube', N=4)

    def sym_head_factory(in_type, out_type, ift_grid):
        import atompaint.layers as ap
        yield from ap.sym_conv_bn_fourier_layer(
                in_type,
                out_type,
                ift_grid=ift_grid,
        )

    def sym_block_factory(in_type, out_type, cond_dim, ift_grid):
        from atompaint.autoencoders.sym_unet import SymUNetBlock
        from atompaint.conditioning import LinearConditionedActivation
        from atompaint.nonlinearities import first_hermite
        from escnn.nn import TensorProductModule, FourierPointwise

        return SymUNetBlock(
                in_type,
                cond_activation=LinearConditionedActivation(
                    cond_dim=cond_dim,
                    activation=TensorProductModule(out_type, out_type),
                ),
                out_activation=FourierPointwise(
                    out_type,
                    grid=ift_grid,
                    function=first_hermite,
                ),
        )

    def sym_downsample_factory(in_type):
        import atompaint.layers as ap
        return ap.sym_pool_conv_layer(in_type)

    def latent_factory(in_type, cond_dim):
        from atompaint.layers import UnwrapTensor
        yield sym_block_factory(
                in_type=in_type,
                out_type=in_type,
                cond_dim=cond_dim,
                ift_grid=ift_grid,
        )
        yield UnwrapTensor()

    def asym_tail_factory(in_channels, out_channels):
        yield nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=0,
        )

    def asym_block_factory(in_channels, out_channels, cond_dim, attention=False):
        import atompaint.autoencoders.asym_unet as ap

        yield ap.AsymConditionedConvBlock(
                in_channels,
                out_channels,
                cond_dim=cond_dim,
        )

        if attention:
            yield ap.AsymAttentionBlock(
                    out_channels,
                    num_heads=2,
                    channels_per_head=out_channels // 2,
            )

    def asym_upsample_factory(channels):
        import atompaint.upsampling as ap
        return ap.Upsample3d(
                size_expr=lambda x: 2*x - 1,
                mode='trilinear',
        )

    def noise_embedding(out_dim):
        import atompaint.conditioning as ap

        embed_dim = 4 * out_dim

        yield ap.SinusoidalEmbedding(
                out_dim=embed_dim,
                min_wavelength=0.1,
                max_wavelength=100,
        )
        yield nn.Linear(embed_dim, out_dim)
        yield nn.ReLU()
        
    unet = SemiSymUNet(
            img_channels=6,
            encoder_types=make_fourier_field_types(
                gspace,
                channels=[2, 4, 8, 16],
                max_frequencies=2,
            ),

            head_factory=partial(sym_head_factory, ift_grid=ift_grid),
            tail_factory=asym_tail_factory,

            encoder_factories=partial(
                sym_block_factory,
                ift_grid=ift_grid,
            ),
            decoder_factories=multipartial[:,1](
                asym_block_factory,
                attention=rows(False, False, True),
            ),
            latent_factory=latent_factory,

            downsample_factory=sym_downsample_factory,
            upsample_factory=asym_upsample_factory,

            cond_dim=16,
            noise_embedding=noise_embedding,
    )
    return KarrasPrecond(
            unet,
            x_shape=(6, 35, 35, 35),
            sigma_data=1,
    )


def _make_x_self_cond(
        *,
        precond: KarrasPrecond,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        labels: Optional[torch.Tensor],
        mask: torch.Tensor,
) -> torch.Tensor:
    with torch.inference_mode(), eval_mode(precond):
        x_self_cond = torch.zeros_like(x_noisy)

        # Need to specially handle this case, because the model will complain 
        # if it gets an empty minibatch.
        if not mask.any():
            return x_self_cond

        x_self_cond_mask = precond(
                x_noisy[mask],
                sigma[mask],
                label=None if labels is None else labels[mask],
                x_self_cond=x_self_cond[mask],
        )

        x_self_cond_mask_iter = iter(x_self_cond_mask)

        for i, mask_i in enumerate(mask):
            if mask_i:
                x_self_cond[i] = next(x_self_cond_mask_iter)

    return x_self_cond

@torch.inference_mode()
def _heun_sde_solver(
        *,
        precond: KarrasPrecond,
        labels: Optional[torch.Tensor] = None,
        params: GenerateParams,
        rng: np.random.Generator,
        device: Optional[torch.device] = None,
        num_images: int,
        on_begin_step: SolverHook = lambda **_: None,
        on_end_step: SolverHook = lambda **_: None,
        progress_bar: ProgressBarFactory = tquiet,
):
    """
    Generate new images, using a model trained to denoise noisy images.

    In technical terms, this function solves stochastic differential equations 
    (SDEs) using Heun's method.  This is a second-order method, which means 
    that the model is evaluated twice for each sampling step.  This extra 
    evaluation allows bigger steps to be taken, so overall this method is more 
    efficient than first-order methods.  The algorithm is based on:

    - Algorithm 2 from [Karras2022]
    - Algorithm 1 from [Lugmayr2022]

    This function is not meant to be used directly by external callers.  
    Instead, it's meant to implement the main logic for both the `generate()` 
    and `inpaint()` functions.

    Arguments:
        precond:
            The model that will be used to denoise the images.  The model 
            should accept a noisy image and the amount of noise in the image, 
            expressed as a standard deviation.  It should return the noise-free 
            version of the image.

            The model must also have attributes `x_shape` and `self_condition`, 
            which respectively give the shape of the expected input, and 
            whether or not self-conditioning is supported.

        rng:
            A pseudorandom number generator, used to generate noise during the 
            sampling process.

        num_images:
            The number of images to generate.

        on_begin_step:
        on_end_step:
            Functions that will be called at the beginning and end of each 
            step, respectively.  The functions will be passed all of the local 
            variables currently in use by the solver, as keyword arguments.  
            In-place modifications to these values will affect the solver's 
            behavior.

            The main intended purpose of these functions is to record
            diagnostic information, i.e. the intermediate images created during 
            the generation process.  The `inpaint()` function also uses 
            `on_begin_step()` to apply the mask.
    """

    assert not precond.training
    device = device or _get_model_device(precond)
    x_shape = num_images, *precond.x_shape
    N = partial(_sample_normal, rng=rng, shape=x_shape, device=device)

    # Each iteration makes use of 3 different noise levels:
    # - σ1: the noise level at the start of the step
    # - σ2: the noise level after churn is added
    # - σ3: the noise level at the end of the step
    #
    # By default, σ1 and σ2 are the same.  If `params.S_churn` is non-zero, 
    # though, σ2 will be greater than σ1.  σ3 is always the smallest of the 
    # three.

    # More nomenclature:
    # - x_noisy_σ*: A generated image, with the specified amount of noise.  
    #   Note that σ* is not the actual standard deviation of the image; it's 
    #   just the standard deviation of the noise, so to get it you'd first have 
    #   to subtract away the noise-free image.
    # - x_clean_σ*: The denoised version of x_noisy_σ*, predicted by the given 
    #   model.  Note that here, the σ suffix does not indicate the amount of 
    #   noise in the image (which should be zero), but rather the amount of 
    #   noise that was removed to make the clean image.

    sigmas = _calc_sigma_schedule(params).to(device)
    x_noisy_σ1 = sigmas[0] * N()
    x_clean_σ1 = torch.zeros(x_shape, device=device)

    def clean_from_noisy(x_noisy, σ, *, x_self_cond):
        return torch.cat([
            precond(
                x_noisy[i:j], σ,
                x_self_cond=x_self_cond[i:j],
                label=labels[i:j] if labels is not None else None,
            )
            for i, j in _get_batch_indices(num_images, params.max_batch_size)
        ])

    for i, (σ1, σ3) in progress_bar(list(enumerate(pairwise(sigmas)))):
        for j in range(params.resample_steps):
            on_begin_step(**locals())

            σ2, x_noisy_σ2 = _add_churn(rng, σ1, x_noisy_σ1, params)
            assert σ3 < σ2

            # It's not obvious why the standard deviation of:
            #
            #   x_noisy_σ2 + (σ3 - σ2) * dx_dσ2
            #
            # would be `σ3`.  However, we can arrive at this result by 
            # carefully applying variance and covariance identities.  
            # First, let's define some shorter variable names:
            #
            #   a = σ2
            #   b = σ3
            #   c = (b - a) / a
            #
            #   X = x_noisy_σ2
            #   Y = x_clean_σ2
            #   Z = x_noisy_σ2 + (σ3 - σ2) * dx_dσ2 = X + c (X - Y)
            #
            # We want to show that Var[Z] = b².
            #
            #   Var[Z] = Var[X + c (X - Y)]
            #          = Var[X] + c² Var[X - Y] + 2c Cov[X, X - Y]
            #          = Var[X] + c² (Var[X] + Var[Y] - 2 Cov[X, Y]) + 2c (Cov[X, X] - Cov[X, Y])
            #
            # Y is the noise-free image, so we can simplify the above 
            # expression by substituting Var[Y] = Cov[X, Y] = 0.  We can also 
            # substitute Cov[X, X] = Var[X]:
            #
            #   Var[Z] = Var[X] + c² Var[X] + 2c Var[X]
            #          = Var[X] (1 + c² + 2c)
            #
            # Now recall that Var[X] = a²:
            #
            #   Var[Z] = a² (1 + (b - a)²/a² + 2(b - a)/a)
            #          = a² + (b - a)² + 2a (b - a)
            #          = a² + (b² - 2ab + a²) + 2ab - 2a²
            #          = b²
            # Q.E.D.

            x_clean_σ2 = clean_from_noisy(
                    x_noisy_σ2, σ2,
                    x_self_cond=x_clean_σ1,
            )
            dx_dσ2 = (x_noisy_σ2 - x_clean_σ2) / σ2
            x_noisy_σ3 = x_noisy_σ3_1st_order = x_noisy_σ2 + (σ3 - σ2) * dx_dσ2

            if σ3 > 0:
                x_clean_σ3 = clean_from_noisy(
                        x_noisy_σ3, σ3,
                        x_self_cond=x_clean_σ2,
                )
                dx_dσ3 = (x_noisy_σ3 - x_clean_σ3) / σ3
                x_noisy_σ3 = x_noisy_σ2 + (σ3 - σ2) * (dx_dσ2 + dx_dσ3) / 2

            on_end_step(**locals())

            # Prepare for the next iteration.  If we're going to take a 
            # resampling step, we need to add noise back to the image to 
            # restore its previous noise level.  Otherwise, we just need to 
            # relabel some variables.

            if j < params.resample_steps - 1:
                x_noisy_σ1 = x_noisy_σ3 + (σ1**2 - σ3**2).sqrt() * N()
            else:
                x_noisy_σ1 = x_noisy_σ3

            x_clean_σ1 = x_clean_σ3 if σ3 > 0 else x_clean_σ2

    x_out = x_clean_σ2

    # On very large tensors, in-place operations are slightly faster.  It's a 
    # minor effect, though.
    x_out.mul_(params.unnormalize_std)
    x_out.add_(params.unnormalize_mean)
    x_out.clamp_(params.clamp_low, params.clamp_high)

    return x_out

def _calc_sigma_schedule(params):
    n = params.noise_steps;  assert n > 1
    i = torch.arange(n + 1)
    inv_rho = 1 / params.rho
    σ_max = params.sigma_max
    σ_min = params.sigma_min

    # See Equation 5 from [Karras2022].
    sigmas = (
            σ_max ** inv_rho +
            (i / (n - 1)) * (σ_min ** inv_rho - σ_max ** inv_rho)
    ) ** params.rho
    sigmas[-1] = 0

    return sigmas

def _add_churn(rng, σ_cur, x_cur, params):
    if not (params.S_min <= σ_cur <= params.S_max):
        return σ_cur, x_cur

    gamma = min(params.S_churn / params.noise_steps, sqrt(2) - 1)
    σ_churn = σ_cur * (1 + gamma)
    eps_churn = _sample_normal(
            rng,
            mean=0,
            std=sqrt(σ_churn**2 - σ_cur**2),
            shape=x_cur.shape,
            device=x_cur.device,
    )

    return σ_churn, x_cur + eps_churn

def _sample_normal(rng, *, mean=0, std=1, shape, device):
    z = rng.normal(loc=mean, scale=std, size=shape)
    return torch.from_numpy(z).float().to(device)

def _sqrt_zero_ok(x):
    return torch.sqrt(x) if x > 0 else 0

def _get_batch_indices(num_images, max_batch_size):
    if max_batch_size is None:
        yield 0, num_images
        return

    i = 0
    while i < num_images:
        j = min(i + max_batch_size, num_images)
        yield i, j
        i = j

def _get_model_device(model: nn.Module):
    from itertools import chain

    # For testing, I sometimes use "models" with no parameters, so it's 
    # convenient for this case to be handled seamlessly.  `KarrasPrecond` has a 
    # buffer, though, so we'll always find a device if we also look for those.
    try:
        return next(chain(model.parameters(), model.buffers())).device
    except StopIteration:
        return 'cpu'
