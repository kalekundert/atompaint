import torch
import torch.nn.functional as F
import lightning as L
import numpy as np
import math

from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment
from einops import rearrange, reduce
from functools import partial
from dataclasses import dataclass
from more_itertools import last
from tqdm.auto import tqdm

def exists(x):
    return x is not None

def identity(t, *args, **kwargs):
    return t

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def randn(rng, *, shape, device, mean=0, std=1):
    x_np = rng.normal(loc=mean, scale=std, size=shape)
    return torch.from_numpy(x_np).float().to(device)


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class HoDiffusion(L.LightningModule):
    """
    A diffusion model based on [Ho2022]_.

    The actual implementation is mostly  taken from the 
    `lucidrains/denoising-diffusion-pytorch`__ repository, with a few 
    modifications to better accommodate 3D atomic data.  This repository 
    combines features from a number of different papers, including DDIM 
    sampling [Song2022], immiscible mixing [Li2024], and more.

    __ https://github.com/lucidrains/denoising-diffusion-pytorch
    """

    # I removed the self-conditioning code, because it required an additional 
    # random number.  I'm strict about requiring all random numbers to come 
    # from the dataset, so before I can support self-conditioning, I'll need to 
    # think about how I want to make datasets that return more flexible sets of 
    # random numbers.

    def __init__(
            self,
            model,
            *,
            opt_factory,
            image_shape: tuple[int],
            timesteps = 1000,
            sampling_timesteps = None,
            objective = 'pred_v',
            beta_schedule = 'sigmoid',
            schedule_fn_kwargs = dict(),
            ddim_sampling_eta = 0.,
            min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
            min_snr_gamma = 5,
            immiscible = False,
    ):
        """
        Arguments:
            model:
                A function that predicts the specified objective given a noisy 
                image and a time index.  Specifically, the model should have 
                the following signature:

                    (x: torch.Tensor, t: torch.Tensor) -> torch.Tensor

                The inputs/outputs should have the following dimensions and 
                data types:

                    ======  =========  =========
                    Tensor  Dimension  Data Type
                    ======  =========  =========
                    x       B, C, *S   float32
                    t       B          int64
                    output  B, C, *S   float32
                    ======  =========  =========

                `B` is the batch dimension, `C` is the number of channels, and 
                `S` are the spatial dimensions of the images (e.g. width, 
                height, and depth for 3D images).

            image_shape:
                The dimensions of the images to be processed by this model.  
                The first value should give the size of the channel dimension.  
                The remaining values should give the size of each spatial 
                dimension (e.g. width, height, and depth for 3D images).
        """
        super().__init__()

        # Extract information from the model.  All of the attributes the model 
        # is expected to have are queried here.
        self.model = model
        self.optimizer = opt_factory(model.parameters())

        if not isinstance(image_shape, tuple):
            raise TypeError('image shape must be a tuple of integers')

        self.image_shape = image_shape
        self.objective = objective

        if objective not in {'pred_noise', 'pred_x0', 'pred_v'}:
            raise ValueError('objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])')

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule: {beta_schedule!r}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # immiscible diffusion

        self.immiscible = immiscible

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

    @property
    def device(self):
        return self.betas.device

    # Training

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        x_clean, noise, t_uniform = x
        assert x_clean.shape[1:] == self.image_shape

        t = (self.num_timesteps * t_uniform).long()
        return self._p_losses(x_clean, noise, t)

    def training_step(self, x):
        loss = self.forward(x)
        self.log('train/loss', loss, on_epoch=True)
        return loss

    def validation_step(self, x):
        loss = self.forward(x)
        self.log('val/loss', loss, on_epoch=True)
        return loss

    def test_step(self, x):
        loss = self.forward(x)
        self.log('test/loss', loss, on_epoch=True)
        return loss

    def _p_losses(self, x_clean, noise, t):
        x = self._q_sample(x_clean=x_clean, noise=noise, t=t)

        model_out = self.model(x, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_clean
        elif self.objective == 'pred_v':
            target = self._predict_v(x_clean, t, noise)
        else:
            raise ValueError(f'unknown objective: {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    @autocast(enabled = False)
    def _q_sample(self, x_clean, noise, t):
        if self.immiscible:
            assign = self._noise_assignment(x_clean, noise)
            noise = noise[assign]

        return (
            extract(self.sqrt_alphas_cumprod, t, x_clean.shape) * x_clean +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_clean.shape) * noise
        )

    def _noise_assignment(self, x_clean, noise):
        x_clean, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (x_clean, noise))
        dist = torch.cdist(x_clean, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    # Sampling

    @torch.inference_mode()
    def sample(
            self,
            rng: np.random.Generator,
            *,
            batch_size: int = 1,
            record_trajectory: bool = False,
    ):
        sample_fn = (
                self._p_sample_loop
                if not self.is_ddim_sampling else
                self._ddim_sample
        )
        sample_gen = sample_fn(
                rng,
                batch_size=batch_size,
        )

        if record_trajectory:
            traj = list(sample_gen)
            return torch.stack(traj, dim=1)
        else:
            return last(sample_gen)

    @torch.inference_mode()
    def _ddim_sample(self, rng, *, batch_size):
        times = torch.linspace(  # [-1, 0, 1, 2, ..., T-1] when `sampling_timesteps == num_timesteps`
                -1, self.num_timesteps - 1,
                steps=self.sampling_timesteps + 1,
        )
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = randn(
                rng,
                shape=(batch_size, *self.image_shape),
                device=self.device,
        )
        imgs = [img]

        eta = self.ddim_sampling_eta

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
            model_pred = self._model_predictions(
                    img, time_cond,
                    clip_x_start=True,
                    rederive_pred_noise=True,
            )

            if time_next < 0:
                img = model_pred.x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = randn(rng, shape=img.shape, device=img.device)

            img = (
                      alpha_next.sqrt() * model_pred.x_start
                    +                 c * model_pred.pred_noise
                    +             sigma * noise
            )

            yield img

    @torch.inference_mode()
    def _p_sample_loop(self, rng, *, batch_size):
        img = randn(
                rng,
                shape=(batch_size, *self.image_shape),
                device=self.device,
        )

        for t in tqdm(
                reversed(range(0, self.num_timesteps)),
                desc='sampling loop time step',
                total=self.num_timesteps,
        ):
            img = self._p_sample(rng, img, t)
            yield img

    @torch.inference_mode()
    def _p_sample(self, rng, x, t: int):
        b, *_, device = *x.shape, self.device

        batched_times = torch.full((b,), t, device=device, dtype=torch.long)

        model_mean, _, model_log_variance, _ = self._p_mean_variance(
                x=x,
                t=batched_times,
                clip_denoised=True,
        )

        noise = randn(rng, shape=x.shape, device=x.device) if t > 0 else 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img

    def _p_mean_variance(self, x, t, clip_denoised = True):
        preds = self._model_predictions(x, t)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(0., 1.)

        model_mean, posterior_variance, posterior_log_variance = self._q_posterior(
                x_start=x_start,
                x_t=x,
                t=t,
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def _q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _model_predictions(self, x, t, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t)
        maybe_clip = partial(torch.clamp, min = 0., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self._predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self._predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self._predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self._predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self._predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def _predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def _predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def _predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def _predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

@dataclass
class ModelPrediction:
    pred_noise: torch.Tensor
    pred_x_start: torch.Tensor

