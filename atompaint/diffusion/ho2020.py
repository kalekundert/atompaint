import torch
import torch.nn.functional as F

from torch.nn import Module
from einops import reduce
from math import pi

# TODO:
# - Save exponential moving averages of model weights [1].  This is apparently 
#   an important hyperparameter.
#
#   [1] https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/

class GaussianDiffusionModule(L.LightningModule):

    def __init__(
            self,
            model,
            *,
            objective: str,
            noise_schedule: 'NoiseSchedule',
            min_snr_loss_weight=False,
            min_snr_gamma=5
    ):
        super().__init__()

        self.model = model
        self.noise_schedule = noise_schedule
        self.objective = objective

        # I don't fully understand the idea behind weighting the loss function.  
        # [Hang2024] argues that clipping the weights can make the model do a 
        # better job of balancing the needs of high- and low-noise predictions.  
        # It also seems like changing the loss weights is necessary to get 
        # equivalence between the different objectives.

        snr = noise_schedule.snr

        # Hang et al. 2024, https://arxiv.org/abs/2303.09556
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        # Convert all tensors to single precision when registering buffers.
        def register_buffer(name, x):
            self.register_buffer(name, x.to(torch.float32), persistent=False)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))
        
    def forward(self, x_clean, noise, t_uniform):

        b = x_start.shape[0]

        phi = torch.distributions.Normal(loc=0, scale=1)
        t_norm = phi.icdf(t_uniform)

        t = scipy.stats.norm().ppf(t_uniform)
        t = torch.from_numpy(t)

        t = rng.integers(self.noise_schedule.timesteps, size=b)
        t = torch.from_numpy(t)

        noise = rng.normal(size=x_start.size)
        noise = torch.from_numpy(noise).to(dtype=x_start.dtype)

        x_t = interpolate_image_with_noise(
                x_start=x_start,
                noise=noise,
                noise_schedule=self.noise_schedule,
                t=t,
        )

        model_out = self.model(x_t, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f"unknown objective: {self.objective!r}")

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * select_timepoints(self.loss_weight, t, 1)
        return loss.mean()

class NoiseSchedule(Module):

    def __init__(self, betas):
        super().__init__()

        beta_min, beta_max = torch.aminmax(betas)
        if beta_min < 0 or beta_max > 1:
            raise ValueError("beta values must be in the range [0, 1]")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.roll(alphas_cumprod, 1)
        alphas_cumprod_prev[0] = 1

        self.num_timesteps = len(betas)

        # Convert all tensors to single precision when registering buffers.
        def register_buffer(name, x):
            self.register_buffer(name, x.to(torch.float32), persistent=False)

        register_buffer('betas', betas)
        register_buffer('alphas', alphas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        register_buffer('snr', alphas_cumprod / (1 - alphas_cumprod))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

def interpolate_image_with_noise(*, x_start, noise, noise_schedule, t):
    # This function implements equation 69 from [arXiv:2208.11970].
    sqrt_alphas_cumprod = select_timepoints(
            noise_schedule.sqrt_alphas_cumprod, t)
    sqrt_one_minus_alphas_cumprod = select_timepoints(
            noise_schedule.sqrt_one_minus_alphas_cumprod, t)
    return (
            sqrt_alphas_cumprod * x_start +
            sqrt_one_minus_alphas_cumprod * noise
    )

def select_timepoints(a, t, dim=3):
    """
    Extract the $\alpha$ values associated with the given timepoints.

    Arguments:
        a: torch.Tensor
            A 1D tensor of size $T$, i.e. where each element represents a 
            different timepoint.  Typically this argument is related to the 
            $\alpha$ values that specify the amount of noise to add in each 
            diffusion step.

        t: torch.Tensor
            A 1D tensor of indices into the *a* tensor.  Typically the number 
            of indices is equal to the batch size.

        dim: int
            The dimensionality of the images being processed, not counting the 
            batch or channel dimensions.  The final tensor is reshaped so that 
            it can be broadcasted with images of this dimensionality.  
            
    Returns:
        A torch.Tensor with containing the values from *a* corresponding to 
        each timepoint in *t*.  The first dimension of this tensor will be the 
        same size as *t*, and the rest will all be size 1.  The number of 
        dimensions will equal *dim* plus 2.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (dim + 1)))

def linear_betas(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_betas(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_betas(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
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
