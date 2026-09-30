"""SDS-guided raster deformation, inspired by Word-As-Image's score gradient.

This is a bounded displacement-field adapter for decomposed PNG inputs, not the
paper's Bezier/diffvg implementation or its conformal loss. See docs/pipeline.md.
"""
from dataclasses import dataclass, asdict
import math

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline

from .generation import resolve_device


@dataclass
class SDSConfig:
    iterations: int = 100
    learning_rate: float = .03
    guidance_scale: float = 100
    grid_size: int = 16
    max_displacement: float = 24
    render_size: int = 256
    tone_weight: float = 10
    smoothness_weight: float = .1
    displacement_weight: float = .01
    min_timestep: int = 50
    max_timestep: int = 950

    def validate(self):
        if self.iterations < 1 or self.grid_size < 2 or self.render_size < 64 or self.render_size % 64:
            raise ValueError('Positive iterations, grid_size >= 2, and render_size a multiple of 64 are required.')
        for key in ['learning_rate', 'guidance_scale', 'max_displacement', 'tone_weight',
                    'smoothness_weight', 'displacement_weight']:
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'{key} must be finite and non-negative.')
        if self.learning_rate == 0 or self.max_displacement == 0:
            raise ValueError('learning_rate and max_displacement must be positive.')
        if not 0 <= self.min_timestep < self.max_timestep <= 1000:
            raise ValueError('Timesteps must satisfy 0 <= min < max <= 1000.')


class RasterDeformation(nn.Module):
    """Optimize geometry only; pixel intensities come from the source via resampling."""
    def __init__(self, subject, mask, grid_size=16, max_displacement=24):
        super().__init__()
        if subject.size != mask.size:
            raise ValueError('Subject and mask must have the same size.')
        m = np.array(mask.convert('L'))
        if set(np.unique(m)) != {0, 255}:
            raise ValueError('Mask must contain both binary regions.')
        self.register_buffer('source', torch.from_numpy(np.array(subject.convert('L'), dtype=np.float32) / 255)[None, None])
        self.register_buffer('mask', torch.from_numpy(m.astype(np.float32) / 255)[None, None])
        if (self.source * self.mask).amax() <= .01:
            raise ValueError('The subject has no visible strokes inside its mask.')
        h, w = m.shape
        y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
        self.register_buffer('identity', torch.stack((x, y), dim=-1)[None])
        self.offsets = nn.Parameter(torch.zeros(1, 2, grid_size, grid_size))
        self.max_displacement = max_displacement
        ys, xs = np.where(m > 0)
        self.box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    def forward(self):
        h, w = self.source.shape[-2:]
        flow = F.interpolate(self.offsets.tanh(), size=(h, w), mode='bilinear', align_corners=True)
        scale = flow.new_tensor([2 * self.max_displacement / (w - 1),
                                 2 * self.max_displacement / (h - 1)])[None, :, None, None]
        grid = self.identity + (flow * scale).permute(0, 2, 3, 1)
        return F.grid_sample(self.source, grid, mode='bilinear', padding_mode='zeros',
                             align_corners=True) * self.mask

    def concept_view(self, rendered, size):
        x0, y0, x1, y1 = self.box
        crop = rendered[:, :, y0:y1, x0:x1]
        h, w = crop.shape[-2:]
        side = max(h, w)
        crop = F.pad(crop, ((side - w) // 2, side - w - (side - w) // 2,
                            (side - h) // 2, side - h - (side - h) // 2))
        crop = F.pad(crop, (max(1, side // 10),) * 4)
        # Diffusion guidance sees dark strokes on white, matching Word-As-Image.
        return 1 - F.interpolate(crop, (size, size), mode='bilinear', align_corners=False).repeat(1, 3, 1, 1)


def score_gradient(latents, noise, unconditional, conditional, alpha, guidance):
    prediction = unconditional + guidance * (conditional - unconditional)
    gradient = alpha.sqrt() * (1 - alpha) * (prediction - noise)
    if not torch.isfinite(gradient).all():
        raise ValueError('Non-finite SDS gradient; use fp32 diffusion computation.')
    return gradient.detach().to(latents.dtype)


class SDSTypography:
    def __init__(self, pipeline):
        self.pipe = pipeline
        if pipeline.unet.config.in_channels != 4 or pipeline.scheduler.config.prediction_type != 'epsilon':
            raise ValueError('SDS requires a four-channel epsilon-prediction Stable Diffusion model.')
        for component in [pipeline.unet, pipeline.vae, pipeline.text_encoder]:
            component.eval().requires_grad_(False)

    @classmethod
    def from_pretrained(cls, model='stable-diffusion-v1-5/stable-diffusion-v1-5', device='auto',
                        precision='auto', variant=None, local_files_only=False):
        device, dtype = resolve_device(device, precision)
        pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=dtype, variant=variant,
                                                       local_files_only=local_files_only).to(device)
        pipe.vae.to(dtype=torch.float32)
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
        return cls(pipe)

    def deform(self, subject, mask, prompt, seed=42, config=None, progress=None):
        cfg = config or SDSConfig()
        cfg.validate()
        if not prompt.strip() or not 0 <= seed < 2**63:
            raise ValueError('A prompt and a seed in [0, 2**63) are required.')
        pipe = self.pipe
        device = pipe._execution_device
        # Small warp stays on CPU for portable grid_sample backward; .to(device)
        # preserves its gradient through the frozen VAE encoder.
        warp = RasterDeformation(subject, mask, cfg.grid_size, cfg.max_displacement)
        with torch.no_grad():
            positive, negative = pipe.encode_prompt(
                'a black and white silhouette of ' + prompt + ', minimal flat vector drawing, white background',
                device, 1, True, '')
        embeddings = torch.cat([negative, positive])
        rng = torch.Generator(device='cpu').manual_seed(seed)
        optimizer = torch.optim.Adam(warp.parameters(), lr=cfg.learning_rate)
        reference = warp.concept_view(warp.source * warp.mask, cfg.render_size).detach()
        reference_tone = F.avg_pool2d(reference, 31, 1, 15)
        history = []
        if cfg.max_timestep > pipe.scheduler.config.num_train_timesteps:
            raise ValueError('max_timestep exceeds the model training schedule.')
        for index in range(cfg.iterations):
            optimizer.zero_grad(set_to_none=True)
            view = warp.concept_view(warp(), cfg.render_size)
            encoded = pipe.vae.encode(view.to(device) * 2 - 1).latent_dist
            eps_vae = torch.randn(encoded.mean.shape, generator=rng).to(device)
            latent = (encoded.mean + encoded.std * eps_vae) * pipe.vae.config.scaling_factor
            with torch.no_grad():
                timestep = torch.randint(cfg.min_timestep, cfg.max_timestep, (1,), generator=rng).to(device)
                noise = torch.randn(latent.shape, generator=rng).to(device)
                noisy = pipe.scheduler.add_noise(latent.detach(), noise, timestep).to(pipe.unet.dtype)
                predicted = pipe.unet(torch.cat([noisy, noisy]), timestep,
                                      encoder_hidden_states=embeddings).sample.float()
                uncond, cond = predicted.chunk(2)
                alpha = pipe.scheduler.alphas_cumprod.to(device)[timestep].reshape(-1, 1, 1, 1)
                gradient = score_gradient(latent, noise, uncond, cond, alpha, cfg.guidance_scale)
            proxy = (gradient * latent).sum(1).mean()
            tone = F.mse_loss(F.avg_pool2d(view, 31, 1, 15), reference_tone)
            flow = warp.offsets.tanh()
            smooth = (flow[:, :, 1:] - flow[:, :, :-1]).square().mean()
            smooth = smooth + (flow[:, :, :, 1:] - flow[:, :, :, :-1]).square().mean()
            displacement = flow.square().mean()
            regularization = cfg.tone_weight * tone + cfg.smoothness_weight * smooth + cfg.displacement_weight * displacement
            loss = proxy + regularization.to(device)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite SDS objective.')
            loss.backward()
            if warp.offsets.grad is None or not torch.isfinite(warp.offsets.grad).all():
                raise ValueError('Invalid deformation gradients.')
            torch.nn.utils.clip_grad_norm_(warp.parameters(), 1.0)
            optimizer.step()
            row = {'iteration': index + 1, 'timestep': timestep.item(), 'sds_proxy': proxy.item(),
                   'tone': tone.item(), 'smoothness': smooth.item(), 'displacement': displacement.item()}
            history.append(row)
            if progress:
                progress(row)
        with torch.no_grad():
            array = (warp()[0, 0].numpy().clip(0, 1) * 255).round().astype(np.uint8)
            offsets = warp.offsets.tanh()[0].numpy() * cfg.max_displacement
        return Image.fromarray(array).convert('RGB'), offsets, {
            'method': 'sds_raster_displacement', 'config': asdict(cfg), 'seed': seed,
            'prompt': prompt, 'device': str(device), 'unet_dtype': str(pipe.unet.dtype),
            'vae_dtype': str(pipe.vae.dtype), 'history': history,
            'displacement_convention': '2xgridxgrid source-sampling offsets in pixels; x then y; bilinear align_corners=True',
            'reference': 'https://github.com/WordAsImage/Word-As-Image',
            'limitations': 'Raster warp, not Bezier optimization; tone and flow regularizers, no ACAP/conformal loss.',
        }
