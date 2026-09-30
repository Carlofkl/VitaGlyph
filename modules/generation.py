"""Single-subject VitaGlyph generation using SD 1.x and two ControlNets.

This is a new, explicit implementation of the final stage, not a wrapper around
the legacy pipeline copies. See docs/generation.md for paper/implementation gaps.
"""

from dataclasses import dataclass, asdict
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetPipeline

from .acg_attention import AttentionState, install_attention, neural_sketch


@dataclass
class GenerationConfig:
    steps: int = 50
    guidance_scale: float = 7.5
    gamma: float = 0.85
    subject_scale: float = 1.1
    surrounding_scale: float = 0.7
    alpha: float = 0.5
    cross_branch_attention: bool = True
    attention_control: bool = True
    attention_max_tokens: int = 1024
    sketch_sigma: float = 1.0
    sketch_threshold: float = 0.5

    def validate(self):
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer.")
        for name in ("guidance_scale", "subject_scale", "surrounding_scale", "sketch_sigma"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in ("gamma", "alpha", "sketch_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1].")
        if self.attention_max_tokens < 1:
            raise ValueError("attention_max_tokens must be positive.")


def compose_noise(subject, surrounding, unconditional, mask, gamma, guidance_scale):
    """Paper Eq. 4 and supplementary Eq. 9; background uses unweighted mask."""
    conditional = gamma * mask * subject + (1 - mask) * surrounding
    return unconditional + guidance_scale * (conditional - unconditional)


def load_inputs(subject, surrounding, mask, width=None, height=None, invert_mask=False):
    """Load full-canvas inputs; explicit output size permits rescaling each canvas."""
    images = []
    for path, mode in ((subject, "RGB"), (surrounding, "RGB"), (mask, "L")):
        with Image.open(path) as image:
            images.append(ImageOps.exif_transpose(image).convert(mode))
    if (width is None) != (height is None):
        raise ValueError("Set both --width and --height, or neither.")
    if len({image.size for image in images}) != 1 and width is None:
        raise ValueError("Inputs have different full-canvas dimensions; specify --resolution or --width/--height "
                         "to explicitly resize each aligned canvas.")
    width, height = (width, height) if width is not None else images[0].size
    if width < 64 or height < 64 or width % 64 or height % 64:
        raise ValueError("Output width and height must be positive multiples of 64 (minimum 64).")
    images = [image.resize((width, height), Image.Resampling.NEAREST if i == 2 else Image.Resampling.LANCZOS)
              for i, image in enumerate(images)]
    if invert_mask:
        images[2] = ImageOps.invert(images[2])
    images[2] = images[2].point(lambda p: 255 if p >= 128 else 0)
    if images[2].getextrema() != (0, 255):
        raise ValueError("Mask must contain both a white subject region and a black surrounding region.")
    return tuple(images)


def resolve_device(name="auto", precision="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type not in ("cuda", "mps", "cpu"):
        raise ValueError("Supported devices: cuda[:index], mps, cpu, auto.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable.")
    dtype = torch.float16 if precision == "fp16" or (precision == "auto" and device.type == "cuda") else torch.float32
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("Use fp32 on CPU.")
    return device, dtype


class VitaGlyphGenerator:
    def __init__(self, pipeline, subject_controlnet):
        self.pipe = pipeline
        self.subject_controlnet = subject_controlnet
        if pipeline.unet.config.in_channels != 4 or pipeline.unet.config.addition_embed_type is not None:
            raise ValueError("This generator supports SD 1.x models, not SDXL or inpainting models.")
        if pipeline.unet.config.time_cond_proj_dim is not None:
            raise ValueError("Time-conditioned/LCM UNets are not supported.")
        self.pipe.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        for model in (self.pipe.unet, self.pipe.controlnet, self.subject_controlnet,
                      self.pipe.vae, self.pipe.text_encoder):
            model.eval()

    @classmethod
    def from_pretrained(cls, base_model, subject_model, surrounding_model,
                        device="auto", precision="auto", local_files_only=False, lora=None, variant=None):
        device, dtype = resolve_device(device, precision)
        common = dict(torch_dtype=dtype, local_files_only=local_files_only)
        subject = ControlNetModel.from_pretrained(subject_model, **common)
        surrounding = ControlNetModel.from_pretrained(surrounding_model, **common)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model, controlnet=surrounding,
                                                                 variant=variant, **common)
        if lora:
            lora_path = Path(lora)
            if not lora_path.is_file():
                raise ValueError(f"LoRA file not found: {lora}")
            pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
        pipe.to(device)
        subject.to(device)
        pipe.enable_vae_slicing()
        # Decode in fp32: half-precision VAE activations can overflow on MPS.
        pipe.vae.to(dtype=torch.float32)
        return cls(pipe, subject)

    @torch.inference_mode()
    def generate(self, subject_image, surrounding_image, mask_image, subject_prompt,
                 surrounding_prompt, negative_prompt="", seed=0, config=None, debug_dir=None):
        config = config or GenerationConfig()
        config.validate()
        if not subject_prompt.strip() or not surrounding_prompt.strip():
            raise ValueError("Both subject and surrounding prompts are required.")
        if len({im.size for im in (subject_image, surrounding_image, mask_image)}) != 1:
            raise ValueError("Input images must have identical canvas dimensions.")
        width, height = subject_image.size
        if width % 64 or height % 64 or min(width, height) < 64:
            raise ValueError("Canvas dimensions must be multiples of 64, minimum 64.")
        if not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63).")
        pipe = self.pipe
        device, dtype = pipe._execution_device, pipe.unet.dtype
        sub_embed, neg_embed = pipe.encode_prompt(subject_prompt, device, 1, True, negative_prompt)
        surr_embed, _ = pipe.encode_prompt(surrounding_prompt, device, 1, False)
        controls = [pipe.prepare_image(im, width, height, 1, 1, device, dtype)
                    for im in (subject_image, surrounding_image)]
        mask = torch.from_numpy(np.array(mask_image.convert("L"), copy=True)).to(device=device)
        mask = (mask >= 128).to(dtype)[None, None]
        if not (mask.any() and (1 - mask).any()):
            raise ValueError("Mask must include subject and surrounding pixels.")
        latent_size = (height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
        latent_mask = F.interpolate(mask, size=latent_size, mode="nearest")
        if not (latent_mask.any() and (1 - latent_mask).any()):
            raise ValueError("Mask loses a region at latent resolution; use a larger region or canvas.")
        # A CPU generator also works with MPS via diffusers.randn_tensor.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        pipe.scheduler.set_timesteps(config.steps, device=device)
        latents = pipe.prepare_latents(1, pipe.unet.config.in_channels, height, width,
                                      dtype, device, generator)
        state = AttentionState(latent_size, config.alpha, config.cross_branch_attention,
                               config.attention_max_tokens)
        debug_dir = Path(debug_dir) if debug_dir else None
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)

        def predict(controlnet, control, embed, latent_input, timestep, scale):
            down, mid = controlnet(latent_input, timestep, encoder_hidden_states=embed,
                                   controlnet_cond=control, conditioning_scale=scale, return_dict=False)
            return pipe.unet(latent_input, timestep, encoder_hidden_states=embed,
                             down_block_additional_residuals=down,
                             mid_block_additional_residual=mid, return_dict=False)[0]

        with install_attention(pipe.unet, state):
            with pipe.progress_bar(total=len(pipe.scheduler.timesteps)) as progress:
                for index, timestep in enumerate(pipe.scheduler.timesteps):
                    state.begin_step()
                    latent_input = pipe.scheduler.scale_model_input(latents, timestep)
                    state.select("unconditional")
                    # Exactly one common negative-prompt prediction, without branch controls.
                    unconditional = pipe.unet(latent_input, timestep, encoder_hidden_states=neg_embed,
                                              return_dict=False)[0]
                    predictions = []
                    for branch, net, control, embed, scale, region in (
                        ("subject", self.subject_controlnet, controls[0], sub_embed, config.subject_scale, mask),
                        ("surrounding", pipe.controlnet, controls[1], surr_embed, config.surrounding_scale, 1 - mask),
                    ):
                        state.select(branch, collect_maps=config.attention_control)
                        prediction = predict(net, control, embed, latent_input, timestep, scale)
                        if config.attention_control:
                            # Same-step probe avoids a circular dependency between control and attention.
                            sketch = neural_sketch(state.maps, (height, width), config.sketch_sigma,
                                                   config.sketch_threshold).to(dtype)
                            fused = torch.maximum(control, sketch * region)
                            if debug_dir and index in (0, len(pipe.scheduler.timesteps) - 1):
                                self._save_tensor(sketch, debug_dir / f"{index:03d}_{branch}_sketch.png")
                                self._save_tensor(fused, debug_dir / f"{index:03d}_{branch}_control.png")
                            state.select(branch)
                            prediction = predict(net, fused, embed, latent_input, timestep, scale)
                        predictions.append(prediction)
                    noise = compose_noise(predictions[0], predictions[1], unconditional,
                                          latent_mask, config.gamma, config.guidance_scale)
                    latents = pipe.scheduler.step(noise, timestep, latents, eta=0.0,
                                                 return_dict=False)[0]
                    if not torch.isfinite(latents).all():
                        raise ValueError(f'Non-finite generation latents at step {index}; use --precision fp32.')
                    progress.update()
        decoded = pipe.vae.decode(latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor,
                                  return_dict=False)[0]
        if not torch.isfinite(decoded).all():
            raise ValueError('Non-finite decoded image; no result was saved.')
        decoded, flags = pipe.run_safety_checker(decoded, device, sub_embed.dtype)
        if flags is not None and any(flags):
            raise ValueError('The model content checker rejected this candidate. No artistic image was saved; '
                             'try another seed or prompt.')
        denormalize = [True] if flags is None else [not flag for flag in flags]
        image = pipe.image_processor.postprocess(decoded, output_type="pil", do_denormalize=denormalize)[0]
        return image, {"seed": seed, "width": width, "height": height, "config": asdict(config),
                       "nsfw_content_detected": None if flags is None else bool(flags[0])}

    @staticmethod
    def _save_tensor(value, path):
        array = (value[0].detach().float().cpu().clamp(0, 1) * 255).round().byte().numpy()
        array = array[0] if array.shape[0] == 1 else array.transpose(1, 2, 0)
        Image.fromarray(array).save(path)
