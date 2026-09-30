"""Subject deformation and surrounding-control preparation for the generation stage."""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import torch
from diffusers import DDIMScheduler, StableDiffusionDepth2ImgPipeline
from transformers import DPTForDepthEstimation

from .generation import resolve_device


class SemanticDepth2ImgPipeline(StableDiffusionDepth2ImgPipeline):
    def prepare_depth_map(self, image, depth_map, batch_size, do_classifier_free_guidance, dtype, device):
        # PyTorch 2.5 MPS lacks bicubic interpolation. Resize/normalize only the
        # small depth condition on CPU; diffusion and depth estimation stay on MPS.
        if torch.device(device).type == 'mps' and depth_map is not None:
            depth = super().prepare_depth_map(image, depth_map.float().cpu(), batch_size,
                                             do_classifier_free_guidance, torch.float32,
                                             torch.device('cpu'))
            return depth.to(device=device, dtype=dtype)
        return super().prepare_depth_map(image, depth_map, batch_size,
                                         do_classifier_free_guidance, dtype, device)


def candidate_preview(subject, depth, surrounding, mask, variants, seed, tile_size=256):
    """Diagnostic contact sheet; never used as a downstream control image."""
    panels = [('Input subject', subject), ('Depth (display only)', depth),
              ('Surrounding scribble', surrounding), ('Subject mask', mask)]
    panels += [(f'sub_{i} | seed {seed + i}', image) for i, image in enumerate(variants)]
    columns = min(4, len(panels))
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new('RGB', (columns * tile_size, rows * (tile_size + 28)), '#252525')
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(panels):
        x, y = (i % columns) * tile_size, (i // columns) * (tile_size + 28)
        tile = ImageOps.contain(image.convert('RGB'), (tile_size, tile_size))
        sheet.paste(tile, (x + (tile_size - tile.width) // 2, y + 28 + (tile_size - tile.height) // 2))
        draw.text((x + 6, y + 7), label, fill='white')
    return sheet


class SemanticTypography:
    def __init__(self, pipeline, hed=None):
        self.pipeline = pipeline
        self.hed = hed
        self.pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        self.last_depth_image = None
        self.last_run = None

    @classmethod
    def from_pretrained(cls, model="sd2-community/stable-diffusion-2-depth", device="auto",
                        precision="auto", hed_model="lllyasviel/Annotators", local_files_only=False,
                        surrounding_preprocessor="hed", variant=None, revision=None):
        if surrounding_preprocessor not in ('hed', 'outline'):
            raise ValueError("surrounding_preprocessor must be hed or outline.")
        device, dtype = resolve_device(device, precision)
        # Load depth weights in fp32 from the start; casting fp16 weights back would
        # not recover their original precision, and DPT includes BatchNorm layers.
        depth_estimator = DPTForDepthEstimation.from_pretrained(
            model, subfolder="depth_estimator", torch_dtype=torch.float32,
            local_files_only=local_files_only, revision=revision,
        )
        pipeline = SemanticDepth2ImgPipeline.from_pretrained(
            model, torch_dtype=dtype, depth_estimator=depth_estimator, local_files_only=local_files_only,
            variant=variant, revision=revision,
        ).to(device)
        pipeline.vae.enable_slicing()
        # Bound attention memory on 16 GB Macs and smaller GPUs.
        pipeline.enable_attention_slicing()
        hed = None
        if surrounding_preprocessor == "hed":
            from controlnet_aux import HEDdetector
            hed = HEDdetector.from_pretrained(hed_model, filename="ControlNetHED.pth",
                                            local_files_only=local_files_only).to(device)
        elif surrounding_preprocessor != "outline":
            raise ValueError("surrounding_preprocessor must be hed or outline.")
        return cls(pipeline, hed)

    @torch.inference_mode()
    def prepare(self, subject, surrounding, mask, prompt, negative_prompt="", seed=42, count=5,
                steps=50, strength=.76, guidance_scale=10, surrounding_preprocessor="hed"):
        self.last_depth_image = None
        self.last_run = None
        if not prompt.strip():
            raise ValueError("A subject prompt is required for Semantic Typography.")
        if len({im.size for im in (subject, surrounding, mask)}) != 1:
            raise ValueError("Subject, surrounding and mask must share a canvas.")
        if min(subject.size) < 64 or any(side % 64 for side in subject.size):
            raise ValueError("Canvas dimensions must be multiples of 64, minimum 64.")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in (count, steps, seed)):
            raise ValueError("Count, steps and seed must be integers.")
        if count < 1 or steps < 1 or not 0 < strength <= 1 or int(steps * strength) < 1:
            raise ValueError("Positive count/steps and strength in (0,1] with steps*strength >= 1 are required.")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and non-negative.")
        if seed < 0 or seed + count > 2**63:
            raise ValueError("Seeds must lie in [0, 2**63).")
        mask = mask.convert('L')
        if set(np.unique(mask)) != {0, 255}:
            raise ValueError("Mask must be binary with both subject (255) and surrounding (0).")
        subject, surrounding = subject.convert('RGB'), surrounding.convert('RGB')
        if surrounding_preprocessor == "hed":
            if self.hed is None:
                raise ValueError("HED was not loaded; use outline or construct the processor with HED.")
            size = min(subject.size)
            scribble = self.hed(surrounding, detect_resolution=size, image_resolution=size,
                                output_type="pil", scribble=True)
            scribble = scribble.convert("RGB").resize(subject.size, Image.Resampling.NEAREST)
        elif surrounding_preprocessor == "outline":
            # Explicit, weight-free alternative for already clean light-on-dark glyphs.
            binary = surrounding.convert("L").point(lambda p: 255 if p >= 128 else 0)
            array = np.array(binary, dtype=np.int16) - np.array(binary.filter(ImageFilter.MinFilter(3)), dtype=np.int16)
            scribble = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")
        else:
            raise ValueError("Unknown surrounding preprocessor.")
        # Extract depth once per input in fp32; avoid fp16 BatchNorm and repeated estimator work.
        pipe = self.pipeline
        depth_pixels = pipe.feature_extractor(images=subject, return_tensors="pt").pixel_values
        depth_pixels = depth_pixels.to(device=pipe._execution_device, dtype=torch.float32)
        depth_map = pipe.depth_estimator(depth_pixels).predicted_depth
        if not torch.isfinite(depth_map).all() or (depth_map.amax() - depth_map.amin()) <= 1e-6:
            raise ValueError("Depth estimator returned a non-finite or constant map.")
        display_depth = (depth_map[0].float() - depth_map.amin()) / (depth_map.amax() - depth_map.amin())
        self.last_depth_image = Image.fromarray(
            (display_depth.cpu().numpy() * 255).round().astype(np.uint8)
        ).resize(subject.size, Image.Resampling.BICUBIC)
        actual_prompt = "a black and white drawing of " + prompt
        deformed = []
        def check_latents(pipeline, step, timestep, tensors):
            if not torch.isfinite(tensors['latents']).all():
                raise ValueError(f'Non-finite diffusion latents at step {step}; try --precision fp32.')
            return tensors

        for current_seed in range(seed, seed + count):
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            pixels = pipe(prompt=actual_prompt, image=subject,
                         depth_map=depth_map, num_inference_steps=steps, strength=strength,
                         guidance_scale=guidance_scale, negative_prompt=negative_prompt,
                         generator=generator, output_type='np',
                         callback_on_step_end=check_latents).images[0]
            if not np.isfinite(pixels).all():
                raise ValueError('Non-finite decoded image; try --precision fp32. No candidates were saved.')
            image = Image.fromarray((pixels * 255).round().clip(0, 255).astype(np.uint8))
            if image.size != subject.size:
                raise ValueError(f"Depth-to-image changed canvas size from {subject.size} to {image.size}.")
            deformed.append(image)
        self.last_run = {
            'diffusion_prompt': actual_prompt,
            'scheduler': type(pipe.scheduler).__name__,
            'scheduler_config': dict(pipe.scheduler.config),
            'denoising_steps': int(steps * strength),
            'execution_device': str(pipe._execution_device),
            'diffusion_dtype': str(pipe.unet.dtype),
            'depth_dtype': str(depth_map.dtype),
            'depth_range': [depth_map.amin().item(), depth_map.amax().item()],
            'subject_postprocessing': 'none; original full-canvas Depth2Img output',
        }
        return deformed, scribble, mask.copy()
