"""Optional SDS raster-deformation stage with the same controls as SemTypo.py."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--subject-prompt', required=True)
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--learning-rate', type=float, default=.03)
    p.add_argument('--guidance-scale', type=float, default=100)
    p.add_argument('--max-displacement', type=float, default=24)
    p.add_argument('--tone-weight', type=float, default=10)
    p.add_argument('--render-size', type=int, default=256)
    p.add_argument('--model', default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    p.add_argument('--variant')
    p.add_argument('--device', default='auto')
    p.add_argument('--precision', choices=['auto', 'fp16', 'fp32'], default='auto')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    import gc
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from modules.generation import load_inputs, resolve_device
    from modules.image_files import find_image
    from modules.regional_decomposition import file_fingerprint
    from modules.sds_typography import SDSConfig, SDSTypography
    try:
        cfg = SDSConfig(iterations=a.iterations, learning_rate=a.learning_rate,
                        guidance_scale=a.guidance_scale, max_displacement=a.max_displacement,
                        tone_weight=a.tone_weight, render_size=a.render_size)
        cfg.validate()
        device, _ = resolve_device(a.device, a.precision)
        if not a.subject_prompt.strip() or not 0 <= a.seed < 2**63:
            raise ValueError('Provide a prompt and seed in [0, 2**63).')
        paths = [find_image(a.input, stem) for stem in ['sub', 'surr', 'mask']]
        subject, surrounding, mask = load_inputs(*paths, a.resolution, a.resolution)
        targets = [a.output / n for n in ['sub_0.png', 'surr.png', 'mask.png', 'preview.png', 'displacement.npy', 'sds.json']]
        if any(path.exists() for path in targets):
            raise ValueError('SDS output already exists; choose a new output folder.')
        if a.dry_run:
            print('SDS inputs and configuration validated; no model loaded.')
            return 0
        # Prepare the small HED output, then release it before loading diffusion.
        from controlnet_aux import HEDdetector
        hed = HEDdetector.from_pretrained('lllyasviel/Annotators', filename='ControlNetHED.pth',
                                         local_files_only=a.local_files_only).to(device)
        surrounding = hed(surrounding, detect_resolution=a.resolution, image_resolution=a.resolution,
                          output_type='pil', scribble=True).convert('RGB').resize(subject.size)
        del hed
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()
        processor = SDSTypography.from_pretrained(a.model, a.device, a.precision, a.variant, a.local_files_only)
        def progress(row):
            if row['iteration'] == 1 or row['iteration'] % 10 == 0 or row['iteration'] == a.iterations:
                print(f"SDS {row['iteration']}/{a.iterations}: tone={row['tone']:.6f}, displacement={row['displacement']:.6f}", flush=True)
        result, offsets, record = processor.deform(subject, mask, a.subject_prompt, a.seed, cfg, progress)
        a.output.mkdir(parents=True, exist_ok=True)
        for name, im in [('sub_0', result), ('surr', surrounding), ('mask', mask)]:
            im.save(a.output / (name + '.png'))
        np.save(a.output / 'displacement.npy', offsets)
        preview = Image.new('RGB', (1024, 284), '#252525')
        draw = ImageDraw.Draw(preview)
        for i, (name, im) in enumerate([('Original subject', subject), ('SDS deformation', result),
                                        ('Surrounding scribble', surrounding), ('Mask', mask)]):
            preview.paste(im.convert('RGB').resize((256, 256)), (256 * i, 28))
            draw.text((256 * i + 6, 7), name, fill='white')
        preview.save(a.output / 'preview.png')
        record.update({'model': a.model, 'variant': a.variant, 'inputs': {
            stem: file_fingerprint(path) for stem, path in zip(['sub', 'surr', 'mask'], paths)}})
        (a.output / 'sds.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(f'Saved SDS controls to {a.output}')
    except (ValueError, OSError) as error:
        p.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
