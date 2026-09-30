"""Deform a decomposed subject and prepare aligned controls for artistic generation."""

import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=Path, default=Path('preds'), help='One decomposition folder, or a parent containing sample folders.')
    parser.add_argument('--output', type=Path, default=Path('outs/CCG'))
    parser.add_argument('--subject-prompt', help='Manual subject description; overrides JSON.')
    parser.add_argument('--prompts', type=Path, help='Prompt JSON for one input folder.')
    parser.add_argument('--prompts-root', type=Path, help='Batch JSON directory; defaults to output_parent/LLM.')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-images', type=int, default=5)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--strength', type=float, default=.76)
    parser.add_argument('--guidance-scale', type=float, default=10)
    parser.add_argument('--positive-prompt', '--positive_prompt', default='clean monochrome drawing, simple background')
    parser.add_argument('--negative-prompt', '--negative_prompt', default='blurry, deformed, low quality, cluttered background')
    parser.add_argument('--model', default='sd2-community/stable-diffusion-2-depth')
    parser.add_argument('--revision', help='Depth2Img model commit or revision.')
    parser.add_argument('--variant', help='Diffusion weight variant, e.g. fp16; depth weights remain fp32.')
    parser.add_argument('--hed-model', default='lllyasviel/Annotators')
    parser.add_argument('--surrounding-preprocessor', choices=('hed', 'outline'), default='hed')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--precision', choices=('auto', 'fp16', 'fp32'), default='auto')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def input_jobs(args):
    from modules.image_files import find_image
    from ConComGen import read_prompts
    if not args.input.is_dir():
        raise ValueError(f'Input directory not found: {args.input}')
    single = any((args.input / ('sub' + suffix)).is_file() for suffix in ('.png', '.jpg', '.jpeg'))
    if not single and args.prompts:
        raise ValueError('--prompts is for one sample; use --prompts-root in batch mode.')
    folders = [args.input] if single else sorted(p for p in args.input.iterdir() if p.is_dir())
    jobs = []
    for folder in folders:
        images = [find_image(folder, stem) for stem in ('sub', 'surr', 'mask')]
        prompt = args.subject_prompt
        prompt_file = None
        if not prompt:
            root = args.prompts_root or args.output.parent / 'LLM'
            prompt_file = args.prompts or root / f'{folder.name}.json'
            prompt, _ = read_prompts(prompt_file)
        output = args.output if single else args.output / folder.name
        jobs.append((images, prompt, output, prompt_file))
    if not jobs:
        raise ValueError('No decomposition folders found.')
    return jobs


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    from ConComGen import append_style
    from modules.generation import load_inputs, resolve_device
    from modules.regional_decomposition import file_fingerprint
    from modules.semantic_typography import SemanticTypography, candidate_preview
    import math
    try:
        if (args.width is None) != (args.height is None):
            raise ValueError('Set both --width and --height.')
        width, height = (args.width, args.height) if args.width is not None else (args.resolution, args.resolution)
        if args.num_images < 1 or args.steps < 1 or not 0 < args.strength <= 1 or int(args.steps * args.strength) < 1:
            raise ValueError('Positive count/steps and strength in (0,1] with steps*strength >= 1 are required.')
        if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
            raise ValueError('guidance-scale must be finite and non-negative.')
        if args.seed < 0 or args.seed + args.num_images > 2**63:
            raise ValueError('Seeds must lie in [0, 2**63).')
        device = 'cuda:' + args.device if args.device.isdigit() else args.device
        resolve_device(device, args.precision)
        jobs = input_jobs(args)
        for paths, prompt, output, _ in jobs:
            load_inputs(*paths, width, height)
            append_style(prompt, args.positive_prompt)
            targets = [output / f'sub_{i}.png' for i in range(args.num_images)]
            targets += [output / name for name in ('surr.png', 'mask.png', 'depth.png', 'preview.png', 'semtypo.json')]
            stale = [path for path in output.glob('sub_*.png')
                     if path.stem[4:].isdigit() and int(path.stem[4:]) >= args.num_images]
            if stale:
                raise ValueError('Output contains extra candidates from an older run; use a new output folder: '
                                 + ', '.join(str(path) for path in stale))
            for target in targets:
                if target.resolve() in {path.resolve() for path in paths}:
                    raise ValueError(f'Output would overwrite input: {target}')
                if target.exists() and not args.overwrite:
                    raise ValueError(f'Output exists: {target}; choose another folder or --overwrite.')
            print(f'Validated {paths[0].parent} -> {output}, {width}x{height}')
        if args.dry_run:
            print(f'Dry run passed: {len(jobs)} sample(s); {args.num_images} variant(s) each.')
            return 0
        processor = SemanticTypography.from_pretrained(args.model, device, args.precision, args.hed_model,
                                                       args.local_files_only, args.surrounding_preprocessor,
                                                       variant=args.variant, revision=args.revision)
        for paths, prompt, output, prompt_file in jobs:
            images = load_inputs(*paths, width, height)
            prompt = append_style(prompt, args.positive_prompt)
            variants, surrounding, mask = processor.prepare(
                *images, prompt, args.negative_prompt, args.seed, args.num_images, args.steps,
                args.strength, args.guidance_scale, args.surrounding_preprocessor,
            )
            output.mkdir(parents=True, exist_ok=True)
            for index, variant in enumerate(variants):
                variant.save(output / f'sub_{index}.png')
            surrounding.save(output / 'surr.png')
            mask.save(output / 'mask.png')
            processor.last_depth_image.save(output / 'depth.png')
            candidate_preview(images[0], processor.last_depth_image, surrounding, mask,
                              variants, args.seed).save(output / 'preview.png')
            record = {'subject_prompt': prompt, 'negative_prompt': args.negative_prompt,
                      'seeds': list(range(args.seed, args.seed + args.num_images)),
                      'steps': args.steps, 'strength': args.strength, 'guidance_scale': args.guidance_scale,
                      'size': [width, height], 'model': args.model, 'hed_model': args.hed_model,
                      'revision': args.revision, 'variant': args.variant, 'precision': args.precision,
                      'surrounding_preprocessor': args.surrounding_preprocessor, 'device': device,
                      'runtime': processor.last_run,
                      'candidates': [{'file': f'sub_{i}.png', 'seed': args.seed + i} for i in range(args.num_images)],
                      'inputs': {name: file_fingerprint(path) for name, path in zip(('sub', 'surr', 'mask'), paths)}}
            if prompt_file:
                record['prompt_file'] = file_fingerprint(prompt_file)
            (output / 'semtypo.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'Saved {len(variants)} subject variants, surr.png and mask.png to {output}')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
