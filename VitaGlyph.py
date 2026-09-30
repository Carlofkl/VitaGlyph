"""Run detection, semantic deformation and artistic rendering with manual prompts."""
import argparse
import hashlib
import json
import shlex
from pathlib import Path
import subprocess
from subprocess import run as execute
import sys

ROOT = Path(__file__).resolve().parent


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_stage(name, command, inputs, outputs, root, resume=False, dry_run=False):
    """Resume only an identical command with unchanged inputs AND output hashes."""
    record_path = root / (name + '.stage.json')
    key = {'command': command, 'inputs': {str(p.resolve()): fingerprint(p) for p in inputs}}
    if resume and record_path.exists():
        previous = json.loads(record_path.read_text())
        if (previous.get('request') == key and all(p.is_file() for p in outputs)
                and previous.get('outputs') == {str(p.resolve()): fingerprint(p) for p in outputs}):
            print(f'[{name}] Reusing verified outputs', flush=True)
            return
    if dry_run:
        print(f'[{name}] ' + shlex.join(command), flush=True)
        return
    if any(p.exists() for p in outputs):
        raise ValueError(f'{name}: existing outputs do not match this request; use a new --output directory.')
    print(f'[{name}] Starting', flush=True)
    execute(command, check=True, cwd=ROOT)
    if not all(p.is_file() for p in outputs):
        raise ValueError(f'{name}: process finished without all required outputs.')
    record_path.write_text(json.dumps({'request': key, 'outputs': {
        str(p.resolve()): fingerprint(p) for p in outputs}}, ensure_ascii=False, indent=2))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--image', type=Path)
    src.add_argument('--text')
    p.add_argument('--font', type=Path)
    p.add_argument('--subject-prompt', required=True)
    p.add_argument('--surrounding-prompt', required=True)
    p.add_argument('--positive-prompt', default='artistic typography, detailed texture, clean background')
    p.add_argument('--negative-prompt', default='blurry, low quality, deformed, cluttered background')
    p.add_argument('--detection-prompt', help='Short detector caption; defaults to subject prompt.')
    p.add_argument('--output', type=Path, default=Path('results/pipeline'))
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--generation-seed', type=int, help='Final-stage seed; defaults to --seed.')
    p.add_argument('--deformation', choices=['depth', 'sds'], default='depth')
    p.add_argument('--strength', type=float, default=.76)
    p.add_argument('--deform-steps', type=int, default=50)
    p.add_argument('--sds-iterations', type=int, default=100)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--guidance-scale', type=float, default=7.5)
    p.add_argument('--gamma', type=float, default=.85)
    p.add_argument('--no-attention-control', action='store_true')
    p.add_argument('--no-cross-attention', action='store_true')
    p.add_argument('--subject-scale', type=float, default=1.1)
    p.add_argument('--surrounding-scale', type=float, default=.7)
    p.add_argument('--min-score', type=float, default=.4)
    p.add_argument('--min-area-ratio', type=float, default=.05)
    p.add_argument('--max-area-ratio', type=float, default=.6)
    p.add_argument('--device', default='auto')
    p.add_argument('--precision', choices=['auto', 'fp16', 'fp32'], default='auto')
    p.add_argument('--deform-precision', choices=['auto', 'fp16', 'fp32'], default='auto')
    p.add_argument('--variant', help='Diffusion checkpoint variant for SemTypo and ACG, e.g. fp16.')
    p.add_argument('--base-model', default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    p.add_argument('--depth-model', default='sd2-community/stable-diffusion-2-depth')
    p.add_argument('--detector-model', default='IDEA-Research/grounding-dino-tiny')
    p.add_argument('--subject-controlnet', default='lllyasviel/sd-controlnet-seg')
    p.add_argument('--surrounding-controlnet', default='lllyasviel/sd-controlnet-scribble')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    from modules.generation import GenerationConfig, resolve_device
    from modules.regional_decomposition import normalize_glyph, render_glyph
    from PIL import Image
    try:
        if not a.subject_prompt.strip() or not a.surrounding_prompt.strip():
            raise ValueError('Both manual prompts are required.')
        if a.resolution < 64 or a.resolution % 64:
            raise ValueError('resolution must be a multiple of 64, at least 64.')
        if not 0 <= a.seed < 2**63:
            raise ValueError('seed must lie in [0, 2**63).')
        generation_seed = a.seed if a.generation_seed is None else a.generation_seed
        if not 0 <= generation_seed < 2**63:
            raise ValueError('generation-seed must lie in [0, 2**63).')
        if not 0 < a.strength <= 1 or a.deform_steps < 1 or int(a.deform_steps * a.strength) < 1:
            raise ValueError('Invalid deformation strength or steps.')
        if a.sds_iterations < 1:
            raise ValueError('sds-iterations must be positive.')
        if not 0 <= a.min_score <= 1 or not 0 <= a.min_area_ratio <= a.max_area_ratio <= 1:
            raise ValueError('Invalid detection thresholds.')
        GenerationConfig(steps=a.steps, guidance_scale=a.guidance_scale,
                         gamma=a.gamma,
                         subject_scale=a.subject_scale, surrounding_scale=a.surrounding_scale).validate()
        resolve_device(a.device, a.precision)
        resolve_device(a.device, a.deform_precision)
        root = a.output.resolve()
        source = a.image.resolve() if a.image else a.font.resolve() if a.font else None
        if source is None or not source.is_file():
            raise ValueError('Provide a source image, or --text and an existing --font.')
        if a.text is not None:
            render_glyph(a.text, source, a.resolution, a.resolution, 32, 0)
            source_args = ['--text', a.text, '--font', str(source), '--width', str(a.resolution),
                           '--height', str(a.resolution)]
        else:
            with Image.open(source) as im:
                normalize_glyph(im)
            source_args = ['--image', str(source)]
        region, controls = root / 'decomposition', root / 'controls'
        common = ['--local-files-only'] if a.local_files_only else []
        variant = ['--variant', a.variant] if a.variant else []
        detect = [sys.executable, str(ROOT / 'RegionDecomp.py'), *source_args,
                  '--subject-prompt', a.detection_prompt or a.subject_prompt,
                  '--model', a.detector_model, '--output', str(region),
                  '--min-score', str(a.min_score), '--min-area-ratio', str(a.min_area_ratio),
                  '--max-area-ratio', str(a.max_area_ratio), *common]
        region_files = [region / n for n in ['raw_image.png', 'sub.png', 'surr.png', 'mask.png', 'pred.png', 'detection.json']]
        deform = [sys.executable, str(ROOT / ('SemTypo.py' if a.deformation == 'depth' else 'SDSDeform.py')),
                  '--input', str(region), '--output', str(controls), '--subject-prompt', a.subject_prompt,
                  '--resolution', str(a.resolution), '--seed', str(a.seed), '--device', a.device,
                  '--precision', a.deform_precision, *variant, *common]
        if a.deformation == 'depth':
            deform += ['--model', a.depth_model, '--num-images', '1', '--strength', str(a.strength),
                       '--steps', str(a.deform_steps)]
            extras = ['depth.png', 'semtypo.json']
        else:
            deform += ['--model', a.base_model, '--iterations', str(a.sds_iterations)]
            extras = ['displacement.npy', 'sds.json']
        control_files = [controls / n for n in ['sub_0.png', 'surr.png', 'mask.png', 'preview.png', *extras]]
        final = root / 'artistic.png'
        generate = [sys.executable, str(ROOT / 'ConComGen.py'), '--subject', str(controls / 'sub_0.png'),
                    '--surrounding', str(controls / 'surr.png'), '--mask', str(controls / 'mask.png'),
                    '--subject-prompt', a.subject_prompt, '--surrounding-prompt', a.surrounding_prompt,
                    '--positive-prompt', a.positive_prompt, '--negative-prompt', a.negative_prompt,
                    '--base-model', a.base_model, '--subject-controlnet', a.subject_controlnet,
                    '--surrounding-controlnet', a.surrounding_controlnet, '--output', str(final),
                    '--steps', str(a.steps), '--seed', str(generation_seed), '--device', a.device,
                    '--precision', a.precision, '--guidance-scale', str(a.guidance_scale),
                    '--gamma', str(a.gamma),
                    '--subject-scale', str(a.subject_scale), '--surrounding-scale', str(a.surrounding_scale),
                    *variant, *common]
        if a.no_attention_control:
            generate.append('--no-attention-control')
        if a.no_cross_attention:
            generate.append('--no-cross-attention')
        if a.dry_run:
            for command in [detect, deform, generate]:
                print(shlex.join(command))
            print('Dry run: inputs/settings validated; no models loaded or outputs written.')
            return 0
        root.mkdir(parents=True, exist_ok=True)
        run_stage('01_detection', detect, [source], region_files, root, a.resume)
        run_stage('02_deformation', deform, region_files, control_files, root, a.resume)
        run_stage('03_generation', generate, control_files, [final, final.with_suffix('.json')], root, a.resume)
        (root / 'pipeline.json').write_text(json.dumps({
            'knowledge_acquisition': False, 'deformation': a.deformation,
            'subject_prompt': a.subject_prompt, 'surrounding_prompt': a.surrounding_prompt,
            'output': str(final), 'sha256': fingerprint(final),
            'stage_records': ['01_detection.stage.json', '02_deformation.stage.json', '03_generation.stage.json'],
        }, ensure_ascii=False, indent=2))
        print(f'Complete: {final}')
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        p.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
