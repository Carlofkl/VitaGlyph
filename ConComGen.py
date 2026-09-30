"""Generate artistic typography from an already deformed subject and aligned controls."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--subject", type=Path, help="Deformed subject control image, on the full canvas.")
    source.add_argument("--input", type=Path, help="Batch directory containing CCG/<name>/ and LLM/<name>.json.")
    parser.add_argument("--surrounding", type=Path, help="Aligned surrounding scribble/edge control image.")
    parser.add_argument("--mask", type=Path, help="White = subject; black = surrounding.")
    parser.add_argument("--subject-prompt", help="Appearance/concept of the subject.")
    parser.add_argument("--surrounding-prompt", help="Texture/style of the surrounding region.")
    parser.add_argument("--prompts", type=Path, help="JSON with sub_prompt and surr_prompt (direct mode).")
    parser.add_argument("--positive-prompt", default="artistic typography, detailed texture, clean background")
    parser.add_argument("--negative-prompt", default="blurry, low quality, deformed, cluttered background")
    parser.add_argument("--output", type=Path, help="Output PNG, or output directory in batch mode.")
    parser.add_argument("--subject-index", type=int, default=0, help="Select sub_<index>.jpg in batch mode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-images", type=int, default=1, help="Sequential outputs with seed, seed+1, ...")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--resolution", type=int, help="Square output; alternative to --width/--height.")
    parser.add_argument("--invert-mask", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--subject-scale", type=float, default=1.1)
    parser.add_argument("--surrounding-scale", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.5, help="Surrounding-key weight in cross-branch attention.")
    parser.add_argument("--no-cross-attention", action="store_true", help="Disable cross-branch key mixing.")
    parser.add_argument("--no-attention-control", action="store_true", help="Disable attention-derived control fusion.")
    parser.add_argument("--attention-max-tokens", type=int, default=1024)
    parser.add_argument("--sketch-sigma", type=float, default=1.0)
    parser.add_argument("--sketch-threshold", type=float, default=0.5)
    parser.add_argument("--base-model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--variant", help="Base diffusion weight variant, e.g. fp16.")
    parser.add_argument("--subject-controlnet", default="lllyasviel/sd-controlnet-seg")
    parser.add_argument("--surrounding-controlnet", default="lllyasviel/sd-controlnet-scribble")
    parser.add_argument("--lora", type=Path, help="Optional local LoRA; omitted by default.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, mps or cpu")
    parser.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--save-debug", action="store_true", help="Save first/last-step attention sketches and fused controls.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/configuration; do not load models or write outputs.")
    return parser


def read_prompts(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Prompt JSON must be an object: {path}")
    subject = data.get("sub_prompt", data.get("subject prompt"))
    surrounding = data.get("surr_prompt", data.get("surrounding prompt"))
    return subject, surrounding


def input_jobs(args):
    from modules.image_files import find_image
    if args.subject:
        if not args.surrounding or not args.mask:
            raise ValueError("Direct mode requires --subject, --surrounding and --mask.")
        subject_prompt, surrounding_prompt = read_prompts(args.prompts) if args.prompts else (None, None)
        output = args.output or Path("results/artistic.png")
        if output.suffix.lower() != ".png":
            raise ValueError("Direct --output must be a .png filename.")
        return [(args.subject, args.surrounding, args.mask,
                 args.subject_prompt or subject_prompt, args.surrounding_prompt or surrounding_prompt, output)]
    if any((args.surrounding, args.mask, args.prompts, args.subject_prompt, args.surrounding_prompt)):
        raise ValueError("Batch mode reads images and prompts from --input; do not mix direct input options.")
    root = args.input / "CCG"
    if not root.is_dir():
        raise ValueError(f"Batch directory not found: {root}")
    output = args.output or Path("results")
    jobs = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        sub_prompt, surr_prompt = read_prompts(args.input / "LLM" / f"{folder.name}.json")
        jobs.append((find_image(folder, f"sub_{args.subject_index}"), find_image(folder, "surr"), find_image(folder, "mask"),
                     sub_prompt, surr_prompt, output / folder.name / "artistic.png"))
    if not jobs:
        raise ValueError(f"No sample directories found: {root}")
    return jobs


def output_path(path, seed, count):
    return path if count == 1 else path.with_name(f"{path.stem}_{seed}{path.suffix}")


def append_style(prompt, style):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Provide non-empty subject and surrounding prompts, or a --prompts JSON file.")
    return ", ".join(part.strip(" ,") for part in (prompt, style) if part.strip(" ,"))


def file_record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    # Keep --help available even before installing the ML dependencies.
    from modules.generation import GenerationConfig, VitaGlyphGenerator, load_inputs, resolve_device

    try:
        if args.resolution is not None:
            if args.width is not None or args.height is not None:
                raise ValueError("Use --resolution or --width/--height, not both.")
            args.width = args.height = args.resolution
        if args.num_images < 1 or args.seed < 0 or args.seed + args.num_images > 2**63:
            raise ValueError("num-images must be positive; seeds must lie in [0, 2**63).")
        if args.subject_index < 0:
            raise ValueError("subject-index must be non-negative.")
        if args.lora and not args.lora.is_file():
            raise ValueError(f"LoRA file not found: {args.lora}")
        config = GenerationConfig(
            steps=args.steps, guidance_scale=args.guidance_scale, gamma=args.gamma,
            subject_scale=args.subject_scale, surrounding_scale=args.surrounding_scale,
            alpha=args.alpha, cross_branch_attention=not args.no_cross_attention,
            attention_control=not args.no_attention_control, attention_max_tokens=args.attention_max_tokens,
            sketch_sigma=args.sketch_sigma, sketch_threshold=args.sketch_threshold,
        )
        config.validate()
        device, dtype = resolve_device(args.device, args.precision)
        jobs = input_jobs(args)
        # Validate every job before downloading/loading any checkpoint.
        for subject, surrounding, mask, sub_prompt, surr_prompt, output in jobs:
            images = load_inputs(subject, surrounding, mask, args.width, args.height, args.invert_mask)
            append_style(sub_prompt, args.positive_prompt)
            append_style(surr_prompt, args.positive_prompt)
            input_paths = {path.resolve() for path in (subject, surrounding, mask)}
            for seed in range(args.seed, args.seed + args.num_images):
                path = output_path(output, seed, args.num_images)
                if path.resolve() in input_paths:
                    raise ValueError(f"Output would overwrite an input image: {path}")
                for target in (path, path.with_suffix(".json")):
                    if target.exists() and not args.overwrite:
                        raise ValueError(f"Output exists: {target}; choose another path or --overwrite.")
            print(f"Validated {subject}: {images[0].width}x{images[0].height} -> {output}")
        if args.dry_run:
            print(f"Dry run passed: {len(jobs)} input(s), {args.num_images} image(s) each, {device}, {dtype}.")
            return 0
        generator = VitaGlyphGenerator.from_pretrained(
            args.base_model, args.subject_controlnet, args.surrounding_controlnet,
            args.device, args.precision, args.local_files_only, args.lora,
            variant=args.variant,
        )
        for subject, surrounding, mask, sub_prompt, surr_prompt, output in jobs:
            images = load_inputs(subject, surrounding, mask, args.width, args.height, args.invert_mask)
            sub_prompt = append_style(sub_prompt, args.positive_prompt)
            surr_prompt = append_style(surr_prompt, args.positive_prompt)
            for seed in range(args.seed, args.seed + args.num_images):
                path = output_path(output, seed, args.num_images)
                debug_dir = path.with_suffix("").with_name(path.stem + "_debug") if args.save_debug else None
                image, metadata = generator.generate(
                    *images, sub_prompt, surr_prompt, args.negative_prompt, seed, config, debug_dir,
                )
                metadata.update({
                    "implementation": "vitaglyph-sd1-acg-v1",
                    "subject_prompt": sub_prompt, "surrounding_prompt": surr_prompt,
                    "negative_prompt": args.negative_prompt,
                    "inputs": {name: file_record(p) for name, p in
                               (("subject", subject), ("surrounding", surrounding), ("mask", mask))},
                    "invert_mask": args.invert_mask,
                    "models": {"base": args.base_model, "subject_controlnet": args.subject_controlnet,
                               "surrounding_controlnet": args.surrounding_controlnet},
                    "lora": file_record(args.lora) if args.lora else None,
                    "base_weight_variant": args.variant, "vae_dtype": str(generator.pipe.vae.dtype),
                    "device": str(device), "dtype": str(dtype), "python": platform.python_version(),
                    "packages": {name: importlib.metadata.version(name) for name in
                                 ("torch", "diffusers", "transformers", "peft", "Pillow")},
                })
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"Saved {path}")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
