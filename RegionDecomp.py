"""Select Grounding DINO's highest-confidence box and export sub, surr and mask."""

import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Existing glyph image.")
    source.add_argument("--text", help="Render this glyph/word with --font before detection.")
    parser.add_argument("--font", type=Path, help="TTF/OTF/TTC font, required with --text.")
    parser.add_argument("--font-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=512, help="Text-rendering canvas width.")
    parser.add_argument("--height", type=int, default=512, help="Text-rendering canvas height.")
    parser.add_argument("--padding", type=int, default=32, help="Text-rendering margin.")
    parser.add_argument("--foreground", choices=("auto", "dark", "light"), default="auto")
    parser.add_argument("--subject-prompt", required=True, help="Detection concept, e.g. 'a rose flower'.")
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--revision", help="Optional immutable Hugging Face model revision.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:index. auto uses CPU on Macs.")
    parser.add_argument("--min-score", type=float, default=0.4, help="Confidence must be at least this value.")
    parser.add_argument("--min-area-ratio", type=float, default=0.05, help="Inclusive minimum box/canvas area ratio.")
    parser.add_argument("--max-area-ratio", type=float, default=0.6, help="Inclusive maximum box/canvas area ratio.")
    parser.add_argument("--output", type=Path, default=Path("preds/sample"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate image/font and paths without loading the detector.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    from PIL import Image
    from modules.regional_decomposition import (GroundingDinoDetector, file_fingerprint, normalize_glyph,
                                                prepare_caption, render_glyph, save_decomposition)

    try:
        prepare_caption(args.subject_prompt)
        if not 0 <= args.min_score <= 1:
            raise ValueError("min-score must lie in [0,1].")
        if not 0 <= args.min_area_ratio <= args.max_area_ratio <= 1:
            raise ValueError("Area bounds must satisfy 0 <= min-area-ratio <= max-area-ratio <= 1.")
        if args.text is not None:
            if args.font is None:
                raise ValueError("--text requires --font.")
            if args.foreground == "light":
                raise ValueError("Rendered text uses dark foreground; use --foreground auto or dark.")
            image = render_glyph(args.text, args.font, args.width, args.height, args.padding, args.font_index)
            source = {"text": args.text, "font": file_fingerprint(args.font), "font_index": args.font_index,
                      "padding": args.padding}
        else:
            if args.font:
                raise ValueError("--font is only used with --text.")
            with Image.open(args.image) as loaded:
                image = loaded.copy()
            source = {"image": file_fingerprint(args.image)}
        glyph, detection_image, polarity = normalize_glyph(image, args.foreground)
        names = ("raw_image.png", "sub.png", "surr.png", "mask.png", "pred.png", "detection.json")
        for name in names:
            path = args.output / name
            if args.image and path.resolve() == args.image.resolve():
                raise ValueError(f"Output would overwrite the source image: {path}")
            if path.exists() and not args.overwrite:
                raise ValueError(f"Output exists: {path}; choose another folder or --overwrite.")
        print(f"Glyph {glyph.width}x{glyph.height}; foreground={polarity}; prompt={args.subject_prompt!r}")
        if args.dry_run:
            print(f"Dry run passed; output directory: {args.output}")
            return 0
        detector = GroundingDinoDetector.from_pretrained(args.model, args.device,
                                                        args.local_files_only, args.revision)
        detection = detector.detect(detection_image, args.subject_prompt, args.min_score,
                                    args.min_area_ratio, args.max_area_ratio)
        metadata = {"source": source, "foreground": polarity, "model": detector.model_id,
                    "model_revision": detector.revision, "device": str(detector.model.device),
                    "min_score": args.min_score, "confidence_comparison": ">=", "min_area_ratio": args.min_area_ratio,
                    "max_area_ratio": args.max_area_ratio}
        record = save_decomposition(args.output, glyph, detection_image, detection, metadata, args.overwrite)
        print(f"Top box xyxy={detection.box}, confidence={detection.score:.6f}")
        print(f"Saved sub.png, surr.png, mask.png and detection metadata to {args.output}")
        if not record["has_surrounding_ink"]:
            print("The top box contains all visible strokes; surr is empty. The highest-score box was kept unchanged.")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
