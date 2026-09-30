"""Grounding DINO top-one region selection and full-canvas glyph decomposition."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


@dataclass(frozen=True)
class Detection:
    # Integer xyxy, right/bottom exclusive, in the original image coordinate system.
    box: tuple
    score: float
    normalized_cxcywh: tuple
    query_index: int
    caption: str
    area_ratio: Optional[float] = None


def prepare_caption(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Provide a non-empty subject description for Grounding DINO.")
    caption = prompt.lower().strip()
    return caption if caption.endswith(".") else caption + "."


def select_top_box(boxes, scores, image_size, caption, min_score=0.4,
                   min_area_ratio=0.05, max_area_ratio=0.6):
    """Filter by box area and confidence, then return the highest-scoring survivor.

    Scores follow the official repository: max sigmoid(logit) across text tokens
    for each object query. Area bounds are inclusive; confidence is at least
    min_score. Area uses the continuous box clipped to the canvas, before rounding
    to integer mask coordinates. There is no NMS or box merging.
    """
    boxes = torch.as_tensor(boxes).detach().float().cpu()
    scores = torch.as_tensor(scores).detach().float().cpu()
    if boxes.ndim != 2 or boxes.shape[-1] != 4 or scores.shape != boxes.shape[:1]:
        raise ValueError("Expected boxes [N,4] in normalized cxcywh format and scores [N].")
    if scores.numel() == 0:
        raise ValueError("Grounding DINO returned no candidate boxes.")
    if not torch.isfinite(scores).all() or not torch.isfinite(boxes).all():
        raise ValueError("Grounding DINO returned non-finite predictions.")
    if not math.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError("min_score must lie in [0,1].")
    if not 0 <= min_area_ratio <= max_area_ratio <= 1:
        raise ValueError("Area bounds must satisfy 0 <= min_area_ratio <= max_area_ratio <= 1.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Confidence scores must lie in [0,1].")
    # Use double precision for corner subtraction, then compare at model precision.
    geometry = boxes.double()
    lower = (geometry[:, :2] - geometry[:, 2:] / 2).clamp(0, 1)
    upper = (geometry[:, :2] + geometry[:, 2:] / 2).clamp(0, 1)
    area_ratios = (upper - lower).clamp_min(0).prod(dim=-1).to(boxes.dtype)
    eligible = ((boxes[:, 2:] > 0).all(dim=-1) & (area_ratios > 0)
                & (area_ratios >= min_area_ratio) & (area_ratios <= max_area_ratio)
                & (scores >= min_score))
    if not eligible.any():
        raise ValueError(f"No box satisfies area ratio [{min_area_ratio}, {max_area_ratio}] "
                         f"and confidence >= {min_score}; no region outputs generated.")
    index = int(scores.masked_fill(~eligible, -torch.inf).argmax())
    score = float(scores[index])
    cx, cy, bw, bh = boxes[index].tolist()
    if bw <= 0 or bh <= 0:
        raise ValueError("Highest-confidence box has non-positive dimensions.")
    width, height = image_size
    x0 = max(0, min(width, math.floor((cx - bw / 2) * width)))
    y0 = max(0, min(height, math.floor((cy - bh / 2) * height)))
    x1 = max(0, min(width, math.ceil((cx + bw / 2) * width)))
    y1 = max(0, min(height, math.ceil((cy + bh / 2) * height)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Highest-confidence box is empty after clipping to the image.")
    return Detection((x0, y0, x1, y1), score, (cx, cy, bw, bh), index, caption,
                     float(area_ratios[index]))


class GroundingDinoDetector:
    """Official IDEA-Research weights through the supported Transformers adapter."""

    def __init__(self, model, processor, model_id="custom", revision=None):
        self.model = model.eval()
        self.processor = processor
        self.model_id = str(model_id)
        self.revision = revision

    @classmethod
    def from_pretrained(cls, model_id="IDEA-Research/grounding-dino-tiny", device="auto",
                        local_files_only=False, revision=None):
        if device == "auto":
            # Portable CPU path on Macs; native GroundingDINO CUDA extensions are not required.
            device = "cuda" if torch.cuda.is_available() else "cpu"
        target = torch.device(device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("Grounding DINO detection supports cpu or cuda[:index]; use cpu on Mac.")
        if target.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested for detection but is unavailable.")
        options = {"local_files_only": local_files_only, "revision": revision}
        processor = AutoProcessor.from_pretrained(model_id, **options)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, disable_custom_kernels=True, **options,
        ).to(target)
        return cls(model, processor, model_id, getattr(model.config, "_commit_hash", None) or revision)

    @torch.inference_mode()
    def detect(self, image, prompt, min_score=0.4, min_area_ratio=0.05, max_area_ratio=0.6):
        caption = prepare_caption(prompt)
        inputs = self.processor(images=image.convert("RGB"), text=caption, return_tensors="pt")
        # Avoid truncating the concept silently (Grounding DINO supports up to 256 tokens).
        if inputs["input_ids"].shape[-1] > self.model.config.max_text_len:
            raise ValueError("Detection prompt is too long; use a short subject description.")
        inputs = inputs.to(self.model.device)
        outputs = self.model(**inputs)
        scores = outputs.logits[0].sigmoid().amax(dim=-1)
        return select_top_box(outputs.pred_boxes[0], scores, image.size, caption,
                              min_score, min_area_ratio, max_area_ratio)


def normalize_glyph(image, foreground="auto"):
    """Return light strokes on black for decomposition and dark strokes for detection.

    Auto polarity assumes a plain background and uses the median border intensity.
    No glyph thresholding is applied, so antialiased strokes are preserved.
    """
    if foreground not in ("auto", "dark", "light"):
        raise ValueError("foreground must be auto, dark or light.")
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = "black" if foreground == "light" else "white"
        canvas = Image.new("RGBA", image.size, background)
        canvas.alpha_composite(rgba)
        image = canvas.convert("RGB")
    gray = image.convert("L")
    if foreground == "auto":
        array = np.asarray(gray)
        border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))
        foreground = "dark" if np.median(border) >= 128 else "light"
    light = ImageOps.invert(gray) if foreground == "dark" else gray
    if light.getextrema()[0] == light.getextrema()[1]:
        raise ValueError("Input glyph is blank or uniform.")
    return light.convert("RGB"), ImageOps.invert(light).convert("RGB"), foreground


def render_glyph(text, font_path, width=512, height=512, padding=32, font_index=0):
    """Fit a single-line glyph/word to a canvas using the specified TTF/OTF/TTC."""
    if not text.strip() or "\n" in text or "\r" in text:
        raise ValueError("Glyph text must be a non-empty single line.")
    if width < 1 or height < 1 or padding < 0 or 2 * padding >= min(width, height):
        raise ValueError("Canvas and padding leave no drawable area.")
    if not Path(font_path).is_file():
        raise ValueError(f"Font file not found: {font_path}")
    # Verify character coverage; missing glyphs must not silently become tofu boxes.
    from fontTools.ttLib import TTFont
    font_data = TTFont(str(font_path), fontNumber=font_index)
    try:
        cmap = font_data.getBestCmap() or {}
        missing = [character for character in dict.fromkeys(text) if not character.isspace() and ord(character) not in cmap]
        if missing:
            raise ValueError(f"Font does not contain these characters: {''.join(missing)}")
    finally:
        font_data.close()
    max_width, max_height = width - 2 * padding, height - 2 * padding
    low, high, best = 1, max(width, height) * 4, None
    while low <= high:
        size = (low + high) // 2
        font = ImageFont.truetype(str(font_path), size=size, index=font_index)
        left, top, right, bottom = font.getbbox(text)
        if right - left <= max_width and bottom - top <= max_height:
            best = font
            low = size + 1
        else:
            high = size - 1
    if best is None:
        raise ValueError("Text cannot fit inside the canvas.")
    left, top, right, bottom = best.getbbox(text)
    image = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(image).text(((width - right + left) / 2 - left,
                               (height - bottom + top) / 2 - top), text, font=best, fill="black")
    return image


def split_regions(light_glyph, detection):
    """Return full-size RGB sub/surr and an exact rectangular uint8 mask."""
    width, height = light_glyph.size
    x0, y0, x1, y1 = detection.box
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError("Detection box must lie inside the glyph canvas.")
    mask = Image.new("L", (width, height), 0)
    mask.paste(255, (x0, y0, x1, y1))
    blank = Image.new("RGB", light_glyph.size, "black")
    glyph = light_glyph.convert("RGB")
    return Image.composite(glyph, blank, mask), Image.composite(blank, glyph, mask), mask


def save_decomposition(output_dir, glyph, detection_image, detection, metadata=None, overwrite=False):
    output_dir = Path(output_dir)
    filenames = ("raw_image.png", "sub.png", "surr.png", "mask.png", "pred.png", "detection.json")
    existing = [str(output_dir / name) for name in filenames if (output_dir / name).exists()]
    if existing and not overwrite:
        raise ValueError(f"Outputs already exist: {', '.join(existing)}; choose another folder or --overwrite.")
    sub, surr, mask = split_regions(glyph, detection)
    annotated = detection_image.copy().convert("RGB")
    x0, y0, x1, y1 = detection.box
    ImageDraw.Draw(annotated).rectangle((x0, y0, x1 - 1, y1 - 1), outline="red", width=2)
    record = dict(metadata or {})
    record.update({"detection": asdict(detection), "size": list(glyph.size),
                   "mask_convention": "white=subject, black=surrounding",
                   "box_convention": "xyxy pixels, right/bottom exclusive",
                   "selection": "highest_confidence_after_filtering", "output_format": "PNG",
                   "has_subject_ink": sub.convert("L").getextrema()[1] >= 128,
                   "has_surrounding_ink": surr.convert("L").getextrema()[1] >= 128})
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in zip(filenames, (glyph, sub, surr, mask, annotated)):
        image.save(output_dir / name)
    (output_dir / "detection.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def file_fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}
