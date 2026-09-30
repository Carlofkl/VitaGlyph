# Regional decomposition and Semantic Typography

Knowledge Acquisition is intentionally omitted. Supply a subject concept yourself;
the final generation stage additionally needs a surrounding prompt.

## Install

```bash
python -m pip install -r requirements-preprocessing.txt
```

Grounding DINO uses the [official IDEA-Research weights](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
through the Transformers integration linked by the
[official repository](https://github.com/IDEA-Research/GroundingDINO). It does not
require cloning/compiling GroundingDINO's native CUDA extension. The existing
generation dependencies already support image-based region detection. The additional
preprocessing dependencies provide font coverage checks, HED and torchvision.

## 1. Detect one region and split the glyph

From an existing glyph image:

```bash
python RegionDecomp.py \
  --image examples/rose_glyph.jpg \
  --subject-prompt 'a rose flower' \
  --output results/decomposition_rose
```

Or render a glyph/word from a font before detection:

```bash
python RegionDecomp.py \
  --text '蔷' --font '/path/to/chinese-font.ttf' \
  --width 512 --height 512 --padding 32 \
  --subject-prompt 'a rose flower' \
  --output results/decomposition_rose
```

TTF, OTF and TTC fonts are supported. Use `--font-index` for TTC collections.
Characters missing from the selected font produce an error instead of silently
rendering replacement boxes. Text rendering fits one line to the requested canvas.
When `--image` is used, original dimensions are preserved.

### Selection rule

- Normalize the detection caption to lowercase with a final period, following the
  [official inference utility](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/util/inference.py).
- Score each query by the maximum sigmoid similarity over text tokens, as in that utility.
- Keep candidates whose box/canvas area ratio is in **[0.05, 0.6]**, inclusive,
  and whose confidence is **at least 0.4**.
- Compute area from the continuous box clipped to the canvas, before rounding
  to integer pixel coordinates. The denominator is the entire image area, not ink area.
- Select exactly one query with the highest score **among those candidates**.
  Equal scores select the first surviving query.
- Convert normalized `cx,cy,w,h` to source-image `x0,y0,x1,y1`, round the lower
  edges down and upper edges up, and clip to the canvas.
- Defaults are `--min-area-ratio 0.05 --max-area-ratio 0.6 --min-score 0.4`.
- If none qualify, report an error and do not write new region outputs. Existing
  outputs from earlier runs are left untouched. There is no fallback to an unfiltered box.
- There is no NMS, box merging or SAM segmentation. Non-finite predictions raise an error.

The score is Grounding DINO's text similarity score, not a calibrated probability
that the selected glyph region is semantically correct. Even an eligible box can
contain all strokes of a small glyph. The CLI reports an empty surrounding region
without substituting another box.

### Output

| File | Content |
| --- | --- |
| `raw_image.png` | Full-canvas glyph, light strokes on black. |
| `sub.png` | Glyph pixels inside the selected rectangle; black everywhere else. |
| `surr.png` | Glyph pixels outside the selected rectangle; black inside. |
| `mask.png` | Hard rectangular mask: 255 inside, 0 outside. |
| `pred.png` | Dark glyph on white with exactly one red box for inspection. |
| `detection.json` | One selected box, area ratio, full-precision score, filter thresholds, model revision, input hash, prompt and conventions. |

All images share the source canvas; **sub is not cropped/rescaled into a separate
object image**. The mask denotes the box, not only the glyph strokes. `sub + surr`
reconstructs `raw_image` exactly, including antialiased pixels. PNG avoids JPEG mask
artifacts. In JSON, right/bottom box coordinates are exclusive.

`--foreground auto` infers dark/light strokes from the border median. Detection is
always performed on dark strokes over white; decomposition exports light strokes
over black, matching the original repository's conventions. For unusual backgrounds
or transparent images, set `--foreground dark` or `light` explicitly. This stage is
for glyph images, not photographic background removal.

Use `--model /path/to/hf-grounding-dino-directory --local-files-only` for offline
weights, or `--revision <commit>` to pin a remote revision. This adapter accepts
Transformers-format weights, not the native repository's standalone `.pth` plus
Python config pair. `--device auto` chooses CUDA if available and CPU otherwise.
Use CPU for detection on Macs; generation can still use MPS.

`--dry-run` checks the image/font and output paths without loading the model.
Existing outputs require `--overwrite`, and the source image cannot be overwritten.

## 2. Deform the subject and prepare the surrounding control

```bash
python SemTypo.py \
  --input results/decomposition_rose \
  --output results/rose_controls \
  --subject-prompt 'a red rose flower blooming' \
  --resolution 512 --seed 42 --num-images 5
```

The input folder is the previous stage's output. SemTypo reads `sub.png`, `surr.png`
and `mask.png` (legacy `.jpg`/`.jpeg` also work), resizes them to an aligned canvas,
then produces:

```text
results/rose_controls/
  sub_0.png ... sub_4.png
  surr.png
  mask.png
  depth.png
  preview.png
  semtypo.json
```

Depth-to-image receives the full-canvas subject and the manually supplied prompt.
The depth estimator runs once in fp32 and its depth map is reused for all candidates.
Each variant uses seed `seed + index`. `--strength` defaults to 0.76, `--steps` to 50,
and `--guidance-scale` to 10. `--width/--height` can override square `--resolution`.
The same nearest-neighbour-resized, binary mask is preserved as lossless PNG.
`depth.png` visualizes the estimated depth, normalized to 0–255 for inspection;
inference uses the original floating-point depth tensor, not this saved image.
`preview.png` shows the input, depth, scribble, mask and all candidates with their
seeds. `semtypo.json` records the actual diffusion prompt (including its monochrome
prefix), scheduler configuration, effective denoising steps, execution device and
dtypes, input hashes, and candidate filenames/seeds.

The subject candidates are the original full-canvas Depth2Img outputs. There is
no additional thresholding, inversion or clipping to the rectangular mask; the
mask controls region composition in the final stage. Inspect the candidates and
choose the desired `sub_N.png` for that stage. Increasing `--strength` gives the
model more freedom to alter the glyph structure; compare 0.60, 0.76 and 0.85 with
the same seed in separate output folders. This is a manual tradeoff between
concept detail and retaining the glyph geometry, not an automatic quality ranking.

The surrounding is processed with HED in scribble mode, using
`lllyasviel/Annotators/ControlNetHED.pth`. `--surrounding-preprocessor outline` is
an explicit weight-free alternative for clean light-on-dark glyphs: it extracts an
inner binary contour. It is not claimed to reproduce HED output.

Model paths are configurable with `--model` and `--hed-model`; `--local-files-only`
applies to both. The default depth model is the accessible community mirror
[`sd2-community/stable-diffusion-2-depth`](https://huggingface.co/sd2-community/stable-diffusion-2-depth).
The old `stabilityai/stable-diffusion-2-depth` endpoint returned HTTP 401 during
implementation. The mirror has the SD2 Depth2Img pipeline/configuration; equality
with the old private local snapshot has not been established. To reproduce an
existing experiment, pass that experiment's own local checkpoint explicitly.
`--revision <commit>` pins a remote Depth2Img revision. `--variant fp16` loads the
smaller diffusion weights. `--precision` separately controls computation dtype;
using fp32 arithmetic on fp16 weights does not restore full-precision weights.
Depth estimation still uses fp32 weights and computation. VAE and attention
slicing are enabled to reduce peak memory. For example, on a Mac:

```bash
python SemTypo.py --input results/decomposition_rose \
  --output results/rose_controls --subject-prompt 'a rose flower blooming' \
  --device mps --precision fp32 --variant fp16 --num-images 3
```

For the pinned PyTorch version, MPS does not implement bicubic depth resampling.
The adapter performs that small condition-resizing/normalization operation in
fp32 on CPU, then transfers the condition to MPS. Model inference remains on MPS;
no global operator fallback is required.
Use fp32 computation on MPS: the tested fp16 run produced non-finite output.
Latents and decoded images are checked for finite values, and an invalid run
raises an error before candidate files are saved.

Existing files require `--overwrite`. If an older run has additional `sub_N.png`
candidates beyond the newly requested count, use a new folder; these files are
not silently reused or deleted.

`--prompts file.json` can replace `--subject-prompt` for one sample. For legacy batch
mode, pass a parent folder containing decomposed sample directories and use
`--prompts-root outs/LLM`; outputs are grouped by sample name. No prompts are
generated or sent to an LLM.

## 3. Render the final artistic typography

```bash
python ConComGen.py \
  --subject results/rose_controls/sub_0.png \
  --surrounding results/rose_controls/surr.png \
  --mask results/rose_controls/mask.png \
  --subject-prompt 'a red rose flower blooming' \
  --surrounding-prompt 'green leaves, slender branches, thorns' \
  --seed 42 --output results/rose.png
```

See [generation.md](generation.md) for generation parameters and model details.
The raw regional `surr.png` contains glyph strokes; the prepared SemTypo `surr.png`
contains the scribble expected by the final ControlNet. Use the latter for generation.

## Verification

Paths in this section refer to local validation runs. Their generated artifacts are not included in Git.

The pretrained `IDEA-Research/grounding-dino-tiny` checkpoint was run on the supplied
rose glyph with `a rose flower` before area/confidence filtering was added.
Its unfiltered highest box was `(29, 27, 472, 488)` on a
500×512 canvas, score approximately 0.467509. It covered all visible strokes, so
the exported surrounding contained only near-black JPEG residue. This is an actual
top-one detector result, not the smaller manually supplied region in the old example.
That box is still rejected because its area is above 0.6. Re-running with the current
5%-60% area bounds and confidence >= 0.4 selects query 1: pixel box
`(112, 292, 391, 488)`, confidence 0.432271, continuous area ratio 0.211826.
The updated `results/decomposition_rose` contains this lower glyph region;
older unfiltered results remain under `results/archive`.

The pretrained HED detector was also run on the existing surrounding glyph and
produced a 512×512 RGB scribble. Tests cover box ranking/clipping, preservation of canvas coordinates, exact region
reconstruction, polarity, font coverage, real small Grounding DINO forward inference,
and SemTypo-to-final-generation compatibility using real small Diffusers components
and a fixed depth fixture. The suite also exercises the real Depth2Img loader setup,
MPS depth resampling, diagnostic outputs and rejection of non-finite decoded images.

Full pretrained SemTypo was run on the updated `results/decomposition_rose` input:
`sd2-community/stable-diffusion-2-depth` revision
`6cb92dd9430a7f6da8d9e99d7b60acdebcc348b7`, fp16 diffusion weights with fp32
computation on MPS, fp32 DPT weights, and pretrained HED. The 512×512 run used
seeds 42/43/44, 50 nominal DDIM steps, strength 0.76 (38 denoising steps), and
guidance 10. Results are in `results/rose_controls`, including the contact sheet.
The mask exactly matches the nearest-neighbour-resized decomposition mask, and
the final generation CLI accepts the exported controls in dry-run mode.

Visual inspection shows altered strokes but weak rose semantics for these 0.76
candidates. Execution success is not a claim of reproducing the paper's image
quality. Final pretrained artistic rendering has not been run with these controls.
A same-seed (42) comparison at strength 0.85, with all other settings unchanged,
is saved under `results/rose_controls_strength085`. It bends the strokes further,
but also does not yet produce an unambiguous rose shape.
