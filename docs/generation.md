# Final-stage artistic typography generation

This stage starts **after subject deformation**. It consumes a deformed subject
control image, a surrounding control image, a subject mask, and two text prompts.
It does not perform glyph rendering, Grounding DINO detection, prompt generation,
or SemTypo deformation.

## Install

Use Python 3.9–3.12 in a separate environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-generation.txt
```

The dependency file is for this generation stage; install `requirements-preprocessing.txt`
for SemTypo/HED and font rendering. See [preprocessing.md](preprocessing.md) for the earlier stages.
CUDA uses fp16 by default. CPU and Apple MPS
use fp32. `--device auto` prefers CUDA, then MPS, then CPU. xformers is not required.
For a CUDA-specific PyTorch wheel, install the matching PyTorch 2.5.1 build first.

## Input contract

| Input | Meaning |
| --- | --- |
| `--subject` | Deformed subject structure, placed at its intended position on the full canvas. Use a light structure on a dark background. |
| `--surrounding` | Prepared surrounding scribble/edge control image, with light edges on a dark background. It is not a photograph or an automatically processed raw environment image. |
| `--mask` | White pixels select the subject region; black pixels select the surrounding region. Use `--invert-mask` for the opposite convention. |
| `--subject-prompt` | Subject concept and appearance. |
| `--surrounding-prompt` | Surrounding texture/style. |

The three images must represent the **same full canvas**, not independently cropped
objects. With no explicit output size, they must have identical dimensions, each
a multiple of 64. `--resolution 512` explicitly stretches all three canvases to
512×512; alternatively use `--width 768 --height 512`. This permits different source
resolutions of the same layout. It does not align an incorrectly positioned crop.
Masks use nearest-neighbour resizing and a threshold of 128; both regions must be
present. Output resolution is also checked against the latent mask.

## Generate from prepared controls

First produce aligned controls using the [pipeline](pipeline.md) or
[preprocessing stages](preprocessing.md). Generated control images are not included in Git.
For the quick-start output directory:

```bash
python ConComGen.py \
  --subject results/my_rose/controls/sub_0.png \
  --surrounding results/my_rose/controls/surr.png \
  --mask results/my_rose/controls/mask.png \
  --subject-prompt 'a blooming red rose flower' \
  --surrounding-prompt 'green leaves, slender branches, thorns' \
  --resolution 512 --seed 45 \
  --output results/rose_variant.png
```

Alternatively, provide a prompt JSON with `--prompts /path/to/prompts.json`.

`--prompts` accepts `sub_prompt` and `surr_prompt`; the paper's `subject prompt`
and `surrounding prompt` field names are also accepted. Explicit prompt arguments
override their respective JSON fields. `--positive-prompt` is appended to both
prompts with a comma; set it to `''` to use only the supplied prompts.

Add `--dry-run` to validate images, prompts, sizes, settings and output paths
without downloading/loading models or writing files. `--help` works before ML
dependencies are installed.

### Model locations

By default, Diffusers downloads these pretrained models into its normal cache:

- Base: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Subject: `lllyasviel/sd-controlnet-seg`
- Surrounding: `lllyasviel/sd-controlnet-scribble`

Local checkpoint directories can be supplied with `--base-model`,
`--subject-controlnet` and `--surrounding-controlnet`. Use `--local-files-only` to
prevent network downloads. These must be compatible SD 1.x Diffusers checkpoints.
The existing SDXL pipeline is not used by this entry point.

LoRA is optional. To use the existing weight:

```bash
--lora loras/vitaglyph_1.safetensors
```

It applies to the shared base model, hence both branches. Its training provenance
and its role in the paper's reported results remain undocumented in the repository;
it is therefore **off by default**.

### Batch and repeated sampling

```bash
python ConComGen.py --input outs --subject-index 4 --resolution 512 \
  --num-images 4 --seed 42 --output results
```

Batch mode reads each `CCG/<name>/sub_<index>`, `surr`, `mask`, and
`LLM/<name>.json`. Images prefer `.png`, falling back to legacy `.jpg`/`.jpeg`.
It processes directories in sorted order and generates seeds
42, 43, 44 and 45 sequentially. Each sample starts with the same seed sequence.

Every output includes a JSON sidecar with prompts, seed, dimensions, input SHA-256
hashes, model identifiers/paths, LoRA hash if used, algorithm settings, device and
package versions. Outputs are not overwritten unless `--overwrite` is supplied.
Model identifiers alone do not lock changing remote weights; use an immutable
local snapshot for exact experiment provenance. Fixed seeds are reproducible on
the same stack; cross-device bitwise identity is not promised.

`--save-debug` additionally saves the first and last steps' neural sketches and
fused controls in `<output_stem>_debug/`.

## Algorithm and explicit implementation choices

Reference: [VitaGlyph v3, sections 3.3 and supplementary II](https://arxiv.org/pdf/2410.01738v3).
The paper does not supply all details needed to implement attention control. The
following choices are part of **this implementation**, not recovered experimental
settings or a claim that the paper's quantitative results have been reproduced.

1. **Backbone.** Use the repository's SD 1.5 baseline. The paper's deployment
   paragraph names SDXL but its limitations paragraph names SD 1.5. Those statements
   cannot establish a single unambiguous reference backbone. One UNet, VAE and text
   encoder are reused across sequential branch evaluations to avoid duplicate
   weights. Conditional residuals and text embeddings remain branch-specific.
2. **Controls.** Use exactly one segmentation ControlNet for the subject and one
   scribble ControlNet for the surrounding. Feed the prepared deformed subject
   directly as its control image, following the existing data convention; this
   stage does not run semantic segmentation or palette conversion.
3. **Noise and CFG.** Both branches see the same current latent at each DDIM step.
   The mask formula is `eps_cond = gamma*M*eps_sub + (1-M)*eps_surr`. The background
   mask is not `1-gamma*M`. Compute one unconditional/negative-prompt UNet prediction
   without ControlNet residuals, then apply
   `eps = eps_uncond + guidance_scale*(eps_cond-eps_uncond)`.
4. **Cross-branch attention.** At matching spatial self-attention layers in the
   denoising UNet, capture the subject keys and replace the surrounding keys with
   `alpha*k_surr + (1-alpha)*k_sub`. Queries and values stay in the surrounding
   branch. Text cross-attention is unchanged. Subject evaluation precedes surrounding
   evaluation at the same timestep. The paper mentions ControlNet insertion as well
   as replacing UNet attention; this implementation uses the latter interpretation.
5. **Attention-derived control.** At every timestep, run a conditional probe using
   the original control, collect mean attention received by spatial keys, normalize
   each map, resize and average across layers, Gaussian-smooth, normalize and
   threshold. Extract a one-pixel inner contour to form a bright-on-dark sketch.
   Clamp additions to the branch's region and take `maximum(original_control,
   region*sketch)`. Re-evaluate that branch using the fused control at the **same
   timestep**. Controls always start from the original image, not accumulated
   sketches. The final subject pass supplies keys for both surrounding passes.
6. **Attention-map cost.** By default only spatial layers with at most 1024 tokens
   contribute saliency maps. Cross-branch key mixing still operates at every spatial
   self-attention layer. Query chunks of 128 bound temporary attention-score memory;
   every query in each selected layer contributes, with no query subsampling.

The map reduction, layer selection, threshold, contour extraction, region clipping
and two-pass ordering are explicit choices because the paper does not prescribe
them. They require visual validation with the full pretrained models. ACG performs
five UNet calls per timestep (one unconditional and two per conditional branch),
so the paper's reported runtime does not describe this implementation.

### Parameters and ablations

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--steps` | 50 | DDIM steps, eta=0 |
| `--guidance-scale` | 7.5 | Shared CFG scale |
| `--gamma` | 0.85 | Subject noise weight |
| `--subject-scale` | 1.1 | Segmentation ControlNet residual scale |
| `--surrounding-scale` | 0.7 | Scribble ControlNet residual scale |
| `--alpha` | 0.5 | Surrounding-key contribution; 1 disables subject-key influence |
| `--attention-max-tokens` | 1024 | Largest spatial attention map used for sketch extraction |
| `--sketch-sigma` | 1.0 | Gaussian sigma in output pixels; 0 disables smoothing |
| `--sketch-threshold` | 0.5 | Threshold after map normalization |

Use `--no-cross-attention` or `--no-attention-control` for individual ablations.
Use both for the noise-composition baseline (three UNet calls per timestep).
This baseline uses corrected mask fusion and shared unconditional prediction; it
does not recreate the legacy script's accidental configurations.

One subject is supported. Multi-concept layout/detection, SDXL, image batches inside
one forward pass, IP-Adapter, and concurrent calls on one generator are outside
this entry point's scope. Old `modules/pipeline_*.py` copies remain for reference;
the new entry point does not import them.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use small, randomly initialized **real** Diffusers UNet/ControlNet/VAE and
Transformers text models, with a local tokenizer. No pretrained downloads are
needed. They cover numerical attention and mask fusion, both conditional branches,
shared step latents, attention-control region boundaries, deterministic CPU outputs,
processor cleanup, local checkpoint loading, PNG/metadata output, and MPS execution
when available. Random-model images validate execution only, not artistic quality.

Full SD 1.5/ControlNet generation quality, CUDA fp16 execution, and peak memory have
not been validated by these small-model tests.
