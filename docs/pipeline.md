# One-command VitaGlyph pipeline

Manual subject/surrounding prompts replace Knowledge Acquisition. No LLM or
knowledge-distillation service is called. The default route is:

`glyph → Grounding DINO → sub/surr/mask → Depth2Img + HED → dual-ControlNet ACG → artistic.png`

## Run

```bash
python -m pip install -r requirements-preprocessing.txt
python VitaGlyph.py \
  --image examples/rose_glyph.jpg \
  --detection-prompt 'a rose flower' \
  --subject-prompt 'a red rose flower blooming' \
  --surrounding-prompt 'green leaves, slender branches and thorns, botanical illustration' \
  --positive-prompt 'botanical watercolor illustration, only flowers and leaves, Chinese calligraphy, plain white background' \
  --negative-prompt 'people, person, human, skin, nudity, body, portrait, blurry, low quality, cluttered background' \
  --output results/my_rose --seed 42 --generation-seed 45 --steps 20 \
  --gamma 1 --guidance-scale 5 --subject-scale 0.8 --surrounding-scale 0.8 \
  --no-attention-control
```

On the tested Mac, append
`--device mps --precision fp16 --deform-precision fp32 --variant fp16`.
`--precision` controls final rendering; `--deform-precision` controls deformation.
VAE decoding in the final stage always uses fp32 to avoid half-precision overflow.
Depth2Img retains its fp32 depth estimator. On CPU, use fp32; CUDA may use fp16.

Use `--text '蔷' --font /path/to/font.ttf` in place of `--image` to render a glyph
first. Text and image inputs enter the same detector and downstream stages.

The detector retains confidence **>= 0.4** and box/image area **5%–60% inclusive**,
then returns the highest-confidence survivor. `--detection-prompt` can be a short
concept independent of the more descriptive generation prompt. No qualifying box
stops the pipeline without inventing another region.

Models are configurable through `--detector-model`, `--depth-model`, `--base-model`,
`--subject-controlnet` and `--surrounding-controlnet`. All accept local snapshot
folders. To run offline, use those local paths plus `--local-files-only`. A snapshot
cached by commit alone may not have the `main` cache reference needed by a bare
model ID; pass the snapshot directory in that case.

## Outputs and continuation

```text
results/my_rose/
  decomposition/       # raw_image, sub, surr, mask, detection preview and JSON
  controls/            # sub_0, HED surr, mask, deformation diagnostics and JSON
  artistic.png
  artistic.json        # final parameters, input hashes, package versions
  01_detection.stage.json
  02_deformation.stage.json
  03_generation.stage.json
  pipeline.json
```

Each stage runs in a separate process so its models leave memory before the next
stage starts. Failures stop immediately. `pipeline.json` is only written after all
three stages succeed. `--resume` skips a completed stage only when its command,
input hashes and output hashes match its stage record. This checks artifacts and
arguments, not changes in remote model revisions or source code; pin local models
and use a new output folder after changing implementation code.

Use a new output folder when changing parameters. Existing mismatched or incomplete
outputs are not silently overwritten. A failed stage that wrote no output can be
retried with `--resume`. `--dry-run` validates the source and parameters and prints
the three stage commands without downloading models or writing outputs; it does
not predict whether the detector will find a valid box.

Non-finite images and candidates rejected by the model's content checker are
errors, not successful black PNGs. If final rendering fails without writing an
output, use `--resume --generation-seed 43` to retry rendering while preserving
the detector and deformation results. `--seed` still governs deformation;
`--generation-seed` only overrides final rendering.

## Optional SDS deformation

```bash
python VitaGlyph.py \
  --image examples/rose_glyph.jpg \
  --detection-prompt 'a rose flower' \
  --subject-prompt 'a rose flower' --surrounding-prompt 'green leaves and branches' \
  --deformation sds --sds-iterations 100 \
  --output results/my_rose_sds
```

The SDS option is an original raster adapter informed by
[Word-As-Image](https://github.com/WordAsImage/Word-As-Image), inspected at commit
`ed72b2b33f7b2fecc5aecc610700973af754b2b7`, particularly its
[`SDSLoss`](https://github.com/Shiriluz/Word-As-Image/blob/ed72b2b33f7b2fecc5aecc610700973af754b2b7/code/losses.py).
It uses the score gradient `sqrt(alpha_bar_t) * (1-alpha_bar_t) * (epsilon_CFG-epsilon)`.
Gradients pass through the frozen VAE encoder to a trainable 16×16 displacement
grid. The frozen UNet predicts the score under `no_grad`; network weights are not
trained. Adam optimizes only displacement, bounded by `tanh` to ±24 source pixels.

The subject-mask bounding box is padded to square and rendered at 256×256 for SDS.
Guidance sees black strokes on white. The exported control remains white strokes
on black, at the original full-canvas coordinates, with zero outside the mask.
A blurred tone penalty, grid smoothness and displacement penalties constrain the
warp. This is **not** Word-As-Image's Bezier/diffvg optimizer or its conformal/ACAP
loss. It cannot add independent strokes or guarantee topology/readability; image
resampling and a bounded warp are deliberate adaptations for this PNG pipeline.
The logged SDS proxy is a gradient surrogate, not a comparable image-quality score.

For direct tuning:

```bash
python SDSDeform.py --input results/rose_pipeline/decomposition \
  --output results/sds_controls --subject-prompt 'a rose flower' \
  --iterations 100 --learning-rate 0.03 --max-displacement 24 \
  --tone-weight 10 --guidance-scale 100 --render-size 256
```

SDS saves `sub_0.png`, `surr.png`, `mask.png`, `preview.png`, `sds.json` with the
per-iteration history, and `displacement.npy` (2×grid×grid, source-sampling offsets
in pixels, x then y). The final generation interface is identical for both methods.
SDS uses backpropagation and is more expensive than the depth route. The CPU warp
and device transfer preserve gradients on MPS without requiring diffvg/CUDA builds.

## Validation

Paths in this section refer to local validation runs. Their generated artifacts are not included in Git.

The 33-test suite covers real small diffusion components, frozen-model SDS gradient
flow and determinism, canvas/mask preservation, stage routing, verified resume,
modified-output rejection, failure propagation, and content-check rejection without
saving a success image. It does not measure typography aesthetics or readability.

A full-pretrained SDS smoke run is saved in `results/sds_rose_smoke`: SD 1.5 fp16
checkpoint weights evaluated in fp32 on MPS, seed 42, three optimization iterations,
128×128 guidance view and a 512×512 output canvas. The learned grid is finite and
nonzero (maximum absolute offset about 2.145 pixels); pixels outside the mask remain
zero. Three iterations verify execution, not convergence or semantic quality.

The complete pretrained **depth route** produced
`results/rose_pipeline/artistic.png` (local validation output, not included in Git).
Its 512×512 output uses deformation seed 42 / strength 0.76 and generation seed 45 /
20 DDIM steps / guidance 5 / gamma 1 / both ControlNet scales 0.8. Cross-branch
attention is **on**, attention-driven control fusion is **off** (fixed controls).
The content checker returned false. Visual inspection confirms a readable stylized
glyph with green upper strokes; the lower subject still lacks clear rose details.
This is a working baseline artifact, not a claim of paper-level visual quality.

Earlier trials with full attention-driven control fusion, gamma 0.85 and guidance
7.5 were rejected by the model's checker. Several parameters changed in the valid
trial, so this does not isolate the cause of rejection. The full fusion code and
switches remain available: remove `--no-attention-control` to enable it. The
working command above intentionally records the configuration that produced the
valid example. The original filtered seed-42 artifact is retained under
`results/rose_pipeline/filtered_attempt_seed42` and is not a usable result.

The final metadata records the fixed model snapshot paths and package versions.
SD 1.5 snapshot: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`;
seg ControlNet: `ecdcb5645b5099c9a7500a504fb9ab3f743c4d96`;
scribble ControlNet: `864edcd5ccc6ee2695eeebea5b4512100c83e7b3`.
The SDS-to-final route is covered by the common interface and routing tests; a
long SDS optimization followed by full pretrained artistic rendering has not yet
been run. The delivered end-to-end artifact uses Depth2Img deformation.
