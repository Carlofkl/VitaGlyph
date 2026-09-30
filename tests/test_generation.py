import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from diffusers import AutoencoderKL, ControlNetModel, DDIMScheduler, StableDiffusionControlNetPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from transformers.models.clip.tokenization_clip import bytes_to_unicode

import ConComGen
from modules.acg_attention import AttentionState, CrossBranchAttnProcessor, attention_saliency, install_attention, neural_sketch
from modules.generation import GenerationConfig, VitaGlyphGenerator, compose_noise, load_inputs


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_noise_fusion_uses_unweighted_background_mask_and_common_unconditional():
    mask = torch.tensor([[[[0.0, 1.0]]]])
    subject = torch.full_like(mask, 4)
    surrounding = torch.full_like(mask, 10)
    unconditional = torch.full_like(mask, 2)
    # gamma=.5, s=3: outside = 2+3*(10-2), inside = 2+3*(.5*4-2).
    assert torch.equal(compose_noise(subject, surrounding, unconditional, mask, .5, 3),
                       torch.tensor([[[[26.0, 2.0]]]]))
    assert torch.equal(compose_noise(subject, surrounding, unconditional, mask, 1, 0), unconditional)


def test_cross_branch_keys_match_explicit_attention():
    torch.manual_seed(3)
    attn = Attention(query_dim=8, heads=2, dim_head=4)
    sub, surr = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    state = AttentionState((2, 2), alpha=.25)
    processor = CrossBranchAttnProcessor("layer", state)
    state.select("subject")
    processor(attn, sub)
    state.select("surrounding")
    actual = processor(attn, surr)
    q = attn.to_q(surr).view(1, 4, 2, 4).transpose(1, 2)
    k = (.25 * attn.to_k(surr) + .75 * attn.to_k(sub)).view(1, 4, 2, 4).transpose(1, 2)
    v = attn.to_v(surr).view(1, 4, 2, 4).transpose(1, 2)
    expected = ((q @ k.transpose(-1, -2) * attn.scale).softmax(-1) @ v)
    expected = expected.transpose(1, 2).reshape(1, 4, 8)
    expected = attn.to_out[1](attn.to_out[0](expected))
    torch.testing.assert_close(actual, expected)
    state.alpha = 1
    torch.testing.assert_close(processor(attn, surr), AttnProcessor2_0()(attn, surr))
    state.select("unconditional")
    torch.testing.assert_close(processor(attn, surr), AttnProcessor2_0()(attn, surr))


def test_chunked_attention_maps_and_empty_evidence():
    q, k = torch.randn(2, 3, 10, 4), torch.randn(2, 3, 10, 4)
    expected = (q @ k.transpose(-1, -2) * .5).softmax(-1).mean((1, 2))
    torch.testing.assert_close(attention_saliency(q, k, .5, chunk_size=3), expected)
    sketch = neural_sketch([torch.ones(1, 1, 4, 4)], (64, 64))
    assert sketch.count_nonzero() == 0
    peak = torch.zeros(1, 1, 8, 8)
    peak[:, :, 2:6, 2:6] = 1
    sketch = neural_sketch([peak], (64, 64))
    assert set(sketch.unique().tolist()) == {0, 1}
    assert sketch[0, 0, 32, 32] == 0 and sketch[0, 0, 0, 0] == 0
    assert sketch.count_nonzero() > 0


def write_inputs(tmp_path, width=64, height=64):
    paths = [tmp_path / name for name in ("subject.png", "surrounding.png", "mask.png")]
    Image.new("RGB", (width, height), "white").save(paths[0])
    Image.new("RGB", (width, height), "black").save(paths[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, :width // 2] = 255
    Image.fromarray(mask).save(paths[2])
    return paths


def test_inputs_keep_alignment_and_binary_masks(tmp_path):
    paths = write_inputs(tmp_path, 128, 64)
    images = load_inputs(*paths, width=256, height=128, invert_mask=True)
    assert all(im.size == (256, 128) for im in images)
    assert images[2].getpixel((0, 0)) == 0
    assert images[2].getpixel((255, 0)) == 255
    with pytest.raises(ValueError, match="multiples of 64"):
        load_inputs(*paths, width=100, height=64)
    Image.new("L", (64, 64)).save(paths[2])
    with pytest.raises(ValueError, match="full-canvas"):
        load_inputs(*paths)
    # Legacy sample masks may use a different resolution of the same canvas.
    small_mask = Image.new("L", (64, 64))
    small_mask.paste(255, (0, 0, 32, 64))
    small_mask.save(paths[2])
    assert all(im.size == (128, 64) for im in load_inputs(*paths, width=128, height=64))


@pytest.fixture(scope="module")
def tiny_generator(tmp_path_factory):
    """Real Diffusers/Transformers components, randomly initialized; no downloads."""
    torch.manual_seed(10)
    tmp = tmp_path_factory.mktemp("tokenizer")
    tokens = list(bytes_to_unicode().values())
    tokens += [token + "</w>" for token in tokens]
    tokens += ["<|startoftext|>", "<|endoftext|>"]
    (tmp / "vocab.json").write_text(json.dumps({token: i for i, token in enumerate(tokens)}))
    (tmp / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(tmp / "vocab.json"), str(tmp / "merges.txt"), model_max_length=32)
    text = CLIPTextModel(CLIPTextConfig(
        vocab_size=len(tokens), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, max_position_embeddings=32,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))
    unet = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32,
        attention_head_dim=4, norm_num_groups=8,
    )
    vae = AutoencoderKL(
        block_out_channels=(16, 16, 32, 32), in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
        latent_channels=4, norm_num_groups=8,
    )
    subject, surrounding = [ControlNetModel.from_unet(
        unet, conditioning_embedding_out_channels=(8, 16, 16, 32)
    ) for _ in range(2)]
    # ControlNet constructors zero the output layers. Make controls observable.
    for controlnet in (subject, surrounding):
        for module in [*controlnet.controlnet_down_blocks, controlnet.controlnet_mid_block,
                       controlnet.controlnet_cond_embedding.conv_out]:
            torch.nn.init.normal_(module.weight, std=.02)
    pipe = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text, tokenizer=tokenizer, unet=unet, controlnet=surrounding,
        scheduler=DDIMScheduler(clip_sample=False), safety_checker=None, feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    return VitaGlyphGenerator(pipe, subject)


def test_real_tiny_pipeline_shared_latents_controls_and_reproducibility(tiny_generator, tmp_path):
    images = load_inputs(*write_inputs(tmp_path, 128, 64))
    pipe = tiny_generator.pipe
    originals = dict(pipe.unet.attn_processors)
    unet_calls, control_calls = [], []

    def record_unet(module, args, kwargs):
        unet_calls.append((args[0].detach().clone(), "down_block_additional_residuals" in kwargs))

    def record_control(module, args, kwargs):
        control_calls.append((module, kwargs["controlnet_cond"].detach().clone()))

    hooks = [pipe.unet.register_forward_pre_hook(record_unet, with_kwargs=True),
             pipe.controlnet.register_forward_pre_hook(record_control, with_kwargs=True),
             tiny_generator.subject_controlnet.register_forward_pre_hook(record_control, with_kwargs=True)]
    config = GenerationConfig(steps=2)
    try:
        first, metadata = tiny_generator.generate(*images, "rose", "leaves", seed=7, config=config,
                                                 debug_dir=tmp_path / "debug")
    finally:
        for hook in hooks:
            hook.remove()
    assert first.size == (128, 64) and first.mode == "RGB"
    assert len(unet_calls) == 10  # One unconditional + two passes per conditional branch, per step.
    assert len(control_calls) == 8
    for start in (0, 5):
        assert not unet_calls[start][1]
        for tensor, controlled in unet_calls[start + 1:start + 5]:
            assert controlled
            torch.testing.assert_close(tensor, unet_calls[start][0])
    # Same-step sketches may only add control pixels in their respective region.
    for start in (0, 4):
        for offset, outside in ((0, slice(64, 128)), (2, slice(0, 64))):
            original, fused = control_calls[start + offset][1], control_calls[start + offset + 1][1]
            torch.testing.assert_close(original[:, :, :, outside], fused[:, :, :, outside])
    assert all(pipe.unet.attn_processors[name] is proc for name, proc in originals.items())
    assert len(list((tmp_path / "debug").glob("*.png"))) == 8
    assert metadata["seed"] == 7
    second, _ = tiny_generator.generate(*images, "rose", "leaves", seed=7, config=config)
    assert np.array_equal(np.array(first), np.array(second))
    baseline, _ = tiny_generator.generate(*images, "rose", "leaves", seed=7,
                                         config=GenerationConfig(steps=2, cross_branch_attention=False,
                                                                 attention_control=False))
    assert not np.array_equal(np.array(first), np.array(baseline))


def test_attention_processors_restored_after_exception(tiny_generator):
    unet = tiny_generator.pipe.unet
    original = dict(unet.attn_processors)
    state = AttentionState((8, 8))
    with pytest.raises(RuntimeError, match="intentional"):
        with install_attention(unet, state):
            state.subject_keys["fake"] = torch.ones(1)
            raise RuntimeError("intentional")
    assert not state.subject_keys
    assert all(unet.attn_processors[name] is proc for name, proc in original.items())


def test_local_checkpoint_loading_and_cpu_generation(tiny_generator, tmp_path):
    base, subject, surrounding = (tmp_path / name for name in ("base", "subject_model", "surrounding_model"))
    tiny_generator.pipe.save_pretrained(base)
    tiny_generator.subject_controlnet.save_pretrained(subject)
    tiny_generator.pipe.controlnet.save_pretrained(surrounding)
    loaded = VitaGlyphGenerator.from_pretrained(str(base), str(subject), str(surrounding),
                                               device="cpu", local_files_only=True)
    images = load_inputs(*write_inputs(tmp_path))
    result, _ = loaded.generate(*images, "rose", "leaves", config=GenerationConfig(steps=1))
    assert result.size == (64, 64)
    assert isinstance(loaded.pipe.scheduler, DDIMScheduler)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_generation(tiny_generator, tmp_path):
    tiny_generator.pipe.to("mps")
    tiny_generator.subject_controlnet.to("mps")
    try:
        images = load_inputs(*write_inputs(tmp_path))
        result, _ = tiny_generator.generate(*images, "rose", "leaves", config=GenerationConfig(steps=1))
        assert result.size == (64, 64)
        assert np.array(result).std() > 0
    finally:
        tiny_generator.pipe.to("cpu")
        tiny_generator.subject_controlnet.to("cpu")


def test_cli_validates_without_loading_models_and_rejects_overwrite(tmp_path, monkeypatch):
    paths = write_inputs(tmp_path)
    monkeypatch.setattr(VitaGlyphGenerator, "from_pretrained", lambda *a, **k: pytest.fail("Model load in dry run"))
    output = tmp_path / "result.png"
    args = ["--subject", str(paths[0]), "--surrounding", str(paths[1]), "--mask", str(paths[2]),
            "--subject-prompt", "rose", "--surrounding-prompt", "leaves", "--output", str(output),
            "--device", "cpu", "--dry-run"]
    assert ConComGen.main(args) == 0
    assert not output.exists()
    output.touch()
    with pytest.raises(SystemExit):
        ConComGen.main(args)


def test_cli_saves_png_and_provenance_with_real_components(tiny_generator, tmp_path, monkeypatch):
    paths = write_inputs(tmp_path)
    monkeypatch.setattr(VitaGlyphGenerator, "from_pretrained", lambda *a, **k: tiny_generator)
    output = tmp_path / "result.png"
    assert ConComGen.main([
        "--subject", str(paths[0]), "--surrounding", str(paths[1]), "--mask", str(paths[2]),
        "--subject-prompt", "rose", "--surrounding-prompt", "leaves", "--output", str(output),
        "--device", "cpu", "--steps", "1", "--num-images", "2", "--seed", "11",
    ]) == 0
    for seed in (11, 12):
        path = tmp_path / f"result_{seed}.png"
        with Image.open(path) as image:
            assert image.size == (64, 64)
        data = json.loads(path.with_suffix(".json").read_text())
        assert data["seed"] == seed and len(data["inputs"]["mask"]["sha256"]) == 64
        assert data["config"]["attention_control"] is True


@pytest.mark.parametrize("config", [GenerationConfig(gamma=2), GenerationConfig(alpha=float("nan")),
                                    GenerationConfig(steps=0), GenerationConfig(subject_scale=-1)])
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        config.validate()


def test_semtypo_real_depth_pipeline_and_downstream_controls(tiny_generator, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from transformers import DPTImageProcessor
    from diffusers import StableDiffusionDepth2ImgPipeline
    from modules.semantic_typography import SemanticTypography, SemanticDepth2ImgPipeline
    import SemTypo

    class DepthEstimator(torch.nn.Module):
        # Real diffusion components with a deterministic depth fixture; no MiDaS download.
        def __init__(self):
            super().__init__()
            self.calls = 0
        @property
        def device(self):
            return torch.device('cpu')
        @property
        def dtype(self):
            return torch.float32
        def forward(self, pixels):
            self.calls += 1
            height, width = pixels.shape[-2:]
            gradient = torch.linspace(0, 1, width, device=pixels.device).expand(pixels.shape[0], height, width)
            return SimpleNamespace(predicted_depth=gradient)

    pipe = tiny_generator.pipe
    unet = UNet2DConditionModel.from_config(pipe.unet.config, in_channels=5)
    depth = DepthEstimator()
    d2i = SemanticDepth2ImgPipeline(
        vae=pipe.vae, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer, unet=unet,
        scheduler=DDIMScheduler(clip_sample=False), depth_estimator=depth,
        feature_extractor=DPTImageProcessor(size={'height': 32, 'width': 32}),
    )
    d2i.set_progress_bar_config(disable=True)
    # Exercise the loader's setup against the real Depth2Img class too: unlike
    # StableDiffusionPipeline, this class has no enable_vae_slicing convenience API.
    monkeypatch.setattr('modules.semantic_typography.DPTForDepthEstimation.from_pretrained',
                        lambda *a, **kw: depth)
    monkeypatch.setattr(StableDiffusionDepth2ImgPipeline, 'from_pretrained', lambda *a, **kw: d2i)
    processor = SemanticTypography.from_pretrained(device='cpu', surrounding_preprocessor='outline')
    assert processor.pipeline.vae.use_slicing
    if torch.backends.mps.is_available():
        # Bicubic depth resampling is unsupported on pinned PyTorch's MPS backend.
        gradient = torch.linspace(0, 1, 32).expand(1, 32, 32)
        condition = d2i.prepare_depth_map(Image.new('RGB', (64, 64)), gradient.to('mps'),
                                         1, True, torch.float16, torch.device('mps'))
        assert condition.device.type == 'mps' and condition.dtype == torch.float16
        assert condition.shape == (2, 1, 8, 8)
        assert torch.isfinite(condition).all()
        assert condition.amin().item() == -1 and condition.amax().item() == 1
    source = tmp_path / 'decomposition'
    source.mkdir()
    paths = write_inputs(source)
    # SemTypo's input stems are sub/surr/mask.
    paths[0].rename(source / 'sub.png')
    paths[1].rename(source / 'surr.png')
    output = tmp_path / 'controls'
    monkeypatch.setattr(SemanticTypography, 'from_pretrained', lambda *a, **kw: processor)
    assert SemTypo.main([
        '--input', str(source), '--output', str(output), '--subject-prompt', 'rose',
        '--resolution', '64', '--num-images', '2', '--steps', '2', '--strength', '.75',
        '--surrounding-preprocessor', 'outline', '--device', 'cpu', '--seed', '10',
    ]) == 0
    assert depth.calls == 1  # Reuse depth across candidates.
    record = json.loads((output / 'semtypo.json').read_text())
    assert record['seeds'] == [10, 11]
    assert record['runtime']['diffusion_prompt'].startswith('a black and white drawing of rose')
    assert record['runtime']['denoising_steps'] == 1
    assert record['runtime']['scheduler'] == 'DDIMScheduler'
    assert record['candidates'] == [{'file': 'sub_0.png', 'seed': 10}, {'file': 'sub_1.png', 'seed': 11}]
    with Image.open(output / 'depth.png') as depth_image:
        assert depth_image.size == (64, 64)
        assert depth_image.getextrema() == (0, 255)
    with Image.open(output / 'preview.png') as preview:
        assert preview.size == (1024, 568)
    assert Image.open(output / 'mask.png').size == (64, 64)
    assert set(np.unique(Image.open(output / 'mask.png'))) == {0, 255}
    prepared = load_inputs(output / 'sub_0.png', output / 'surr.png', output / 'mask.png')
    result, _ = tiny_generator.generate(*prepared, 'rose', 'leaves', config=GenerationConfig(steps=1))
    assert result.size == (64, 64)
    # Avoid silently feeding stale variants to the downstream batch generator.
    with pytest.raises(SystemExit):
        SemTypo.main(['--input', str(source), '--output', str(output), '--subject-prompt', 'rose',
                      '--num-images', '1', '--overwrite', '--dry-run'])
    # Public Python API enforces the same binary mask contract as the CLI loader.
    with pytest.raises(ValueError, match='Mask must be binary'):
        processor.prepare(prepared[0], prepared[1], Image.new('L', (64, 64), 127), 'rose',
                          surrounding_preprocessor='outline')
    monkeypatch.setattr(SemanticDepth2ImgPipeline, '__call__',
                        lambda *a, **kw: SimpleNamespace(images=[np.full((64, 64, 3), np.nan)]))
    with pytest.raises(ValueError, match='Non-finite decoded image'):
        processor.prepare(*prepared, 'rose', count=1, surrounding_preprocessor='outline')


def test_batch_generation_prefers_png_and_accepts_legacy_jpg(tmp_path):
    folder = tmp_path / 'CCG' / 'rose'
    folder.mkdir(parents=True)
    for name in ['sub_0.png', 'sub_0.jpg', 'surr.png', 'mask.png']:
        Image.new('RGB', (64, 64)).save(folder / name)
    (tmp_path / 'LLM').mkdir()
    (tmp_path / 'LLM/rose.json').write_text(json.dumps({'sub_prompt': 'rose', 'surr_prompt': 'leaves'}))
    args = ConComGen.build_parser().parse_args(['--input', str(tmp_path)])
    job = ConComGen.input_jobs(args)[0]
    assert job[0].suffix == '.png'
    (folder / 'sub_0.png').unlink()
    assert ConComGen.input_jobs(args)[0][0].suffix == '.jpg'


def test_sds_gradient_reaches_geometry_through_frozen_vae(tiny_generator):
    from modules.sds_typography import SDSTypography, SDSConfig, RasterDeformation, score_gradient
    subject = Image.new('L', (64, 64), 0)
    subject.paste(255, (20, 15, 32, 45))
    mask = Image.new('L', (64, 64), 0)
    mask.paste(255, (8, 8, 56, 56))
    warp = RasterDeformation(subject, mask, 4, 8)
    torch.testing.assert_close(warp(), warp.source * warp.mask, atol=1e-5, rtol=1e-5)
    actual = score_gradient(torch.zeros(1), torch.ones(1), torch.ones(1) * 2,
                            torch.ones(1) * 3, torch.tensor(.25), 4)
    torch.testing.assert_close(actual, torch.tensor([1.875]))
    runner = SDSTypography(tiny_generator.pipe)
    cfg = SDSConfig(iterations=2, render_size=64, grid_size=4, max_displacement=8)
    image, offsets, record = runner.deform(subject, mask, 'rose', seed=5, config=cfg)
    assert np.isfinite(offsets).all() and np.abs(offsets).max() > 0
    assert np.abs(offsets).max() <= 8
    assert (np.array(image)[np.array(mask) == 0] == 0).all()
    assert len(record['history']) == 2
    for model in [runner.pipe.unet, runner.pipe.vae, runner.pipe.text_encoder]:
        assert all(not param.requires_grad and param.grad is None for param in model.parameters())
    again, repeated, _ = runner.deform(subject, mask, 'rose', seed=5, config=cfg)
    np.testing.assert_array_equal(offsets, repeated)
    np.testing.assert_array_equal(np.array(image), np.array(again))
    rendered, _ = tiny_generator.generate(image, Image.new('RGB', image.size), mask,
                                         'rose', 'leaves', config=GenerationConfig(steps=1))
    assert rendered.size == image.size


def test_rejected_generation_is_not_saved_as_success(tiny_generator, tmp_path, monkeypatch):
    paths = write_inputs(tmp_path)
    monkeypatch.setattr(VitaGlyphGenerator, 'from_pretrained', lambda *a, **kw: tiny_generator)
    monkeypatch.setattr(tiny_generator.pipe, 'run_safety_checker', lambda image, *a: (image, [True]))
    output = tmp_path / 'rejected.png'
    with pytest.raises(SystemExit):
        ConComGen.main(['--subject', str(paths[0]), '--surrounding', str(paths[1]), '--mask', str(paths[2]),
                       '--subject-prompt', 'rose', '--surrounding-prompt', 'leaves', '--output', str(output),
                       '--device', 'cpu', '--steps', '1'])
    assert not output.exists() and not output.with_suffix('.json').exists()
