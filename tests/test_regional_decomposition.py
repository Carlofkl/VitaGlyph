import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest
import torch
from transformers import (BertConfig, BertTokenizerFast, GroundingDinoConfig,
                          GroundingDinoForObjectDetection, GroundingDinoImageProcessor,
                          GroundingDinoProcessor, SwinConfig)

import RegionDecomp
from modules.regional_decomposition import (Detection, GroundingDinoDetector, normalize_glyph,
                                           prepare_caption, render_glyph, save_decomposition,
                                           select_top_box, split_regions)


def test_filter_candidates_before_ranking():
    boxes = [[.5, .5, .8, .8], [.25, .25, .1, .1], [.5, .5, .5, 1], [.5, .5, 1, .5]]
    result = select_top_box(boxes, [.99, .98, .8, .9], (200, 100), 'rose.')
    assert result.query_index == 3
    assert result.score == pytest.approx(.9) and result.area_ratio == .5
    assert result.box == (0, 25, 200, 75)
    with pytest.raises(ValueError, match='No box satisfies'):
        select_top_box(boxes, [.99, .98, .3999, .3], (200, 100), 'rose.')


@pytest.mark.parametrize('ratio', [.05, .6])
def test_area_and_confidence_endpoints_are_inclusive(ratio):
    boxes = [[.5, .5, ratio, 1]]
    assert select_top_box(boxes, [.4], (200, 100), 'rose.').area_ratio == pytest.approx(ratio)
    with pytest.raises(ValueError, match='No box satisfies'):
        select_top_box(boxes, [.3999], (200, 100), 'rose.')


@pytest.mark.parametrize('ratio', [.0499, .6001])
def test_reject_area_outside_bounds(ratio):
    with pytest.raises(ValueError, match='No box satisfies'):
        select_top_box([[.5, .5, ratio, 1]], [.99], (200, 100), 'rose.')


def test_box_clipping_ties_and_bad_predictions():
    result = select_top_box([[0, .5, 1, 1], [.5, .5, .5, 1]], [.8, .8], (100, 80), 'rose.')
    assert result.box == (0, 0, 50, 80) and result.query_index == 0
    assert result.area_ratio == .5  # Only the in-canvas portion counts.
    with pytest.raises(ValueError, match='no candidate'):
        select_top_box(torch.empty(0, 4), [], (10, 10), 'rose.')
    with pytest.raises(ValueError, match='non-finite'):
        select_top_box([[.5, .5, .2, .2]], [float('nan')], (10, 10), 'rose.')
    with pytest.raises(ValueError, match='No box satisfies'):
        select_top_box([[.5, .5, 0, .2], [.5, .5, .2, .2]], [.9, .8], (10, 10), 'rose.')


def test_decomposition_exactly_reconstructs_grayscale_glyph_and_preserves_canvas(tmp_path):
    values = np.arange(80 * 100, dtype=np.uint8).reshape(80, 100)
    glyph = Image.fromarray(values).convert('RGB')
    detection = Detection((10, 20, 40, 60), .95, (.25, .5, .3, .5), 3, 'rose.')
    sub, surr, mask = split_regions(glyph, detection)
    assert sub.size == surr.size == mask.size == glyph.size
    assert set(np.unique(mask)) == {0, 255}
    assert np.array_equal(np.array(sub).astype(int) + np.array(surr).astype(int), np.array(glyph))
    assert mask.getpixel((10, 20)) == 255 and mask.getpixel((40, 60)) == 0
    record = save_decomposition(tmp_path, glyph, glyph, detection)
    assert len(list(tmp_path.glob('*.png'))) == 5
    assert record['selection'] == 'highest_confidence_after_filtering'
    assert json.loads((tmp_path / 'detection.json').read_text())['detection']['box'] == [10, 20, 40, 60]
    with pytest.raises(ValueError, match='already exist'):
        save_decomposition(tmp_path, glyph, glyph, detection)


def test_polarity_and_transparency():
    dark = Image.new('RGB', (64, 64), 'white')
    ImageDraw.Draw(dark).rectangle((20, 20, 40, 40), fill='black')
    light, detect, polarity = normalize_glyph(dark)
    assert polarity == 'dark' and detect.getpixel((0, 0)) == (255, 255, 255)
    assert light.getpixel((20, 20)) == (255, 255, 255)
    same, _, polarity = normalize_glyph(light)
    assert polarity == 'light' and np.array_equal(light, same)
    transparent = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(transparent).rectangle((20, 20, 40, 40), fill=(0, 0, 0, 255))
    assert np.array_equal(normalize_glyph(transparent)[0], light)
    with pytest.raises(ValueError, match='blank'):
        normalize_glyph(Image.new('RGB', (32, 32), 'white'))


def test_render_font_fits_and_rejects_missing_glyphs(tmp_path):
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    # A local test font keeps this test independent of installed system fonts.
    builder = FontBuilder(1000, isTTF=True)
    order = ['.notdef', 'R', 'o', 's', 'e']
    builder.setupGlyphOrder(order)
    builder.setupCharacterMap({ord(c): c for c in order[1:]})
    glyphs = {}
    for name in order:
        pen = TTGlyphPen(None)
        pen.moveTo((50, 0))
        pen.lineTo((550, 0))
        pen.lineTo((550, 700))
        pen.lineTo((50, 700))
        pen.closePath()
        glyphs[name] = pen.glyph()
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics({name: (600, 50) for name in order})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({'familyName': 'TestGlyph', 'styleName': 'Regular', 'uniqueFontIdentifier': 'TestGlyph',
                            'fullName': 'TestGlyph', 'psName': 'TestGlyph'})
    builder.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    builder.setupPost()
    font_path = tmp_path / 'font.ttf'
    builder.save(str(font_path))
    glyph = render_glyph('Rose', font_path, 256, 128, padding=16)
    _, detect, _ = normalize_glyph(glyph)
    ink = np.asarray(detect.convert('L')) < 128
    ys, xs = np.where(ink)
    assert xs.min() >= 15 and xs.max() < 241 and ys.min() >= 15 and ys.max() < 113
    with pytest.raises(ValueError, match='does not contain'):
        render_glyph('\U0010ffff', font_path)


def test_real_tiny_grounding_dino_forward_and_top_one(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        vocab = [f'[unused{i}]' for i in range(1013)]
        for index, word in {0: '[PAD]', 100: '[UNK]', 101: '[CLS]', 102: '[SEP]',
                            103: 'rose', 104: 'flower', 1012: '.'}.items():
            vocab[index] = word
        vocab_file = tmp_path / 'vocab.txt'
        vocab_file.write_text('\n'.join(vocab))
        tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))
        processor = GroundingDinoProcessor(GroundingDinoImageProcessor(
            size={'shortest_edge': 64, 'longest_edge': 96}), tokenizer)
        config = GroundingDinoConfig(
            backbone_config=SwinConfig(embed_dim=16, depths=[1, 1, 1], num_heads=[1, 2, 4],
                                       window_size=2, out_features=['stage2', 'stage3']),
            text_config=BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1,
                                   num_attention_heads=4, intermediate_size=64),
            d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=4,
            decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64,
            num_feature_levels=3, num_queries=5, max_text_len=32, disable_custom_kernels=True,
        )
        detector = GroundingDinoDetector(GroundingDinoForObjectDetection(config), processor)
        image = Image.new('RGB', (96, 64), 'white')
        ImageDraw.Draw(image).rectangle((20, 20, 60, 50), fill='black')
        # Random-model forward smoke check, not a detector-quality test.
        detection = detector.detect(image, 'Rose Flower', min_score=0, min_area_ratio=0, max_area_ratio=1)
        assert detection.caption == 'rose flower.'
        assert len(detection.box) == 4 and 0 <= detection.score <= 1
        assert 0 <= detection.query_index < 5
    finally:
        torch.set_num_threads(previous)


def test_region_cli_saves_one_box_and_preserves_source(tmp_path, monkeypatch):
    image = Image.new('RGB', (128, 64), 'white')
    ImageDraw.Draw(image).rectangle((20, 10, 90, 50), fill='black')
    source = tmp_path / 'source.png'
    image.save(source)
    original = source.read_bytes()
    result = Detection((10, 5, 70, 60), .9, (.3, .5, .4, .8), 2, 'rose.')
    class FakeDetector:
        model_id = 'test'
        revision = 'test-revision'
        model = SimpleNamespace(device='cpu')
        def detect(self, image, prompt, min_score, min_area_ratio, max_area_ratio):
            assert image.size == (128, 64)
            assert (min_score, min_area_ratio, max_area_ratio) == (.4, .05, .6)
            return result
    monkeypatch.setattr(GroundingDinoDetector, 'from_pretrained', lambda *a, **kw: FakeDetector())
    output = tmp_path / 'region'
    argv = ['--image', str(source), '--subject-prompt', 'rose', '--output', str(output)]
    assert RegionDecomp.main(argv + ['--dry-run']) == 0
    assert not output.exists()
    assert RegionDecomp.main(argv) == 0
    assert source.read_bytes() == original
    assert Image.open(output / 'sub.png').size == (128, 64)
    assert json.loads((output / 'detection.json').read_text())['detection']['score'] == .9


def test_cli_no_eligible_box_writes_nothing(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    image = Image.new('RGB', (64, 64), 'white')
    ImageDraw.Draw(image).rectangle((10, 10, 30, 40), fill='black')
    image.save(source)
    class EmptyDetector:
        def detect(self, image, prompt, min_score, min_area_ratio, max_area_ratio):
            return select_top_box([[.5, .5, 1, 1]], [.99], image.size, prompt,
                                  min_score, min_area_ratio, max_area_ratio)
    monkeypatch.setattr(GroundingDinoDetector, 'from_pretrained', lambda *a, **kw: EmptyDetector())
    output = tmp_path / 'empty'
    with pytest.raises(SystemExit) as exc:
        RegionDecomp.main(['--image', str(source), '--subject-prompt', 'rose', '--output', str(output)])
    assert exc.value.code == 2
    assert not output.exists()
