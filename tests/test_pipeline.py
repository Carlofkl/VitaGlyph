import json
from pathlib import Path

import pytest
from PIL import Image
import VitaGlyph


@pytest.mark.parametrize('method', ['depth', 'sds'])
def test_three_stage_routing_resume_and_modified_artifact(tmp_path, monkeypatch, method):
    source = tmp_path / 'glyph.png'
    glyph = Image.new('RGB', (64, 64), 'white')
    glyph.paste('black', (20, 10, 40, 50))
    glyph.save(source)
    output = tmp_path / 'pipeline'
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        target = Path(command[command.index('--output') + 1])
        script = Path(command[1]).name
        if script == 'RegionDecomp.py':
            names = ['raw_image.png', 'sub.png', 'surr.png', 'mask.png', 'pred.png', 'detection.json']
        elif script == 'SemTypo.py':
            names = ['sub_0.png', 'surr.png', 'mask.png', 'depth.png', 'preview.png', 'semtypo.json']
        elif script == 'SDSDeform.py':
            names = ['sub_0.png', 'surr.png', 'mask.png', 'preview.png', 'displacement.npy', 'sds.json']
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (64, 64)).save(target)
            target.with_suffix('.json').write_text('{}')
            return
        target.mkdir(parents=True)
        for name in names:
            if name.endswith('.png'):
                Image.new('RGB', (64, 64)).save(target / name)
            else:
                (target / name).write_text('{}')
    monkeypatch.setattr(VitaGlyph, 'execute', run)
    args = ['--image', str(source), '--subject-prompt', 'rose', '--surrounding-prompt', 'leaves',
            '--output', str(output), '--resolution', '64', '--device', 'cpu',
            '--generation-seed', '44', '--positive-prompt', 'botanical watercolor',
            '--gamma', '1', '--no-attention-control', '--deformation', method]
    assert VitaGlyph.main(args + ['--dry-run']) == 0
    assert not output.exists() and not calls
    assert VitaGlyph.main(args) == 0
    assert [Path(c[1]).name for c in calls] == [
        'RegionDecomp.py', 'SemTypo.py' if method == 'depth' else 'SDSDeform.py', 'ConComGen.py']
    record = json.loads((output / 'pipeline.json').read_text())
    assert record['knowledge_acquisition'] is False and (output / 'artistic.png').exists()
    assert '--min-score' in calls[0] and calls[0][calls[0].index('--min-score') + 1] == '0.4'
    assert calls[1][calls[1].index('--seed') + 1] == '42'
    assert calls[2][calls[2].index('--seed') + 1] == '44'
    assert calls[2][calls[2].index('--positive-prompt') + 1] == 'botanical watercolor'
    assert '--no-attention-control' in calls[2] and '--no-attention-control' not in calls[1]
    assert calls[2][calls[2].index('--gamma') + 1] == '1.0'
    assert VitaGlyph.main(args + ['--resume']) == 0
    assert len(calls) == 3
    (output / 'controls/sub_0.png').write_bytes(b'changed')
    with pytest.raises(SystemExit):
        VitaGlyph.main(args + ['--resume'])
    assert len(calls) == 3


def test_stage_failure_stops_before_next_stage(tmp_path, monkeypatch):
    source = tmp_path / 'glyph.png'
    glyph = Image.new('RGB', (64, 64), 'white')
    glyph.paste('black', (20, 10, 40, 50))
    glyph.save(source)
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        raise VitaGlyph.subprocess.CalledProcessError(2, command)
    monkeypatch.setattr(VitaGlyph, 'execute', fail)
    with pytest.raises(SystemExit):
        VitaGlyph.main(['--image', str(source), '--subject-prompt', 'rose', '--surrounding-prompt', 'leaves',
                       '--output', str(tmp_path / 'out'), '--device', 'cpu'])
    assert len(calls) == 1
    assert not (tmp_path / 'out/pipeline.json').exists()
