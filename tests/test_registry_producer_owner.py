"""Producer writes route to shared ownership without changing pinned read/provenance roots."""
import importlib.util
import json
from pathlib import Path
import pytest
from natlang_neuralese.common import data_registry as dr

ROOT = Path(__file__).resolve().parents[1]


def script(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem + '_owner_test', ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('relative,args', [
    ('scripts/rl_tasks/register_authored_held.py', ('missing', 'stamp', False)),
    ('scripts/rl_tasks/register_source_held.py', ('missing', 'prefix', 'stamp', 'label', False)),
])
def test_explicit_held_packet_pin_refused_before_any_output(relative, args, tmp_path, monkeypatch):
    module = script(relative)
    pin = tmp_path / 'pin'
    pin.mkdir()
    marker = pin / '.natlang-pin.json'
    marker.write_text(json.dumps({'source': str(tmp_path / 'owner')}))
    monkeypatch.setattr(module, 'REPO', pin)
    with pytest.raises(ValueError, match='pin'):
        module.register(*args)
    assert list(pin.iterdir()) == [marker]


def test_render_entry_paths_use_owner_while_code_and_source_snapshot_stay_pinned(tmp_path, monkeypatch):
    module = script('scripts/neuralese_data/register_native_turn_render.py')
    owner, pin = tmp_path / 'owner', tmp_path / 'pin'
    monkeypatch.setattr(module, 'ROOT', pin)
    monkeypatch.setattr(dr, 'registry_owner_repo', lambda: owner)
    args = module.parse(['--id', 'render-id', '--render-dir', str(owner / 'renders/current'),
                         '--source-corpus', 'source-id'])
    assert args.path == 'renders/current'
    assert module.ROOT == pin


def test_artifact_registration_refuses_pin_before_copy_or_registry_changes(tmp_path):
    from natlang_neuralese import artifacts
    pin = tmp_path / 'pin'
    pin.mkdir()
    marker = pin / '.natlang-pin.json'
    marker.write_text('{}')
    with pytest.raises(ValueError, match='pin'):
        artifacts.register(pin, {'id': 'artifact-id'}, {'file.nz': tmp_path / 'missing'})
    assert list(pin.iterdir()) == [marker]
