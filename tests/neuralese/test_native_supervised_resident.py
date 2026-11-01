"""Direct SFT owns a real resident plan before model allocation."""
import json
from pathlib import Path
import pytest
from natlang_neuralese.rl.native_supervised_resident import ensure_plan
from natlang_neuralese.train import native_resident as resident
from test_native_supervised_boundary import fixture


def test_direct_launch_creates_usable_authenticated_plan(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    directory = tmp_path / 'direct' / 'resident-data'
    config = {**f.config, 'resident_data_directory': str(directory)}
    ref = ensure_plan(config)
    plan = json.loads(Path(ref['path']).read_text())
    assert plan['code'] == resident.code_inventory(Path(resident.__file__).parents[1])
    manager = resident.ResidentAdoption(directory, stage_id='sft',
        recipe_binding=f.manager.recipe_binding, train=f.manager.train, held=f.manager.held)
    assert manager.plan == plan
    before = Path(ref['path']).read_bytes()
    assert ensure_plan(config) == ref
    assert Path(ref['path']).read_bytes() == before


def test_changed_launch_source_rejected_before_plan_creation(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    directory = tmp_path / 'direct' / 'resident-data'
    config = {**f.config, 'resident_data_directory': str(directory),
              'seed_native_graphs': str(f.paths['new'])}
    with pytest.raises(ValueError, match='launch source roles'):
        ensure_plan(config)
    assert not (directory.parent / 'recipe-plan.json').exists()


def test_existing_authority_cannot_be_replaced_with_new_runtime(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    path = f.directory.parent / 'recipe-plan.json'
    old = path.read_bytes()
    package = tmp_path / 'changed-package'
    package.mkdir()
    (package / 'changed.py').write_text('changed = True\n')
    with pytest.raises(ValueError, match='distinct drained authority directory'):
        ensure_plan(f.config, package=package)
    assert path.read_bytes() == old
