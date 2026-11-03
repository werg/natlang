"""The Pop dev16 watcher discovers cloud SFT-window controllers (DGX hub copies restored from verified exports)."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(monkeypatch, workspace, cloud):
    monkeypatch.setenv('NATLANG_LFM3_WORKSPACE', str(workspace))
    monkeypatch.setenv('NATLANG_CLOUD_RUNS', str(cloud))
    sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
    spec = importlib.util.spec_from_file_location('pop_behavior_eval_t', ROOT / 'scripts/neuralese_data/pop_behavior_eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cloud_window_controller_is_discovered_with_its_own_runtime(tmp_path, monkeypatch):
    workspace, cloud = tmp_path / 'ws', tmp_path / 'ext'
    runtime = tmp_path / 'runtime-v44'
    runtime.mkdir()
    (runtime / 'runtime-input-closure.json').write_text('{}')
    window = cloud / 'natlang-lfm3-cloud-x' / 'corrected-mix' / 'w000000'
    window.mkdir(parents=True)
    workspace.mkdir()
    (window / 'config.json').write_text(json.dumps({'actor_environment': {'NATLANG_REPO': str(runtime)}}))
    (window / 'state.json').write_text(json.dumps({'last_learner_progress': {
        'checkpoint': str(window / 'learner' / 'checkpoint.pt'), 'checkpoint_sha256': 'a' * 64, 'step': 64}}))
    module = load(monkeypatch, workspace, cloud)
    found = list(module.controller_progress())
    assert [path for path, _ in found] == [window / 'state.json']
    assert module.lineage_runtime(window / 'state.json') == str(runtime)
