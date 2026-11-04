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


def test_solo_window_controllers_are_discovered(tmp_path, monkeypatch):
    workspace, cloud = tmp_path / 'ws', tmp_path / 'ext'
    cloud.mkdir()
    runtime = tmp_path / 'runtime-v49'
    runtime.mkdir()
    (runtime / 'runtime-input-closure.json').write_text('{}')
    solo = workspace / 'ops' / 'solo-1104'
    for name, sha in (('out', 'b' * 64), ('w1', 'c' * 64)):
        (solo / name).mkdir(parents=True)
        (solo / name / 'state.json').write_text(json.dumps({'last_learner_progress': {
            'checkpoint': str(solo / name / 'learner' / 'checkpoint.pt'), 'checkpoint_sha256': sha, 'step': 5}}))
    # The first window's control sits beside the run root; later windows keep theirs beside their state.
    (solo / 'config.json').write_text(json.dumps({'actor_environment': {'NATLANG_REPO': str(runtime)}}))
    (solo / 'w1' / 'config.json').write_text(json.dumps({'actor_environment': {'NATLANG_REPO': str(runtime)}}))
    module = load(monkeypatch, workspace, cloud)
    assert [path for path, _ in module.controller_progress()] == [solo / 'out' / 'state.json', solo / 'w1' / 'state.json']
    assert module.lineage_runtime(solo / 'out' / 'state.json') == str(runtime)
    assert module.lineage_runtime(solo / 'w1' / 'state.json') == str(runtime)


def test_unscored_arm_is_reported_unmatched_not_an_error():
    from natlang_neuralese.train.promotion import compare
    cases = [0, 8]
    packet = {'path': '/p', 'sha256': 'f' * 64}
    declaration = {'schema': 'natlang.behavior-promotion-declaration/1', 'packet': packet, 'cases': cases,
                   'compare_to': 'parent', 'min_delta': 1, 'max_case_regressions': None, 'confirmation': None}

    def arm(sha, successes, outcomes):
        return ({'status': 'finished', 'training_admission': False, 'contract': {'checkpoint_sha256': sha, 'step': 1},
                 'summary': {'eligible': successes is not None, 'schema': 'natlang.in-run-task-eval/1',
                             'task_successes': successes, 'cases': 2, 'packet_sha256': packet['sha256'],
                             'case_indices': cases, 'outcome_counts': {}}},
                {'packet_sha256': packet['sha256'], 'case_indices': cases, 'task_successes': successes, 'cases': 2,
                 'cases_detail': [{'id': f'c{i}', 'outcome': o} for i, o in enumerate(outcomes)]})
    unscored, unscored_detail = arm('a' * 64, None, ['unresolved', 'semantic_mismatch'])
    scored, scored_detail = arm('b' * 64, 1, ['task_success', 'semantic_mismatch'])
    receipt = compare(declaration, phase='development', candidate=unscored, reference=scored,
                      candidate_detail=unscored_detail, reference_detail=scored_detail)
    assert receipt['matched'] is False and receipt['promoted'] is False
    assert any('not a finished, eligible, scored run' in reason for reason in receipt['unmatched_reasons'])
