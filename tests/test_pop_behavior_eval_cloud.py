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


def test_rl_actor_marker_holds_evaluations_and_pending_eval_is_read_only(tmp_path, monkeypatch):
    """A Pop RL rollout actor owns the GPU while its marker is fresh; a stale marker is a crashed collection."""
    import time
    workspace = tmp_path / 'ws'
    workspace.mkdir()
    monkeypatch.setenv('NATLANG_POP_EVAL_ROOT', str(tmp_path / 'eval'))
    module = load(monkeypatch, workspace, tmp_path / 'ext')
    root = tmp_path / 'eval'
    root.mkdir()
    assert module.rl_actor_active() is False
    (root / 'rl-actor-active.json').write_text(json.dumps({'started': time.time()}))
    assert module.rl_actor_active() is True
    slept = []
    monkeypatch.setattr(module.time, 'sleep', lambda s: (slept.append(s), (root / 'rl-actor-active.json').unlink()))
    module.wait_for_rl_actor()          # waits once, until the teardown hook removes the marker
    assert slept == [60]
    (root / 'rl-actor-active.json').write_text(json.dumps({'started': time.time() - module.RL_ACTOR_MAX_SECONDS - 1}))
    assert module.rl_actor_active() is False
    state = {'candidates': {}, 'controllers': {}, 'parent': {'sha256': 'p' * 64}}
    assert module.eval_pending(state) is False
    state['candidates']['x'] = {'status': 'running'}
    assert module.eval_pending(state) is True


def _watcher(tmp_path, monkeypatch):
    workspace = tmp_path / 'ws'
    workspace.mkdir()
    monkeypatch.setenv('NATLANG_POP_EVAL_ROOT', str(tmp_path / 'eval'))
    module = load(monkeypatch, workspace, tmp_path / 'ext')
    (tmp_path / 'eval' / 'requests').mkdir(parents=True)
    state = {'candidates': {}, 'controllers': {}, 'parent': {'sha256': 'p' * 64}, 'parent_evals': {}}
    return module, state, tmp_path / 'eval' / 'requests'


def _request(requests, name, sha, runtime, **extra):
    (requests / f'{name}.json').write_text(json.dumps({'checkpoint': '/ck.pt', 'sha256': sha, 'runtime': runtime,
                                                       'label': 'x', 'step': 7, **extra}))


def _events(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]


def test_duplicate_request_is_logged_but_an_explicit_attempt_or_new_runtime_runs_fresh(tmp_path, monkeypatch, capsys):
    module, state, requests = _watcher(tmp_path, monkeypatch)
    sha = 'a' * 64
    state['candidates'][sha] = {'sha256': sha, 'status': 'done', 'runtime': '/old-runtime', 'path': '/ck.pt'}
    # The same checkpoint under the same runtime and no attempt label is an idempotent duplicate, with a disposition.
    _request(requests, '1', sha, '/old-runtime')
    module.discover(state)
    assert list(state['candidates']) == [sha]
    duplicate = [e for e in _events(capsys) if e.get('event') == 'request-duplicate']
    assert len(duplicate) == 1 and 'already done' in duplicate[0]['disposition'] and '--attempt' in duplicate[0]['hint']
    # The review's case: same SHA, another runtime, a new repeat label.
    _request(requests, '2', sha, '/new-runtime', attempt='repeat 2')
    module.discover(state)
    fresh = state['candidates'][f'{sha}#repeat-2']
    assert fresh['status'] == 'queued' and fresh['runtime'] == '/new-runtime' and fresh['attempt'] == 'repeat-2'
    assert state['candidates'][sha]['status'] == 'done'
    assert any(e.get('event') == 'requested' and e['attempt'] == 'repeat-2' for e in _events(capsys))
    # A new attempt label under the SAME runtime is also a fresh run; repeating that label is a duplicate again.
    _request(requests, '3', sha, '/old-runtime', attempt='again')
    module.discover(state)
    assert state['candidates'][f'{sha}#again']['status'] == 'queued'
    _request(requests, '4', sha, '/old-runtime', attempt='again')
    module.discover(state)
    assert len(state['candidates']) == 3
    assert any(e.get('event') == 'request-duplicate' for e in _events(capsys))
    # Another runtime without any label is a different measurement as well (keyed by the runtime's name).
    _request(requests, '5', sha, '/other-runtime')
    module.discover(state)
    assert state['candidates'][f'{sha}#other-runtime']['status'] == 'queued'
    # A failed candidate is requeued under its own key, as before.
    state['candidates'][sha]['status'] = 'failed'
    _request(requests, '6', sha, '/old-runtime')
    module.discover(state)
    assert state['candidates'][sha]['status'] == 'queued'


def code(pop='a' * 40, dgx='d' * 40):
    return {'dgx': {'git_commit': dgx, 'dirty': False}, 'pop': {'git_commit': pop, 'dirty': False}}


def test_execution_identity_binds_driver_and_image_and_records_the_code_commits(tmp_path, monkeypatch):
    module, _, _ = _watcher(tmp_path, monkeypatch)
    monkeypatch.setattr(module, 'pop_image_id', lambda: 'sha256:' + '1' * 64)
    driver = tmp_path / 'driver.py'
    driver.write_text('print(1)')
    monkeypatch.setattr(module, 'DRIVER', driver)
    base = module.execution_identity(code())
    assert module.execution_identity(code()) is base
    # the code that ran is provenance, not part of the identity: another Pop commit keeps the same identity (and so pairs)
    newer = module.execution_identity(code(pop='b' * 40))
    assert newer['identity_sha256'] == base['identity_sha256'] and newer['code']['pop']['git_commit'] == 'b' * 40
    driver.write_text('print(2)')
    changed_driver = module.execution_identity(code())
    assert changed_driver['identity_sha256'] != base['identity_sha256']
    monkeypatch.setattr(module, 'pop_image_id', lambda: 'sha256:' + '2' * 64)
    assert module.execution_identity(code())['identity_sha256'] == changed_driver['identity_sha256']  # cached
    module._IDENTITY_CACHE.clear()
    assert module.execution_identity(code())['identity_sha256'] != changed_driver['identity_sha256']  # image
    identity = module.execution_identity(code())
    launch = {'code_commit': 'a' * 40, 'driver_sha256': identity['fields']['driver_sha256'],
              'task_spec_sha256': module.SPEC_SHA256, 'image_id': 'sha256:' + '2' * 64,
              'execution_identity_sha256': identity['identity_sha256']}
    assert module.verify_launch(identity, launch) == []
    problems = module.verify_launch(identity, {**launch, 'driver_sha256': 'x', 'image_id': 'sha256:other'})
    assert len(problems) == 2 and problems[0].startswith('driver_sha256') and problems[1].startswith('image_id')
    assert any(p.startswith('execution_identity_sha256') for p in module.verify_launch(identity, {**launch, 'execution_identity_sha256': None}))


def _receipt(sha, step, successes, python='p' * 64):
    return {'status': 'finished', 'training_admission': False, 'contract': {'checkpoint_sha256': sha, 'step': step, 'spec_sha256': 's' * 64},
            'actor_info': {'prompt_rendering_identity_sha256': 'r' * 64, 'serving_config_identity_sha256': 'v' * 64,
                           'rl_policy_binding': {'python_runtime': {'sha256': python}, 'serving_precision': {'p': 1},
                                                 'history_reasoning': {'h': 1}, 'native_identity': {'tokenizer_fingerprint': 't'}}},
            'summary': {'eligible': True, 'schema': 'natlang.in-run-task-eval/1', 'task_successes': successes, 'cases': 2,
                        'packet_sha256': 'f' * 64, 'case_indices': [0, 8], 'outcome_counts': {}, 'prompt_rendering': 'full'}}


def test_promotion_requires_the_same_verified_execution_identity():
    from natlang_neuralese.train.promotion import compare
    declaration = {'schema': 'natlang.behavior-promotion-declaration/1', 'packet': {'path': '/p', 'sha256': 'f' * 64},
                   'cases': [0, 8], 'compare_to': 'parent', 'min_delta': 1, 'max_case_regressions': None, 'confirmation': None}
    run = lambda **kw: compare(declaration, phase='development', candidate=_receipt('a' * 64, 2, 2),
                               reference=_receipt('b' * 64, 1, 1), **kw)
    execution = lambda identity, verified=True: {'identity_sha256': identity, 'verified': verified,
                                                 'fields': {'driver_sha256': identity, 'image_id': 'i'}}
    # The receipts alone are equal on every tracked field: matched, exactly as before identities existed.
    assert run()['promoted'] is True
    same = run(candidate_execution=execution('x' * 64), reference_execution=execution('x' * 64))
    assert same['promoted'] is True and same['execution_identity_sha256'] == 'x' * 64
    other = run(candidate_execution=execution('x' * 64), reference_execution=execution('y' * 64))
    assert other['matched'] is False and other['promoted'] is False
    assert 'identical execution environment: driver_sha256' in other['unmatched_reasons'][0]
    unverified = run(candidate_execution=execution('x' * 64, False), reference_execution=execution('x' * 64))
    assert unverified['matched'] is False and 'unverified' in unverified['unmatched_reasons'][0]
    one = run(candidate_execution=execution('x' * 64))
    assert one['matched'] is False and 'only one arm' in one['unmatched_reasons'][0]


def test_collect_checks_the_launch_record_against_the_requested_identity(tmp_path, monkeypatch):
    module, _, _ = _watcher(tmp_path, monkeypatch)
    monkeypatch.setattr(module, 'pop_image_id', lambda: 'sha256:' + '1' * 64)
    identity = module.execution_identity(code())
    good = {'container_id': 'k' * 64, 'code_commit': 'a' * 40, 'driver_sha256': identity['fields']['driver_sha256'],
            'task_spec_sha256': module.SPEC_SHA256, 'image_id': 'sha256:' + '1' * 64, 'arm': 'raw',
            'execution_identity_sha256': identity['identity_sha256']}
    completed = {'summary': {'task_successes': 3, 'cases': 16},
                 'actor_info': {'rl_policy_binding': {'code': {'git_commit': 'a' * 40}}}}
    served = {}
    def fake_run(argv, **kwargs):
        destination = Path(argv[-1])
        destination.write_text(json.dumps(served['launch'] if destination.name == 'launch.json' else completed))
    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    monkeypatch.setattr(module, 'pop_python', lambda *a, **k: {'cases_detail': []})
    pruned = []
    monkeypatch.setattr(module, 'remove_pop_work', pruned.extend)
    prep = {'source': {'sha256': 'a' * 64}, 'step': 5}
    served['launch'] = good
    result = module.collect('good', '/work', prep, identity)
    assert '/work/store' in pruned and not any('evaluation' in path for path in pruned)   # the done marker stays
    assert result['execution']['verified'] is True and result['execution']['launch']['container_id'] == 'k' * 64
    assert result['execution']['served_code_commit'] == 'a' * 40
    served['launch'] = {**good, 'driver_sha256': 'bad'}
    result = module.collect('bad', '/work', prep, identity)
    assert result['execution']['verified'] is False and 'driver_sha256' in result['execution']['problems'][0]


def test_pop_work_is_removed_inside_a_root_container_scoped_to_the_eval_root(tmp_path, monkeypatch):
    module, _, _ = _watcher(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(module, 'ssh', lambda command, **kw: calls.append(command))
    module.remove_pop_work([])
    assert calls == []
    module.remove_pop_work([module.POP_EVAL_ROOT + '/rl-actor-1-abc', module.POP_EVAL_ROOT + '/raw-2-def/store'])
    assert '-v ' + module.POP_EVAL_ROOT + ':/r' in calls[0] and calls[0].endswith('-- /r/rl-actor-1-abc /r/raw-2-def/store')
    import pytest
    with pytest.raises(ValueError):
        module.remove_pop_work(['/elsewhere'])


def test_model_facing_request_is_its_own_diagnostic_measurement(tmp_path, monkeypatch):
    module, state, requests = _watcher(tmp_path, monkeypatch)
    sha = 'a' * 64
    state['candidates'][sha] = {'sha256': sha, 'status': 'done', 'runtime': '/rt', 'path': '/ck.pt'}
    # The same checkpoint and runtime with a model-facing variant is never a duplicate of the regular evaluation, and is
    # never keyed as the checkpoint's own entry (a later regular discovery of that SHA is not suppressed by it).
    _request(requests, '1', sha, '/rt', model_facing='none')
    _request(requests, '2', sha, '/rt', model_facing='default')
    _request(requests, '3', sha, '/rt', model_facing='none')   # an idempotent repeat of request 1
    module.discover(state)
    assert sorted(state['candidates']) == sorted([sha, f'{sha}#mf-none', f'{sha}#mf-default'])
    arm = state['candidates'][f'{sha}#mf-none']
    assert arm['model_facing'] == 'none' and arm['status'] == 'queued' and arm['ema'] is False
    assert module.model_facing_value('effect-receipt+read-file-binding') == 'effect-receipt,read-file-binding'
    # The variant is part of the execution identity (so never paired with, or cached as, a regular evaluation) and is
    # checked against the launch record.
    monkeypatch.setattr(module, 'pop_image_id', lambda: 'sha256:' + '1' * 64)
    regular = module.execution_identity(code())
    none = module.execution_identity(code(), 'none')
    assert none['identity_sha256'] != regular['identity_sha256'] != module.execution_identity(code(), 'default')['identity_sha256']
    launch = {'driver_sha256': none['fields']['driver_sha256'], 'task_spec_sha256': module.SPEC_SHA256,
              'image_id': 'sha256:' + '1' * 64, 'execution_identity_sha256': none['identity_sha256'], 'model_facing': 'none'}
    assert module.verify_launch(none, launch) == []
    assert any(p.startswith('model_facing') for p in module.verify_launch(none, {**launch, 'model_facing': None}))
