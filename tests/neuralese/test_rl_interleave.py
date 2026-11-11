import json
from pathlib import Path

import pytest

from natlang_neuralese.common.hashing import sha256_file_hex_cached as sha
from natlang_neuralese.rl import interleave, production
from natlang_neuralese.rl.curriculum import DEFAULT as CURRICULUM
from natlang_neuralese.train.recipe import load_recipe

LOSS = {'schema': 'jolt-plus/1', 'beta_s': 1.0, 'beta_t': 0.5,
        'outcome': {'schema': 'score-centered/1', 'k': 128, 'eps': 1e-6, 'importance': 'tis'}}


def recipe(tmp_path, **parameters):
    cases = tmp_path / 'cases.jsonl'
    cases.write_text(json.dumps({'id': 'task-0', 'split': 'train'}) + '\n')
    stage = {'id': 'interleaved_rl', 'kind': 'native_interleaved_rl', 'requires': ['sft'],
             'inputs': {'rl_task_source': 'rl.tasks'},
             'parameters': {'sft_stage': 'sft', 'rl_rounds_per_window': 2, 'rows_per_round': 16, 'group': 8,
                            'task_curriculum': CURRICULUM, **parameters}}
    value = {'schema': 'natlang.neuralese-training-recipe/1', 'id': 'native-fixture',
             'identity_qualification': 'native_state_runtime_qualified',
             'input_bindings': {'rl.tasks': {'path': str(cases), 'sha256': sha(cases)}},
             'stages': [{'id': 'sft', 'kind': 'native_state_sft', 'requires': [],
                         'parameters': {'refresh_after_updates': 64}}, stage]}
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(value))
    return path, cases


def test_interleaved_stage_requires_its_windowed_sft_stage_and_valid_controls(tmp_path):
    path, _ = recipe(tmp_path)
    assert load_recipe(path)['stages'][1]['kind'] == 'native_interleaved_rl'
    value = json.loads(path.read_text())
    value['stages'][0]['parameters'] = {}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='refresh_after_updates'):
        load_recipe(path)
    for bad, match in (({'rl_rounds_per_window': 0}, 'rl_rounds_per_window'), ({'group': 1}, 'group'),
                       ({'task_curriculum': {**CURRICULUM, 'spread_target': 3}}, 'spread_target'),
                       ({'start_with': 'both'}, 'start_with')):
        path, _ = recipe(tmp_path, **bad)
        with pytest.raises(ValueError, match=match):
            load_recipe(path)
    value = json.loads(recipe(tmp_path)[0].read_text())
    value['stages'][1]['requires'] = []
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='require an earlier native_state_sft'):
        load_recipe(path)


def test_declared_stage_owns_the_controller_task_source_and_budget(tmp_path):
    path, cases = recipe(tmp_path, rl_policy_loss=LOSS, max_model_requests=64)
    config = {'production_recipe': str(path), 'production_stage': 'sft', 'interleaved_rl_stage': 'interleaved_rl',
              'cases': '/old/cases.jsonl', 'rows_per_round': 4, 'group': 4, 'policy_loss': None}
    state = {'round': 5}
    bound, declared = interleave.bind_stage(config, state, tmp_path / 'state.json')
    assert bound['cases'] == str(cases.resolve()) and (bound['rows_per_round'], bound['group']) == (16, 8)
    assert bound['policy_loss'] == LOSS and bound['task_curriculum'] == CURRICULUM
    assert declared['sft_window_updates'] == 64
    assert state['interleave']['rl_rounds_at_last_window'] == 5 and len(state['interleave']['stage_bindings']) == 1
    interleave.bind_stage(config, state, tmp_path / 'state.json')
    assert len(state['interleave']['stage_bindings']) == 1
    assert not interleave.window_due(declared, state)
    state['round'] = 6
    assert not interleave.window_due(declared, state)
    # The controller retunes the cadence live (one SFT window per RL round); the change is a recorded binding.
    _, every_round = interleave.bind_stage({**config, 'rl_rounds_per_window': 1}, state, tmp_path / 'state.json')
    assert interleave.window_due(every_round, state) and every_round['parameters']['rl_rounds_per_window'] == 1
    assert state['interleave']['stage_bindings'][-1]['rl_rounds_per_window'] == 1
    with pytest.raises(ValueError, match='positive integer'):
        interleave.bind_stage({**config, 'rl_rounds_per_window': 0}, state, tmp_path / 'state.json')
    state['round'] = 7
    assert interleave.window_due(declared, state)
    with pytest.raises(ValueError, match='different SFT stage'):
        interleave.bind_stage({**config, 'production_stage': 'other'}, {}, tmp_path / 'state.json')
    with pytest.raises(ValueError, match='mandatory'):
        interleave.bind_stage({**config, 'mandatory_sft_only': True}, {}, tmp_path / 'state.json')
    first = {}
    interleave.bind_stage(json.loads(json.dumps(config)), first, tmp_path / 'state.json')
    path, _ = recipe(tmp_path, start_with='sft')
    started = {}
    _, declared = interleave.bind_stage(config, started, tmp_path / 'state.json')
    assert interleave.window_due(declared, started)


def test_sft_window_adopts_its_earned_boundary_and_stops_when_the_stage_is_complete(tmp_path, monkeypatch):
    from natlang_neuralese.rl import native_supervised
    path, _ = recipe(tmp_path)
    parent = tmp_path / 'parent.pt'
    parent.write_bytes(b'parent')
    out = tmp_path / 'controller'
    out.mkdir()
    config = {'production_recipe': str(path), 'production_stage': 'sft', 'interleaved_rl_stage': 'interleaved_rl',
              'out': str(out), 'checkpoint': str(parent)}
    state = {'round': 2, 'checkpoint': str(parent)}
    state_path = out / 'state.json'
    config, declared = interleave.bind_stage(config, state, state_path)
    calls = []

    def continuation(source, stage, *, checkpoint, out):
        assert stage['id'] == 'sft' and 'interleaved_rl_stage' not in source
        return {'checkpoint': checkpoint, 'out': out, 'mandatory_sft_only': True,
                'resident_data_directory': source['resident_data_directory']}, {}

    def execute(window, window_state, window_state_path, control, *, payload=b'trained'):
        calls.append(window)
        learner = Path(window['out']) / 'learner'
        learner.mkdir(parents=True, exist_ok=True)
        (learner / 'checkpoint.pt').write_bytes(payload)
        (learner / 'report.json').write_text(json.dumps({'continuation_boundary': {
            'kind': 'semantic-refresh', 'reached': True, 'new_optimizer_updates': 64}}))
        window_state.update(phase='awaiting_semantic_refresh',
                            pending_native_checkpoint=str(learner / 'checkpoint.pt'))
        return window_state

    monkeypatch.setattr(native_supervised, 'continuation_config', continuation)
    monkeypatch.setattr(native_supervised, 'execute', execute)
    monkeypatch.setattr(production, 'initial_controller_state', lambda c: {'checkpoint': c['checkpoint']})
    assert interleave.run_sft_window(config, declared, state, state_path, None) == 'adopted'
    window = out / 'sft-window-000000'
    assert state['checkpoint'] == str(window / 'learner/checkpoint.pt')
    assert state['actor_policy_pending_load'] == sha(window / 'learner/checkpoint.pt')
    [record] = state['interleave']['windows']
    assert record['parent']['sha256'] == sha(parent) and record['new_optimizer_updates'] == 64
    assert state['interleave']['rl_rounds_at_last_window'] == 2 and state['phase'] == 'collect'
    assert json.loads((window / 'config.json').read_text())['interleaved_window']['sft_updates'] == 64
    assert not interleave.window_due(declared, state)
    # A window whose boundary leaves the head unchanged retires future SFT windows.
    state['round'] = 4
    monkeypatch.setattr(native_supervised, 'execute',
        lambda w, s, p, c: execute(w, s, p, c, payload=Path(state['checkpoint']).read_bytes()))
    assert interleave.run_sft_window(config, declared, state, state_path, None) == 'complete'
    assert state['interleave']['sft_complete'] and not interleave.window_due(declared, {**state, 'round': 99})


def test_interrupted_window_resumes_from_its_earned_slot(tmp_path, monkeypatch):
    from natlang_neuralese.rl import native_supervised
    from natlang_neuralese.train import checkpoint_policy
    path, _ = recipe(tmp_path, start_with='sft')
    parent = tmp_path / 'parent.pt'
    parent.write_bytes(b'parent')
    out = tmp_path / 'controller'
    out.mkdir()
    config = {'production_recipe': str(path), 'production_stage': 'sft', 'interleaved_rl_stage': 'interleaved_rl',
              'out': str(out), 'checkpoint': str(parent)}
    state = {'round': 0, 'checkpoint': str(parent)}
    config, declared = interleave.bind_stage(config, state, out / 'state.json')
    monkeypatch.setattr(native_supervised, 'continuation_config', lambda source, stage, *, checkpoint, out: (
        {'checkpoint': checkpoint, 'out': out}, {}))
    monkeypatch.setattr(production, 'initial_controller_state', lambda c: {'checkpoint': c['checkpoint']})
    monkeypatch.setattr(checkpoint_policy.CheckpointPolicy, 'load',
                        staticmethod(lambda path: {'training': {'optimizer': {'x': 1}, 'rng': {'y': 1}}}))
    seen = []

    def interrupted(window, window_state, window_state_path, control):
        seen.append(window.get('rolling_sft_resume'))
        learner = Path(window['out']) / 'learner'
        learner.mkdir(parents=True, exist_ok=True)
        if len(seen) == 1:
            (learner / 'checkpoint.pt').write_bytes(b'partial')
            raise RuntimeError('interrupted')
        (learner / 'checkpoint.pt').write_bytes(b'done')
        (learner / 'report.json').write_text(json.dumps({'continuation_boundary': {'new_optimizer_updates': 3}}))
        window_state.update(phase='awaiting_semantic_refresh', pending_native_checkpoint=str(learner / 'checkpoint.pt'))
        return window_state

    monkeypatch.setattr(native_supervised, 'execute', interrupted)
    assert interleave.window_due(declared, state)
    with pytest.raises(RuntimeError):
        interleave.run_sft_window(config, declared, state, out / 'state.json', None)
    assert state['phase'] == 'interleaved_sft_window'
    assert interleave.run_sft_window(config, declared, state, out / 'state.json', None) == 'adopted'
    assert seen[0] is None and seen[1]['previous_checkpoint_sha256'] == sha(parent)
    assert state['interleave']['windows'][0]['rolling_resumes'] == [seen[1]]


def test_production_run_alternates_declared_sft_windows_with_rl_rounds(tmp_path, monkeypatch):
    from natlang_neuralese.rl import native_raw_policy
    monkeypatch.setattr(native_raw_policy, 'prepare_raw_policy', lambda *a, **kw: {'tokenizer': 'bound-parent'})
    path, cases = recipe(tmp_path, start_with='sft', rl_rounds_per_window=1, rows_per_round=1, group=2,
                         task_curriculum={**CURRICULUM, 'min_rows_per_round': 1, 'min_group': 2})
    parent = tmp_path / 'parent.pt'
    parent.write_bytes(b'full state')
    out = tmp_path / 'production'
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'out': str(out), 'checkpoint': str(parent), 'cases': '/unused/old.jsonl',
        'production_recipe': str(path), 'production_stage': 'sft', 'interleaved_rl_stage': 'interleaved_rl',
        'endpoints': ['http://unused'], 'learner_command': ['learn'], 'after_update': ['publish'],
        'native_state': True, 'base': 'unused'}))
    order = []

    def window(config, declared, state, state_path, control):
        order.append(('sft', state['round'], config['cases']))
        state['interleave'].update(sft_windows=state['interleave']['sft_windows'] + 1,
                                   rl_rounds_at_last_window=state['round'])
        state['interleave'].pop('pending_initial_window', None)
        Path(state_path).write_text(json.dumps(state))
        return 'adopted'

    class Stop(Exception):
        pass

    def collect(**kwargs):
        order.append(('rl', kwargs['rows'], kwargs['group']))
        raise Stop

    monkeypatch.setattr(interleave, 'run_sft_window', window)
    monkeypatch.setattr(production, 'collect', collect)
    with pytest.raises(Stop):
        production.run(config, rounds=1)
    assert order == [('sft', 0, str(cases.resolve())), ('rl', 1, 2)]
    state = json.loads((out / 'state.json').read_text())
    assert state['interleave']['stage'] == 'interleaved_rl' and state['interleave']['sft_windows'] == 1
    selection = json.loads((out / 'round-000000-selection.json').read_text())
    assert selection['budget']['group'] == 2 and selection['curriculum']['schema'] == CURRICULUM['schema']
