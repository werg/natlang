import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('build_idiom_mixture', Path(__file__).resolve().parents[2] / 'scripts/neuralese_data/build_idiom_mixture.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_prompt_selection_preserves_single_view_and_selects_one_paired_view():
    for header in (False, True):
        assert mod.keep_prompt_view('teacher-turn-full', 'p', {'p'}, set(), header)
        assert mod.keep_prompt_view('teacher-turn', 'p', set(), {'p'}, header)
        assert mod.keep_prompt_view('teacher-turn', 'p', {'p'}, {'p'}, header) == header
        assert mod.keep_prompt_view('teacher-turn-full', 'p', {'p'}, {'p'}, header) != header


def test_mixture_content_identity_retains_decision_targets_and_state_dependencies():
    import copy
    graph = {'tokenizer_fingerprint': 'fixture', 'nodes': [{'name': 'n', 'token_ids': [1, 2], 'target_weights': [0, 0]}],
             'decisions': [{'name': 'd', 'options': ['n'], 'label': 0}], 'external_states': {'s': {'id': 'one'}}}
    changed_label = copy.deepcopy(graph)
    changed_label['decisions'][0]['label'] = 1
    changed_state = copy.deepcopy(graph)
    changed_state['external_states']['s']['id'] = 'two'
    assert mod.content_key(graph) != mod.content_key(changed_label)
    assert mod.content_key(graph) != mod.content_key(changed_state)
    assert mod.content_key(graph) == mod.content_key(copy.deepcopy(graph))


def put(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_held_closes_through_ordinary_communication_and_prior(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'REPO', tmp_path)
    ordinary=tmp_path/'ordinary.jsonl'; comm=tmp_path/'comm.jsonl'
    put(ordinary,[{'split':'train','source_groups':['b','c']}, {'split':'train','source_groups':['x']}])
    put(comm,[{'split':'train','source_groups':['a','b']}, {'split':'train','source_groups':['x','y']}])
    monkeypatch.setattr(mod, 'graph_files', lambda render, split: [('program-header', ordinary)])
    config={'renders':['fixture'],'communication':[{'kind':'native-graphs','path':str(comm)}]}
    policy={'held_carve':{'fraction':1.0,'salt':'s','max_rows':10}}
    carve, held = mod.carve_groups(config,policy,held_groups={'a'},prior_components=[('c','d')])
    assert held == {'a','b','c','d'}
    assert carve == {'x','y'}
    assert mod.communication_is_held({'source_groups':['y']}, [{}], held, carve)
    assert not mod.communication_is_held({'source_groups':['z']}, [{}], held, carve)
    with pytest.raises(ValueError,match='prior train sources are connected'):
        mod.carve_groups(config,policy,trained_groups={'d'},held_groups={'a'},prior_components=[('c','d')])


def test_held_closure_applies_without_carve(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'REPO', tmp_path)
    comm=tmp_path/'comm.jsonl';put(comm,[{'split':'train','source_groups':['a','b']}])
    carve, held = mod.carve_groups({'renders':[],'communication':[{'kind':'native-graphs','path':str(comm)}]},
                                   {},held_groups={'a'})
    assert carve == set() and held == {'a','b'}


def test_held_closure_includes_full_teacher_and_bounded_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'REPO', tmp_path)
    full = tmp_path / 'full.jsonl'; bounded = tmp_path / 'bounded.jsonl'
    put(full, [{'split': 'train', 'source_groups': ['held', 'bridge']}])
    put(bounded, [{'split': 'train', 'source_groups': ['bridge', 'end']}])
    config = {'renders': [], 'teacher_graphs_full': [str(full)],
              'bounded_graphs': [{'path': str(bounded)}]}
    _, held = mod.carve_groups(config, {}, held_groups={'held'})
    assert held == {'held', 'bridge', 'end'}


def test_resumed_connectivity_reapplies_current_held_and_checks_source(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'REPO', tmp_path)
    rows = tmp_path / 'rows.jsonl'; put(rows, [{'source_groups': ['a', 'b']}])
    monkeypatch.setattr(mod, 'graph_files', lambda render, split: [('program-header', rows)])
    config = {'renders': ['one']}; checkpoint = tmp_path / 'scan.pkl'
    _, first = mod.carve_groups(config, {}, held_groups={'a'}, checkpoint_path=checkpoint)
    _, second = mod.carve_groups(config, {}, held_groups={'unrelated'}, checkpoint_path=checkpoint)
    assert first == {'a', 'b'} and second == {'unrelated'}
    put(rows, [{'source_groups': ['a', 'c']}])
    with pytest.raises(ValueError, match='inputs or protection changed'):
        mod.carve_groups(config, {}, held_groups={'a'}, checkpoint_path=checkpoint)


def cand(i, family, cluster, tier=2, tokens=10):
    return {'id': f'r{i}', 'tier': tier, 'target_tokens': tokens,
            'features': {'family': family, 'near_duplicate_cluster': cluster}}


def test_balance_caps_clusters_then_families_round_robin():
    import collections
    rows = [cand(i, 'big', f'world{i % 4}') for i in range(400)] + [cand(1000 + i, 'small', f's{i}') for i in range(20)]
    policy = {'balance': {'cluster_rows': 30, 'family_share': 0.0, 'family_floor_rows': 40}}
    dropped = collections.Counter()
    out = mod.balance(rows, policy, dropped)
    big = [c for c in out if c['features']['family'] == 'big']
    assert len(big) == 40 and len([c for c in out if c['features']['family'] == 'small']) == 20
    assert collections.Counter(c['features']['near_duplicate_cluster'] for c in big) == {f'world{i}': 10 for i in range(4)}
    assert dropped['balance: near-duplicate cluster cap'] == 400 - 120
    assert dropped['balance: family cap (big)'] == 80


def test_balance_exempt_family_and_bounded_tiers():
    import collections
    rows = [cand(i, 'trajectory_continuation_research', f'c{i}') for i in range(100)] + \
           [cand(200 + i, 'acq', f'a{i}', tier=6) for i in range(50)]
    policy = {'balance': {'cluster_rows': 1, 'family_share': 0.0, 'family_floor_rows': 5,
                          'exempt_family_pattern': '^trajectory_continuation_'},
              'bounded_mixes': {'external-agent': {'tier': 6, 'share': 0.1}}}
    out = mod.balance(rows, policy, collections.Counter())
    assert len(out) == 150


def test_bounded_sample_share_of_natlang_tokens():
    import collections
    rows = [cand(i, 'n', f'c{i}', tier=2, tokens=100) for i in range(10)]
    rows += [{**cand(100 + i, f'D{i % 2}', f'a{i}', tier=6, tokens=10), 'mix': 'external-agent'} for i in range(40)]
    policy = {'bounded_mixes': {'external-agent': {'tier': 6, 'share': 0.1}}}
    dropped = collections.Counter()
    out = mod.bounded_sample(rows, policy, dropped)
    chosen = [c for c in out if c['tier'] == 6]
    assert sum(c['target_tokens'] for c in chosen) == 100
    assert collections.Counter(c['features']['family'] for c in chosen) == {'D0': 5, 'D1': 5}


def test_component_carve_draws_once_per_component(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'REPO', tmp_path)
    rows = tmp_path / 'rows.jsonl'
    many = [f'g{i:03d}' for i in range(200)]
    put(rows, [{'split': 'train', 'source_groups': many}, {'split': 'train', 'source_groups': ['solo']}])
    monkeypatch.setattr(mod, 'graph_files', lambda render, split: [('program-header', rows)])
    config = {'renders': ['fixture']}
    any_group = {'held_carve': {'fraction': 0.05, 'salt': 's', 'max_rows': 10}}
    carve, _ = mod.carve_groups(config, any_group)
    assert set(many) <= carve  # one of 200 groups falls under 5%: the whole folder component is held
    per_component = {'held_carve': {'fraction': 0.05, 'salt': 's', 'max_rows': 10, 'unit': 'component'}}
    carve, _ = mod.carve_groups(config, per_component)
    assert bool(set(many) & carve) == mod.carved([min(many)], per_component['held_carve'])


def test_header_view_split_is_deterministic():
    share = sum(mod.header_view(f'd{i}', 0.2) for i in range(5000)) / 5000
    assert 0.17 < share < 0.23 and mod.header_view('x', 0.2) == mod.header_view('x', 0.2)


def test_held_identity_retains_colliding_views_and_world_custody():
    import copy
    first = {'id': 'same-source-id', 'split': 'test', 'source_groups': ['world-a'],
             'tokenizer_fingerprint': 'fixture',
             'nodes': [{'name': 'n', 'token_ids': [1, 2], 'target_weights': [0, 1]}]}
    different_view = copy.deepcopy(first)
    different_view['nodes'][0]['token_ids'] = [3, 2]
    different_world = copy.deepcopy(first)
    different_world['source_groups'] = ['world-b']
    assert mod.content_key(first) == mod.content_key(different_world)
    seen = set()
    rows = [mod.held_graph_identity(row, seen) for row in (first, different_view, different_world)]
    assert len({row['id'] for row in rows}) == 3
    assert set().union(*(set(row['source_groups']) for row in rows)) == {'world-a', 'world-b'}
    assert all(row['held_identity']['original_id'] == 'same-source-id' for row in rows)
    assert all('same-source-id' in row['source_record_ids'] for row in rows)
    assert mod.held_graph_identity(copy.deepcopy(first), seen) is None
    assert first['id'] == 'same-source-id' and 'held_identity' not in first


def test_source_identity_custody_distinguishes_excluded_trajectory_and_recovery_context():
    custody = {'hard_ids': {'eval:held'}, 'context_only_ids': {'decision:failed'},
               'excluded_trajectories': {'teacher-program:bad'}}
    reason = mod.source_custody_reason
    assert reason({'id': 'x', 'conversion': {'source_row_id': 'teacher-program:bad:decision:1'}}, custody)
    assert reason({'id': 'x', 'source_ids': ['eval:held']}, custody)
    assert reason({'id': 'x', 'conversion': {'source_row_id': 'decision:failed'}}, custody)
    assert reason({'id': 'x', 'source_ids': ['decision:failed'],
                   'conversion': {'source_row_id': 'decision:good'}}, custody) is None
