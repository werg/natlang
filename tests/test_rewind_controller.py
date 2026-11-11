"""rewind_controller on a synthetic interleaved controller: three RL rounds and two SFT windows (rl_rounds_per_window 2:
rounds 0-1, window 0, rounds 2-3, window 1), a collected-but-untrained round 4."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training/neuralese'))
SPEC = importlib.util.spec_from_file_location('rewind_controller', ROOT / 'scripts/neuralese_data/rewind_controller.py')
rw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rw)

HEX = {name: (name * 64)[:64] for name in 'abcdefgh'}   # policies: a=start, b=after r0, c=after r1, d=window0, e=after r2, f=after r3, g=window1
# round r: collected from CAUSE[r], its update produces RESULT[r]
CAUSE = ['a', 'b', 'd', 'e', 'g']
RESULT = ['b', 'c', 'e', 'f']
TASKS = {r: [f't{r}-{i}' for i in range(3)] + ['shared'] for r in range(5)}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def group(task, mean):
    return {'task_id': task, 'scored': 8, 'mean': mean, 'spread': 0 < mean < 1, 'outcomes': ['task_success'] * 8, 'route': 'x',
            'efficiency_spread': False, 'reporting_spread': False}


def build(tmp_path, monkeypatch):
    out = tmp_path / 'w'
    out.mkdir()
    history, observed, yields, counts, accounted = {}, {}, [], {}, []
    from natlang_neuralese.rl.curriculum import record_visit
    for r in range(5):
        groups = [group(t, 0.5 if t == 'shared' else 0.0) for t in TASKS[r]]
        write(out / f'round-{r:06d}/score.json', {'groups': groups, 'scored': 32, 'round_marker': r})
        write(out / f'round-{r:06d}-selection.json', {'selection': [{'row': 10 * r + i, 'samples': 8} for i in range(4)],
              'cases': [{'id': t} for t in TASKS[r]], 'budget': {'offset': 10 * r, 'rows': 4, 'group': 8}})
        if r == 4:
            continue
        write(out / f'round-{r:06d}/learner/report.json', {'completed': True, 'round': r})
        write(out / f'round-{r:06d}/learner/rl-data-receipt.json', {'tasks_selected': [{'task_id': t, 'split': 'train'} for t in TASKS[r]]})
        (out / f'round-{r:06d}/learner/checkpoint.pt').write_bytes(bytes([r]) * 8)
        write(out / f'round-{r:06d}/learner-parameters.json', {'lr': r})
        counted = {g['task_id']: record_visit(history.get(g['task_id']), r) for g in groups}
        for g in groups:
            history[g['task_id']] = {**g, **counted[g['task_id']], 'observed_policy_id': HEX[CAUSE[r]], 'observed_round': r,
                                     'collection_attempt': 0}
        yields.append({'round': r, 'collection_attempt': 0, 'policy_id': HEX[CAUSE[r]], 'spread_fraction': 0.5})
        counts[str(out / f'round-{r:06d}/learner')] = {'supervised_tokens': 100 + r, 'target_tokens': 100 + r}
        accounted.append(str(out / f'round-{r:06d}/learner'))
    for r in (0, 1, 2, 3):
        pass
    updates = [{'round': r, 'checkpoint': {'path': str(out / f'round-{r:06d}/learner/checkpoint.pt'), 'sha256': HEX[RESULT[r]], 'bytes': 8},
                'schema': 'natlang.rl-production-completed-update/1'} for r in range(4)]
    for w, (parent, result, c) in enumerate((('c', 'd', 2), ('f', 'g', 4))):
        (out / f'sft-window-{w:06d}/learner').mkdir(parents=True)
        (out / f'sft-window-{w:06d}/learner/checkpoint.pt').write_bytes(b'w' * 8)
    windows = [{'window': w, 'controller_round': c, 'parent': {'sha256': HEX[p], 'path': 'p'},
                'checkpoint': {'sha256': HEX[q], 'path': str(out / f'sft-window-{w:06d}/learner/checkpoint.pt'), 'bytes': 8},
                'new_optimizer_updates': 8, 'completed_stage': False} for w, (p, q, c) in enumerate((('c', 'd', 2), ('f', 'g', 4)))]
    # distillation queue: window 0 admitted package p0 (rollout of round 1, actor b) partially; window 1 admitted the rest and
    # p1 (round 2, actor d) and p2 (round 3, actor e), refused p3 whose actor is off the lineage
    queue_dir = out / 'memetic/distill'
    packages = []
    for number, actors in enumerate((['b'], ['d'], ['e'], ['z'])):
        path = queue_dir / f'package-{number:04d}/receipt.json'
        write(path, {'schema': 'natlang.memetic-distillation-package/1', 'family': 'rollout-distillation',
                     'episode_actors': [HEX.get(a, 'z' * 64) for a in actors], 'number': number})
        packages.append(path)
    (queue_dir / 'admission-queue.jsonl').write_text(''.join(json.dumps({'package': str(p), 'family': 'rollout-distillation', 'rows': 2}) + '\n' for p in packages))
    sha = {n: rw._sha256(p) for n, p in enumerate(packages)}
    settled0 = out / 'sft-window-000000/memetic-admission/recipe-settled.json'
    write(settled0, {'settled': 0})
    write(out / 'sft-window-000001/memetic-admission/recipe-settled.json', {'settled': 1})
    ledger = {'schema': 'natlang.memetic-distillation-admission/1', 'recipe': rw._file_ref(out / 'sft-window-000001/memetic-admission/recipe-settled.json'),
              'admissions': [{'window': 0, 'recipe': {'path': str(out / 'sft-window-000000/memetic-admission/recipe-append.json'), 'sha256': 'x'}, 'record': {}},
                             {'window': 1, 'recipe': {'path': str(out / 'sft-window-000001/memetic-admission/recipe-append.json'), 'sha256': 'x'}, 'record': {}}],
              'packages': {sha[0]: {'path': 'p0', 'status': 'admitted', 'window': 1}, sha[1]: {'path': 'p1', 'status': 'admitted', 'window': 1},
                           sha[2]: {'path': 'p2', 'status': 'partial', 'window': 1}},
              'rows': {'r-p0a': {'package': sha[0], 'disposition': 'admitted', 'window': 0}, 'r-p0b': {'package': sha[0], 'disposition': 'admitted', 'window': 1},
                       'r-p1a': {'package': sha[1], 'disposition': 'admitted', 'window': 1}, 'r-p2a': {'package': sha[2], 'disposition': 'deferred', 'window': 1}}}
    state = {'schema': 'natlang.rl-production-state/1', 'round': 4, 'trained_rounds': 4, 'offset': 40, 'phase': 'requested_stop',
             'checkpoint': windows[1]['checkpoint']['path'], 'actor_policy_pending_load': HEX['g'], 'production_complete': False,
             'completed_update_history': updates, 'last_completed_update': updates[-1], 'group_stats': [], 'collection_attempt': 0,
             'trained_stage_counts': counts, 'trained_supervised_tokens': sum(v['supervised_tokens'] for v in counts.values()),
             'trained_target_tokens': sum(v['target_tokens'] for v in counts.values()), 'accounted_stages': accounted,
             'trained_task_ids': sorted({t for r in range(4) for t in TASKS[r]}),
             'attempted_task_ids': sorted({t for r in range(4) for t in TASKS[r]}), 'collected_task_ids': sorted({t for r in range(4) for t in TASKS[r]}),
             'available_tasks': 10, 'collected_task_fraction': 0.5, 'task_curriculum_history': history, 'task_curriculum_spread_yield': yields,
             'latest_reward': json.loads((out / 'round-000003/score.json').read_text()),
             'last_learner_progress': {'progress_of': 3}, 'update_learner_parameters': {'lr': 3},
             'precollection_history': [{'round': 1, 'behavior_policy_id': HEX['a']}, {'round': 3, 'behavior_policy_id': HEX['d']}],
             'held_eval_transitions': [{'recorded_at_round': 2, 'from_round': 3}],
             'rollout_distillation_transitions': [{'recorded_at_round': 1}], 'task_curriculum_transitions': [{'round': 0}],
             'interleave': {'schema': 'natlang.rl-sft-interleave/1', 'stage': 's', 'sft_windows': 2, 'rl_rounds_at_last_window': 4,
                            'windows': windows, 'stage_bindings': [{'round': 0, 'x': 1}, {'round': 4, 'x': 2}],
                            'memetic': {'active_round': None, 'offset': 16, 'pending': None}, 'memetic_distillation': ledger}}
    write(out / 'state.json', state)
    write(out / 'config.json', {'cases': str(out / 'cases.jsonl'), 'out': str(out)})
    (out / 'cases.jsonl').write_text(''.join(json.dumps({'id': t}) + '\n' for t in sorted({t for r in range(5) for t in TASKS[r]} | {'unused'})))
    (out / 'controller.lock').write_text('')
    (out / 'round-000003-pre').mkdir()          # round 3 was pre-collected: its selection predates update 2
    (out / 'round-000003/pre-collection.json').write_text('{}')
    monkeypatch.setattr(rw, '_learner_progress', lambda path: {'progress_of': json.loads(Path(path).read_text())['round']})
    return out, state


def test_rewind_to_the_last_update_reruns_the_window_it_made_due(tmp_path, monkeypatch):
    out, state = build(tmp_path, monkeypatch)
    planned = rw.plan(out, HEX['f'], every=4, now=1)
    new = planned['new']
    assert (new['round'], new['trained_rounds'], new['phase'], new['checkpoint']) == (4, 4, 'collect', state['completed_update_history'][3]['checkpoint']['path'])
    assert new['actor_policy_pending_load'] == HEX['f']
    assert new['interleave']['sft_windows'] == 1 and new['interleave']['rl_rounds_at_last_window'] == 2
    assert [w['window'] for w in new['interleave']['windows']] == [0]
    # everything the round-3 update committed is still there; the distillation ledger is back to window 0's decisions
    for key in ('trained_task_ids', 'attempted_task_ids', 'collected_task_ids', 'task_curriculum_history', 'latest_reward', 'last_learner_progress',
                'update_learner_parameters', 'precollection_history', 'trained_stage_counts', 'offset', 'trained_supervised_tokens'):
        assert new[key] == state[key], key
    ledger = new['interleave']['memetic_distillation']
    assert set(ledger['rows']) == {'r-p0a'} and [a['window'] for a in ledger['admissions']] == [0]
    assert sorted(p['status'] for p in ledger['packages'].values()) == ['partial', 'refused'] and ledger['recipe']['path'].endswith('window-000000/memetic-admission/recipe-settled.json')
    # the package whose actor is off the lineage is refused, the others (actors on the lineage) are pending again
    assert [r['actors'] for r in planned['record']['memetic_distillation']['refused_off_lineage']] == [['z' * 64]]
    names = {src.name for src, _ in planned['moves']}
    assert names == {'round-000004', 'round-000004-selection.json', 'sft-window-000001'}
    assert new['rewind_history'][-1]['to']['checkpoint_sha256'] == HEX['f']
    assert planned['record']['rolled_back_policies'] == [HEX['g']]


def test_rewind_to_an_earlier_update_restores_every_per_round_field(tmp_path, monkeypatch):
    out, state = build(tmp_path, monkeypatch)
    planned = rw.plan(out, HEX['e'], every=4, now=2)       # after round 2's update: round 3 about to collect
    new = planned['new']
    assert (new['round'], new['trained_rounds'], new['offset']) == (3, 3, 30)
    assert [h['round'] for h in new['completed_update_history']] == [0, 1, 2] and new['last_completed_update']['round'] == 2
    assert sorted(new['trained_stage_counts']) == [str(out / f'round-{r:06d}/learner') for r in range(3)]
    assert new['trained_supervised_tokens'] == 100 + 101 + 102 and len(new['accounted_stages']) == 3
    kept = {t for r in range(3) for t in TASKS[r]}
    assert set(new['trained_task_ids']) == set(new['attempted_task_ids']) == set(new['collected_task_ids']) == kept
    assert 't3-0' not in new['task_curriculum_history'] and new['task_curriculum_history']['shared']['visit_rounds'] == [0, 1, 2]
    assert new['task_curriculum_history']['shared']['observed_round'] == 2
    assert new['latest_reward']['round_marker'] == 2 and new['last_learner_progress'] == {'progress_of': 2}
    assert new['update_learner_parameters'] == {'lr': 2} and [y['round'] for y in new['task_curriculum_spread_yield']] == [0, 1, 2]
    assert new['precollection_history'] == [{'round': 1, 'behavior_policy_id': HEX['a']}]
    assert new['held_eval_transitions'] and new['rollout_distillation_transitions'] and new['task_curriculum_transitions']
    assert new['interleave']['stage_bindings'] == [{'round': 0, 'x': 1}]
    assert new['interleave']['memetic'] == {'pending': None}           # round 3 was the first memetic round
    assert new['interleave']['sft_windows'] == 1 and 'memetic_distillation' in new['interleave']
    # round 3 had a pre-collection, so its sealed selection stays; its (consumed) pre-collection and everything later go
    names = {src.name for src, _ in planned['moves']}
    assert names == {'round-000003', 'round-000003-pre', 'round-000004', 'round-000004-selection.json', 'sft-window-000001'}
    assert [r['actors'] for r in planned['record']['memetic_distillation']['refused_off_lineage']] == [['z' * 64]]
    assert planned['record']['rolled_back_policies'] == sorted([HEX['f'], HEX['g']])
    reselect = rw.plan(out, HEX['e'], every=4, now=2, reselect=True)
    assert 'round-000003-selection.json' in {src.name for src, _ in reselect['moves']}


def test_rewind_to_a_window_result(tmp_path, monkeypatch):
    out, state = build(tmp_path, monkeypatch)
    planned = rw.plan(out, HEX['d'], every=4, now=3)       # window 0's result: round 2 about to collect
    new = planned['new']
    assert (new['round'], new['checkpoint'], new['actor_policy_pending_load']) == (2, state['interleave']['windows'][0]['checkpoint']['path'], HEX['d'])
    assert [h['round'] for h in new['completed_update_history']] == [0, 1] and new['interleave']['sft_windows'] == 1
    assert new['interleave']['rl_rounds_at_last_window'] == 2 and new['offset'] == 20
    assert new['interleave']['memetic']['pending'] is None
    assert {src.name for src, _ in planned['moves']} >= {'round-000002', 'round-000003', 'sft-window-000001'}


def test_refusals(tmp_path, monkeypatch):
    out, state = build(tmp_path, monkeypatch)
    with pytest.raises(rw.Refusal, match='not a committed checkpoint'):
        rw.plan(out, HEX['h'])
    broken = copy.deepcopy(state)
    broken['task_curriculum_history']['t0-0']['mean'] = 0.123      # the live history disagrees with the replayed rounds
    write(out / 'state.json', broken)
    with pytest.raises(rw.Refusal, match='disagrees'):
        rw.plan(out, HEX['e'], every=4)
    write(out / 'state.json', state)
    with open(out / 'controller.lock', 'rb') as holder:
        import fcntl
        fcntl.flock(holder, fcntl.LOCK_EX)
        assert any('controller.lock' in p for p in rw.running_problems(out))
        assert rw.main([str(out), '--to-policy', HEX['f']]) == 2         # a real run refuses
    assert not list(out.glob('*rolled-back*'))
    assert rw.main([str(out), '--to-policy', HEX['h'], '--dry-run']) == 2


def test_apply_moves_directories_keeps_the_old_state_and_is_resumable_by_the_controller(tmp_path, monkeypatch):
    out, state = build(tmp_path, monkeypatch)
    assert rw.main([str(out), '--to-policy', HEX['f'], '--dry-run', '--memetic-every', '4']) == 0
    assert json.loads((out / 'state.json').read_text()) == state and not list(out.glob('*rolled-back*'))   # dry run writes nothing
    assert rw.main([str(out), '--to-policy', HEX['f'], '--memetic-every', '4', '--reason', 'test', '--no-verify-sha']) == 0
    new = json.loads((out / 'state.json').read_text())
    assert new['interleave']['sft_windows'] == 1 and new['rewind_history'][0]['reason'] == 'test'
    assert not (out / 'round-000004').exists() and not (out / 'sft-window-000001').exists()
    moved = [Path(m['to']) for m in new['rewind_history'][0]['moved']]
    assert len(moved) == 3 and all(p.exists() for p in moved)
    backup = Path(new['rewind_history'][0]['state_backup'])
    assert json.loads(backup.read_text()) == state
    # the window the cadence makes due is due again; a second rewind to the same point only has the controller's state to move
    from natlang_neuralese.rl import interleave
    assert interleave.window_due({'parameters': {'rl_rounds_per_window': 2}}, new) is True
