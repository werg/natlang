"""CPU checks of grpo.py's task supply arguments: errata exclusion by default, one episode budget with held eval."""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/neuralese_data"))
import grpo  # noqa: E402
from natlang_neuralese.rl import held_eval  # noqa: E402

PARSE_ARGS = argparse.ArgumentParser.parse_args


def parsed(monkeypatch, *extra):
    """grpo.py train's parsed arguments (main stops before any work)."""
    seen = {}

    def capture(self, args=None, namespace=None):
        seen['args'] = PARSE_ARGS(self, args, namespace)
        raise SystemExit(0)
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', capture)
    monkeypatch.setattr(sys, 'argv', ['grpo.py', 'train', '--work', 'w', '--checkpoint', 'c', '--checkpoint-sha256', 'x',
                                      *extra])
    with pytest.raises(SystemExit):
        grpo.main()
    return seen['args']


def test_errata_files_default_to_the_errata_directory_with_an_opt_out(tmp_path, monkeypatch):
    (tmp_path / 'b.jsonl').write_text('{"id": "y"}\n')
    (tmp_path / 'a.jsonl').write_text('{"id": "x"}\n')
    (tmp_path / 'notes.txt').write_text('not errata')
    monkeypatch.setattr(grpo, 'ERRATA', str(tmp_path))
    args = parsed(monkeypatch)
    assert args.exclude_ids == [] and not args.no_default_errata
    assert grpo.errata_files(args) == [str(tmp_path / 'a.jsonl'), str(tmp_path / 'b.jsonl')]
    assert grpo.errata_files(parsed(monkeypatch, '--no-default-errata')) == []
    assert grpo.errata_files(parsed(monkeypatch, '--exclude-ids', 'own.jsonl')) == ['own.jsonl']
    monkeypatch.setattr(grpo, 'ERRATA', str(tmp_path / 'missing'))
    assert grpo.errata_files(SimpleNamespace(exclude_ids=[], no_default_errata=False)) == []


def test_training_and_held_eval_share_one_episode_budget(monkeypatch):
    assert held_eval.EPISODE_MODEL_CALLS == 40
    assert parsed(monkeypatch).max_model_requests == held_eval.EPISODE_MODEL_CALLS
    assert parsed(monkeypatch, '--max-model-requests', '64').max_model_requests == 64
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', PARSE_ARGS)
    seen = {}
    monkeypatch.setattr(held_eval, 'measure', lambda *a, **k: seen.update(k))
    held_eval.main(['measure', '--manifest', 'm.json', '--out', 'o', '--endpoint', 'http://x=1', '--policy-id', 'p'])
    assert seen['max_model_requests'] == held_eval.EPISODE_MODEL_CALLS


def test_reward_defaults_to_graded_with_a_binary_opt_out(monkeypatch):
    assert parsed(monkeypatch).reward == 'graded'
    assert parsed(monkeypatch, '--reward', 'binary').reward == 'binary'


def test_explore_order_defaults_to_rows_and_family_round_robin_visits_families_in_turn(monkeypatch):
    import random
    assert parsed(monkeypatch).explore_order == 'row'
    assert parsed(monkeypatch, '--explore-order', 'family').explore_order == 'family'
    rows = [{'id': f'big-{i}', 'family': 'big'} for i in range(20)] + [{'id': 's-0', 'family': 'small'},
                                                                        {'id': 's-1', 'family': 'small'},
                                                                        {'id': 't-0', 'family': 'tiny'}]
    random.Random(0).shuffle(rows)
    order = grpo.family_round_robin(rows, random.Random(0))
    assert sorted(r['id'] for r in order) == sorted(r['id'] for r in rows)
    assert {r['family'] for r in order[:3]} == {'big', 'small', 'tiny'}         # every family in the first round
    assert {r['family'] for r in order[3:5]} == {'big', 'small'}
    assert [r['family'] for r in order[5:]] == ['big'] * 18
    big = [r['id'] for r in rows if r['family'] == 'big']
    assert [r['id'] for r in order if r['family'] == 'big'] == big              # each family keeps its shuffled order
    assert order == grpo.family_round_robin(rows, random.Random(0))             # deterministic in the seed


def _shape_row(i, shape):
    return {'id': f't{i}', 'shape': shape}


def test_cap_shapes_bounds_every_prefix_and_keeps_every_row():
    rows = [_shape_row(i, 'big') for i in range(80)] + [_shape_row(80 + i, f's{i}') for i in range(120)]
    out = grpo.cap_shapes(rows, lambda r: r['shape'], share=0.1, floor=4)
    assert sorted(r['id'] for r in out) == sorted(r['id'] for r in rows)      # nothing dropped
    last_other = max(n for n, row in enumerate(out, 1) if row['shape'] != 'big')
    big = 0
    for n, row in enumerate(out[:last_other], 1):                             # while other shapes remain
        big += row['shape'] == 'big'
        assert big <= max(4, 0.1 * n)
    assert 10 <= big <= 14                                                    # and big ones enter as the prefix grows
    assert [r['shape'] for r in out[:4]] == ['big'] * 4                       # the floor: a short prefix keeps its head
    assert out[4]['shape'] == 's0'
    assert grpo.cap_shapes(rows, lambda r: r['shape'], share=0) == rows        # opt-out: the order unchanged
    assert grpo.cap_shapes(rows[80:], lambda r: r['shape'], share=0.1) == rows[80:]   # distinct shapes: unchanged


def test_cap_shapes_spreads_the_tail_over_capped_shapes():
    rows = [_shape_row(i, 'a') for i in range(30)] + [_shape_row(30 + i, 'b') for i in range(30)]
    out = grpo.cap_shapes(rows, lambda r: r['shape'], share=0.1, floor=2)
    assert [r['shape'] for r in out[:8]] == ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'b']
    assert [r['id'] for r in out if r['shape'] == 'a'] == [f't{i}' for i in range(30)]   # each shape keeps its order


def test_explore_walk_orders_by_shape_cap(monkeypatch):
    rows = [_shape_row(i, 'big') for i in range(40)] + [_shape_row(40 + i, f's{i}') for i in range(40)]
    monkeypatch.setattr(grpo, 'task_shapes', lambda tasks: {r['id']: r['shape'] for r in tasks})
    walk = grpo.ExploreWalk(rows, shape_cap=0.1)
    assert sum(r['shape'] == 'big' for r in walk.tasks[:40]) <= 4
    assert grpo.ExploreWalk(rows).tasks == rows
    assert parsed(monkeypatch).explore_shape_cap == grpo.SHAPE_CAP


def test_program_shape_abstracts_item_data_but_not_the_template():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/rl_tasks'))
    from measure_repetition import program_shape

    def row(repo, path, question='how many lines were added', root='investigate_history.nl'):
        text = (f'---\nargs: {{}}\nreturns: "{{ added: number }}"\n---\ngit is a read-only git service over the newest 40 '
                f'commits of the {repo} repository. Over all the commits, {question} to {path}?')
        return {'id': repo, 'semantics': {'root': root, 'files': {root: text}, 'inputs': {'x': repo}}}
    a = program_shape(row('ncsoft/promotionImpact', 'man/detectOutliers.Rd'))
    assert a == program_shape(row('torvalds/linux', 'kernel/sched_core.c'))        # another item, one template
    assert a == program_shape(row('torvalds/linux', 'kernel/x.c', root='other_name.nl'))   # the root's stem abstracted
    assert a != program_shape(row('torvalds/linux', 'kernel/x.c', question='how many commits touched'))
    assert program_shape({'id': 'x'}) == 'id:x'


def test_pool_load_leaves_out_rows_the_curriculum_refuses_with_an_opt_out(tmp_path, monkeypatch):
    import json
    import expert_iteration as ei
    from natlang_neuralese.rl import curriculum
    rows = [{'id': 'q', 'curriculum': {'rl': {'qualification': 'qualified'}}}, {'id': 'plain'},
            {'id': 'pending', 'curriculum': {'rl': {'qualification': 'source-only-live-pending'}}}]
    pool = tmp_path / 'pool.jsonl'
    pool.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    monkeypatch.setattr(held_eval, 'training_exclusion_ids', lambda *a, **k: set())
    monkeypatch.setattr(grpo, 'ERRATA', str(tmp_path / 'none'))
    assert [curriculum.selection_refusal(r) is None for r in rows] == [True, True, False]
    with pytest.raises(ValueError, match='qualification'):
        curriculum.metadata(rows[2])                                    # the curriculum's refusal: the same predicate
    args = SimpleNamespace(pool=[str(pool)], held_manifest=None, exclude_family_regex=None, seed=0)
    tasks, _ = grpo.load_tasks(ei.pool_args(args), tmp_path)           # expert-iteration select and collect
    assert sorted(r['id'] for r in tasks) == ['plain', 'q']
    assert json.loads((tmp_path / 'pool.json').read_text())['excluded_unqualified_ids'] == 1
    args.include_unqualified = True                                     # the opt-out
    assert len(grpo.load_tasks(ei.pool_args(args), None)[0]) == 3
    assert not parsed(monkeypatch).include_unqualified                  # grpo train: left out by default
