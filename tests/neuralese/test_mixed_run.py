"""The mixed run (plans/neuralese/NEURALESE_DIRECT_V1.md "The mixed run", owner 2026-10-13): train.direct_sft
interleaves e2e producer->consumer updates (train.trajectories.build_e2e_objective) with its direct updates on one
engine, one pair of projections and one optimizer."""

import hashlib
import json

import pytest

from natlang_neuralese.train.direct_sft import e2e_argv, e2e_cursors, parse
from test_call_recurrence import _direct_stand_in


def test_the_schedule_is_a_fraction_of_all_updates_and_a_function_of_the_direct_step():
    for fraction in (0.25, 0.5, 0.1, 0.75, 1 / 3):
        cursors = [list(e2e_cursors(step, fraction)) for step in range(1, 401)]
        flat = [cursor for chunk in cursors for cursor in chunk]
        assert flat == list(range(len(flat))), 'every e2e cursor once, in order'
        for step in range(1, 401):
            done = sum(len(chunk) for chunk in cursors[:step])
            assert abs(done - fraction * (done + step)) < 1, 'after any direct update a fraction p are e2e'
        # A resume at any step continues the same cursors (no second counter).
        assert [list(e2e_cursors(step, fraction)) for step in range(173, 401)] == cursors[172:]
    assert [len(e2e_cursors(step, 0.25)) for step in range(1, 7)] == [0, 0, 1, 0, 0, 1]
    assert [len(e2e_cursors(step, 0.5)) for step in range(1, 4)] == [1, 1, 1]


def test_the_e2e_options_become_the_embedded_trainers_arguments(tmp_path):
    a = parse(['--heads', 'h.pt', '--out', str(tmp_path), '--records', 'r.jsonl', '--steps', '2',
               '--token-choice-projection', '--e2e-mix', json.dumps({
                   'fraction': 0.25, 'records': ['data/cohort'], 'pieces': '/abs/pieces.jsonl',
                   'options': {'record_source': 'trajectories', 'write_depth': 1, 'context_coverage': None,
                               'task_eval': {'cases': 2}, 'pin_first': True, 'shuffle': False}})])
    argv = e2e_argv(a.e2e_mix, a, '/repo')
    assert argv[:4] == ['--records', '/repo/data/cohort', '--pieces', '/abs/pieces.jsonl']
    assert argv[argv.index('--soft-parts') + 1] == 'text' and argv[argv.index('--handover') + 1] == 'written'
    assert argv[argv.index('--write-depth') + 1] == '1' and '--context-coverage' not in argv
    assert json.loads(argv[argv.index('--task-eval') + 1]) == {'cases': 2}
    assert '--pin-first' in argv and '--no-shuffle' in argv
    implied = parse(['--heads', 'h.pt', '--out', str(tmp_path), '--records', 'r.jsonl', '--steps', '2', '--e2e-mix',
                     json.dumps({'fraction': 0.25, 'records': ['r'], 'pieces': ['p'],
                                 'token_choice': {'lr': 3e-6, 'optimizer': 'muon'},
                                 'writer': {'lr': 2e-6, 'optimizer': 'adamw'}})])
    assert implied.token_choice_projection and implied.token_choice_lr == 3e-6
    assert (implied.neuralese_projection_lr, implied.neuralese_projection_optimizer) == (2e-6, 'adamw')
    with pytest.raises(SystemExit):
        parse(['--heads', 'h.pt', '--out', str(tmp_path), '--records', 'r.jsonl', '--steps', '2',
               '--e2e-mix', json.dumps({'fraction': 1.0, 'records': ['r'], 'pieces': ['p']})])


E2E_OPTIONS = {'steps': 1, 'batch': 1, 'eval': 2, 'write_depth': 1, 'tokens_per_vector': 1,
               'writer_supervision': 'native-value', 'writer_length_policy': 'native-value',
               'stop_supervision': 'generated-length', 'writer_text_weight': 0, 'distill': 0,
               'record_order': 'family-interleave'}


def _mixed(tmp_path, direct, records, pieces, *extra):
    from natlang_neuralese.train import direct_sft
    return direct_sft.main([
        '--heads', str(tmp_path / 'raw-token-heads.pt'), '--records', str(direct), '--pieces', str(pieces),
        '--out', str(tmp_path / 'mixed'), '--device', 'cpu', '--steps', '3', '--eval-every', '3',
        '--tokens', '512', '--target-tokens', '64', '--held-documents', '1', '--projection-hidden', '16',
        '--latent-optimizer', 'muonsr', '--drift-rows', '0', '--no-drift-gate', '--token-choice-projection',
        '--token-choice-lr', '3e-6', '--neuralese-projection-lr', '3e-6',
        '--e2e-mix', json.dumps({'fraction': 0.5, 'records': [str(records)], 'pieces': [str(pieces)],
                                 'options': E2E_OPTIONS}), *extra])


def test_a_mixed_run_runs_both_kinds_and_resumes_both_cursors(loaded, tmp_path, monkeypatch):
    """Fraction 0.5: one e2e update after each direct update. Stopped after update 2 and resumed, the run continues
    with direct update 3 and e2e cursor 2; the log has per-kind seconds, supervised tokens and tokens/s, and the eval
    carries the e2e objective's paired held readers."""
    records, pieces, _ = _direct_stand_in(loaded, tmp_path, monkeypatch)
    # The direct documents: the same calls, admitted, with independent train/held source groups (the e2e records keep
    # theirs).
    direct = tmp_path / 'direct-records.jsonl'
    direct.write_text(''.join(json.dumps({**record, 'source_groups': [f"direct-{record['split']}"],
                                          'training_admission': {'approved': True}}) + '\n'
                              for record in map(json.loads, records.read_text().splitlines())))
    _mixed(tmp_path, direct, records, pieces, '--stop-at', '2', '--checkpoint-every', '1')
    _mixed(tmp_path, direct, records, pieces)
    out = tmp_path / 'mixed'
    rows = [json.loads(line) for line in (out / 'train.jsonl').read_text().splitlines()]
    assert [(row['kind'], row['step'], row.get('e2e_cursor')) for row in rows] == [
        ('direct', 1, None), ('e2e', 1, 0), ('direct', 2, None), ('e2e', 2, 1), ('direct', 3, None), ('e2e', 3, 2)]
    for row in rows:
        assert row['seconds'] > 0 and row['supervised_tokens'] > 0 and row['supervised_tokens_per_second'] > 0
    assert all(row['record_id'] == 'caller' and row['write_lengths'] for row in rows if row['kind'] == 'e2e')
    evaluations = [json.loads(line) for line in (out / 'e2e-eval.jsonl').read_text().splitlines()]
    # The baseline, the stop-at end of the first run (an evaluation like any run end) and the final update.
    assert [e['step'] for e in evaluations] == [0, 2, 3] and all(e['written'] for e in evaluations)
    report = json.loads((out / 'report.json').read_text())
    assert report['e2e']['step'] == 3 and report['steps'] == 3


def test_both_lines_hand_off_into_the_same_mixed_run():
    """Lockstep: both launch recipes carry one e2e_mix and continue their running direct_sft by an objective
    continuation that adds only it; the e2e base no longer declares a readiness moment."""
    from pathlib import Path
    from natlang_neuralese.train.recipe import resolve_recipe_data, validate_e2e_mix
    recipes = Path(__file__).resolve().parents[2] / 'training' / 'neuralese' / 'recipes'
    mixed = {line: resolve_recipe_data(recipes / f'{name}.json') for line, name in (
        ('mellum', 'neuralese-mixed-mellum-v4'), ('lfm', 'neuralese-mixed-lfm-qad-broad-v1'))}
    assert mixed['mellum']['e2e_mix'] == mixed['lfm']['e2e_mix']
    for recipe in mixed.values():
        validate_e2e_mix(recipe['e2e_mix'], {stage['id']: stage['kind'] for stage in recipe['stages']})
        assert recipe['objective_continuation']['stage'] == 'direct_sft'
        assert recipe['objective_continuation']['previous']['e2e_mix'] is None
        assert recipe['e2e_mix']['token_choice'] == recipe['e2e_mix']['writer'] == {'lr': 3e-06, 'optimizer': 'muon'}
    base = resolve_recipe_data(recipes / 'neuralese-e2e-v1.json')
    assert 'parent_readiness' not in base
    manifest = json.loads((recipes.parents[1] / 'corpus-manifests' /
                           'neuralese-e2e-cohort-20261014-v4.json').read_text())
    files = {item['path']: item['sha256'] for item in manifest['files']}
    for role in ('records', 'pieces'):
        assert mixed['lfm']['e2e_mix'][role] == {
            'path': manifest['path'] + '/' + role + '.jsonl', 'sha256': files[role + '.jsonl']}


def test_neuralese_spans_with_an_edge_become_context_only_merged_with_declared_spans(tmp_path):
    """Option A (NEURALESE_DIRECT_V1.md "Neuralese outputs in direct samples"): with e2e_mix active, a Neuralese output
    span whose value an e2e edge trains is context only in direct updates; spans without an edge are counted."""
    from natlang_neuralese.train.direct_sft import context_only_spans, merged_context_only_spans, neuralese_spans
    sidecar = tmp_path / 'neuralese-spans.jsonl'
    sidecar.write_text(json.dumps({'id': 'a', 'spans': [[10, 14, 'result:x', True], [20, 25, 'note:y', False]]}) + '\n'
                       + json.dumps({'id': 'b', 'spans': [[3, 5, 'result:z', False]]}) + '\n')
    assert neuralese_spans({'fraction': 0.25}, tmp_path) is None
    edges, plain = neuralese_spans({'neuralese_spans': {'path': sidecar.name,
                                                      'sha256': hashlib.sha256(sidecar.read_bytes()).hexdigest()}},
                                   tmp_path)
    assert edges == {'a': [[10, 14]]} and plain == 2
    merged = merged_context_only_spans([[0, 4], [12, 18]], edges['a'] + [[30, 31]])
    assert merged == [[0, 4], [10, 18], [30, 31]]
    assert context_only_spans({'context_only_spans': merged, 'token_ids': list(range(40))}) == [(0, 4), (10, 18),
                                                                                               (30, 31)]
