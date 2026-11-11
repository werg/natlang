"""Queued source withdrawals are applied by the window boundary as one proof-bound source correction, with the packages."""
import hashlib
import json
from pathlib import Path

import pytest

from natlang_neuralese.common.jsonio import write_json_durable as write_json
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.rl import interleave, memetic_admission as admission, source_withdrawal as withdrawal
from natlang_neuralese.train.native_source_correction import apply_source_correction, source_correction_dependencies
from test_memetic_admission import FP, graph, lineage, package, window_harness

LEAVES = [[('keep', 'ordinary'), ('tpl-a', 'tpl'), ('tpl-b', 'tpl')], [('tpl-c', 'tpl')],
          [('tpl-d', 'tpl'), ('other', 'ordinary')]]
EARNED = 'tpl-b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_of(train):
    """The parent's saved SFT frame: every row has one of two visits earned except the earned row ``EARNED``."""
    source = NativeGraphSource(train, FP, required_split='train')
    ids = [e['id'] for e in source.entries]
    return {'source_receipts': [{'sha256': source.file_sha256}], 'source_row_order': ids,
            'source_row_bindings': {e['id']: e['binding'] for e in source.entries},
            'required_visits': {i: 2 for i in ids}, 'exposure_counts': {i: 2 if i == EARNED else 1 for i in ids}}


def world(tmp_path, monkeypatch, *, focus=False, **parameters):
    """A lineage whose source holds three leaves of rows and families; the controller's checkpoint frame and model id
    are the test's."""
    config, declared, state, out, train = lineage(
        tmp_path, quota=True, geometry=True, focus=focus, leaves=LEAVES, extra_scales={'tpl': .5},
        distillation_max_share=1., distillation_family_scale=.2, **parameters)
    frame = frame_of(train)
    frame['options'] = {'ce_supervision_policy_binding': stage_inputs(config['production_recipe'],
                                                                      declared['sft_stage'])['ce_supervision_policy']}
    monkeypatch.setattr(admission, '_parent_sft_frame', lambda checkpoint, stage: frame)
    monkeypatch.setattr(withdrawal, '_model_id', lambda checkpoint: 'lfm:test-model')
    monkeypatch.setattr(admission, '_focus_state', lambda checkpoint: None)
    seen = []
    window_harness(monkeypatch, seen)
    return config, declared, state, out, train, frame, seen


def plan(tmp_path, rows, caps=None, name='plan.json', **extra):
    document = {'schema': withdrawal.PLAN_SCHEMA, 'reason': 'test: withdraw useless template rows',
                'withdraw': {ident: {'family': family, 'criterion': 'test criterion'} for ident, family in rows},
                'visit_cap': {ident: {'required_visits': value} for ident, value in (caps or {}).items()}, **extra}
    path = tmp_path / name
    write_json(path, document)
    return path


def stage_inputs(recipe_path, stage):
    from natlang_neuralese.train.recipe import load_recipe, resolve_stage_inputs
    return resolve_stage_inputs(load_recipe(recipe_path), stage, {})


def run_window(config, declared, state, out):
    return interleave.run_sft_window(config, declared, state, out / 'state.json', None)


def test_a_queued_withdrawal_and_a_package_become_one_source_correction_at_the_boundary(tmp_path, monkeypatch):
    config, declared, state, out, train, frame, seen = world(tmp_path, monkeypatch)
    originals = {p: sha(p) for p in tmp_path.glob('*.jsonl')}
    # tpl-a: part of a leaf; tpl-c: a whole leaf; tpl-b: earned since the plan was made (kept); keep: visit cap
    path = plan(tmp_path, [('tpl-a', 'tpl'), ('tpl-b', 'tpl'), ('tpl-c', 'tpl'), ('gone', 'tpl')], caps={'keep': 1, 'other': 5})
    line = withdrawal.queue_plan(out, path, log=lambda text: None)
    assert line['sha256'] == sha(path) and withdrawal.queue_plan(out, path)['sha256'] == sha(path)
    assert len((out / 'memetic' / 'distill' / 'withdrawal-queue.jsonl').read_text().splitlines()) == 1  # idempotent
    package(out, [graph('memetic:a', 6), graph('memetic:b', 8)], sha(config['checkpoint']))
    assert run_window(config, declared, state, out) == 'adopted'
    assert not state['interleave'].get('memetic_distillation_failures')
    recipe_path, stage = seen[0]
    inputs = stage_inputs(recipe_path, stage)
    old_source = NativeGraphSource(train, FP, required_split='train')
    successor = NativeGraphSource(inputs['native_graphs']['path'], FP, required_split='train')
    # exact retained prefix and order, then the appended rows; the withdrawn rows are out of the schedule only
    retained = [(e['id'], e['binding']) for e in old_source.entries if e['id'] not in ('tpl-a', 'tpl-c')]
    assert [(e['id'], e['binding']) for e in successor.entries[:len(retained)]] == retained
    assert [e['id'] for e in successor.entries[len(retained):]] == ['memetic:a', 'memetic:b']
    assert {p: sha(p) for p in tmp_path.glob('*.jsonl')} == originals  # no original file is touched
    # the proof: deferred rows with their visit debt, added rows, the quota revision, bound as the stage's correction
    proof_ref = inputs['source_correction']
    source_correction_dependencies({'path': proof_ref['path'], 'sha256': proof_ref['sha256']})
    proof = json.loads(Path(proof_ref['path']).read_text())
    ledger_rows = [json.loads(x) for x in Path(proof['row_bindings_path']).read_text().splitlines()]
    assert {(r['id'], r['disposition']) for r in ledger_rows} == {
        ('tpl-a', 'deferred'), ('tpl-c', 'deferred'), ('memetic:a', 'added'), ('memetic:b', 'added')}
    assert proof['prior_source']['sha256'] == sha(train) and proof['current_source']['sha256'] == successor.file_sha256
    assert [json.loads(x)['id'] for x in Path(proof['quota_revision']['row_bindings_path']).read_text().splitlines()] == ['keep']
    assert inputs['continued_native_graphs']['sha256'] == sha(train)
    # the trainer's own application of that proof to the parent's earned frame
    training = {**frame, 'source_receipts': [old_source.receipt()], 'options': {'epochs': 2., '_communication_ids': []},
                'pending_source_rows': {}, 'global_step': 9, 'optimizer': {}, 'rng': {}, 'weights': {}}
    result, summary = apply_source_correction(
        training, proof_ref['path'], model_id='lfm:test-model', stage_id='sft', current_source_receipt=successor.receipt(),
        current_bindings=successor.bindings, current_order=[e['id'] for e in successor.entries],
        current_communication_ids=[])
    assert set(result['pending_source_rows']) == {'tpl-a', 'tpl-c'}  # visit debt and credit stay in the pending ledger
    assert result['required_visits']['keep'] == 1 and result['exposure_counts'][EARNED] == 2
    assert result['exposure_counts']['memetic:a'] == 0 and EARNED in result['source_row_bindings']
    # CE policy: family map, objective transition, source-bound quota basis, scales
    policy = json.loads(Path(inputs['ce_supervision_policy']['path']).read_text())
    family_map = json.loads(Path(policy['objective']['family_balance']['map']['path']).read_text())
    assert family_map['source_sha256'] == successor.file_sha256
    assert [(r['id'], r['family']) for r in family_map['rows']] == [
        ('keep', 'ordinary'), ('tpl-b', 'tpl'), ('tpl-d', 'tpl'), ('other', 'ordinary'),
        ('memetic:a', 'memetic-guidance'), ('memetic:b', 'memetic-guidance')]
    assert policy['quota_aware_basis']['candidate_source_sha256'] == successor.file_sha256
    assert 'from' in policy['objective_transition'] and 'memetic-guidance' in policy['objective']['family_balance']['scales']
    # packed geometry follows the final rows; the descriptor binds the final source
    geometry = json.loads(Path(stage['parameters']['sft_packed_geometry']['path']).read_text())
    assert len(geometry['kv_slots']) == len(successor.entries) == 6
    # the ledger and the admission record
    memetic = state['interleave']['memetic_distillation']
    [entry] = memetic['withdrawals'].values()
    assert entry['status'] == 'applied' and entry['window'] == 0
    record = json.loads(Path(memetic['admissions'][0]['record']['path']).read_text())
    assert record['withdrawal']['rows'] == {'tpl': 2} and record['withdrawal']['earned_rows_kept'] == {'tpl': 1}
    assert record['withdrawal']['visit_capped'] == 1 and record['rows_added'] == 2
    # settled: the next window continues on the changed source and neither re-withdraws nor re-appends
    state['round'] = 3
    assert run_window(config, declared, state, out) == 'adopted'
    later = stage_inputs(*seen[1])
    assert later['native_graphs']['sha256'] == successor.file_sha256 and 'continued_native_graphs' not in later
    assert len(memetic['admissions']) == 1


def test_a_withdrawal_alone_changes_the_source_and_keeps_the_pending_focus_on_the_filtered_leaves(tmp_path, monkeypatch):
    config, declared, state, out, train, frame, seen = world(tmp_path, monkeypatch, focus=3)  # keep, tpl-a, tpl-b
    old_focus = stage_inputs(config['production_recipe'], declared['sft_stage'])['pending_focus']
    monkeypatch.setattr(admission, '_focus_state',
                        lambda checkpoint: {'state': 'active', 'request': {'sha256': old_focus['sha256']}})
    withdrawal.queue_plan(out, plan(tmp_path, [('tpl-a', 'tpl'), ('tpl-c', 'tpl')]), log=lambda text: None)
    assert run_window(config, declared, state, out) == 'adopted'
    assert not state['interleave'].get('memetic_distillation_failures')
    inputs = stage_inputs(*seen[0])
    from natlang_neuralese.train import native_pending_focus
    document, _ = native_pending_focus.dependencies({'path': inputs['pending_focus']['path'],
                                                     'sha256': inputs['pending_focus']['sha256']})
    assert [r['id'] for r in document['rows']] == ['keep', 'tpl-b']  # tpl-a left the request
    assert document['supersedes']['request_sha256'] == old_focus['sha256']
    new_source = NativeGraphSource(inputs['native_graphs']['path'], FP, required_split='train')
    from natlang_neuralese.train.native_cohort_schedule import source_cohort_ids
    assert {r['source_sha256'] for r in document['rows']} == set(source_cohort_ids(new_source)[:2])
    assert [e['id'] for e in new_source.entries] == ['keep', 'tpl-b', 'tpl-d', 'other']
    assert inputs['native_graphs']['sha256'] == new_source.file_sha256  # the filtered manifest itself, no wrapper
    record = json.loads(Path(state['interleave']['memetic_distillation']['admissions'][0]['record']['path']).read_text())
    assert record['pending_focus']['withdrawn_rows_dropped'] == 1 and record['rows_added'] == 0


def test_an_earned_row_is_never_withdrawn_and_a_plan_with_nothing_left_is_consumed(tmp_path, monkeypatch):
    config, declared, state, out, train, frame, seen = world(tmp_path, monkeypatch)
    path = plan(tmp_path, [(EARNED, 'tpl')], caps={'keep': 1})
    withdrawal.queue_plan(out, path, log=lambda text: None)
    assert run_window(config, declared, state, out) == 'adopted'
    assert seen[0][0] == config['production_recipe']  # the window runs on the existing source
    [entry] = state['interleave']['memetic_distillation']['withdrawals'].values()
    assert entry['status'] == 'empty' and entry['earned_kept'] == {'tpl': 1} and entry['visit_caps_dropped'] == 1
    assert not state['interleave'].get('memetic_distillation_failures')
    assert not withdrawal.pending_plans(config['out'], state['interleave']['memetic_distillation'])
    # resolve() itself: nothing earned, rows no longer active or changed are skipped, never withdrawn
    family = {'a': 'tpl', 'b': 'tpl', 'c': 'tpl'}
    bindings = {'a': 'x', 'b': 'y', 'c': 'z'}
    frame = {'exposure_counts': {'a': 2, 'b': 1}, 'required_visits': {'a': 2, 'b': 2, 'c': 3}}
    resolved = withdrawal.resolve([('p', {'withdraw': {
        'a': {'family': 'tpl', 'criterion': 'k'}, 'b': {'family': 'tpl', 'criterion': 'k', 'binding': 'other'},
        'c': {'family': 'tpl', 'criterion': 'k'}, 'gone': {'family': 'tpl', 'criterion': 'k'}}})], frame, family, bindings)
    assert set(resolved['withdrawn']) == {'c'} and resolved['earned_kept'] == {'tpl': 1}
    assert resolved['skipped']['p'] == {'inactive': 1, 'binding_changed': 1}


def test_a_plan_about_another_family_map_is_refused_once_and_a_malformed_plan_never_stops_the_boundary(tmp_path, monkeypatch):
    config, declared, state, out, train, frame, seen = world(tmp_path, monkeypatch)
    wrong = plan(tmp_path, [('keep', 'tpl')], name='wrong.json')  # keep is an 'ordinary' row
    withdrawal.queue_plan(out, wrong, log=lambda text: None)
    bad = tmp_path / 'bad.json'
    write_json(bad, {'schema': 'natlang.something-else/1'})
    with pytest.raises(withdrawal.PlanRefused):
        withdrawal.queue_plan(out, bad)
    # a queued plan that changed after it was queued is refused at the boundary
    late = plan(tmp_path, [('tpl-a', 'tpl')], name='late.json')
    withdrawal.queue_plan(out, late, log=lambda text: None)
    write_json(late, {**json.loads(late.read_text()), 'reason': 'edited after queueing'})
    package(out, [graph('memetic:a', 6)], sha(config['checkpoint']))
    assert run_window(config, declared, state, out) == 'adopted'
    memetic = state['interleave']['memetic_distillation']
    assert sorted(item['status'] for item in memetic['withdrawals'].values()) == ['refused', 'refused']
    assert not state['interleave'].get('memetic_distillation_failures')
    # the package still joined: a refused withdrawal never holds an append back
    assert memetic['admissions'] and stage_inputs(*seen[0])['native_graphs']['sha256'] != sha(train)
    assert 'source_correction' not in stage_inputs(*seen[0])
    assert not withdrawal.pending_plans(config['out'], memetic)


def test_a_failing_withdrawal_is_recorded_stays_pending_and_the_package_is_appended_without_it(tmp_path, monkeypatch):
    config, declared, state, out, train, frame, seen = world(tmp_path, monkeypatch)
    path = plan(tmp_path, [('tpl-a', 'tpl')])
    withdrawal.queue_plan(out, path, log=lambda text: None)
    package(out, [graph('memetic:a', 6)], sha(config['checkpoint']))

    def broken(*args, **kwargs):
        raise ValueError('exact current own source index is unavailable for leaf')

    monkeypatch.setattr(withdrawal, 'filter_leaf_files', broken)
    assert run_window(config, declared, state, out) == 'adopted'  # the controller is not stopped
    [failure] = state['interleave']['memetic_distillation_failures']
    assert failure['with_withdrawal'] and 'index is unavailable' in failure['error']
    inputs = stage_inputs(*seen[0])
    assert 'source_correction' not in inputs and inputs['continued_native_graphs']['sha256'] == sha(train)
    memetic = state['interleave']['memetic_distillation']
    assert 'withdrawals' not in memetic or not memetic['withdrawals']  # nothing settled
    [pending] = withdrawal.pending_plans(config['out'], memetic)
    assert pending['sha256'] == sha(path)


def test_withdrawal_queue_is_a_file_the_package_queue_readers_never_see(tmp_path, monkeypatch):
    """An older controller reads only admission-queue.jsonl: a queued withdrawal waits for the first boundary of a
    controller that has this module, and never refuses or breaks a package append."""
    config, declared, state, out, train, frame, seen = world(tmp_path, monkeypatch)
    withdrawal.queue_plan(out, plan(tmp_path, [('tpl-a', 'tpl')]), log=lambda text: None)
    assert withdrawal.queue_path(out) != admission.queue_path(out)
    assert admission._queued(out) == [] and not admission.unseen_packages(out, state)


def test_manifest_rewrite_drops_whole_leaves_and_swaps_filtered_ones(tmp_path):
    (tmp_path / 'manifests').mkdir()
    manifest = tmp_path / 'm.json'
    manifest.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': '/x/a.jsonl', 'sha256': 'a'}, {'kind': 'native-graphs', 'path': '/x/b.jsonl', 'sha256': 'b'},
        {'kind': 'native-graphs', 'path': '/x/c.jsonl', 'sha256': 'c'}]}))
    doc, changed = withdrawal.rewrite_manifest(
        manifest, {'a': None, 'b': {'path': '/y/b2.jsonl', 'sha256': 'b2', 'bytes': 3}}, tmp_path)
    assert changed and [s['sha256'] for s in doc['sources']] == ['b2', 'c'] and doc['sources'][0]['bytes'] == 3
    assert withdrawal.rewrite_manifest(manifest, {'a': None, 'b': None, 'c': None}, tmp_path)[0] is None
