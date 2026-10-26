import hashlib
import json
from pathlib import Path

import pytest

from natlang_neuralese.train.native_source_correction import (
    SCHEMA, apply_source_correction, source_correction_dependencies,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _receipt(path, digest, ids, source_hashes):
    sources = []
    for row_id, source_sha in zip(ids, source_hashes):
        sources.append({'path': f'/src/{source_sha}.jsonl.zst', 'sha256': source_sha,
                        'rows': 1, 'id': row_id})
    return {'path': str(path), 'sha256': digest, 'rows': len(ids), 'sources': sources}


def _write_proof(tmp_path, prior, current, rows, model_id='lfm2:inventory', protected_held=None):
    row_path = tmp_path / 'rows.jsonl'
    row_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    proof_path = tmp_path / 'source-correction.json'
    proof = {'schema': SCHEMA, 'model_id': model_id, 'stage_id': 'native_full_sft',
             'prior_source': {'path': prior['path'], 'sha256': prior['sha256']},
             'current_source': {'path': current['path'], 'sha256': current['sha256']},
             'row_bindings_path': str(row_path), 'row_bindings_sha256': _sha(row_path)}
    if protected_held is not None:
        held_path = tmp_path / 'held-rows.jsonl'
        held_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n'
                                      for row in protected_held['rows']))
        proof['protected_held_correction'] = {
            'schema': 'natlang.native-protected-held-correction/1',
            'prior_source': {'path': protected_held['prior']['path'],
                             'sha256': protected_held['prior']['sha256']},
            'current_source': {'path': protected_held['current']['path'],
                               'sha256': protected_held['current']['sha256']},
            'row_bindings_path': str(held_path), 'row_bindings_sha256': _sha(held_path),
        }
    proof_path.write_text(json.dumps(proof, sort_keys=True))
    return proof_path


def _row(row_id, old_binding, new_binding, old_sha, new_sha, disposition, reason=None):
    return {'id': row_id, 'old_binding': old_binding, 'new_binding': new_binding,
            'old_source_sha256': old_sha, 'new_source_sha256': new_sha,
            'disposition': disposition, 'reason': reason}


def test_source_correction_preserves_state_and_resets_only_corrected_rows(tmp_path):
    old_ids = ['ordinary', 'edge:fixed', 'edge:pending']
    old_source_hashes = ['sha-ordinary', 'sha-old-fixed', 'sha-old-pending']
    prior = _receipt('/old/train.json', 'old-manifest', old_ids, old_source_hashes)
    current_ids = ['ordinary', 'edge:fixed', 'new:append']
    current_source_hashes = ['sha-ordinary', 'sha-new-fixed', 'sha-new-row']
    current = _receipt('/new/train.json', 'new-manifest', current_ids, current_source_hashes)
    optimizer, rng, weights = object(), object(), object()
    frame = {'identity': {'stage_id': 'native_full_sft'}, 'source_receipts': [prior, {'path': '/held', 'sha256': 'held'}],
             'source_row_bindings': {'ordinary': {'h': 'o'}, 'edge:fixed': {'h': 'old'},
                                     'edge:pending': {'h': 'pending'}},
             'source_row_order': old_ids, 'exposure_counts': {'ordinary': 1, 'edge:fixed': 1, 'edge:pending': 1},
             'required_visits': {'ordinary': 1, 'edge:fixed': 1, 'edge:pending': 1},
             'completed_foundations': [], 'step': 73, 'global_step': 73,
             'optimizer': optimizer, 'rng': rng, 'weights': weights,
             'options': {'stage_id': 'native_full_sft', 'epochs': 1,
                         '_communication_ids': ['edge:fixed', 'edge:pending']},
             'sft_progress': {'native_full_sft': {'source_receipts': [prior, {'path': '/held', 'sha256': 'held'}],
                 'source_row_bindings': {'ordinary': {'h': 'o'}, 'edge:fixed': {'h': 'old'},
                                         'edge:pending': {'h': 'pending'}},
                 'source_row_order': old_ids, 'exposure_counts': {'ordinary': 1, 'edge:fixed': 1, 'edge:pending': 1},
                 'required_visits': {'ordinary': 1, 'edge:fixed': 1, 'edge:pending': 1},
                 'completed_foundations': ['old-foundation'], 'options': {'stage_id': 'native_full_sft', 'epochs': 1}}}}
    new_bindings = {'ordinary': {'h': 'o'}, 'edge:fixed': {'h': 'new'}, 'new:append': {'h': 'added'}}
    proof = _write_proof(tmp_path, prior, current, [
        _row('edge:fixed', {'h': 'old'}, {'h': 'new'}, 'sha-old-fixed', 'sha-new-fixed', 'corrected'),
        _row('edge:pending', {'h': 'pending'}, None, 'sha-old-pending', None, 'deferred', 'source remains under repair'),
        _row('new:append', None, {'h': 'added'}, None, 'sha-new-row', 'added'),
    ])
    result, summary = apply_source_correction(frame, proof, model_id='lfm2:inventory',
        stage_id='native_full_sft', current_source_receipt=current,
        current_bindings=new_bindings, current_order=current_ids,
        current_communication_ids=['edge:fixed'])

    assert summary == {'changed': True, 'proof_sha256': _sha(proof), 'corrected_rows': 1,
                       'deferred_rows': 1, 'added_rows': 1, 'protected_held_rows': 0}
    assert result['optimizer'] is optimizer and result['rng'] is rng and result['weights'] is weights
    assert result['global_step'] == 73
    assert result['source_row_bindings'] == new_bindings
    assert result['source_row_order'] == current_ids
    assert result['exposure_counts'] == {'ordinary': 1, 'edge:fixed': 0, 'edge:pending': 1, 'new:append': 0}
    assert result['required_visits']['edge:fixed'] == 1
    assert result['pending_source_rows']['edge:pending'] == {
        'binding': {'h': 'pending'}, 'source_sha256': 'sha-old-pending',
        'required_visits': 1, 'reason': 'source remains under repair'}
    assert result['completed_foundations'] == []
    assert result['sft_progress']['native_full_sft']['exposure_counts']['edge:fixed'] == 0


def test_corrected_rows_may_move_after_the_unchanged_earned_prefix(tmp_path):
    prior=_receipt('/old/train.json','old',['ordinary','edge','last'],
                   ['sha-ordinary','sha-old-edge','sha-last'])
    current=_receipt('/new/train.json','new',['ordinary','last','edge'],
                     ['sha-ordinary','sha-last','sha-new-edge'])
    bindings={'ordinary':{'h':'ordinary'},'last':{'h':'last'},'edge':{'h':'new'}}
    proof=_write_proof(tmp_path,prior,current,[
        _row('edge',{'h':'old'},{'h':'new'},'sha-old-edge','sha-new-edge','corrected')])
    training={'source_receipts':[prior,{'path':'/held','sha256':'held'}],
        'source_row_bindings':{'ordinary':{'h':'ordinary'},'edge':{'h':'old'},'last':{'h':'last'}},
        'source_row_order':['ordinary','edge','last'],
        'exposure_counts':{'ordinary':2,'edge':1,'last':3},
        'required_visits':{'ordinary':1,'edge':1,'last':1},'pending_source_rows':{},
        'options':{'epochs':1,'_communication_ids':['edge']},'step':7,'global_step':7,
        'optimizer':{'state':{'kept':True}},'rng':{'seed':4},
        'sft_progress':{'native_full_sft':{'completed_foundations':['old-foundation']}}}
    result,summary=apply_source_correction(training,proof,model_id='lfm2:inventory',
        stage_id='native_full_sft',current_source_receipt=current,current_bindings=bindings,
        current_order=['ordinary','last','edge'],current_communication_ids=['edge'])
    assert summary['corrected_rows']==1
    assert result['source_row_order']==['ordinary','last','edge']
    assert result['exposure_counts']=={'ordinary':2,'edge':0,'last':3}
    assert result['sft_progress']['native_full_sft']['completed_foundations'] == []
    assert result['options']['foundation_qualification_history'][0]['prior_completed_foundations'] == ['old-foundation']
    assert result['options']['source_correction_history'][0]['corrected_rows'][0]['prior_exposure'] == 1
    assert result['options']['source_correction_history'][0]['new_corrected_row_exposure_added'] == 0
    assert len(source_correction_dependencies({'path': str(proof), 'sha256': _sha(proof)})) == 2


def test_pending_row_can_be_corrected_later_without_losing_original_debt(tmp_path):
    old_id = 'edge:late'
    first_prior = _receipt('/old/all.json', 'old-all', ['ordinary', old_id], ['sha-o', 'sha-late'])
    first_current = _receipt('/new/partial.json', 'new-partial', ['ordinary'], ['sha-o'])
    training = {'source_receipts': [first_prior], 'source_row_bindings': {'ordinary': {'h': 'o'}, old_id: {'h': 'late'}},
        'source_row_order': ['ordinary', old_id], 'exposure_counts': {'ordinary': 1, old_id: 1},
        'required_visits': {'ordinary': 1, old_id: 2}, 'pending_source_rows': {}, 'completed_foundations': [],
        'step': 9, 'optimizer': object(), 'rng': object(), 'options': {'stage_id': 'native_full_sft', 'epochs': 1,
            '_communication_ids': [old_id]}, 'sft_progress': {}}
    first = _write_proof(tmp_path, first_prior, first_current, [
        _row(old_id, {'h': 'late'}, None, 'sha-late', None, 'deferred', 'awaiting corrected producer graph')],
        model_id='mellum:inventory')
    training, _ = apply_source_correction(training, first, model_id='mellum:inventory', stage_id='native_full_sft',
        current_source_receipt=first_current, current_bindings={'ordinary': {'h': 'o'}}, current_order=['ordinary'],
        current_communication_ids=[])
    second_current = _receipt('/new/complete.json', 'new-complete', ['ordinary', old_id], ['sha-o', 'sha-late-corrected'])
    second = _write_proof(tmp_path, first_current, second_current, [
        _row(old_id, {'h': 'late'}, {'h': 'corrected'}, 'sha-late', 'sha-late-corrected', 'corrected')],
        model_id='mellum:inventory')
    result, summary = apply_source_correction(training, second, model_id='mellum:inventory', stage_id='native_full_sft',
        current_source_receipt=second_current,
        current_bindings={'ordinary': {'h': 'o'}, old_id: {'h': 'corrected'}}, current_order=['ordinary', old_id],
        current_communication_ids=[old_id])
    assert summary['corrected_rows'] == 1
    assert result['pending_source_rows'] == {}
    assert result['source_row_bindings'][old_id] == {'h': 'corrected'}
    assert result['exposure_counts'][old_id] == 0
    assert result['required_visits'][old_id] == 1
    assert result['options']['source_correction_history'][0]['deferred_rows'] == [old_id]


@pytest.mark.parametrize('mutate', ['ordinary-change', 'missing-proof-row', 'reordered'])
def test_source_correction_rejects_unproved_changes(tmp_path, mutate):
    prior = _receipt('/old/train.json', 'old', ['keep', 'edge'], ['sha-keep', 'sha-edge'])
    current = _receipt('/new/train.json', 'new', ['keep', 'edge'], ['sha-keep', 'sha-edge-new'])
    bindings = {'keep': {'v': 1}, 'edge': {'v': 2}}
    row = _row('edge', {'v': 1}, {'v': 2}, 'sha-edge', 'sha-edge-new', 'corrected')
    proof = _write_proof(tmp_path, prior, current, [row])
    if mutate == 'ordinary-change':
        bindings['keep'] = {'v': 9}
    if mutate == 'missing-proof-row':
        bindings.pop('edge')
    order = ['keep', 'edge'] if mutate != 'reordered' else ['edge', 'keep']
    training = {'source_receipts': [prior], 'source_row_bindings': {'keep': {'v': 1}, 'edge': {'v': 1}},
        'source_row_order': ['keep', 'edge'], 'exposure_counts': {}, 'required_visits': {},
        'step': 1, 'options': {'stage_id': 'native_full_sft', 'epochs': 1, '_communication_ids': ['edge']}}
    with pytest.raises(ValueError):
        apply_source_correction(training, proof, model_id='lfm2:inventory', stage_id='native_full_sft',
            current_source_receipt=current, current_bindings=bindings, current_order=order,
            current_communication_ids=['edge'])


def test_source_correction_requires_every_new_row_in_the_proof_ledger(tmp_path):
    prior = _receipt('/old/train.json', 'old', ['edge'], ['sha-edge'])
    current = _receipt('/new/train.json', 'new', ['edge', 'unreviewed'], ['sha-edge-new', 'sha-unreviewed'])
    proof = _write_proof(tmp_path, prior, current, [
        _row('edge', {'h': 'old'}, {'h': 'new'}, 'sha-edge', 'sha-edge-new', 'corrected')])
    training = {'source_receipts': [prior], 'source_row_bindings': {'edge': {'h': 'old'}},
        'source_row_order': ['edge'], 'exposure_counts': {'edge': 1}, 'required_visits': {'edge': 1},
        'options': {'epochs': 1, '_communication_ids': ['edge']}}
    with pytest.raises(ValueError, match='exactly cover'):
        apply_source_correction(training, proof, model_id='lfm2:inventory', stage_id='native_full_sft',
            current_source_receipt=current,
            current_bindings={'edge': {'h': 'new'}, 'unreviewed': {'h': 'unreviewed'}},
            current_order=['edge', 'unreviewed'], current_communication_ids=['edge'])


def test_source_correction_rejects_replacement_that_is_no_longer_a_communication_graph(tmp_path):
    prior = _receipt('/old/train.json', 'old', ['edge'], ['sha-edge'])
    current = _receipt('/new/train.json', 'new', ['edge'], ['sha-edge-new'])
    proof = _write_proof(tmp_path, prior, current, [
        _row('edge', {'h': 'old'}, {'h': 'new'}, 'sha-edge', 'sha-edge-new', 'corrected')])
    training = {'source_receipts': [prior], 'source_row_bindings': {'edge': {'h': 'old'}},
        'source_row_order': ['edge'], 'exposure_counts': {'edge': 1}, 'required_visits': {'edge': 1},
        'options': {'epochs': 1, '_communication_ids': ['edge']}}
    with pytest.raises(ValueError, match='communication row'):
        apply_source_correction(training, proof, model_id='lfm2:inventory', stage_id='native_full_sft',
            current_source_receipt=current, current_bindings={'edge': {'h': 'new'}},
            current_order=['edge'], current_communication_ids=[])


def test_protected_held_correction_keeps_membership_groups_and_exposure_unchanged(tmp_path):
    prior_train = _receipt('/old/train.json', 'old-train', ['edge'], ['old-train-leaf'])
    current_train = _receipt('/new/train.json', 'new-train', ['edge'], ['new-train-leaf'])
    old_held = _receipt('/old/held.json', 'old-held', ['held:a', 'held:b'],
                        ['old-held-a', 'old-held-b'])
    new_held = _receipt('/new/held.json', 'new-held', ['held:a', 'held:b'],
                        ['new-held-a', 'old-held-b'])
    bindings = {'edge': {'sha': 'new-train'}}
    held_bindings = {'held:a': {'sha': 'new-held'}, 'held:b': {'sha': 'old-held-b'}}
    protected = {'prior': old_held, 'current': new_held, 'rows': [
        {'id': 'held:a', 'old_binding': {'sha': 'old-held'}, 'new_binding': {'sha': 'new-held'},
         'old_source_sha256': 'old-held-a', 'new_source_sha256': 'new-held-a',
         'old_groups': ['world:held-a'], 'new_groups': ['world:held-a'], 'disposition': 'corrected'}]}
    proof = _write_proof(tmp_path, prior_train, current_train,
        [_row('edge', {'sha': 'old-train'}, {'sha': 'new-train'}, 'old-train-leaf', 'new-train-leaf', 'corrected')],
        protected_held=protected)
    proof_doc = json.loads(proof.read_text())
    proof_doc['row_bindings_path'] = 'rows.jsonl'
    proof_doc['protected_held_correction']['row_bindings_path'] = 'held-rows.jsonl'
    evidence_dir = tmp_path / 'evidence'
    evidence_dir.mkdir()
    evidence_file = evidence_dir / 'provenance.json'
    evidence_file.write_text('{"authenticated":true}')
    proof_doc['evidence'] = [{'path': 'evidence/provenance.json', 'sha256': _sha(evidence_file)}]
    proof.write_text(json.dumps(proof_doc, sort_keys=True))
    training = {'source_receipts': [prior_train, old_held],
        'source_row_bindings': {'edge': {'sha': 'old-train'}}, 'source_row_order': ['edge'],
        'held_row_bindings': {'held:a': {'sha': 'old-held'}, 'held:b': {'sha': 'old-held-b'}},
        'exposure_counts': {'edge': 1}, 'required_visits': {'edge': 1}, 'pending_source_rows': {},
        'options': {'epochs': 1.5, '_communication_ids': ['edge']},
        'sft_progress': {'native_full_sft': {'source_receipts': [prior_train, old_held],
            'source_row_bindings': {'edge': {'sha': 'old-train'}}, 'source_row_order': ['edge'],
            'held_row_bindings': {'held:a': {'sha': 'old-held'}, 'held:b': {'sha': 'old-held-b'}},
            'exposure_counts': {'edge': 1}, 'required_visits': {'edge': 1}, 'options': {}}}}
    held_context = {'prior_receipt': old_held, 'current_receipt': new_held,
        'prior_bindings': training['held_row_bindings'], 'current_bindings': held_bindings,
        'prior_order': ['held:a', 'held:b'], 'current_order': ['held:a', 'held:b'],
        'prior_groups': {'world:held-a', 'world:held-b'}, 'current_groups': {'world:held-a', 'world:held-b'},
        'prior_groups_by_id': {'held:a': ['world:held-a'], 'held:b': ['world:held-b']},
        'current_groups_by_id': {'held:a': ['world:held-a'], 'held:b': ['world:held-b']}}
    result, summary = apply_source_correction(training, proof, model_id='lfm2:inventory',
        stage_id='native_full_sft', current_source_receipt=current_train,
        current_bindings=bindings, current_order=['edge'], current_communication_ids=['edge'],
        protected_held=held_context)
    assert summary['protected_held_rows'] == 1
    assert result['held_row_bindings'] == held_bindings
    assert result['source_receipts'][1] == new_held
    assert result['sft_progress']['native_full_sft']['held_row_bindings'] == held_bindings
    assert result['exposure_counts'] == {'edge': 0}
    assert result['required_visits']['edge'] == 2
    assert len(source_correction_dependencies({'path': str(proof), 'sha256': _sha(proof)})) == 4


def test_protected_held_coverage_rebinds_only_after_exact_same_class_rows():
    from natlang_neuralese.train.native_held import rebind_held_coverage_after_exact_correction
    previous=[{'id':'h1','binding':'old','communication':True},
              {'id':'h2','binding':'same','communication':False}]
    current=[{'id':'h1','binding':'new','communication':True},
             {'id':'h2','binding':'same','communication':False}]
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    coverage={'source_sha256':'old-source','classes':{
        'communication':{'rows':1,'row_prefix_sha256':canonical_json_sha256_hex([['h1','old']]),
                         'visits':[4],'cursor':0},
        'ordinary':{'rows':1,'row_prefix_sha256':canonical_json_sha256_hex([['h2','same']]),
                    'visits':[2],'cursor':0}}}
    rebound=rebind_held_coverage_after_exact_correction(previous,current,coverage,source_sha256='new-source')
    assert rebound['source_sha256']=='new-source'
    assert rebound['classes']['communication']['visits']==[4]
    assert rebound['classes']['communication']['row_prefix_sha256']==canonical_json_sha256_hex([['h1','new']])
    moved=[{'id':'h1','binding':'new','communication':False},
           {'id':'h2','binding':'same','communication':False}]
    with pytest.raises(ValueError,match='class membership/order'):
        rebind_held_coverage_after_exact_correction(previous,moved,coverage,source_sha256='new-source')


def test_source_correction_uses_fractional_epoch_schedule_for_fresh_rows(tmp_path):
    prior = _receipt('/old/train.json', 'old', ['ordinary', 'edge'], ['sha-o', 'sha-edge'])
    current = _receipt('/new/train.json', 'new', ['ordinary', 'edge', 'added'],
                       ['sha-o', 'sha-edge-new', 'sha-added'])
    proof = _write_proof(tmp_path, prior, current, [
        _row('edge', {'h': 'old'}, {'h': 'new'}, 'sha-edge', 'sha-edge-new', 'corrected'),
        _row('added', None, {'h': 'added'}, None, 'sha-added', 'added')])
    training = {'source_receipts': [prior], 'source_row_bindings': {'ordinary': {'h': 'o'}, 'edge': {'h': 'old'}},
        'source_row_order': ['ordinary', 'edge'], 'exposure_counts': {'ordinary': 1, 'edge': 1},
        'required_visits': {'ordinary': 1, 'edge': 1}, 'completed_foundations': [], 'options': {
            'epochs': 1.5, '_communication_ids': ['edge']}}
    result, _ = apply_source_correction(training, proof, model_id='lfm2:inventory', stage_id='native_full_sft',
        current_source_receipt=current, current_bindings={'ordinary': {'h': 'o'}, 'edge': {'h': 'new'},
            'added': {'h': 'added'}}, current_order=['ordinary', 'edge', 'added'],
        current_communication_ids=['edge'])
    # 1.5 epochs over three ordered rows gives two visits to the first two,
    # one to the final row; the communication row also has its one-visit floor.
    assert result['required_visits'] == {'ordinary': 1, 'edge': 2, 'added': 1}


def test_active_recipe_stage_accepts_only_exact_source_correction_handoff(tmp_path):
    from natlang_neuralese.train.recipe import _source_correction_continuation
    prior = _receipt('/inputs/old.json', 'old-manifest', ['edge'], ['old-leaf'])
    current = _receipt('/inputs/new.json', 'new-manifest', ['edge'], ['new-leaf'])
    proof = _write_proof(tmp_path, prior, current, [
        _row('edge', {'h': 'old'}, {'h': 'new'}, 'old-leaf', 'new-leaf', 'corrected')])
    old = {'native_graphs': {'path': prior['path'], 'sha256': prior['sha256'], 'binding': 'native_graphs'},
        'held_native_graphs': {'path': '/inputs/held.json', 'sha256': 'held', 'binding': 'held_native_graphs'}}
    new = {'native_graphs': {'path': current['path'], 'sha256': current['sha256'], 'binding': 'native_graphs'},
        'continued_native_graphs': {'path': prior['path'], 'sha256': prior['sha256'], 'binding': 'continued_native_graphs'},
        'held_native_graphs': old['held_native_graphs'],
        'source_correction': {'path': str(proof), 'sha256': _sha(proof), 'binding': 'source_correction'}}
    assert _source_correction_continuation(old, new)
    altered = dict(new, held_native_graphs={'path': '/inputs/other-held.json',
        'sha256': 'changed', 'binding': 'held_native_graphs'})
    assert not _source_correction_continuation(old, altered)
