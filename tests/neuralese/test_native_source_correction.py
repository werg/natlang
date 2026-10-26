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


def _write_proof(tmp_path, prior, current, rows, model_id='lfm2:inventory'):
    row_path = tmp_path / 'rows.jsonl'
    row_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    proof_path = tmp_path / 'source-correction.json'
    proof = {'schema': SCHEMA, 'model_id': model_id, 'stage_id': 'native_full_sft',
             'prior_source': {'path': prior['path'], 'sha256': prior['sha256']},
             'current_source': {'path': current['path'], 'sha256': current['sha256']},
             'row_bindings_path': str(row_path), 'row_bindings_sha256': _sha(row_path)}
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
                       'deferred_rows': 1, 'added_rows': 1}
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
    assert result['sft_progress']['native_full_sft']['completed_foundations'] == []
    assert result['options']['foundation_qualification_history'][0]['prior_completed_foundations'] == ['old-foundation']
    assert result['options']['source_correction_history'][0]['corrected_rows'][0]['prior_exposure'] == 1
    assert result['options']['source_correction_history'][0]['new_corrected_row_exposure_added'] == 0
    assert result['options']['source_correction_history'][0]['added_rows'] == ['new:append']
    assert len(source_correction_dependencies({'path': str(proof), 'sha256': _sha(proof)})) == 2
    replay, replay_summary = apply_source_correction(result, proof, model_id='lfm2:inventory',
        stage_id='native_full_sft', current_source_receipt=current,
        current_bindings=new_bindings, current_order=current_ids,
        current_communication_ids=['edge:fixed'])
    assert replay is result and replay_summary['changed'] is False
    with pytest.raises(ValueError, match='replay differs'):
        apply_source_correction(result, proof, model_id='lfm2:inventory',
            stage_id='native_full_sft', current_source_receipt=prior,
            current_bindings=new_bindings, current_order=current_ids,
            current_communication_ids=['edge:fixed'])


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
