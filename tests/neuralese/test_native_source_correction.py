import hashlib
import json
import copy
from pathlib import Path

import pytest

from natlang_neuralese.train.native_source_correction import (
    SCHEMA, apply_source_correction, source_correction_dependencies,
    validate_held_quality_hold,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _receipt(path, digest, ids, source_hashes):
    sources = []
    for row_id, source_sha in zip(ids, source_hashes):
        sources.append({'path': f'/src/{source_sha}.jsonl.zst', 'sha256': source_sha,
                        'rows': 1, 'id': row_id})
    return {'path': str(path), 'sha256': digest, 'rows': len(ids), 'sources': sources}


def _write_proof(tmp_path, prior, current, rows, model_id='lfm2:inventory', protected_held=None,
                 name='source-correction'):
    row_path = tmp_path / (name + '-rows.jsonl')
    row_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    proof_path = tmp_path / (name + '.json')
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


def _row(row_id, old_binding, new_binding, old_sha, new_sha, disposition, reason=None,
         old_communication=None):
    row = {'id': row_id, 'old_binding': old_binding, 'new_binding': new_binding,
            'old_source_sha256': old_sha, 'new_source_sha256': new_sha,
            'disposition': disposition, 'reason': reason}
    if old_communication is not None:
        row['old_communication'] = old_communication
    return row


def test_prior_receipt_path_change_requires_exact_authenticated_source_map(tmp_path):
    from natlang_neuralese.train.native_source_correction import _receipt_matches_with_staged_relocation
    staged=tmp_path/'staged-held.json';staged.write_bytes(b'authenticated held manifest bytes')
    source_map=tmp_path/'source-map.json'
    source_map.write_text(json.dumps({'schema':'natlang.staged-source-map/1',
        'paths':{'/runtime/held.json':str(staged.resolve())}},sort_keys=True))
    proof_path=tmp_path/'source-correction.json'
    proof={'source_map':{'path':source_map.name,'sha256':_sha(source_map)},
        'prior_held_relocation':{'schema':'natlang.staged-receipt-relocation/1',
            'logical_path':'/runtime/held.json','staged_path':str(staged.resolve()),
            'sha256':_sha(staged),'source_map_path':str(source_map.resolve()),
            'source_map_sha256':_sha(source_map)}}
    actual={'path':'/runtime/held.json','sha256':_sha(staged)}
    expected={'path':str(staged),'sha256':_sha(staged)}
    assert _receipt_matches_with_staged_relocation(expected,actual,proof,proof_path)
    assert _receipt_matches_with_staged_relocation(actual,expected,proof,proof_path)
    assert not _receipt_matches_with_staged_relocation(expected,
        {'path':'/runtime/other-held.json','sha256':_sha(staged)},proof,proof_path)
    assert not _receipt_matches_with_staged_relocation(expected,
        {'path':'/runtime/held.json','sha256':'0'*64},proof,proof_path)


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
        'required_visits': 1, 'reason': 'source remains under repair',
        'communication': True,
        'classification_provenance': {
            'schema': 'natlang.pending-row-classification-provenance/1',
            'row_id': 'edge:pending', 'model_id': 'lfm2:inventory',
            'stage_id': 'native_full_sft',
            'classification_source': 'saved_checkpoint_communication_ids',
            'proof_path': str(proof.resolve()), 'proof_sha256': _sha(proof),
            'source_receipt': {'path': str(Path(prior['path']).resolve()), 'sha256': prior['sha256']}},
        'retired_exposure': {'visits': 1, 'required_visits': 1,
            'source_receipt': {'path': str(Path(prior['path']).resolve()), 'sha256': prior['sha256']}}}
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
    import torch

    old_id = 'edge:late'
    first_prior = _receipt('/old/all.json', 'old-all', ['ordinary', old_id], ['sha-o', 'sha-late'])
    first_current = _receipt('/new/partial.json', 'new-partial', ['ordinary'], ['sha-o'])
    optimizer = {'state': {'exp_avg': torch.tensor([1.0, 2.0])}, 'step': 17}
    rng = {'cpu': torch.arange(8, dtype=torch.uint8)}
    weights = {'w': torch.tensor([3.0, 4.0])}
    same_source_carried = {'source_receipts': [first_prior],
        'source_row_bindings': {'ordinary': {'h': 'o'}, old_id: {'h': 'late'}},
        'source_row_order': ['ordinary', old_id], 'exposure_counts': {'ordinary': 1, old_id: 1},
        'required_visits': {'ordinary': 1, old_id: 2}, 'pending_source_rows': {},
        'options': {'epochs': 1, '_communication_ids': [old_id]},
        'metrics': {'held': {'loss': 1.25}, 'marker': 'carried'} }
    unrelated_dpo = {'source_receipts': [_receipt('/unrelated/dpo.json', 'dpo-v1', ['pair'], ['pair-sha'])],
        'source_row_bindings': {'pair': {'h': 'pair'}}, 'source_row_order': ['pair'],
        'exposure_counts': {'pair': 4}, 'required_visits': {'pair': 6},
        'options': {'stage_id': 'native_dpo', 'objective': {'beta': 0.2}, 'sentinel': {'bytes': 'unchanged'}},
        'metrics': {'held': {'loss': 9.5}, 'sentinel': 'unchanged'} }
    training = {'source_receipts': [first_prior], 'source_row_bindings': {'ordinary': {'h': 'o'}, old_id: {'h': 'late'}},
        'source_row_order': ['ordinary', old_id], 'exposure_counts': {'ordinary': 1, old_id: 1},
        'required_visits': {'ordinary': 1, old_id: 2}, 'pending_source_rows': {}, 'completed_foundations': [],
        'step': 9, 'global_step': 9, 'optimizer': optimizer, 'rng': rng, 'weights': weights,
        'options': {'stage_id': 'native_full_sft', 'epochs': 1, '_communication_ids': [old_id]},
        'sft_progress': {'native_full_sft': same_source_carried,
            'native_supervised-carried':copy.deepcopy(same_source_carried), 'native_dpo': unrelated_dpo}}
    first = _write_proof(tmp_path, first_prior, first_current, [
        _row(old_id, {'h': 'late'}, None, 'sha-late', None, 'deferred', 'awaiting corrected producer graph')],
        model_id='mellum:inventory')
    training, first_summary = apply_source_correction(training, first, model_id='mellum:inventory', stage_id='native_full_sft',
        current_source_receipt=first_current, current_bindings={'ordinary': {'h': 'o'}}, current_order=['ordinary'],
        current_communication_ids=[])
    assert first_summary['deferred_rows'] == 1
    assert training['pending_source_rows'][old_id]['communication'] is True
    assert training['pending_source_rows'][old_id]['retired_exposure'] == {
        'visits': 1, 'required_visits': 2, 'source_receipt': {'path': first_prior['path'], 'sha256': first_prior['sha256']}}
    assert training['sft_progress']['native_supervised-carried']['pending_source_rows'][old_id]['communication'] is True
    assert training['sft_progress']['native_dpo'] == unrelated_dpo

    # Exercise the exact durable boundary and normal resume's current-source
    # option rebuild, which drops the omitted row from _communication_ids.
    slot = tmp_path / 'saved-checkpoint.pt'
    torch.save(training, slot)
    training = torch.load(slot, map_location='cpu', weights_only=False)
    assert old_id not in training['options']['_communication_ids']
    assert old_id not in training['sft_progress']['native_supervised-carried']['options']['_communication_ids']
    before_numeric = {key: torch.clone(value) for key, value in training['weights'].items()}
    before_optimizer = torch.clone(training['optimizer']['state']['exp_avg'])
    before_rng = torch.clone(training['rng']['cpu'])
    second_current = _receipt('/new/complete.json', 'new-complete', ['ordinary', old_id], ['sha-o', 'sha-late-corrected'])
    second = _write_proof(tmp_path, first_current, second_current, [
        _row(old_id, {'h': 'late'}, {'h': 'corrected'}, 'sha-late', 'sha-late-corrected', 'corrected',
             old_communication=True)],
        model_id='mellum:inventory', name='second-correction')
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
    assert result['options']['source_correction_history'][1]['corrected_rows'][0]['prior_exposure'] == 1
    assert result['sft_progress']['native_supervised-carried']['pending_source_rows'] == {}
    assert result['sft_progress']['native_dpo'] == unrelated_dpo
    assert all(torch.equal(result['weights'][key], value) for key, value in before_numeric.items())
    assert torch.equal(result['optimizer']['state']['exp_avg'], before_optimizer)
    assert torch.equal(result['rng']['cpu'], before_rng)


def test_pending_ordinary_row_cannot_be_reclassified_as_communication(tmp_path):
    old_id = 'ordinary:late'
    prior = _receipt('/old/all.json', 'old-all', ['ordinary', old_id], ['sha-o', 'sha-late'])
    reduced = _receipt('/new/partial.json', 'new-partial', ['ordinary'], ['sha-o'])
    training = {'source_receipts':[prior], 'source_row_bindings':{'ordinary':{'h':'o'}, old_id:{'h':'late'}},
        'source_row_order':['ordinary',old_id], 'exposure_counts':{'ordinary':1,old_id:2},
        'required_visits':{'ordinary':1,old_id:3}, 'pending_source_rows':{},
        'options':{'stage_id':'native_full_sft','epochs':1,'_communication_ids':[]}, 'sft_progress':{}}
    first = _write_proof(tmp_path, prior, reduced, [
        _row(old_id, {'h':'late'}, None, 'sha-late', None, 'deferred', 'pending ordinary source')])
    training,_=apply_source_correction(training,first,model_id='lfm2:inventory',stage_id='native_full_sft',
        current_source_receipt=reduced,current_bindings={'ordinary':{'h':'o'}},current_order=['ordinary'],
        current_communication_ids=[])
    assert training['pending_source_rows'][old_id]['communication'] is False
    restored = _receipt('/new/repaired.json','repaired',['ordinary',old_id],['sha-o','sha-late-new'])
    second = _write_proof(tmp_path,reduced,restored,[
        _row(old_id,{'h':'late'},{'h':'new'},'sha-late','sha-late-new','corrected',
             old_communication=False)], name='second-ordinary-correction')
    with pytest.raises(ValueError,match='replacement is not an exact changed communication row'):
        apply_source_correction(training,second,model_id='lfm2:inventory',stage_id='native_full_sft',
            current_source_receipt=restored,current_bindings={'ordinary':{'h':'o'},old_id:{'h':'new'}},
            current_order=['ordinary',old_id],current_communication_ids=[old_id])


def test_pending_communication_class_survives_source_append_and_redeferral(tmp_path):
    import torch

    edge = 'edge:late'
    source_a = _receipt('/a/train.json', 'source-a', ['ordinary', edge], ['sha-o', 'sha-edge'])
    source_b = _receipt('/b/train.json', 'source-b', ['ordinary'], ['sha-o'])
    training = {'source_receipts':[source_a],
        'source_row_bindings':{'ordinary':{'h':'o'},edge:{'h':'edge'}},
        'source_row_order':['ordinary',edge], 'exposure_counts':{'ordinary':2,edge:3},
        'required_visits':{'ordinary':4,edge:5}, 'pending_source_rows':{},
        'options':{'stage_id':'native_full_sft','epochs':1,'_communication_ids':[edge]},
        'sft_progress':{'native_full_sft':{'source_receipts':[source_a],
            'source_row_bindings':{'ordinary':{'h':'o'},edge:{'h':'edge'}},
            'source_row_order':['ordinary',edge], 'exposure_counts':{'ordinary':2,edge:3},
            'required_visits':{'ordinary':4,edge:5},
            'options':{'stage_id':'native_full_sft','epochs':1,'_communication_ids':[edge]}}},
        'optimizer':{'state':{'moment':torch.tensor([1.25])}},
        'rng':{'cpu':torch.arange(4,dtype=torch.uint8)},'weights':{'w':torch.tensor([2.5])},
        'global_step':11}
    proof1 = _write_proof(tmp_path, source_a, source_b,[
        _row(edge,{'h':'edge'},None,'sha-edge',None,'deferred','source still under repair',True)],
        model_id='mellum:inventory',name='defer-one')
    training,_=apply_source_correction(training,proof1,model_id='mellum:inventory',
        stage_id='native_full_sft',current_source_receipt=source_b,
        current_bindings={'ordinary':{'h':'o'}},current_order=['ordinary'],current_communication_ids=[])
    original_pending=copy.deepcopy(training['pending_source_rows'][edge])
    source_c = _receipt('/c/train.json','source-c',['ordinary','append:1'],['sha-o','sha-append'])
    proof2 = _write_proof(tmp_path,source_b,source_c,[
        _row(edge,{'h':'edge'},None,'sha-edge',None,'deferred','source still under repair',True),
        _row('append:1',None,{'h':'append'},None,'sha-append','added')],
        model_id='mellum:inventory',name='append-and-redefer')
    tampered=copy.deepcopy(training)
    tampered['pending_source_rows'][edge]['communication']=False
    with pytest.raises(ValueError,match='class differs from its saved deferral record'):
        apply_source_correction(tampered,proof2,model_id='mellum:inventory',
            stage_id='native_full_sft',current_source_receipt=source_c,
            current_bindings={'ordinary':{'h':'o'},'append:1':{'h':'append'}},
            current_order=['ordinary','append:1'],current_communication_ids=[])
    training,_=apply_source_correction(training,proof2,model_id='mellum:inventory',
        stage_id='native_full_sft',current_source_receipt=source_c,
        current_bindings={'ordinary':{'h':'o'},'append:1':{'h':'append'}},
        current_order=['ordinary','append:1'],current_communication_ids=[])
    assert training['pending_source_rows'][edge]==original_pending
    # A real serialization boundary preserves the proof lineage used by the
    # later correction, while the active communication list excludes the row.
    slot=tmp_path/'lineage.pt';torch.save(training,slot)
    training=torch.load(slot,map_location='cpu',weights_only=False)
    source_d = _receipt('/d/train.json','source-d',
        ['ordinary','append:1',edge],['sha-o','sha-append','sha-edge-fixed'])
    proof3 = _write_proof(tmp_path,source_c,source_d,[
        _row(edge,{'h':'edge'},{'h':'edge-fixed'},'sha-edge','sha-edge-fixed','corrected',
             old_communication=True)],model_id='mellum:inventory',name='reintroduce-after-append')
    result,summary=apply_source_correction(training,proof3,model_id='mellum:inventory',
        stage_id='native_full_sft',current_source_receipt=source_d,
        current_bindings={'ordinary':{'h':'o'},'append:1':{'h':'append'},edge:{'h':'edge-fixed'}},
        current_order=['ordinary','append:1',edge],current_communication_ids=[edge])
    assert summary['corrected_rows']==1
    assert result['pending_source_rows']=={}
    assert result['exposure_counts'][edge]==0
    assert result['required_visits'][edge]==1
    assert result['options']['_communication_ids']==[edge]
    assert len(result['options']['source_correction_history'])==3
    assert result['options']['source_correction_history'][-1]['corrected_rows'][0]['prior_exposure']==3
    assert torch.equal(result['optimizer']['state']['moment'],training['optimizer']['state']['moment'])
    assert torch.equal(result['rng']['cpu'],training['rng']['cpu'])
    assert torch.equal(result['weights']['w'],training['weights']['w'])


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


def test_source_correction_does_not_rebind_a_frame_from_another_source(tmp_path):
    from natlang_neuralese.train.native_continuation import raw_communication_rows

    prior = _receipt('/old/train.json', 'old', ['edge'], ['sha-edge'])
    current = _receipt('/new/train.json', 'new', ['edge'], ['sha-edge-new'])
    proof = _write_proof(tmp_path, prior, current, [
        _row('edge', {'h': 'old'}, {'h': 'new'}, 'sha-edge', 'sha-edge-new', 'corrected')])
    unrelated = _receipt('/unrelated/train.json', 'unrelated', ['edge'], ['sha-other'])
    old_frame = {'source_receipts': [prior], 'source_row_bindings': {'edge': {'h': 'old'}},
        'source_row_order': ['edge'], 'exposure_counts': {'edge': 1}, 'required_visits': {'edge': 1},
        'options': {'epochs': 1, '_communication_ids': ['edge']}}
    unrelated_frame = {**old_frame, 'source_receipts': [unrelated],
        'options': {'stage_id':'native_dpo','objective':{'beta':0.3},'marker':{'keep':True}},
        'metrics': {'held':{'loss':8.75},'task_eval':{'successes':2},'marker':'untouched'}}
    unrelated_before = copy.deepcopy(unrelated_frame)
    training = {'source_receipts': [prior], 'source_row_bindings': {'edge': {'h': 'old'}},
        'source_row_order': ['edge'], 'exposure_counts': {'edge': 1}, 'required_visits': {'edge': 1},
        'options': {'epochs': 1, '_communication_ids': ['edge']},
        'sft_progress': {'native_full_sft': old_frame, 'other-source-stage': unrelated_frame}}

    result, _ = apply_source_correction(training, proof, model_id='lfm2:inventory',
        stage_id='native_full_sft', current_source_receipt=current,
        current_bindings={'edge': {'h': 'new'}}, current_order=['edge'],
        current_communication_ids=['edge'])

    untouched = result['sft_progress']['other-source-stage']
    assert untouched == unrelated_before
    active_rows = raw_communication_rows(result['sft_progress'])
    assert set(active_rows) == {'edge'}
    assert active_rows['edge']['binding'] == {'h': 'new'}


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
    from natlang_neuralese.train.native_held import held_selection

    prior_train = _receipt('/old/train.json', 'old-train', ['edge'], ['old-train-leaf'])
    current_train = _receipt('/new/train.json', 'new-train', ['edge'], ['new-train-leaf'])
    old_held = _receipt('/old/held.json', 'old-held', ['held:a', 'held:b'],
                        ['old-held-a', 'old-held-b'])
    new_held = _receipt('/new/held.json', 'new-held', ['held:a', 'held:b'],
                        ['new-held-a', 'old-held-b'])
    bindings = {'edge': {'sha': 'new-train'}}
    held_bindings = {'held:a': {'sha': 'new-held'}, 'held:b': {'sha': 'old-held-b'}}
    prior_entries = [
        {'id':'held:a','binding':{'sha':'old-held'},'groups':['world:held-a'],
         'communication':False,'forward_tokens':3},
        {'id':'held:b','binding':{'sha':'old-held-b'},'groups':['world:held-b'],
         'communication':False,'forward_tokens':3},]
    current_entries = [
        {'id':'held:a','binding':{'sha':'new-held'},'groups':['world:held-a'],
         'communication':False,'forward_tokens':5},
        {'id':'held:b','binding':{'sha':'old-held-b'},'groups':['world:held-b'],
         'communication':False,'forward_tokens':3},]
    _, carried_coverage, _ = held_selection(prior_entries, source_sha256='old-held', final=True)
    assert carried_coverage['forward_token_visits'] == 6
    protected = {'prior': old_held, 'current': new_held, 'rows': [
        {'id': 'held:a', 'old_binding': {'sha': 'old-held'}, 'new_binding': {'sha': 'new-held'},
         'old_source_sha256': 'old-held-a', 'new_source_sha256': 'new-held-a',
         'old_groups': ['world:held-a'], 'new_groups': ['world:held-a'], 'disposition': 'corrected'}]}
    proof = _write_proof(tmp_path, prior_train, current_train,
        [_row('edge', {'sha': 'old-train'}, {'sha': 'new-train'}, 'old-train-leaf', 'new-train-leaf', 'corrected')],
        protected_held=protected)
    proof_doc = json.loads(proof.read_text())
    proof_doc['row_bindings_path'] = 'source-correction-rows.jsonl'
    proof_doc['protected_held_correction']['row_bindings_path'] = 'held-rows.jsonl'
    evidence_dir = tmp_path / 'evidence'
    evidence_dir.mkdir()
    evidence_file = evidence_dir / 'provenance.json'
    evidence_file.write_text('{"authenticated":true}')
    proof_doc['evidence'] = [{'path': 'evidence/provenance.json', 'sha256': _sha(evidence_file)}]
    proof.write_text(json.dumps(proof_doc, sort_keys=True))
    carried_frame = {'source_receipts':[prior_train,old_held],
        'source_row_bindings':{'edge':{'sha':'old-train'}},'source_row_order':['edge'],
        'held_row_bindings':{'held:a':{'sha':'old-held'},'held:b':{'sha':'old-held-b'}},
        'exposure_counts':{'edge':2},'required_visits':{'edge':3},
        'options':{'objective':'separate-but-same-source'},
        'metrics':{'held':{'loss':1.0},'held_coverage':carried_coverage}}
    training = {'source_receipts': [prior_train, old_held],
        'source_row_bindings': {'edge': {'sha': 'old-train'}}, 'source_row_order': ['edge'],
        'held_row_bindings': {'held:a': {'sha': 'old-held'}, 'held:b': {'sha': 'old-held-b'}},
        'exposure_counts': {'edge': 1}, 'required_visits': {'edge': 1}, 'pending_source_rows': {},
        'options': {'epochs': 1.5, '_communication_ids': ['edge']},
        'sft_progress': {'native_full_sft': {'source_receipts': [prior_train, old_held],
            'source_row_bindings': {'edge': {'sha': 'old-train'}}, 'source_row_order': ['edge'],
            'held_row_bindings': {'held:a': {'sha': 'old-held'}, 'held:b': {'sha': 'old-held-b'}},
            'exposure_counts': {'edge': 1}, 'required_visits': {'edge': 1}, 'options': {}},
            'carried-same-source': carried_frame}}
    held_context = {'prior_receipt': old_held, 'current_receipt': new_held,
        'prior_bindings': training['held_row_bindings'], 'current_bindings': held_bindings,
        'prior_order': ['held:a', 'held:b'], 'current_order': ['held:a', 'held:b'],
        'prior_groups': {'world:held-a', 'world:held-b'}, 'current_groups': {'world:held-a', 'world:held-b'},
        'prior_groups_by_id': {'held:a': ['world:held-a'], 'held:b': ['world:held-b']},
        'current_groups_by_id': {'held:a': ['world:held-a'], 'held:b': ['world:held-b']},
        'prior_entries':prior_entries, 'current_entries':current_entries}
    result, summary = apply_source_correction(training, proof, model_id='lfm2:inventory',
        stage_id='native_full_sft', current_source_receipt=current_train,
        current_bindings=bindings, current_order=['edge'], current_communication_ids=['edge'],
        protected_held=held_context)
    assert summary['protected_held_rows'] == 1
    assert result['held_row_bindings'] == held_bindings
    assert result['source_receipts'][1] == new_held
    assert result['sft_progress']['native_full_sft']['held_row_bindings'] == held_bindings
    rebound_coverage = result['sft_progress']['carried-same-source']['metrics']['held_coverage']
    assert rebound_coverage['source_sha256'] == 'new-held'
    assert rebound_coverage['classes']['ordinary']['visits'] == [0, 1]
    assert rebound_coverage['graph_visits'] == 1
    assert rebound_coverage['forward_token_visits'] == 3
    assert rebound_coverage['retired_target_visit_history'] == [{
        'id':'held:a','class':'ordinary','old_binding_sha256':
            __import__('natlang_neuralese.common.hashing',fromlist=['canonical_json_sha256_hex']).canonical_json_sha256_hex({'sha':'old-held'}),
        'visits':1,'forward_tokens_per_visit':3,'retired_forward_token_visits':3,
        'reason':'held target binding changed'}]
    assert result['exposure_counts'] == {'edge': 0}
    assert result['required_visits']['edge'] == 2
    assert len(source_correction_dependencies({'path': str(proof), 'sha256': _sha(proof)})) == 4


def test_held_quality_hold_authenticates_exact_pending_rows_and_is_carried_in_checkpoint(tmp_path):
    prior_train = _receipt('/old/train.json', 'old-train', ['edge'], ['a'*64])
    current_train = _receipt('/new/train.json', 'new-train', ['edge'], ['b'*64])
    held = _receipt('/new/held.json', 'held-current', ['held:bad', 'held:good'], ['c'*64, 'd'*64])
    entries = [
        {'id':'held:bad','binding':{'graph':'bad'},'groups':['world:a'],'communication':False},
        {'id':'held:good','binding':{'graph':'good'},'groups':['world:b'],'communication':False},
    ]
    rows_path=tmp_path/'quality-rows.jsonl'
    rows_path.write_text(json.dumps({'id':'held:bad','binding':{'graph':'bad'},'source_sha256':'c'*64,
        'reason':'truncated producer target','groups':['world:a']},sort_keys=True)+'\n')
    hold_path=tmp_path/'held-quality-hold.json'
    hold_path.write_text(json.dumps({'schema':'natlang.native-held-quality-hold/1',
        'held_source':held,'row_bindings_path':rows_path.name,'row_bindings_sha256':_sha(rows_path),
        'excluded_rows':1,'pending_sources':[{'sha256':'c'*64,'rows':1,
        'reason':'truncated producer target'}]},sort_keys=True))
    reference={'path':str(hold_path),'sha256':_sha(hold_path)}
    quality=validate_held_quality_hold(reference,held_receipt=held,held_entries=entries)
    assert quality['excluded_ids']=={'held:bad'}
    bad_entries=[dict(entries[0],groups=['world:wrong']),entries[1]]
    with pytest.raises(ValueError,match='exact source/binding/groups'):
        validate_held_quality_hold(reference,held_receipt=held,held_entries=bad_entries)

    correction_rows=tmp_path/'rows.jsonl'
    correction_rows.write_text(json.dumps(_row('edge',{'graph':'old'},{'graph':'new'},
        'a'*64,'b'*64,'corrected'),sort_keys=True)+'\n')
    held_rows=tmp_path/'held-rows.jsonl';held_rows.write_text('')
    proof_path=tmp_path/'source-correction.json'
    proof={'schema':SCHEMA,'model_id':'lfm2:inventory','stage_id':'native_full_sft',
        'prior_source':{'path':prior_train['path'],'sha256':prior_train['sha256']},
        'current_source':{'path':current_train['path'],'sha256':current_train['sha256']},
        'row_bindings_path':correction_rows.name,'row_bindings_sha256':_sha(correction_rows),
        'protected_held_correction':{'schema':'natlang.native-protected-held-correction/1',
            'prior_source':{'path':held['path'],'sha256':held['sha256']},
            'current_source':{'path':held['path'],'sha256':held['sha256']},
            'row_bindings_path':held_rows.name,'row_bindings_sha256':_sha(held_rows)},
        'held_quality_hold':{'path':hold_path.name,'sha256':_sha(hold_path)},
        'pending_held_correction':{'remaining_original_sources':[{'source':{'sha256':'c'*64},
            'rows':1,'reason':'truncated producer target'}]}}
    proof_path.write_text(json.dumps(proof,sort_keys=True))
    training={'source_receipts':[prior_train,held],
        'source_row_bindings':{'edge':{'graph':'old'}},'source_row_order':['edge'],
        'held_row_bindings':{'held:bad':{'graph':'bad'},'held:good':{'graph':'good'}},
        'exposure_counts':{'edge':1},'required_visits':{'edge':1},'pending_source_rows':{},
        'options':{'epochs':1,'_communication_ids':['edge']},
        'sft_progress':{'native_full_sft':{'source_receipts':[prior_train,held],
            'source_row_bindings':{'edge':{'graph':'old'}},'source_row_order':['edge'],
            'held_row_bindings':{'held:bad':{'graph':'bad'},'held:good':{'graph':'good'}},
            'exposure_counts':{'edge':1},'required_visits':{'edge':1},'options':{}}}}
    held_context={'prior_receipt':held,'current_receipt':held,
        'prior_bindings':training['held_row_bindings'],'current_bindings':training['held_row_bindings'],
        'prior_order':['held:bad','held:good'],'current_order':['held:bad','held:good'],
        'prior_groups':{'world:a','world:b'},'current_groups':{'world:a','world:b'},
        'prior_groups_by_id':{'held:bad':['world:a'],'held:good':['world:b']},
        'current_groups_by_id':{'held:bad':['world:a'],'held:good':['world:b']},
        'prior_entries':entries,'current_entries':entries}
    result,_=apply_source_correction(training,proof_path,model_id='lfm2:inventory',
        stage_id='native_full_sft',current_source_receipt=current_train,
        current_bindings={'edge':{'graph':'new'}},current_order=['edge'],
        current_communication_ids=['edge'],protected_held=held_context)
    binding=result['options']['held_quality_hold_binding']
    assert binding['excluded_rows']==1 and binding['source_sha256']=='held-current'
    assert result['sft_progress']['native_full_sft']['options']['held_quality_hold_binding']==binding


def test_protected_held_coverage_rebinds_only_after_exact_same_class_rows():
    from natlang_neuralese.train.native_held import rebind_held_coverage_after_exact_correction
    previous=[{'id':'h1','binding':'old','communication':True,'forward_tokens':3},
              {'id':'h2','binding':'same','communication':False,'forward_tokens':2}]
    current=[{'id':'h1','binding':'new','communication':True,'forward_tokens':3},
             {'id':'h2','binding':'same','communication':False,'forward_tokens':2}]
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    coverage={'source_sha256':'old-source','graph_visits':6,'completed_sweeps':2,
        'forward_token_visits':16,'classes':{
        'communication':{'rows':1,'row_prefix_sha256':canonical_json_sha256_hex([['h1','old']]),
                         'visits':[4],'cursor':0},
        'ordinary':{'rows':1,'row_prefix_sha256':canonical_json_sha256_hex([['h2','same']]),
                    'visits':[2],'cursor':0}}}
    rebound=rebind_held_coverage_after_exact_correction(previous,current,coverage,source_sha256='new-source')
    assert rebound['source_sha256']=='new-source'
    assert rebound['classes']['communication']['visits']==[0]
    assert rebound['classes']['communication']['row_prefix_sha256']==canonical_json_sha256_hex([['h1','new']])
    assert rebound['classes']['ordinary']['visits']==[2]
    assert rebound['graph_visits']==2 and rebound['completed_sweeps']==0
    assert rebound['forward_token_visits']==4
    assert rebound['retired_target_visit_history']==[{'id':'h1','class':'communication',
        'old_binding_sha256':canonical_json_sha256_hex('old'),'visits':4,
        'forward_tokens_per_visit':3,'retired_forward_token_visits':12,
        'reason':'held target binding changed'}]
    moved=[{'id':'h1','binding':'new','communication':False},
           {'id':'h2','binding':'same','communication':False}]
    with pytest.raises(ValueError,match='class membership/order'):
        rebind_held_coverage_after_exact_correction(previous,moved,coverage,source_sha256='new-source')


def test_carried_held_correction_accepts_only_exact_authenticated_source_relocation(tmp_path):
    from natlang_neuralese.train.native_source_correction import _propagate_source_correction_to_carried_frames
    from natlang_neuralese.train.native_held import held_selection

    staged=tmp_path/'staged-held.jsonl';staged.write_bytes(b'held\n')
    logical='/work/native-production/data/held.jsonl'
    source_map=tmp_path/'source-map.json'
    source_map.write_text(json.dumps({'schema':'natlang.staged-source-map/1',
        'paths':{logical:str(staged.resolve())}},sort_keys=True))
    held_sha=_sha(staged)
    map_sha=_sha(source_map)
    proof=tmp_path/'correction-with-relocation.json'
    proof.write_text(json.dumps({'schema':SCHEMA,
        'source_map':{'path':str(source_map),'sha256':map_sha},
        'prior_held_relocation':{'schema':'natlang.staged-receipt-relocation/1',
            'logical_path':logical,'staged_path':str(staged.resolve()),
            'sha256':held_sha,'source_map_path':str(source_map.resolve()),
            'source_map_sha256':map_sha}},sort_keys=True))
    prior_train=_receipt('/old/train.json','train-old',['edge'],['sha-edge'])
    current_train=_receipt('/new/train.json','train-new',['edge'],['sha-edge-new'])
    prior_held={'path':str(staged.resolve()),'sha256':held_sha,'rows':1,
        'sources':[{'path':str(staged.resolve()),'sha256':held_sha,'rows':1,'id':'held:a'}]}
    current_held=_receipt('/new/held.json','held-new',['held:a'],['sha-held-new'])
    prior_entries=[{'id':'held:a','binding':{'target':'old'},'groups':['world:a'],
        'communication':False,'forward_tokens':3}]
    current_entries=[{'id':'held:a','binding':{'target':'new'},'groups':['world:a'],
        'communication':False,'forward_tokens':3}]
    _,coverage,_=held_selection(prior_entries,source_sha256=held_sha,final=True)
    carried={'source_receipts':[prior_train,{'path':logical,'sha256':held_sha}],
        'source_row_bindings':{'edge':{'graph':'old'}},'source_row_order':['edge'],
        'exposure_counts':{'edge':1},'required_visits':{'edge':2},
        'held_row_bindings':{'held:a':prior_entries[0]['binding']},
        'metrics':{'held_coverage':coverage},
        'options':{'epochs':1,'_communication_ids':['edge']}}
    unrelated_held={**carried,'source_receipts':[prior_train,{'path':'/other/held.json','sha256':'other'}],
        'held_row_bindings':{'held:a':{'target':'other'}}}
    progress={'carried':carried,'different-held-source':unrelated_held}
    protected={'prior_receipt':prior_held,'current_receipt':current_held,
        'prior_bindings':{'held:a':prior_entries[0]['binding']},
        'current_bindings':{'held:a':current_entries[0]['binding']},
        'prior_entries':prior_entries,'current_entries':current_entries}
    transitions=_propagate_source_correction_to_carried_frames(progress,stage_id='native_full_sft',
        prior_receipt=prior_train,current_receipt=current_train,
        prior_bindings={'edge':{'graph':'old'}},current_bindings={'edge':{'graph':'new'}},
        current_order=['edge'],current_communication_ids=['edge'],corrected={'edge'},deferred=set(),
        added=set(),pending={},protected_held=protected,proof=json.loads(proof.read_text()),proof_path=proof)
    assert transitions[0]['stage_id']=='carried'
    assert progress['carried']['source_receipts'][1]==current_held
    assert progress['carried']['held_row_bindings']=={'held:a':{'target':'new'}}
    rebound=progress['carried']['metrics']['held_coverage']
    assert rebound['classes']['ordinary']['visits']==[0]
    assert rebound['forward_token_visits']==0
    assert progress['different-held-source']['source_receipts'][1]['path']=='/other/held.json'
    assert progress['different-held-source']['held_row_bindings']=={'held:a':{'target':'other'}}


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


@pytest.mark.parametrize('relative_ledger', [False, True])
def test_live_recipe_correction_accepts_proved_held_repair_and_portable_ledgers(tmp_path, relative_ledger):
    from natlang_neuralese.train.recipe import _source_correction_continuation
    prior = _receipt('/inputs/old.json', 'old-manifest', ['edge'], ['old-leaf'])
    current = _receipt('/inputs/new.json', 'new-manifest', ['edge'], ['new-leaf'])
    old_held = _receipt('/inputs/held.json', 'old-held', ['held'], ['old-held-leaf'])
    new_held = _receipt('/inputs/held-repaired.json', 'new-held', ['held'], ['new-held-leaf'])
    proof = _write_proof(tmp_path, prior, current,
        [_row('edge', {'h':'old'}, {'h':'new'}, 'old-leaf', 'new-leaf', 'corrected')],
        protected_held={'prior':old_held, 'current':new_held,
            'rows':[_row('held', {'h':'old'}, {'h':'new'}, 'old-held-leaf', 'new-held-leaf', 'corrected')]})
    document = json.loads(proof.read_text())
    if relative_ledger:
        document['row_bindings_path'] = Path(document['row_bindings_path']).name
        document['protected_held_correction']['row_bindings_path'] = Path(document['protected_held_correction']['row_bindings_path']).name
        proof.write_text(json.dumps(document))
    def ref(source, label):
        return {'path':source['path'], 'sha256':source['sha256'], 'binding':label}
    old = {'native_graphs':ref(prior,'train'), 'held_native_graphs':ref(old_held,'held')}
    new = {'native_graphs':ref(current,'new-train'), 'continued_native_graphs':ref(prior,'prior-train'),
           'held_native_graphs':ref(new_held,'new-held'), 'continued_held_native_graphs':ref(old_held,'prior-held'),
           'source_correction':{'path':str(proof),'sha256':_sha(proof),'binding':'correction'}}
    assert _source_correction_continuation(old,new)
    bad = copy.deepcopy(new);bad['held_native_graphs']['sha256']='unproved'
    assert not _source_correction_continuation(old,bad)
    bad = copy.deepcopy(new);bad['continued_held_native_graphs']['path']='/inputs/unrelated.json'
    assert not _source_correction_continuation(old,bad)
    bad = copy.deepcopy(new);bad['unrelated_input']={'path':'/smuggled','sha256':'bad','binding':'other'}
    assert not _source_correction_continuation(old,bad)
