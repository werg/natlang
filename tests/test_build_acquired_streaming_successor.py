import json
import math
from types import SimpleNamespace

import pytest

from scripts.neuralese_data.build_acquired_streaming_successor import FAMILY, appended_manifest, load_signals, scales_for_append, verify_diagnosis_proof


def test_sealed_successor_protection_binding_is_usable_and_conflicts_fail_closed():
    from scripts.neuralese_data.build_acquired_streaming_successor import receipt_protection_digest
    digest='a'*64
    assert receipt_protection_digest({'protection_digest':digest})==digest
    assert receipt_protection_digest({'current_protection_digest':digest})==digest
    assert receipt_protection_digest({'protection_digest':digest,'current_protection_digest':digest})==digest
    for receipt in ({}, {'protection_digest':digest,'current_protection_digest':'b'*64}):
        with pytest.raises(ValueError,match='protection union'):
            receipt_protection_digest(receipt)


def test_repeated_mass_admission_recovers_sealed_append_and_rejects_unearned_mass(tmp_path,monkeypatch):
    from scripts.neuralese_data import build_acquired_streaming_successor as module
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    monkeypatch.setattr(module,'policy_objective',lambda policy:policy['objective'])
    def write(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return module.bind(path)
    objective={'name':'own-outputs/1','family_balance':{'scales':{'coding':1}}}
    base_source={'sha256':'a'*64};current_source={'sha256':'b'*64}
    unscaled_policy=write('unscaled.json',{'objective':{**objective,'family_balance':None}})
    policy=write('policy.json',{'objective':objective});family=write('family.json',{})
    census_ref=write('base-census.json',{'schema':'natlang.own-output-mass-census/1',
        'objective_without_family_scales':True,'manifest':base_source,'rows':2,'policy':unscaled_policy,
        'families':{'coding':{'rows':2,'unscaled_mass':20.}},'unscaled_mass_total':20.})
    receipt=write('base-mass-receipt.json',{'census':census_ref,'map':family})
    base=write('base.json',{'source':base_source,'rows':2,'ce_policy':policy,'family_map':family,
        'objective_census':census_ref,'objective_census_receipt':receipt})
    screen=write('screen.json',{})
    delta=write('delta.json',{'schema':'natlang.acquired-effective-learning-census/1',
        'screen':screen,'ce_policy':policy,'objective_sha256':canonical_json_sha256_hex(objective),
        'rows':1,'category_mass':{'context':0.,'assistant_context':0.,'reply':5.}})
    previous={'schema':module.SCHEMA,'source':current_source,'rows':3,'new_rows':1,
        'predecessor':base,'predecessor_source':base_source,'effective_loss_census':delta,
        'source_screen':screen,'new_family':'general-agent',
        'family_balance':{'unscaled_mass':{'coding':20.,'general-agent':5.}}}
    path=tmp_path/'previous.json';path.write_text(json.dumps(previous))
    census,proof=module.predecessor_mass_census(previous,path,objective)
    assert census['rows']==3 and census['unscaled_mass_total']==25.
    assert census['families']['general-agent']=={'rows':1,'unscaled_mass':5.}
    assert proof['delta']==delta and proof['predecessor_mass']['census']==census_ref
    previous['family_balance']['unscaled_mass']['general-agent']=500.
    with pytest.raises(ValueError,match='declared family mass'):
        module.predecessor_mass_census(previous,path,objective)


def test_preparation_uses_exact_preserved_builder_after_checkout_changes(tmp_path):
    from scripts.neuralese_data import build_acquired_streaming_successor as module
    source=tmp_path/'builder.py';source.write_text('actual preparation code')
    reference=module.bind(source);prepared=tmp_path/'receipt.json';prepared.write_text('{}')
    preserved=tmp_path/('builder-'+reference['sha256']+'.py');preserved.write_bytes(source.read_bytes())
    source.write_text('new checkout implementation')
    assert module.checked_preparation_builder(prepared,reference)==module.bind(preserved)
    preserved.write_text('wrong historical copy')
    with pytest.raises(ValueError):module.checked_preparation_builder(prepared,reference)


def test_capacity_descriptor_matches_strict_runtime_contract():
    from scripts.neuralese_data.build_acquired_streaming_successor import sft_capacity_descriptor
    source={'path': '/source', 'sha256': 'a'*64}
    geometry={'path': '/geometry', 'sha256': 'b'*64}
    descriptor=sft_capacity_descriptor(source, geometry)
    assert set(descriptor)=={'schema', 'source', 'geometry'}
    assert descriptor['schema']=='natlang.native-sft-capacity-cases/1'
    assert descriptor['source']==source and descriptor['geometry']==geometry


def census():
    return {'families': {'coding': {'unscaled_mass': 153308050.0},
                         'native-communication': {'unscaled_mass': 14715643.0},
                         'natlang': {'unscaled_mass': 42728014.0}},
            'unscaled_mass_total': 210751707.0}


def test_scales_preserve_old_effective_proportions_and_mean_one_when_uncapped():
    old = {'coding': .179308, 'native-communication': 6.226787, 'natlang': 2.144522}
    plan = scales_for_append(census(), old, 10_000_000, share=.07)
    assert plan['effective_shares'][FAMILY] == pytest.approx(.07)
    assert plan['total_effective_mass'] == pytest.approx(plan['total_unscaled_mass'], rel=1e-12)
    assert max(plan['scales'].values()) <= 8
    prior = plan['previous_effective_mass']
    for left, right in [('coding', 'natlang'), ('native-communication', 'natlang')]:
        assert plan['effective_shares'][left] / plan['effective_shares'][right] == pytest.approx(prior[left] / prior[right])


def test_cap_redistributes_mass_without_exceeding_eight():
    old = {'coding': .179308, 'native-communication': 6.226787, 'natlang': 2.144522}
    plan = scales_for_append(census(), old, 300_000_000, share=.07)
    assert plan['capped_families']
    assert max(plan['scales'].values()) <= 8
    assert sum(plan['effective_shares'].values()) == pytest.approx(1)
    assert plan['total_effective_mass'] == pytest.approx(plan['total_unscaled_mass'])
    assert plan['effective_shares'][FAMILY] > .07  # cap makes the requested share infeasible


def test_repeated_admission_extends_existing_family_without_erasing_prior_mass():
    prior={'families':{name:{'unscaled_mass':100.} for name in ('coding','native-communication','natlang',FAMILY)},
        'unscaled_mass_total':400.}
    scales={name:1. for name in prior['families']}
    plan=scales_for_append(prior,scales,50.,share=.25)
    assert plan['unscaled_mass'][FAMILY]==150.
    assert plan['previous_effective_mass'][FAMILY]==100.
    assert plan['total_unscaled_mass']==450. and plan['total_effective_mass']==pytest.approx(450.)
    assert plan['effective_shares'][FAMILY]==pytest.approx(.25)
    assert all(value==1. for value in scales.values())
    capped=scales_for_append(prior,scales,500000.,share=.25)
    assert max(capped['scales'].values())<=8 and capped['capped_families']
    assert capped['total_effective_mass']==pytest.approx(capped['total_unscaled_mass'])


@pytest.mark.parametrize('new_mass', [0, -1, float('inf'), float('nan')])
def test_invalid_new_mass_is_rejected(new_mass):
    with pytest.raises(ValueError, match='finite positive'):
        scales_for_append(census(), {'coding': 1., 'native-communication': 1., 'natlang': 1.}, new_mass)


def row(ident, *, mass=3., diagnosis=False):
    return {'id': ident, 'graph_sha256': 'a' * 64, 'source_store': 'upstream/store',
            'source_groups': ['world:' + ident], 'source_row_id': 'source:' + ident,
            'source_curated': 'source-curated-third-party-SFT', 'source_label_diagnosis': diagnosis,
            'oracle_verified': False, 'native_execution_verified': False, 'forward_tokens': 8,
            'raw_positive_tokens': 6, 'effective_positive_tokens': 3,
            'category_mass': {'context': 0., 'assistant_context': 0., 'reply': mass}, 'eligible': True}


def receipts(rows):
    return ({'rows': len(rows), 'counts': {'forward_tokens': sum(x['forward_tokens'] for x in rows),
        'raw_positive_tokens': sum(x['raw_positive_tokens'] for x in rows),
        'effective_positive_tokens': sum(x['effective_positive_tokens'] for x in rows),
        'source_label_diagnosis_rows': sum(x['source_label_diagnosis'] for x in rows),
        'oracle_verified_rows': 0, 'zero_effective_rows': 0},
        'category_mass': {'context': 0., 'assistant_context': 0., 'reply': sum(x['category_mass']['reply'] for x in rows)},
        'source_stores': {'upstream/store': len(rows)}}, {'output': {'rows': len(rows)}})


def test_census_stream_reconciles_mass_provenance_and_predecessor_ids(tmp_path):
    rows = [row('a'), row('b', mass=4., diagnosis=True)]
    source = tmp_path / 'signals.jsonl'; source.write_text(''.join(json.dumps(item) + '\n' for item in rows))
    receipt, screen = receipts(rows)
    loaded, mass, counts, stores = load_signals(source, census=receipt, screen=screen,
                                                protected=SimpleNamespace(identities=set()), old_ids={'old'})
    assert loaded == rows and mass['reply'] == 7 and counts['source_label_diagnosis_rows'] == 1
    assert stores == {'upstream/store': 2}
    with pytest.raises(ValueError, match='predecessor graph ID'):
        load_signals(source, census=receipt, screen=screen, protected=SimpleNamespace(identities=set()), old_ids={'a'})
    rows[0]['category_mass']['context'] = 1.
    source.write_text(''.join(json.dumps(item) + '\n' for item in rows))
    with pytest.raises(ValueError, match='context loss'):
        load_signals(source, census=receipt, screen=screen, protected=SimpleNamespace(identities=set()), old_ids=set())


def test_diagnosis_proof_binds_exact_source_subset_and_program(tmp_path):
    rows = [row('a', diagnosis=True), row('b')]
    proof = {'schema': 'natlang.acquired-source-diagnosis-target-verification/1',
             'screen': {'sha256': 'b' * 64}, 'ce_policy': {'sha256': 'c' * 64},
             'verified_rows': 1, 'rows': [{'id': 'a', 'source_store': 'upstream/store', 'target_sha256': 'd' * 64}]}
    proof_path = tmp_path / 'proof.json'; proof_path.write_text(json.dumps(proof))
    program = tmp_path / 'verify.py'; program.write_text('print(1)\n')
    result = verify_diagnosis_proof(rows, proof_path, program, {'sha256': 'b' * 64}, {'sha256': 'c' * 64})
    assert result['verified_rows'] == 1 and result['program']['path'] == str(program)
    proof['rows'][0]['id'] = 'unrelated'; proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='mismatched row'):
        verify_diagnosis_proof(rows, proof_path, program, {'sha256': 'b' * 64}, {'sha256': 'c' * 64})


def test_appended_manifest_preserves_relative_predecessor_reference_and_controls(tmp_path):
    leaf = tmp_path / 'old.jsonl'; leaf.write_text('{}\n')
    source = tmp_path / 'sources.json'
    source.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'custom': {'kept': True},
        'sources': [{'kind': 'native-graphs', 'path': leaf.name, 'sha256': 'a' * 64, 'training_priority': 2}]}))
    graph = tmp_path / 'new.jsonl'; graph.write_text('{}\n')
    result = appended_manifest(source, graph, {'sha256': 'b' * 64, 'bytes': 3})
    assert result['custom'] == {'kept': True}
    assert result['sources'][0] == {'kind': 'native-graphs', 'path': str(leaf), 'sha256': 'a' * 64, 'training_priority': 2}
    assert result['sources'][1]['path'] == str(graph) and result['sources'][1]['training_priority'] == 1


def test_conversion_custody_reconciles_every_receipt_without_reading_graphs(tmp_path, monkeypatch):
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    from scripts.neuralese_data import build_acquired_streaming_successor as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    builder = tmp_path / 'scripts/neuralese_data/build_acquired_full_session_graphs.py'; builder.parent.mkdir(parents=True)
    writer = tmp_path / 'scripts/neuralese_data/build_native_graph_text_corpus.py'; writer.write_text('writer')
    converter = tmp_path / 'training/neuralese/natlang_neuralese/data/native_graph_conversion.py'; converter.parent.mkdir(parents=True)
    builder.write_text('builder'); converter.write_text('converter')
    screen_program = tmp_path / 'screen.py'; screen_program.write_text('screen')
    output = {'path': str(tmp_path / 'graph-that-need-not-be-opened.zst'), 'sha256': 'a' * 64, 'bytes': 100, 'rows': 1}
    source_export = {'path': str(tmp_path / 'export.json'), 'sha256': 'b' * 64}
    source = {'path': str(tmp_path / 'raw.jsonl'), 'sha256': 'c' * 64}
    receipt = {'schema': 'natlang.acquired-full-session-graphs/1', 'training_admission': False,
        'builder_sha256': module.bind(builder)['sha256'], 'runtime': {'files': {'native_graph_conversion': module.bind(converter), 'graph_writer': module.bind(writer)}},
        'outputs': [output], 'counts': {'graphs:train': 1}, 'source_record_count': 1,
        'export_receipt': source_export, 'source': source}
    receipt_path = tmp_path / 'receipt.json'; receipt_path.write_text(json.dumps(receipt))
    receipt_sha = module.bind(receipt_path)['sha256']
    entry = {'receipt': str(receipt_path), 'receipt_sha256': receipt_sha, 'builder_sha256': receipt['builder_sha256'],
        'outputs': [output], 'counts': receipt['counts'], 'source_record_count': 1, 'export_sha256': 'b' * 64}
    inventory = [{'path': str(receipt_path), 'sha256': receipt_sha, 'source_export': source_export, 'source': source, 'output': output}]
    screen = {'input_receipts': {'count': 1, 'sha256': canonical_json_sha256_hex(inventory)}}
    evidence = module.verify_conversion_custody({'receipts': [entry]}, screen, screen_program)
    assert evidence['conversion_receipts'] == 1
    writer.write_text('changed writer')
    with pytest.raises(ValueError, match='converter'):
        module.verify_conversion_custody({'receipts': [entry]}, screen, screen_program)
    writer.write_text('writer')
    screen['input_receipts']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='inventory'):
        module.verify_conversion_custody({'receipts': [entry]}, screen, screen_program)


def test_successor_recipe_rebinds_roles_from_static_predecessor_without_dropping_correction():
    from scripts.neuralese_data.build_acquired_streaming_successor import bind_successor_recipe_inputs
    recipe={'input_bindings':{'static_train':{'path':'old','sha256':'a'},'static_ce':{},'older':{},'static_proof':{},'held':{}},'stages':[{'id':'native_joint_sft_v4','inputs':{
        'native_graphs':'static_train','foundation_native_graphs':'static_train','continued_native_graphs':'older',
        'ce_supervision_policy':'static_ce','source_correction':'static_proof','held_native_graphs':'held'}}]}
    stage=bind_successor_recipe_inputs(recipe,{'path':'old','sha256':'a','bytes':1},
        {'path':'new','sha256':'b','bytes':2},{'path':'ce','sha256':'c','bytes':3})
    assert stage['inputs']['native_graphs']==stage['inputs']['foundation_native_graphs']=='static_train'
    assert recipe['input_bindings'][stage['inputs']['native_graphs']]=={'path':'new','sha256':'b'}
    assert recipe['input_bindings'][stage['inputs']['continued_native_graphs']]=={'path':'old','sha256':'a'}
    assert stage['inputs']['source_correction']=='static_proof' and stage['inputs']['held_native_graphs']=='held'
    assert not {'static_ce','older'}&set(recipe['input_bindings'])
    assert {'static_proof','held'}<=set(recipe['input_bindings'])


def test_shared_named_binding_stays_valid_for_recipe_continuation():
    from natlang_neuralese.train.recipe import _data_continuation
    old={'native_graphs':{'path':'old','sha256':'a','binding':'static_train'},
         'foundation_native_graphs':{'path':'old','sha256':'a','binding':'static_train'},
         'held_native_graphs':{'path':'held','sha256':'h','binding':'held'},
         'source_correction':{'path':'proof','sha256':'p','binding':'proof'}}
    new={**old,'native_graphs':{'path':'new','sha256':'b','binding':'static_train'},
         'foundation_native_graphs':{'path':'new','sha256':'b','binding':'static_train'},
         'continued_native_graphs':{'path':'old','sha256':'a','binding':'previous'}}
    assert _data_continuation(old,new)
    new['native_graphs']['binding']='renamed'
    new['foundation_native_graphs']['binding']='renamed'
    assert not _data_continuation(old,new)


@pytest.mark.parametrize('tamper', ['none', 'builder', 'screen', 'predecessor', 'partition', 'admission'])
def test_reviewed_partition_reuse_binds_exact_preparation(tmp_path, tamper):
    from scripts.neuralese_data.build_acquired_streaming_successor import bind, reviewed_partition_count, PREPARED_SCHEMA
    def write(name, value):
        path=tmp_path/name
        path.write_text(json.dumps(value))
        return bind(path)
    screen=write('screen.json', {'screen':1}); predecessor=write('predecessor.json', {'source':1})
    old=write('old.json', {'source':0});builder=write('builder.py', {'reviewed':True})
    active={'path':'active','sha256':'a','rows':3}
    partition=write('partition.json', {'schema':'natlang.acquired-position-pending-partition/1',
        'predecessor':old,'active_graph':active,'pending_graph':{'rows':2}})
    prep={'schema':PREPARED_SCHEMA,'training_admission':False,'screen':screen,'predecessor':predecessor,'builder':builder}
    if tamper=='screen': prep['screen']['sha256']='bad'
    if tamper=='predecessor': prep['predecessor']['sha256']='bad'
    if tamper=='admission': prep['training_admission']=True
    ref=write('prep.json',prep); partition_screen={'pending_partition':partition,'output':active}
    if tamper=='partition': partition_screen['output']={'sha256':'bad'}
    digest='bad' if tamper=='builder' else builder['sha256']
    args=(ref['path'],digest,screen['path'],predecessor['path'],partition_screen,old['path'])
    if tamper=='none': assert reviewed_partition_count(*args)==2
    else:
        with pytest.raises(ValueError): reviewed_partition_count(*args)
