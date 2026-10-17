"""Filesystem/lifecycle fixtures only; these are not reference or generation rows."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.coreweave import teacher_generation_guardian as guardian
from scripts.coreweave import teacher_source_sequence as sequence


def put(path,value):
    guardian.write(path,value);return guardian.binding(path)


def setup(tmp_path, *, bad=None, packet_count=2):
    root=tmp_path/'generation';root.mkdir();control=tmp_path/'control';control.mkdir()
    runtime=tmp_path/'runtime';runtime.mkdir();runtime_file=runtime/'cli.js';runtime_file.write_bytes(b'qualified runtime')
    runtime_inventory={'cli.js':guardian.binding(runtime_file)}
    runtime_sha=hashlib.sha256(json.dumps(runtime_inventory,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    protection={'path':'original/protection.json','sha256':'a'*64,'held_groups':3,'quality_groups':4}
    holds=[{'path':'original/holds.jsonl','sha256':'b'*64,'bytes':10}]
    authority={'protection':protection,'decision_hold_bindings':holds,'reference_runtime_sha256':runtime_sha}
    entries=[];packets=[]
    for index in range(packet_count):
        folder=tmp_path/f'packet{index}';folder.mkdir()
        record={'version':'natlang.program/2','id':f'train-task-{index}','split':'train','source_groups':[f'group-{index}'],
            'semantics':{'expected':1}}
        raw=json.dumps(record,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
        candidate_file=folder/'candidate.jsonl';candidate_file.write_bytes(raw+b'\n')
        candidate=put(folder/'candidate-proof.json',{'cases':1,'cases_binding':guardian.binding(candidate_file)})
        cases=folder/'qualified.jsonl'
        cases.write_bytes(raw.replace(b'"expected":1',b'"expected":1.0')+b'\n'if bad=='changed_program'else raw+b'\n')
        cases_ref=guardian.binding(cases)
        inputs=put(folder/'inputs.json',{'cases':guardian.binding(candidate_file),
            'runtime':runtime_inventory,'authority':{'protection':protection,'decision_holds':holds}})
        verdicts=folder/'verdicts.jsonl'
        verdict={'program_sha256':hashlib.sha256(raw).hexdigest(),'passed':True,'model_calls':0,
            'training_admission':False,'reference_traces_are_teacher_generations':False}
        verdicts.write_text(''if bad=='missing_verdict'else json.dumps(verdict)+'\n')
        qualification=put(folder/'qualification.json',{'schema':'natlang.existing-native-reference-qualification/1',
            'passed':True,'model_calls':0,'training_admission':False,'reference_traces_are_teacher_generations':False,
            'qualified_cases_binding':cases_ref,'qualified_cases':1,'candidate_cases':1,'inputs':inputs,
            'runtime_inventory_sha256':runtime_sha if bad!='wrong_runtime'else 'f'*64,
            'verdicts':guardian.binding(verdicts)})
        selection=put(folder/'selection.json',{'schema':'natlang.strong-teacher-code-source-selection/1',
            'cases':1,'cases_sha256':cases_ref['sha256'],'training_admission':False,
            'held_and_pending_intersections':0 if bad!='protected'else 1,
            'protection':protection,'decision_hold_bindings':holds,'reference_qualification':qualification})
        join=put(folder/'join.json',{'schema':'natlang.native-teacher-global-protection-join/1','status':'passed',
            'selected_packet_sha256':cases_ref['sha256'],'held_group_intersections':[],
            'quality_group_intersections':[],'pending_group_intersections':[],
            'current_protection':protection,'decision_hold_bindings':holds})
        prepared=put(folder/'prepared.json',{'cases':cases_ref,'qualification':qualification,
            'references':{'selection':selection,'global_protection_join':join,'selected_cases':cases_ref}})
        entries.append({'ordinal':index,'candidate_packet':candidate,'qualified_packet':prepared,
                        'started':False,'closed':False,'production_adopted':False});packets.append(candidate)
    inventory=put(tmp_path/'inventory.json',{'packets':packets})
    manifest=tmp_path/'ready.json';put(manifest,{'schema':'natlang.strong-teacher-ready-packet-sequence/1',
        'candidate_inventory':inventory,'entries':entries,'training_admission':False,'teacher_generations':0})
    anchor=root/'evidence-collections-v2/batch-0021';anchor.mkdir(parents=True)
    (anchor/'source-cases.jsonl').write_bytes(b'original selected140 packet\n')
    anchor_sha=guardian.binding(anchor/'source-cases.jsonl')['sha256']
    put(anchor/'queue.jsonl',{'original':'queue'})
    put(anchor/'teacher-source.json',{'source_cases_sha256':anchor_sha})
    boundary=put(control/'boundary21.json',{'batch':21,'batch_complete':True,
        'bindings':{name:guardian.binding(anchor/name)for name in ('queue.jsonl','teacher-source.json')}})
    current=root/'code-source-v3.json';current.write_bytes(b'{ "old" : "keep exact bytes" }')
    declaration={'anchor':{'batch':21,'cases_sha256':anchor_sha},'ready_manifest':str(manifest),
        'candidate_inventory':inventory,
        'authority':authority,'source_configuration':str(current),'initial_configuration':guardian.binding(current),
        'on_unready':'retain-current','task_runtime':str(runtime),'task_runtime_bindings':[guardian.binding(runtime_file)]}
    config={'root':str(root),'task_runtime':str(runtime),'source_sequence':declaration}
    state={'last_boundary':boundary,'owner':None,'unknown_owner_field':{'keep':'exact'}}
    put(control/'state.json',state)
    return control,state,config


def seal(folder,state,config,batch):
    active=json.loads(Path(config['source_sequence']['source_configuration']).read_text())
    root=Path(config['root'])/'evidence-collections-v2'/f'batch-{batch:04d}';root.mkdir()
    (root/'source-cases.jsonl').write_bytes(Path(active['cases']).read_bytes())
    put(root/'queue.jsonl',{'actual':'sealed fixture queue'})
    put(root/'teacher-source.json',{'source_cases_sha256':active['cases_sha256']})
    return {name:guardian.binding(root/name)for name in ('source-cases.jsonl','queue.jsonl','teacher-source.json')}


def test_batch21_stays_exact_until_its_selected_packet_is_sealed(tmp_path):
    folder,state,config=setup(tmp_path)
    old=Path(config['source_sequence']['source_configuration']).read_bytes()
    config['source_sequence']['ready_manifest']='not-staged-yet'
    config['source_sequence']['task_runtime_bindings']=[]
    sequence.prepare_source_sequence(folder,state,config,21)
    assert Path(config['source_sequence']['source_configuration']).read_bytes()==old
    assert 'source_sequence'not in state


def test_actual_sealed_and_closed_lifecycles_advance_order_without_rewriting_current(tmp_path):
    folder,state,config=setup(tmp_path)
    anchor=Path(config['root'])/'evidence-collections-v2/batch-0021'
    before={str(p):p.read_bytes()for p in anchor.iterdir()}
    sequence.prepare_source_sequence(folder,state,config,22)
    assert state['source_sequence']['next_ordinal']==0
    assert 'started'not in state['source_sequence']
    first=seal(folder,state,config,22)
    sequence.observe_source_sequence(folder,state,config)
    assert '22'in state['source_sequence']['started']
    assert 'closed'not in state['source_sequence']
    with pytest.raises(ValueError,match='preceding completed boundary'):
        sequence.prepare_source_sequence(folder,state,config,23)
    state['last_boundary']=put(folder/'boundary22.json',{'batch':22,'batch_complete':True,
        'bindings':{name:first[name]for name in ('queue.jsonl','teacher-source.json')},
        'individual_outcomes':[{'key':'task','success':False}]})
    sequence.observe_source_sequence(folder,state,config)
    assert state['source_sequence']['next_ordinal']==1
    assert '22'in state['source_sequence']['closed']
    sequence.prepare_source_sequence(folder,state,config,23)
    current=json.loads(Path(config['source_sequence']['source_configuration']).read_text())
    assert json.loads(Path(current['cases']).read_text())['id']=='train-task-1'
    assert state['unknown_owner_field']=={'keep':'exact'}
    assert {str(p):p.read_bytes()for p in anchor.iterdir()}==before
    assert all(guardian.binding(Path(ref['path']))==ref for ref in first.values())


@pytest.mark.parametrize('bad',['changed_program','missing_verdict','wrong_runtime','protected'])
def test_unqualified_or_changed_packets_cannot_replace_current_source(tmp_path,bad):
    folder,state,config=setup(tmp_path,bad=bad)
    current=Path(config['source_sequence']['source_configuration']);old=current.read_bytes()
    with pytest.raises(ValueError):sequence.prepare_source_sequence(folder,state,config,22)
    assert current.read_bytes()==old
    assert not state['source_sequence'].get('pending')


@pytest.mark.parametrize('where',['before_configuration','after_configuration'])
def test_interrupted_publication_reuses_exact_intent_without_advancing_packet(tmp_path,monkeypatch,where):
    folder,state,config=setup(tmp_path);actual_write=sequence.write
    target=Path(config['source_sequence']['source_configuration']);interrupted=False
    def interrupt(path,value):
        nonlocal interrupted
        path=Path(path)
        fail=(path==target if where=='before_configuration'else
              path==folder/'state.json'and 'configuration'in(state.get('source_sequence')or{}))
        if fail and not interrupted:
            interrupted=True;raise OSError('injected publication interruption')
        return actual_write(path,value)
    with monkeypatch.context()as patch:
        patch.setattr(sequence,'write',interrupt)
        with pytest.raises(OSError,match='injected'):sequence.prepare_source_sequence(folder,state,config,22)
    restored=json.loads((folder/'state.json').read_text())
    intent=copy.deepcopy(restored['source_sequence']['pending']['intent'])
    sequence.prepare_source_sequence(folder,restored,config,22)
    assert restored['source_sequence']['pending']['intent']==intent
    assert guardian.binding(Path(intent['path']))==intent
    assert restored['source_sequence']['next_ordinal']==0
    sequence.verify_prepared_source(restored,config,22)


def test_manifest_reordering_and_external_configuration_mutation_fail_closed(tmp_path):
    folder,state,config=setup(tmp_path)
    sequence.prepare_source_sequence(folder,state,config,22)
    target=Path(config['source_sequence']['source_configuration']);raw=target.read_bytes()
    manifest=Path(config['source_sequence']['ready_manifest']);value=json.loads(manifest.read_text());value['entries'].reverse();put(manifest,value)
    with pytest.raises(ValueError,match='prefix'):sequence.prepare_source_sequence(folder,state,config,22)
    assert target.read_bytes()==raw
    value['entries'].reverse();put(manifest,value)
    target.write_bytes(b'{"unrelated":"operator edit"}')
    with pytest.raises(ValueError,match='outside its intent'):sequence.prepare_source_sequence(folder,state,config,22)
    assert target.read_bytes()==b'{"unrelated":"operator edit"}'


def test_explicit_unready_policy_retains_current_without_crediting_a_new_packet(tmp_path):
    folder,state,config=setup(tmp_path)
    manifest=Path(config['source_sequence']['ready_manifest']);value=json.loads(manifest.read_text());value['entries']=[];put(manifest,value)
    current=Path(config['source_sequence']['source_configuration']);before=current.read_bytes()
    sequence.prepare_source_sequence(folder,state,config,22)
    assert state['source_sequence']['next_ordinal']==0
    assert not state['source_sequence'].get('pending')
    assert current.read_bytes()==before


def test_guardian_write_preserves_legacy_json_bytes_and_can_snapshot_exact_raw(tmp_path):
    path=tmp_path/'json';value={'unicode':'ä','nested':[1,1.0]};guardian.write(path,value)
    assert path.read_bytes()==(json.dumps(value,indent=2)+'\n').encode()
    raw=b'{ "old" : 1 }';guardian.write(path,raw);assert path.read_bytes()==raw


def test_verified_json_parses_saved_bytes_instead_of_rereading_replaced_path(tmp_path,monkeypatch):
    path=tmp_path/'proof.json';reference=put(path,{'exact':'verified original'})
    read=Path.read_bytes
    def replace_after_read(self):
        raw=read(self)
        if self==path:self.write_bytes(b'{"exact":"different replacement"}')
        return raw
    with monkeypatch.context()as patch:
        patch.setattr(Path,'read_bytes',replace_after_read)
        assert sequence._json(reference,{})=={'exact':'verified original'}
    with pytest.raises(ValueError,match='binding changed'):sequence._json(reference,{})


def test_pending_launch_rechecks_current_declared_protection_and_runtime(tmp_path,monkeypatch):
    folder,state,config=setup(tmp_path);target=Path(config['source_sequence']['source_configuration'])
    before=target.read_bytes();write=sequence.write
    def fail_publication(path,value):
        if Path(path)==target:raise OSError('before CAS')
        return write(path,value)
    with monkeypatch.context()as patch:
        patch.setattr(sequence,'write',fail_publication)
        with pytest.raises(OSError,match='before CAS'):sequence.prepare_source_sequence(folder,state,config,22)
    config['source_sequence']['authority']['reference_runtime_sha256']='f'*64
    with pytest.raises(ValueError,match='authority differs'):
        sequence.prepare_source_sequence(folder,state,config,22)
    assert target.read_bytes()==before


def test_mutable_ready_pointer_cannot_invalidate_historical_manifest_receipt(tmp_path):
    folder,state,config=setup(tmp_path)
    sequence.prepare_source_sequence(folder,state,config,22)
    reference=copy.deepcopy(state['source_sequence']['pending']['intent'])
    intent=json.loads(Path(reference['path']).read_text());historical=intent['manifest']
    pointer=Path(config['source_sequence']['ready_manifest']);value=json.loads(pointer.read_text());value['new_progress_note']='later qualifier progress';put(pointer,value)
    sequence.prepare_source_sequence(folder,state,config,22)
    assert guardian.binding(Path(historical['path']))==historical
    assert state['source_sequence']['pending']['intent']==reference


def test_supervisor_candidate_inventory_is_not_replaced_by_a_ready_pointer(tmp_path):
    folder,state,config=setup(tmp_path)
    pointer=Path(config['source_sequence']['ready_manifest']);value=json.loads(pointer.read_text())
    value['candidate_inventory']=put(tmp_path/'undeclared-candidates.json',{'packets':[]});put(pointer,value)
    with pytest.raises(ValueError,match='supervisor declaration'):
        sequence.prepare_source_sequence(folder,state,config,22)


def test_reference_runtime_continuation_is_candidate_bound_and_legacy_packet_stays_exact(tmp_path):
    folder,state,config=setup(tmp_path)
    declaration=config['source_sequence'];manifest_path=Path(declaration['ready_manifest'])
    manifest=json.loads(manifest_path.read_text());entry=manifest['entries'][0]
    prepared_path=Path(entry['qualified_packet']['path']);prepared=json.loads(prepared_path.read_text())
    qual_path=Path(prepared['qualification']['path']);qual=json.loads(qual_path.read_text())
    inputs_path=Path(qual['inputs']['path']);inputs=json.loads(inputs_path.read_text())
    inputs['runtime']={'new-exact-reference-helper.js':{'sha256':'c'*64,'bytes':12,'path':'new-exact-reference-helper.js'}}
    runtime_sha=hashlib.sha256(json.dumps(inputs['runtime'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    qual['inputs']=put(inputs_path,inputs);qual['runtime_inventory_sha256']=runtime_sha
    prepared['qualification']=put(qual_path,qual)
    selection_path=Path(prepared['references']['selection']['path']);selection=json.loads(selection_path.read_text())
    selection['reference_qualification']=prepared['qualification']
    prepared['references']['selection']=put(selection_path,selection)
    entry['qualified_packet']=put(prepared_path,prepared);put(manifest_path,manifest)
    with pytest.raises(ValueError,match='authority differs'):
        sequence._packet(entry,declaration)
    declaration['authority']['reference_runtime_by_candidate']={entry['candidate_packet']['sha256']:runtime_sha}
    assert sequence._packet(entry,declaration)['cases_sha256']
    # The unrelated original packet remains valid under its old exact runtime.
    assert sequence._packet(manifest['entries'][1],declaration)['cases_sha256']
    sequence.prepare_source_sequence(folder,state,config,22)
    assert state['source_sequence']['pending']['ordinal']==0
    declaration['authority']['reference_runtime_by_candidate']={'d'*64:runtime_sha}
    with pytest.raises(ValueError,match='authority differs'):
        sequence._packet(entry,declaration)


@pytest.mark.parametrize('binding',[[],{'not-a-candidate-sha':'a'*64},{'a'*64:True},{'a'*64:'B'*64}])
def test_reference_runtime_continuation_rejects_malformed_authority(tmp_path,binding):
    _,_,config=setup(tmp_path);declaration=config['source_sequence']
    manifest=json.loads(Path(declaration['ready_manifest']).read_text())
    declaration['authority']['reference_runtime_by_candidate']=binding
    with pytest.raises(ValueError,match='malformed'):
        sequence._packet(manifest['entries'][0],declaration)


def incomplete_boundary(folder, state, config, batch):
    """Use the real guardian accounting and retry preparer, never a success label."""
    source=Path(config['root'])/'evidence-collections-v2'/f'batch-{batch:04d}'
    entries=[dict(key=f'task-{index}',source=str(source/'source-cases.jsonl'),index=index,
        count=1,seed=400000+index,jobs=str(source/f'jobs-{index}'),
        output=str(source/f'results-{index}.jsonl'),log=str(source/f'collector-{index}.log'))
        for index in range(2)]
    (source/'queue.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in entries))
    journal=[dict(key=entries[0]['key'],event='finish',output_accounting=dict(
        version='natlang.supervisor_output_accounting/2',manifest_valid=True,manifest_errors=[],
        expected_jobs=1,complete=True,job_states=[dict(state='result')]))]
    (source/'journal.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in journal))
    score=dict(schema='natlang.s7-task-rollout-score/1',rollouts=2,
               outcome_counts=dict(semantic_mismatch=1,infrastructure=1))
    put(source/'score.json',score)
    decision=guardian.assess(entries,journal,[0,2],score)
    decision.update(batch=batch,bindings={name:guardian.binding(source/name)
        for name in('queue.jsonl','journal.jsonl','score.json','teacher-source.json')})
    state['last_boundary']=put(folder/'boundaries'/f'batch-{batch:04d}.json',decision)
    guardian.prepare_retry(folder,decision,entries)
    return source,entries,decision


def test_drained_anchor_with_bound_repair_debt_starts_independent_next_packet(tmp_path):
    folder,state,config=setup(tmp_path)
    source,entries,decision=incomplete_boundary(folder,state,config,21)
    originals={p.name:p.read_bytes()for p in source.iterdir()}
    boundary=copy.deepcopy(state['last_boundary'])
    sequence.prepare_source_sequence(folder,state,config,22)
    assert state['source_sequence']['pending']['ordinal']==0
    assert guardian.binding(Path(boundary['path']))==boundary
    assert {p.name:p.read_bytes()for p in source.iterdir()}==originals
    closure=json.loads((folder/'source-sequence/drained-boundaries/batch-0021.json').read_text())
    assert closure['batch_complete']is False and closure['training_admission']is False
    assert closure['repair_required_keys']==['task-1']
    assert decision['completion']['exact_completed_queue_keys']==1
    assert decision['batch_complete']is False
    assert state['unknown_owner_field']=={'keep':'exact'}


def test_started_packet_closes_with_repair_debt_without_claiming_completion(tmp_path):
    folder,state,config=setup(tmp_path)
    sequence.prepare_source_sequence(folder,state,config,22)
    seal(folder,state,config,22)
    # Observe the actual sealed queue before collection, just as the guardian does.
    source,entries,decision=incomplete_boundary(folder,state,config,22)
    sequence.observe_source_sequence(folder,state,config)
    assert state['source_sequence']['next_ordinal']==1
    closed=json.loads(Path(state['source_sequence']['closed']['22']['path']).read_text())
    assert 'drained_with_repair_debt'in closed
    assert json.loads(Path(closed['boundary']['path']).read_text())['batch_complete']is False
    sequence.prepare_source_sequence(folder,state,config,23)
    assert state['source_sequence']['pending']['ordinal']==1


@pytest.mark.parametrize('tamper',['missing_queue','changed_seed','changed_prepared_original','missing_prepared','changed_journal'])
def test_incomplete_boundary_requires_exact_repair_and_original_custody(tmp_path,tamper):
    folder,state,config=setup(tmp_path)
    source,entries,decision=incomplete_boundary(folder,state,config,21)
    repair=folder/'repairs/batch-0021';queue=repair/'queue.jsonl';prepared=repair/'task-1/prepared.json'
    if tamper=='missing_queue':queue.unlink()
    elif tamper=='missing_prepared':prepared.unlink()
    elif tamper=='changed_seed':
        row=json.loads(queue.read_text());row['seed']+=1;queue.write_text(json.dumps(row)+'\n')
    elif tamper=='changed_prepared_original':
        value=json.loads(prepared.read_text());value['original']['seed']=float(value['original']['seed']);put(prepared,value)
    else:(source/'journal.jsonl').write_text('')
    before=Path(config['source_sequence']['source_configuration']).read_bytes()
    with pytest.raises((ValueError,FileNotFoundError)):
        sequence.prepare_source_sequence(folder,state,config,22)
    assert Path(config['source_sequence']['source_configuration']).read_bytes()==before
    assert not state.get('source_sequence')


def test_unassigned_global_accounting_error_cannot_be_called_drained_with_repair(tmp_path):
    folder,state,config=setup(tmp_path)
    source,entries,decision=incomplete_boundary(folder,state,config,21)
    decision['completion']['repair_required'].append(dict(key=None,reason='unassigned global failure'))
    decision['repair_required']=decision['completion']['repair_required']
    state['last_boundary']=put(folder/'boundaries/batch-0021.json',decision)
    with pytest.raises(ValueError,match='accounting changed'):
        sequence.prepare_source_sequence(folder,state,config,22)
