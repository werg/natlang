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


def context_deferrals(folder,state,config,ordinals):
    declaration=config['source_sequence'];ready_path=Path(declaration['ready_manifest']);ready=json.loads(ready_path.read_text())
    packets=[];programs=[]
    for ordinal in ordinals:
        entry=ready['entries'][ordinal];p=json.loads(Path(entry['qualified_packet']['path']).read_text())
        rows=[json.loads(line)for line in Path(p['cases']['path']).read_bytes().splitlines()]
        row=rows[0];digest=hashlib.sha256(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        programs.append(dict(program_id=row['id'],program_sha256=digest))
        packets.append(dict(ordinal=ordinal,candidate_packet=entry['candidate_packet'],qualified_packet=entry['qualified_packet'],cases=p['cases'],excluded_program_ids=[row['id']],source_cases=len(rows),excluded_cases=1,retained_cases=len(rows)-1))
    audit=put(folder/'context-audit.json',{'training_admission':False})
    proof=dict(schema='natlang.teacher-source-context-exclusions/1',original_ready=guardian.binding(ready_path),source_context_audit=audit,primary_source_audit=audit,original_context_source=audit,excluded_programs=programs,deferred_packets=packets,training_admission=False,model_calls=0)
    declaration['context_deferrals']=put(folder/'context-deferrals.json',proof)
    return proof


def test_exact_future_deferral_preserves_observed_prefix_and_records_debt(tmp_path):
    folder,state,config=setup(tmp_path)
    original=json.loads(Path(config['source_sequence']['ready_manifest']).read_text())['entries']
    context_deferrals(folder,state,config,[0])
    sequence.prepare_source_sequence(folder,state,config,22)
    actual=state['source_sequence'];assert actual['observed_entries']==original
    assert actual['pending']['ordinal']==1 and actual['next_ordinal']==1
    assert not actual.get('closed') and not actual.get('started')
    debt=json.loads(Path(actual['context_deferred']['0']['path']).read_text())
    assert debt['excluded_program_ids']==['train-task-0']
    assert debt['training_admission']is False and debt['generation_admission']is False
    assert debt['source_cases']==debt['excluded_cases']+debt['retained_cases']==1
    current=json.loads(Path(config['source_sequence']['source_configuration']).read_text())
    assert json.loads(Path(current['cases']).read_text())['id']=='train-task-1'
    # Repeated reconciliation does not consume another packet or invent closure.
    before=copy.deepcopy(actual);sequence.prepare_source_sequence(folder,state,config,22)
    assert state['source_sequence']==before


def test_context_deferral_cannot_skip_pending_current_packet(tmp_path):
    folder,state,config=setup(tmp_path)
    sequence.prepare_source_sequence(folder,state,config,22)
    current=Path(config['source_sequence']['source_configuration']);before=current.read_bytes()
    context_deferrals(folder,state,config,[0])
    with pytest.raises(ValueError,match='unstarted'):
        sequence.prepare_source_sequence(folder,state,config,22)
    assert current.read_bytes()==before and state['source_sequence']['pending']['ordinal']==0
    assert not state['source_sequence'].get('context_deferred')


@pytest.mark.parametrize('mutation',["unknown_id","wrong_program","lost_sibling","repeated_packet"])
def test_invalid_context_deferral_cannot_advance_or_publish(tmp_path,mutation):
    folder,state,config=setup(tmp_path);proof=context_deferrals(folder,state,config,[0])
    if mutation=='unknown_id':proof['deferred_packets'][0]['excluded_program_ids']=['unknown']
    elif mutation=='wrong_program':proof['excluded_programs'][0]['program_sha256']='0'*64
    elif mutation=='lost_sibling':proof['deferred_packets'][0]['retained_cases']=1
    else:proof['deferred_packets'].append(copy.deepcopy(proof['deferred_packets'][0]))
    config['source_sequence']['context_deferrals']=put(folder/'context-deferrals.json',proof)
    current=Path(config['source_sequence']['source_configuration']);before=current.read_bytes()
    with pytest.raises(ValueError):sequence.prepare_source_sequence(folder,state,config,22)
    assert current.read_bytes()==before and not state['source_sequence'].get('pending')
    assert not state['source_sequence'].get('context_deferred')


def test_context_debt_allows_fresh_qualified_packet_append_after_real_closure(tmp_path):
    folder,state,config=setup(tmp_path,packet_count=3);decl=config['source_sequence']
    ready_path=Path(decl['ready_manifest']);full=json.loads(ready_path.read_text());fresh=full['entries'][-1]
    initial={**full,'entries':full['entries'][:2]};put(ready_path,initial)
    context_deferrals(folder,state,config,[0]);sequence.prepare_source_sequence(folder,state,config,22)
    sealed=seal(folder,state,config,22)
    state['last_boundary']=put(folder/'boundary22.json',{'batch':22,'batch_complete':True,'bindings':{key:sealed[key]for key in('queue.jsonl','teacher-source.json')}})
    sequence.observe_source_sequence(folder,state,config)
    assert state['source_sequence']['next_ordinal']==2 and '22'in state['source_sequence']['closed']
    appended=put(folder/'ready-appended.json',{**initial,'entries':[*initial['entries'],fresh]})
    decl['ready_manifest']=appended['path'];sequence.prepare_source_sequence(folder,state,config,23)
    assert state['source_sequence']['pending']['ordinal']==2
    assert '0'in state['source_sequence']['context_deferred']
    current=json.loads(Path(decl['source_configuration']).read_text())
    assert json.loads(Path(current['cases']).read_text())['id']=='train-task-2'


def test_context_deferral_cannot_reclassify_genuinely_started_closed_work(tmp_path):
    folder,state,config=setup(tmp_path)
    sequence.prepare_source_sequence(folder,state,config,22);sealed=seal(folder,state,config,22)
    state['last_boundary']=put(folder/'boundary22.json',{'batch':22,'batch_complete':True,'bindings':{key:sealed[key]for key in('queue.jsonl','teacher-source.json')}})
    sequence.observe_source_sequence(folder,state,config)
    before=copy.deepcopy(state);context_deferrals(folder,state,config,[0])
    with pytest.raises(ValueError,match='unstarted'):
        sequence.prepare_source_sequence(folder,state,config,23)
    assert state==before


def test_context_deferral_reuses_durable_debt_after_cursor_write_interruption(tmp_path,monkeypatch):
    folder,state,config=setup(tmp_path);context_deferrals(folder,state,config,[0])
    original=sequence.write
    def fail_state(path,value):
        if Path(path)==folder/'state.json':raise OSError('assigned cursor interruption')
        return original(path,value)
    monkeypatch.setattr(sequence,'write',fail_state)
    with pytest.raises(OSError,match='cursor interruption'):
        sequence.prepare_source_sequence(folder,state,config,22)
    debt=folder/'source-sequence/context-deferred/ordinal-0000.json'
    before=debt.read_bytes();state=json.loads((folder/'state.json').read_text())
    monkeypatch.setattr(sequence,'write',original)
    sequence.prepare_source_sequence(folder,state,config,22)
    assert debt.read_bytes()==before
    assert state['source_sequence']['pending']['ordinal']==1
    assert len(state['source_sequence']['context_deferred'])==1


def test_past_context_debt_is_reauthenticated_and_cannot_be_silently_removed(tmp_path):
    folder,state,config=setup(tmp_path);proof=context_deferrals(folder,state,config,[0])
    sequence.prepare_source_sequence(folder,state,config,22)
    stored=state['source_sequence']['context_deferred']['0'];path=Path(stored['path']);original=path.read_bytes()
    # A new proof may stop deferring future packets; historical debt is retained
    # and its exact original proof stays bound, rather than overwritten.
    config['source_sequence']['context_deferrals']=put(folder/'deferrals-successor.json',{**proof,'deferred_packets':[]})
    sequence.prepare_source_sequence(folder,state,config,22)
    assert state['source_sequence']['context_deferred']['0']==stored
    assert path.read_bytes()==original
    path.write_bytes(original+b' ')
    with pytest.raises(ValueError,match='binding changed'):
        sequence.prepare_source_sequence(folder,state,config,22)


def test_disabling_context_deferral_policy_does_not_drop_historical_proof_checks(tmp_path):
    folder,state,config=setup(tmp_path);context_deferrals(folder,state,config,[0])
    sequence.prepare_source_sequence(folder,state,config,22)
    stored=state['source_sequence']['context_deferred']['0'];receipt=json.loads(Path(stored['path']).read_text())
    config['source_sequence'].pop('context_deferrals')
    sequence.prepare_source_sequence(folder,state,config,22)
    proof=Path(receipt['context_exclusions']['path']);proof.write_bytes(proof.read_bytes()+b' ')
    with pytest.raises(ValueError,match='binding changed'):
        sequence.prepare_source_sequence(folder,state,config,22)


def requalify_with_current_protection(entry, protection, holds):
    prepared_path=Path(entry['qualified_packet']['path']);prepared=json.loads(prepared_path.read_bytes())
    qual_path=Path(prepared['qualification']['path']);qual=json.loads(qual_path.read_bytes())
    inputs_path=Path(qual['inputs']['path']);inputs=json.loads(inputs_path.read_bytes())
    inputs['authority'].update(protection=protection,decision_holds=holds)
    qual['inputs']=put(inputs_path,inputs);prepared['qualification']=put(qual_path,qual)
    selection_path=Path(prepared['references']['selection']['path']);selection=json.loads(selection_path.read_bytes())
    selection.update(protection=protection,decision_hold_bindings=holds,reference_qualification=prepared['qualification'])
    prepared['references']['selection']=put(selection_path,selection)
    join_path=Path(prepared['references']['global_protection_join']['path']);join=json.loads(join_path.read_bytes())
    join.update(current_protection=protection,decision_hold_bindings=holds)
    prepared['references']['global_protection_join']=put(join_path,join)
    entry['qualified_packet']=put(prepared_path,prepared)


def test_fresh_protection_is_exact_candidate_bound_and_preserves_historical_packet(tmp_path):
    folder,state,config=setup(tmp_path);decl=config['source_sequence'];manifest_path=Path(decl['ready_manifest']);manifest=json.loads(manifest_path.read_bytes());entry=manifest['entries'][0]
    original_second=copy.deepcopy(manifest['entries'][1]);second_bytes=Path(original_second['qualified_packet']['path']).read_bytes()
    protection={**decl['authority']['protection'],'path':'current/protection.json','sha256':'d'*64}
    holds=[{'path':'current/decision-hold.jsonl','sha256':'e'*64,'bytes':9}]
    requalify_with_current_protection(entry,protection,holds)
    with pytest.raises(ValueError,match='authority differs'):sequence._packet(entry,decl)
    exact={'protection':protection,'decision_hold_bindings':holds}
    decl['authority']['reference_protection_by_candidate']={entry['candidate_packet']['sha256']:exact}
    assert sequence._packet(entry,decl)['rows']==1
    assert sequence._packet(original_second,decl)['rows']==1
    assert Path(original_second['qualified_packet']['path']).read_bytes()==second_bytes
    put(manifest_path,manifest);sequence.prepare_source_sequence(folder,state,config,22)
    assert state['source_sequence']['pending']['ordinal']==0
    decl['authority']['reference_protection_by_candidate']={'f'*64:exact}
    with pytest.raises(ValueError,match='authority differs'):sequence._packet(entry,decl)


@pytest.mark.parametrize('tamper',['selection','protection_join','inputs'])
def test_candidate_current_authority_requires_every_independent_proof(tmp_path,tamper):
    _,_,config=setup(tmp_path);decl=config['source_sequence'];manifest=json.loads(Path(decl['ready_manifest']).read_bytes());entry=manifest['entries'][0]
    protection={**decl['authority']['protection'],'sha256':'d'*64};holds=decl['authority']['decision_hold_bindings']
    requalify_with_current_protection(entry,protection,holds)
    decl['authority']['reference_protection_by_candidate']={entry['candidate_packet']['sha256']:{'protection':protection,'decision_hold_bindings':holds}}
    pp=Path(entry['qualified_packet']['path']);prepared=json.loads(pp.read_bytes())
    if tamper=='inputs':
        qp=Path(prepared['qualification']['path']);qual=json.loads(qp.read_bytes());ip=Path(qual['inputs']['path']);inputs=json.loads(ip.read_bytes());inputs['authority']['protection']={'sha256':'f'*64};qual['inputs']=put(ip,inputs);prepared['qualification']=put(qp,qual)
        sp=Path(prepared['references']['selection']['path']);selection=json.loads(sp.read_bytes());selection['reference_qualification']=prepared['qualification'];prepared['references']['selection']=put(sp,selection)
    else:
        role='selection'if tamper=='selection'else'global_protection_join';path=Path(prepared['references'][role]['path']);value=json.loads(path.read_bytes());value['protection'if role=='selection'else'current_protection']={'sha256':'f'*64};prepared['references'][role]=put(path,value)
    entry['qualified_packet']=put(pp,prepared)
    with pytest.raises(ValueError,match='authority differs|protection proof differs'):sequence._packet(entry,decl)


@pytest.mark.parametrize('value',[[],{'bad':{}},{'a'*64:{}},{'a'*64:{'protection':{'path':'x','sha256':'d'*64},'decision_hold_bindings':[{'path':'h','sha256':'e'*64,'bytes':True}]}},{'a'*64:{'protection':{'path':'x','sha256':'d'*64},'decision_hold_bindings':[],'unknown':True}}])
def test_candidate_protection_mapping_rejects_malformed_authority(tmp_path,value):
    _,_,config=setup(tmp_path);decl=config['source_sequence'];entry=json.loads(Path(decl['ready_manifest']).read_bytes())['entries'][0]
    decl['authority']['reference_protection_by_candidate']=value
    with pytest.raises(ValueError,match='malformed'):sequence._packet(entry,decl)
