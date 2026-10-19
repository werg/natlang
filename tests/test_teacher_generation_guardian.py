import copy
import json
from pathlib import Path
import pytest
from scripts.coreweave import teacher_generation_guardian as guardian


def example(tmp_path):
    entry = {'key':'case-0','source':str(tmp_path/'source.jsonl'),'index':0,'seed':91,
        'jobs':str(tmp_path/'original/jobs'),'output':str(tmp_path/'original/results.jsonl'),
        'log':str(tmp_path/'original/collector.log')}
    finish = {'event':'finish','key':'case-0','status':'repetition','resource_limit_reason':'identical_turn_limit',
        'output_accounting':{'version':'natlang.supervisor_output_accounting/2','manifest_valid':True,
        'expected_jobs':1,'complete':False,'job_states':[{'state':'partial_without_terminal_result'}]}}
    score = {'schema':'natlang.s7-task-rollout-score/1','rollouts':1,'outcome_counts':{'unresolved':1}}
    return entry, finish, score


def test_unresolved_batch_remains_incomplete_with_exact_repair_debt(tmp_path):
    entry, finish, score = example(tmp_path)
    decision = guardian.assess([entry],[finish],[2],score)
    assert decision['batch_complete'] is False
    assert decision['repair_required'] == [{'key':'case-0','reason':'missing or unresolved collector accounting'}]
    assert decision['individual_outcomes'][0] == {'key':'case-0','source':entry['source'],
        'index':0,'seed':91,'finish':finish}
    assert not decision['completion']['training_admission']


def test_complete_semantic_failure_is_not_relabelled_success(tmp_path):
    entry, finish, score = example(tmp_path)
    finish['output_accounting'].update(complete=True,job_states=[{'state':'result'}])
    score['outcome_counts']={'semantic_mismatch':1}
    decision=guardian.assess([entry],[finish],[2],score)
    assert decision['batch_complete'] and not decision['repair_required']
    assert decision['individual_outcomes'][0]['finish']['status']=='repetition'


def test_retry_publication_is_recoverable_and_preserves_source_seed_and_original(tmp_path):
    entry,finish,score=example(tmp_path); original=copy.deepcopy(entry)
    decision=guardian.assess([entry],[finish],[2],score);decision['batch']=13
    guardian.write(tmp_path/'boundaries/batch-0013.json',decision)
    rows=guardian.prepare_retry(tmp_path,decision,[entry]);queue=tmp_path/'repairs/batch-0013/queue.jsonl'
    assert rows==guardian.prepare_retry(tmp_path,decision,[entry])
    assert entry==original and rows[0]['source']==entry['source'] and rows[0]['seed']==91
    assert rows[0]['jobs']!=entry['jobs'] and queue.exists()
    # Recover the crash window between per-key intent and atomic queue publication.
    queue.unlink();assert guardian.prepare_retry(tmp_path,decision,[entry])==rows
    changed={**entry,'seed':92}
    with pytest.raises(ValueError,match='source/seed'):guardian.prepare_retry(tmp_path,decision,[changed])


def test_prepared_live_feeder_is_recovered_without_second_launch(tmp_path,monkeypatch):
    identity={'pid':19,'start_ticks':'123','boot_id':'boot','argv':[b'python',b'feed-evidence-v7.py']}
    monkeypatch.setattr(guardian,'recover_token',lambda token:identity)
    monkeypatch.setattr(guardian.subprocess,'Popen',lambda *a,**k:pytest.fail('duplicate launch'))
    state={'prepared_launch':{'token':'token','batch':14}}
    guardian.launch_feeder(tmp_path,state,{'root':str(tmp_path)},14)
    assert state['owner']['pid']==19 and state['active_batch']==14 and 'prepared_launch' not in state


def test_exited_prepared_feeder_reconciles_materialized_batch_without_replay(tmp_path,monkeypatch):
    (tmp_path/'evidence-collections-v2/batch-0014').mkdir(parents=True)
    monkeypatch.setattr(guardian,'recover_token',lambda token:None)
    monkeypatch.setattr(guardian.subprocess,'Popen',lambda *a,**k:pytest.fail('replayed batch'))
    state={'prepared_launch':{'token':'token','batch':14}}
    guardian.launch_feeder(tmp_path,state,{'root':str(tmp_path)},14)
    assert state['owner'] is None and state['active_batch']==14 and 'prepared_launch' not in state


def test_existing_unowned_batch_is_rejected_before_launch(tmp_path,monkeypatch):
    (tmp_path/'evidence-collections-v2/batch-0014').mkdir(parents=True)
    monkeypatch.setattr(guardian.subprocess,'Popen',lambda *a,**k:pytest.fail('duplicate launch'))
    with pytest.raises(ValueError,match='already exists'):guardian.launch_feeder(tmp_path,{}, {'root':str(tmp_path)},14)


def test_operator_drain_prevents_new_feeder_or_repair(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();control=tmp_path/'control';control.mkdir();drain=tmp_path/'DRAIN';drain.touch()
    config={'root':str(root),'control':str(control),'predecessor_pid':999999,'predecessor_script':'feed-evidence-v6.py',
        'predecessor_batch':13,'drain_file':str(drain),'teacher_identity':{'path':'/immutable/actor.json','sha256':'a'*64}};p=tmp_path/'config.json';p.write_text(json.dumps(config))
    guardian.write(control/'state.json',{'config':guardian.binding(p),'owner':None,'active_batch':None,'next_batch':14})
    monkeypatch.setattr(guardian,'launch_feeder',lambda *a:pytest.fail('feeder during drain'))
    monkeypatch.setattr(guardian,'launch_repair',lambda *a:pytest.fail('repair during drain'))
    monkeypatch.setattr(guardian.time,'sleep',lambda seconds:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):guardian.run(p)
    assert json.loads((control/'phase.json').read_text())['phase']=='operator_drained'


def repair_fixture(tmp_path):
    entry,finish,score=example(tmp_path);decision=guardian.assess([entry],[finish],[2],score);decision['batch']=13
    guardian.write(tmp_path/'boundaries/batch-0013.json',decision)
    guardian.prepare_retry(tmp_path,decision,[entry])
    config={'task_runtime':'/qualified/ts-host','node_bin':'/node/bin','queue_worker':'/repo/run_bonsai_queue.py',
        'server':'http://localhost:8001','model_id':'teacher'}
    return config,tmp_path/'repairs/batch-0013/launch.json'


def test_repair_pre_spawn_crash_recovers_same_token_and_queue(tmp_path,monkeypatch):
    config,receipt=repair_fixture(tmp_path);state={}
    monkeypatch.setattr(guardian,'recover_token',lambda *a,**k:None)
    monkeypatch.setattr(guardian.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('crash before spawn')))
    with pytest.raises(RuntimeError):guardian.launch_repair(tmp_path,state,config)
    before=json.loads(receipt.read_text());assert before['current']['state']=='prepared'
    identity={'pid':23,'start_ticks':'123','boot_id':'boot'}
    calls=[]
    monkeypatch.setattr(guardian.subprocess,'Popen',lambda command,**kwargs:(calls.append((command,kwargs)) or type('Child',(),{'pid':23})()))
    monkeypatch.setattr(guardian,'process',lambda pid:identity)
    guardian.launch_repair(tmp_path,state,config)
    after=json.loads(receipt.read_text());assert after['current']['token']==before['current']['token']
    assert after['current']['queue']==before['current']['queue'] and len(calls)==1
    assert calls[0][1]['env'][guardian.TOKEN]==before['current']['token']


def test_repair_post_spawn_pre_receipt_crash_recovers_live_process_without_duplicate(tmp_path,monkeypatch):
    config,receipt=repair_fixture(tmp_path);state={}
    monkeypatch.setattr(guardian,'recover_token',lambda *a,**k:None)
    monkeypatch.setattr(guardian.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('crash')))
    with pytest.raises(RuntimeError):guardian.launch_repair(tmp_path,state,config)
    identity={'pid':24,'start_ticks':'124','boot_id':'boot','argv':[b'python',b'run_bonsai_queue.py']}
    monkeypatch.setattr(guardian,'recover_token',lambda *a,**k:identity)
    monkeypatch.setattr(guardian.subprocess,'Popen',lambda *a,**k:pytest.fail('duplicate repair'))
    guardian.launch_repair(tmp_path,state,config)
    assert state['repair_owner']['pid']==24
    assert json.loads(receipt.read_text())['current']['state']=='launched'


def test_actual_terminal_failed_retry_is_not_repeated(tmp_path,monkeypatch):
    config,receipt=repair_fixture(tmp_path);state={}
    monkeypatch.setattr(guardian,'recover_token',lambda *a,**k:None)
    monkeypatch.setattr(guardian.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('crash')))
    with pytest.raises(RuntimeError):guardian.launch_repair(tmp_path,state,config)
    value=json.loads(receipt.read_text());attempt=value['current'];entry=json.loads(Path(attempt['queue']).read_text())
    Path(attempt['journal']).write_text(json.dumps({'event':'finish','key':entry['key'],'status':'repetition'})+'\n')
    monkeypatch.setattr(guardian.subprocess,'Popen',lambda *a,**k:pytest.fail('repeated terminal failed repair'))
    guardian.launch_repair(tmp_path,state,config);guardian.launch_repair(tmp_path,state,config)
    value=json.loads(receipt.read_text());assert value['state']=='attempt_exhausted'
    assert value['terminal_keys'][entry['key']]['status']=='repetition'


def test_interrupted_repair_retries_distinct_outputs_and_preserves_old_journal(tmp_path,monkeypatch):
    config,receipt=repair_fixture(tmp_path);state={}
    monkeypatch.setattr(guardian,'recover_token',lambda *a,**k:None)
    monkeypatch.setattr(guardian.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('crash')))
    with pytest.raises(RuntimeError):guardian.launch_repair(tmp_path,state,config)
    before=json.loads(receipt.read_text());attempt=before['current'];entry=json.loads(Path(attempt['queue']).read_text())
    journal=Path(attempt['journal']);journal.write_text(json.dumps({'event':'start','key':entry['key']})+'\n');raw=journal.read_bytes()
    with pytest.raises(RuntimeError):guardian.launch_repair(tmp_path,state,config)
    after=json.loads(receipt.read_text());retry=json.loads(Path(after['current']['queue']).read_text())
    assert retry['seed']==entry['seed'] and retry['index']==entry['index'] and retry['source']==entry['source']
    assert retry['jobs']!=entry['jobs'] and journal.read_bytes()==raw and len(after['attempts'])==1


def test_same_control_runtime_revision_retains_active_owner_debt_and_next_batch(tmp_path):
    config={'root':'/work/generation','control':str(tmp_path),'task_runtime':'/old/ts-host','server':'http://old',
        'model_id':'old-teacher','feeder':'/old/feed-evidence.py'}
    p=tmp_path/'config.json';p.write_text(json.dumps(config));state={'owner':{'pid':31,'start_ticks':'a','boot_id':'boot'},
        'next_batch':15,'pending_debt':['batch13:case-0'],'task_runtime':'/actual-active-old/ts-host'}
    guardian.config_revision(tmp_path,state,p);old_snapshot=state['config_snapshot']
    p.write_text(json.dumps({**config,'task_runtime':'/qualified/new/ts-host','server':'http://new','model_id':'new-teacher'}))
    result=guardian.config_revision(tmp_path,state,p)
    assert result['task_runtime']=='/qualified/new/ts-host' and state['task_runtime']=='/actual-active-old/ts-host'
    assert state['owner']['pid']==31 and state['next_batch']==15 and state['pending_debt']==['batch13:case-0']
    assert len(state['config_history'])==2 and Path(old_snapshot['path']).exists()
    assert state['config_history'][-1]['previous']==old_snapshot


def test_unrelated_pid_reuse_is_not_an_active_batch_worker(tmp_path,monkeypatch):
    # A reused PID in an old metadata list is irrelevant: only exact Python
    # queue-worker argv with this frozen batch path can block its boundary.
    source=tmp_path/'batch-0013';source.mkdir()
    actual=guardian.process
    monkeypatch.setattr(guardian,'process',lambda pid:{'pid':int(pid),'start_ticks':'new','boot_id':'boot',
        'argv':[b'python',b'/unrelated/worker.py']} if str(pid)=='1' else None)
    assert guardian.live_batch_workers(source)==[]


def test_actual_teacher_identity_uses_existing_producer_and_exact_live_engine(tmp_path,monkeypatch):
    actor=tmp_path/'existing-serving-producer.json';actor.write_text(json.dumps({'repo':'own-hosted-teacher','revision':'pinned','quant':'nvfp4'}))
    engine={'pid':41,'start_ticks':'engine-1','boot_id':'boot','argv':[b'python',b'serve',b'--pinned-model']}
    monkeypatch.setattr(guardian,'process',lambda pid:engine)
    expected={**guardian.public_identity(engine),'argv_sha256':guardian.hashlib.sha256(b'\0'.join(engine['argv'])).hexdigest()}
    p=tmp_path/'execution-identity.json';guardian.write(p,{'schema':'natlang.self-hosted-teacher-execution-identity/1',
        'actor':guardian.binding(actor),'processes':[expected],'server':'http://teacher','model_id':'own'})
    proof=guardian.verify_teacher_identity(p,server='http://teacher',model_id='own')
    assert proof['actor']==guardian.binding(actor) and proof['processes']==[expected]
    with pytest.raises(ValueError,match='endpoint'):guardian.verify_teacher_identity(p,server='http://different',model_id='own')
    engine['start_ticks']='reused-pid'
    with pytest.raises(ValueError,match='engine process'):guardian.verify_teacher_identity(p,server='http://teacher',model_id='own')
    engine['start_ticks']='engine-1';actor.write_text('{}')
    with pytest.raises(ValueError,match='producer bytes'):guardian.verify_teacher_identity(p,server='http://teacher',model_id='own')


def test_prepared_never_started_repair_revises_actor_without_claiming_old_actor_called(tmp_path,monkeypatch):
    config,receipt=repair_fixture(tmp_path);state={}
    monkeypatch.setattr(guardian,'recover_token',lambda *a,**k:None)
    monkeypatch.setattr(guardian.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('crash before spawn')))
    with pytest.raises(RuntimeError):guardian.launch_repair(tmp_path,state,config)
    before=json.loads(receipt.read_text())
    with pytest.raises(RuntimeError):guardian.launch_repair(tmp_path,state,{**config,'model_id':'new-quant-teacher','server':'http://new-engine'})
    after=json.loads(receipt.read_text())
    assert after['attempts'][0]['state']=='superseded_unstarted_configuration_revision'
    assert after['current']['model_id']=='new-quant-teacher' and after['current']['server']=='http://new-engine'
    assert after['current']['token']!=before['current']['token']


def preserved_slot_fixture(tmp_path, *, completed=False, closed=False, active_current=False,
                           batch=46, root=None, control=None):
    root=root or tmp_path/'generation';control=control or tmp_path/'control'
    home=root/'evidence-collections-v2'/f'batch-{batch:04d}'
    home.mkdir(parents=True);control.mkdir(parents=True,exist_ok=True)
    sealed={}
    for name,value in [('source-cases.jsonl',b'{"case":"kept"}\n'),
                       ('queue.jsonl',b'{"key":"case"}\n'),
                       ('teacher-source.json',json.dumps({'batch':batch,'actual_task_runtime':'/runtime/epoch-a'}).encode())]:
        (home/name).write_bytes(value);sealed[name]=guardian.binding(home/name)
    intent_path=control/f'source-sequence/intents/batch-{batch:04d}.json'
    guardian.write(intent_path,{'batch':batch,'prefetch':{'task_runtime':'/runtime/epoch-a'}})
    intent=guardian.binding(intent_path)
    started_path=control/f'source-sequence/started/batch-{batch:04d}.json'
    guardian.write(started_path,{'batch':batch,'sealed':sealed,'selection_intent':intent})
    started=guardian.binding(started_path)
    owner={'pid':9021+batch,'start_ticks':f'12345-{batch}','boot_id':'boot-a'}
    slot={'batch':batch,'owner':owner,'selection_intent':intent,'task_runtime':'/runtime/epoch-a'}
    state={'active_batch':45,'source_sequence':{'started':{str(batch):started}}}
    if active_current:
        state['active_batch']=batch;state['owner']=owner;state['primary_launches']={str(batch):slot}
    elif completed:
        boundary_path=control/f'boundaries/batch-{batch:04d}.json'
        guardian.write(boundary_path,{'batch':batch,'batch_complete':True,'bindings':{
            name:sealed[name] for name in ('queue.jsonl','teacher-source.json')}})
        boundary=guardian.binding(boundary_path);slot['boundary']=boundary
        state['primary_completed_ahead']={str(batch):slot}
    else:
        state['primary_active_ahead']={str(batch):slot}
    if closed:
        if not completed:raise AssertionError('closed fixture needs completed batch')
        closure_path=control/f'source-sequence/closed/batch-{batch:04d}.json'
        guardian.write(closure_path,{'batch':batch,'started':started,'boundary':slot['boundary']})
        state['source_sequence']['closed']={str(batch):guardian.binding(closure_path)}
        state['primary_completed_ahead'].clear()
        state['primary_launches']={str(batch):slot}
    expected={'batch':batch,'owner':{'identity':owner,'argv':['python','feeder.py']},'sealed':sealed}
    return root,control,state,expected,started,owner,copy.deepcopy(slot)


def test_preserved_slot_verifier_accepts_live_and_completed_boundary_custody(tmp_path,monkeypatch):
    root,control,state,expected,started,owner,expected_state_slot=preserved_slot_fixture(tmp_path)
    monkeypatch.setattr(guardian,'alive',lambda identity:identity==owner)
    monkeypatch.setattr(guardian,'process',lambda pid:{**owner,'argv':[b'python',b'feeder.py']})
    live=guardian.verify_preserved_primary_slot(root,control,state,expected,started,expected_state_slot)
    assert live['custody_route']=='live-owner' and live['task_runtime']=='/runtime/epoch-a'
    root,control,state,expected,started,owner,expected_state_slot=preserved_slot_fixture(tmp_path/'completed',completed=True)
    monkeypatch.setattr(guardian,'alive',lambda identity:False)
    done=guardian.verify_preserved_primary_slot(root,control,state,expected,started,expected_state_slot)
    assert done['custody_route']=='authenticated-completed-ahead'
    assert done['boundary']==state['primary_completed_ahead']['46']['boundary']


def test_preserved_slot_verifier_accepts_only_exact_closed_source_boundary(tmp_path,monkeypatch):
    root,control,state,expected,started,_,expected_state_slot=preserved_slot_fixture(tmp_path,completed=True,closed=True)
    monkeypatch.setattr(guardian,'alive',lambda identity:False)
    result=guardian.verify_preserved_primary_slot(root,control,state,expected,started,expected_state_slot)
    assert result['custody_route']=='authenticated-closed-boundary'


def test_manifest_operator_entry_resolves_captured_slots_and_started_receipts(tmp_path,monkeypatch):
    root,control,state,expected,started,owner,state_slot=preserved_slot_fixture(tmp_path,completed=True)
    monkeypatch.setattr(guardian,'alive',lambda identity:False)
    handover={'slots':[expected], 'bound_source_sequence':{'started':{'46':started}},
        'bound_primary_prefetch':None, 'bound_primary_completed_ahead':{'46':state_slot},
        'bound_primary_active_ahead':{}, 'bound_primary_launches':{}}
    result=guardian.verify_preserved_primary_slots(root,control,state,handover)
    assert result['batches']==[46] and result['training_admission'] is False
    handover['bound_primary_completed_ahead']['46']['boundary']['sha256']='0'*64
    with pytest.raises(ValueError,match='preserved guardian slot differs from its captured handover state'):
        guardian.verify_preserved_primary_slots(root,control,state,handover)


def test_deployed_v3_manifest_shape_directly_verifies_active_and_closed_46_48(tmp_path,monkeypatch):
    root=tmp_path/'generation';control=tmp_path/'control'
    active=preserved_slot_fixture(tmp_path/'batch-45',active_current=True,batch=45,root=root,control=control)
    closed_cases=[preserved_slot_fixture(tmp_path/f'batch-{batch}',completed=True,closed=True,
        batch=batch,root=root,control=control) for batch in (46,48)]
    cases=[active,*closed_cases]
    started={str(item[3]['batch']):item[4] for item in cases}
    # This is the deployed v3 field shape: the public handover manifest uses
    # bound_prefetch/bound_completed_ahead/bound_active_ahead, not new aliases.
    manifest={'slots':[item[3] for item in cases], 'active_batch':45,
        'original_owner':active[5],
        'bound_source_sequence':{'started':started},
        'bound_prefetch':None,
        'bound_completed_ahead':{str(item[3]['batch']):item[6] for item in closed_cases},
        'bound_active_ahead':{}}
    state={'active_batch':45,'owner':active[5],
        'source_sequence':{'started':started,
        'closed':{str(item[3]['batch']):item[2]['source_sequence']['closed'][str(item[3]['batch'])]
            for item in closed_cases}},
        # 46 and 48 have been naturally archived from every slot map; their
        # only current custody is the source-sequence closed receipt/boundary.
        'primary_launches':{'45':active[6]}}
    monkeypatch.setattr(guardian,'alive',lambda identity:identity==active[5])
    monkeypatch.setattr(guardian,'process',lambda pid:{**active[5],'argv':[b'python',b'feeder.py']})
    result=guardian.verify_preserved_primary_slots(root,control,state,manifest)
    assert result['batches']==[45,46,48]
    assert [slot['custody_route'] for slot in result['slots']]==[
        'live-owner',
        'authenticated-closed-boundary','authenticated-closed-boundary']
    assert result['training_admission'] is False
    manifest['bound_completed_ahead']['46']['boundary']['sha256']='0'*64
    with pytest.raises(ValueError,match='archived slot boundary differs'):
        guardian.verify_preserved_primary_slots(root,control,state,manifest)


@pytest.mark.parametrize('mutation', ['missing-slot','wrong-owner','wrong-cases','wrong-runtime','arbitrary-boundary','missing-boundary'])
def test_preserved_slot_verifier_rejects_missing_or_changed_custody(tmp_path,monkeypatch,mutation):
    root,control,state,expected,started,owner,expected_state_slot=preserved_slot_fixture(tmp_path,completed=True)
    monkeypatch.setattr(guardian,'alive',lambda identity:False)
    if mutation=='missing-slot':state['primary_completed_ahead'].clear()
    elif mutation=='wrong-owner':state['primary_completed_ahead']['46']['owner']={**owner,'start_ticks':'999'}
    elif mutation=='wrong-cases':expected['sealed']['source-cases.jsonl']['sha256']='0'*64
    elif mutation=='wrong-runtime':state['primary_completed_ahead']['46']['task_runtime']='/runtime/epoch-b'
    elif mutation=='arbitrary-boundary':state['primary_completed_ahead']['46']['boundary']=guardian.binding(root/'evidence-collections-v2/batch-0046/queue.jsonl')
    elif mutation=='missing-boundary':(control/'boundaries/batch-0046.json').unlink()
    with pytest.raises((ValueError,KeyError,FileNotFoundError),match='preserved|source|boundary|owner|slot|runtime|batch|batch-0046'):
        guardian.verify_preserved_primary_slot(root,control,state,expected,started,expected_state_slot)
