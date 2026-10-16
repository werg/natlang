import copy
import hashlib
import json
from pathlib import Path
import pytest
from scripts.coreweave.teacher_external_reconciliation import reconcile_snapshot


def fixture(tmp_path):
    files = {}
    def store(source, value, *, lines=False, compact=False):
        path = tmp_path/'files'/source.lstrip('/')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':') if compact else None)+'\n' for row in value) if lines else json.dumps(value))
        item={'source':source,'path':str(path.relative_to(tmp_path)),
              'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
        files[source]=item
        return {key:item['source'] if key=='path' else item[key] for key in ('path','sha256','bytes')}
    packet=store('/remote/packet.jsonl',[{'id':'task0'},{'id':'task1'}],lines=True)
    original_source='/remote/original/teacher-source.json'
    original_queue='/remote/original/queue.jsonl'
    original={'schema':'natlang.sandbox-strong-teacher-collection/1',
        'source_cases':packet['path'],'source_cases_sha256':packet['sha256'],'seed_base':31,'protected_selection':{'packet':'train'}}
    parent=store(original_source,original)
    entries=[{'key':f'case{i}','index':i,'count':1,'seed':31+i,'source':packet['path'],
        'jobs':f'/remote/original/{i}/jobs','output':f'/remote/original/{i}/results.jsonl',
        'log':f'/remote/original/{i}/log'} for i in range(2)]
    queue=store(original_queue,entries,lines=True)
    store('/remote/original/journal.jsonl',[{'event':'start','key':'case0'}],lines=True)
    actor=store('/remote/retry/actor.json',{'model':'actual-teacher'})
    execution=store('/remote/retry/execution.json',{'actor':actor})
    drain=store('/remote/retry/drain.json',{'original_incomplete':True})
    continuation={**original,'schema':'natlang.sandbox-strong-teacher-continuation/1',
        'original_source':parent,'original_queue':queue,'drain_frontier':drain,'actor':actor,
        'actual_teacher_execution_identity':execution,'source_indices_seeds_preserved':True,'entries':2}
    retry_source='/remote/retry/teacher-source.json';retry_queue='/remote/retry/queue.jsonl'
    store(retry_source,continuation)
    retries=[{**entry,**{field:entry[field].replace('/original/','/retry/') for field in ('jobs','output','log')}} for entry in entries]
    store(retry_queue,retries,lines=True)
    events=[{'event':'start','key':'case0','teacher_identity':{'actor':actor,'identity':execution}},
        {'event':'finish','key':'case0','status':'complete','output_accounting':{
            'version':'natlang.supervisor_output_accounting/2','manifest_valid':True,
            'complete':True,'expected_jobs':1,'job_states':[{'state':'result'}]}}]
    store('/remote/retry/journal-worker-0.jsonl',events,lines=True)
    def publish():
        index=tmp_path/'index.json';index.write_text(json.dumps({'files':list(files.values())}));return index
    kwargs={'original_source':original_source,'original_queue':original_queue,
            'continuations':[(retry_source,retry_queue)]}
    return store,publish,kwargs,entries,retries,events,continuation


def test_join_preserves_original_incomplete_and_only_closes_bound_terminal_keys(tmp_path):
    store,publish,kwargs,entries,retries,events,continuation=fixture(tmp_path)
    result=reconcile_snapshot(publish(),**kwargs)
    assert set(result['terminal_keys'])=={'case0'} and result['unresolved_keys']==['case1']
    assert result['terminal_keys']['case0'][0]['finish']==events[-1]
    assert result['original_batch_complete'] is False and result['training_admission'] is False
    assert len(result['collections'][1]['continuation_proofs'])==5


@pytest.mark.parametrize('tamper',['seed','unknown-key','duplicate-finish','actor','missing-start','queue-parent','bytes'])
def test_join_rejects_unbound_or_relabelled_completion(tmp_path,tamper):
    store,publish,kwargs,entries,retries,events,continuation=fixture(tmp_path)
    if tamper=='seed':
        retries[0]['seed']+=1;store('/remote/retry/queue.jsonl',retries,lines=True)
    elif tamper=='unknown-key':
        events[-1]['key']='unknown';store('/remote/retry/journal-worker-0.jsonl',events,lines=True)
    elif tamper=='duplicate-finish':
        store('/remote/retry/journal-worker-1.jsonl',[events[-1]],lines=True)
    elif tamper=='actor':
        events[0]['teacher_identity']['actor']['sha256']='0'*64
        store('/remote/retry/journal-worker-0.jsonl',events,lines=True)
    elif tamper=='missing-start':store('/remote/retry/journal-worker-0.jsonl',events[1:],lines=True)
    elif tamper=='queue-parent':
        continuation['original_queue']['sha256']='0'*64
        store('/remote/retry/teacher-source.json',continuation)
    index=publish()
    if tamper=='bytes':(tmp_path/'files/remote/retry/actor.json').write_text('changed')
    with pytest.raises(ValueError):reconcile_snapshot(index,**kwargs)


def test_packet_rendering_bytes_may_differ_but_all_ordered_values_must_match(tmp_path):
    store,publish,kwargs,entries,retries,events,continuation=fixture(tmp_path)
    rendered=store('/remote/rendered.jsonl',[{'id':'task0'},{'id':'task1'}],lines=True,compact=True)
    assert rendered['sha256'] != continuation['source_cases_sha256']
    for entry in entries:entry['source']=rendered['path']
    for entry in retries:entry['source']=rendered['path']
    queue=store('/remote/original/queue.jsonl',entries,lines=True)
    continuation['original_queue']=queue
    store('/remote/retry/teacher-source.json',continuation)
    store('/remote/retry/queue.jsonl',retries,lines=True)
    assert set(reconcile_snapshot(publish(),**kwargs)['terminal_keys'])=={'case0'}
    store('/remote/rendered.jsonl',[{'id':'task1'},{'id':'task0'}],lines=True)
    with pytest.raises(ValueError,match='selected collection packet'):reconcile_snapshot(publish(),**kwargs)


def adoption_fixture(tmp_path,monkeypatch):
    from scripts.coreweave import teacher_generation_guardian as guardian
    from scripts.coreweave import teacher_external_reconciliation as external
    folder=tmp_path/'control';folder.mkdir();root=tmp_path/'root'
    def save(path,value):guardian.write(path,value);return guardian.binding(path)
    packet=tmp_path/'packet.jsonl';packet.write_text('{}\n')
    queue=root/'evidence-collections-v2/batch-0014/queue.jsonl'
    original_source=str(queue.parent/'teacher-source.json')
    entries=[{'key':'remaining','index':0,'seed':31,'source':str(packet),
              'jobs':str(tmp_path/'old/jobs'),'output':str(tmp_path/'old/results'),'log':str(tmp_path/'old/log')}]
    save(queue,entries[0]);queue.write_text(json.dumps(entries[0])+'\n')
    oldconfig={'root':str(root),'control':str(folder),'feeder':'/old/feeder','server':'http://localhost:8001',
        'model_id':'old','task_runtime':'/old/runtime','teacher_identity':{'path':'/old/identity'},
        'predecessor_pid':1,'predecessor_script':'/old/feeder','predecessor_batch':14}
    oldstate={'owner':None,'active_batch':14,'config_value':oldconfig}
    save(folder/'state.json',oldstate);oldbinding=save(folder/'old-state.json',oldstate)
    feeder=folder/'feed-evidence-v8.py';feeder.write_text('pass\n')
    teacher=save(folder/'teacher.json',{'schema':'actual-test-teacher'})
    config={**oldconfig,'feeder':str(feeder),'model_id':'new','task_runtime':'/new/runtime','teacher_identity':teacher}
    config_path=folder/'config.json';configbinding=save(config_path,config)
    owner={'pid':21,'start_ticks':'33','boot_id':'boot','argv':[b'python',str(feeder).encode()]}
    transition={'schema':'natlang.teacher-generation-external-transition/1','original_batch_complete':False,
        'original_state':oldbinding,'original_batch':14,'batch':15,'owner':guardian.public_identity(owner),
        'process_token':'token','config':configbinding,'feeder':guardian.binding(feeder),
        'task_runtime':config['task_runtime'],'teacher_identity':teacher,
        'source_authority':save(folder/'authority.json',{}),
        'completed_continuation':save(folder/'exits.json',{'returncodes':[0]})}
    transition_path=folder/'transition.json';save(transition_path,transition)
    index=tmp_path/'index.json';save(index,{'files':[{'source':str(queue),**guardian.binding(queue)},
                                                    {'source':str(packet),**guardian.binding(packet)}]})
    evidence={'schema':'natlang.teacher-external-terminal-reconciliation/1','snapshot':guardian.binding(index),
        'original_batch_complete':False,'training_admission':False,'original_source':original_source,
        'original_queue':str(queue),'expected_keys':1,'terminal_keys':{},'unresolved_keys':['remaining'],
        'collections':[{'source':original_source,'queue':str(queue)}]}
    evidence_path=folder/'reconciliation.json';save(evidence_path,evidence)
    monkeypatch.setattr(external,'reconcile_snapshot',lambda *args,**kwargs:copy.deepcopy(evidence))
    monkeypatch.setattr(guardian,'alive',lambda value:False)
    monkeypatch.setattr(guardian,'recover_token',lambda token:owner)
    monkeypatch.setattr(guardian,'verify_teacher_identity',lambda *args,**kwargs:{})
    monkeypatch.setattr(guardian,'live_batch_workers',lambda path:[])
    return guardian,external,(folder,config_path,transition_path,evidence_path),oldstate,transition


def test_adoption_preserves_debt_and_actual_successor_without_duplicate_launch(tmp_path,monkeypatch):
    guardian,external,args,oldstate,transition=adoption_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(guardian.subprocess,'Popen',lambda *a,**kw:pytest.fail('duplicate feeder'))
    state=external.adopt_external_transition(*args)
    assert state['active_batch']==15 and state['owner']==transition['owner']
    assert state['external_transitions'][0]['unresolved_keys']==['remaining']
    decision=json.loads((args[0]/'boundaries/batch-0014.json').read_text())
    assert decision['batch_complete'] is False
    retry=json.loads((args[0]/'repairs/batch-0014/queue.jsonl').read_text())
    assert retry['key']=='batch-0014:remaining' and retry['seed']==31
    assert external.adopt_external_transition(*args)==state


@pytest.mark.parametrize('tamper',['live-old','wrong-token','teacher-bytes','live-workers','changed-original','fake-terminal'])
def test_adoption_rejects_changed_owners_and_unverified_dispositions(tmp_path,monkeypatch,tamper):
    guardian,external,args,oldstate,transition=adoption_fixture(tmp_path,monkeypatch)
    if tamper=='live-old':monkeypatch.setattr(guardian,'alive',lambda value:True)
    elif tamper=='wrong-token':monkeypatch.setattr(guardian,'recover_token',lambda token:None)
    elif tamper=='teacher-bytes':Path(transition['teacher_identity']['path']).write_text('{}')
    elif tamper=='live-workers':monkeypatch.setattr(guardian,'live_batch_workers',lambda path:[{'pid':1}])
    elif tamper=='changed-original':guardian.write(args[0]/'state.json',{**oldstate,'active_batch':13})
    elif tamper=='fake-terminal':
        value=json.loads(args[3].read_text());value['unresolved_keys']=[];guardian.write(args[3],value)
    with pytest.raises(ValueError):external.adopt_external_transition(*args)
    if tamper!='changed-original':assert json.loads((args[0]/'state.json').read_text())==oldstate


def test_adoption_recovers_after_config_preparation_before_atomic_owner_commit(tmp_path,monkeypatch):
    guardian,external,args,oldstate,transition=adoption_fixture(tmp_path,monkeypatch)
    original=guardian.config_revision
    def crash(*args,**kwargs):
        original(*args,**kwargs)
        raise RuntimeError('interrupted before owner commit')
    monkeypatch.setattr(guardian,'config_revision',crash)
    with pytest.raises(RuntimeError):external.adopt_external_transition(*args)
    assert json.loads((args[0]/'state.json').read_text())==oldstate
    monkeypatch.setattr(guardian,'config_revision',original)
    assert external.adopt_external_transition(*args)['owner']==transition['owner']


def test_packet_jsonl_preserves_unicode_line_separator_inside_values(tmp_path):
    store,publish,kwargs,entries,retries,events,continuation=fixture(tmp_path)
    packet=store('/remote/packet.jsonl',[{'id':'task0','text':'a\u2028b'},{'id':'task1'}],lines=True)
    original_path='/remote/original/teacher-source.json'
    original=json.loads((tmp_path/'files'/original_path.lstrip('/')).read_text())
    original['source_cases_sha256']=packet['sha256']
    continuation['source_cases_sha256']=packet['sha256']
    continuation['original_source']=store(original_path,original)
    store('/remote/retry/teacher-source.json',continuation)
    assert set(reconcile_snapshot(publish(),**kwargs)['terminal_keys'])=={'case0'}


@pytest.mark.parametrize('omitted',['manifest','error'])
def test_unindexed_existing_budget_file_cannot_close_a_retry(tmp_path,omitted):
    store,publish,kwargs,entries,retries,events,continuation=fixture(tmp_path)
    error=store('/remote/retry/0/jobs/error.json',{'index':0,'program_id':'task0',
        'code':'NATLANG_MODEL_REQUEST_BUDGET','source_sha256':continuation['source_cases_sha256'],
        'program_ir_sha256':'a'*64})
    manifest=store('/remote/retry/0/results.jsonl.manifest.json',{
        'version':'natlang.teacher_batch.native/1','source':entries[0]['source'],
        'source_sha256':continuation['source_cases_sha256'],'range':{'start':0,'count':1},'missing':[0]})
    events[-1].update(status='incomplete',resource_limit_reason='model_request_budget',
        output_accounting={'version':'natlang.supervisor_output_accounting/2','manifest_valid':True,
            'complete':False,'expected_jobs':1,'job_states':[{'index':0,'program_id':'task0','digest':'a'*64,
                'state':'failed','resource_limit_reason':'model_request_budget','terminal_error':error}]})
    store('/remote/retry/journal-worker-0.jsonl',events,lines=True)
    index=publish()
    assert 'case0' in reconcile_snapshot(index,**kwargs)['terminal_keys']
    value=json.loads(index.read_text());reference=manifest if omitted=='manifest' else error
    value['files']=[item for item in value['files'] if item['source']!=reference['path']]
    index.write_text(json.dumps(value))
    assert (tmp_path/'files'/reference['path'].lstrip('/')).is_file()
    result=reconcile_snapshot(index,**kwargs)
    assert 'case0' not in result['terminal_keys'] and 'case0' in result['unresolved_keys']
