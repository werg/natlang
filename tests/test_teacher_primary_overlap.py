"""Real source/queue files exercise bounded reservations; no model execution."""
import copy
import json
from pathlib import Path

import pytest
from scripts.coreweave import teacher_generation_guardian as g
from scripts.coreweave import teacher_source_sequence as seq
from test_teacher_source_sequence import setup,put,seal


def active(tmp_path,monkeypatch,*,bad=None):
    folder,state,config=setup(tmp_path,bad=bad,packet_count=4)
    seq.prepare_source_sequence(folder,state,config,22)
    sealed=seal(folder,state,config,22)
    seq.observe_source_sequence(folder,state,config)
    state.update(owner={'pid':123,'start_ticks':'4','boot_id':'boot'},active_batch=22)
    monkeypatch.setattr(seq,'alive',lambda owner:bool(owner))
    config['primary_parallel_batches']=2;config['collection_repo']=str(tmp_path/'collection-repo')
    worker=tmp_path/'run_bonsai_queue.py';worker.write_text('# exact worker\n')
    config.update(queue_worker=str(worker),collection_worker=g.binding(worker),node_bin=str(tmp_path/'node/bin'))
    return folder,state,config,sealed


def reserve(tmp_path,monkeypatch):
    folder,state,config,sealed=active(tmp_path,monkeypatch)
    before=Path(config['source_sequence']['source_configuration']).read_bytes()
    reservation=seq.prepare_prefetched_source(folder,state,config,23)
    assert reservation['ordinal']==1
    assert state['source_sequence']['pending']['batch']==22
    assert state['source_sequence']['next_ordinal']==0
    assert Path(config['source_sequence']['source_configuration']).read_bytes()==before
    assert seq.prepare_prefetched_source(folder,state,config,23)==reservation
    value=seq._json(seq._json(reservation['intent'],config['source_sequence'])['new_configuration'],config['source_sequence'])
    root=Path(config['root'])/'evidence-collections-v2/batch-0023';root.mkdir()
    (root/'source-cases.jsonl').write_bytes(Path(value['cases']).read_bytes())
    put(root/'queue.jsonl',{'actual':'prefetched fixture queue'})
    put(root/'teacher-source.json',{'batch':23,'source_cases_sha256':value['cases_sha256'],'actual_task_runtime':config['task_runtime']})
    started=seq.observe_prefetched_source(folder,state,config)
    assert started==state['source_sequence']['started']['23']
    return folder,state,config,sealed,reservation


def close_predecessor(folder,state,config,sealed):
    state['owner']=None
    state['last_boundary']=put(folder/'boundary22.json',{'batch':22,'batch_complete':True,
        'bindings':{name:sealed[name]for name in ('queue.jsonl','teacher-source.json')}})
    seq.observe_source_sequence(folder,state,config)


def test_reserve_materialize_and_adopt_only_after_actual_closure(tmp_path,monkeypatch):
    folder,state,config,sealed,reservation=reserve(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='unfinished predecessor'):
        seq.adopt_prefetched_source(folder,state,config,23)
    assert '23'not in state['source_sequence'].get('closed',{})
    close_predecessor(folder,state,config,sealed)
    seq.adopt_prefetched_source(folder,state,config,23)
    assert state['source_sequence']['pending']==reservation
    assert state['source_sequence']['next_ordinal']==1
    assert 'prefetch'not in state['source_sequence']
    # Crash between source adoption and guardian owner publication is idempotent.
    assert seq.adopt_prefetched_source(folder,state,config,23)==reservation


@pytest.mark.parametrize('mutation',['wrong_batch','source','runtime','intent'])
def test_prefetch_tamper_fails_without_changing_current_source(tmp_path,monkeypatch,mutation):
    folder,state,config,sealed=active(tmp_path,monkeypatch)
    target=Path(config['source_sequence']['source_configuration']);before=target.read_bytes()
    if mutation=='wrong_batch':batch=24
    else:batch=23
    if mutation=='source':(Path(config['root'])/'evidence-collections-v2/batch-0022/source-cases.jsonl').write_text('changed')
    if mutation=='runtime':Path(config['source_sequence']['task_runtime_bindings'][0]['path']).write_text('changed')
    if mutation=='intent':state['source_sequence']['pending']['intent']['sha256']='0'*64
    with pytest.raises(ValueError):seq.prepare_prefetched_source(folder,state,config,batch)
    assert target.read_bytes()==before
    assert 'prefetch'not in state['source_sequence']


def test_two_slots_forbid_third_reservation(tmp_path,monkeypatch):
    folder,state,config,_,_=reserve(tmp_path,monkeypatch)
    with pytest.raises(ValueError):seq.prepare_prefetched_source(folder,state,config,24)


def test_changed_future_protection_is_not_a_valid_reservation(tmp_path,monkeypatch):
    folder,state,config,_=active(tmp_path,monkeypatch)
    manifest=json.loads(Path(config['source_sequence']['ready_manifest']).read_text())
    ref=manifest['entries'][1]['qualified_packet'];prepared=json.loads(Path(ref['path']).read_text())
    selection=prepared['references']['selection'];value=json.loads(Path(selection['path']).read_text());value['held_and_pending_intersections']=1
    put(Path(selection['path']),value)
    with pytest.raises(ValueError):seq.prepare_prefetched_source(folder,state,config,23)
    assert 'prefetch'not in state['source_sequence']


def test_adoption_does_not_accept_modified_reserved_packet(tmp_path,monkeypatch):
    folder,state,config,sealed,reservation=reserve(tmp_path,monkeypatch)
    close_predecessor(folder,state,config,sealed)
    (Path(config['root'])/'evidence-collections-v2/batch-0023/source-cases.jsonl').write_text('changed')
    with pytest.raises(ValueError):seq.adopt_prefetched_source(folder,state,config,23)
    assert state['source_sequence'].get('pending')is None


def tail(tmp_path,*,unstarted=0,unfinished=2):
    home=tmp_path/'evidence-collections-v2/batch-0028';home.mkdir(parents=True)
    rows=[{'key':str(i)}for i in range(8)]
    (home/'queue.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
    events=[]
    for i in range(8-unstarted):
        events.append({'event':'start','key':str(i)})
        if i<8-unstarted-unfinished:events.append({'event':'finish','key':str(i),'status':'complete'})
    journal=home/'journal-worker-0.jsonl';journal.write_text(''.join(json.dumps(e)+'\n'for e in events))
    return home,journal


def test_tail_requires_all_started_and_bounded_actual_work(tmp_path):
    home,journal=tail(tmp_path)
    value=g.primary_tail_frontier(tmp_path,28,6)
    assert value['remaining_keys']==['6','7']and not value['unstarted_keys']
    assert value['queue']==g.binding(home/'queue.jsonl')
    assert g.primary_tail_frontier(tmp_path,28,1)is None


def test_unstarted_work_never_grants_an_extra_batch(tmp_path):
    tail(tmp_path,unstarted=1,unfinished=1)
    assert g.primary_tail_frontier(tmp_path,28,6)is None


def test_tail_rejects_foreign_batch_journal(tmp_path):
    _,journal=tail(tmp_path)
    with journal.open('a')as out:out.write(json.dumps({'event':'start','key':'foreign'})+'\n')
    with pytest.raises(ValueError,match='another batch'):g.primary_tail_frontier(tmp_path,28,6)


def test_owner_restart_recovers_exact_token_without_spawn(tmp_path,monkeypatch):
    folder=tmp_path;source=tmp_path/'source.json';source.write_text('{}');feeder=tmp_path/'teacher_collection_feeder.py';feeder.write_text('')
    slot={'batch':29,'token':'exact','owner':None,'feeder':g.binding(feeder),'source_configuration':g.binding(source)}
    state={'primary_prefetch':slot};owner={'pid':7,'start_ticks':'8','boot_id':'boot','argv':[]}
    monkeypatch.setattr(g,'recover_token',lambda token,**kwargs:owner if token=='exact'else None)
    monkeypatch.setattr(g.subprocess,'Popen',lambda *a,**k:pytest.fail('duplicate feeder spawn'))
    g._spawn_primary_prefetch(folder,state,{'root':str(tmp_path)},slot,12)
    assert slot['owner']==g.public_identity(owner)
    assert json.loads((folder/'state.json').read_text())['primary_prefetch']['owner']==slot['owner']


def test_materialized_owner_loss_never_relaunches_existing_batch(tmp_path,monkeypatch):
    home=tmp_path/'evidence-collections-v2/batch-0029';home.mkdir(parents=True)
    slot={'batch':29,'token':'exact','owner':None,'feeder':{'path':'teacher_collection_feeder.py'}}
    monkeypatch.setattr(g,'recover_token',lambda *a,**k:None);monkeypatch.setattr(g,'alive',lambda x:False)
    monkeypatch.setattr(g.subprocess,'Popen',lambda *a,**k:pytest.fail('duplicate materialized feeder'))
    g._spawn_primary_prefetch(tmp_path,{}, {'root':str(tmp_path)},slot,12)


@pytest.mark.parametrize('value',[0,3,True,'2'])
def test_invalid_slot_limits_reject(value):
    with pytest.raises(ValueError):g._primary_overlap_settings({'primary_parallel_batches':value})


def test_started_reservation_keeps_original_policy_when_future_priorities_change(tmp_path,monkeypatch):
    from test_teacher_source_sequence import priority_setup
    folder,state,config=priority_setup(tmp_path,minimum=1)
    seq.prepare_source_sequence(folder,state,config,22);sealed=seal(folder,state,config,22)
    seq.observe_source_sequence(folder,state,config)
    state.update(owner={'pid':123,'start_ticks':'4','boot_id':'boot'},active_batch=22)
    monkeypatch.setattr(seq,'alive',lambda owner:bool(owner));config['primary_parallel_batches']=2;config['collection_repo']=str(tmp_path/'collection-repo')
    reservation=seq.prepare_prefetched_source(folder,state,config,23)
    assert reservation['ordinal']==3
    value=seq._json(seq._json(reservation['intent'],config['source_sequence'])['new_configuration'],config['source_sequence'])
    root=Path(config['root'])/'evidence-collections-v2/batch-0023';root.mkdir()
    (root/'source-cases.jsonl').write_bytes(Path(value['cases']).read_bytes())
    put(root/'queue.jsonl',{'actual':'queue'})
    put(root/'teacher-source.json',{'batch':23,'source_cases_sha256':value['cases_sha256'],'actual_task_runtime':config['task_runtime']})
    seq.observe_prefetched_source(folder,state,config)
    policy=seq._json(config['source_sequence']['priority_packets'],config['source_sequence'])
    # New unstarted priority4 becomes first, but the real already-started3
    # retains its original immutable selection, source and obligation.
    config['source_sequence']['priority_packets']=put(folder/'new-priority.json',{**policy,'entries':list(reversed(policy['entries']))})
    close_predecessor(folder,state,config,sealed)
    seq.adopt_prefetched_source(folder,state,config,23)
    assert state['source_sequence']['pending']==reservation
    assert seq._json(reservation['intent'],config['source_sequence'])['priority']['policy']!=config['source_sequence']['priority_packets']


@pytest.mark.parametrize('when',['before','after'])
def test_reserved_adoption_recovers_interrupted_configuration_publication(tmp_path,monkeypatch,when):
    folder,state,config,sealed,reservation=reserve(tmp_path,monkeypatch)
    close_predecessor(folder,state,config,sealed)
    original=seq._reconcile_publication
    def crash(*args):
        if when=='after':original(*args)
        raise RuntimeError('injected interrupted publication')
    monkeypatch.setattr(seq,'_reconcile_publication',crash)
    with pytest.raises(RuntimeError,match='interrupted'):
        seq.adopt_prefetched_source(folder,state,config,23)
    state=json.loads((folder/'state.json').read_text())
    monkeypatch.setattr(seq,'_reconcile_publication',original)
    # The production loop observes persisted state before adopting a slot.
    seq.observe_source_sequence(folder,state,config)
    seq.adopt_prefetched_source(folder,state,config,23)
    assert state['source_sequence']['pending']==reservation
    assert seq.binding(config['source_sequence']['source_configuration'])==state['source_sequence']['configuration']


def publisher_overlap_fixture():
    import hashlib
    from types import SimpleNamespace
    root='/work/generation';control='/work/control';remote={}
    def store(path,value):
        raw=value if isinstance(value,bytes)else json.dumps(value).encode()
        remote[path]=raw
        return {'path':path,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    feeder=store(control+'/overlap.py',b'# exact feeder\n')
    worker=store(control+'/run_bonsai_queue.py',b'# exact worker\n')
    context={'repo':'/work/source','queue_worker':worker,'node_bin':'/work/node/bin'}
    runtime='/work/runtime'
    guardian={'control':control,'root':root,'feeder':control+'/feed.py','task_runtime':runtime,
        'primary_parallel_batches':2,'overlap_feeder':feeder,'collection_repo':context['repo'],
        'queue_worker':worker['path'],'collection_worker':worker,'node_bin':context['node_bin'],
        'teacher_identity':{'sha256':'b'*64},'source_sequence':{'anchor':{'batch':21,'cases_sha256':'a'*64}}}
    config=store(control+'/config.json',guardian)
    historical=store(control+'/config-history/'+config['sha256']+'.json',remote[config['path']])
    state={'schema':'natlang.teacher-generation-guardian/1','config':config,
        'owner':{'pid':42,'start_ticks':'777','boot_id':'boot'},'active_batch':22,
        'source_sequence':{'started':{}}}
    for batch in(22,23):
        home=f'{root}/evidence-collections-v2/batch-{batch:04d}'
        cases=store(home+'/source-cases.jsonl',json.dumps({'program_ir':{'id':f'program-{batch}'}}).encode()+b'\n')
        entry={'key':f'task-{batch}','index':0,'count':1,'sample':0,'seed':batch,'task_id':f'program-{batch}',
            'source':cases['path'],'output':home+'/out/0/results.jsonl','jobs':home+'/out/0/jobs'}
        queue=store(home+'/queue.jsonl',json.dumps(entry).encode()+b'\n')
        teacher=store(home+'/teacher-source.json',{'schema':'natlang.sandbox-strong-teacher-collection/1','batch':batch,
            'source_cases_sha256':cases['sha256'],'actor':{'sha256':'a'*64},'teacher':{'model':'qwen-test'},
            'actual_teacher_execution_identity':{'sha256':'b'*64},'actual_task_runtime':runtime,
            'collection_controller':{'sha256':'c'*64},'runtime_source_receipt':{'sha256':'d'*64},
            'teacher_configuration_source':{'sha256':'e'*64},'source_provenance':{'sha256':'f'*64}})
        candidate=store(control+f'/candidate-{batch}.json',{'cases':cases['path'],'cases_sha256':cases['sha256']})
        intent=store(control+f'/intent-{batch}.json',{'batch':batch,'ordinal':batch-22,'new_configuration':candidate,
            'prefetch':{'task_runtime':runtime}})
        started=store(control+f'/source-sequence/started/batch-{batch:04d}.json',{'batch':batch,'ordinal':batch-22,
            'sealed':{'source-cases.jsonl':cases,'queue.jsonl':queue,'teacher-source.json':teacher},'selection_intent':intent})
        state['source_sequence']['started'][str(batch)]=started
        phase={'phase':'collecting','batch':batch,'worker_count':1,'worker_pids':[batch+500],
            'out':home,'cases':1,'teacher':'qwen-test'}
        phasepath=root+'/feed-evidence-phase.json'if batch==22 else root+f'/feeder-phases/batch-{batch:04d}.json'
        store(phasepath,phase);store(home+'/queue-worker-0.jsonl',remote[queue['path']]);store(home+'/journal-worker-0.jsonl',b'')
        if batch==23:
            slot={'batch':23,'owner':{'pid':43,'start_ticks':'888','boot_id':'boot'},'feeder':feeder,
                'guardian_config':historical,'collection_context':context,'task_runtime':runtime,'phase_path':phasepath,'selection_intent':intent,'source_configuration':candidate}
            state['primary_prefetch']=slot
            state['source_sequence']['prefetch']={'batch':23,'intent':intent}
    store(control+'/state.json',state);remote['/proc/sys/kernel/random/boot_id']=b'boot\n'
    probes=[]
    def execute(argv,**kwargs):
        if argv[0]=='cat':out='boot\n'
        elif 'glob'in argv[2]:out=json.dumps({'queues':['queue-worker-0.jsonl'],'journals':['journal-worker-0.jsonl']})
        else:
            probes.append(argv)
            expected=state['owner']if argv[-1]=='22'else state['primary_prefetch']['owner']
            assert argv[-5:]==[str(expected['pid']),expected['start_ticks'],'boot',guardian['feeder']if argv[-1]=='22'else feeder['path'],argv[-1]]
            out=json.dumps({'valid':True})
        return SimpleNamespace(result=lambda **_:SimpleNamespace(returncode=0,stdout=out,stderr=''))
    publisher={'schema':'natlang.teacher-incremental-export-publisher/1','sandbox_id':'sid','sandbox_boot_id':'boot',
        'guardian_config':config,'control_dir':control,'generation_root':root,'anchor_batch':21,'anchor_cases_sha256':'a'*64,
        'directory_root':'/external/export','role':'bulk','chunk_size':32,'exporter_path':'/owner/export.py',
        'identity':{'sandbox_boot_id':'boot','actor_sha256':'a'*64,'execution_identity_sha256':'b'*64,
            'teacher_config_sha256':'e'*64,'runtime_path':runtime}}
    return publisher,SimpleNamespace(exec=execute),remote,state,store,probes


def test_publisher_two_slots_keep_exact_sources_processes_and_directories_separate():
    from scripts.coreweave.teacher_incremental_export import discover_guardian_batch
    publisher,sandbox,remote,state,store,probes=publisher_overlap_fixture()
    first=discover_guardian_batch(publisher,sandbox,remote.__getitem__)
    second=discover_guardian_batch(publisher,sandbox,remote.__getitem__,batch_slot=23)
    assert first['batch']==22 and second['batch']==23
    assert first['queue_path']!=second['queue_path']and first['directory']!=second['directory']
    assert first['identity']['source_cases_sha256']!=second['identity']['source_cases_sha256']
    assert [argv[-1]for argv in probes]==['22','23']
    # Fresh process restart reconstructs the same exact custody without relying
    # on a cached local discovery object.
    assert discover_guardian_batch(publisher,sandbox,remote.__getitem__,batch_slot=23)==second


@pytest.mark.parametrize('tamper',['third','source','feeder','runtime','owner_boot','intent'])
def test_publisher_overlap_never_grants_wrong_slot_or_source_custody(tamper):
    from scripts.coreweave.teacher_incremental_export import discover_guardian_batch
    publisher,sandbox,remote,state,store,_=publisher_overlap_fixture()
    slot=state['primary_prefetch'];batch=23
    if tamper=='third':batch=24
    elif tamper=='source':remote['/work/generation/evidence-collections-v2/batch-0023/source-cases.jsonl']=b'foreign source'
    elif tamper=='feeder':remote[slot['feeder']['path']]=b'changed'
    elif tamper=='runtime':slot['task_runtime']='/wrong/runtime'
    elif tamper=='owner_boot':slot['owner']['boot_id']='wrong'
    else:slot['selection_intent']['sha256']='0'*64
    store('/work/control/state.json',state)
    with pytest.raises(ValueError):discover_guardian_batch(publisher,sandbox,remote.__getitem__,batch_slot=batch)


def test_actual_owned_feeder_is_recovered_after_guardian_owner_publication_gap(tmp_path,monkeypatch):
    import os,time,subprocess,sys
    monkeypatch.setenv('PATH',str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''))
    feeder=tmp_path/'owned-overlap-feeder.py';feeder.write_text('import time;time.sleep(30)\n')
    candidate=put(tmp_path/'source.json',{'actual':'immutable candidate'})
    slot={'batch':23,'token':'exact-overlap-fixture-'+str(os.getpid()),'owner':None,
        'feeder':g.binding(feeder),'source_configuration':candidate,'task_runtime':'/actual/runtime'}
    config={'root':str(tmp_path),'collection_repo':str(tmp_path),'queue_worker':str(tmp_path/'run_bonsai_queue.py'),'node_bin':str(Path(sys.executable).parent),'teacher_identity':{'path':'/fixture/identity'},'server':'http://fixture','model_id':'fixture'}
    worker=Path(config['queue_worker']);worker.write_text('# actual worker\n')
    slot['collection_context']={'repo':config['collection_repo'],'queue_worker':g.binding(worker),'node_bin':config['node_bin']}
    state={'owner':None};children=[];actual_popen=g.subprocess.Popen
    def launch(*args,**kwargs):
        child=actual_popen(*args,**kwargs);children.append(child);return child
    monkeypatch.setattr(g.subprocess,'Popen',launch)
    monkeypatch.setattr(g,'verify_teacher_identity',lambda *a,**k:None)
    try:
        g._spawn_primary_prefetch(tmp_path,state,config,slot,12)
        assert len(children)==1 and g.alive(slot['owner'])
        original=slot['owner'];slot['owner']=None
        time.sleep(.05)
        g._spawn_primary_prefetch(tmp_path,state,config,slot,12)
        assert len(children)==1 and slot['owner']==original
        env=Path(f"/proc/{original['pid']}/environ").read_bytes().split(b'\0')
        assert ('NATLANG_TEACHER_SOURCE_CONFIGURATION_SHA256='+candidate['sha256']).encode()in env
        assert b'NATLANG_TEACHER_START_BATCH=23'in env
        # This exact child remains owned; no other feeder is signalled.
    finally:
        for child in children:
            child.terminate();child.wait(timeout=3)


def feeder_loader():
    # Execute the actual pure function from the production feeder without
    # importing its real server/paths or manufacturing another implementation.
    import ast,hashlib
    path=Path(g.__file__).with_name('teacher_collection_feeder.py')
    tree=ast.parse(path.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='load_source_configuration')
    namespace={'pathlib':__import__('pathlib'),'hashlib':hashlib,'json':json}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['load_source_configuration']


@pytest.mark.parametrize('mutation',[None,'config','cases','count','missing_digest'])
def test_feeder_reads_exact_reserved_configuration_and_cases(tmp_path,mutation):
    load=feeder_loader();cases=tmp_path/'cases.jsonl';cases.write_text('{"id":"actual"}\n')
    value={'cases':str(cases),'cases_sha256':g.binding(cases)['sha256'],'rows':1}
    if mutation=='count':value['rows']=2
    ref=put(tmp_path/'source.json',value)
    digest=ref['sha256'];size=ref['bytes']
    if mutation=='config':Path(ref['path']).write_text('{}')
    elif mutation=='cases':cases.write_text('{"id":"foreign"}\n')
    elif mutation=='missing_digest':digest=None
    if mutation:
        with pytest.raises(ValueError):load(ref['path'],digest,size)
    else:assert load(ref['path'],digest,size)==value


def test_actual_publisher_service_exports_two_independent_frontiers_and_resumes(tmp_path,monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from scripts.coreweave import teacher_incremental_export as module,source_archive
    publisher,sandbox,remote,state,store,_=publisher_overlap_fixture()
    publisher.update(directory_root=str(tmp_path/'exports'),chunk_size=1)
    path=tmp_path/'publisher.json';path.write_text(json.dumps(publisher))
    for batch in(22,23):
        home=f'/work/generation/evidence-collections-v2/batch-{batch:04d}'
        entry=json.loads(remote[home+'/queue.jsonl'])
        payload=json.dumps({'trajectory':[{'task':entry['task_id'],'batch':batch}]}).encode()+b'\n'
        remote[entry['output']]=payload
        store(entry['output']+'.manifest.json',{'version':'natlang.teacher_batch.native/1',
            'source':entry['source'],'source_sha256':hashlib.sha256(remote[entry['source']]).hexdigest(),
            'range':{'start':entry['index'],'count':1},'completed':1,'missing':[],
            'output_sha256':hashlib.sha256(payload).hexdigest()})
        finish={'event':'finish','key':entry['key'],'output_accounting':{
            'version':'natlang.supervisor_output_accounting/2','manifest_valid':True,'manifest_errors':[],
            'complete':True,'expected_jobs':1,'exact_result_rows':1,'output_sha256':hashlib.sha256(payload).hexdigest(),
            'job_states':[{'state':'result'}]}}
        remote[home+'/journal-worker-0.jsonl']=b''.join(json.dumps(row).encode()+b'\n'for row in(
            {'event':'start','key':entry['key']},finish))
    monkeypatch.setattr(module,'_cwctl_module',lambda:SimpleNamespace(load_env_file=lambda:None))
    monkeypatch.setattr(module,'scoped_sdk_session',lambda _: (sandbox,remote.__getitem__))
    actual_pack=source_archive.pack
    monkeypatch.setattr(source_archive,'pack',lambda *a,**k:actual_pack(*a,**{**k,'reserve_bytes':0}))
    uploaded=[]
    def export(tranche,*,exporter,role,receipt_path):
        # Exercise real archives/receipts/frontier durability; local streamed
        # readback stands in for provider transport, never for training admission.
        raw=Path(tranche['archive_receipt']['archive']).read_bytes()
        identity={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        assert identity=={'bytes':tranche['archive_receipt']['archive_bytes'],
                          'sha256':tranche['archive_receipt']['archive_sha256']}
        receipt_path.parent.mkdir(parents=True,exist_ok=True)
        receipt_path.write_text(json.dumps({'private':True,'run':'test/run','name':'archive',
            'identity':identity,'independent_verification':identity,'verified_at':123.0}))
        uploaded.append(tranche['batch'])
        return module.accept_private_receipt(receipt_path,tranche)
    monkeypatch.setattr(module,'export_private_tranche',export)
    result=module.run_guardian_export_service(path,poll_seconds=0,max_polls=2,allowed_spool_root=tmp_path)
    assert uploaded==[22,23]  # second poll reuses each durable exact receipt
    for batch in(22,23):
        frontier=json.loads((tmp_path/f'exports/batch-{batch:04d}/frontier.json').read_text())
        assert frontier['batch']==batch and frontier['completed_keys']==[f'task-{batch}']
        assert len(frontier['tranches'])==1
    assert result['active_identity']['batch']==22
    assert result['overlap_batches']['23']['active_identity']['batch']==23
    assert result['closed_boundaries']=={}
    assert not result['training_admission']and not result['exposure']


def test_tail_observation_yields_on_inflight_partial_journal_record(tmp_path):
    home,journal=tail(tmp_path)
    with journal.open('ab')as stream:stream.write(b'{"event":"finish","key":')
    assert g.primary_tail_frontier(tmp_path,28,6)is None


def test_finished_early_next_batch_keeps_own_boundary_until_predecessor_closes(tmp_path,monkeypatch):
    folder,state,config,sealed=active(tmp_path,monkeypatch)
    reservation=seq.prepare_prefetched_source(folder,state,config,23)
    intent=seq._json(reservation['intent'],config['source_sequence']);value=seq._json(intent['new_configuration'],config['source_sequence'])
    home=Path(config['root'])/'evidence-collections-v2/batch-0023';home.mkdir()
    (home/'source-cases.jsonl').write_bytes(Path(value['cases']).read_bytes())
    entry={'key':'task-23','index':0,'count':1,'seed':23000,'source':str(home/'source-cases.jsonl'),
        'jobs':str(home/'jobs'),'output':str(home/'results.jsonl')}
    (home/'queue.jsonl').write_text(json.dumps(entry)+'\n')
    put(home/'teacher-source.json',{'batch':23,'source_cases_sha256':value['cases_sha256'],'actual_task_runtime':config['task_runtime']})
    finish={'event':'finish','key':entry['key'],'output_accounting':{
        'version':'natlang.supervisor_output_accounting/2','manifest_valid':True,
        'manifest_errors':[],'expected_jobs':1,'complete':True,'job_states':[{'state':'result'}]}}
    (home/'journal.jsonl').write_text(json.dumps(finish)+'\n')
    put(home/'score.json',{'schema':'natlang.s7-task-rollout-score/1','rollouts':1,'outcome_counts':{'task_success':1}})
    phase=Path(config['root'])/'feeder-phases/batch-0023.json'
    put(phase,{'phase':'batch_complete','batch':23,'returncodes':[0]})
    feeder=tmp_path/'overlap.py';feeder.write_text('# fixture feeder\n')
    config['overlap_feeder']=g.binding(feeder)
    slot={'batch':23,'owner':None,'token':'never-spawn-materialized','workers':12,
        'phase_path':str(phase),'feeder':config['overlap_feeder'],'task_runtime':config['task_runtime'],
        'source_configuration':intent['new_configuration'],'selection_intent':reservation['intent']}
    state['primary_prefetch']=slot
    g.maintain_primary_prefetch(folder,state,config)
    ref=slot['boundary'];assert json.loads(Path(ref['path']).read_text())['batch_complete']is True
    assert state['source_sequence']['pending']['batch']==22
    assert '23'not in state['source_sequence'].get('closed',{})
    assert state['source_sequence']['next_ordinal']==0
    # A guardian restart reuses the immutable terminal boundary, never spawns
    # another feeder or invents closure of the still-running predecessor.
    g.maintain_primary_prefetch(folder,state,config);assert slot['boundary']==ref
    close_predecessor(folder,state,config,sealed);state['next_batch']=23
    g.adopt_primary_prefetch(folder,state,config)
    assert state['active_batch']==23 and state['source_sequence']['pending']==reservation
    state['last_boundary']=ref;seq.observe_source_sequence(folder,state,config)
    assert set(state['source_sequence']['closed'])=={'22','23'}
    assert state['source_sequence']['next_ordinal']==2


def test_predecessor_exit_recovers_unpublished_next_owner_and_drain_blocks_new_spawn(tmp_path,monkeypatch):
    folder,state,config,sealed=active(tmp_path,monkeypatch)
    reserved=seq.prepare_prefetched_source(folder,state,config,23)
    intent=seq._json(reserved['intent'],config['source_sequence'])
    feeder=tmp_path/'feeder.py';feeder.write_text('# exact fixture\n')
    config.update(overlap_feeder=g.binding(feeder),drain_file=str(tmp_path/'DRAIN'))
    slot={'batch':23,'token':'token-before-owner-write','owner':None,'workers':12,
        'feeder':config['overlap_feeder'],'task_runtime':config['task_runtime'],
        'selection_intent':reserved['intent'],'source_configuration':intent['new_configuration']}
    state['primary_prefetch']=slot
    close_predecessor(folder,state,config,sealed);state['next_batch']=23
    Path(config['drain_file']).touch()
    monkeypatch.setattr(g,'recover_token',lambda *a,**kw:None)
    monkeypatch.setattr(g,'alive',lambda owner:bool(owner))
    monkeypatch.setattr(g.subprocess,'Popen',lambda *a,**kw:pytest.fail('DRAIN must not spawn a new feeder'))
    assert not g.adopt_primary_prefetch(folder,state,config)
    assert state['primary_prefetch']==slot and state.get('owner')is None
    recovered={'pid':55,'start_ticks':'44','boot_id':'boot','argv':[b'python',str(feeder).encode()]}
    monkeypatch.setattr(g,'recover_token',lambda *a,**kw:recovered)
    # Existing work is recovered even under DRAIN; it is never replaced by a
    # second spawn and it retains the original exact source reservation.
    assert g.adopt_primary_prefetch(folder,state,config)
    assert state['owner']==g.public_identity(recovered)
    assert state['source_sequence']['pending']==reserved


@pytest.mark.parametrize('maximum,expected_next',[(1,0),(2,4)])
def test_live_priority_predecessor_counts_service_without_fake_closure(tmp_path,monkeypatch,maximum,expected_next):
    from test_teacher_source_sequence import priority_setup
    folder,state,config=priority_setup(tmp_path,minimum=0)
    decl=config['source_sequence'];proof=seq._json(decl['priority_packets'],decl)
    decl['priority_packets']=put(folder/'priority-limits.json',{**proof,'max_consecutive':maximum})
    seq.prepare_source_sequence(folder,state,config,22)
    assert state['source_sequence']['pending']['ordinal']==3
    sealed=seal(folder,state,config,22);seq.observe_source_sequence(folder,state,config)
    state.update(owner={'pid':123,'start_ticks':'4','boot_id':'boot'},active_batch=22)
    monkeypatch.setattr(seq,'alive',lambda owner:bool(owner));config['primary_parallel_batches']=2;config['collection_repo']=str(tmp_path/'collection-repo')
    reserved=seq.prepare_prefetched_source(folder,state,config,23)
    assert reserved['ordinal']==expected_next
    assert state['source_sequence'].get('closed',{})=={}
    assert state['source_sequence'].get('priority_streak',0)==0
    intent=seq._json(reserved['intent'],decl);value=seq._json(intent['new_configuration'],decl)
    home=Path(config['root'])/'evidence-collections-v2/batch-0023';home.mkdir()
    (home/'source-cases.jsonl').write_bytes(Path(value['cases']).read_bytes())
    put(home/'queue.jsonl',{'actual':'reserved queue'})
    put(home/'teacher-source.json',{'batch':23,'source_cases_sha256':value['cases_sha256'],'actual_task_runtime':config['task_runtime']})
    seq.observe_prefetched_source(folder,state,config)
    close_predecessor(folder,state,config,sealed)
    seq.adopt_prefetched_source(folder,state,config,23)
    assert state['source_sequence']['pending']==reserved
    assert state['source_sequence']['priority_streak']==1
    assert set(state['source_sequence']['closed'])=={'22'}
    if maximum==1:assert 'priority'not in intent  # bounded ordinary service remains mandatory
    else:assert intent['priority']['resume_ordinal']==0


def test_prefetch_start_rejects_another_batch_header_even_with_same_case_bytes(tmp_path,monkeypatch):
    folder,state,config,sealed,reserved=reserve(tmp_path,monkeypatch)
    header=Path(config['root'])/'evidence-collections-v2/batch-0023/teacher-source.json'
    value=json.loads(header.read_text());value['batch']=24;put(header,value)
    with pytest.raises(ValueError,match='reserved batch'):
        seq.observe_prefetched_source(folder,state,config)


def test_started_overlap_keeps_original_feeder_snapshot_across_future_helper_revision():
    from scripts.coreweave.teacher_incremental_export import discover_guardian_batch
    publisher,sandbox,remote,state,store,_=publisher_overlap_fixture()
    path=publisher['guardian_config']['path'];guardian=json.loads(remote[path])
    guardian['overlap_feeder']=store('/work/control/future-overlap.py',b'# future helper\n')
    latest=store(path,guardian);publisher['guardian_config']=latest;state['config']=latest
    store('/work/control/state.json',state)
    result=discover_guardian_batch(publisher,sandbox,remote.__getitem__,batch_slot=23)
    assert result['batch']==23
    # The running owner still uses its actual prior helper, not the future one.
    ref=state['primary_prefetch']['guardian_config'];remote[ref['path']]=b'changed snapshot'
    with pytest.raises(ValueError,match='snapshot changed'):
        discover_guardian_batch(publisher,sandbox,remote.__getitem__,batch_slot=23)
