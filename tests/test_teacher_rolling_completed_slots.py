"""Release completed collection capacity while preserving ordered cursor custody."""
import copy
import json
from pathlib import Path

import pytest
from scripts.coreweave import teacher_generation_guardian as g
from scripts.coreweave import teacher_source_sequence as seq
from scripts.coreweave.teacher_incremental_export import _guardian_lookahead_slots
from test_teacher_primary_overlap import reserve,close_predecessor
from test_teacher_source_sequence import put


def finish_slot(folder,state,config,reserved):
    batch=reserved['batch'];started=seq._json(state['source_sequence']['started'][str(batch)],config['source_sequence'])
    ref=put(folder/'boundaries'/f'batch-{batch:04d}.json',{'batch':batch,'batch_complete':True,
        'bindings':{name:started['sealed'][name]for name in ('queue.jsonl','teacher-source.json')}})
    slot={'batch':batch,'owner':None,'selection_intent':reserved['intent'],'boundary':ref}
    state['primary_prefetch']=slot
    seq.archive_completed_prefetch(folder,state,config,slot)
    return slot


def materialize(folder,state,config,reserved):
    batch=reserved['batch'];decl=config['source_sequence']
    candidate=seq._json(seq._json(reserved['intent'],decl)['new_configuration'],decl)
    root=Path(config['root'])/'evidence-collections-v2'/f'batch-{batch:04d}';root.mkdir()
    (root/'source-cases.jsonl').write_bytes(Path(candidate['cases']).read_bytes())
    put(root/'queue.jsonl',{'actual':f'queue-{batch}'})
    put(root/'teacher-source.json',{'batch':batch,'source_cases_sha256':candidate['cases_sha256'],
        'actual_task_runtime':config['task_runtime']})
    seq.observe_prefetched_source(folder,state,config)


def test_multiple_completed_slots_release_capacity_and_adopt_original_order(tmp_path,monkeypatch):
    folder,state,config,sealed,first=reserve(tmp_path,monkeypatch)
    owner=copy.deepcopy(state['owner']);ambient=Path(config['source_sequence']['source_configuration']).read_bytes()
    finish_slot(folder,state,config,first)
    second=seq.prepare_prefetched_source(folder,state,config,24)
    assert second['ordinal']==2
    materialize(folder,state,config,second);finish_slot(folder,state,config,second)
    third=seq.prepare_prefetched_source(folder,state,config,25)
    assert third['ordinal']==3
    assert state['owner']==owner and state['active_batch']==22
    assert Path(config['source_sequence']['source_configuration']).read_bytes()==ambient
    assert state['source_sequence']['next_ordinal']==0
    assert set(state['primary_completed_ahead'])=={'23','24'}
    # Restart retains every committed reservation; it never selects a new packet.
    state=json.loads((folder/'state.json').read_text())
    assert seq.prepare_prefetched_source(folder,state,config,25)==third
    close_predecessor(folder,state,config,sealed)
    for reserved in (first,second):
        batch=reserved['batch'];seq.adopt_prefetched_source(folder,state,config,batch)
        assert state['source_sequence']['pending']==reserved
        assert state['source_sequence']['prefetch']==third
        state['last_boundary']=state['primary_completed_ahead'][str(batch)]['boundary']
        seq.observe_source_sequence(folder,state,config)
        state['primary_completed_ahead'].pop(str(batch))
    assert set(state['source_sequence']['closed'])=={'22','23','24'}
    assert state['source_sequence']['next_ordinal']==3
    assert state['source_sequence']['prefetch']==third


@pytest.mark.parametrize('tamper',['gap','boundary_bytes','wrong_boundary_batch','source','live_owner','reservation'])
def test_completed_ahead_custody_never_releases_foreign_or_unclosed_work(tmp_path,monkeypatch,tamper):
    folder,state,config,_,reserved=reserve(tmp_path,monkeypatch)
    slot=finish_slot(folder,state,config,reserved)
    if tamper=='gap':state['primary_completed_ahead']['24']=state['primary_completed_ahead'].pop('23')
    elif tamper=='boundary_bytes':Path(slot['boundary']['path']).write_text('{}')
    elif tamper=='wrong_boundary_batch':slot['boundary']=put(folder/'boundaries/batch-0023.json',{'batch':24,'batch_complete':True})
    elif tamper=='source':(Path(config['root'])/'evidence-collections-v2/batch-0023/source-cases.jsonl').write_text('changed')
    elif tamper=='live_owner':slot['owner']={'pid':8}
    else:state['source_sequence']['prefetch_history']['23']['intent']['sha256']='0'*64
    with pytest.raises((ValueError,KeyError)):
        seq.prepare_prefetched_source(folder,state,config,24)
    assert state['source_sequence']['pending']['batch']==22
    assert 'prefetch'not in state['source_sequence']


def test_publisher_accepts_completed_distance_but_rejects_missing_slot_custody():
    state={'active_batch':33,'primary_completed_ahead':{'34':{'batch':34,'boundary':{'path':'bound34'}}},
        'primary_prefetch':{'batch':35}}
    assert list(_guardian_lookahead_slots(state))==[34,35]
    # Atomic primary closure may precede immediate adoption of completed34.
    state.update(active_batch=None,next_batch=34)
    assert list(_guardian_lookahead_slots(state))==[34,35]
    for changed in ({'primary_completed_ahead':{}},{'primary_prefetch':{'batch':36}},
                    {'primary_completed_ahead':{'34':{'batch':34}}}):
        with pytest.raises(ValueError):_guardian_lookahead_slots({**state,**changed})


def test_guardian_releases_completed_pool_and_launches_next_without_signalling_old_owner(tmp_path,monkeypatch):
    folder,state,config,_,reserved=reserve(tmp_path,monkeypatch)
    first_owner=copy.deepcopy(state['owner']);completed=finish_slot(folder,state,config,reserved)
    # Exercise the actual guardian transition, starting immediately before its
    # durable archive rather than bypassing the production orchestration.
    state['source_sequence']['prefetch']=state['source_sequence']['prefetch_history'].pop('23')
    state.pop('primary_completed_ahead');state['primary_prefetch']=completed
    phase=folder/'phase23.json';put(phase,{'phase':'batch_complete','batch':23})
    feeder=folder/'feeder.py';feeder.write_text('# immutable feeder\n')
    completed.update(phase_path=str(phase),workers=24,token='original23',feeder=g.binding(feeder))
    config.update(primary_parallel_batches=2,primary_tail_max_running=6,overlap_workers=24,
        overlap_feeder=g.binding(feeder),drain_file=str(folder/'DRAIN'))
    state['config_snapshot']=put(folder/'config-snapshot.json',config)
    launches=[]
    def spawn(folder_,state_,config_,slot,workers):
        launches.append((slot['batch'],workers))
        if slot['batch']==24:slot['owner']={'pid':124,'start_ticks':'new','boot_id':'boot'}
    monkeypatch.setattr(g,'_spawn_primary_prefetch',spawn)
    monkeypatch.setattr(g,'alive',lambda owner:bool(owner))
    monkeypatch.setattr(g,'_source_sequence_helpers',lambda:seq)
    monkeypatch.setattr(g,'boundary',lambda *a,**k:(seq._json(completed['boundary'],config['source_sequence']),[]))
    monkeypatch.setattr(g,'prepare_retry',lambda *a:None)
    monkeypatch.setattr(g,'primary_tail_frontier',lambda *a:{'active_keys':['original-long-tail']})
    g.maintain_primary_prefetch(folder,state,config)
    assert launches==[(23,24),(24,24)]
    assert state['owner']==first_owner and state['active_batch']==22
    assert state['primary_prefetch']['batch']==24
    assert state['primary_completed_ahead']['23']['boundary']==completed['boundary']
    # Restart reuses the exact new pool and does not archive/reselect old23 again.
    materialize(folder,state,config,state['source_sequence']['prefetch'])
    g.maintain_primary_prefetch(folder,state,config)
    assert launches==[(23,24),(24,24),(24,24)]
    assert state['source_sequence']['prefetch']['ordinal']==2


def test_guardian_archived_adoption_retains_later_live_pool(tmp_path,monkeypatch):
    folder,state,config,sealed,first=reserve(tmp_path,monkeypatch)
    completed=finish_slot(folder,state,config,first)
    feeder=folder/'feeder.py';feeder.write_text('# feeder\n')
    completed.update(token='retained23',workers=24,feeder=g.binding(feeder),task_runtime=config['task_runtime'])
    later=seq.prepare_prefetched_source(folder,state,config,24)
    later_slot={'batch':24,'owner':{'pid':124}}
    state['primary_prefetch']=later_slot
    close_predecessor(folder,state,config,sealed)
    state.update(active_batch=None,next_batch=23)
    monkeypatch.setattr(g,'_spawn_primary_prefetch',lambda *a:None)
    monkeypatch.setattr(g,'alive',lambda owner:bool(owner))
    monkeypatch.setattr(g,'_source_sequence_helpers',lambda:seq)
    assert g.adopt_primary_prefetch(folder,state,config)
    assert state['active_batch']==23 and state['owner']is None
    assert state['primary_prefetch']==later_slot
    assert not state['primary_completed_ahead']
    assert state['source_sequence']['prefetch']==later


def publisher_with_completed_and_later_slot():
    from types import SimpleNamespace
    from test_teacher_primary_overlap import publisher_overlap_fixture
    publisher,_,remote,state,store,_=publisher_overlap_fixture()
    old=state['primary_prefetch'];root='/work/generation';control='/work/control'
    boundary=store(control+'/boundaries/batch-0023.json',{'batch':23})
    old['boundary']=boundary;state['primary_completed_ahead']={'23':old}
    home=root+'/evidence-collections-v2/batch-0024'
    cases=store(home+'/source-cases.jsonl',b'{"program_ir":{"id":"program-24"}}\n')
    original_home=root+'/evidence-collections-v2/batch-0023'
    entry=json.loads(remote[original_home+'/queue.jsonl']);entry.update(key='task-24',task_id='program-24',
        source=cases['path'],output=home+'/out/0/results.jsonl',jobs=home+'/out/0/jobs',seed=24)
    queue=store(home+'/queue.jsonl',json.dumps(entry).encode()+b'\n')
    teacher=json.loads(remote[original_home+'/teacher-source.json']);teacher.update(batch=24,source_cases_sha256=cases['sha256'])
    source=store(home+'/teacher-source.json',teacher)
    candidate=store(control+'/candidate-24.json',{'cases':cases['path'],'cases_sha256':cases['sha256']})
    intent=store(control+'/intent-24.json',{'batch':24,'ordinal':2,'new_configuration':candidate,'prefetch':{'task_runtime':old['task_runtime']}})
    started=store(control+'/source-sequence/started/batch-0024.json',{'batch':24,'ordinal':2,
        'sealed':{'source-cases.jsonl':cases,'queue.jsonl':queue,'teacher-source.json':source},'selection_intent':intent})
    phase=root+'/feeder-phases/batch-0024.json';store(phase,{'phase':'collecting','batch':24,'worker_count':1,
        'worker_pids':[524],'out':home,'cases':1,'teacher':'qwen-test'})
    store(home+'/queue-worker-0.jsonl',remote[queue['path']]);store(home+'/journal-worker-0.jsonl',b'')
    state['primary_prefetch']={**old,'batch':24,'owner':{'pid':44,'start_ticks':'999','boot_id':'boot'},
        'phase_path':phase,'selection_intent':intent,'source_configuration':candidate}
    state['primary_prefetch'].pop('boundary')
    state['source_sequence']['prefetch_history']={'23':state['source_sequence']['prefetch']}
    state['source_sequence']['prefetch']={'batch':24,'intent':intent}
    state['source_sequence']['started']['24']=started
    store(control+'/state.json',state)
    def execute(argv,**kwargs):
        if argv[0]=='cat':out='boot\n'
        elif 'glob'in argv[2]:out=json.dumps({'queues':['queue-worker-0.jsonl'],'journals':['journal-worker-0.jsonl']})
        elif argv[-1]=='23':out=json.dumps({'valid':False,'missing':True,'boot_id':'boot'})
        else:
            assert argv[-5:]==['44','999','boot',old['feeder']['path'],'24']
            out=json.dumps({'valid':True})
        return SimpleNamespace(result=lambda **_:SimpleNamespace(returncode=0,stdout=out,stderr=''))
    return publisher,SimpleNamespace(exec=execute),remote,state,store


def test_actual_publisher_discovers_farther_live_slot_and_exact_completed_boundary():
    from scripts.coreweave.teacher_incremental_export import discover_guardian_batch
    publisher,sandbox,remote,state,store=publisher_with_completed_and_later_slot()
    old=discover_guardian_batch(publisher,sandbox,remote.__getitem__,batch_slot=23)
    assert old['draining']and old['verified_owner_transition']
    assert old['closed_boundaries']==[state['primary_completed_ahead']['23']['boundary']]
    current=discover_guardian_batch(publisher,sandbox,remote.__getitem__,batch_slot=24)
    assert current['batch']==24 and current['identity']['source_cases_sha256']!=old.get('identity',{}).get('source_cases_sha256')
    assert current['queue_path'].endswith('batch-0024/queue.jsonl')


@pytest.mark.parametrize('tamper',['runtime','boundary','gap','source'])
def test_actual_publisher_rejects_corrupt_distant_slot_authority(tamper):
    from scripts.coreweave.teacher_incremental_export import discover_guardian_batch
    publisher,sandbox,remote,state,store=publisher_with_completed_and_later_slot()
    if tamper=='runtime':state['primary_prefetch']['task_runtime']='/wrong/runtime'
    elif tamper=='boundary':remote[state['primary_completed_ahead']['23']['boundary']['path']]=b'changed'
    elif tamper=='gap':state['primary_completed_ahead']={}
    else:remote['/work/generation/evidence-collections-v2/batch-0024/source-cases.jsonl']=b'wrong'
    store('/work/control/state.json',state)
    with pytest.raises(ValueError):
        discover_guardian_batch(publisher,sandbox,remote.__getitem__,batch_slot=23 if tamper=='boundary'else 24)


def test_primary_natural_exit_during_archive_preserves_completed_and_defers_new_launch(tmp_path,monkeypatch):
    folder,state,config,sealed,reserved=reserve(tmp_path,monkeypatch)
    state['owner']=None
    finish_slot(folder,state,config,reserved)
    assert state['primary_completed_ahead']['23']['batch']==23
    assert state['source_sequence']['pending']['batch']==22
    with pytest.raises(ValueError,match='live predecessor'):
        seq.prepare_prefetched_source(folder,state,config,24)
    # Existing main-loop closure/adoption can now continue in exact source order.
    close_predecessor(folder,state,config,sealed)
    assert seq.adopt_prefetched_source(folder,state,config,23)==reserved


def test_archive_crash_after_boundary_before_state_publication_reuses_original_reservation(tmp_path,monkeypatch):
    folder,state,config,_,reserved=reserve(tmp_path,monkeypatch)
    seq.write(folder/'state.json',state)
    original=seq.write
    def crash(path,value):
        if Path(path)==folder/'state.json':raise RuntimeError('injected archive publication crash')
        return original(path,value)
    monkeypatch.setattr(seq,'write',crash)
    with pytest.raises(RuntimeError,match='publication crash'):
        finish_slot(folder,state,config,reserved)
    state=json.loads((folder/'state.json').read_text())
    assert state['source_sequence']['prefetch']==reserved
    assert 'primary_completed_ahead'not in state
    # The already-written boundary is immutable; replay consumes it once.
    monkeypatch.setattr(seq,'write',original)
    finish_slot(folder,state,config,reserved)
    next_reserved=seq.prepare_prefetched_source(folder,state,config,24)
    assert next_reserved['ordinal']==2
    assert state['source_sequence']['prefetch_history']['23']==reserved
