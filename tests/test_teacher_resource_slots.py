"""Recover real sealed multi-tail source custody without executing a model."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from scripts.coreweave import teacher_generation_guardian as g
from scripts.coreweave import teacher_source_sequence as seq
from scripts.coreweave import teacher_incremental_export as export
from test_teacher_source_sequence import setup, put


def pool(tmp_path,monkeypatch,*,ahead=3,work_items=8,active_count=1,unstarted=1):
    folder,state,config=setup(tmp_path,packet_count=9)
    feeder=tmp_path/'feed-evidence-owned.py';feeder.write_text('# owned feeder\n')
    worker=tmp_path/'run_bonsai_queue.py';worker.write_text('# owned worker\n')
    config.update(primary_parallel_batches=2,primary_worker_capacity=48,overlap_workers=24,
        primary_tail_max_running=6,feeder=str(feeder),overlap_feeder=g.binding(feeder),
        collection_repo=str(tmp_path),collection_worker=g.binding(worker),queue_worker=str(worker),
        node_bin='/node',drain_file=str(tmp_path/'DRAIN'))
    state['config_snapshot']=put(folder/'config.json',config)
    owners={}
    monkeypatch.setattr(g,'alive',lambda owner:bool(owner)and owner['pid']in owners)
    monkeypatch.setattr(seq,'alive',g.alive)
    monkeypatch.setattr(g,'process',lambda pid:owners.get(int(pid)))
    seq.prepare_source_sequence(folder,state,config,22)
    state.update(active_batch=22,owner={'pid':122,'start_ticks':'22','boot_id':'boot'})
    owners[122]={**state['owner'],'argv':[b'python',str(feeder).encode()]}

    def materialize(batch,configuration,workers=12):
        value=seq._json(configuration,config['source_sequence'])
        home=Path(config['root'])/'evidence-collections-v2'/f'batch-{batch:04d}';home.mkdir()
        (home/'source-cases.jsonl').write_bytes(Path(value['cases']).read_bytes())
        rows=[{'key':f'{batch}-{i}'}for i in range(work_items)]
        (home/'queue.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in rows))
        put(home/'teacher-source.json',{'batch':batch,'source_cases_sha256':value['cases_sha256'],
            'actual_task_runtime':config['task_runtime'],'concurrency':workers})
        events=[]
        for i,row in enumerate(rows[:work_items-unstarted]):
            events.append({'event':'start','key':row['key']})
            if i<work_items-unstarted-active_count:events.append({'event':'finish','key':row['key']})
        (home/'journal-worker-0.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in events))
        return home

    current=state['source_sequence']['pending']
    intent=seq._json(current['intent'],config['source_sequence'])
    materialize(22,intent['new_configuration']);seq.observe_source_sequence(folder,state,config)
    for batch in range(23,23+ahead):
        reservation=seq.prepare_prefetched_source(folder,state,config,batch)
        intent=seq._json(reservation['intent'],config['source_sequence'])
        materialize(batch,intent['new_configuration'],24)
        seq.observe_prefetched_source(folder,state,config)
        owner={'pid':100+batch,'start_ticks':str(batch),'boot_id':'boot'}
        owners[100+batch]={**owner,'argv':[b'python',str(feeder).encode()]}
        slot={'batch':batch,'owner':owner,'workers':24,'token':f'token-{batch}',
            'feeder':g.binding(feeder),'guardian_config':state['config_snapshot'],
            'source_configuration':intent['new_configuration'],'selection_intent':reservation['intent'],
            'task_runtime':config['task_runtime'],'collection_context':intent['prefetch']['resource_refill']['collection_context'],
            'phase_path':str(Path(config['root'])/'feeder-phases'/f'batch-{batch:04d}.json')}
        state['primary_prefetch']=slot
        if batch<22+ahead:seq.park_active_prefetch(folder,state,config,copy.deepcopy(slot))
    return folder,state,config,owners


def test_many_live_tails_refill_within_worker_envelope_and_preserve_cursors(tmp_path,monkeypatch):
    folder,state,config,owners=pool(tmp_path,monkeypatch,ahead=4)
    before=copy.deepcopy(state);observed=g.primary_capacity_observation(folder,state,config)
    assert [row['workers']for row in observed['slots']]==[12,24,24,24,24]
    decision=g.primary_refill_admission(observed,24,48)
    assert decision['potential_workers']==10  # Active AND unstarted keys of all five batches.
    spawned=[]
    monkeypatch.setattr(g,'_spawn_primary_prefetch',lambda *args:spawned.append(args[3]['batch']))
    g.maintain_primary_prefetch(folder,state,config)
    assert state['primary_prefetch']['batch']==27
    assert set(state['primary_active_ahead'])=={'23','24','25','26'}
    assert state['source_sequence']['pending']==before['source_sequence']['pending']
    assert state['source_sequence']['next_ordinal']==0
    assert not state['source_sequence'].get('closed')
    assert [slot['batch']for slot in g.primary_ahead_slots(state).values()]==[23,24,25,26,27]
    receipt=json.loads(Path(state['primary_prefetch']['capacity_admission']['path']).read_bytes())
    assert receipt['potential_workers']+receipt['new_workers']==34
    # Before the new owner seals its source, another poll cannot reserve anything else.
    g.maintain_primary_prefetch(folder,state,config)
    assert state['primary_prefetch']['batch']==27


def test_exact_nice_exec_transition_preserves_feeder_owner_custody(tmp_path,monkeypatch):
    folder,state,config,owners=pool(tmp_path,monkeypatch,ahead=0)
    owner=owners[state['owner']['pid']]
    owner['argv']=[b'nice',b'-n',b'10',b'python',b'-u',config['feeder'].encode()]
    observed=g.primary_capacity_observation(folder,state,config)
    assert observed['slots'][0]['owner']==state['owner']
    # A different wrapper cannot smuggle a script path beyond the old prefix.
    owner['argv'][0]=b'unrelated-wrapper'
    with pytest.raises(ValueError,match='exact feeder'):
        g.primary_capacity_observation(folder,state,config)


@pytest.mark.parametrize('change',['capacity','startup','dead','tamper','workers','partitions'])
def test_unknown_or_excess_work_never_grants_capacity(tmp_path,monkeypatch,change):
    folder,state,config,owners=pool(tmp_path,monkeypatch,ahead=2)
    if change=='dead':owners.pop(123)
    if change=='tamper':
        home=Path(config['root'])/'evidence-collections-v2/batch-0023'
        (home/'teacher-source.json').write_text('{}')
    if change=='workers':state['primary_active_ahead']['23']['workers']=12
    if change in ('tamper','workers'):
        with pytest.raises(ValueError):g.primary_capacity_observation(folder,state,config)
        return
    observed=g.primary_capacity_observation(folder,state,config)
    if change=='dead':assert observed is None;return
    if change=='capacity':assert g.primary_refill_admission(observed,24,29)is None;return
    if change=='startup':
        for slot in observed['slots']:
            frontier=slot['frontier'];frontier.update(started_keys=[],finished_keys=[],active_keys=[],
                unstarted_keys=frontier['remaining_keys'])
        assert g.primary_refill_admission(observed,24,48)is None;return
    observed['slots'][0]['frontier']['active_keys']=[]
    with pytest.raises(ValueError,match='partitions'):g.primary_refill_admission(observed,24,48)


def test_changed_journal_or_owner_invalidates_capacity_snapshot(tmp_path,monkeypatch):
    folder,state,config,owners=pool(tmp_path,monkeypatch)
    observed=g.primary_capacity_observation(folder,state,config)
    assert g._capacity_still_current(observed)
    owners[123]['argv'].append(b'changed')
    assert not g._capacity_still_current(observed)
    owners[123]['argv'].pop()
    journal=Path(observed['slots'][0]['frontier']['journals'][0]['path'])
    journal.write_bytes(journal.read_bytes()+b'\n')
    assert not g._capacity_still_current(observed)


def test_completed_middle_slot_keeps_later_live_slots_and_serial_adoption(tmp_path,monkeypatch):
    folder,state,config,owners=pool(tmp_path,monkeypatch,ahead=3)
    slot=state['primary_active_ahead']['24'];owners.pop(124)
    started=seq._json(state['source_sequence']['started']['24'],config['source_sequence'])
    slot['boundary']=put(folder/'boundaries/batch-0024.json',{'batch':24,'batch_complete':True,
        'bindings':{name:started['sealed'][name]for name in ('queue.jsonl','teacher-source.json')}})
    seq.archive_completed_prefetch(folder,state,config,copy.deepcopy(slot))
    assert set(state['primary_completed_ahead'])=={'24'}
    assert set(state['primary_active_ahead'])=={'23'}
    assert g.primary_ahead_slots(state)[25]==state['primary_prefetch']
    # Crash after publication reuses all original started references on reload.
    state=json.loads((folder/'state.json').read_text())
    assert [item['batch']for item in seq._prefetch_service_chain(folder,state,config)]==[22,23,24]
    primary=seq._json(state['source_sequence']['started']['22'],config['source_sequence'])
    state.update(owner=None,last_boundary=put(folder/'boundary22.json',{'batch':22,'batch_complete':True,
        'bindings':{name:primary['sealed'][name]for name in ('queue.jsonl','teacher-source.json')}}))
    seq.observe_source_sequence(folder,state,config);state['next_batch']=23
    monkeypatch.setattr(g,'_spawn_primary_prefetch',lambda *args:None)
    assert g.adopt_primary_prefetch(folder,state,config)
    assert state['active_batch']==23 and state['primary_prefetch']['batch']==25
    assert set(state['primary_completed_ahead'])=={'24'}
    assert state['source_sequence']['next_ordinal']==1


def test_primary_death_during_park_preserves_every_source_without_new_reservation(tmp_path,monkeypatch):
    folder,state,config,owners=pool(tmp_path,monkeypatch,ahead=2)
    original=seq.park_active_prefetch
    def park(*args):
        original(*args);owners.pop(122)
    monkeypatch.setattr(seq,'park_active_prefetch',park)
    monkeypatch.setattr(g,'_spawn_primary_prefetch',lambda *args:None)
    g.maintain_primary_prefetch(folder,state,config)
    assert not state.get('primary_prefetch')
    assert set(state['primary_active_ahead'])=={'23','24'}
    assert set(state['source_sequence']['started'])=={'22','23','24'}
    assert not state['source_sequence'].get('prefetch')
    # Even though the primary died, archive validation can recover the parked custody.
    assert [row['batch']for row in seq._prefetch_service_chain(folder,state,config,require_live=False)]==[22,23,24]
    with pytest.raises(ValueError,match='live predecessor'):seq.prepare_prefetched_source(folder,state,config,25)


def test_interleaved_inventory_shared_with_flat_publisher(monkeypatch):
    state={'active_batch':22,'primary_active_ahead':{'23':{'batch':23}},
        'primary_completed_ahead':{'24':{'batch':24,'boundary':{'sha256':'proof'}}},
        'primary_prefetch':{'batch':25}}
    expected={23:state['primary_active_ahead']['23'],24:state['primary_completed_ahead']['24'],25:state['primary_prefetch']}
    assert export._guardian_lookahead_slots(state)==expected
    path=Path(export.__file__);monkeypatch.syspath_prepend(str(path.parent))
    spec=importlib.util.spec_from_file_location('flat_teacher_export_test',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    try:
        spec.loader.exec_module(module)
        assert module._guardian_lookahead_slots(state)==expected
    finally:sys.modules.pop(spec.name,None)
    state['primary_active_ahead']['24']={'batch':24}
    with pytest.raises(ValueError):export._guardian_lookahead_slots(state)


@pytest.mark.parametrize('tamper',[None,'source','runtime','gap','envelope'])
def test_actual_publisher_preserves_live_tail_and_later_completed_publication(tamper):
    from types import SimpleNamespace
    from test_teacher_rolling_completed_slots import publisher_with_completed_and_later_slot
    publisher,_,remote,state,store=publisher_with_completed_and_later_slot()
    # Reverse the old completed/live arrangement: the earlier batch is still
    # productive, while a later batch has sealed and released its workers.
    earlier=state.pop('primary_completed_ahead')['23'];earlier.pop('boundary')
    state['primary_active_ahead']={'23':earlier}
    later=state.pop('primary_prefetch')
    later['boundary']=store('/work/control/boundaries/batch-0024.json',{'batch':24})
    state['primary_completed_ahead']={'24':later}
    state['source_sequence']['prefetch_history']['24']=state['source_sequence'].pop('prefetch')
    config=json.loads(remote[publisher['guardian_config']['path']]);config['primary_worker_capacity']=48
    current=store(publisher['guardian_config']['path'],config);publisher['guardian_config']=current;state['config']=current
    if tamper=='source':remote['/work/generation/evidence-collections-v2/batch-0023/source-cases.jsonl']=b'foreign'
    elif tamper=='runtime':earlier['task_runtime']='/wrong/runtime'
    elif tamper=='gap':state['primary_active_ahead']={}
    elif tamper=='envelope':
        config.pop('primary_worker_capacity');current=store(publisher['guardian_config']['path'],config)
        publisher['guardian_config']=current;state['config']=current
    store('/work/control/state.json',state)
    def execute(argv,**kwargs):
        if argv[0]=='cat':out='boot\n'
        elif 'glob'in argv[2]:out=json.dumps({'queues':['queue-worker-0.jsonl'],'journals':['journal-worker-0.jsonl']})
        elif argv[-1]=='24':out=json.dumps({'valid':False,'missing':True,'boot_id':'boot'})
        else:
            assert argv[-5:]==['43','888','boot',earlier['feeder']['path'],'23']
            out=json.dumps({'valid':True})
        return SimpleNamespace(result=lambda **_:SimpleNamespace(returncode=0,stdout=out,stderr=''))
    sandbox=SimpleNamespace(exec=execute)
    if tamper:
        with pytest.raises(ValueError):export.discover_guardian_batch(publisher,sandbox,remote.__getitem__,batch_slot=23)
        return
    live=export.discover_guardian_batch(publisher,sandbox,remote.__getitem__,batch_slot=23)
    assert live['batch']==23 and live['queue_path'].endswith('batch-0023/queue.jsonl')
    terminal=export.discover_guardian_batch(publisher,sandbox,remote.__getitem__,batch_slot=24)
    assert terminal['verified_owner_transition'] and terminal['closed_boundaries']==[later['boundary']]


def test_explicit_capacity_refills_before_legacy_low_water_mark(tmp_path,monkeypatch):
    folder,state,config,owners=pool(tmp_path,monkeypatch,ahead=0,work_items=16,active_count=8,unstarted=0)
    original=copy.deepcopy(state['source_sequence']['pending'])
    assert g.primary_tail_frontier(config['root'],22,6)is None
    observation=g.primary_capacity_observation(folder,state,config)
    decision=g.primary_refill_admission(observation,24,48)
    assert decision['potential_workers']==8 and decision['new_workers']==24
    assert g.primary_refill_admission(observation,24,31)is None
    spawned=[]
    monkeypatch.setattr(g,'_spawn_primary_prefetch',lambda *args:spawned.append(args[3]['batch']))
    g.maintain_primary_prefetch(folder,state,config)
    assert spawned==[23] and state['primary_prefetch']['batch']==23
    assert state['source_sequence']['pending']==original and not state['source_sequence'].get('closed')
    proof=json.loads(Path(state['primary_prefetch']['capacity_admission']['path']).read_bytes())
    assert proof['potential_workers']+proof['new_workers']==32<=proof['worker_capacity']==48


def test_absent_worker_envelope_preserves_legacy_low_water_mark(tmp_path,monkeypatch):
    folder,state,config,_=pool(tmp_path,monkeypatch,ahead=0,work_items=16,active_count=8,unstarted=0)
    config.pop('primary_worker_capacity')
    monkeypatch.setattr(g,'_spawn_primary_prefetch',lambda *args:pytest.fail('Legacy mode must retain its old low-water rule'))
    g.maintain_primary_prefetch(folder,state,config)
    assert not state.get('primary_prefetch') and not state['source_sequence'].get('prefetch')
