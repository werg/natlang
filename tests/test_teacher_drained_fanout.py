"""Exact ownership and key partition at a cooperative case boundary."""
import copy,json
from pathlib import Path
import pytest
from scripts.coreweave import teacher_generation_guardian as g


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'generation';folder=root/'control';folder.mkdir(parents=True)
    out=root/'evidence-collections-v2/batch-0015';out.mkdir(parents=True)
    queue=out/'queue.jsonl';entries=[{'key':str(i),'index':i,'seed':90+i,'source':'original','jobs':f'jobs{i}'} for i in range(6)]
    queue.write_text(''.join(json.dumps(e)+'\n' for e in entries))
    journal=out/'journal-worker-0.jsonl';journal.write_text(json.dumps({'event':'start','key':'0'})+'\n'+json.dumps({'event':'finish','key':'0','status':'complete'})+'\n')
    old={'pid':10,'boot_id':'boot','start_ticks':'100'};new={'pid':20,'boot_id':'boot','start_ticks':'200'}
    feeder=folder/'feed-evidence-resume.py';feeder.write_text('exact awaiting feeder')
    config={'root':str(root),'control':str(folder),'drain_file':str(folder/'DRAIN'),'feeder':'old-feeder','teacher_identity':{'path':'actor'},'server':'server','model_id':'model'}
    g.write(folder/'old-config.json',config)
    state={'active_batch':15,'next_batch':16,'owner':old,'repair_debt':['keep'], 'config':g.binding(folder/'old-config.json'),'config_value':config}
    g.write(folder/'state.json',state);g.write(folder/'original-state.json',state)
    (folder/'DRAIN').touch();g.write(folder/'drain.json',{'owner':old,'children':[{'pid':11,'boot_id':'boot','start_ticks':'110'}],'drain_file':str(folder/'DRAIN')})
    g.write(out/'teacher-source.json',{'original_actor':'unchanged'})
    g.write(folder/'frontier.json',g.fanout_frontier(queue,[journal],3))
    g.write(folder/'new-config.json',{**config,'feeder':str(feeder)})
    transition={'schema':'natlang.teacher-drained-fanout-transition/1','id':'fanout-24','batch':15,
        'original_state':g.binding(folder/'original-state.json'),'drain':g.binding(folder/'drain.json'),
        'config':g.binding(folder/'new-config.json'),'feeder':g.binding(feeder),
        'frontier':g.binding(folder/'frontier.json'),'source':g.binding(out/'teacher-source.json'),
        'process_token':'exact-token','owner':new}
    path=folder/'transition.json';g.write(path,transition)
    monkeypatch.setattr(g,'alive',lambda identity:identity==new)
    monkeypatch.setattr(g,'recover_token',lambda token:{**new,'argv':[b'python',str(feeder).encode()]})
    monkeypatch.setattr(g,'live_batch_workers',lambda source:[])
    monkeypatch.setattr(g,'verify_teacher_identity',lambda *a,**k:{'passed':True})
    return folder,path,transition,out,entries,new


def test_drained_fanout_preserves_exact_keys_seeds_debt_and_same_batch(tmp_path,monkeypatch):
    folder,path,t,out,entries,new=fixture(tmp_path,monkeypatch)
    before=(out/'queue.jsonl').read_bytes();source=(out/'teacher-source.json').read_bytes()
    actual=g.adopt_drained_fanout(folder,path)
    assert actual['active_batch']==15 and actual['next_batch']==16 and actual['repair_debt']==['keep']
    assert actual['owner']==new and (folder/'fanout/fanout-24/ARMED.json').exists()
    assert (out/'queue.jsonl').read_bytes()==before and (out/'teacher-source.json').read_bytes()==source
    plan=json.loads(Path(t['frontier']['path']).read_text())
    flattened=[e for shard in plan['shards'] for e in shard]
    assert sorted(flattened,key=lambda e:e['index'])==entries[1:]
    assert g.adopt_drained_fanout(folder,path)==actual


@pytest.mark.parametrize('change',['owner_alive','child_alive','actual_worker_alive','source_changed','queue_changed','journal_changed','omitted_journal','actor_changed','reused_pid','state_changed','escaped_id'])
def test_bad_fanout_never_publishes_successor(tmp_path,monkeypatch,change):
    folder,path,t,out,entries,new=fixture(tmp_path,monkeypatch)
    if change=='owner_alive':monkeypatch.setattr(g,'alive',lambda identity:True)
    elif change=='child_alive':monkeypatch.setattr(g,'alive',lambda identity:identity and identity['pid']==11)
    elif change=='actual_worker_alive':monkeypatch.setattr(g,'live_batch_workers',lambda source:[{'pid':11}])
    elif change=='source_changed':(out/'teacher-source.json').write_text('{}')
    elif change=='queue_changed':(out/'queue.jsonl').write_text('{}\n')
    elif change=='journal_changed':(out/'journal-worker-0.jsonl').write_text('{}\n')
    elif change=='omitted_journal':(out/'journal-worker-1.jsonl').write_text('')
    elif change=='actor_changed':
        config=json.loads(Path(t['config']['path']).read_text());config['server']='another';g.write(t['config']['path'],config);t['config']=g.binding(t['config']['path']);g.write(path,t)
    elif change=='reused_pid':monkeypatch.setattr(g,'recover_token',lambda token:None)
    elif change=='state_changed':g.write(folder/'state.json',{'owner':'newer'})
    elif change=='escaped_id':t['id']='../escape';g.write(path,t)
    before=(folder/'state.json').read_bytes()
    with pytest.raises((ValueError,KeyError)):g.adopt_drained_fanout(folder,path)
    assert (folder/'state.json').read_bytes()==before
    assert not(folder/'fanout/fanout-24/ARMED.json').exists()


def test_frontier_rejects_inflight_case_and_conflicting_outcomes(tmp_path):
    queue=tmp_path/'queue';queue.write_text('{"key":"a","seed":91}\n');journal=tmp_path/'journal'
    journal.write_text('{"event":"start","key":"a"}\n')
    with pytest.raises(ValueError,match='unfinished'):g.fanout_frontier(queue,[journal],24)
    journal.write_text('{"event":"finish","key":"a","status":"complete"}\n{"event":"finish","key":"a","status":"failed"}\n')
    with pytest.raises(ValueError,match='conflicting'):g.fanout_frontier(queue,[journal],24)
