"""Exact diagnostic plumbing; these fixtures are not live feedback measurements."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('native_marker_feedback',ROOT/'scripts/neuralese_data/qualify_native_marker_feedback.py')
feedback=importlib.util.module_from_spec(spec);spec.loader.exec_module(feedback)


def failed_response():
    failure={'schema':'natlang.native-marker-failure/1','code':'neuralese-native-state-close',
        'message':'neuralese-native-state-close: produce end without an open span',
        'trigger':'sample','sample_index':0,'fed':False,'own_input_count':1,
        'generated_input_count':0,'max_tokens':4,'stop_token_ids':[2]}
    identity={key:('lfm2' if key=='architecture' else '1'*64) for key in feedback.IDENTITY_KEYS}
    return {'choices':[{'finish_reason':'error','message':{'role':'assistant','content':'actual-draw'}}],
        'x_natlang_native_failure':failure,'x_natlang_rl_capture':{
            'schema':'natlang.native-rl-sampling/4','policy':identity,'finish_reason':'error',
            'protocol_failure':copy.deepcopy(failure),'prompt':[{'kind':'tokens','token_ids':[5]}],
            'samples':[{'index':0,'token_id':7,'terminal':False}],'generated_inputs':[]}}


def test_legacy_projection_requires_exact_real_capture_boundary():
    value=failed_response()
    assert feedback.project_response(value,'candidate')==(200,value)
    assert feedback.project_response(value,'baseline')==(400,{'error':{
        'code':value['x_natlang_native_failure']['code'],'message':value['x_natlang_native_failure']['message']}})
    tampered=copy.deepcopy(value);tampered['x_natlang_native_failure']['fed']=True
    with pytest.raises(ValueError,match='actual failed capture'):feedback.project_response(tampered,'baseline')
    tampered=copy.deepcopy(value);tampered['x_natlang_rl_capture']['generated_inputs']=[{'token_id':7,'sample_index':0}]
    with pytest.raises(ValueError,match='fed-input'):feedback.project_response(tampered,'candidate')
    tampered=copy.deepcopy(value);tampered.pop('x_natlang_native_failure')
    with pytest.raises(ValueError,match='lacks its typed'):feedback.project_response(tampered,'baseline')
    tampered=copy.deepcopy(value);tampered['choices'][0]['finish_reason']='stop'
    with pytest.raises(ValueError,match='executable'):feedback.project_response(tampered,'candidate')
    ordinary={'choices':[{'message':{'content':'literal neuralese-native-state-close'}}]}
    assert feedback.project_response(ordinary,'baseline')==(200,ordinary)


def test_recursive_view_projection_preserves_the_actual_failed_source_turn():
    value=failed_response();view={'source_turns':[{'response':value}],
        'x_natlang_native_failure':copy.deepcopy(value['x_natlang_native_failure'])}
    assert feedback.project_response(view,'baseline')[0]==400
    view['x_natlang_native_failure']['sample_index']=1
    with pytest.raises(ValueError,match='actual failed capture'):feedback.project_response(view,'baseline')


def test_durable_observer_rejects_actor_swap_and_keeps_exact_failed_wire(tmp_path):
    value=failed_response();identity=value['x_natlang_rl_capture']['policy']
    proxy=feedback.RecordingProxy('http://unused','baseline',tmp_path,identity)
    request={'messages':[{'role':'user','content':'ordinary input'}]}
    assert proxy.observe('/v1/chat/completions',request,value,1)[0]==400
    record=json.loads((tmp_path/'000000.json').read_text())
    assert record['response']==value and record['request']==request
    assert feedback.bound(record['bundle']).is_file()
    restored=feedback.RecordingProxy('http://unused','candidate',tmp_path,identity)
    assert restored.next==1
    swapped=copy.deepcopy(value);swapped['x_natlang_rl_capture']['policy']['checkpoint_sha256']='2'*64
    with pytest.raises(ValueError,match='actor changed'):restored.observe('/v1/chat/completions',request,swapped,2)
    assert not (tmp_path/'000001.json').exists()
    with pytest.raises(ValueError,match='omitted'):restored.observe('/v1/chat/completions',request,{'choices':[]},2)


def test_feedback_counts_require_exact_same_call_context_and_runtime_text():
    prompt=[{'role':'user','content':'original'}]
    previous={'variant':'candidate','request':{'messages':prompt},'expected_runtime_feedback':'actual runtime error',
        'started_at':1,'finished_at':2,'protocol_failures':1}
    reply={'variant':'candidate','request':{'messages':prompt+[
        {'role':'assistant','content':'actual-draw'},{'role':'user','content':'actual runtime error'}]},
        'started_at':3,'finished_at':5,'protocol_failures':0}
    metrics=feedback.feedback_metrics([previous,reply])
    assert metrics['actual_feedback_exposures']==1 and metrics['protocol_accepted_repairs']==1
    assert metrics['feedback_replies'][0]['repair_elapsed_seconds']==2
    wrong=copy.deepcopy(reply);wrong['request']['messages'][0]['content']='another call'
    assert feedback.feedback_metrics([previous,wrong])['actual_feedback_exposures']==0
    repeated=copy.deepcopy(reply);repeated['protocol_failures']=1
    assert feedback.feedback_metrics([previous,repeated])['repeated_protocol_failures_after_feedback']==1
    legacy=copy.deepcopy(previous);legacy['variant']='baseline'
    with pytest.raises(ValueError,match='unexpectedly'):feedback.feedback_metrics([legacy,reply])


def test_parent_receipt_and_immutable_inputs_fail_closed(tmp_path):
    path=tmp_path/'parent.json';sha='a'*64
    receipt={'name':f'lfm/{sha}/lfm-native-checkpoint-step-8451.pt','private':True,
        'identity':{'sha256':sha,'bytes':123},'independent_verification':{'sha256':sha,'bytes':123}}
    feedback.immutable(path,receipt);reference=feedback.binding(path)
    assert feedback.trained_parent(reference)==sha
    with pytest.raises(ValueError,match='immutable'):feedback.immutable(path,{**receipt,'private':False})
    path.write_text(json.dumps({**receipt,'private':False}))
    with pytest.raises(ValueError,match='input changed'):feedback.trained_parent(reference)
    with pytest.raises(ValueError,match='independently verified'):feedback.trained_parent(feedback.binding(path))


def test_runner_restart_reuses_sealed_proxy_and_executor_without_claiming_live_samples(tmp_path,monkeypatch):
    import argparse
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    import subprocess
    import threading
    source=tmp_path/'source.json';source.write_text('{}')
    selection=tmp_path/'selection.json';selection.write_text('[]')
    parent=tmp_path/'parent.json';sha='a'*64
    feedback.immutable(parent,{'name':f'lfm/{sha}/lfm-native-checkpoint-step-8451.pt','private':True,
        'identity':{'sha256':sha,'bytes':123},'independent_verification':{'sha256':sha,'bytes':123}})
    runtime=tmp_path/'runtime';runtime.mkdir()
    variant=tmp_path/'candidate';variant.mkdir()
    packet=variant/'packet.jsonl';packet.write_text('{}\n')
    queue=variant/'queue.jsonl';queue.write_text('')
    node=tmp_path/'node';node.write_text('pinned executor bytes')
    row={'id':'fixture-only','split':'train'}
    plan={'schema':'natlang.native-marker-feedback-window/1','parent_receipt':feedback.binding(parent),
        'checkpoint_sha256':sha,'source':feedback.binding(source),'selection':feedback.binding(selection),
        'cases':[{'id':row['id'],'source_index':0,'record_sha256':feedback.digest(row)}],
        'packets':{'candidate':feedback.binding(packet)},'queues':{'candidate':feedback.binding(queue)},
        'runtimes':{'candidate':{'path':str(runtime),'inventory':{'test-only':'fixture'}}}}
    feedback.immutable(tmp_path/'plan.json',plan)
    monkeypatch.setattr(feedback,'select_rows',lambda *args:({},[(0,row)]))
    monkeypatch.setattr(feedback,'task_runtime_inventory',lambda path:{'test-only':'fixture'})
    monkeypatch.setattr(feedback.subprocess,'check_output',lambda *args,**kw:'v24.0.1\n')
    calls=[]
    def collect(command,**kwargs):
        calls.append(command)
        assert kwargs['env']['PATH'].split(':')[0]==str(node.parent)
        (variant/'journal.jsonl').write_text(''.join(json.dumps({'event':'finish','key':'same-window'})+'\n' for _ in range(2)))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(feedback.subprocess,'run',collect)
    identity={key:('lfm2' if key=='architecture' else sha) for key in feedback.IDENTITY_KEYS}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            raw=json.dumps({'native_state':identity,'capabilities':['native.protocol-recovery']}).encode()
            self.send_response(200);self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    args=argparse.Namespace(plan=tmp_path/'plan.json',variant='candidate',node=node,
        endpoint=f'http://127.0.0.1:{server.server_address[1]}',model_id='trained-student')
    try:
        first=feedback.run(args);second=feedback.run(args)
        assert calls[0]==calls[1]
        assert first==second and second['actual_sampled_turns']==0
        assert second['feedback_quality_passed'] is None and second['finished_windows']==1
        assert second['finish_events']==2
        packet.write_text('changed packet\n')
        with pytest.raises(ValueError,match='input changed'):feedback.run(args)
        assert len(calls)==2
    finally:server.shutdown();server.server_close()


def test_preparation_seals_twelve_train_shapes_and_exact_four_seed_queues(tmp_path,monkeypatch):
    import argparse
    cases=tmp_path/'cases.jsonl'
    rows=[{'id':f'train-{i}','split':'train','curriculum':{'reference':{'root':[['eval',{'code':'return 7;'}]]}}} for i in range(12)]
    cases.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    manifest=tmp_path/'source.json';manifest.write_text('{}')
    selection=tmp_path/'selection.json';selection.write_text(json.dumps([row['id'] for row in rows]))
    parent=tmp_path/'parent.json';sha='a'*64
    feedback.immutable(parent,{'name':f'lfm/{sha}/lfm-native-checkpoint-step-8451.pt','private':True,
        'identity':{'sha256':sha,'bytes':123},'independent_verification':{'sha256':sha,'bytes':123}})
    monkeypatch.setattr(feedback,'resolve_task_source',lambda path:{'cases':cases,'manifest':manifest})
    monkeypatch.setattr(feedback,'snapshot_task_runtime',lambda runtime,directory:Path(runtime))
    monkeypatch.setattr(feedback,'task_runtime_inventory',lambda runtime:{'test-only':'fixture'})
    args=argparse.Namespace(out=tmp_path/'out',task_source=manifest,selection=selection,actor_receipt=parent,
        baseline_runtime=tmp_path/'baseline',candidate_runtime=tmp_path/'candidate',root_mode='authored',
        temperature=.6,seed_base=17000,max_model_requests=3,action_max_tokens=512,case_seconds=120)
    plan=feedback.prepare(args)
    assert feedback.prepare(args)==plan
    value=json.loads(plan.read_text());queues=[]
    for variant in ('baseline','candidate'):
        queue=[json.loads(line) for line in feedback.bound(value['queues'][variant]).read_text().splitlines()]
        assert len(queue)==48 and len({entry['seed'] for entry in queue})==48
        assert all(entry['action_max_tokens']==512 and not entry['text_neuralese_emulation'] for entry in queue)
        assert feedback.bound(value['packets'][variant]).is_file()
        queues.append([entry['seed'] for entry in queue])
    assert queues[0]==queues[1]
    rows[-1]['split']='test';cases.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError,match='protected/nontrain'):feedback.prepare(args)


@pytest.mark.parametrize('sampled',[False,True])
def test_first_capture_stops_host_only_windows_and_preserves_sampled_first_case(tmp_path,sampled):
    import os
    import sys
    runner=tmp_path/'collector.py'
    runner.write_text('''import sys,json,pathlib
queue,journal,observations=map(pathlib.Path,sys.argv[1:])
observations.mkdir(exist_ok=True)
finished={json.loads(line)['key'] for line in journal.read_text().splitlines()} if journal.exists() else set()
for line in queue.read_text().splitlines():
 row=json.loads(line)
 if row['key'] in finished:continue
 with journal.open('a') as out:out.write(json.dumps({'key':row['key'],'event':'finish'})+'\\n')
 if row['sampled']:(observations/(row['key']+'.json')).write_text(json.dumps({'sampled_turns':1}))
''')
    rows=[{'key':f'{i:06d}','sampled':sampled} for i in range(48)]
    queue=tmp_path/'queue.jsonl';queue.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    journal=tmp_path/'journal.jsonl';observations=tmp_path/'observations'
    command=[sys.executable,str(runner),str(queue),str(journal),str(observations)]
    if not sampled:
        with pytest.raises(RuntimeError,match='remaining queue was not run'):
            feedback.run_queue(command,directory=tmp_path,env=os.environ,require_first_capture=True)
        assert len(journal.read_text().splitlines())==1
    else:
        assert feedback.run_queue(command,directory=tmp_path,env=os.environ,require_first_capture=True).returncode==0
        assert len(journal.read_text().splitlines())==48
        assert len({json.loads(line)['key'] for line in journal.read_text().splitlines()})==48
        # Restart uses the same completed key and journal; no first draw repeated.
        assert feedback.run_queue(command,directory=tmp_path,env=os.environ,require_first_capture=True).returncode==0
        assert len(journal.read_text().splitlines())==48
    proof=json.loads((tmp_path/'first-capture-check.json').read_text())
    assert proof['passed'] is sampled and proof['training_admitted'] is False
