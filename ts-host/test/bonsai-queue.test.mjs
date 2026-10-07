import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

test('structured collection request budget remains an incomplete resource-limited attempt', () => {
  const output = execFileSync('python3', ['-c', `
import importlib.util,json,tempfile,pathlib
spec=importlib.util.spec_from_file_location('queue','../scripts/run_bonsai_queue.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
with tempfile.TemporaryDirectory() as d:
    p=pathlib.Path(d)
    (p/'000002.error.json').write_text(json.dumps(dict(index=2,program_id='test',code='NATLANG_MODEL_REQUEST_BUDGET',error='limit')))
    result=queue.output_accounting(dict(jobs=d,output=str(p/'out'),source=str(p/'source'),_resolved_jobs=[dict(index=2,key='000002-x',program_id='test',digest='digest')]))
    assert result['complete'] is False
    assert result['job_states'][0]['state']=='failed'
    assert result['job_states'][0]['resource_limit_reason']=='model_request_budget'
    print(json.dumps(result['resource_limit_reason']))
`], { cwd: new URL('..', import.meta.url), encoding: 'utf8' });
  assert.equal(JSON.parse(output), 'model_request_budget');
});

test('Bonsai activity counts live replies separately from replay and keeps timeout outcomes explicit', () => {
  const output = execFileSync('python3', ['-c', `
import importlib.util,json,tempfile,pathlib
spec=importlib.util.spec_from_file_location('queue','../scripts/run_bonsai_queue.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
with tempfile.TemporaryDirectory() as d:
    p=pathlib.Path(d)
    reply={'raw_response':{'id':'live'},'completion_tokens':10,'calls':[['eval',{'code':'1'}]]}
    (p/'000002-x.partial.json').write_text(json.dumps({'program_id':'test','provenance':{'program_ir_sha256':'digest'},'turns':[
        {'response':{'calls':[['eval',{'code':'replay'}]]}}, {'response':reply}, {'response':reply}]}))
    print(json.dumps(queue.partial_metrics({'jobs':d,'index':2,'_resolved_jobs':[{'key':'000002-x','program_id':'test','digest':'digest'}]})))
`], { cwd: new URL('..', import.meta.url), encoding: 'utf8' });
  assert.deepEqual(JSON.parse(output), { saved_turns: 3, fresh_model_replies: 2, completion_tokens: 20,
    repeated_action_sets: 1, repeated_request_hashes: 0, unique_action_sets: 1, unique_request_hashes: 0,
    repetition_scope: 'action shapes across all child calls; not a semantic stall detector' });
});

test('the supervisor distinguishes a hard time budget from missing reply activity', () => {
  const output = execFileSync('python3', ['-c', `
import importlib.util,json,tempfile,pathlib
from unittest.mock import patch,MagicMock
spec=importlib.util.spec_from_file_location('queue','../scripts/run_bonsai_queue.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
statuses=[]
for budget in [60,600]:
    with tempfile.TemporaryDirectory() as d:
        p=pathlib.Path(d);clock=[0.0];stopped=[False]
        child=MagicMock()
        def wait(timeout=None):
            if stopped[0]: return -15
            clock[0]+=timeout
            raise queue.subprocess.TimeoutExpired('test',timeout)
        def stop(): stopped[0]=True
        child.wait.side_effect=wait;child.terminate.side_effect=stop
        entry=dict(_resolved_jobs=[dict(key='test',program_id='test',digest='digest',index=0)],key='test',index=0,jobs=d,source='source',output='out',seed=1,log=str(p/'log'),case_seconds=budget)
        (p/'queue').write_text(json.dumps(entry)+'\\n')
        health=MagicMock();health.__enter__.return_value.status=200
        with patch.object(queue.urllib.request,'urlopen',return_value=health),patch.object(queue.subprocess,'Popen',return_value=child),patch.object(queue.time,'monotonic',side_effect=lambda:clock[0]):
            queue.run_queue(p/'queue',p/'journal',p,600,no_observation_seconds=300)
        events=[json.loads(line) for line in (p/'journal').read_text().splitlines()]
        statuses.append(events[-1]['status'])
        assert any(e['event']=='activity' for e in events)
print(json.dumps(statuses))
`], { cwd: new URL('..', import.meta.url), encoding: 'utf8' });
  assert.deepEqual(JSON.parse(output.trim().split('\n').at(-1)), ['timeout', 'no_observation_limit']);
});

test('the same bounded supervisor configures one Luna request including planning without the local server', () => {
  const output = execFileSync('python3', ['-c', `
import importlib.util,json,tempfile,pathlib
from unittest.mock import patch,MagicMock
spec=importlib.util.spec_from_file_location('queue','../scripts/run_bonsai_queue.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
with tempfile.TemporaryDirectory() as d:
    p=pathlib.Path(d);child=MagicMock();child.wait.return_value=0
    entry=dict(_resolved_jobs=[dict(key='luna',program_id='test',digest='digest',index=0)],key='luna',index=0,jobs=d,source='source',output='out',seed=1,log=str(p/'log'),max_model_requests=568,text_neuralese_emulation=True)
    (p/'queue').write_text(json.dumps(entry)+'\\n')
    with patch.object(queue.subprocess,'Popen',return_value=child) as start:
        queue.run_queue(p/'queue',p/'journal',p,1200,model_id='gpt-6-luna',provider='openai-codex',model_concurrency=1,execution_plans=True)
    print(json.dumps(next(call.args[0] for call in start.call_args_list if '--workers' in call.args[0])))
`], { cwd: new URL('..', import.meta.url), encoding: 'utf8' });
  const command = JSON.parse(output.trim().split('\n').at(-1));
  const value = flag => command[command.indexOf(flag) + 1];
  assert.equal(value('--workers'), '1');
  assert.equal(value('--model-concurrency'), '1');
  assert.equal(value('--model-id'), 'gpt-6-luna');
  assert.equal(value('--provider'), 'openai-codex');
  assert.equal(value('--max-model-requests'), '568');
  assert.ok(command.includes('--execution-plans'));
  assert.ok(command.includes('--text-neuralese-emulation'));
  assert.ok(!command.includes('--server'));
});
