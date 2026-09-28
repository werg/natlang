import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

test('Bonsai activity counts live replies separately from replay and keeps timeout outcomes explicit', () => {
  const output = execFileSync('python3', ['-c', `
import importlib.util,json,tempfile,pathlib
spec=importlib.util.spec_from_file_location('queue','../scripts/run_bonsai_queue.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
with tempfile.TemporaryDirectory() as d:
    p=pathlib.Path(d)
    reply={'raw_response':{'id':'live'},'completion_tokens':10,'calls':[['eval',{'code':'1'}]]}
    (p/'000002-x.partial.json').write_text(json.dumps({'turns':[
        {'response':{'calls':[['eval',{'code':'replay'}]]}}, {'response':reply}, {'response':reply}]}))
    print(json.dumps(queue.partial_metrics({'jobs':d,'index':2})))
`], { cwd: new URL('..', import.meta.url), encoding: 'utf8' });
  assert.deepEqual(JSON.parse(output), { saved_turns: 3, fresh_model_replies: 2, completion_tokens: 20,
    repeated_action_sets: 1, unique_action_sets: 1 });
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
        entry=dict(key='test',index=0,jobs=d,source='source',output='out',seed=1,log=str(p/'log'),case_seconds=budget)
        (p/'queue').write_text(json.dumps(entry)+'\\n')
        with patch.object(queue.subprocess,'Popen',return_value=child),patch.object(queue.time,'monotonic',side_effect=lambda:clock[0]):
            queue.run_queue(p/'queue',p/'journal',p,600)
        events=[json.loads(line) for line in (p/'journal').read_text().splitlines()]
        statuses.append(events[-1]['status'])
        assert any(e['event']=='activity' for e in events)
print(json.dumps(statuses))
`], { cwd: new URL('..', import.meta.url), encoding: 'utf8' });
  assert.deepEqual(JSON.parse(output.trim().split('\n').at(-1)), ['timeout', 'inactivity_timeout']);
});
