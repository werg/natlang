import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { NativeRuntime } from '../dist/native/runtime.js';
import { TypeEnv } from '../dist/index.js';
import { buildPending } from '../dist/native/values.js';
import { NativeSession } from '../dist/native/runtime.js';
import { dump } from '../dist/native/values.js';
import { NativeToolAgent } from '../dist/native/agent.js';
import { checkedDefinitions } from '../dist/native/codebase.js';
import { loadFunctionFile } from '../dist/native/source.js';
import { dumpState } from '../dist/native/values.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';

const root = fileURLToPath(new URL('../..', import.meta.url));
const python = process.env.NATLANG_PYTHON;
const SCRIPT = `import json,sys\nfrom natlang.runtime import Runtime\nfrom natlang.values import load_program,dump\nroot=load_program(json.load(sys.stdin))\nout,value=Runtime(None).run_root(root)\nprint(json.dumps({'kind':out.kind,'value':dump(value)},sort_keys=True))`;

function assertTraceParity(actual, expected) {
  const stable = events => events.map(event => {
    assert.ok(Number.isFinite(Date.parse(event.observed_at)));
    assert.ok(Number.isInteger(event.elapsed_ms) && event.elapsed_ms >= 0);
    const { observed_at, elapsed_ms, duration_ms, tool_schema_bytes, ...rest } = event;
    if (duration_ms !== undefined) assert.ok(Number.isInteger(duration_ms) && duration_ms >= 0);
    if (tool_schema_bytes !== undefined) assert.ok(Number.isInteger(tool_schema_bytes) && tool_schema_bytes > 0);
    return rest;
  });
  assert.deepEqual(stable(actual), stable(expected));
}

test('native default system prompt stays aligned with Python tool agent', { skip: !python }, async () => {
  const reference = readFileSync(new URL('../../natlang/prompts/tools_small.md', import.meta.url), 'utf8');
  assert.equal(TOOLS_PROMPT, reference);
  let seen;
  const agent = new NativeToolAgent(request => { seen = request.messages[0].content;
    return { calls: [], text: '', completion_tokens: 1 }; });
  await new NativeRuntime({ agent: session => agent.run(session) }).runRoot({ $lambda: {
    type: 'Lambda<{}, Num>', instructions: 'Write one.' } });
  assert.equal(seen, reference + '\nFor run_code, always name an engine offered in its current tool schema.');
});

test('model tool-call IDs survive feedback history in both runtimes', { skip: !python }, async () => {
  const script = `import json
from natlang.decoder import ChatTurn
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.values import load_program
class Driver:
 def __init__(self): self.history=None; self.turns=0
 def chat(self,messages,tools,*,temperature,seed,max_tokens):
  self.turns+=1
  if self.turns==1:
   return ChatTurn(calls=[('read',{'path':'instructions'})],raw_calls=[{'id':'browser_42','type':'function','function':{'name':'read','arguments':'{"path":"instructions"}'}}],completion_tokens=1)
  self.history=messages
  return ChatTurn(calls=[],completion_tokens=1)
driver=Driver()
Runtime(lambda _:ToolAgent(driver)).run_root(load_program({'$lambda':{'type':'Lambda<{}, Num>','instructions':'Read instructions.'}}))
print(json.dumps([m.get('tool_call_id') for m in driver.history if m['role']=='tool']))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  let history, turns = 0;
  const agent = new NativeToolAgent(request => {
    if (++turns === 1) return { calls: [['read', { path: 'instructions' }]],
      raw_calls: [{ id: 'browser_42', type: 'function',
        function: { name: 'read', arguments: '{"path":"instructions"}' } }], completion_tokens: 1 };
    history = request.messages;
    return { calls: [], completion_tokens: 1 };
  });
  await new NativeRuntime({ agent: session => agent.run(session) }).runRoot({ $lambda: {
    type: 'Lambda<{}, Num>', instructions: 'Read instructions.' } });
  assert.deepEqual(history.filter(message => message.role === 'tool').map(message => message.tool_call_id),
    JSON.parse(py.stdout));
});

test('long conversations checkpoint from durable lambda state in both runtimes', { skip: !python }, async () => {
  const program = { $lambda: { type: 'Lambda<{}, { a: Num, b: Num }>',
    instructions: 'Write a as 1 and b as 2.' } };
  const script = `import json,sys
from natlang.decoder import ChatTurn
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.values import load_program,dump
program=json.load(sys.stdin)
class Driver:
 def __init__(self): self.requests=[]
 def chat(self,messages,tools,*,temperature,seed,max_tokens):
  self.requests.append((list(messages),tools))
  n=len(self.requests)
  if n==1: return ChatTurn(calls=[('write',{'path':'return/a','type':'Num','value':1})],completion_tokens=1)
  if n==2: return ChatTurn(text='Need to fill b.',completion_tokens=1)
  if n==3: return ChatTurn(calls=[('write',{'path':'return/b','type':'Num','value':2})],completion_tokens=1)
  return ChatTurn(completion_tokens=1)
driver=Driver()
out,value=Runtime(lambda _:ToolAgent(driver,segment_turns=1)).run_root(load_program(program))
fresh=driver.requests[2][0]
print(json.dumps({'kind':out.kind,'value':dump(value),'checkpoint_tools':len(driver.requests[1][1]),
 'fresh_messages':len(fresh),'has_note':any('Earlier working note' in str(m.get('content')) for m in fresh)}))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(program), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  let turns = 0, checkpointTools = -1, freshMessages = -1, hasNote = false;
  const agent = new NativeToolAgent(request => {
    turns++;
    if (turns === 1) return { calls: [['write', { path: 'return/a', type: 'Num', value: 1 }]],
      completion_tokens: 1 };
    if (turns === 2) { checkpointTools = request.tools.length; return { text: 'Need to fill b.',
      completion_tokens: 1 }; }
    if (turns === 3) {
      freshMessages = request.messages.length;
      hasNote = request.messages.some(message => String(message.content).includes('Earlier working note'));
      return { calls: [['write', { path: 'return/b', type: 'Num', value: 2 }]],
        completion_tokens: 1 };
    }
    return { calls: [], completion_tokens: 1 };
  }, { segmentTurns: 1 });
  const runtime = new NativeRuntime({ agent: session => agent.run(session) });
  const result = await runtime.runRoot(program);
  assert.deepEqual({ kind: result.outcome.kind, value: result.value, checkpoint_tools: checkpointTools,
    fresh_messages: freshMessages, has_note: hasNote }, JSON.parse(py.stdout));
  assert.ok(runtime.trace.events.some(event => event.kind === 'checkpoint'));
});

test('continuation notes and effect journals round-trip in both runtimes', { skip: !python }, () => {
  const program = { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Continue.',
    continuation_note: 'Need the final count.', effects_journal: [
      { seq: 1, capability: 'send', ok: true }] } };
  const script = `import json,sys
from natlang.values import load_program,dump
print(json.dumps(dump(load_program(json.load(sys.stdin)))))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(program), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  assert.deepEqual(dump(buildPending(program)), JSON.parse(py.stdout));
});

test('continuation state is visible through the same tools and workspace text', { skip: !python }, () => {
  const program = { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Continue.',
    continuation_note: 'Need the final count.', effects_journal: [
      { seq: 1, capability: 'send', ok: true }] } };
  const script = `import json,sys
from natlang.runtime import Runtime,Session
from natlang.types import TypeEnv
from natlang.values import load_program
from natlang.surface import ToolSurface
session=Session(Runtime(None,executors={'typescript-host':object()},engine_selection=True),
                load_program(json.load(sys.stdin)),TypeEnv())
surface=ToolSurface()
print(json.dumps({'state':surface.render_state(session),'tools':surface.tools(session)}))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(program), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const session = new NativeSession(new NativeRuntime(), buildPending(program), new TypeEnv());
  const agent = new NativeToolAgent(() => ({ calls: [] }));
  const expected = JSON.parse(py.stdout);
  assert.equal(agent.opening(session), expected.state);
  assert.deepEqual(agent.tools(session), expected.tools);
});

test('default model turn has no implicit token limit in either runtime', { skip: !python }, async () => {
  const script = `import json\nfrom natlang.decoder import ChatTurn\nfrom natlang.runtime import Runtime\nfrom natlang.tool_agent import ToolAgent\nfrom natlang.values import load_program\nclass Driver:\n def __init__(self): self.limits=[]\n def chat(self, messages, tools, *, temperature, seed, max_tokens):\n  self.limits.append(max_tokens)\n  return ChatTurn(calls=[], text='done', completion_tokens=1)\ndriver=Driver()\nagent=ToolAgent(driver)\nout,_=Runtime(lambda _: agent).run_root(load_program({'$lambda': {'type':'Lambda<{}, Num>', 'instructions':'Return one.'}}))\nprint(json.dumps({'limits':driver.limits,'kind':out.kind}))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const limits = [];
  const agent = new NativeToolAgent(request => { limits.push(request.max_tokens);
    return { calls: [], text: 'done', completion_tokens: 1 }; });
  const outcome = await new NativeRuntime({ agent: session => agent.run(session) }).runRoot({ $lambda: {
    type: 'Lambda<{}, Num>', instructions: 'Return one.' } });
  const expected = JSON.parse(py.stdout);
  assert.deepEqual(limits, expected.limits);
  assert.equal(outcome.outcome.kind, expected.kind);
});

test('long structured episodes cross former turn and action limits in both runtimes', { skip: !python }, async () => {
  const script = `import json\nfrom natlang.decoder import ChatTurn\nfrom natlang.runtime import Runtime\nfrom natlang.tool_agent import ToolAgent\nfrom natlang.values import load_program,dump\nclass Driver:\n def __init__(self): self.turns=0; self.limits=[]\n def chat(self, messages, tools, *, temperature, seed, max_tokens):\n  self.turns+=1; self.limits.append(max_tokens)\n  if self.turns<=130: return ChatTurn(calls=[('read',{'path':'instructions'})],completion_tokens=1)\n  if self.turns==131: return ChatTurn(calls=[('write',{'path':'return','type':'Num','value':7})],completion_tokens=1)\n  return ChatTurn(calls=[],completion_tokens=1)\ndriver=Driver(); agent=ToolAgent(driver)\nout,value=Runtime(lambda _:agent).run_root(load_program({'$lambda':{'type':'Lambda<{}, Num>','instructions':'Return seven.'}}))\nprint(json.dumps({'kind':out.kind,'value':dump(value),'turns':driver.turns,'unbounded':all(x is None for x in driver.limits)}))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  let turns = 0;
  const limits = [];
  const agent = new NativeToolAgent(request => {
    turns++; limits.push(request.max_tokens);
    if (turns <= 130) return { calls: [['read', { path: 'instructions' }]], completion_tokens: 1 };
    if (turns === 131) return { calls: [['write', { path: 'return', type: 'Num', value: 7 }]],
      completion_tokens: 1 };
    return { calls: [], completion_tokens: 1 };
  });
  const result = await new NativeRuntime({ agent: session => agent.run(session) }).runRoot({ $lambda: {
    type: 'Lambda<{}, Num>', instructions: 'Return seven.' } });
  assert.deepEqual({ kind: result.outcome.kind, value: result.value, turns,
    unbounded: limits.every(value => value === null) }, JSON.parse(py.stdout));
});

test('native reducer matches Python outcomes for finite crisp programs', { skip: !python }, async () => {
  const leaf = (type, code) => ({ $lambda: { type, code } });
  const fixtures = [
    leaf('Lambda<{}, Num>', 'return 4 + 3;'),
    { $map: { type: 'Map<Num, Num>', over: [1, 2, 3],
      fn: leaf('Lambda<{ item: Num }, Num>', 'return args.item * 2;') } },
    { $fold: { type: 'Fold<Num, Num>', over: [2, 3], init: 1,
      step: leaf('Lambda<{ acc: Num, item: Num }, Num>', 'return args.acc + args.item;') } },
    { $iterate: { type: 'Iterate<Num>', init: 0, max: 5, state_name: 'value', check_name: 'value',
      step: leaf('Lambda<{ value: Num }, Num>', 'return args.value + 1;'),
      check: leaf('Lambda<{ value: Num }, Bool>', 'return args.value >= 3;') } },
    leaf('Lambda<{}, Num>', 'return "bad";'),
  ];
  for (const fixture of fixtures) {
    const py = spawnSync(python, ['-c', SCRIPT], { cwd: root, input: JSON.stringify(fixture), encoding: 'utf8' });
    assert.equal(py.status, 0, py.stderr);
    const expected = JSON.parse(py.stdout);
    const actual = await new NativeRuntime().runRoot(fixture);
    assert.equal(actual.outcome.kind, expected.kind);
    if (actual.outcome.kind === 'done') assert.deepEqual(dump(actual.value), expected.value);
  }
});

test('native write actions match Python typed outcomes', { skip: !python }, () => {
  const doc = { $lambda: { type: 'Lambda<{}, { name: Text, amount: Num }>', instructions: 'Write the record.' } };
  const calls = [['write', { path: 'return/amount', type: 'Num', value: 'lots' }],
    ['write', { path: 'return', type: '{ name: Text, amount: Num }', value: { name: 'Ada', amount: 4 } }]];
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program,dump\ndoc,calls=json.load(sys.stdin)\nroot=load_program(doc)\ns=Session(Runtime(None),root,TypeEnv())\nprint(json.dumps({'kinds':[s.apply(n,a).kind for n,a in calls], 'value':dump(root.ret)}))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify([doc, calls]), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const lam = buildPending(doc);
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const actual = calls.map(([name, args]) => session.apply(name, args).kind);
  assert.deepEqual(actual, expected.kinds);
  assert.deepEqual(dump(lam.return), expected.value);
});

test('native action diagnostics match Python for rejected copy, edit, and reduce', { skip: !python }, async () => {
  const doc = { $lambda: { type: 'Lambda<{ item: Text }, { answer: Text, count: Num }>',
    instructions: 'Complete the record.', args: { item: 'abc' },
    return: { answer: { $lambda: { type: 'Lambda<{ missing: Text }, Text>', instructions: 'Use missing.' } } } } };
  const calls = [['copy', { from: 'args/item', to: 'return/count' }],
    ['edit', { path: 'args/item', old: 'a', new: 'z' }],
    ['run', { paths: 'return/answer' }],
    ['write', { path: 'return/answer/args/missing', type: 'Text', value: 'ready' }]];
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program\ndoc,calls=json.load(sys.stdin)\nroot=load_program(doc)\ns=Session(Runtime(None),root,TypeEnv())\nprint(json.dumps([{'kind':r.kind,'codes':r.codes} for n,a in calls for r in [s.apply(n,a)]]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify([doc, calls]), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const session = new NativeSession(new NativeRuntime(), buildPending(doc), new TypeEnv());
  const actual = [];
  for (const [name, args] of calls) {
    const result = await session.applyAsync(name, args);
    actual.push({ kind: result.kind, codes: result.codes ?? [] });
  }
  assert.deepEqual(actual, expected);
});

test('native pending loader rejects malformed fields and capabilities at Python paths', { skip: !python }, () => {
  const docs = [
    { $lambda: { type: 'Lambda<{}, Num>', code: 'return 1;', surprise: true } },
    { $lambda: { type: 'Lambda<{}, Num>', code: 'return 1;', effects: 'out.emit' } },
    { $lambda: { type: 'Lambda<{ x: Num }, Num>', code: 'return args.x;', args: [1] } },
    { $map: { type: 'Map<Num, Num>', over: [1], fn: { $lambda: {
      type: 'Lambda<{ item: Num }, Num>', code: 'return args.item;' } }, surprise: true } },
  ];
  const script = `import json,sys\nfrom natlang.values import load_program\nfrom natlang.diag import Reject\nfor doc in json.load(sys.stdin):\n try: load_program(doc); print(json.dumps(None))\n except Reject as e: print(json.dumps([[d.path,d.code] for d in e.diags]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(docs), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = py.stdout.trim().split('\n').map(JSON.parse);
  const actual = docs.map(doc => {
    try { buildPending(doc); return null; }
    catch (error) { return error.diagnostics.map(item => [item.path, item.code]); }
  });
  assert.deepEqual(actual, expected);
});

test('native checked-source revision uses Python canonical key ordering', { skip: !python }, () => {
  const entries = { A: { returns: 'Num', code: 'return 1;' },
    a: { returns: 'Num', code: 'return 2;' } };
  const script = `import json,sys\nfrom natlang.codebase import from_definitions\nprint(from_definitions(json.load(sys.stdin),'A').revision)`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(entries), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  assert.equal(checkedDefinitions(entries, 'A').revision, py.stdout.trim());
});

test('native disk source loader reproduces Python triage program state', { skip: !python }, () => {
  const path = `${root}/examples/triage/main.nl`;
  const script = `import json,sys\nfrom pathlib import Path\nfrom natlang.host import load\nfrom natlang.values import dump_state\nprint(json.dumps(dump_state(load(Path(sys.argv[1]),{}))))`;
  const py = spawnSync(python, ['-c', script, path], { cwd: root, encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  assert.deepEqual(dumpState(loadFunctionFile(path)), JSON.parse(py.stdout));
});

test('native checked definitions preserve Python isolated uses scopes', { skip: !python }, () => {
  const entries = { main: { args: { input: 'Alias' }, returns: 'Num',
    instructions: 'Use helper.', types: { Alias: 'Num' }, uses: { helper: 'helper' } },
  helper: { args: { value: 'Num' }, returns: 'Num', code: 'return args.value + 1;' } };
  const script = `import json,sys\nfrom natlang.host import load_definitions\nfrom natlang.values import dump_state\n_,root=load_definitions(json.load(sys.stdin),'main')\nprint(json.dumps(dump_state(root)))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(entries), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  assert.deepEqual(dumpState(checkedDefinitions(entries, 'main').instantiate()), JSON.parse(py.stdout));
});

test('native workspace text and core tool alternatives agree with Python surface', { skip: !python }, () => {
  const doc = { $lambda: { type: 'Lambda<{ number: Num, text?: Text }, { count: Num, label: Text }>',
    instructions: 'Copy the number and label it.', args: { number: 3 },
    return: { count: 3 } } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program\nfrom natlang.surface import ToolSurface\nroot=load_program(json.load(sys.stdin))\ns=Session(Runtime(None),root,TypeEnv())\nt=ToolSurface()\ntools={x['function']['name']:x['function']['parameters'] for x in t.tools(s)}\nprint(json.dumps({'state':t.render_state(s),'missing':t.missing(s),'read':tools['read']['x-natlang-alternatives'],'write':tools['write']['x-natlang-alternatives'],'value':tools['write']['properties']['value']}))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const session = new NativeSession(new NativeRuntime(), buildPending(doc), new TypeEnv());
  const agent = new NativeToolAgent(() => ({ calls: [] }));
  const definitions = Object.fromEntries(agent.tools(session).map(item =>
    [item.function.name, item.function.parameters]));
  assert.equal(agent.opening(session), expected.state);
  assert.equal(agent.missing(session), expected.missing);
  assert.deepEqual(definitions.read['x-natlang-alternatives'], expected.read);
  assert.deepEqual(definitions.write['x-natlang-alternatives'], expected.write);
  assert.deepEqual(definitions.write.properties.value, expected.value);
});

test('native complete leaf tool schema equals Python tools-v3', { skip: !python }, () => {
  const doc = { $lambda: { type: 'Lambda<{ x: Num }, Num>', instructions: 'Return x.', args: { x: 3 } } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program\nfrom natlang.surface import ToolSurface\ns=Session(Runtime(None,executors={'typescript-host':object()},engine_selection=True),load_program(json.load(sys.stdin)),TypeEnv())\nprint(json.dumps(ToolSurface().tools(s)))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const session = new NativeSession(new NativeRuntime(), buildPending(doc), new TypeEnv());
  assert.deepEqual(new NativeToolAgent(() => ({ calls: [] })).tools(session), expected);
});

test('native marked request text matches Python program and function listing', { skip: !python }, async () => {
  const docs = [
    { $lambda: { type: 'Lambda<{ item: Num }, Num>', instructions: '1. Double the item.\n2. Write the answer.',
      args: { item: 4 }, codebase: { double: { args: { item: 'Num' }, returns: 'Num', code: 'return args.item * 2;' } } } },
    { $lambda: { type: 'Lambda<{}, Num>', instructions: Array.from({ length: 11 }, (_, index) => `${index + 1}. Step.`).join('\n'),
      codebase: { a: { returns: 'Num', description: 'A number.', code: 'return 1;' },
        longer: { returns: 'Num', description: 'Another number.', code: 'return 2;' } } } },
  ];
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program\nfrom natlang.surface import ToolSurface\nprint(json.dumps([ToolSurface().render_request(Session(Runtime(None),load_program(doc),TypeEnv())) for doc in json.load(sys.stdin)]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(docs), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const actual = [];
  for (const doc of docs) {
    let request;
    const agent = new NativeToolAgent(turn => { request = turn.messages[1].content;
      return { calls: [], text: '', completion_tokens: 1 }; });
    await new NativeRuntime({ agent: session => agent.run(session) }).runRoot(doc);
    actual.push(request);
  }
  assert.deepEqual(actual, expected);
});

test('native review prompts and tools match Python across prompt variants', { skip: !python }, async () => {
  for (const variant of ['baseline', 'repeat_instructions', 'checklist']) {
    let initial, review;
    const calls = [['edit', { path: 'instructions', old: 'seven', new: '7' }]];
    const agent = new NativeToolAgent(request => { initial = request.messages;
      return { calls, completion_tokens: 1 }; }, { review: {
      scope: 'actions', prompt: variant, driver: request => { review = request;
        return { calls: [['review_write', { reason: 'Reject the edit.', decision: 'withdraw' }]], completion_tokens: 1 }; },
    } });
    await new NativeRuntime({ agent: session => agent.run(session) }).runRoot({ $lambda: {
      type: 'Lambda<{}, Num>', instructions: 'Return seven.' } });
    const script = `import json,sys\nfrom natlang.tool_agent import review_messages,review_tools\nmessages,calls,variant=json.load(sys.stdin)\nprint(json.dumps({'prompt':review_messages(messages,calls,0,variant)[-1]['content'],'tools':review_tools()}))`;
    const py = spawnSync(python, ['-c', script], { cwd: root,
      input: JSON.stringify([initial, calls, variant]), encoding: 'utf8' });
    assert.equal(py.status, 0, py.stderr);
    const expected = JSON.parse(py.stdout);
    assert.equal(review.messages.at(-1).content, expected.prompt, variant);
    assert.deepEqual(review.tools, expected.tools);
  }
});

test('native complete checked-call and completion-mark schema equals Python tools-v3', { skip: !python }, () => {
  const doc = { $lambda: { type: 'Lambda<{ item: Num, items: Num[] }, Num>',
    instructions: '1. Double the item.\n2. Write the answer.', args: { item: 4, items: [2, 3] },
    codebase: { double: { args: { item: 'Num' }, returns: 'Num', code: 'return args.item * 2;' },
      is_enough: { args: { value: 'Num' }, returns: 'Bool', code: 'return args.value > 10;' } } } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program\nfrom natlang.surface import ToolSurface\ns=Session(Runtime(None,executors={'typescript-host':object()},engine_selection=True),load_program(json.load(sys.stdin)),TypeEnv())\nprint(json.dumps(ToolSurface().tools(s)))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const session = new NativeSession(new NativeRuntime(), buildPending(doc), new TypeEnv());
  const agent = new NativeToolAgent(() => ({ calls: [] }));
  assert.deepEqual(agent.tools(session), expected);
});

test('native checked-call rejection codes agree with Python for malformed bindings', { skip: !python }, async () => {
  const doc = { $lambda: { type: 'Lambda<{ item: Num, word: Text, flags: Bool[] }, Num>',
    instructions: 'Use double.', args: { item: 4, word: 'hello', flags: [true] },
    codebase: { double: { args: { item: 'Num' }, returns: 'Num', code: 'return args.item * 2;' } } } };
  const calls = [
    ['call', { function: 'missing', to: 'let/result' }],
    ['call', { function: 'double', to: 'let/result', inputs: {} }],
    ['call', { function: 'double', to: 'let/result', inputs: { unknown: 'args/item' } }],
    ['call', { function: 'double', to: 'let/result', inputs: { item: 'args/word' } }],
    ['call', { function: 'double', to: 'let/result', over: 'args/word' }],
    ['call', { function: 'double', to: 'let/result', over: 'args/flags' }],
  ];
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program\ndoc,calls=json.load(sys.stdin)\ns=Session(Runtime(None),load_program(doc),TypeEnv())\nprint(json.dumps([{'kind':r.kind,'codes':r.codes} for n,a in calls for r in [s.apply(n,a)]]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify([doc, calls]), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const session = new NativeSession(new NativeRuntime(), buildPending(doc), new TypeEnv());
  const actual = [];
  for (const [name, args] of calls) {
    const result = await session.applyAsync(name, args);
    actual.push({ kind: result.kind, codes: result.codes ?? [] });
  }
  assert.deepEqual(actual, expected);
});

test('native complete pending-child tool schema equals Python tools-v3', { skip: !python }, () => {
  const doc = { $lambda: { type: 'Lambda<{}, { answer: Text }>', instructions: 'Use the nested task.',
    return: { answer: { $lambda: { type: 'Lambda<{ question: Text }, Text>', instructions: 'Answer the question.' } } } } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program\nfrom natlang.surface import ToolSurface\ns=Session(Runtime(None,executors={'typescript-host':object()},engine_selection=True),load_program(json.load(sys.stdin)),TypeEnv())\nprint(json.dumps(ToolSurface().tools(s)))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const session = new NativeSession(new NativeRuntime(), buildPending(doc), new TypeEnv());
  const agent = new NativeToolAgent(() => ({ calls: [] }));
  assert.deepEqual(agent.tools(session), expected);
});

test('native checked functions compose Map, Fold and Iterate like Python', { skip: !python }, async () => {
  const doc = { $lambda: { type: 'Lambda<{ nums: Num[], start: Num }, { mapped: Num[], sum: Num, finish: Num }>',
    instructions: 'Map, fold, and iterate.', args: { nums: [1, 2, 3], start: 0 }, codebase: {
      double: { args: { item: 'Num' }, returns: 'Num', code: 'return args.item * 2;' },
      add: { args: { acc: 'Num', item: 'Num' }, returns: 'Num', code: 'return args.acc + args.item;' },
      step: { args: { value: 'Num' }, returns: 'Num', code: 'return args.value + 1;' },
      done: { args: { value: 'Num' }, returns: 'Bool', code: 'return args.value >= 3;' },
    } } };
  const calls = [
    ['call', { function: 'double', to: 'return/mapped', over: 'args/nums' }],
    ['call', { function: 'add', to: 'return/sum', over: 'args/nums', init: 0 }],
    ['call', { function: 'step', to: 'return/finish', init: 'args/start', until: 'done', max: 5 }],
  ];
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program,dump\ndoc,calls=json.load(sys.stdin)\nroot=load_program(doc)\ns=Session(Runtime(None),root,TypeEnv())\nresults=[s.apply(n,a) for n,a in calls]\nprint(json.dumps({'kinds':[r.kind for r in results], 'codes':[r.codes for r in results], 'value':dump(root.ret)}))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify([doc, calls]), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const lam = buildPending(doc), session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const actual = [];
  for (const [name, args] of calls) actual.push(await session.applyAsync(name, args));
  assert.deepEqual(actual.map(result => result.kind), expected.kinds);
  assert.deepEqual(actual.map(result => result.codes ?? []), expected.codes);
  assert.deepEqual(dump(lam.return), expected.value);
});

test('native editable function copies keep the checked source immutable', { skip: !python }, async () => {
  const doc = { $lambda: { type: 'Lambda<{ value: Num }, Num>', instructions: 'Use a copied function.',
    args: { value: 3 }, codebase: { inc: { args: { value: 'Num' }, returns: 'Num',
      code: 'return args.value + 1;' } } } };
  const calls = [
    ['write', { path: 'let/twice', type: 'Function<inc>' }],
    ['edit', { path: 'let/twice/code', old: '+ 1', new: '* 2' }],
    ['call', { function: 'let/twice', to: 'return', inputs: { value: 'args/value' } }],
  ];
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program,dump\ndoc,calls=json.load(sys.stdin)\nroot=load_program(doc)\ns=Session(Runtime(None),root,TypeEnv())\nresults=[s.apply(n,a) for n,a in calls]\nprint(json.dumps({'kinds':[r.kind for r in results], 'codes':[r.codes for r in results], 'value':dump(root.ret)}))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify([doc, calls]), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const lam = buildPending(doc), session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const actual = [];
  for (const [name, args] of calls) actual.push(await session.applyAsync(name, args));
  assert.deepEqual(actual.map(result => result.kind), expected.kinds);
  assert.deepEqual(actual.map(result => result.codes ?? []), expected.codes);
  assert.deepEqual(dump(lam.return), expected.value);
  assert.equal(lam.codebase.inc.code, 'return args.value + 1;\n');
});

test('native tool rejection text includes Python diagnostic hints', { skip: !python }, () => {
  const doc = { $lambda: { type: 'Lambda<{ x: Text }, Text>', instructions: 'Return x.', args: { x: 'abc' } } };
  const calls = [
    ['edit', { path: 'instructions', old: 'missing', new: 'x' }],
    ['write', { path: 'args/x', type: 'Text', value: 'z' }],
  ];
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program\ndoc,calls=json.load(sys.stdin)\ns=Session(Runtime(None),load_program(doc),TypeEnv())\nprint(json.dumps([s.apply(n,a).text for n,a in calls]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify([doc, calls]), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const session = new NativeSession(new NativeRuntime(), buildPending(doc), new TypeEnv());
  assert.deepEqual(calls.map(([name, args]) => session.apply(name, args).text), expected);
});

test('native trace preserves declared effect order before a failed eval', { skip: !python }, async () => {
  const doc = { $lambda: { type: 'Lambda<{}, Num>', effects: ['out.emit'],
    code: "fx.out.emit({ id: 'first' }); throw new Error('failed');" } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime\nfrom natlang.trace import TraceRecorder\nfrom natlang.values import load_program\nsink=TraceRecorder({})\nrt=Runtime(None,trace_sink=sink)\nout,value=rt.run_root(load_program(json.load(sys.stdin)))\nprint(json.dumps({'outcome':out.kind,'emitted':rt.emitted,'effects':[(e['phase'],e['capability']) for e in sink.events if e['kind']=='effect'],'evals':[e['phase'] for e in sink.events if e['kind']=='eval'],'trace':sink.events[1:]}))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const runtime = new NativeRuntime();
  const actual = await runtime.runRoot(doc);
  assert.equal(actual.outcome.kind, expected.outcome);
  assert.deepEqual(actual.emitted, expected.emitted);
  assert.deepEqual(runtime.trace.events.filter(event => event.kind === 'effect').map(event =>
    [event.phase, event.capability]), expected.effects);
  assert.deepEqual(runtime.trace.events.filter(event => event.kind === 'eval').map(event => event.phase), expected.evals);
  assert.equal(runtime.trace.events.find(event => event.kind === 'eval' && event.phase === 'start').engine,
    'typescript-host');
  assertTraceParity(runtime.trace.events.slice(1).map(event => {
    if (event.kind !== 'eval' || event.phase !== 'start') return event;
    const { declared_engine, ...rest } = event;
    return { ...rest, engine: declared_engine };
  }), expected.trace);
  assert.deepEqual(runtime.trace.reconstruct(), runtime.trace.finalState());
});

test('native finite crisp reduction trace matches Python events apart from executor identity', { skip: !python }, async () => {
  const doc = { $lambda: { type: 'Lambda<{}, Num>', code: 'return 7;' } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime\nfrom natlang.trace import TraceRecorder\nfrom natlang.values import load_program\nsink=TraceRecorder({})\nRuntime(None,trace_sink=sink).run_root(load_program(json.load(sys.stdin)))\nprint(json.dumps(sink.events[1:]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const runtime = new NativeRuntime();
  await runtime.runRoot(doc);
  const actual = runtime.trace.events.slice(1).map(event => {
    if (event.kind !== 'eval' || event.phase !== 'start') return event;
    const { declared_engine, ...rest } = event;
    return { ...rest, engine: declared_engine };
  });
  assertTraceParity(actual, expected);
});

test('native model-driven leaf trace matches Python actions and state observations', { skip: !python }, async () => {
  const doc = { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Write seven.' } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime\nfrom natlang.trace import TraceRecorder\nfrom natlang.values import load_program\nfrom natlang.tool_agent import ToolAgent\nfrom natlang.decoder import ChatTurn\nclass Decoder:\n def __init__(self): self.turns=iter([ChatTurn([('write',{'path':'return','type':'Num','value':7})],completion_tokens=1),ChatTurn([], 'done',completion_tokens=1)])\n def chat(self,*args,**kwargs): return next(self.turns)\nsink=TraceRecorder({})\nRuntime(lambda lam:ToolAgent(Decoder()),trace_sink=sink).run_root(load_program(json.load(sys.stdin)))\nprint(json.dumps(sink.events[1:]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  let turn = 0;
  const agent = new NativeToolAgent(() => ++turn === 1 ?
    { calls: [['write', { path: 'return', type: 'Num', value: 7 }]], completion_tokens: 1 } :
    { calls: [], text: 'done', completion_tokens: 1 });
  const runtime = new NativeRuntime({ agent: session => agent.run(session) });
  await runtime.runRoot(doc);
  const actual = runtime.trace.events.slice(1).map(event => event.kind === 'action' ?
    { ...event, surface: 'tools-v2' } : event);
  assertTraceParity(actual, expected);
});

test('native nested model Map trace matches Python invocation and action order', { skip: !python }, async () => {
  const doc = { $lambda: { type: 'Lambda<{ items: Num[] }, Num[]>', instructions: 'Double every item.',
    args: { items: [2, 3] }, codebase: { double: { args: { item: 'Num' }, returns: 'Num',
      instructions: 'Double the item.' } } } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime\nfrom natlang.trace import TraceRecorder\nfrom natlang.values import load_program\nfrom natlang.tool_agent import ToolAgent\nfrom natlang.decoder import ChatTurn\nclass Decoder:\n def __init__(self,lam): self.lam=lam; self.index=0\n def chat(self,*args,**kwargs):\n  self.index+=1\n  if self.index>1: return ChatTurn([], 'done',completion_tokens=1)\n  if self.lam.fn_name=='double': return ChatTurn([('write',{'path':'return','type':'Num','value':self.lam.in_['item']*2})],completion_tokens=1)\n  return ChatTurn([('call',{'function':'double','to':'return','over':'args/items'})],completion_tokens=1)\nsink=TraceRecorder({})\nRuntime(lambda lam:ToolAgent(Decoder(lam)),trace_sink=sink).run_root(load_program(json.load(sys.stdin)))\nprint(json.dumps(sink.events[1:]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const runtime = new NativeRuntime({ agent: session => {
    let index = 0;
    const agent = new NativeToolAgent(() => {
      index++;
      if (index > 1) return { calls: [], text: 'done', completion_tokens: 1 };
      if (session.lam.functionName === 'double') return { calls: [['write', {
        path: 'return', type: 'Num', value: session.lam.args.item * 2 }]], completion_tokens: 1 };
      return { calls: [['call', { function: 'double', to: 'return', over: 'args/items' }]], completion_tokens: 1 };
    });
    return agent.run(session);
  } });
  await runtime.runRoot(doc);
  const actual = runtime.trace.events.slice(1).map(event => event.kind === 'action' ?
    { ...event, surface: 'tools-v2' } : event);
  assertTraceParity(actual, expected);
});

test('native nested model Fold trace matches Python state and invocation order', { skip: !python }, async () => {
  const doc = { $lambda: { type: 'Lambda<{ items: Num[] }, Num>', instructions: 'Sum every item.',
    args: { items: [2, 3] }, codebase: { add: { args: { acc: 'Num', item: 'Num' }, returns: 'Num',
      instructions: 'Add the item to the accumulator.' } } } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime\nfrom natlang.trace import TraceRecorder\nfrom natlang.values import load_program\nfrom natlang.tool_agent import ToolAgent\nfrom natlang.decoder import ChatTurn\nclass Decoder:\n def __init__(self,lam): self.lam=lam; self.index=0\n def chat(self,*args,**kwargs):\n  self.index+=1\n  if self.index>1: return ChatTurn([], 'done',completion_tokens=1)\n  if self.lam.fn_name=='add': return ChatTurn([('write',{'path':'return','type':'Num','value':self.lam.in_['acc']+self.lam.in_['item']})],completion_tokens=1)\n  return ChatTurn([('call',{'function':'add','to':'return','over':'args/items','init':0})],completion_tokens=1)\nsink=TraceRecorder({})\nRuntime(lambda lam:ToolAgent(Decoder(lam)),trace_sink=sink).run_root(load_program(json.load(sys.stdin)))\nprint(json.dumps(sink.events[1:]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const runtime = new NativeRuntime({ agent: session => {
    let index = 0;
    const agent = new NativeToolAgent(() => {
      index++;
      if (index > 1) return { calls: [], text: 'done', completion_tokens: 1 };
      if (session.lam.functionName === 'add') return { calls: [['write', {
        path: 'return', type: 'Num', value: session.lam.args.acc + session.lam.args.item }]], completion_tokens: 1 };
      return { calls: [['call', { function: 'add', to: 'return', over: 'args/items', init: 0 }]], completion_tokens: 1 };
    });
    return agent.run(session);
  } });
  await runtime.runRoot(doc);
  const actual = runtime.trace.events.slice(1).map(event => event.kind === 'action' ?
    { ...event, surface: 'tools-v2' } : event);
  assertTraceParity(actual, expected);
});

test('native nested model Iterate trace matches Python step and check order', { skip: !python }, async () => {
  const doc = { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Count to two.',
    codebase: { inc: { args: { state: 'Num' }, returns: 'Num', instructions: 'Increment the state.' },
      enough: { args: { value: 'Num' }, returns: 'Bool', instructions: 'Check if value reached two.' } } } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime\nfrom natlang.trace import TraceRecorder\nfrom natlang.values import load_program\nfrom natlang.tool_agent import ToolAgent\nfrom natlang.decoder import ChatTurn\nclass Decoder:\n def __init__(self,lam): self.lam=lam; self.index=0\n def chat(self,*args,**kwargs):\n  self.index+=1\n  if self.index>1: return ChatTurn([], 'done',completion_tokens=1)\n  if self.lam.fn_name=='inc': return ChatTurn([('write',{'path':'return','type':'Num','value':self.lam.in_['state']+1})],completion_tokens=1)\n  if self.lam.fn_name=='enough': return ChatTurn([('write',{'path':'return','type':'Bool','value':self.lam.in_['value']>=2})],completion_tokens=1)\n  return ChatTurn([('call',{'function':'inc','to':'return','until':'enough','init':0,'max':4})],completion_tokens=1)\nsink=TraceRecorder({})\nRuntime(lambda lam:ToolAgent(Decoder(lam)),trace_sink=sink).run_root(load_program(json.load(sys.stdin)))\nprint(json.dumps(sink.events[1:]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const runtime = new NativeRuntime({ agent: session => {
    let index = 0;
    const agent = new NativeToolAgent(() => {
      index++;
      if (index > 1) return { calls: [], text: 'done', completion_tokens: 1 };
      if (session.lam.functionName === 'inc') return { calls: [['write', {
        path: 'return', type: 'Num', value: session.lam.args.state + 1 }]], completion_tokens: 1 };
      if (session.lam.functionName === 'enough') return { calls: [['write', {
        path: 'return', type: 'Bool', value: session.lam.args.value >= 2 }]], completion_tokens: 1 };
      return { calls: [['call', { function: 'inc', to: 'return', until: 'enough', init: 0, max: 4 }]], completion_tokens: 1 };
    });
    return agent.run(session);
  } });
  await runtime.runRoot(doc);
  const actual = runtime.trace.events.slice(1).map(event => event.kind === 'action' ?
    { ...event, surface: 'tools-v2' } : event);
  assertTraceParity(actual, expected);
});

test('native quiesced root resumes with the same Python trace transitions', { skip: !python }, async () => {
  const doc = { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return seven once available.' } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime\nfrom natlang.trace import TraceRecorder\nfrom natlang.values import load_program\nfrom natlang.tool_agent import ToolAgent\nfrom natlang.decoder import ChatTurn\nturns=[0]\nclass Decoder:\n def chat(self,*args,**kwargs):\n  turns[0]+=1\n  if turns[0]==1: return ChatTurn([('report_blocker',{'missing':'the number is unavailable'})],completion_tokens=1)\n  if turns[0]==2: return ChatTurn([('write',{'path':'return','type':'Num','value':7})],completion_tokens=1)\n  return ChatTurn([], 'done',completion_tokens=1)\nsink=TraceRecorder({})\nrt=Runtime(lambda lam:ToolAgent(Decoder()),trace_sink=sink)\nnode=load_program(json.load(sys.stdin))\nrt.run_root(node)\nrt.run_root(node)\nprint(json.dumps(sink.events[1:]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  let turns = 0;
  const agent = new NativeToolAgent(() => {
    turns++;
    if (turns === 1) return { calls: [['report_blocker', { missing: 'the number is unavailable' }]], completion_tokens: 1 };
    if (turns === 2) return { calls: [['write', { path: 'return', type: 'Num', value: 7 }]], completion_tokens: 1 };
    return { calls: [], text: 'done', completion_tokens: 1 };
  });
  const runtime = new NativeRuntime({ agent: session => agent.run(session) });
  const node = buildPending(doc);
  await runtime.runRoot(node);
  await runtime.runRoot(node);
  const actual = runtime.trace.events.slice(1).map(event => event.kind === 'action' ?
    { ...event, surface: 'tools-v2' } : event);
  assertTraceParity(actual, expected);
});

test('native live Fold stream matches Python waiting and resumed trace', { skip: !python }, async () => {
  const doc = { $fold: { type: 'Fold<Num, Num>', init: 0,
    step: { $lambda: { type: 'Lambda<{ acc: Num, item: Num }, Num>', code: 'return args.acc + args.item;' } } } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime\nfrom natlang.trace import TraceRecorder\nfrom natlang.values import load_program\nfrom natlang.streams import QueueSource,StreamBuffer\nsink=TraceRecorder({})\nsource=QueueSource();source.put(2)\nnode=load_program(json.load(sys.stdin));node.over=StreamBuffer(source)\nrt=Runtime(None,trace_sink=sink)\nrt.run_root(node)\nsource.put(3);source.close()\nrt.run_root(node)\nprint(json.dumps(sink.events[1:]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const polls = [{ kind: 'item', value: 2 }, { kind: 'empty' },
    { kind: 'item', value: 3 }, { kind: 'closed' }];
  const runtime = new NativeRuntime({ stream: { poll: () => polls.shift() } });
  const node = buildPending(doc);
  await runtime.runRoot(node);
  await runtime.runRoot(node);
  const actual = runtime.trace.events.slice(1).map(event => {
    if (event.kind !== 'eval' || event.phase !== 'start') return event;
    const { declared_engine, ...rest } = event;
    return { ...rest, engine: declared_engine };
  });
  assertTraceParity(actual, expected);
});

test('native approved review preserves Python proposal trace and audit decisions', { skip: !python }, async () => {
  const doc = { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return seventeen.' } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime\nfrom natlang.trace import TraceRecorder\nfrom natlang.values import load_program\nfrom natlang.tool_agent import ToolAgent\nfrom natlang.decoder import ChatTurn\nclass Decoder:\n def __init__(self): self.index=0\n def chat(self,*args,**kwargs):\n  self.index+=1\n  if self.index==1: return ChatTurn([('write',{'path':'return','type':'Num','value':17})],completion_tokens=1,value_confidence=[{'geometric_mean':0.1}])\n  if self.index==2: return ChatTurn([('review_write',{'decision':'approve','reason':'The value is correct.'})],completion_tokens=1)\n  return ChatTurn([], 'done',completion_tokens=1)\nsink=TraceRecorder({})\nproposals=[];reviews=[]\nRuntime(lambda lam:ToolAgent(Decoder(),careful_threshold=0.5,proposals=proposals,reviews=reviews),trace_sink=sink).run_root(load_program(json.load(sys.stdin)))\nprint(json.dumps({'events':sink.events[1:],'proposals':[{'released':p['released'],'confidence':p['value_confidence']} for p in proposals], 'reviews':[{'decision':r['decision'],'trigger':r['trigger'],'order':r['order'],'prompt_variant':r['prompt_variant']} for r in reviews]}))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  let turns = 0;
  const driver = () => ++turns === 1 ? { calls: [['write', { path: 'return', type: 'Num', value: 17 }]],
    completion_tokens: 1, value_confidence: [{ geometric_mean: 0.1 }] } : turns === 2 ?
    { calls: [['review_write', { decision: 'approve', reason: 'The value is correct.' }]], completion_tokens: 1 } :
    { calls: [], text: 'done', completion_tokens: 1 };
  const agent = new NativeToolAgent(driver, { review: { threshold: 0.5, driver } });
  const runtime = new NativeRuntime({ agent: session => agent.run(session) });
  await runtime.runRoot(doc);
  const actual = runtime.trace.events.slice(1).map(event => event.kind === 'action' ?
    { ...event, surface: 'tools-v2' } : event);
  assertTraceParity(actual, expected.events);
  assert.deepEqual(agent.proposals.map(item => ({ released: item.released, confidence: item.value_confidence })),
    expected.proposals);
  assert.deepEqual(agent.reviews.map(item => ({ decision: item.decision, trigger: item.trigger,
    order: item.order, prompt_variant: item.prompt_variant })), expected.reviews);
});

test('native withdrawn review retries from unchanged state like Python', { skip: !python }, async () => {
  const doc = { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return seventeen.' } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime\nfrom natlang.trace import TraceRecorder\nfrom natlang.values import load_program\nfrom natlang.tool_agent import ToolAgent\nfrom natlang.decoder import ChatTurn\nclass Decoder:\n def __init__(self): self.index=0\n def chat(self,*args,**kwargs):\n  self.index+=1\n  if self.index==1: return ChatTurn([('write',{'path':'return','type':'Num','value':99})],completion_tokens=1,value_confidence=[{'geometric_mean':0.1}])\n  if self.index==2: return ChatTurn([('review_write',{'decision':'withdraw','reason':'The value must be seventeen.'})],completion_tokens=1)\n  if self.index==3: return ChatTurn([('write',{'path':'return','type':'Num','value':17})],completion_tokens=1)\n  return ChatTurn([], 'done',completion_tokens=1)\nsink=TraceRecorder({})\nproposals=[];reviews=[]\nRuntime(lambda lam:ToolAgent(Decoder(),careful_threshold=0.5,withdrawal_policy='retry',proposals=proposals,reviews=reviews),trace_sink=sink).run_root(load_program(json.load(sys.stdin)))\nprint(json.dumps({'events':sink.events[1:],'proposals':[{'released':p['released'],'withdrawn':p.get('withdrawn',False)} for p in proposals], 'reviews':[{'decision':r['decision'],'trigger':r['trigger']} for r in reviews]}))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  let turns = 0;
  const driver = () => ++turns === 1 ? { calls: [['write', { path: 'return', type: 'Num', value: 99 }]],
    completion_tokens: 1, value_confidence: [{ geometric_mean: 0.1 }] } : turns === 2 ?
    { calls: [['review_write', { decision: 'withdraw', reason: 'The value must be seventeen.' }]], completion_tokens: 1 } :
    turns === 3 ? { calls: [['write', { path: 'return', type: 'Num', value: 17 }]], completion_tokens: 1 } :
    { calls: [], text: 'done', completion_tokens: 1 };
  const agent = new NativeToolAgent(driver, { review: { threshold: 0.5, withdrawalPolicy: 'retry', driver } });
  const runtime = new NativeRuntime({ agent: session => agent.run(session) });
  await runtime.runRoot(doc);
  const actual = runtime.trace.events.slice(1).map(event => event.kind === 'action' ?
    { ...event, surface: 'tools-v2' } : event);
  assertTraceParity(actual, expected.events);
  assert.deepEqual(agent.proposals.map(item => ({ released: item.released, withdrawn: item.withdrawn ?? false })),
    expected.proposals);
  assert.deepEqual(agent.reviews.map(item => ({ decision: item.decision, trigger: item.trigger })), expected.reviews);
});

test('native completion-mark output matches Python compact listing', { skip: !python }, () => {
  const doc = { $lambda: { type: 'Lambda<{}, Num>',
    instructions: '1. Do this.\n2. Do that.\n3. Skip this.\n4. Finish.' } };
  const calls = [
    ['mark_done', { start: 1, end: 2 }],
    ['mark_done', { start: 3, skipped: true }],
    ['mark_done', { start: 7 }],
  ];
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program\ndoc,calls=json.load(sys.stdin)\ns=Session(Runtime(None),load_program(doc),TypeEnv())\nprint(json.dumps([s.apply(n,a).text for n,a in calls]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify([doc, calls]), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const session = new NativeSession(new NativeRuntime(), buildPending(doc), new TypeEnv());
  assert.deepEqual(calls.map(([name, args]) => session.apply(name, args).text), expected);
});

test('native successful writes and edits include Python progress text', { skip: !python }, () => {
  const doc = { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Write one.' } };
  const calls = [
    ['write', { path: 'let/draft', type: 'Text', value: 'one word' }],
    ['edit', { path: 'let/draft', old: 'one', new: 'two' }],
    ['write', { path: 'return', type: 'Num', value: 1 }],
  ];
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program\ndoc,calls=json.load(sys.stdin)\ns=Session(Runtime(None),load_program(doc),TypeEnv())\nprint(json.dumps([s.apply(n,a).text for n,a in calls]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify([doc, calls]), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const session = new NativeSession(new NativeRuntime(), buildPending(doc), new TypeEnv());
  assert.deepEqual(calls.map(([name, args]) => session.apply(name, args).text), expected);
});

test('native blocker and error reports preserve Python result text', { skip: !python }, () => {
  const doc = { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Use unavailable information.' } };
  const calls = [
    ['report_blocker', { missing: 'x' }],
    ['report_error', { message: 'x' }],
    ['report_blocker', { missing: 'The source document is missing.' }],
    ['report_error', { message: 'The requested result is impossible.' }],
  ];
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program\ndoc,calls=json.load(sys.stdin)\ns=Session(Runtime(None),load_program(doc),TypeEnv())\nprint(json.dumps([{'kind':r.kind,'text':r.text} for n,a in calls for r in [s.apply(n,a)]]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify([doc, calls]), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const session = new NativeSession(new NativeRuntime(), buildPending(doc), new TypeEnv());
  assert.deepEqual(calls.map(([name, args]) => {
    const result = session.apply(name, args); return { kind: result.kind, text: result.text };
  }), expected);
});

test('native ranged reads match Python line and item numbering', { skip: !python }, () => {
  const doc = { $lambda: { type: 'Lambda<{ lines: Text, items: Num[] }, Num>',
    instructions: 'Read the supplied inputs.', args: { lines: 'first\nsecond\nthird', items: [10, 20, 30] } } };
  const calls = [
    ['read', { path: 'args/lines', start: 2, end: 3 }],
    ['read', { path: 'args/items', start: 1, end: 2 }],
    ['read', { path: 'args/items', start: 0 }],
  ];
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program\ndoc,calls=json.load(sys.stdin)\ns=Session(Runtime(None),load_program(doc),TypeEnv())\nprint(json.dumps([{'kind':r.kind,'text':r.text} for n,a in calls for r in [s.apply(n,a)]]))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify([doc, calls]), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const session = new NativeSession(new NativeRuntime(), buildPending(doc), new TypeEnv());
  assert.deepEqual(calls.map(([name, args]) => {
    const result = session.apply(name, args); return { kind: result.kind, text: result.text };
  }), expected);
});
