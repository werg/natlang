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
import { TOOLS_PROMPT } from '../dist/native/prompt.js';

const root = fileURLToPath(new URL('../..', import.meta.url));
const python = process.env.NATLANG_PYTHON;
const SCRIPT = `import json,sys\nfrom natlang.runtime import Runtime\nfrom natlang.values import load_program,dump\nroot=load_program(json.load(sys.stdin))\nout,value=Runtime(None).run_root(root)\nprint(json.dumps({'kind':out.kind,'value':dump(value)},sort_keys=True))`;

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

test('native checked-call and completion-mark alternatives agree with Python surface', { skip: !python }, () => {
  const doc = { $lambda: { type: 'Lambda<{ item: Num, items: Num[] }, Num>',
    instructions: '1. Double the item.\n2. Write the answer.', args: { item: 4, items: [2, 3] },
    codebase: { double: { args: { item: 'Num' }, returns: 'Num', code: 'return args.item * 2;' },
      is_enough: { args: { value: 'Num' }, returns: 'Bool', code: 'return args.value > 10;' } } } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program\nfrom natlang.surface import ToolSurface\ns=Session(Runtime(None),load_program(json.load(sys.stdin)),TypeEnv())\ntools={x['function']['name']:x['function']['parameters'] for x in ToolSurface().tools(s)}\nprint(json.dumps({n:tools[n] for n in ('call','write','mark_done')}))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const session = new NativeSession(new NativeRuntime(), buildPending(doc), new TypeEnv());
  const agent = new NativeToolAgent(() => ({ calls: [] }));
  const actual = Object.fromEntries(agent.tools(session).map(item =>
    [item.function.name, item.function.parameters]));
  for (const name of ['call', 'write', 'mark_done']) {
    assert.deepEqual(actual[name]?.['x-natlang-alternatives'], expected[name]?.['x-natlang-alternatives'], name);
  }
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

test('native tool schema exposes only unbound child inputs of pending tasks', { skip: !python }, () => {
  const doc = { $lambda: { type: 'Lambda<{}, { answer: Text }>', instructions: 'Use the nested task.',
    return: { answer: { $lambda: { type: 'Lambda<{ question: Text }, Text>', instructions: 'Answer the question.' } } } } };
  const script = `import json,sys\nfrom natlang.runtime import Runtime,Session\nfrom natlang.types import TypeEnv\nfrom natlang.values import load_program\nfrom natlang.surface import ToolSurface\ns=Session(Runtime(None),load_program(json.load(sys.stdin)),TypeEnv())\ntools={x['function']['name']:x['function']['parameters'] for x in ToolSurface().tools(s)}\nprint(json.dumps({'write':tools['write']['x-natlang-alternatives'],'read':tools['read']['x-natlang-alternatives']}))`;
  const py = spawnSync(python, ['-c', script], { cwd: root, input: JSON.stringify(doc), encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const expected = JSON.parse(py.stdout);
  const session = new NativeSession(new NativeRuntime(), buildPending(doc), new TypeEnv());
  const agent = new NativeToolAgent(() => ({ calls: [] }));
  const tools = Object.fromEntries(agent.tools(session).map(item => [item.function.name, item.function.parameters]));
  assert.deepEqual(tools.write['x-natlang-alternatives'], expected.write);
  assert.deepEqual(tools.read['x-natlang-alternatives'], expected.read);
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
  assert.equal(lam.codebase.inc.code, 'return args.value + 1;');
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
  const script = `import json,sys\nfrom natlang.runtime import Runtime\nfrom natlang.trace import TraceRecorder\nfrom natlang.values import load_program\nsink=TraceRecorder({})\nrt=Runtime(None,trace_sink=sink)\nout,value=rt.run_root(load_program(json.load(sys.stdin)))\nprint(json.dumps({'outcome':out.kind,'emitted':rt.emitted,'effects':[(e['phase'],e['capability']) for e in sink.events if e['kind']=='effect'],'evals':[e['phase'] for e in sink.events if e['kind']=='eval']}))`;
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
  assert.deepEqual(runtime.trace.reconstruct(), runtime.trace.finalState());
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
