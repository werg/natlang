import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { NativeRuntime } from '../dist/native/runtime.js';
import { TypeEnv } from '../dist/index.js';
import { buildPending } from '../dist/native/values.js';
import { NativeSession } from '../dist/native/runtime.js';
import { dump } from '../dist/native/values.js';

const root = fileURLToPath(new URL('../..', import.meta.url));
const python = process.env.NATLANG_PYTHON;
const SCRIPT = `import json,sys\nfrom natlang.runtime import Runtime\nfrom natlang.values import load_program,dump\nroot=load_program(json.load(sys.stdin))\nout,value=Runtime(None).run_root(root)\nprint(json.dumps({'kind':out.kind,'value':dump(value)},sort_keys=True))`;

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
