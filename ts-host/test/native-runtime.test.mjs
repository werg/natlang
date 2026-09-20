import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NativeRuntime } from '../dist/native/runtime.js';

const crisp = (type, code, extra = {}) => ({ $lambda: { type, code, engine: 'typescript-host', ...extra } });

test('native reducer executes crisp root, Map and Fold without Python', async () => {
  const one = await new NativeRuntime().runRoot(crisp('Lambda<{}, Num>', 'return 7;'));
  assert.equal(one.outcome.kind, 'done'); assert.equal(one.value, 7);
  const map = await new NativeRuntime().runRoot({ $map: { type: 'Map<Num, Num>', over: [2, 4],
    fn: crisp('Lambda<{ item: Num }, Num>', 'return args.item * 3;') } });
  assert.equal(map.outcome.kind, 'done'); assert.deepEqual(map.value, [6, 12]);
  const fold = await new NativeRuntime().runRoot({ $fold: { type: 'Fold<Num, Num>', over: [2, 4], init: 1,
    step: crisp('Lambda<{ acc: Num, item: Num }, Num>', 'return args.acc + args.item;') } });
  assert.equal(fold.outcome.kind, 'done'); assert.equal(fold.value, 7);
});

test('native episodes write through typed actions and reject wrong values', async () => {
  let actions;
  const runtime = new NativeRuntime({ agent: session => {
    const wrong = session.apply('write', { path: 'return', type: 'Num', value: 'oops' });
    const right = session.apply('write', { path: 'return', type: 'Num', value: 9 });
    actions = [wrong, right];
    session.finish();
  } });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return nine.' } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 9);
  assert.equal(actions[0].kind, 'rejected'); assert.deepEqual(actions[0].codes, ['type-mismatch']);
  assert.equal(actions[1].kind, 'ok');
});

test('native Fold can wait for more input and resume', async () => {
  let poll = 0;
  const source = { poll: () => ++poll === 1 ? { kind: 'item', value: 2 } :
    poll === 2 ? { kind: 'empty' } : poll === 3 ? { kind: 'item', value: 3 } : { kind: 'closed' } };
  const runtime = new NativeRuntime({ stream: source });
  const root = { $fold: { type: 'Fold<Num, Num>', over: [], init: 1,
    step: crisp('Lambda<{ acc: Num, item: Num }, Num>', 'return args.acc + args.item;') } };
  const first = await runtime.runRoot(root);
  assert.equal(first.outcome.kind, 'waiting');
  const second = await runtime.runRoot(first.value);
  assert.equal(second.outcome.kind, 'done'); assert.equal(second.value, 6);
});

test('native Iterate uses a checked step and Boolean completion condition', async () => {
  const result = await new NativeRuntime().runRoot({ $iterate: {
    type: 'Iterate<Num>', init: 0, max: 5, state_name: 'value', check_name: 'value',
    step: crisp('Lambda<{ value: Num }, Num>', 'return args.value + 1;'),
    check: crisp('Lambda<{ value: Num }, Bool>', 'return args.value >= 3;'),
  } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 3);
});
