import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NativeRuntime } from '../dist/native/runtime.js';
import { NativeToolAgent } from '../dist/native/agent.js';
import { checkedDefinitions } from '../dist/native/codebase.js';

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

test('native codebase call binds typed inputs and reduces a child', async () => {
  const actions = [];
  const runtime = new NativeRuntime({ agent: async session => {
    actions.push(await session.applyAsync('call', { function: 'double', to: 'return',
      inputs: { item: 'args/item' } }));
    session.finish();
  } });
  const result = await runtime.runRoot({ $lambda: {
    type: 'Lambda<{ item: Num }, Num>', instructions: 'Double the item.', args: { item: 8 },
    codebase: { double: { args: { item: 'Num' }, returns: 'Num', engine: 'typescript-host',
      code: 'return args.item * 2;' } },
  } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 16);
  assert.equal(actions[0].kind, 'done');
});

test('native model-turn loop drives tool actions without Python', async () => {
  let turn = 0;
  const agent = new NativeToolAgent(() => ++turn === 1 ? {
    calls: [['write', { path: 'return', type: 'Num', value: 11 }]], completion_tokens: 3,
  } : { calls: [], text: 'done', completion_tokens: 2 });
  const runtime = new NativeRuntime({ agent: session => agent.run(session) });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return eleven.' } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 11); assert.equal(turn, 2);
});

test('native checked definitions snapshot and link source calls', async () => {
  const source = { main: { args: { price: 'Num' }, returns: 'Num',
    instructions: 'Use double.', uses: { double: 'helper' } },
    helper: { args: { item: 'Num' }, returns: 'Num', code: 'return args.item * 2;' } };
  const graph = checkedDefinitions(source, 'main');
  source.helper.code = 'return 999;';
  const runtime = new NativeRuntime({ agent: async session => {
    const out = await session.applyAsync('call', { function: 'double', to: 'return',
      inputs: { item: 'args/price' } });
    assert.equal(out.kind, 'done'); session.finish();
  } });
  const result = await runtime.runRoot(graph.instantiate({ price: 8 }));
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 16);
  assert.throws(() => checkedDefinitions({ a: { returns: 'Num', code: 'return 1;', uses: { a: 'a' } } }, 'a'), /recursion/);
});

test('native source calls lower to Map and Fold combinators', async () => {
  const actions = [];
  const runtime = new NativeRuntime({ agent: async session => {
    actions.push(await session.applyAsync('call', { function: 'double', to: 'let/doubled', over: 'args/items' }));
    actions.push(await session.applyAsync('call', { function: 'add', to: 'return', over: 'let/doubled', init: 0 }));
    session.finish();
  } });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{ items: Num[] }, Num>',
    instructions: 'Double then add.', args: { items: [1, 2, 3] }, codebase: {
      double: { args: { item: 'Num' }, returns: 'Num', code: 'return args.item * 2;' },
      add: { args: { acc: 'Num', item: 'Num' }, returns: 'Num', code: 'return args.acc + args.item;' },
    } } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 12);
  assert.deepEqual(actions.map(x => x.kind), ['done', 'done']);
});
