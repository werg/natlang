import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NativeRuntime } from '../dist/native/runtime.js';
import { NativeToolAgent } from '../dist/native/agent.js';
import { checkedDefinitions } from '../dist/native/codebase.js';
import { readTrace } from '../../web/natlang_lite.mjs';
import { deriveSeed } from '../dist/native/trace.js';
import { admitNativeTrace } from '../dist/native/scenario.js';
import { buildPending, dumpState, MISSING } from '../dist/native/values.js';
import { NativeSession } from '../dist/native/runtime.js';
import { TypeEnv } from '../dist/native/types.js';

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

test('native reads may inspect read-only inputs', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ state: { head: Text } }, Text>',
    instructions: 'Inspect the state.', args: { state: { head: 'manifest-1' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = session.apply('read', { path: 'args/state' });
  assert.equal(result.kind, 'ok');
  assert.deepEqual(result.value, { head: 'manifest-1' });
});

test('explicit inference and action limits still apply', async () => {
  const program = { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return seven.' } };
  const runtime = new NativeRuntime();
  assert.equal(runtime.options.maxEpisodes, undefined);
  assert.equal(runtime.options.maxDepth, undefined);
  const capped = new NativeSession(new NativeRuntime({ maxActions: 2, maxToolCalls: 3 }),
    buildPending(program), new TypeEnv());
  assert.equal(capped.apply('read', { path: 'instructions' }).kind, 'ok');
  assert.equal(capped.apply('read', { path: 'instructions' }).kind, 'ok');
  assert.equal(capped.apply('read', { path: 'instructions' }).kind, 'budget');
  const cappedAgent = new NativeToolAgent(() => ({ calls: [['read', { path: 'instructions' }]],
    completion_tokens: 1 }), { maxTurns: 2 });
  const stopped = await new NativeRuntime({ agent: session => cappedAgent.run(session) }).runRoot(program);
  assert.equal(stopped.outcome.kind, 'quiesced');
  assert.match(stopped.outcome.detail, /budget exhausted/);
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
  assert.equal(runtime.trace.coverage().live_source_reconstructable, false);
  assert.equal(runtime.trace.replayObservations().outcome, 'done');
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

test('native model loop returns validation to caller by default and can nudge locally', async () => {
  const root = { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Write a number.' } };
  let defaultTurns = 0;
  const caller = new NativeToolAgent(() => { defaultTurns++; return { calls: [], text: '7', completion_tokens: 1 }; });
  const first = await new NativeRuntime({ agent: session => caller.run(session) }).runRoot(root);
  assert.equal(first.outcome.kind, 'quiesced');
  assert.match(first.outcome.detail, /validation failed: `return` has not been written yet/);
  assert.equal(defaultTurns, 1);
  let localTurns = 0;
  const local = new NativeToolAgent(() => ++localTurns === 1 ? { calls: [], text: '7', completion_tokens: 1 } :
    localTurns === 2 ? { calls: [['write', { path: 'return', type: 'Num', value: 7 }]], completion_tokens: 1 } :
      { calls: [], text: 'done', completion_tokens: 1 }, { validationFeedback: 'local' });
  const second = await new NativeRuntime({ agent: session => local.run(session) }).runRoot(root);
  assert.equal(second.outcome.kind, 'done'); assert.equal(second.value, 7); assert.equal(localTurns, 3);
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

test('native traces reconstruct in the shared offline reader', async () => {
  const runtime = new NativeRuntime();
  const result = await runtime.runRoot(crisp('Lambda<{}, Num>', 'return 14;'));
  assert.equal(result.outcome.kind, 'done');
  assert.equal(runtime.trace.reconstruct(), 14);
  const inspected = readTrace(runtime.trace.events);
  assert.equal(inspected.outcome, 'done');
  assert.equal(inspected.reconstructed, 14);
  assert.equal(runtime.trace.events[0].engine_contracts['typescript-host'].native_state_replayable, false);
});

test('native model seeds match Python derivation vectors', async () => {
  assert.equal(deriveSeed(43, '', 1, 'model-turn', 0), 1079124865);
  assert.equal(deriveSeed(43, 'return/0', 1, 'model-turn', 0), 365401298);
  const seen = [];
  const agent = new NativeToolAgent(request => {
    seen.push(request.seed);
    return seen.length === 1 ? { calls: [['write', { path: 'return', type: 'Bool', value: true }]], completion_tokens: 1 } :
      { calls: [], text: 'done', completion_tokens: 1 };
  });
  const runtime = new NativeRuntime({ agent: session => agent.run(session), seedPolicy: { mode: 'derived', root: 43 } });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{}, Bool>', instructions: 'Return true.' } });
  assert.equal(result.outcome.kind, 'done');
  assert.deepEqual(seen, [1079124865, deriveSeed(43, '', 1, 'model-turn', 1)]);
});

test('nested pending paths preserve types, frozen slots, and writable child results', async () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{}, { score: Num }>', instructions: 'Compute a score.',
    return: { score: { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Find it.' } } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  assert.equal(session.apply('read', { path: 'return/score/instructions' }).kind, 'ok');
  assert.equal(session.apply('write', { path: 'return/score/return', type: 'Num', value: 5 }).kind, 'ok');
  assert.equal(session.apply('write', { path: 'return/score/args/unknown', type: 'Num', value: 5 }).kind, 'rejected');
  assert.equal(session.apply('edit', { path: 'args/name', old: 'a', new: 'b' }).kind, 'rejected');
  assert.equal(lam.return.score.return, 5);
});

test('failed local writes roll back declaration; source copy, delete and effect scope are checked', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ item: Num }, Num>', instructions: 'Use item.', args: { item: 3 } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  assert.equal(session.apply('write', { path: 'let/a', type: 'Num', value: 'bad' }).kind, 'rejected');
  assert.equal(Object.hasOwn(lam.letTypes, 'a'), false);
  assert.equal(session.apply('write', { path: 'let/a', type: 'Num', source: 'args/item' }).kind, 'ok');
  assert.equal(lam.let.a, 3);
  assert.equal(session.apply('copy', { from: 'let/a', to: 'return' }).kind, 'ok');
  assert.equal(lam.return, 3);
  assert.equal(session.apply('delete', { path: 'let/a' }).kind, 'ok');
  assert.equal(Object.hasOwn(lam.let, 'a'), false);
  assert.equal(session.apply('write', { path: 'return', type: 'Num', value: { $lambda: {
    type: 'Lambda<{}, Num>', code: 'return 1;', effects: ['out.emit'] } } }).kind, 'rejected');
  assert.equal(lam.return, 3);
  assert.notEqual(lam.return, MISSING);
});

test('native tool schemas narrow to typed slots and expand as workspace values appear', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ item: Num }, { label: "yes" | "no", count: Num }>',
    instructions: 'Classify and count.', args: { item: 2 }, codebase: {
      count: { args: { item: 'Num' }, returns: 'Num', code: 'return args.item;' },
    } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const agent = new NativeToolAgent(() => ({ calls: [] }));
  const tools = agent.tools(session);
  const write = tools.find(item => item.function.name === 'write').function.parameters;
  assert.ok(write['x-natlang-alternatives'].some(alt => alt.path?.const === 'return/label'));
  assert.ok(write['x-natlang-alternatives'].some(alt => alt.path?.const === 'return/count'));
  const read = tools.find(item => item.function.name === 'read').function.parameters;
  assert.ok(read.properties.path.enum.includes('args/item'));
  assert.ok(!read.properties.path.enum.includes('return/label'));
  assert.equal(session.apply('write', { path: 'return/label', type: '"yes" | "no"', value: 'yes' }).kind, 'ok');
  const readAfter = agent.tools(session).find(item => item.function.name === 'read').function.parameters;
  assert.ok(readAfter.properties.path.enum.includes('return/label'));
});

test('native tool alternatives bind paths to their declared types and readable sources', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ number: Num, words: Text[] }, { count: Num, label: Text }>',
    instructions: 'Copy the number and label the words.', args: { number: 3, words: ['a', 'b'] },
    codebase: { identity: { args: { item: 'Num' }, returns: 'Num', code: 'return args.item;' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const definitions = new NativeToolAgent(() => ({ calls: [] })).tools(session);
  const parameters = name => definitions.find(item => item.function.name === name).function.parameters;
  const writes = parameters('write')['x-natlang-alternatives'];
  assert.ok(writes.some(alt => alt.path?.const === 'return/count' && alt.type?.const === 'Num' &&
    alt.value?.type === 'number'));
  assert.ok(writes.some(alt => alt.path?.const === 'return/count' && alt.source?.enum.includes('args/number')));
  assert.ok(!writes.some(alt => alt.path?.const === 'return/label' && alt.source?.enum.includes('args/number')));
  assert.ok(parameters('read')['x-natlang-alternatives'].some(alt => alt.path?.const === 'args/words' &&
    alt.start?.enum.includes(0) && alt.end?.enum.includes(1)));
  assert.ok(parameters('call')['x-natlang-alternatives'].some(alt => alt.function?.const === 'identity' &&
    alt.inputs?.properties?.item?.description === 'workspace path to Num'));
  assert.ok(parameters('call')['x-natlang-alternatives'].some(alt => alt.function?.const === 'identity' &&
    alt.values?.properties?.item?.type === 'number'));
});

test('native calls accept typed literal values and reject double binding', async () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ item: Num }, Num>',
    instructions: 'Return a selected value.', args: { item: 2 }, codebase: {
      add: { args: { left: 'Num', right: 'Num' }, returns: 'Num', code: 'return args.left + args.right;' },
    } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('call', { function: 'add', to: 'return',
    inputs: { left: 'args/item' }, values: { right: 5 } });
  assert.equal(result.kind, 'done');
  assert.equal(lam.return, 7);
  const overlap = await session.applyAsync('call', { function: 'add', to: 'let/nope',
    inputs: { left: 'args/item' }, values: { left: 3, right: 4 } });
  assert.equal(overlap.kind, 'rejected');
  assert.ok(overlap.codes.includes('bad-call'));
});

test('native opening state distinguishes absent inputs, empty text, partial records, and pending tasks', () => {
  const agent = new NativeToolAgent(() => ({ calls: [] }));
  const lam = buildPending({ $lambda: { type: 'Lambda<{ text?: Text }, { label: Text, count: Num }>',
    instructions: 'Summarize.', args: {} } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  assert.match(agent.opening(session), /args\/text \(Text, read-only\): not supplied/);
  assert.equal(agent.missing(session), '`return` has not been written yet. Write a { label: Text, count: Num } to `return`.');
  lam.args.text = '';
  assert.match(agent.opening(session), /args\/text \(Text, read-only\): ""/);
  session.apply('write', { path: 'return/label', type: 'Text', value: 'ok' });
  assert.match(agent.opening(session), /return \(\{ label: Text, count: Num \}\): partly written/);
  assert.match(agent.opening(session), /still missing: return\/count \(Num\)/);
  assert.match(agent.missing(session), /`return` is missing: return\/count \(Num\)\./);
});

test('complete native state snapshot resumes only unfinished Map slots', async () => {
  const program = { $map: { type: 'Map<Text, Text>', over: ['a', 'bad', 'c'],
    fn: { $lambda: { type: 'Lambda<{ item: Text }, Text>', instructions: 'Echo item.' } } } };
  const firstRuntime = new NativeRuntime({ agent: session => {
    if (session.lam.args.item === 'bad') return 'unreadable item';
    session.apply('write', { path: 'return', value: session.lam.args.item });
    session.finish();
  } });
  const first = await firstRuntime.runRoot(program);
  assert.equal(first.outcome.kind, 'quiesced');
  const snapshot = dumpState(first.value);
  assert.equal(snapshot.$map.slots.length, 3);
  const resumed = new NativeRuntime({ agent: session => {
    session.apply('write', { path: 'return', value: session.lam.args.item === 'bad' ? 'b' : session.lam.args.item });
    session.finish();
  } });
  const result = await resumed.runRoot(snapshot);
  assert.equal(result.outcome.kind, 'done');
  assert.deepEqual(result.value, ['a', 'b', 'c']);
  assert.equal(resumed.episodesStarted, 1);
});

test('crisp code may replace itself with an unreduced typed task', async () => {
  const result = await new NativeRuntime().runRoot(crisp('Lambda<{}, Num>',
    'return lambda({ type: "Lambda<{}, Num>", instructions: "Compute seven." });'));
  assert.equal(result.outcome.kind, 'replaced');
  assert.equal(result.value.nodeKind, 'lambda');
  assert.equal(result.value.status, 'unreduced');
});

test('model write cannot invent an unchecked anonymous task', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return one.' } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = session.apply('write', { path: 'return', type: 'Num', value: {
    $lambda: { type: 'Lambda<{}, Num>', instructions: 'Guess one.' },
  } });
  assert.deepEqual(result.codes, ['anonymous-lambda']);
  assert.equal(lam.return, MISSING);
});

test('review withdrawal prevents a proposed write and permits one corrected retry', async () => {
  let proposed = 0, reviewed = 0;
  const agent = new NativeToolAgent(() => {
    proposed++;
    if (proposed === 1) return { calls: [['write', { path: 'return', type: 'Num', value: 99 }]],
      value_confidence: [0.1], completion_tokens: 1 };
    if (proposed === 2) return { calls: [['write', { path: 'return', type: 'Num', value: 7 }]],
      value_confidence: [0.9], completion_tokens: 1 };
    return { calls: [], text: 'done', completion_tokens: 1 };
  }, { review: { threshold: 0.5, withdrawalPolicy: 'retry', driver: () => {
    reviewed++;
    return { calls: [['review_write', { reason: 'The proposed value ignores the instruction.', decision: 'withdraw' }]],
      completion_tokens: 1 };
  } } });
  const runtime = new NativeRuntime({ agent: session => agent.run(session) });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return seven.' } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 7);
  assert.equal(reviewed, 1);
  assert.ok(runtime.trace.events.some(event => event.kind === 'proposal' && event.phase === 'withdrawn'));
  assert.equal(runtime.trace.events.filter(event => event.kind === 'action' && event.name === 'write').length, 1);
});

test('native scenario admission checks exact calls and ordered effects without replay', async () => {
  const runtime = new NativeRuntime({ agent: async session => {
    await session.applyAsync('call', { function: 'double', to: 'return', inputs: { item: 'args/item' } });
    session.finish();
  } });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{ item: Num }, Num>',
    instructions: 'Call double.', args: { item: 2 }, codebase: {
      double: { args: { item: 'Num' }, returns: 'Num', code: 'return args.item * 2;' },
    } } });
  assert.equal(result.value, 4);
  const contract = { outcome: 'done', value: 4, constrainedCalls: [
    { function: 'double', to: 'return', inputs: { item: 'args/item' } }], requiredActions: [
    { name: 'call', arguments: { function: 'double', to: 'return' } }] };
  assert.equal(admitNativeTrace(runtime.trace, contract).admitted, true);
  assert.throws(() => admitNativeTrace(runtime.trace, { ...contract, constrainedCalls: [
    { function: 'double', to: 'return/size', inputs: { item: 'args/item' } }] }), /destination or inputs/);
  const effects = new NativeRuntime();
  const effected = await effects.runRoot({ $lambda: { type: 'Lambda<{}, Num>', code: 'fx.out.emit({ x: 1 }); return 3;',
    effects: ['out.emit'] } });
  assert.equal(effected.value, 3);
  assert.equal(admitNativeTrace(effects.trace, { outcome: 'done', value: 3,
    effects: [['out.emit', [{ x: 1 }]]] }).admitted, true);
});

test('native run_code can await a declared asynchronous capability', async () => {
  const runtime = new NativeRuntime({ capabilities: { 'counter.add': async ([n]) => n + 4 },
    agent: async session => {
      const result = await session.applyAsync('run_code', { engine: 'typescript-host', code: 'await fx.counter.add(3)' });
      assert.equal(result.kind, 'ok'); assert.equal(result.value, 7);
      session.apply('write', { path: 'return', value: result.value });
      session.finish();
    } });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'Compute seven.',
    effects: ['counter.add'] } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 7);
});

test('bounded independent Map work overlaps while preserving slot order and shared budget', async () => {
  let active = 0, peak = 0;
  const runtime = new NativeRuntime({ mapWorkers: 2, parallelModelSafe: true, maxEpisodes: 4,
    agent: async session => {
      active++; peak = Math.max(peak, active);
      await new Promise(resolve => setTimeout(resolve, 8));
      session.apply('write', { path: 'return', value: session.lam.args.item * 2 });
      session.finish(); active--;
    } });
  const result = await runtime.runRoot({ $map: { type: 'Map<Num, Num>', over: [1, 2, 3, 4],
    fn: { $lambda: { type: 'Lambda<{ item: Num }, Num>', instructions: 'Double item.' } } } });
  assert.equal(result.outcome.kind, 'done');
  assert.deepEqual(result.value, [2, 4, 6, 8]);
  assert.equal(runtime.episodesStarted, 4);
  assert.equal(peak, 2);
  assert.equal(runtime.trace.events.filter(event => event.kind === 'map_slot').length, 4);
});

test('native en-passant done marks validate before work and follow successful writes', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'First step.\nSecond step.',
    codebase: { one: { returns: 'Num', code: 'return 1;' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  assert.deepEqual(session.apply('write', { path: 'return', value: 7, done: [1, 9] }).codes, ['bad-range']);
  assert.equal(lam.return, MISSING);
  assert.equal(session.apply('write', { path: 'return', value: 'bad', done: 1 }).kind, 'rejected');
  assert.deepEqual(lam.marks, {});
  assert.equal(session.apply('write', { path: 'return', value: 7, done: 1 }).kind, 'ok');
  assert.equal(lam.marks[1], 'done');
  const tools = new NativeToolAgent(() => ({ calls: [] })).tools(session);
  const mark = tools.find(item => item.function.name === 'mark_done');
  assert.deepEqual(mark.function.parameters['x-natlang-alternatives'][0].start.enum, [2]);
  assert.ok(tools.find(item => item.function.name === 'write').function.parameters.properties.done);
});

test('native retry reopens a completed natural-language child with its draft value', async () => {
  const runtime = new NativeRuntime({ agent: async session => {
    if (session.lam.functionName === 'draft') {
      session.apply('write', { path: 'return', value: session.lam.body.includes('short') ? 'short' : 'too long' });
      session.finish(); return;
    }
    assert.equal((await session.applyAsync('call', { function: 'draft', to: 'let/headline' })).kind, 'done');
    assert.equal(session.apply('retry', { path: 'let/headline', feedback: 'Write a short headline.' }).kind, 'ok');
    assert.equal((await session.applyAsync('run', { paths: 'let/headline' })).kind, 'done');
    session.apply('write', { path: 'return', source: 'let/headline' });
    session.finish();
  } });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{}, Text>', instructions: 'Draft then refine.',
    codebase: { draft: { returns: 'Text', instructions: 'Write a headline.' } } } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 'short');
});

test('failed checked call does not leave an empty local declaration', async () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'Do the work.',
    codebase: { emit: { returns: 'Num', effects: ['out.emit'], code: 'fx.out.emit(1); return 1;' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('call', { function: 'emit', to: 'let/draft' });
  assert.deepEqual(result.codes, ['effect-wider-than-parent']);
  assert.equal(Object.hasOwn(lam.letTypes, 'draft'), false);
});
