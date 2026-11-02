import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ReplayServices } from '../dist/calls/replay.js';
import { hexDigest } from '../dist/native/hash.js';

const canonical = value => JSON.stringify(value, (_key,item) => item && typeof item === 'object' && !Array.isArray(item) ? Object.fromEntries(Object.entries(item).sort(([a],[b]) => a.localeCompare(b))) : item);
function contract(effects, methods) { const body={schema:'natlang.independent-service-replay/1',semantics:'stateless-independent',methods,effects_sha256:hexDigest(canonical(effects))};return {...body,sha256:hexDigest(canonical(body))}; }

test('strict replay preserves ordered source observations and exact argument types', () => {
  const effects = [
    { service: 'browser', method: 'search', args: [{ query: 'x', page: 1 }], result: 'first', async: false, complete: true },
    { service: 'browser', method: 'search', args: [{ query: 'x', page: 1 }], result: 'changed', async: false, complete: true },
    { service: 'browser', method: 'open', args: [{ id: 3 }], result: 'page', async: false, complete: true },
  ];
  const replay = new ReplayServices(effects, [], { mode: 'ordered-exact' });
  assert.equal(replay.services.browser.search({ page: 1, query: 'x' }), 'first');
  assert.throws(() => replay.services.browser.open({ id: 3 }), /no recorded result/);
  assert.equal(replay.services.browser.search({ query: 'x', page: 1 }), 'changed');
  assert.throws(() => replay.services.browser.open({ id: '3' }), /no recorded result/);
  assert.equal(replay.services.browser.open({ id: 3 }), 'page');
  assert.equal(replay.unused().length, 0);
});

test('independent replay requires an explicit contract and retains exact scalar types', () => {
  const effects = [{ service: 'orders', method: 'refund', args: [999], result: 'ok', async: false, complete: true }];
  assert.throws(() => new ReplayServices(effects, [], { mode:'independent-exact' }), /source-bound stateless service contract/);
  const valid = contract(effects,['orders.refund']);
  assert.throws(() => new ReplayServices(effects, [], {mode:'independent-exact', contract:{...valid,effects_sha256:'f'.repeat(64)}}), /source-bound/);
  assert.throws(() => new ReplayServices(effects, [], {mode:'independent-exact', contract:contract(effects,['orders.lookup'])}), /source-bound/);
  const replay = new ReplayServices(effects, [], { mode:'independent-exact', contract:valid });
  assert.throws(() => replay.services.orders.refund('999'), /no recorded result/);
  assert.equal(replay.services.orders.refund(999), 'ok');
});

test('public replay propagates strict sequencing to runtime services', async () => {
  const { replay } = await import('../dist/calls/replay.js');
  const { currentFrame } = await import('../dist/runtime/context.js');
  const { createNatlangRuntime } = await import('../dist/runtime/runtime.js');
  const { nodePlatform } = await import('../dist/runtime/node.js');
  const runtime = createNatlangRuntime({ platform:nodePlatform(), model: async () => { throw new Error('unexpected model'); }, calls: false });
  const effects = [
    { service:'browser', method:'open', args:[{id:1}], result:'first', async:false, complete:true },
    { service:'browser', method:'open', args:[{id:2}], result:'second', async:false, complete:true },
  ];
  const wrong = await replay(runtime, () => currentFrame().services.browser.open({id:2}), {}, effects, {replayMode:'ordered-exact'});
  assert.match(wrong.error, /no recorded result/);
  assert.equal(wrong.divergences.length, 1);
  const defaultResult = await replay(runtime, () => currentFrame().services.browser.open({id:2}), {}, effects);
  assert.match(defaultResult.error, /no recorded result/);
  const independent = await replay(runtime, () => currentFrame().services.browser.open({id:2}), {}, effects,
    {replayMode:'independent-exact',contract:contract(effects,['browser.open'])});
  assert.equal(independent.value, 'second');
});
