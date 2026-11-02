import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ReplayServices } from '../dist/calls/replay.js';

test('strict replay preserves ordered source observations and exact argument types', () => {
  const effects = [
    { service: 'browser', method: 'search', args: [{ query: 'x', page: 1 }], result: 'first', async: false, complete: true },
    { service: 'browser', method: 'search', args: [{ query: 'x', page: 1 }], result: 'changed', async: false, complete: true },
    { service: 'browser', method: 'open', args: [{ id: 3 }], result: 'page', async: false, complete: true },
  ];
  const replay = new ReplayServices(effects, [], { strictSequence: true });
  assert.equal(replay.services.browser.search({ page: 1, query: 'x' }), 'first');
  assert.throws(() => replay.services.browser.open({ id: 3 }), /no recorded result/);
  assert.equal(replay.services.browser.search({ query: 'x', page: 1 }), 'changed');
  assert.throws(() => replay.services.browser.open({ id: '3' }), /no recorded result/);
  assert.equal(replay.services.browser.open({ id: 3 }), 'page');
  assert.equal(replay.unused().length, 0);
});

test('historical replay still permits its recorded loose scalar equivalence', () => {
  const replay = new ReplayServices([{ service: 'orders', method: 'refund', args: [999], result: 'ok', async: false, complete: true }]);
  assert.equal(replay.services.orders.refund('999'), 'ok');
  assert.deepEqual(replay.divergences, []);
});
