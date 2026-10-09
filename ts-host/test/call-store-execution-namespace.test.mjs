import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, defineNatlang } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

test('call-store task namespaces use UUID-backed execution identity', async () => {
  const root = defineNatlang('---\nargs: {}\nreturns: number\n---\nReturn one.\n');
  const ids = [];
  for (let index = 0; index < 2; index++) {
    const traces = [];
    const model = scriptedModel(() => 'return 1;');
    assert.equal(await createNatlangRuntime({ model: model.driver, trace: trace => traces.push(trace) })
      .run(() => root()), 1);
    assert.equal(traces.length, 1);
    assert.equal(traces[0].parentCallId, null);
    assert.match(traces[0].taskId, /^task-\d+-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
    assert.match(traces[0].callId, new RegExp(`^${traces[0].taskId}/1$`));
    ids.push(traces[0].callId);
  }
  assert.notEqual(ids[0], ids[1], 'separate executions cannot replace or parent-link each other in CallStore');
});
