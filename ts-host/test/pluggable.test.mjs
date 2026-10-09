import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, pluggable, pluggableMode } from '../dist/runtime/node.js';

const parts = (calls = []) => ({
  crisp: n => { calls.push('crisp'); return { total: n * 2 }; },
  nl: async n => { calls.push('nl'); return { total: n * 2 + (n > 5 ? 1 : 0) }; },
});

test('crisp and nl modes run one implementation', async () => {
  const calls = [];
  assert.deepEqual(await pluggable(parts(calls), 'crisp')(2), { total: 4 });
  assert.deepEqual(await pluggable(parts(calls), 'nl')(2), { total: 4 });
  assert.deepEqual(calls, ['crisp', 'nl']);
  assert.throws(() => pluggable(parts(), 'both'), /"crisp", "nl" or "shadow"/);
});

test('shadow mode runs both, serves one and records agreement and disagreement as pluggable_shadow events', async () => {
  const traces = [], calls = [];
  const runtime = createNatlangRuntime({ trace: trace => traces.push(trace) });
  const part = pluggable(parts(calls), 'shadow', { name: 'doubling' });
  const served = await runtime.run(async () => [await part(2), await part(9)]);
  assert.deepEqual(served, [{ total: 4 }, { total: 19 }], 'the nl side is served by default');
  assert.deepEqual(calls.sort(), ['crisp', 'crisp', 'nl', 'nl']);
  const events = traces.flatMap(trace => trace.events).filter(event => event.kind === 'pluggable_shadow');
  assert.deepEqual(events.map(event => event.agree), [true, false]);
  assert.equal(events[1].name, 'doubling'); assert.equal(events[1].served, 'nl');
  assert.equal(events[1].crisp, JSON.stringify({ total: 18 }));
  const crispServed = await runtime.run(() => pluggable(parts(), 'shadow', { serve: 'crisp' })(9));
  assert.deepEqual(crispServed, { total: 18 });
});

test('shadow mode records a failure of the side that is not served and fails when the served side fails', async () => {
  const traces = [];
  const runtime = createNatlangRuntime({ trace: trace => traces.push(trace) });
  const part = pluggable({ crisp: () => { throw new Error('crisp broke'); }, nl: () => 'ok' }, 'shadow');
  assert.equal(await runtime.run(() => part()), 'ok');
  const [event] = traces.flatMap(trace => trace.events).filter(item => item.kind === 'pluggable_shadow');
  assert.equal(event.agree, false); assert.match(event.crisp, /crisp broke/);
  await assert.rejects(() => runtime.run(() => pluggable({ crisp: () => 1, nl: () => { throw new Error('nl broke'); } }, 'shadow')()), /nl broke/);
  assert.equal(await pluggable({ crisp: () => 1, nl: () => 1 }, 'shadow')(), 1, 'outside a task it still compares and serves');
});

test('the older spellings of nl are accepted, an absent setting takes the default, and anything else is refused', async () => {
  const calls = [];
  for (const old of ['natlang', 'natural-language']) assert.deepEqual(await pluggable(parts(calls), old)(2), { total: 4 });
  assert.deepEqual(calls, ['nl', 'nl']);
  assert.deepEqual(await pluggable(parts(calls), undefined)(2), { total: 4 });
  assert.deepEqual(await pluggable(parts(calls), undefined, { default: 'crisp' })(2), { total: 4 });
  assert.deepEqual(calls, ['nl', 'nl', 'nl', 'crisp']);
  assert.deepEqual(['crisp', 'nl', 'shadow', 'natlang', 'natural-language', undefined].map(mode => pluggableMode(mode)),
    ['crisp', 'nl', 'shadow', 'nl', 'nl', 'nl']);
  assert.equal(pluggableMode(undefined, 'crisp'), 'crisp');
  assert.throws(() => pluggableMode('both'), /deprecated/);
});
