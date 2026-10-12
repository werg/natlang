import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const { mapLimit } = await import(pathToFileURL(`${process.cwd()}/src/pool.js`).href);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
test('order and limit', async () => {
  let running = 0, peak = 0;
  const delays = [25, 5, 15, 1, 30, 2, 10, 4];
  const out = await mapLimit(delays, 3, async (ms, i) => { running++; peak = Math.max(peak, running); await sleep(ms); running--; return `${i}:${ms}`; });
  assert.deepEqual(out, delays.map((ms, i) => `${i}:${ms}`));
  assert.equal(peak, 3);
});
test('limit above length', async () => {
  assert.deepEqual(await mapLimit([3, 1, 2], 10, async (ms) => { await sleep(ms); return ms; }), [3, 1, 2]);
});
test('rejection propagates', async () => {
  await assert.rejects(mapLimit([1, 2, 3], 2, async (x) => { if (x === 2) throw new Error('boom'); return x; }), /boom/);
});
