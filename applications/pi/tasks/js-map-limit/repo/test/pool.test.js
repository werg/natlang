import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mapLimit } from '../src/pool.js';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

test('keeps input order', async () => {
  const out = await mapLimit([30, 10, 20], 3, async (ms) => { await sleep(ms); return ms * 2; });
  assert.deepEqual(out, [60, 20, 40]);
});
test('empty input', async () => {
  assert.deepEqual(await mapLimit([], 2, async (x) => x), []);
});
