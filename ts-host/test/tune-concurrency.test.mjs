import assert from 'node:assert/strict';
import { test } from 'node:test';
import { DEFAULT_TUNE_CONCURRENCY, mapBounded, tuneConcurrency } from '../dist/tune/execution.js';

test('tune roots run a bounded number at once and keep their input order', async () => {
  let running = 0, peak = 0;
  const results = await mapBounded([5, 1, 4, 2, 3], 2, async (delay, index) => {
    running++; peak = Math.max(peak, running);
    await new Promise(resolve => setTimeout(resolve, delay * 3));
    running--;
    return `${index}:${delay}`;
  });
  assert.deepEqual(results, ['0:5', '1:1', '2:4', '3:2', '4:3']);
  assert.equal(peak, 2);
  assert.deepEqual(await mapBounded([], 4, async () => 1), []);
});

test('a failing root stops new work and is rethrown after the started roots settle', async () => {
  const started = [];
  await assert.rejects(mapBounded([0, 1, 2, 3, 4, 5], 2, async index => {
    started.push(index);
    await new Promise(resolve => setTimeout(resolve, 2));
    if (index === 1) throw new Error('root 1 failed');
    return index;
  }), /root 1 failed/);
  assert.ok(started.length < 6);
});

test('tune.concurrency defaults to batched roots and rejects nonsense', () => {
  assert.equal(tuneConcurrency(undefined), DEFAULT_TUNE_CONCURRENCY);
  assert.equal(tuneConcurrency(1), 1);
  assert.throws(() => tuneConcurrency(0), /positive integer/);
  assert.throws(() => tuneConcurrency(2.5), /positive integer/);
});
