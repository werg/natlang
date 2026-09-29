import { test } from 'node:test';
import assert from 'node:assert/strict';
import { rateLimited, transportFailure, retryAfterMs, retryWaitMs } from '../dist/teacher/retry.js';

test('transient provider failures are retried, model and authentication errors are not', () => {
  for (const error of [new Error('HTTP 503'), { status: 500 }, new Error('fetch failed'), new Error('usage_limit_reached')])
    assert.equal(transportFailure(error), true);
  for (const error of [new Error('incorrect answer'), { status: 401 }, new Error('invalid API key')])
    assert.equal(transportFailure(error), false);
  assert.equal(rateLimited({ status: 429 }), true);
});
test('delays double, jitter is capped, and zero policy delay remains available', () => {
  const error = new Error('rate limit');
  assert.deepEqual([0,1,2,3].map(n => retryWaitMs(error, n, 15000, () => 0.5)), [45000,90000,120000,120000]);
  assert.equal(retryWaitMs(error, 10, 15000, () => 1), 120000);
  assert.equal(retryWaitMs(new Error('fetch failed'), 10, 15000, () => 1), 30000);
  assert.equal(retryWaitMs(error, 10, 0), 0);
});
test('supplied retry delays are minimums, including dates and flattened SDK errors', () => {
  const now = Date.parse('2026-09-29T08:00:00Z');
  assert.equal(retryAfterMs({ headers: { 'Retry-After': '180' } }, now), 180000);
  assert.equal(retryAfterMs({ response: { headers: new Headers({ 'retry-after': 'Tue, 29 Sep 2026 08:02:00 GMT' }) } }, now), 120000);
  assert.equal(retryAfterMs(new Error('{"retry_after_ms":1234}')), 1234);
  assert.equal(retryAfterMs(new Error('{"resets_in_seconds":3600}')), 3600000);
  assert.equal(retryAfterMs(new Error('Please try again in 12 seconds')), 12000);
  assert.equal(retryAfterMs({ headers: { 'retry-after': 'nonsense' } }), 0);
  assert.equal(retryWaitMs({ status: 429, retry_after: 180 }, 0, 15000, () => 0), 180000);
});
