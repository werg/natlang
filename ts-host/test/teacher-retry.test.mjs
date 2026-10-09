import { test } from 'node:test';
import assert from 'node:assert/strict';
import { rateLimited, transportFailure, retryAfterMs, retryWaitMs, providerFinishReason, providerRequestRetryable } from '../dist/teacher/retry.js';

test('transient provider failures are retried, model and authentication errors are not', () => {
  for (const error of [new Error('HTTP 503'), { status: 500 }, new Error('fetch failed'), new Error('usage_limit_reached'), new Error('Connection error.'), new Error('Connection error')])
    assert.equal(transportFailure(error), true);
  for (const error of [new Error('incorrect answer'), { status: 401 }, new Error('invalid API key'), new Error('invalid connection error handling in model code')])
    assert.equal(transportFailure(error), false);
  assert.equal(rateLimited({ status: 429 }), true);
});
test('same-request retries respect explicit provider non-retryability and non-bridge tool failures', () => {
  assert.equal(providerRequestRetryable(Object.assign(new Error('model HTTP 502'), { status: 502, providerCode: 'NON_BRIDGE_TOOL_USE' })), false);
  assert.equal(providerRequestRetryable(Object.assign(new Error('model HTTP 503'), { status: 503, providerRetryable: false })), false);
  assert.equal(providerRequestRetryable(Object.assign(new Error('model HTTP 503'), { status: 503 })), true);
  for (const error of [
    Object.assign(new Error('context size has been exceeded'), { status: 503 }),
    Object.assign(new Error('model HTTP 503'), { status: 503, providerCode: 'context_length_exceeded' }),
    Object.assign(new Error('model HTTP 503'), { status: 503, providerCode: 'schema_validation_error' }),
    Object.assign(new Error('model HTTP 503'), { status: 503, providerCode: 'authentication_error' }),
    Object.assign(new Error('model HTTP 503'), { status: 503, providerCode: 'content_filter' }),
    Object.assign(new Error('teacher provider turn exceeded deadline'), { status: 503, code: 'NATLANG_PROVIDER_REQUEST_TIMEOUT' }),
  ]) assert.equal(providerRequestRetryable(error), false, error.providerCode ?? error.message);
  assert.equal(transportFailure(Object.assign(new Error('model HTTP 503'), { status: 503,
    code: 'NATLANG_PROVIDER_REQUEST_RETRIES_EXHAUSTED' })), false);
});
test('only explicit provider error/network finish reasons join bounded transport retries', () => {
  assert.equal(providerFinishReason(new Error('Provider finish_reason: error')), 'error');
  assert.equal(providerFinishReason(new Error('Provider finish_reason: network_error')), 'network_error');
  assert.equal(transportFailure(new Error('Provider finish_reason: error')), true);
  assert.equal(transportFailure(new Error('Provider finish_reason: network_error')), true);
  for (const reason of ['content_filter', 'length', 'tool_calls', 'stop', 'invalid_request', 'authentication_error', 'error: server rejected payload']) {
    const error = new Error(`Provider finish_reason: ${reason}`);
    assert.equal(providerFinishReason(error), undefined, reason);
    assert.equal(transportFailure(error), false, reason);
  }
  assert.equal(transportFailure(new Error('model returned malformed tool arguments after 1 attempt: invalid JSON')), false);
  assert.equal(transportFailure({ status: 401, message: 'Provider finish_reason: error' }), false);
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
test('loopback OpenCode JSON provider errors preserve status and retry-after for collection', () => {
  const body = JSON.stringify({ error: { message: 'Rate limit exceeded', provider_status_code: 429,
    provider_retryable: true, retry_after_ms: 7000 } });
  const error = Object.assign(new Error(`model HTTP 429: ${body}`), { status: 429 });
  assert.equal(rateLimited(error), true);
  assert.equal(transportFailure(error), true);
  assert.equal(retryAfterMs(error), 7000);
});
