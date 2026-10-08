import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createOpenCodeStructuredTurnBackend } from '../scripts/opencode-structured-turn.mjs';
import { createOpenCodeLoopbackChatAdapter } from '../scripts/opencode-loopback-chat-adapter.mjs';

test('matching OpenCode session.error surfaces provider 429 before the prompt deadline', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'opencode-event-error-'));
  const calls = [];
  let subscriptionReady = false;
  let promptStartedResolve;
  const promptStarted = new Promise(resolve => { promptStartedResolve = resolve; });
  const providerError = {
    type: 'session.error',
    properties: { sessionID: 'session-target', error: { name: 'APIError', data: {
      message: 'Rate limit exceeded. Please try again later.', statusCode: 429,
      isRetryable: true, responseBody: '{"error":"rate limited"}',
      responseHeaders: { authorization: 'must-not-be-captured' }
    } } }
  };
  const client = {
    tool: { ids: async () => ({ data: ['bash', 'invalid'] }) },
    event: { subscribe: async ({ signal }) => {
      await new Promise(resolve => setTimeout(resolve, 30));
      subscriptionReady = true;
      calls.push('subscribed');
      return { stream: (async function* () {
      yield { type: 'session.error', properties: { sessionID: 'unrelated', error: providerError.properties.error } };
      await promptStarted;
      yield providerError;
      await new Promise(resolve => signal.addEventListener('abort', resolve, { once: true }));
      })() };
    } },
    session: {
      create: async () => ({ data: { id: 'session-target' } }),
      prompt: async (_args, { signal }) => new Promise((_, reject) => {
        assert.equal(subscriptionReady, true, 'event stream must be established before prompt starts');
        calls.push('prompt');
        promptStartedResolve();
        signal.addEventListener('abort', () => reject(signal.reason), { once: true });
      }),
      abort: async () => { calls.push('abort'); return { data: true }; },
      messages: async () => ({ data: [] }),
      delete: async () => { calls.push('delete'); return { data: true }; }
    }
  };
  const backend = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode',
    modelID: 'space-bunny-free', directory });

  try {
    await assert.rejects(backend({ messages: [], tools: [], tool_choice: 'auto' }), error => {
      assert.equal(error.code, 'OPENCODE_PROVIDER_ERROR');
      assert.equal(error.providerStatusCode, 429);
      assert.equal(error.providerRetryable, true);
      assert.equal(error.transportDiagnostic.classification, 'provider_request_failure');
      assert.equal(error.transportDiagnostic.upstream_error.status_code, 429);
      return true;
    });
    assert.deepEqual(calls, ['subscribed', 'prompt', 'abort', 'delete']);
    const sidecarDir = join(directory, '.natlang-transport-failures');
    const { readdir } = await import('node:fs/promises');
    const [sidecar] = await readdir(sidecarDir);
    const receipt = JSON.parse(await readFile(join(sidecarDir, sidecar), 'utf8'));
    assert.equal(receipt.upstream_error.response_body.redacted_preview, '{"error":"rate limited"}');
    assert.equal(JSON.stringify(receipt).includes('must-not-be-captured'), false);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('OpenCode session.error from another session does not replace the prompt result', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'opencode-event-scope-'));
  const final = { info: { id: 'assistant-final', role: 'assistant' }, parts: [
    { type: 'text', text: '{"content":"ok","toolCalls":[]}' }
  ] };
  const client = {
    tool: { ids: async () => ({ data: ['invalid'] }) },
    event: { subscribe: async ({ signal }) => ({ stream: (async function* () {
      yield { type: 'session.error', properties: { sessionID: 'other', error: {
        name: 'APIError', data: { message: 'Rate limit exceeded', statusCode: 429, isRetryable: true }
      } } };
      await new Promise(resolve => signal.addEventListener('abort', resolve, { once: true }));
    })() }) },
    session: {
      create: async () => ({ data: { id: 'session-target' } }),
      prompt: async () => ({ data: final }),
      messages: async () => ({ data: [final] }),
      delete: async () => ({ data: true })
    }
  };
  const backend = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode',
    modelID: 'space-bunny-free', directory });
  try {
    const result = await backend({ messages: [], tools: [], tool_choice: 'auto' });
    assert.equal(result.text, 'ok');
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('loopback preserves provider 429 and Retry-After for the collector retry policy', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'opencode-http-retry-'));
  const client = {
    tool: { ids: async () => ({ data: ['invalid'] }) },
    event: { subscribe: async ({ signal }) => ({ stream: (async function* () {
      yield { type: 'session.error', properties: { sessionID: 'session-http', error: {
        name: 'APIError', data: { message: 'Rate limit exceeded', statusCode: 429, isRetryable: true,
          responseHeaders: { 'retry-after': '7' }, responseBody: '{"error":"rate limited"}' }
      } } };
      await new Promise(resolve => signal.addEventListener('abort', resolve, { once: true }));
    })() }) },
    session: {
      create: async () => ({ data: { id: 'session-http' } }),
      prompt: async (_args, { signal }) => new Promise((_, reject) => {
        signal.addEventListener('abort', () => reject(signal.reason), { once: true });
      }),
      abort: async () => ({ data: true }),
      messages: async () => ({ data: [] }),
      delete: async () => ({ data: true })
    }
  };
  const adapter = await createOpenCodeLoopbackChatAdapter({ client, providerID: 'opencode',
    modelID: 'space-bunny-free', directory });
  try {
    const response = await fetch(`${adapter.url}/v1/chat/completions`, { method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'opencode/space-bunny-free', messages: [], tools: [], stream: true }) });
    const body = await response.json();
    assert.equal(response.status, 429);
    assert.equal(body.error.provider_status_code, 429);
    assert.equal(body.error.provider_retryable, true);
    assert.equal(body.error.retry_after_ms, 7000);
    assert.match(JSON.stringify(body), /"retry_after_ms":7000/);
  } finally {
    await adapter.close();
    await rm(directory, { recursive: true, force: true });
  }
});
