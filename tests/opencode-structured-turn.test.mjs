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
    event: { subscribe: async ({ directory: eventDirectory }, { signal, fetch, sseMaxRetryAttempts }) => {
      assert.equal(eventDirectory, directory);
      assert.equal(sseMaxRetryAttempts, 1);
      calls.push('subscribed');
      return { stream: (async function* () {
      const response = await fetch(new Request(`http://opencode.test/event?directory=${encodeURIComponent(eventDirectory)}`));
      assert.equal(response.status, 200);
      subscriptionReady = true;
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
    modelID: 'space-bunny-free', directory,
    eventFetchImpl: async () => new Response('', { status: 200, headers: { 'content-type': 'text/event-stream' } }) });

  try {
    await assert.rejects(backend({ messages: [], tools: [], tool_choice: 'auto' }), error => {
      assert.equal(error.code, 'OPENCODE_PROVIDER_ERROR');
      assert.equal(error.providerStatusCode, 429);
      assert.equal(error.providerRetryable, true);
      assert.equal(error.transportDiagnostic.classification, 'provider_request_failure');
      assert.equal(error.transportDiagnostic.upstream_error.status_code, 429);
      assert.equal(error.transportDiagnostic.upstream_error.event_type, 'session.error');
      assert.equal(error.transportDiagnostic.upstream_error.event_session_id, 'session-target');
      assert.equal(error.transportDiagnostic.upstream_error.event_stream.status, 200);
      assert.equal(error.transportDiagnostic.upstream_error.event_stream.directory, directory);
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

test('request timeout diagnostics are classified as infrastructure timeouts', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'opencode-request-timeout-'));
  const timeout = Object.assign(new Error('OpenCode turn timed out'), { code: 'REQUEST_TIMEOUT' });
  const client = {
    tool: { ids: async () => ({ data: ['invalid'] }) },
    session: {
      create: async () => ({ data: { id: 'session-timeout' } }),
      prompt: async () => { throw timeout; },
      messages: async () => ({ data: [] }),
      delete: async () => ({ data: true })
    }
  };
  const backend = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode',
    modelID: 'space-bunny-free', directory });
  try {
    await assert.rejects(backend({ messages: [], tools: [], tool_choice: 'auto' }), error => {
      assert.equal(error.code, 'REQUEST_TIMEOUT');
      assert.equal(error.transportDiagnostic.classification, 'request_timeout');
      assert.equal(error.transportDiagnostic.failure_phase, 'session_prompt');
      assert.equal(error.transportDiagnostic.event_stream.mode, 'SDK event subscription unavailable');
      assert.equal(error.transportDiagnostic.event_stream.ready, false);
      return true;
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('failure evidence retains matching-session retry status without inferring HTTP metadata', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'opencode-session-retry-status-'));
  const timeout = Object.assign(new Error('OpenCode turn timed out'), { code: 'REQUEST_TIMEOUT' });
  const client = {
    tool: { ids: async () => ({ data: ['invalid'] }) },
    event: { subscribe: async ({ directory: eventDirectory }, { signal, fetch }) => ({ stream: (async function* () {
      await fetch(new Request(`http://opencode.test/event?directory=${encodeURIComponent(eventDirectory)}`));
      yield { type: 'session.status', properties: { sessionID: 'session-retry', status: {
        type: 'retry', attempt: 3, message: 'Provider rate limit; retrying shortly',
        action: { provider: 'opencode', label: 'Retry', message: 'Please retry after delay' }
      } } };
      await new Promise(resolve => signal.addEventListener('abort', resolve, { once: true }));
    })() }) },
    session: {
      create: async () => ({ data: { id: 'session-retry' } }),
      prompt: async () => { await new Promise(resolve => setTimeout(resolve, 10)); throw timeout; },
      messages: async () => ({ data: [] }),
      delete: async () => ({ data: true })
    }
  };
  const backend = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode',
    modelID: 'space-bunny-free', directory,
    eventFetchImpl: async () => new Response('', { status: 200, headers: { 'content-type': 'text/event-stream' } }) });
  try {
    await assert.rejects(backend({ messages: [], tools: [], tool_choice: 'auto' }), error => {
      const eventStream = error.transportDiagnostic.event_stream;
      assert.equal(eventStream.latest_session_status.type, 'retry');
      assert.equal(eventStream.latest_session_status.attempt, 3);
      assert.equal(eventStream.latest_session_status.action.provider, 'opencode');
      assert.equal(error.providerStatusCode, undefined, 'retry prose is not an HTTP status');
      return true;
    });
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
    event: { subscribe: async ({ directory: eventDirectory }, { signal, fetch }) => ({ stream: (async function* () {
      await fetch(new Request(`http://opencode.test/event?directory=${encodeURIComponent(eventDirectory)}`));
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
    modelID: 'space-bunny-free', directory,
    eventFetchImpl: async () => new Response('', { status: 200, headers: { 'content-type': 'text/event-stream' } }) });
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
    event: { subscribe: async ({ directory: eventDirectory }, { signal, fetch }) => ({ stream: (async function* () {
      await fetch(new Request(`http://opencode.test/event?directory=${encodeURIComponent(eventDirectory)}`));
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
    modelID: 'space-bunny-free', directory,
    eventFetchImpl: async () => new Response('', { status: 200, headers: { 'content-type': 'text/event-stream' } }) });
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
