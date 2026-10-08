import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createOpenCodeStructuredTurnBackend } from '../scripts/opencode-structured-turn.mjs';

test('matching OpenCode session.error surfaces provider 429 before the prompt deadline', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'opencode-event-error-'));
  const calls = [];
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
    event: { subscribe: async ({ signal }) => ({ stream: (async function* () {
      yield { type: 'session.error', properties: { sessionID: 'unrelated', error: providerError.properties.error } };
      yield providerError;
      await new Promise(resolve => signal.addEventListener('abort', resolve, { once: true }));
    })() }) },
    session: {
      create: async () => ({ data: { id: 'session-target' } }),
      prompt: async (_args, { signal }) => new Promise((_, reject) => {
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
    assert.deepEqual(calls, ['abort', 'delete']);
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
