import test from 'node:test';
import assert from 'node:assert/strict';
import { buildOpenCodeStructuredPrompt, createOpenCodeStructuredTurnBackend,
  openCodeStructuredTurnBridgeId } from '../../scripts/opencode-structured-turn.mjs';
import { createOpenCodeLoopbackChatAdapter, openCodeLoopbackBridgeId } from '../../scripts/opencode-loopback-chat-adapter.mjs';

function fakeClient({ structured, parts = [], promptError, promptInfoError, onPrompt } = {}) {
  const calls = [];
  let sequence = 0;
  return {
    calls,
    session: {
      async create(params, options) {
        calls.push({ method: 'create', params, options });
        return { data: { id: `ses-${++sequence}` } };
      },
      async prompt(params, options) {
        calls.push({ method: 'prompt', params, options });
        await onPrompt?.(params, options);
        if (promptError) throw promptError;
        return { data: { info: { id: 'assistant-final', role: 'assistant', structured, error: promptInfoError,
          tokens: { input: 77, output: 12 } }, parts } };
      },
      async messages(params, options) {
        calls.push({ method: 'messages', params, options });
        return { data: [{ info: { id: 'user-1', role: 'user' }, parts: [] },
          { info: { id: 'assistant-final', role: 'assistant', structured, tokens: { input: 77, output: 12 } }, parts }] };
      },
      async abort(params) {
        calls.push({ method: 'abort', params });
        return { data: true };
      },
      async delete(params) {
        calls.push({ method: 'delete', params });
        return { data: true };
      }
    }
  };
}

const request = {
  invocation_id: 'private-collector-metadata',
  messages: [
    { role: 'system', content: 'Natlang instructions' },
    { role: 'user', content: 'Read the file' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'prior-1', type: 'function', function: { name: 'read_file', arguments: '{"path":"a.txt"}' } }] },
    { role: 'tool', tool_call_id: 'prior-1', content: 'file text' }
  ],
  tools: [{ type: 'function', function: { name: 'read_file', description: 'Read a file', parameters: {
    type: 'object', properties: { path: { type: 'string' } }, required: ['path'], additionalProperties: false
  } } }],
  tool_choice: 'required', temperature: 0, seed: null, max_tokens: 300
};

test('builds structured prompt with exact Natlang transcript and declared action schema', () => {
  const prompt = buildOpenCodeStructuredPrompt(request, { providerID: 'opencode', modelID: 'exo-free' });
  assert.equal(prompt.body.model.providerID, 'opencode');
  assert.equal(prompt.body.model.modelID, 'exo-free');
  assert.equal(prompt.body.format.type, 'json_schema');
  assert.deepEqual(prompt.body.format.schema.required, ['content', 'toolCalls']);
  assert.deepEqual(prompt.body.format.schema.properties.toolCalls.items.properties.name.enum, ['read_file']);
  assert.equal(Object.hasOwn(prompt.body, 'tools'), false, 'leave OpenCode tool availability at its configured defaults');
  const payload = JSON.parse(prompt.body.parts[0].text);
  assert.equal(payload.protocol, openCodeStructuredTurnBridgeId);
  assert.deepEqual(payload.messages, request.messages);
  assert.deepEqual(payload.tools, request.tools);
  assert.equal(payload.invocation_id, undefined, 'collector-only invocation identity is not model context');
});

test('maps structured output to a clearly labeled Natlang action and always deletes its session', async () => {
  const client = fakeClient({ structured: { content: '', toolCalls: [{ name: 'read_file', arguments: { path: 'a.txt' } }] },
    parts: [{ type: 'tool', tool: 'StructuredOutput', state: { status: 'completed' } }] });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free', agent: 'build',
    directory: '/tmp/natlang-opencode-test' });
  const result = await turn(request);
  assert.deepEqual(result.calls, [['read_file', { path: 'a.txt' }]]);
  assert.equal(result.text, undefined);
  assert.equal(result.prompt_tokens, 77);
  assert.equal(result.completion_tokens, 12);
  assert.equal(result.raw_calls, undefined, 'do not invent provider-native tool call records');
  assert.equal(result.raw_response.transport, openCodeStructuredTurnBridgeId);
  assert.equal(result.raw_response.session_messages_audited, 2);
  assert.equal(result.raw_response.assistant_steps[0].message_id, 'assistant-final');
  assert.equal(result.raw_response.assistant_steps[0].structured_output.toolCalls[0].name, 'read_file');
  assert.match(result.raw_response.fidelity, /not native provider tool-call output/);
  assert.deepEqual(client.calls.map(call => call.method), ['create', 'prompt', 'messages', 'delete']);
});

test('maps text-only turns and accepts no tools when Natlang offered none', async () => {
  const client = fakeClient({ structured: { content: 'Done.', toolCalls: [] } });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  const result = await turn({ messages: [{ role: 'user', content: 'Say done' }], tools: [],
    tool_choice: 'auto', temperature: undefined, seed: null, max_tokens: 50 });
  assert.deepEqual(result.text, 'Done.');
  assert.equal(result.calls, undefined);
  const promptCall = client.calls.find(call => call.method === 'prompt');
  assert.equal(promptCall.params.format.schema.properties.toolCalls.maxItems, 0);
});

test('rejects unavailable calls and OpenCode tool execution instead of laundering them as Natlang actions', async () => {
  const unknown = fakeClient({ structured: { content: '', toolCalls: [{ name: 'bash', arguments: { command: 'echo x' } }] } });
  const turn = createOpenCodeStructuredTurnBackend({ client: unknown, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  await assert.rejects(turn(request), /unavailable Natlang tool/);
  assert.equal(unknown.calls.at(-1).method, 'delete');

  const external = fakeClient({ structured: { content: '', toolCalls: [] },
    parts: [{ type: 'tool', tool: 'bash', state: { status: 'completed' } }] });
  const turn2 = createOpenCodeStructuredTurnBackend({ client: external, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  await assert.rejects(turn2({ ...request, tool_choice: 'auto' }), /executed non-bridge tool/);
  assert.equal(external.calls.at(-1).method, 'delete');
});

test('rejects OpenCode built-in tool use found anywhere in session history', async () => {
  const client = fakeClient({ structured: { content: '', toolCalls: [] } });
  const original = client.session.messages;
  client.session.messages = async (...args) => {
    const result = await original(...args);
    result.data.unshift({ info: { role: 'assistant' }, parts: [
      { type: 'tool', tool: 'read', state: { status: 'completed' } }
    ] });
    return result;
  };
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  await assert.rejects(turn({ ...request, tool_choice: 'auto' }), /executed non-bridge tool\(s\) in the session: read/);
  assert.equal(client.calls.at(-1).method, 'delete');
});

test('rejects session history at the audit page bound as potentially truncated', async () => {
  const client = fakeClient({ structured: { content: '', toolCalls: [] } });
  client.session.messages = async () => ({ data: Array.from({ length: 1000 }, () => ({ info: {}, parts: [] })) });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  await assert.rejects(turn({ ...request, tool_choice: 'auto' }), /incomplete or possibly truncated/);
});

test('rejects empty session history or history missing the final assistant message', async () => {
  for (const history of [[], [{ info: { id: 'user-1', role: 'user' }, parts: [] }]]) {
    const client = fakeClient({ structured: { content: 'ok', toolCalls: [] } });
    client.session.messages = async () => ({ data: history });
    const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
      directory: '/tmp/natlang-opencode-test' });
    await assert.rejects(turn({ ...request, tool_choice: 'auto' }), /audit is empty|does not contain the exact final/);
  }
});

test('aggregates and preserves usage and cost across assistant steps', async () => {
  const client = fakeClient({ structured: { content: 'ok', toolCalls: [] } });
  client.session.messages = async () => ({ data: [
    { info: { id: 'assistant-retry', role: 'assistant', providerID: 'opencode', modelID: 'exo-free',
      cost: 0.004, finish: 'retry', tokens: { input: 11, output: 3, reasoning: 1, cache: { read: 2, write: 1 } },
      structured: { content: '', toolCalls: [] } }, parts: [] },
    { info: { id: 'assistant-final', role: 'assistant', providerID: 'opencode', modelID: 'exo-free',
      cost: 0.006, finish: 'stop', tokens: { input: 17, output: 5 }, structured: { content: 'ok', toolCalls: [] } }, parts: [] }
  ] });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  const result = await turn({ ...request, tool_choice: 'auto' });
  assert.equal(result.prompt_tokens, 28);
  assert.equal(result.completion_tokens, 8);
  assert.equal(result.raw_response.assistant_total_cost, 0.01);
  assert.deepEqual(result.raw_response.assistant_steps.map(step => step.message_id), ['assistant-retry', 'assistant-final']);
  assert.deepEqual(result.raw_response.assistant_steps[0].tokens.cache, { read: 2, write: 1 });
});

test('enforces required tool calls and cleans up when parsing fails', async () => {
  const client = fakeClient({ structured: { content: 'No call', toolCalls: [] } });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  await assert.rejects(turn(request), /required a tool call/);
  assert.equal(client.calls.at(-1).method, 'delete');
});

test('preserves safe OpenCode provider status and retryability for supervisor backoff', async () => {
  const client = fakeClient({ promptInfoError: { name: 'APIError', data: {
    message: 'temporary endpoint failure', statusCode: 503, isRetryable: true,
    responseHeaders: { authorization: 'must not leak' }, responseBody: 'secret body'
  } } });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  await assert.rejects(turn({ ...request, tool_choice: 'auto' }), error => {
    assert.equal(error.code, 'OPENCODE_PROVIDER_ERROR');
    assert.equal(error.providerStatusCode, 503);
    assert.equal(error.providerRetryable, true);
    assert.equal(JSON.stringify(error).includes('must not leak'), false);
    return true;
  });
});

test('forwards abort signals and aborts then deletes the isolated session', async () => {
  const controller = new AbortController();
  const client = fakeClient({ onPrompt: async (_params, options) => {
    assert.equal(options.signal, controller.signal);
    controller.abort(new Error('stop requested'));
    throw controller.signal.reason;
  } });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  await assert.rejects(turn(request, controller.signal), /stop requested/);
  assert.deepEqual(client.calls.map(call => call.method), ['create', 'prompt', 'abort', 'delete']);
});

test('bounds a hung OpenCode session cleanup', async () => {
  const client = fakeClient({ structured: { content: 'ok', toolCalls: [] } });
  client.session.delete = async () => new Promise(() => {});
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test', cleanupTimeoutMs: 25 });
  await assert.rejects(turn({ ...request, tool_choice: 'auto' }), /session cleanup timed out/);
});

function chatRequest(overrides = {}) {
  return { model: 'opencode/exo-free', messages: [{ role: 'user', content: 'Read a.txt' }],
    tools: request.tools, tool_choice: 'required', stream: false, temperature: 0, max_tokens: 32, ...overrides };
}

test('loopback HTTP adapter returns OpenAI shape with explicit bridge provenance', async t => {
  const client = fakeClient({ structured: { content: '', toolCalls: [{ name: 'read_file', arguments: { path: 'a.txt' } }] } });
  const adapter = await createOpenCodeLoopbackChatAdapter({ client, providerID: 'opencode', modelID: 'exo-free',
    modelAlias: 'opencode/exo-free', directory: '/tmp/natlang-opencode-test', maxConcurrency: 1 });
  t.after(() => adapter.close());
  assert.equal(Object.isFrozen(adapter.config), true);
  assert.equal(adapter.config.host, '127.0.0.1');
  const response = await fetch(`${adapter.url}/v1/chat/completions`, { method: 'POST',
    headers: { 'content-type': 'application/json' }, body: JSON.stringify(chatRequest()) });
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.object, 'chat.completion');
  assert.equal(body.choices[0].finish_reason, 'tool_calls');
  assert.equal(body.choices[0].message.tool_calls[0].function.name, 'read_file');
  assert.deepEqual(JSON.parse(body.choices[0].message.tool_calls[0].function.arguments), { path: 'a.txt' });
  assert.equal(body.x_natlang_bridge.id, openCodeLoopbackBridgeId);
  assert.equal(body.x_natlang_bridge.native_provider_tool_calls, false);
  assert.equal(body.x_natlang_bridge.incremental_token_streaming, false);
  assert.equal(body.raw_response.session_id, 'ses-1');
  assert.equal(body.raw_response.structured_output.content, '');
  assert.equal(body.raw_response.session_messages_audited, 2);
  assert.deepEqual(client.calls.map(call => call.method), ['create', 'prompt', 'messages', 'delete']);
});

test('loopback adapter returns buffered JSON for collector stream=true and rejects scoring', async t => {
  const client = fakeClient({ structured: { content: 'Completed', toolCalls: [] } });
  const adapter = await createOpenCodeLoopbackChatAdapter({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  t.after(() => adapter.close());
  const streamed = await fetch(`${adapter.url}/v1/chat/completions`, { method: 'POST',
    headers: { 'content-type': 'application/json' }, body: JSON.stringify(chatRequest({ stream: true, tool_choice: 'auto' })) });
  assert.equal(streamed.headers.get('content-type')?.startsWith('application/json'), true);
  const buffered = await streamed.json();
  assert.equal(buffered.choices[0].message.content, 'Completed');
  assert.equal(buffered.x_natlang_bridge.stream_requested, true);
  assert.equal(buffered.x_natlang_bridge.stream_honored, false);

  const scoring = await fetch(`${adapter.url}/v1/chat/completions`, { method: 'POST',
    headers: { 'content-type': 'application/json' }, body: JSON.stringify(chatRequest({ prompt_logprobs: 0 })) });
  assert.equal(scoring.status, 501);
  assert.equal((await scoring.json()).error.code, 'decision_unsupported');
});

test('loopback preserves provider 503 retryability without forwarding provider headers or body', async t => {
  const client = fakeClient({ promptInfoError: { name: 'APIError', data: {
    message: 'temporary unavailable', statusCode: 503, isRetryable: true,
    responseHeaders: { authorization: 'secret' }, responseBody: 'secret provider response'
  } } });
  const adapter = await createOpenCodeLoopbackChatAdapter({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  t.after(() => adapter.close());
  const response = await fetch(`${adapter.url}/v1/chat/completions`, { method: 'POST',
    headers: { 'content-type': 'application/json' }, body: JSON.stringify(chatRequest({ tool_choice: 'auto' })) });
  assert.equal(response.status, 503);
  const body = await response.json();
  assert.equal(body.error.provider_status_code, 503);
  assert.equal(body.error.provider_retryable, true);
  assert.equal(JSON.stringify(body).includes('secret'), false);
});

test('loopback adapter enforces loopback bind, body bound, and concurrency limit', async t => {
  const client = fakeClient({ structured: { content: 'ok', toolCalls: [] }, onPrompt: async () => {
    await new Promise(resolve => setTimeout(resolve, 60));
  } });
  await assert.rejects(createOpenCodeLoopbackChatAdapter({ client, providerID: 'opencode', modelID: 'exo-free', host: '0.0.0.0' }),
    /loopback address/);
  const adapter = await createOpenCodeLoopbackChatAdapter({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test', maxConcurrency: 1, maxBodyBytes: 1024 });
  t.after(() => adapter.close());
  const first = fetch(`${adapter.url}/v1/chat/completions`, { method: 'POST',
    headers: { 'content-type': 'application/json' }, body: JSON.stringify(chatRequest({ tool_choice: 'auto' })) });
  await new Promise(resolve => setTimeout(resolve, 5));
  const second = await fetch(`${adapter.url}/v1/chat/completions`, { method: 'POST',
    headers: { 'content-type': 'application/json' }, body: JSON.stringify(chatRequest({ tool_choice: 'auto' })) });
  assert.equal(second.status, 429);
  await first;
  const oversized = await fetch(`${adapter.url}/v1/chat/completions`, { method: 'POST',
    headers: { 'content-type': 'application/json' }, body: JSON.stringify({ tooBig: 'x'.repeat(2048) }) });
  assert.equal(oversized.status, 413);
});
