import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { buildOpenCodeStructuredPrompt, buildOpenCodeToolPolicy, buildOpenCodeSessionPermissions, createOpenCodeStructuredTurnBackend,
  openCodeStructuredTurnBridgeId } from '../../scripts/opencode-structured-turn.mjs';
import { createOpenCodeLoopbackChatAdapter, openCodeLoopbackBridgeId } from '../../scripts/opencode-loopback-chat-adapter.mjs';

const ACTION_TOOL = 'natlang_action_bridge_submit_action';
const actionPart = (name = 'read_file', args = { path: 'a.txt' }) => ({ type: 'tool', tool: ACTION_TOOL,
  state: { status: 'completed', input: { name, arguments: args }, output: 'ACTION_RECORDED' } });

function fakeClient({ structured, text, parts = [], promptError, promptInfoError, onPrompt, events = [],
  permissionReplyData = true, toolIds = ['invalid', 'read', 'bash', 'plugin_search'],
  mcpInitiallyConnected = true, mcpFailure = '' } = {}) {
  const calls = [];
  const mcpCalls = [];
  let mcpConnected = mcpInitiallyConnected;
  let sequence = 0;
  const assistantText = text ?? JSON.stringify(structured ?? { content: 'ok', toolCalls: [] });
  const assistantParts = [...parts, { type: 'text', text: assistantText }];
  return {
    calls,
    mcpCalls,
    mcp: {
      async status(params, options) {
        mcpCalls.push({ method: 'mcp.status', params, options });
        return { data: { natlang_action_bridge: mcpConnected ? { status: 'connected' } :
          mcpFailure ? { status: 'failed', error: mcpFailure } : { status: 'disabled' } } };
      },
      async connect(params, options) {
        mcpCalls.push({ method: 'mcp.connect', params, options });
        mcpConnected = !mcpFailure;
        return { data: { status: mcpConnected ? 'connected' : 'failed' } };
      }
    },
    tool: {
      async ids(params, options) {
        calls.push({ method: 'tool.ids', params, options });
        return { data: toolIds };
      }
    },
    session: {
      async create(params, options) {
        calls.push({ method: 'create', params, options });
        return { data: { id: `ses-${++sequence}` } };
      },
      async prompt(params, options) {
        calls.push({ method: 'prompt', params, options });
        await onPrompt?.(params, options);
        if (promptError) throw promptError;
        return { data: { info: { id: 'assistant-final', role: 'assistant', error: promptInfoError,
          tokens: { input: 77, output: 12 } }, parts: assistantParts } };
      },
      async messages(params, options) {
        calls.push({ method: 'messages', params, options });
        return { data: [{ info: { id: 'user-1', role: 'user' }, parts: [] },
          { info: { id: 'assistant-final', role: 'assistant', tokens: { input: 77, output: 12 } }, parts: assistantParts }] };
      },
      async abort(params) {
        calls.push({ method: 'abort', params });
        return { data: true };
      },
      async delete(params) {
        calls.push({ method: 'delete', params });
        return { data: true };
      },
      permission: {
        async reply(params) {
          calls.push({ method: 'permission.v2.reply', params });
          return { data: undefined };
        }
      }
    },
    permission: {
      async reply(params) {
        calls.push({ method: 'permission.reply', params });
        return { data: permissionReplyData };
      }
    },
    ...(events.length ? { event: { async subscribe(_params, options) {
      await options.fetch(new Request('http://127.0.0.1/event'));
      return { stream: (async function* () { for (const event of events) yield event; })() };
    } } } : {})
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

test('builds a JSON text prompt with exact Natlang transcript and declared response schema', () => {
  const prompt = buildOpenCodeStructuredPrompt(request, { providerID: 'opencode', modelID: 'exo-free' });
  assert.equal(prompt.body.model.providerID, 'opencode');
  assert.equal(prompt.body.model.modelID, 'exo-free');
  assert.equal(Object.hasOwn(prompt.body, 'format'), false, 'avoid invalid stored JSON Schema metadata in OpenCode history');
  assert.deepEqual(prompt.responseSchema.required, ['content', 'toolCalls']);
  assert.deepEqual(prompt.responseSchema.properties.toolCalls.items.properties.name.enum, ['read_file']);
  assert.equal(prompt.responseSchema.properties.toolCalls.minItems, 1);
  assert.equal(Object.hasOwn(prompt.body, 'tools'), false, 'SDK native inventory is retained through default prompt behavior');
  const payload = JSON.parse(prompt.body.parts[0].text);
  assert.equal(payload.protocol, openCodeStructuredTurnBridgeId);
  assert.deepEqual(payload.messages, request.messages);
  assert.deepEqual(payload.tools, request.tools);
  assert.deepEqual(payload.response_schema, prompt.responseSchema);
  assert.match(prompt.body.system, /not provider-enforced JSON Schema/);
  assert.equal(payload.invocation_id, undefined, 'collector-only invocation identity is not model context');
});

test('retains the full default inventory and builds a wildcard-ask plus exact action-MCP session policy', () => {
  assert.deepEqual(buildOpenCodeToolPolicy(['invalid', 'bash', 'mcp_browser_search', ACTION_TOOL, 'new_plugin']), {
    wildcard: 'ask', action_tool: ACTION_TOOL,
    retained_default_inventory: ['bash', 'invalid', 'mcp_browser_search', ACTION_TOOL, 'new_plugin']
  });
  assert.deepEqual(buildOpenCodeSessionPermissions(['read_file']), [
    { permission: '*', pattern: '*', action: 'ask' },
    { permission: ACTION_TOOL, pattern: '*', action: 'allow' }
  ]);
  assert.deepEqual(buildOpenCodeSessionPermissions([]), [
    { permission: '*', pattern: '*', action: 'ask' },
    { permission: ACTION_TOOL, pattern: '*', action: 'deny' }
  ]);
  assert.deepEqual(buildOpenCodeToolPolicy(['bash']).retained_default_inventory, ['bash']);
  assert.throws(() => buildOpenCodeToolPolicy(['bash', ACTION_TOOL, ACTION_TOOL]), /duplicate IDs/);
  assert.throws(() => buildOpenCodeToolPolicy(['*', ACTION_TOOL]), /reserved wildcard ID/);
  assert.throws(() => buildOpenCodeToolPolicy([null, ACTION_TOOL]), /nonempty IDs/);
});

test('uses connected MCP status for dynamic action registration separate from built-in tool IDs', async () => {
  const client = fakeClient({ toolIds: ['bash', 'read'], parts: [actionPart()] });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test', eventFetchImpl: async () => new Response('data: {}\n\n',
      { headers: { 'content-type': 'text/event-stream' } }) });
  const result = await turn(request);
  assert.deepEqual(result.calls, [['read_file', { path: 'a.txt' }]]);
  assert.deepEqual(client.calls.map(call => call.method), ['tool.ids', 'create', 'prompt', 'messages', 'delete']);
});

test('maps the isolated action-MCP record to a Natlang call and always deletes its session', async () => {
  const client = fakeClient({ structured: { content: '', toolCalls: [] }, parts: [actionPart()] });
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
  assert.deepEqual(result.raw_response.assistant_steps[0].parsed_json_text.toolCalls, []);
  assert.match(result.raw_response.fidelity, /not provider-enforced JSON Schema or native provider tool-call output/);
  assert.equal(result.raw_response.output_contract, 'exact JSON text parsed and validated by the bridge');
  assert.deepEqual(result.raw_response.open_code_tool_policy, {
    control: 'official SDK session.create permission rules; prompt default inventory retained',
    action_mcp: { name: 'natlang_action_bridge', status: 'connected', tool_id: ACTION_TOOL,
      tool_verified_by_connected_MCP_tools_list: true },
    inventory: 'official SDK tool.ids built-in inventory; connected MCP tools are resolved separately by OpenCode',
    inventory_ids: ['bash', 'invalid', 'plugin_search', 'read'],
    wildcard_action: 'ask', allowed_tool_ids: [ACTION_TOOL], rejected_permission_requests: [],
    session_history_audited: true, non_bridge_tool_parts_observed: false,
    native_execution_prevention: 'not established by permission policy and post-turn history audit'
  });
  assert.deepEqual(buildOpenCodeToolPolicy(['invalid', 'read', 'bash', 'plugin_search', ACTION_TOOL]).retained_default_inventory,
    ['bash', 'invalid', ACTION_TOOL, 'plugin_search', 'read']);
  const promptCall = client.calls.find(call => call.method === 'prompt');
  assert.equal(Object.hasOwn(promptCall.params, 'tools'), false);
  assert.deepEqual(client.calls.find(call => call.method === 'create').params.permission, buildOpenCodeSessionPermissions(['read_file']));
  assert.equal(Object.hasOwn(promptCall.params, 'format'), false);
  assert.deepEqual(client.calls.map(call => call.method), ['tool.ids', 'create', 'prompt', 'messages', 'delete']);
  assert.deepEqual(client.mcpCalls.map(call => call.method), ['mcp.status']);
});

test('connects the configured MCP through the official SDK before checking dynamic tool IDs', async () => {
  const client = fakeClient({ structured: { content: 'No action.', toolCalls: [] }, mcpInitiallyConnected: false });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  await turn({ ...request, tool_choice: 'auto' });
  assert.deepEqual(client.mcpCalls.map(call => call.method), ['mcp.status', 'mcp.connect', 'mcp.status']);
  assert.deepEqual(client.mcpCalls[1].params, { name: 'natlang_action_bridge', directory: '/tmp/natlang-opencode-test' });
  assert.equal(client.calls[0].method, 'tool.ids');
});

test('preserves bounded, redacted MCP connection diagnostics before session creation', async () => {
  const client = fakeClient({ mcpInitiallyConnected: false,
    mcpFailure: 'spawn failed Authorization: Bearer abcdefghijklmnop OPENCODE_API_KEY=sk-secret-value' });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  await assert.rejects(turn(request), error => {
    assert.match(error.message, /status: failed/);
    assert.doesNotMatch(error.message, /abcdefghijklmnop|sk-secret-value/);
    return true;
  });
  assert.deepEqual(client.calls, []);
});

test('rejects every asked built-in permission with the official SDK and records that audit', async () => {
  const client = fakeClient({ structured: { content: 'No action.', toolCalls: [] },
    events: [{ id: 'e-1', type: 'permission.asked', properties: { id: 'perm-1', sessionID: 'ses-1',
      permission: 'bash', patterns: ['*'], metadata: {}, always: [] } }],
    onPrompt: async () => new Promise(resolve => setTimeout(resolve, 20)) });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test', eventFetchImpl: async () => new Response('data: {}\n\n',
      { headers: { 'content-type': 'text/event-stream' } }) });
  const result = await turn({ ...request, tool_choice: 'auto' });
  assert.deepEqual(client.calls.find(call => call.method === 'permission.reply').params, {
    requestID: 'perm-1', directory: '/tmp/natlang-opencode-test', reply: 'reject',
    message: 'Natlang bridge rejects every requested OpenCode tool except its pre-authorized action MCP tool.'
  });
  assert.deepEqual(result.raw_response.open_code_tool_policy.rejected_permission_requests, [{
    request_id: 'perm-1', permission: 'bash', patterns: ['*'], reply: 'reject', succeeded: true
  }]);
});

test('fails closed when the SDK does not confirm permission rejection', async () => {
  const client = fakeClient({ structured: { content: 'No action.', toolCalls: [] }, permissionReplyData: false,
    events: [{ id: 'e-1', type: 'permission.asked', properties: { id: 'perm-1', sessionID: 'ses-1',
      permission: 'bash', patterns: ['*'], metadata: {}, always: [] } }] });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test', eventFetchImpl: async () => new Response('data: {}\n\n',
      { headers: { 'content-type': 'text/event-stream' } }) });
  await assert.rejects(turn({ ...request, tool_choice: 'auto' }), /permission request could not be rejected/);
  assert.equal(client.calls.at(-1).method, 'delete');
});

test('rejects v2 permission requests through the official session permission reply API', async () => {
  const client = fakeClient({ structured: { content: 'No action.', toolCalls: [] },
    events: [{ id: 'e-2', type: 'permission.v2.asked', properties: { id: 'perm-v2', sessionID: 'ses-1',
      action: 'bash', resources: ['command:*'] } }] });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'ling-3.1-flash-free',
    directory: '/tmp/natlang-opencode-test', eventFetchImpl: async () => new Response('data: {}\n\n',
      { headers: { 'content-type': 'text/event-stream' } }) });
  const result = await turn({ ...request, tool_choice: 'auto' });
  assert.deepEqual(client.calls.find(call => call.method === 'permission.v2.reply').params, {
    sessionID: 'ses-1', requestID: 'perm-v2', directory: '/tmp/natlang-opencode-test', reply: 'reject',
    message: 'Natlang bridge rejects every requested OpenCode tool except its pre-authorized action MCP tool.'
  });
  assert.deepEqual(result.raw_response.open_code_tool_policy.rejected_permission_requests, [{
    request_id: 'perm-v2', permission: 'bash', patterns: ['command:*'], reply: 'reject', succeeded: true
  }]);
});

test('maps text-only turns and denies the action MCP when Natlang offered no tools', async () => {
  const client = fakeClient({ structured: { content: 'Done.', toolCalls: [] } });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  const result = await turn({ messages: [{ role: 'user', content: 'Say done' }], tools: [],
    tool_choice: 'auto', temperature: undefined, seed: null, max_tokens: 50 });
  assert.deepEqual(result.text, 'Done.');
  assert.equal(result.calls, undefined);
  const promptCall = client.calls.find(call => call.method === 'prompt');
  assert.equal(JSON.parse(promptCall.params.parts[0].text).response_schema.properties.toolCalls.maxItems, 0);
  assert.deepEqual(client.calls.find(call => call.method === 'create').params.permission, buildOpenCodeSessionPermissions([]));
});

test('rejects markdown or malformed JSON text instead of extracting or repairing it', async () => {
  for (const text of ['```json\n{"content":"ok","toolCalls":[]}\n```', '{"content":"ok"}']) {
    const client = fakeClient({ text });
    const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
      directory: '/tmp/natlang-opencode-test' });
    await assert.rejects(turn({ ...request, tool_choice: 'auto' }), /not one valid JSON action object|exactly content and toolCalls/);
  }
});

test('preserves bounded, redacted assistant evidence and classifies parse/schema failures before session cleanup', async t => {
  const scratch = mkdtempSync(join(tmpdir(), 'natlang-opencode-failure-test-'));
  t.after(() => rmSync(scratch, { recursive: true, force: true }));
  const invalidText = `Bearer abc123 api_key="secret-value" ${'x'.repeat(20_000)}`;
  const invalidClient = fakeClient({ text: invalidText });
  const invalidTurn = createOpenCodeStructuredTurnBackend({ client: invalidClient, providerID: 'opencode',
    modelID: 'exo-free', directory: scratch });
  await assert.rejects(invalidTurn({ ...request, tool_choice: 'auto' }), error => {
    const diagnostic = error.transportDiagnostic;
    assert.equal(diagnostic.version, 'natlang.opencode_transport_failure/1');
    assert.equal(diagnostic.classification, 'invalid_json_text');
    assert.equal(diagnostic.provider_id, 'opencode');
    assert.equal(diagnostic.model_id, 'exo-free');
    assert.equal(diagnostic.session_id, 'ses-1');
    assert.equal(diagnostic.assistant_text.bytes, Buffer.byteLength(invalidText));
    assert.equal(diagnostic.assistant_text.preview_truncated, true);
    assert.equal(diagnostic.assistant_text.redacted, true);
    assert.match(diagnostic.assistant_text.redacted_preview, /Bearer \[REDACTED\]/);
    assert.doesNotMatch(diagnostic.assistant_text.redacted_preview, /abc123|secret-value/);
    assert.equal(diagnostic.parts.length, 1);
    assert.ok(diagnostic.parts[0].bytes > Buffer.byteLength(invalidText), 'part digest covers the serialized part record');
    const sidecar = JSON.parse(readFileSync(diagnostic.evidence_path, 'utf8'));
    assert.equal(sidecar.assistant_text.bytes, Buffer.byteLength(invalidText));
    assert.match(sidecar.assistant_text.redacted_preview, /Bearer \[REDACTED\]/);
    assert.doesNotMatch(sidecar.assistant_text.redacted_preview, /abc123|secret-value/);
    assert.equal(sidecar.assistant_text.preview_truncated, true);
    assert.ok(Buffer.byteLength(sidecar.assistant_text.redacted_preview) <= 16 * 1024);
    assert.equal(invalidClient.calls.at(-1).method, 'delete');
    return true;
  });

  for (const [structured, classification, toolChoice] of [
    [{ content: 'ok', toolCalls: [{ name: 'not_declared', arguments: {} }] }, 'schema_invalid_json_text', 'auto'],
    [{ content: 'ok', toolCalls: [] }, 'schema_invalid_json_text', 'required']
  ]) {
    const client = fakeClient({ structured });
    const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
      directory: '/tmp/natlang-opencode-test' });
    await assert.rejects(turn({ ...request, tool_choice: toolChoice }), error => {
      assert.equal(error.transportDiagnostic.classification, classification);
      assert.equal(error.transportDiagnostic.session_id, 'ses-1');
      assert.equal(client.calls.at(-1).method, 'delete');
      return true;
    });
  }
});

test('classifies empty assistant text and native OpenCode tool refusal without preserving tool arguments', async () => {
  const emptyClient = fakeClient({ text: '' });
  const emptyTurn = createOpenCodeStructuredTurnBackend({ client: emptyClient, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  await assert.rejects(emptyTurn({ ...request, tool_choice: 'auto' }), error => {
    assert.equal(error.transportDiagnostic.classification, 'empty_assistant_text');
    assert.equal(error.transportDiagnostic.assistant_text.bytes, 0);
    return true;
  });

  const nativeClient = fakeClient({ structured: { content: '', toolCalls: [] },
    parts: [{ type: 'tool', tool: 'bash', state: { status: 'completed', input: { command: 'secret command' } } }] });
  const nativeTurn = createOpenCodeStructuredTurnBackend({ client: nativeClient, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  await assert.rejects(nativeTurn({ ...request, tool_choice: 'auto' }), error => {
    assert.equal(error.transportDiagnostic.classification, 'native_tool_refusal');
    assert.equal(error.transportDiagnostic.parts[0].tool, 'bash');
    assert.equal(JSON.stringify(error.transportDiagnostic).includes('secret command'), false);
    return true;
  });
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
  await assert.rejects(turn2({ ...request, tool_choice: 'auto' }), /contains non-bridge tool part/);
  assert.equal(external.calls.at(-1).method, 'delete');
});

test('retains an exact invalid-handler rejection as evidence without treating it as a Natlang action', async () => {
  const error = "Model tried to call unavailable tool 'read_file'. Available tools: bash, read";
  const rejected = { type: 'tool', tool: 'invalid', state: { status: 'completed', title: 'Invalid Tool',
    input: { tool: 'read_file', error }, output: `The arguments provided to the tool are invalid: ${error}`, metadata: {} } };
  const client = fakeClient({ structured: { content: 'No action.', toolCalls: [] }, parts: [rejected] });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  const result = await turn({ ...request, tool_choice: 'auto' });
  assert.equal(result.calls, undefined);
  assert.equal(result.text, 'No action.');
  assert.deepEqual(result.raw_response.rejected_native_tool_attempts, [{ rejected_tool_name: 'read_file', rejection: error,
    handler: 'OpenCode InvalidTool', status: 'completed', protocol_record: rejected }]);
  assert.deepEqual(result.raw_response.open_code_tool_parts, [{ name: 'invalid', status: 'completed' }]);
});

test('does not tolerate malformed invalid-handler records or rejected names absent from the Natlang request', async () => {
  const validError = "Model tried to call unavailable tool 'eval'. Available tools: read";
  const validState = { status: 'completed', title: 'Invalid Tool', input: { tool: 'eval', error: validError },
    output: `The arguments provided to the tool are invalid: ${validError}`, metadata: {} };
  const invalidParts = [
    { type: 'tool', tool: 'invalid', state: { ...validState, output: 'a different handler output' } },
    { type: 'tool', tool: 'invalid', state: { ...validState, status: 'error' } },
    { type: 'tool', tool: 'invalid', state: { ...validState, input: { ...validState.input, tool: 'bash' } } },
    { type: 'tool', tool: 'StructuredOutput', state: { status: 'completed' } }
  ];
  for (const part of invalidParts) {
    const client = fakeClient({ structured: { content: 'ok', toolCalls: [] }, parts: [part] });
    const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
      directory: '/tmp/natlang-opencode-test' });
    await assert.rejects(turn({ ...request, tool_choice: 'auto' }), /contains non-bridge tool part/);
    assert.equal(client.calls.at(-1).method, 'delete');
  }
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
  await assert.rejects(turn({ ...request, tool_choice: 'auto' }), error => {
    assert.match(error.message, /contains non-bridge tool part\(s\): read/);
    assert.equal(error.transportDiagnostic.classification, 'native_tool_refusal');
    assert.equal(error.transportDiagnostic.parts[0].tool, 'read');
    return true;
  });
  assert.equal(client.calls.at(-1).method, 'delete');
});

test('requests the complete session history without imposing an SDK page limit', async () => {
  const client = fakeClient({ structured: { content: '', toolCalls: [] } });
  const history = Array.from({ length: 1001 }, (_, index) => ({ info: { id: `assistant-${index}`, role: 'assistant' }, parts: [] }));
  history.push({ info: { id: 'assistant-final', role: 'assistant' }, parts: [] });
  client.session.messages = async params => {
    assert.equal(Object.hasOwn(params, 'limit'), false);
    return { data: history };
  };
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  const result = await turn({ ...request, tool_choice: 'auto' });
  assert.equal(result.raw_response.session_messages_audited, 1002);
  assert.equal(result.raw_response.assistant_steps.at(-1).message_id, 'assistant-final');
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

test('enforces required tool calls and cleans up when MCP did not record an action', async () => {
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
    // The prompt runs under its own controller, which the caller's abort is forwarded to.
    assert.ok(options.signal instanceof AbortSignal);
    controller.abort(new Error('stop requested'));
    assert.equal(options.signal.aborted, true);
    assert.equal(options.signal.reason, controller.signal.reason);
    throw options.signal.reason;
  } });
  const turn = createOpenCodeStructuredTurnBackend({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: '/tmp/natlang-opencode-test' });
  await assert.rejects(turn(request, controller.signal), /stop requested/);
  assert.deepEqual(client.calls.map(call => call.method), ['tool.ids', 'create', 'prompt', 'abort', 'delete']);
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

test('loopback HTTP adapter returns OpenAI shape from an isolated action-MCP record', async t => {
  const client = fakeClient({ structured: { content: '', toolCalls: [] }, parts: [actionPart()] });
  const adapter = await createOpenCodeLoopbackChatAdapter({ client, providerID: 'opencode', modelID: 'exo-free',
    modelAlias: 'opencode/exo-free', directory: '/tmp/natlang-opencode-test', maxConcurrency: 1 });
  t.after(() => adapter.close());
  assert.equal(Object.isFrozen(adapter.config), true);
  assert.equal(adapter.config.host, '127.0.0.1');
  assert.match(adapter.config.nativeOpenCodeToolPolicy, /default inventory retained/);
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
  assert.deepEqual(client.calls.map(call => call.method), ['tool.ids', 'create', 'prompt', 'messages', 'delete']);
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

test('loopback forwards only the bounded bridge transport diagnostic on protocol failure', async t => {
  const client = fakeClient({ text: 'not json', parts: Array.from({ length: 8 }, () => ({ type: 'reasoning', text: 'x'.repeat(4096) })) });
  const scratch = mkdtempSync(join(tmpdir(), 'natlang-opencode-loopback-failure-test-'));
  t.after(() => rmSync(scratch, { recursive: true, force: true }));
  const adapter = await createOpenCodeLoopbackChatAdapter({ client, providerID: 'opencode', modelID: 'exo-free',
    directory: scratch });
  t.after(() => adapter.close());
  const response = await fetch(`${adapter.url}/v1/chat/completions`, { method: 'POST',
    headers: { 'content-type': 'application/json' }, body: JSON.stringify(chatRequest({ tool_choice: 'auto' })) });
  assert.equal(response.status, 502);
  const responseText = await response.text();
  assert.ok(Buffer.byteLength(responseText) < 1_985, 'collector error capture preserves the complete diagnostic body under its 2 KB cap');
  const body = JSON.parse(responseText);
  assert.equal(body.error.transport_diagnostic.classification, 'invalid_json_text');
  assert.equal(body.error.transport_diagnostic.session_id, 'ses-1');
  assert.equal(body.error.transport_diagnostic.parts_omitted_count, 5);
  assert.equal(body.error.transport_diagnostic.parts.length, 4);
  assert.equal(JSON.stringify(body).includes('authorization'), false);
  assert.deepEqual(client.calls.map(call => call.method), ['tool.ids', 'create', 'prompt', 'delete']);
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

test('action MCP falls back to its supported protocol version and records actions without executing them', t => {
  const scratch = mkdtempSync(join(tmpdir(), 'natlang-opencode-mcp-protocol-test-'));
  t.after(() => rmSync(scratch, { recursive: true, force: true }));
  const actionLog = join(scratch, 'actions.jsonl');
  const server = resolve('scripts/opencode-natlang-action-mcp-server.mjs');
  const input = [
    { jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-11-25' } },
    { jsonrpc: '2.0', id: 2, method: 'initialize', params: { protocolVersion: '2024-11-05' } },
    { jsonrpc: '2.0', id: 3, method: 'tools/list', params: {} },
    { jsonrpc: '2.0', id: 4, method: 'tools/call', params: { name: 'submit_action', arguments: {
      name: 'read_file', arguments: { path: 'a.txt' }
    } } }
  ].map(value => JSON.stringify(value)).join('\n') + '\n';
  const child = spawnSync(process.execPath, [server], { input, encoding: 'utf8', env: {
    PATH: process.env.PATH ?? '', HOME: tmpdir(), NATLANG_OPENCODE_ACTION_LOG: actionLog
  } });
  assert.equal(child.status, 0, child.stderr);
  const replies = child.stdout.trim().split('\n').map(line => JSON.parse(line));
  assert.equal(replies[0].id, 1);
  assert.equal(replies[0].result.protocolVersion, '2024-11-05');
  assert.equal(replies[1].result.protocolVersion, '2024-11-05');
  assert.deepEqual(replies[2].result.tools.map(tool => tool.name), ['submit_action']);
  assert.equal(replies[3].result.content[0].text, 'ACTION_RECORDED');
  const recorded = JSON.parse(readFileSync(actionLog, 'utf8'));
  assert.equal(recorded.name, 'read_file');
  assert.deepEqual(recorded.arguments, { path: 'a.txt' });
});
