import assert from 'node:assert/strict';
import { chmod, mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { abortOpenCodeSession, createOpenCodeCliChatAdapter } from '../scripts/opencode-cli-chat-adapter.mjs';

async function fixture({ actions = [{ name: 'probe_tool', arguments: { value: 1 } }], extraEvents = [], permissionEvent, retryEvent, exitCode = 0, timeoutMs = 1500, abortError = { name: 'MessageAbortedError', message: '' }, abortDelayMs = 0, abortConfirmed = true, modelVariant, agentName } = {}) {
  const root = await mkdtemp(join(tmpdir(), 'natlang-opencode-boundary-'));
  const outputDirectory = join(root, 'output'); await mkdir(outputDirectory);
  const actionLogPath = join(root, 'actions.jsonl'); await writeFile(actionLogPath, '');
  const markerPath = join(root, 'abort.marker');
  const argvPath = join(root, 'argv.json');
  const cliPath = join(root, 'fake-cli.mjs');
  const script = `#!/usr/bin/env node
import fs from 'node:fs';
const sid = 'fake-session-1';
const actions = JSON.parse(process.env.FAKE_TOOL_ACTIONS || '[]');
const send = event => process.stdout.write(JSON.stringify(event) + '\\n');
fs.writeFileSync(process.env.FAKE_ARGV, JSON.stringify(process.argv.slice(2)));
send({ type: 'step_start', sessionID: sid, part: { type: 'step-start', sessionID: sid } });
for (let i = 0; i < actions.length; i++) {
  const action = actions[i];
  fs.appendFileSync(process.env.NATLANG_OPENCODE_ACTION_LOG, JSON.stringify(action) + '\\n');
  send({ type: 'tool_use', sessionID: sid, part: { type: 'tool', tool: 'natlang_action_bridge_submit_action', callID: 'call-' + i, state: { status: 'completed' } } });
}
for (const event of JSON.parse(process.env.FAKE_EXTRA_EVENTS || '[]')) send(event);
send({ type: 'step_finish', sessionID: sid, part: { type: 'step-finish', sessionID: sid, reason: 'tool-calls' } });
await new Promise(resolve => setTimeout(resolve, 100));
if (fs.existsSync(process.env.FAKE_ABORT_MARKER)) {
  send({ type: 'text', sessionID: sid, part: { type: 'text', sessionID: sid, messageID: 'msg-final', text: 'Explanatory final text after action.' } });
} else {
  send({ type: 'step_start', sessionID: sid, part: { type: 'step-start', sessionID: sid } });
  send({ type: 'text', sessionID: sid, part: { type: 'text', sessionID: sid, messageID: 'msg-final', text: '{"content":"echo","toolCalls":[]}' } });
}
process.exit(Number(process.env.FAKE_EXIT_CODE || 0));
`;
  await writeFile(cliPath, script, { mode: 0o700 }); await chmod(cliPath, 0o700);
  const eventClients = new Set();
  const eventServer = createServer((req, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
    res.write(': connected\n\n'); eventClients.add(res); res.on('close', () => eventClients.delete(res));
    if (permissionEvent || retryEvent) setTimeout(() => {
      if (res.destroyed) return;
      if (permissionEvent) res.write(`data: ${JSON.stringify(permissionEvent)}\n\n`);
      if (retryEvent) res.write(`data: ${JSON.stringify(retryEvent)}\n\n`);
    }, 25);
  });
  await new Promise(resolve => eventServer.listen(0, '127.0.0.1', resolve));
  const eventAddress = eventServer.address();
  const aborted = new Set(); const abortCalls = []; const observedStatusArgs = [];
  const permissionCalls = [];
  const client = {
    _client: { permissionReply: async options => { permissionCalls.push(options); return { data: true }; } },
    async postSessionIdPermissionsPermissionId(options) { return this._client.permissionReply(options); },
    mcp: { status: async () => ({ data: { natlang_action_bridge: { status: 'connected' } } }),
      connect: async () => ({ data: true }) },
    tool: { ids: async () => ({ data: ['read', 'write', 'bash'] }) },
    session: {
      abort: async options => {
        abortCalls.push(options); aborted.add(options.path.id);
        await writeFile(markerPath, 'aborted');
        for (const stream of eventClients) stream.write(`data: ${JSON.stringify({ type: 'session.error', properties: { sessionID: options.path.id, error: abortError } })}\n\n`);
        if (abortDelayMs) await new Promise(resolve => setTimeout(resolve, abortDelayMs));
        return { data: abortConfirmed };
      },
      status: async options => {
        observedStatusArgs.push(options);
        return { data: { 'fake-session-1': { type: aborted.has('fake-session-1') ? 'idle' : 'busy' } } };
      },
      messages: async () => ({ data: [] })
    }
  };
  const adapter = await createOpenCodeCliChatAdapter({ cliPath, client,
    baseUrl: `http://127.0.0.1:${eventAddress.port}`, directory: root, outputDirectory,
    modelAlias: 'fixture/free', modelID: 'free', actionLogPath, timeoutMs, maxRequestMs: timeoutMs,
    modelVariant, agentName,
    env: { PATH: process.env.PATH, HOME: root, NATLANG_OPENCODE_ACTION_LOG: actionLogPath,
      FAKE_ABORT_MARKER: markerPath, FAKE_ARGV: argvPath, FAKE_TOOL_ACTIONS: JSON.stringify(actions),
      FAKE_EXTRA_EVENTS: JSON.stringify(extraEvents), FAKE_EXIT_CODE: String(exitCode) } });
  return { root, adapter, eventServer, abortCalls, observedStatusArgs, permissionCalls, actionLogPath, outputDirectory, argvPath,
    async close() { await adapter.close(); eventServer.closeAllConnections(); await new Promise(resolve => eventServer.close(resolve)); await rm(root, { recursive: true, force: true }); } };
}

async function invoke(adapter, { toolChoice } = {}) {
  return fetch(`${adapter.url}/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'fixture/free', messages: [{ role: 'user', content: 'Record the action.' }],
      ...(toolChoice ? { tool_choice: toolChoice } : {}),
      tools: [{ type: 'function', function: { name: 'probe_tool', parameters: { type: 'object', properties: { value: { type: 'number' } } } } }] }) });
}

test('completed audited action stops only its session after the tool-calls step', async () => {
  const f = await fixture({ abortDelayMs: 50 });
  try {
    const response = await invoke(f.adapter); const body = await response.json();
    assert.equal(response.status, 200);
    const argv = JSON.parse(await readFile(f.argvPath, 'utf8'));
    assert.equal(argv[argv.indexOf('--title') + 1], 'Natlang teacher turn', 'explicit documented CLI title avoids auto-title generation from the full prompt');
    assert.equal(body.choices[0].message.content, null);
    assert.equal(body.choices[0].message.tool_calls.length, 1);
    assert.deepEqual(JSON.parse(body.choices[0].message.tool_calls[0].function.arguments), { value: 1 });
    assert.equal(f.abortCalls.length, 1);
    assert.deepEqual(f.abortCalls[0], { path: { id: 'fake-session-1' }, query: { directory: f.root } });
    const diagnostic = JSON.parse((await readFile(join(f.outputDirectory, 'cli-invocations.jsonl'), 'utf8')).trim());
    assert.equal(diagnostic.terminal_action_boundary.status, 'aborted');
    assert.equal(diagnostic.final_text_status, 'invalid_json_final_text_ignored');
    assert.equal(diagnostic.provider_step_telemetry.started, 1);
  } finally { await f.close(); }
});

test('dedicated configured agent is selected explicitly by the CLI invocation', async () => {
  const f = await fixture({ agentName: 'build' });
  try {
    const response = await invoke(f.adapter); assert.equal(response.status, 200);
    const argv = JSON.parse(await readFile(f.argvPath, 'utf8'));
    assert.equal(argv[argv.indexOf('--agent') + 1], 'build');
  } finally { await f.close(); }
});

test('documented reasoning variant is passed separately from the immutable model ID', async () => {
  const f = await fixture({ modelVariant: 'low' });
  try {
    const response = await invoke(f.adapter); assert.equal(response.status, 200);
    const argv = JSON.parse(await readFile(f.argvPath, 'utf8'));
    assert.equal(argv[argv.indexOf('--model') + 1], 'fixture/free');
    assert.equal(argv[argv.indexOf('--variant') + 1], 'low');
    const diagnostic = JSON.parse((await readFile(join(f.outputDirectory, 'cli-invocations.jsonl'), 'utf8')).trim());
    assert.equal(diagnostic.cli_model_variant, 'low');
  } finally { await f.close(); }
});

test('native permission event reaches the official SDK reply with its required receiver', async () => {
  const f = await fixture({ permissionEvent: { type: 'permission.asked', properties: {
    sessionID: 'fake-session-1', id: 'per_native', permission: 'bash' } } });
  try {
    const response = await invoke(f.adapter);
    assert.equal(response.status, 502);
    const diagnostic = JSON.parse((await readFile(join(f.outputDirectory, 'cli-invocations.jsonl'), 'utf8')).trim());
    assert.deepEqual(diagnostic.permission_rejections, [{ v2: false, sessionID: 'fake-session-1', requestID: 'per_native',
      permission: 'bash', reply: 'reject', succeeded: true }]);
    assert.deepEqual(f.permissionCalls, [{ path: { id: 'fake-session-1', permissionID: 'per_native' },
      query: { directory: f.root }, body: { response: 'reject' } }]);
  } finally { await f.close(); }
});

test('provider retry reports bridge 503 separately from unavailable upstream status', async () => {
  const f = await fixture({ actions: [], retryEvent: { type: 'session.status', properties: {
    sessionID: 'fake-session-1', status: { type: 'retry', attempt: 2, message: 'Upstream request failed: Endpoint is unavailable.' }
  } } });
  try {
    const response = await invoke(f.adapter); const body = await response.json();
    assert.equal(response.status, 503);
    assert.equal(body.error.code, 'provider_retry');
    assert.equal(body.error.status_scope, 'bridge');
    assert.equal(body.error.bridge_status_code, 503);
    assert.equal(body.error.provider_status_code, null);
    assert.equal(body.error.upstream_http_status, null);
    assert.equal(body.error.provider_retryable, true);
    assert.equal(body.error.retry_origin, 'opencode_session_status');
    assert.equal(body.error.provider_message, 'Upstream request failed: Endpoint is unavailable.');
    const diagnostic = JSON.parse((await readFile(join(f.outputDirectory, 'cli-invocations.jsonl'), 'utf8')).trim());
    assert.equal(diagnostic.provider_error_provenance.provider_id, 'opencode');
    assert.equal(diagnostic.provider_error_provenance.status_scope, 'bridge');
    assert.equal(diagnostic.provider_error_provenance.upstream_http_status, null);
    assert.equal(diagnostic.provider_error_provenance.provider_status_code, null);
    assert.equal(diagnostic.provider_error_provenance.bridge_status_code, 503);
  } finally { await f.close(); }
});

test('terminal boundary preserves every action in a multi-action step', async () => {
  const actions = [1, 2].map(value => ({ name: 'probe_tool', arguments: { value } }));
  const f = await fixture({ actions });
  try {
    const response = await invoke(f.adapter); const body = await response.json();
    assert.equal(response.status, 200);
    assert.deepEqual(body.choices[0].message.tool_calls.map(call => JSON.parse(call.function.arguments)), [{ value: 1 }, { value: 2 }]);
    assert.equal(f.abortCalls.length, 1);
    const diagnostic = JSON.parse((await readFile(join(f.outputDirectory, 'cli-invocations.jsonl'), 'utf8')).trim());
    assert.equal(diagnostic.terminal_action_boundary.action_record_count, 2);
    assert.equal(diagnostic.cli_tool_use_audit.filter(use => use.bridge).length, 2);
  } finally { await f.close(); }
});

test('official CLI no-op invalid Natlang attempt does not block an audited MCP action', async () => {
  const error = "Model tried to call unavailable tool 'probe_tool'. Available tools: bash, invalid.";
  const invalidEvent = { type: 'tool_use', sessionID: 'fake-session-1', part: { type: 'tool', tool: 'invalid',
    callID: 'call-invalid', state: { status: 'completed', title: 'Invalid Tool',
      input: { tool: 'probe_tool', error },
      output: `The arguments provided to the tool are invalid: ${error}`, metadata: { truncated: false },
      time: { start: 10, end: 11 } } } };
  const f = await fixture({ extraEvents: [invalidEvent] });
  try {
    const response = await invoke(f.adapter, { toolChoice: 'required' }); const body = await response.json();
    assert.equal(response.status, 200);
    assert.equal(body.choices[0].finish_reason, 'tool_calls');
    assert.equal(body.choices[0].message.tool_calls.length, 1);
    assert.equal(f.abortCalls.length, 1, 'the completed audited action still reaches the terminal abort');
    const diagnostic = JSON.parse((await readFile(join(f.outputDirectory, 'cli-invocations.jsonl'), 'utf8')).trim());
    assert.equal(diagnostic.non_bridge_tool_use_count, 0);
    assert.equal(diagnostic.rejected_natlang_tool_attempts.length, 1);
    assert.equal(diagnostic.rejected_natlang_tool_attempts[0].rejected_tool_name, 'probe_tool');
    assert.equal(diagnostic.terminal_action_boundary.status, 'aborted');
  } finally { await f.close(); }
});

test('official CLI no-op invalid tool attempt alone cannot satisfy required action', async () => {
  const error = "Model tried to call unavailable tool 'probe_tool'. Available tools: bash, invalid.";
  const invalidEvent = { type: 'tool_use', sessionID: 'fake-session-1', part: { type: 'tool', tool: 'invalid',
    callID: 'call-invalid', state: { status: 'completed', title: 'Invalid Tool',
      input: { tool: 'probe_tool', error },
      output: `The arguments provided to the tool are invalid: ${error}`, metadata: { truncated: false },
      time: { start: 10, end: 11 } } } };
  const f = await fixture({ actions: [], extraEvents: [invalidEvent] });
  try {
    const response = await invoke(f.adapter, { toolChoice: 'required' }); const body = await response.json();
    assert.equal(response.status, 502);
    assert.equal(body.error.code, 'MISSING_REQUIRED_ACTION');
    assert.equal(f.abortCalls.length, 0, 'a rejected invalid attempt is not an executed action');
  } finally { await f.close(); }
});

test('terminal action boundary does not hide non-bridge tool use or failed CLI exit', async () => {
  const nonBridge = await fixture({ extraEvents: [{ type: 'tool_use', sessionID: 'fake-session-1',
    part: { type: 'tool', tool: 'read', callID: 'native-1', state: { status: 'completed' } } }] });
  try {
    const response = await invoke(nonBridge.adapter); const body = await response.json();
    assert.equal(response.status, 502);
    assert.equal(body.error.code, 'NON_BRIDGE_TOOL_USE');
    assert.equal(nonBridge.abortCalls.length, 0);
  } finally { await nonBridge.close(); }

  const failed = await fixture({ exitCode: 7 });
  try {
    const response = await invoke(failed.adapter); const body = await response.json();
    assert.equal(response.status, 502);
    assert.equal(body.error.code, 'CLI_EXIT');
    assert.equal(failed.abortCalls.length, 1);
  } finally { await failed.close(); }
});

test('audited action does not hide a provider error event from the official CLI', async () => {
  const f = await fixture({ extraEvents: [{ type: 'error', error: { name: 'ProviderError', message: 'upstream failed' } }] });
  try {
    const response = await invoke(f.adapter); const body = await response.json();
    assert.equal(response.status, 502);
    assert.equal(body.error.code, 'CLI_EVENT_ERROR');
    assert.equal(f.abortCalls.length, 1);
  } finally { await f.close(); }
});

test('only the exact same-session MessageAbortedError from a confirmed terminal abort is suppressed', async () => {
  const f = await fixture({ abortError: { name: 'ProviderError', message: 'upstream failed' } });
  try {
    const response = await invoke(f.adapter); const body = await response.json();
    assert.equal(response.status, 502);
    assert.equal(body.error.code, 'SESSION_ERROR');
    const diagnostic = JSON.parse((await readFile(join(f.outputDirectory, 'cli-invocations.jsonl'), 'utf8')).trim());
    assert.deepEqual(diagnostic.session_errors.map(error => error.name), ['ProviderError']);
  } finally { await f.close(); }
});

test('standard nested Aborted error is ignored only for the confirmed terminal-action session', async () => {
  const expected = await fixture({ extraEvents: [{ type: 'error', sessionID: 'fake-session-1',
    error: { name: 'MessageAbortedError', data: { message: 'Aborted' } } }] });
  try {
    const response = await invoke(expected.adapter); const body = await response.json();
    assert.equal(response.status, 200);
    assert.equal(body.choices[0].message.tool_calls.length, 1);
    const diagnostic = JSON.parse((await readFile(join(expected.outputDirectory, 'cli-invocations.jsonl'), 'utf8')).trim());
    assert.equal(diagnostic.ignored_confirmed_terminal_abort_events, 1);
    assert.deepEqual(diagnostic.cli_event_errors.map(error => error.message), ['Aborted']);
  } finally { await expected.close(); }

  for (const event of [
    { type: 'error', sessionID: 'other-session', error: { name: 'MessageAbortedError', data: { message: 'Aborted' } } },
    { type: 'error', sessionID: 'fake-session-1', error: { name: 'MessageAbortedError', data: { message: 'not aborted' } } }
  ]) {
    const f = await fixture({ extraEvents: [event] });
    try {
      const response = await invoke(f.adapter); const body = await response.json();
      assert.equal(response.status, 502);
      assert.equal(body.error.code, 'CLI_EVENT_ERROR');
    } finally { await f.close(); }
  }
});

test('same-session session.error may arrive while confirmed terminal abort is pending', async () => {
  const f = await fixture({ abortError: { name: 'MessageAbortedError', data: { message: 'Aborted' } }, abortDelayMs: 50 });
  try {
    const response = await invoke(f.adapter); const body = await response.json();
    assert.equal(response.status, 200);
    assert.equal(body.choices[0].message.tool_calls.length, 1);
    const diagnostic = JSON.parse((await readFile(join(f.outputDirectory, 'cli-invocations.jsonl'), 'utf8')).trim());
    assert.equal(diagnostic.terminal_action_boundary.status, 'aborted');
    assert.deepEqual(diagnostic.session_errors.map(error => error.message), ['Aborted']);
  } finally { await f.close(); }
});

test('unconfirmed abort never admits an audited action, even with the expected abort error event', async () => {
  const f = await fixture({ abortDelayMs: 50, abortConfirmed: false });
  try {
    const response = await invoke(f.adapter); const body = await response.json();
    assert.equal(response.status, 502);
    assert.equal(body.error.code, 'SESSION_ERROR');
    assert.equal(f.abortCalls.length, 1);
    const diagnostic = JSON.parse((await readFile(join(f.outputDirectory, 'cli-invocations.jsonl'), 'utf8')).trim());
    assert.equal(diagnostic.terminal_action_boundary.status, 'abort_not_confirmed');
    assert.deepEqual(diagnostic.session_errors.map(error => error.name), ['MessageAbortedError']);
  } finally { await f.close(); }
});

test('timeout aborts the observed server session and waits for idle status', async () => {
  const f = await fixture({ actions: [], timeoutMs: 100 });
  // Replace the normal fake with an emitter that announces a session, then never finishes.
  await writeFile(join(f.root, 'fake-cli.mjs'), `#!/usr/bin/env node\nprocess.stdout.write(JSON.stringify({type:'step_start',sessionID:'fake-session-1',part:{type:'step-start',sessionID:'fake-session-1'}})+'\\n'); setInterval(()=>{},1000);\n`, { mode: 0o700 });
  await chmod(join(f.root, 'fake-cli.mjs'), 0o700);
  try {
    const response = await invoke(f.adapter); const body = await response.json();
    assert.equal(response.status, 504);
    assert.equal(body.error.code, 'REQUEST_TIMEOUT');
    assert.equal(f.abortCalls.length, 1);
    assert.deepEqual(f.abortCalls[0], { path: { id: 'fake-session-1' }, query: { directory: f.root } });
    const diagnostic = JSON.parse((await readFile(join(f.outputDirectory, 'cli-invocations.jsonl'), 'utf8')).trim());
    assert.equal(diagnostic.server_session_cleanup_reason, 'timeout');
    assert.deepEqual(diagnostic.server_session_cleanup.map(x => x.status), ['idle']);
  } finally { await f.close(); }
});

test('session cleanup aborts retry states using the SDK path/query contract', async () => {
  let state = 'retry'; const calls = [];
  const client = { session: {
    status: async options => ({ data: { 'session-retry': { type: state } } }),
    abort: async options => { calls.push(options); state = 'idle'; return { data: true }; }
  } };
  const result = await abortOpenCodeSession(client, 'session-retry', '/tmp/fake-session');
  assert.deepEqual(result, { ok: true, status: 'idle', abort_confirmed: true });
  assert.deepEqual(calls, [{ path: { id: 'session-retry' }, query: { directory: '/tmp/fake-session' } }]);
});
