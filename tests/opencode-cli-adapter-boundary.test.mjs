import assert from 'node:assert/strict';
import { chmod, mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { abortOpenCodeSession, createOpenCodeCliChatAdapter } from '../scripts/opencode-cli-chat-adapter.mjs';

async function fixture({ actions = [{ name: 'probe_tool', arguments: { value: 1 } }], extraEvents = [], exitCode = 0, timeoutMs = 1500 } = {}) {
  const root = await mkdtemp(join(tmpdir(), 'natlang-opencode-boundary-'));
  const outputDirectory = join(root, 'output'); await mkdir(outputDirectory);
  const actionLogPath = join(root, 'actions.jsonl'); await writeFile(actionLogPath, '');
  const markerPath = join(root, 'abort.marker');
  const cliPath = join(root, 'fake-cli.mjs');
  const script = `#!/usr/bin/env node
import fs from 'node:fs';
const sid = 'fake-session-1';
const actions = JSON.parse(process.env.FAKE_TOOL_ACTIONS || '[]');
const send = event => process.stdout.write(JSON.stringify(event) + '\\n');
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
  const eventServer = createServer((req, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
    res.write(': connected\n\n');
  });
  await new Promise(resolve => eventServer.listen(0, '127.0.0.1', resolve));
  const eventAddress = eventServer.address();
  const aborted = new Set(); const abortCalls = []; const observedStatusArgs = [];
  const client = {
    mcp: { status: async () => ({ data: { natlang_action_bridge: { status: 'connected' } } }),
      connect: async () => ({ data: true }) },
    tool: { ids: async () => ({ data: ['read', 'write', 'bash'] }) },
    session: {
      abort: async options => {
        abortCalls.push(options); aborted.add(options.path.id);
        await writeFile(markerPath, 'aborted');
        return { data: true };
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
    env: { PATH: process.env.PATH, HOME: root, NATLANG_OPENCODE_ACTION_LOG: actionLogPath,
      FAKE_ABORT_MARKER: markerPath, FAKE_TOOL_ACTIONS: JSON.stringify(actions),
      FAKE_EXTRA_EVENTS: JSON.stringify(extraEvents), FAKE_EXIT_CODE: String(exitCode) } });
  return { root, adapter, eventServer, abortCalls, observedStatusArgs, actionLogPath, outputDirectory,
    async close() { await adapter.close(); eventServer.closeAllConnections(); await new Promise(resolve => eventServer.close(resolve)); await rm(root, { recursive: true, force: true }); } };
}

async function invoke(adapter) {
  return fetch(`${adapter.url}/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'fixture/free', messages: [{ role: 'user', content: 'Record the action.' }],
      tools: [{ type: 'function', function: { name: 'probe_tool', parameters: { type: 'object', properties: { value: { type: 'number' } } } } }] }) });
}

test('completed audited action stops only its session after the tool-calls step', async () => {
  const f = await fixture();
  try {
    const response = await invoke(f.adapter); const body = await response.json();
    assert.equal(response.status, 200);
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
