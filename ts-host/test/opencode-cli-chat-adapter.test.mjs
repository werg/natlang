import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { createServer } from 'node:http';
import { createOpenCodeCliChatAdapter } from '../../scripts/opencode-cli-chat-adapter.mjs';

const repository = resolve(import.meta.dirname, '../..');

test('CLI adapter maps a real local MCP record and cancels the idle SSE reader', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'natlang-opencode-cli-adapter-'));
  const scratch = join(temp, 'scratch');
  const output = join(temp, 'out');
  await mkdir(scratch); await mkdir(output);
  const log = join(output, 'action-mcp-calls.jsonl');
  await writeFile(log, '');
  const fakeCli = join(temp, 'fake-opencode');
  const mcp = resolve(repository, 'scripts/opencode-natlang-action-mcp-server.mjs');
  await writeFile(fakeCli, `#!/usr/bin/env node\nimport { spawnSync } from 'node:child_process';\nconst input = [\n JSON.stringify({jsonrpc:'2.0',id:1,method:'initialize',params:{protocolVersion:'2025-11-25'}}),\n JSON.stringify({jsonrpc:'2.0',method:'notifications/initialized'}),\n JSON.stringify({jsonrpc:'2.0',id:2,method:'tools/call',params:{name:'submit_action',arguments:{name:'diagnostic_record',arguments:{value:'LOCAL_MCP_OK'}}}})\n].join('\\n')+'\\n';\nconst result=spawnSync(process.execPath,[${JSON.stringify(mcp)}],{input,encoding:'utf8',env:process.env});\nif(result.status!==0) process.exit(result.status??1);\nprocess.stdout.write(JSON.stringify({type:'text',part:{text:'intermediate',messageID:'msg-before'},sessionID:'fake'})+'\\n');\nprocess.stdout.write(JSON.stringify({type:'step_start',part:{type:'step-start'}})+'\\n');\nprocess.stdout.write(JSON.stringify({type:'step_finish',part:{type:'step-finish',tokens:{total:11,input:7,output:3,reasoning:1,cache:{read:0,write:0}}}})+'\\n');\nprocess.stdout.write(JSON.stringify({type:'text',part:{text:JSON.stringify({content:'ready',toolCalls:[]}),messageID:'msg-final'},sessionID:'fake'})+'\\n');\n`);
  const { chmod } = await import('node:fs/promises'); await chmod(fakeCli, 0o755);
  const permissionReplies = [];
  const client = {
    mcp: { status: async () => ({ data: { natlang_action_bridge: { status: 'connected' } } }),
      connect: async () => ({ data: true }) },
    tool: { ids: async () => ({ data: ['read', 'bash'] }) },
    permission: { reply: async args => { permissionReplies.push(args); return { data: true }; } },
    session: { permission: { reply: async args => { permissionReplies.push(args); return {}; } },
      abort: async () => ({ data: true }) }
  };
  const eventServer = createServer((req, res) => {
    if (req.url.startsWith('/event')) { res.writeHead(200, { 'content-type': 'text/event-stream' }); res.write(': ready\n\n'); return; }
    res.writeHead(404); res.end();
  });
  await new Promise(resolveListen => eventServer.listen(0, '127.0.0.1', resolveListen));
  const eventPort = eventServer.address().port;
  const adapter = await createOpenCodeCliChatAdapter({ cliPath: fakeCli, client,
    baseUrl: `http://127.0.0.1:${eventPort}`, directory: scratch, outputDirectory: output,
    actionLogPath: log, modelID: 'ling-3.1-flash-free', env: { ...process.env, NATLANG_OPENCODE_ACTION_LOG: log, NATLANG_OPENCODE_MCP_HANDSHAKE_LOG: join(output, 'mcp-handshake.jsonl') }, maxRequestMs: 3000, timeoutMs: 3000 });
  try {
    const response = await fetch(`${adapter.url}/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'opencode/ling-3.1-flash-free', messages: [{ role: 'user', content: 'record diagnostic' }],
        tools: [{ type: 'function', function: { name: 'diagnostic_record', parameters: { type: 'object' } } }] }) });
    const responseText = await response.text();
    assert.equal(response.status, 200, responseText);
    const completion = JSON.parse(responseText);
    assert.equal(completion.choices[0].finish_reason, 'tool_calls');
    assert.equal(completion.choices[0].message.tool_calls[0].function.name, 'diagnostic_record');
    assert.deepEqual(JSON.parse(completion.choices[0].message.tool_calls[0].function.arguments), { value: 'LOCAL_MCP_OK' });
    assert.equal(completion.raw_response.action_record_count, 1);
    assert.equal(completion.raw_response.provider_step_telemetry.finished, 1);
    assert.equal(completion.raw_response.provider_step_telemetry.tokens.total, 11);
    assert.equal(completion.raw_response.provider_request_count, null);
    assert.deepEqual(permissionReplies, []);
    assert.equal((await readFile(join(output, 'cli-stdout.raw'), 'utf8')).includes('LOCAL_MCP_OK'), false);
  } finally {
    await adapter.close();
    eventServer.closeAllConnections();
    await new Promise(resolveClose => eventServer.close(resolveClose));
  }
});


test('CLI adapter rejects a native permission request and stops the child', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'natlang-opencode-cli-permission-'));
  const scratch = join(temp, 'scratch'), output = join(temp, 'out');
  await mkdir(scratch); await mkdir(output);
  const log = join(output, 'action-mcp-calls.jsonl'); await writeFile(log, '');
  const fakeCli = join(temp, 'fake-opencode');
  await writeFile(fakeCli, '#!/usr/bin/env node\nprocess.on("SIGINT",()=>process.exit(0)); setInterval(()=>{},1000);\n');
  const { chmod } = await import('node:fs/promises'); await chmod(fakeCli, 0o755);
  const permissionReplies = [];
  const client = {
    mcp: { status: async () => ({ data: { natlang_action_bridge: { status: 'connected' } } }), connect: async () => ({ data: true }) },
    tool: { ids: async () => ({ data: ['read', 'bash'] }) },
    permission: { reply: async args => { permissionReplies.push(args); return { error: { message: 'reply transport failed' } }; } },
    session: { permission: { reply: async args => { permissionReplies.push(args); return {}; } }, abort: async () => ({ data: true }) }
  };
  const eventServer = createServer((req, res) => {
    if (req.url.startsWith('/event')) {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      res.write(`data: ${JSON.stringify({ type: 'permission.asked', properties: { id: 'permission-1', sessionID: 'session-1', permission: 'bash' } })}\n\n`);
      return;
    }
    res.writeHead(404); res.end();
  });
  await new Promise(resolveListen => eventServer.listen(0, '127.0.0.1', resolveListen));
  const adapter = await createOpenCodeCliChatAdapter({ cliPath: fakeCli, client,
    baseUrl: `http://127.0.0.1:${eventServer.address().port}`, directory: scratch, outputDirectory: output,
    actionLogPath: log, modelID: 'ling-3.1-flash-free', env: process.env, maxRequestMs: 4000, timeoutMs: 4000 });
  try {
    const response = await fetch(`${adapter.url}/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'opencode/ling-3.1-flash-free', messages: [{ role: 'user', content: 'do not run a tool' }], tools: [] }) });
    assert.equal(response.status, 502);
    const body = await response.json(); assert.equal(body.error.code, 'native_permission');
    assert.equal(permissionReplies.length, 1); assert.equal(permissionReplies[0].reply, 'reject');
    const attempts = (await readFile(join(output, 'cli-invocations.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
    assert.equal(attempts.length, 1); assert.equal(attempts[0].permission_rejections[0].succeeded, false);
    assert.match(attempts[0].permission_rejections[0].error, /reply transport failed/);
  } finally {
    await adapter.close(); eventServer.closeAllConnections();
    await new Promise(resolveClose => eventServer.close(resolveClose));
  }
});

test('CLI adapter rejects a non-bridge tool-use event and records its provenance', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'natlang-opencode-cli-tool-audit-'));
  const scratch = join(temp, 'scratch'), output = join(temp, 'out');
  await mkdir(scratch); await mkdir(output);
  const log = join(output, 'action-mcp-calls.jsonl'); await writeFile(log, '');
  const fakeCli = join(temp, 'fake-opencode');
  await writeFile(fakeCli, `#!/usr/bin/env node
process.stdout.write(JSON.stringify({type:'tool_use',part:{type:'tool',tool:'bash',callID:'call-native'}})+'\\n');
process.stdout.write(JSON.stringify({type:'text',part:{text:JSON.stringify({content:'done',toolCalls:[]}),messageID:'msg-final'}})+'\\n');
`);
  const { chmod } = await import('node:fs/promises'); await chmod(fakeCli, 0o755);
  const client = {
    mcp: { status: async () => ({ data: { natlang_action_bridge: { status: 'connected' } } }), connect: async () => ({ data: true }) },
    tool: { ids: async () => ({ data: ['read', 'bash'] }) },
    permission: { reply: async () => ({ data: true }) },
    session: { permission: { reply: async () => ({}) }, abort: async () => ({ data: true }) }
  };
  const eventServer = createServer((req, res) => {
    if (req.url.startsWith('/event')) { res.writeHead(200, { 'content-type': 'text/event-stream' }); res.write(': ready\\n\\n'); return; }
    res.writeHead(404); res.end();
  });
  await new Promise(resolveListen => eventServer.listen(0, '127.0.0.1', resolveListen));
  const adapter = await createOpenCodeCliChatAdapter({ cliPath: fakeCli, client,
    baseUrl: `http://127.0.0.1:${eventServer.address().port}`, directory: scratch, outputDirectory: output,
    actionLogPath: log, modelID: 'ling-3.1-flash-free', env: process.env, maxRequestMs: 3000, timeoutMs: 3000 });
  try {
    const response = await fetch(`${adapter.url}/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'opencode/ling-3.1-flash-free', messages: [{ role: 'user', content: 'return text' }], tools: [] }) });
    assert.equal(response.status, 502);
    assert.equal((await response.json()).error.code, 'NON_BRIDGE_TOOL_USE');
    const attempt = JSON.parse((await readFile(join(output, 'cli-invocations.jsonl'), 'utf8')).trim());
    assert.deepEqual(attempt.cli_tool_use_audit, [{ event_type: 'tool_use', part_type: 'tool', name: 'bash', call_id: 'call-native', bridge: false }]);
  } finally {
    await adapter.close(); eventServer.closeAllConnections();
    await new Promise(resolveClose => eventServer.close(resolveClose));
  }
});

test('CLI adapter keeps step and bridge-tool telemetry when a provider retry aborts the turn', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'natlang-opencode-cli-retry-audit-'));
  const scratch = join(temp, 'scratch'), output = join(temp, 'out');
  await mkdir(scratch); await mkdir(output);
  const log = join(output, 'action-mcp-calls.jsonl'); await writeFile(log, '');
  const fakeCli = join(temp, 'fake-opencode');
  await writeFile(fakeCli, `#!/usr/bin/env node
process.on('SIGINT',()=>process.exit(0));
for (const e of [
 {type:'step_start',part:{type:'step-start'}},
 {type:'tool_use',part:{type:'tool',tool:'natlang_action_bridge_submit_action',callID:'call-bridge'}},
 {type:'step_finish',part:{type:'step-finish',tokens:{total:17,input:12,output:4,reasoning:1,cache:{read:0,write:0}}}}
]) process.stdout.write(JSON.stringify(e)+'\\n');
setInterval(()=>{},1000);
`);
  const { chmod } = await import('node:fs/promises'); await chmod(fakeCli, 0o755);
  const client = { mcp: { status: async () => ({ data: { natlang_action_bridge: { status: 'connected' } } }), connect: async () => ({ data: true }) },
    tool: { ids: async () => ({ data: ['read', 'bash'] }) }, permission: { reply: async () => ({ data: true }) },
    session: { permission: { reply: async () => ({}) }, abort: async () => ({ data: true }) } };
  const eventServer = createServer((req, res) => {
    if (req.url.startsWith('/event')) {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      setTimeout(() => res.write(`data: ${JSON.stringify({ type: 'session.status', properties: { sessionID: 'fake-session', status: { type: 'retry', attempt: 1, message: 'endpoint unavailable' } } })}\n\n`), 40);
      return;
    }
    res.writeHead(404); res.end();
  });
  await new Promise(resolveListen => eventServer.listen(0, '127.0.0.1', resolveListen));
  const adapter = await createOpenCodeCliChatAdapter({ cliPath: fakeCli, client,
    baseUrl: `http://127.0.0.1:${eventServer.address().port}`, directory: scratch, outputDirectory: output,
    actionLogPath: log, modelID: 'ling-3.1-flash-free', env: process.env, maxRequestMs: 3000, timeoutMs: 3000 });
  try {
    const response = await fetch(`${adapter.url}/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'opencode/ling-3.1-flash-free', messages: [{ role: 'user', content: 'return one tool action' }], tools: [] }) });
    assert.equal(response.status, 503);
    assert.equal((await response.json()).error.code, 'provider_retry');
    const attempt = JSON.parse((await readFile(join(output, 'cli-invocations.jsonl'), 'utf8')).trim());
    assert.equal(attempt.provider_step_telemetry.started, 1);
    assert.equal(attempt.provider_step_telemetry.finished, 1);
    assert.equal(attempt.provider_step_telemetry.tokens.total, 17);
    assert.equal(attempt.cli_tool_use_audit[0].name, 'natlang_action_bridge_submit_action');
    assert.equal(attempt.cli_tool_use_audit[0].bridge, true);
    assert.equal(attempt.provider_retries[0].message, 'endpoint unavailable');
  } finally {
    await adapter.close(); eventServer.closeAllConnections();
    await new Promise(resolveClose => eventServer.close(resolveClose));
  }
});
