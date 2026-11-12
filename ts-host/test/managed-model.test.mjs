import test from 'node:test';
import assert from 'node:assert/strict';
import { chmodSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createManagedModelSession, inspectLlamaServer, executorIdentityForChoice } from '../dist/model/index.js';

test('executor identities redact credentials and fingerprint behavior headers', () => {
  const choice = { kind: 'external', model: 'deployment-v1', endpoint: 'https://user:password@example.test/v1?access_token=private',
    apiKeyEnv: 'PRIVATE_KEY', headers: { Authorization: 'Bearer private', 'X-Route': 'blue' },
    request: { temperature: 0.1, access_token: 'private', nested: { refreshToken: 'private' } } };
  const identity = executorIdentityForChoice(choice);
  assert.equal(identity.configuration.endpoint, 'https://example.test/v1');
  assert.equal(identity.configuration.request.temperature, 0.1);
  assert.ok(!JSON.stringify(identity).includes('private'));
  assert.ok(!JSON.stringify(identity).includes('password'));
  assert.deepEqual(executorIdentityForChoice({ ...choice, headers: { ...choice.headers, Authorization: 'Bearer rotated' } }), identity);
  assert.notEqual(executorIdentityForChoice({ ...choice, headers: { ...choice.headers, 'X-Route': 'green' } }).configuration.headersHash,
    identity.configuration.headersHash);
});

test('managed model startup can be prepared eagerly and its child is closed with the session', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-managed-model-'));
  const executable = join(root, 'fake-llama-server'), marker = join(root, 'events'), model = join(root, 'model.gguf');
  const template = join(root, 'template.jinja');
  writeFileSync(model, 'fixture'); writeFileSync(template, 'fixture');
  writeFileSync(executable, `#!${process.execPath}
if (process.argv.includes('--version')) { console.log('version: 0.8.0 (build 12069, commit a0c92ce36)'); process.exit(0); }
const http = require('node:http'), fs = require('node:fs');
const args = process.argv.slice(2), port = Number(args[args.indexOf('--port') + 1]);
fs.appendFileSync(${JSON.stringify(marker)}, 'started\\n');
const server = http.createServer((request, response) => {
  response.setHeader('content-type', 'application/json');
  if (request.url === '/health') response.end(JSON.stringify({ status: 'ok' }));
  else { let body = ''; request.on('data', chunk => body += chunk); request.on('end', () =>
    response.end(JSON.stringify({ choices: [{ message: { content: 'ready', tool_calls: [] } }], usage: { completion_tokens: 1 } }))); }
});
server.listen(port, '127.0.0.1');
process.on('SIGTERM', () => { fs.appendFileSync(${JSON.stringify(marker)}, 'stopped\\n'); server.close(() => process.exit(0)); });
`);
  chmodSync(executable, 0o755);
  const environment = { ...process.env, NATLANG_LLAMA_SERVER: executable, NATLANG_SHARED_SERVER: 'off',
    NATLANG_MODEL_PATH: model, NATLANG_TEMPLATE: template, NATLANG_MODEL_START_TIMEOUT_MS: '5000' };
  const session = createManagedModelSession({}, environment);
  assert.equal(existsSync(marker), false);
  const prepared = await session.prepare();
  assert.equal(prepared.running, true);
  assert.equal(readFileSync(marker, 'utf8').trim(), 'started');
  const result = await session.turn({ messages: [], tools: [], temperature: 0, seed: 1, max_tokens: 4 });
  assert.equal(result.text, 'ready');
  assert.equal(session.status().running, true);
  await session.close();
  assert.deepEqual(readFileSync(marker, 'utf8').trim().split('\n'), ['started', 'stopped']);
  await assert.rejects(session.prepare(), /model session is closed/);
});

test('configured endpoints stay externally owned', async () => {
  const session = createManagedModelSession({ endpoint: 'http://127.0.0.1:9', model: 'remote' }, process.env);
  assert.equal(session.status().source, 'external');
  assert.equal((await session.prepare()).source, 'external');
  await session.close();
});

test('a host can grant managed runtime installation at the first semantic turn', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-managed-consent-'));
  const executable = join(root, 'fake-llama-server'), marker = join(root, 'events'), model = join(root, 'model.gguf');
  const template = join(root, 'template.jinja');
  writeFileSync(model, 'fixture'); writeFileSync(template, 'fixture');
  writeFileSync(executable, `#!${process.execPath}
if (process.argv.includes('--version')) { console.log('version: 0.8.0 (build 12069, commit a0c92ce36)'); process.exit(0); }
const http = require('node:http'), fs = require('node:fs');
const args = process.argv.slice(2), port = Number(args[args.indexOf('--port') + 1]);
fs.appendFileSync(${JSON.stringify(marker)}, 'started\\n');
const server = http.createServer((_request, response) => response.end(JSON.stringify({ choices: [{ message: { content: 'ready', tool_calls: [] } }] })));
server.listen(port, '127.0.0.1');
process.on('SIGTERM', () => server.close(() => process.exit(0)));
`);
  chmodSync(executable, 0o755);
  const environment = { ...process.env, PATH: '', NATLANG_RUNTIME_HOME: join(root, 'runtime'), NATLANG_SHARED_SERVER: 'off',
    NATLANG_MODEL_PATH: model, NATLANG_TEMPLATE: template, NATLANG_MODEL_START_TIMEOUT_MS: '5000' };
  let requests = 0;
  const session = createManagedModelSession({}, environment, process.stderr, { ensureRuntime: async discovery => {
    requests++; assert.equal(discovery.selected, null);
    return inspectLlamaServer(executable, 'managed', undefined, environment);
  } });
  assert.equal(requests, 0);
  await session.turn({ messages: [], tools: [], temperature: 0, seed: 1, max_tokens: 4 });
  assert.equal(requests, 1); assert.equal(existsSync(marker), true);
  await session.close();
});

test('tuned soft states load as native state spans and requests carry them as native_state parts', async () => {
  const { nativeStateSentinel } = await import('../dist/native/neuralese.js');
  const { nativeStateMessages } = await import('../dist/model/openai-compatible.js');
  const state = 'nzs1_' + 'a'.repeat(52), other = 'nzs1_' + 'b'.repeat(52);
  const messages = [{ role: 'system', content: `Rules: ${nativeStateSentinel(state)} end` }, { role: 'user', content: 'plain' }];
  assert.deepEqual(nativeStateMessages(messages, new Set([state]))[0].content,
    [{ type: 'text', text: 'Rules: ' }, { type: 'native_state', id: state }, { type: 'text', text: ' end' }]);
  assert.equal(nativeStateMessages(messages, new Set([state]))[1], messages[1], 'text without states is unchanged');
  assert.throws(() => nativeStateMessages(messages, new Set([other])), /weights-unavailable: native state/);

  const fake = (supports) => {
    const root = mkdtempSync(join(tmpdir(), 'natlang-native-state-'));
    const executable = join(root, 'fake-llama-server'), log = join(root, 'log'), model = join(root, 'model.gguf'), template = join(root, 'template.jinja');
    writeFileSync(model, 'fixture'); writeFileSync(template, 'fixture');
    writeFileSync(executable, `#!${process.execPath}
if (process.argv.includes('--version')) { console.log('version: 0.8.0 (build 12069, commit a0c92ce36)'); process.exit(0); }
if (process.argv.includes('--help')) { console.log(${JSON.stringify(supports ? '--lora-per-sequence\n--native-state FNAME' : '--lora')}); process.exit(0); }
const http = require('node:http'), fs = require('node:fs');
const args = process.argv.slice(2), port = Number(args[args.indexOf('--port') + 1]);
fs.appendFileSync(${JSON.stringify(log)}, JSON.stringify({ args }) + '\\n');
const server = http.createServer((request, response) => {
  response.setHeader('content-type', 'application/json');
  if (request.url === '/health') return response.end(JSON.stringify({ status: 'ok' }));
  let body = ''; request.on('data', chunk => body += chunk); request.on('end', () => {
    fs.appendFileSync(${JSON.stringify(log)}, JSON.stringify({ body: JSON.parse(body) }) + '\\n');
    response.end(JSON.stringify({ choices: [{ message: { content: 'ok', tool_calls: [] } }], usage: { completion_tokens: 1 } })); });
});
server.listen(port, '127.0.0.1');
process.on('SIGTERM', () => server.close(() => process.exit(0)));
`);
    chmodSync(executable, 0o755);
    return { log, environment: { ...process.env, NATLANG_LLAMA_SERVER: executable, NATLANG_MODEL_PATH: model, NATLANG_TEMPLATE: template,
      NATLANG_MODEL_START_TIMEOUT_MS: '5000' } };
  };
  const fork = fake(true);
  const session = createManagedModelSession({}, fork.environment, process.stderr, { nativeStates: () => [{ id: state, path: '/release/exports/state.gguf' }] });
  await session.turn({ messages, tools: [], temperature: 0, seed: 1, max_tokens: 4 });
  await session.close();
  const records = readFileSync(fork.log, 'utf8').trim().split('\n').map(line => JSON.parse(line));
  const args = records.find(record => record.args).args;
  assert.equal(args[args.indexOf('--native-state') + 1], '/release/exports/state.gguf');
  assert.deepEqual(records.find(record => record.body).body.messages[0].content[1], { type: 'native_state', id: state });
  // A server without --native-state cannot serve soft states: the session refuses to start rather than run them as text.
  const upstream = fake(false);
  const refused = createManagedModelSession({}, upstream.environment, process.stderr, { nativeStates: () => [{ id: state, path: '/x.gguf' }] });
  await assert.rejects(refused.prepare(), /weights-unavailable: .* cannot import native state spans/);
  await refused.close();
});

test('managed sessions share one server per model: started once, kept while leased, stopped when unused', async () => {
  const { listSharedServers } = await import('../dist/model/shared-server.js');
  const root = mkdtempSync(join(tmpdir(), 'natlang-shared-model-'));
  const executable = join(root, 'fake-llama-server'), model = join(root, 'model.gguf'), template = join(root, 'template.jinja'), starts = join(root, 'starts');
  writeFileSync(model, 'fixture weights'); writeFileSync(template, 'fixture');
  writeFileSync(executable, `#!${process.execPath}
if (process.argv.includes('--version')) { console.log('version: 0.8.0 (build 12069, commit a0c92ce36)'); process.exit(0); }
if (process.env.FAKE_FAIL) { console.error('cannot load the model file'); process.exit(3); }
const args = process.argv.slice(2), port = Number(args[args.indexOf('--port') + 1]);
require('node:fs').appendFileSync(${JSON.stringify(starts)}, 'started\\n');
const s = require('node:http').createServer((q, r) => { r.setHeader('content-type', 'application/json');
  if (q.url === '/health') return r.end('{"status":"ok"}');
  if (q.url === '/slots') return r.end('[{"is_processing":false}]');
  let b = ''; q.on('data', c => b += c); q.on('end', () => r.end(JSON.stringify({ choices: [{ message: { content: 'shared', tool_calls: [] } }] }))); });
s.listen(port, '127.0.0.1');
process.on('SIGTERM', () => s.close(() => process.exit(0)));
`);
  chmodSync(executable, 0o755);
  const environment = { ...process.env, NATLANG_LLAMA_SERVER: executable, NATLANG_MODEL_PATH: model, NATLANG_TEMPLATE: template,
    NATLANG_STATE_HOME: join(root, 'state'), NATLANG_CACHE_HOME: join(root, 'cache'), NATLANG_MODEL_START_TIMEOUT_MS: '5000',
    NATLANG_SHARED_SERVER_IDLE_S: '1', NATLANG_WARM_PREFIXES: 'off' };
  const quiet = { write: () => true };
  const until = async (condition, what) => {
    for (let waited = 0; waited < 15000; waited += 100) { if (condition()) return; await new Promise(resolve => setTimeout(resolve, 100)); }
    assert.fail(`timed out waiting for ${what}`);
  };
  const first = createManagedModelSession({}, environment, quiet), second = createManagedModelSession({}, environment, quiet);
  const [a, b] = await Promise.all([first.prepare(), second.prepare()]);
  assert.equal(a.endpoint, b.endpoint);
  assert.equal(a.shared, true); assert.equal(b.shared, true);
  assert.deepEqual(readFileSync(starts, 'utf8').trim().split('\n'), ['started'], 'one server for both sessions');
  assert.equal((await first.turn({ messages: [], tools: [], temperature: 0, seed: 1, max_tokens: 4 })).text, 'shared');
  const [record] = listSharedServers(environment);
  assert.equal(record.endpoint, a.endpoint);
  // The server outlives the session that started it while another holds a lease, and a later session reuses it.
  await first.close();
  await new Promise(resolve => setTimeout(resolve, 2500));
  const third = createManagedModelSession({}, environment, quiet);
  assert.equal((await third.prepare()).endpoint, a.endpoint);
  assert.equal(readFileSync(starts, 'utf8').trim().split('\n').length, 1);
  await second.close(); await third.close();
  // Unused for the idle time: the supervisor stops it and withdraws its record.
  await until(() => listSharedServers(environment).length === 0, 'the unused server to stop');
  // Another model file is another server; a private session (NATLANG_SHARED_SERVER=off) starts its own.
  const priv = createManagedModelSession({}, { ...environment, NATLANG_SHARED_SERVER: 'off' }, quiet);
  assert.equal((await priv.prepare()).shared, undefined);
  await priv.close();
  // A start that fails reports the server's output to the session that waited for it.
  const failing = createManagedModelSession({}, { ...environment, FAKE_FAIL: '1' }, quiet);
  await assert.rejects(failing.prepare(), /exited during startup[\s\S]*cannot load the model file/);
  await failing.close();
});

test('a profile request refines the stock model request field by field', async () => {
  const { resolveModelChoice } = await import('../dist/model/config.js');
  const choice = resolveModelChoice({ local: { stock: 'lfm3-1.7b-q4_k_m' },
    request: { thinking_budget_tokens: 512, chat_template_kwargs: { enable_thinking: true } } });
  assert.deepEqual(choice.request, { temperature: 0, thinking_budget_tokens: 512,
    chat_template_kwargs: { preserve_thinking: true, enable_thinking: true } });
  assert.deepEqual(resolveModelChoice({}).request, { temperature: 0, chat_template_kwargs: { preserve_thinking: true } });
});
