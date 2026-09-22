import test from 'node:test';
import assert from 'node:assert/strict';
import { chmodSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createManagedModelSession, inspectLlamaServer } from '../dist/model/index.js';

test('managed model startup can be prepared eagerly and its child is closed with the session', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-managed-model-'));
  const executable = join(root, 'fake-llama-server'), marker = join(root, 'events'), model = join(root, 'model.gguf');
  const template = join(root, 'template.jinja');
  writeFileSync(model, 'fixture'); writeFileSync(template, 'fixture');
  writeFileSync(executable, `#!${process.execPath}
if (process.argv.includes('--version')) { console.log('version: 0.4.1 (build 10964, commit b29c606e2)'); process.exit(0); }
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
  const environment = { ...process.env, NATLANG_LLAMA_SERVER: executable,
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
if (process.argv.includes('--version')) { console.log('version: 0.4.1 (build 10964, commit b29c606e2)'); process.exit(0); }
const http = require('node:http'), fs = require('node:fs');
const args = process.argv.slice(2), port = Number(args[args.indexOf('--port') + 1]);
fs.appendFileSync(${JSON.stringify(marker)}, 'started\\n');
const server = http.createServer((_request, response) => response.end(JSON.stringify({ choices: [{ message: { content: 'ready', tool_calls: [] } }] })));
server.listen(port, '127.0.0.1');
process.on('SIGTERM', () => server.close(() => process.exit(0)));
`);
  chmodSync(executable, 0o755);
  const environment = { ...process.env, PATH: '', NATLANG_RUNTIME_HOME: join(root, 'runtime'),
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
