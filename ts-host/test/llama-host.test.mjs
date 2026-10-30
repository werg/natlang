import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createLlamaServer, ensureStockModel, localSlotPlan, localModelPrerequisites, managedModelSha256, discoverLlamaRuntime,
  stockModelPath, nodeLlamaHost } from '../dist/llama/index.js';
import { openAICompatibleModelTurn } from '../dist/model/openai-compatible.js';

// A host that touches nothing of the machine: its processes, network and log are the test's own.
function fixtureHost(root, { fetch, spawn } = {}) {
  const log = [], spawned = [];
  const model = join(root, 'model.gguf'), template = join(root, 'template.jinja');
  writeFileSync(model, 'fixture weights'); writeFileSync(template, 'fixture');
  const directories = { runtime: join(root, 'runtime'), models: join(root, 'models'), modelHashes: join(root, 'hashes'),
    warmPrefixes: join(root, 'warm'), state: join(root, 'state'), checkoutModels: null, templates: [root] };
  const processes = {
    spawn(command, args, options) {
      spawned.push({ command, args: [...args], options });
      return spawn?.(command, args, options) ?? { pid: 4242, hasExited: () => false, exited: new Promise(() => {}), kill() {} };
    },
    run(command, args) {
      if (args.includes('--version')) return { status: 0, stdout: 'version: 0.8.0 (build 12069, commit a0c92ce36)\n', stderr: '' };
      if (args.includes('--help')) return { status: 0, stdout: '--lora-per-sequence\n--native-state FNAME', stderr: '' };
      return { status: 1, stdout: '', stderr: `unexpected command ${command}` };
    },
    async freePort() { return 45678; },
  };
  const environment = { PATH: '', NATLANG_LLAMA_SERVER: join(root, 'llama-server'), NATLANG_MODEL_PATH: model, NATLANG_TEMPLATE: template,
    NATLANG_MODEL_START_TIMEOUT_MS: '2000' };
  writeFileSync(environment.NATLANG_LLAMA_SERVER, 'not run: the host probes it');
  const host = { environment, directories, processes, log: { write: text => log.push(text) },
    fetch: fetch ?? (async url => url.endsWith('/health') ? new Response('{"status":"ok"}') : new Response('', { status: 404 })) };
  return { host, log, spawned, model };
}

test('the package spawns through the host and sizes, arguments and warm directory come from the host\'s settings', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-llama-host-'));
  const requests = [];
  const { host, log, spawned } = fixtureHost(root, { fetch: async (url, init) => { requests.push(String(url)); return new Response('{"status":"ok"}'); } });
  const server = createLlamaServer(host, { local: { parallel: 3, contextTokens: 4096, args: ['--flash-attn', 'on'] } });
  assert.equal(server.status().running, false);
  const endpoint = await server.start();
  assert.equal(endpoint, 'http://127.0.0.1:45678');
  assert.equal(spawned.length, 1, 'one server however many callers');
  await server.start();
  assert.equal(spawned.length, 1);
  const { command, args } = spawned[0];
  assert.equal(command, host.environment.NATLANG_LLAMA_SERVER);
  const arg = name => args[args.indexOf(name) + 1];
  assert.equal(arg('--port'), '45678'); assert.equal(arg('--parallel'), '3'); assert.equal(arg('-c'), String(3 * 4096));
  assert.equal(arg('-m'), host.environment.NATLANG_MODEL_PATH); assert.equal(arg('--chat-template-file'), host.environment.NATLANG_TEMPLATE);
  assert.equal(arg('--slot-save-path'), host.directories.warmPrefixes, 'warm states live where the host says');
  assert.equal(arg('--flash-attn'), 'on', 'profile arguments are passed through');
  assert.ok(existsSync(host.directories.warmPrefixes));
  assert.deepEqual(requests, [endpoint + '/health']);
  assert.match(log.join(''), /starting managed model model\.gguf with 3 slots \(4096 tokens each; local\.parallel\)/);
  assert.equal(server.plan().slots, 3);
  assert.equal(server.status().running, true);
  await server.close();
  assert.equal(server.status().running, false);
  await assert.rejects(server.start(), /model session is closed/);
});

test('a host that supervises llama-server itself launches it; the package plans around it and stops it through the host', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-llama-launch-'));
  const { host, spawned } = fixtureHost(root);
  const launches = []; let stopped = 0;
  const server = createLlamaServer(host, { local: { parallel: 2 }, loraAdapters: () => [{ id: 'a'.repeat(8), path: '/a.gguf' }],
    launch: async (launch, signal) => { launches.push(launch); assert.equal(signal.aborted, false); return { stop: async () => { stopped++; } }; } });
  assert.equal(await server.start(), 'http://127.0.0.1:45678');
  assert.equal(spawned.length, 0, 'the package spawned nothing');
  assert.equal(launches.length, 1);
  assert.ok(launches[0].args.includes('--lora-per-sequence') && launches[0].args.includes('--lora'));
  assert.equal(server.loraIndex.get('a'.repeat(8)), 0);
  assert.equal(launches[0].args.includes('--slot-save-path'), false, 'adapters change the KV: no warm states');
  await server.close();
  assert.equal(stopped, 1);
  // closing while it starts aborts the launch the host is running
  let seen;
  const slow = createLlamaServer(host, { launch: (_launch, signal) => new Promise((_resolve, reject) => {
    seen = signal; signal.addEventListener('abort', () => reject(signal.reason ?? new Error('aborted')), { once: true }); }) });
  const starting = slow.start();
  starting.catch(() => undefined);
  await new Promise(wake => setTimeout(wake, 20));
  await slow.close();
  await assert.rejects(starting);
  assert.equal(seen.aborted, true);
});

test('a server that exits while starting fails the start with its error output', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-llama-fail-'));
  const { host } = fixtureHost(root, { spawn: (_c, _a, options) => {
    setTimeout(() => options.onStderr('error: unknown model architecture'), 5);
    let exited = false;
    const exit = new Promise(resolve => setTimeout(() => { exited = true; resolve(); }, 30));
    return { pid: 1, hasExited: () => exited, exited: exit, kill() {} };
  }, fetch: async () => { throw new Error('connection refused'); } });
  const server = createLlamaServer(host);
  await assert.rejects(server.start(), /exited during startup\nerror: unknown model architecture/);
  await server.close();
});

test('an attached server is planned and warmed through the host\'s fetch, with its credentials, and never started', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-llama-attach-'));
  const posts = [];
  const { host, spawned, log } = fixtureHost(root, { fetch: async (url, init) => {
    const path = String(url).replace('http://127.0.0.1:9000', '');
    posts.push({ path, method: init?.method ?? 'GET', authorization: new Headers(init?.headers).get('authorization'),
      body: init?.body ? JSON.parse(init.body) : undefined });
    if (path === '/slots') return new Response('[{"is_processing":false},{"is_processing":false}]');
    if (path === '/apply-template') {
      const { messages } = JSON.parse(init.body);
      return new Response(JSON.stringify({ prompt: `<|im_start|>system\n${messages[0].content}<|im_end|>\n<|im_start|>user\n${messages[1].content}<|im_end|>\n` }));
    }
    if (path === '/completion') return new Response(JSON.stringify({ timings: { prompt_n: 321 } }));
    if (path.startsWith('/slots/0?action=save')) {
      writeFileSync(join(host.directories.warmPrefixes, JSON.parse(init.body).filename), 'slot state'); return new Response('{}');
    }
    return new Response('{}');
  } });
  mkdirSync(host.directories.warmPrefixes, { recursive: true });
  const sha = createHash('sha256').update('supervised weights').digest('hex');
  const server = createLlamaServer(host, { headers: { authorization: 'Bearer secret' },
    attach: { endpoint: 'http://127.0.0.1:9000', slots: 2, contextPerSlot: 8192, modelSha256: sha, runtime: 'b12069ga0c92ce', warm: true } });
  assert.equal(await server.start(), 'http://127.0.0.1:9000');
  assert.equal(spawned.length, 0);
  assert.equal(server.plan().slots, 2); assert.equal(server.plan().totalContext, 16384);
  const request = { messages: [{ role: 'system', content: 'You run one call. '.repeat(40) }, { role: 'user', content: 'hi' }], tools: [] };
  const { endpoint, release } = await server.admit(request);
  assert.equal(endpoint, 'http://127.0.0.1:9000');
  release();
  const paths = posts.map(post => `${post.method} ${post.path}`);
  assert.deepEqual(paths, ['GET /slots', 'POST /apply-template', 'POST /apply-template', 'POST /slots/0?action=erase',
    'POST /completion', 'POST /slots/0?action=save', 'POST /slots/1?action=restore']);
  assert.ok(posts.every(post => post.authorization === 'Bearer secret'), 'every request to the server carries the host\'s credentials');
  const warmed = server.status().warm;
  assert.equal(warmed.length, 1); assert.equal(warmed[0].tokens, 321); assert.equal(warmed[0].restoredSlots, 2); assert.equal(warmed[0].created, true);
  assert.match(warmed[0].file, new RegExp(`-${sha.slice(0, 12)}-b12069ga0c92ce-p2x8192\\.slot$`), 'keyed by the model, the build and the slot layout');
  assert.ok(readdirSync(host.directories.warmPrefixes).some(name => name.endsWith('.slot')));
  assert.match(log.join(''), /prefilled the 321-token prompt opening/);
  // a second server over the same states restores instead of prefilling
  const again = createLlamaServer(host, { headers: { authorization: 'Bearer secret' },
    attach: { endpoint: 'http://127.0.0.1:9000', slots: 2, contextPerSlot: 8192, modelSha256: sha, runtime: 'b12069ga0c92ce', warm: true } });
  posts.length = 0;
  (await again.admit(request)).release();
  assert.equal(again.status().warm[0].created, false);
  assert.deepEqual(posts.map(post => post.path), ['/slots', '/slots/0?action=restore', '/slots/1?action=restore']);
  // a busy server is not warmed
  const busy = createLlamaServer({ ...host, fetch: async url => new Response(String(url).endsWith('/slots') ? '[{"is_processing":true}]' : '{}') },
    { attach: { endpoint: 'http://127.0.0.1:9000', slots: 1, contextPerSlot: 8192, modelSha256: sha, runtime: null, warm: true } });
  (await busy.admit(request)).release();
  assert.equal(busy.status().warm.length, 0);
  await server.close(); await again.close(); await busy.close();
});

test('decision scoring runs against a server the host supervises: the package\'s endpoint feeds the endpoint driver\'s scorer', async () => {
  const seen = [];
  const http = createServer((request, response) => {
    let body = ''; request.on('data', chunk => body += chunk);
    request.on('end', () => {
      seen.push({ url: request.url, authorization: request.headers.authorization, body: JSON.parse(body) });
      response.setHeader('content-type', 'application/json');
      const { items } = JSON.parse(body);
      response.end(JSON.stringify({ results: items.map(item => ({ log_probs: item.continuations.map((_c, index) => -index - 1), tokens: item.continuations.map(() => 2) })) }));
    });
  });
  await new Promise(resolve => http.listen(0, '127.0.0.1', resolve));
  try {
    const root = mkdtempSync(join(tmpdir(), 'natlang-llama-score-'));
    const { host } = fixtureHost(root);
    const server = createLlamaServer(host, { attach: { endpoint: `http://127.0.0.1:${http.address().port}`, slots: 1, contextPerSlot: 8192,
      modelSha256: 'a'.repeat(64), runtime: null } });
    const endpoint = await server.start();
    const driver = openAICompatibleModelTurn({ endpoint, model: 'm', apiKey: 'k', scoreEndpoint: true, scoreAll: true });
    const scores = await driver.decide({ messages: [{ role: 'user', content: 'q' }], options: ['"a"', '"b"'] });
    assert.deepEqual(scores.log_probs, [-1, -2]);
    assert.equal(seen[0].url, '/v1/natlang/score'); assert.equal(seen[0].authorization, 'Bearer k');
    assert.deepEqual(seen[0].body.items[0].continuations, ['"a"', '"b"']);
    await server.close();
  } finally { await new Promise(resolve => http.close(resolve)); }
});

test('a stock release with sliding-window layers is served with --swa-full; a configured model file is not', async () => {
  const bytes = Buffer.from('a tiny stand-in for a GGUF file');
  const release = { id: 'swa', label: 'SWA', family: 'fixture', file: 'swa.gguf', quant: 'Q', bytes: bytes.length,
    sha256: createHash('sha256').update(bytes).digest('hex'), contextTokens: 2048, kvBytesPerToken: 1000, fullSlidingWindowCache: true,
    downloadUrl: 'https://models.invalid/swa.gguf', template: 'template.jinja', source: { repository: 'x/y', revision: 'r' } };
  const serve = async (configured, stock) => {
    const { host, spawned } = fixtureHost(mkdtempSync(join(tmpdir(), 'natlang-llama-swa-')));
    if (!configured) { delete host.environment.NATLANG_MODEL_PATH; delete host.environment.NATLANG_TEMPLATE; }
    mkdirSync(host.directories.models, { recursive: true });
    writeFileSync(stockModelPath(host, release), bytes);
    const server = createLlamaServer(host, { release: stock, local: { parallel: 2 } });
    await server.start();
    await server.close();
    return spawned[0].args;
  };
  assert.ok((await serve(false, release)).includes('--swa-full'), 'the stock file needs the full window cache to reuse a cached prompt prefix');
  assert.equal((await serve(true, release)).includes('--swa-full'), false, 'a configured file is another model');
  assert.equal((await serve(false, { ...release, fullSlidingWindowCache: undefined })).includes('--swa-full'), false);
});

test('stock model download and verification go through the host\'s fetch, directories and log', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-llama-download-'));
  const bytes = Buffer.from('a tiny stand-in for a GGUF file');
  const release = { id: 'fixture', label: 'Fixture', family: 'fixture', file: 'fixture.gguf', quant: 'Q', bytes: bytes.length,
    sha256: createHash('sha256').update(bytes).digest('hex'), contextTokens: 2048, kvBytesPerToken: 1000,
    downloadUrl: 'https://models.invalid/fixture.gguf', template: 'fixture.jinja', source: { repository: 'x/y', revision: 'r' } };
  const fetched = [];
  const { host, log } = fixtureHost(root, { fetch: async (url, init) => { fetched.push(String(url)); return new Response(bytes); } });
  delete host.environment.NATLANG_MODEL_PATH;
  const path = await ensureStockModel(host, release);
  assert.equal(path, stockModelPath(host, release));
  assert.ok(path.startsWith(host.directories.models));
  assert.deepEqual(fetched, ['https://models.invalid/fixture.gguf']);
  assert.deepEqual(readFileSync(path), bytes);
  assert.match(log.join(''), /downloading Fixture from x\/y/);
  await ensureStockModel(host, release);
  assert.equal(fetched.length, 1, 'a verified file is not downloaded again');
  assert.equal(await managedModelSha256(host, release), release.sha256, 'a stock model\'s hash is the declared one');
  // a download that does not match its size and SHA-256 is refused and leaves nothing behind
  const corrupt = { ...release, file: 'corrupt.gguf', sha256: '0'.repeat(64) };
  const refused = fixtureHost(mkdtempSync(join(tmpdir(), 'natlang-llama-corrupt-')), { fetch: async () => new Response(bytes) });
  delete refused.host.environment.NATLANG_MODEL_PATH;
  await assert.rejects(ensureStockModel(refused.host, corrupt), /failed its size or SHA-256 check/);
  assert.equal(existsSync(stockModelPath(refused.host, corrupt)), false);
  assert.deepEqual(readdirSync(refused.host.directories.models), []);
});

test('prerequisites, slot plans and runtime discovery read the host\'s directories, settings and probes', () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-llama-plan-'));
  const { host, model } = fixtureHost(root);
  const found = discoverLlamaRuntime(host);
  assert.equal(found.selected.source, 'explicit'); assert.equal(found.runtimeRoot, host.directories.runtime);
  const prerequisites = localModelPrerequisites(host);
  assert.equal(prerequisites.available, true); assert.equal(prerequisites.modelPath, model);
  const plan = localSlotPlan(host, { memoryBudgetMiB: 4096, contextTokens: 8192, kvBytesPerToken: 100000 }, model);
  assert.equal(plan.budgetSource, 'memoryBudgetMiB'); assert.equal(plan.contextPerSlot, 8192);
  assert.ok(plan.slots >= 1 && plan.slots <= 8);
  // the Node binding derives the same roots from the environment the CLI uses
  const node = nodeLlamaHost({ NATLANG_CACHE_HOME: join(root, 'cache'), NATLANG_STATE_HOME: join(root, 'state'), NATLANG_RUNTIME_HOME: join(root, 'rt') });
  assert.equal(node.directories.warmPrefixes, join(root, 'cache', 'warm-prefixes'));
  assert.equal(node.directories.models, join(root, 'cache', 'models'));
  assert.equal(node.directories.state, join(root, 'state'));
  assert.equal(node.directories.runtime, join(root, 'rt'));
});

test('the llama entry bundles on its own: its closure is the llama directory and three leaf modules', async () => {
  const { build } = await import('esbuild');
  const result = await build({ entryPoints: [new URL('../dist/llama/index.js', import.meta.url).pathname], bundle: true, platform: 'node',
    format: 'esm', write: false, metafile: true, logLevel: 'silent' });
  const inputs = Object.keys(result.metafile.inputs).filter(file => !file.startsWith('node:')).map(file => file.replace(/^.*?dist\//, ''));
  assert.deepEqual(inputs.filter(file => !file.startsWith('llama/')).sort(),
    ['llama-fork-source.js', 'llama-runtime-release.js', 'model-catalog.js', 'package/directories.js']);
});
