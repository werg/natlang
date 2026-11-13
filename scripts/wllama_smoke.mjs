#!/usr/bin/env node
/**
 * The in-page engine against the managed llama-server on the same model: decision scores of a few items, once plain and
 * once with `cache_messages`, in headless Chrome (JSPI build). Run it after `update_llama_runtime.mjs --fork` vendors a
 * new release (docs/ENGINE_PARITY.md).
 *
 *   node scripts/wllama_smoke.mjs --model MODEL.gguf --server LLAMA_SERVER [--chrome PATH] [--tolerance NATS]
 *
 * Needs playwright (npx playwright, or a wllama checkout's node_modules via NODE_PATH) and Chrome. Exits 1 when the
 * page ranks an item's options differently from the native server, differs by more than the tolerance (default 1 nat:
 * quantized matmuls in WebAssembly and AVX round differently), or scores a repeated item differently.
 */
import { spawn } from 'node:child_process';
import { createReadStream, mkdtempSync, rmSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2), value = name => args.includes(name) ? args[args.indexOf(name) + 1] : undefined;
const model = value('--model'), server = value('--server'), chrome = value('--chrome') ?? '/usr/bin/google-chrome';
const tolerance = Number(value('--tolerance') ?? 1);
if (!model || !server) throw new Error('usage: node scripts/wllama_smoke.mjs --model MODEL.gguf --server LLAMA_SERVER [--chrome PATH]');
const vendor = resolve(dirname(fileURLToPath(import.meta.url)), '../ts-host/vendor/wllama');
const { chromium } = createRequire(import.meta.url)('playwright');

const isolation = { 'cross-origin-opener-policy': 'same-origin', 'cross-origin-embedder-policy': 'require-corp',
  'cross-origin-resource-policy': 'same-origin' };
const http = createServer((request, response) => {
  const path = new URL(request.url, 'http://page').pathname;
  if (path === '/') { response.writeHead(200, { 'content-type': 'text/html', ...isolation }); return response.end('<html></html>'); }
  const file = path === '/model.gguf' ? model : join(vendor, path);
  try {
    const type = { '.js': 'text/javascript', '.wasm': 'application/wasm' }[extname(file)] ?? 'application/octet-stream';
    response.writeHead(200, { 'content-type': type, 'content-length': statSync(file).size, ...isolation });
    createReadStream(file).pipe(response);
  } catch { response.writeHead(404); response.end(); }
}).listen(0, '127.0.0.1');

const system = { role: 'system', content: 'Classify the support ticket. Answer with one JSON string.' };
const options = ['"billing"', '"technical"', '"other"'];
const items = [
  { messages: [system, { role: 'user', content: 'My invoice charged me twice this month.' }], continuations: options },
  { messages: [system, { role: 'user', content: 'The app crashes when I open settings.' }], continuations: options, cache_messages: 1 },
  { messages: [system, { role: 'user', content: 'Can I change the email on my account?' }], continuations: options, cache_messages: 1 },
];

const port = 18000 + Math.floor(Math.random() * 1000);
const native = spawn(server, ['-m', model, '--port', String(port), '-c', '4096', '-np', '2', '--swa-full'], { stdio: 'ignore' });
let expected;
try {
  for (let attempt = 0; attempt < 300 && !expected; attempt++) {
    await new Promise(done => setTimeout(done, 1000));
    try {
      const response = await fetch(`http://127.0.0.1:${port}/v1/natlang/score`, { method: 'POST',
        headers: { 'content-type': 'application/json' }, body: JSON.stringify({ items }) });
      if (response.ok) expected = (await response.json()).results.map(result => result.log_probs);
    } catch { /* still loading */ }
  }
} finally { native.kill(); }
if (!expected) throw new Error('the native server did not answer');

const profile = mkdtempSync(join(tmpdir(), 'natlang-wllama-smoke-'));
const browser = await chromium.launchPersistentContext(profile, { executablePath: chrome,
  args: ['--enable-features=WebAssemblyJSPI', '--js-flags=--experimental-wasm-jspi'] });
let page;
try {
  const tab = await browser.newPage();
  await tab.goto(`http://127.0.0.1:${http.address().port}/`);
  page = await tab.evaluate(async items => {
    const { Wllama } = await import('/esm/index.js');
    const engine = new Wllama({ default: '/esm/wasm/wllama.wasm' });
    await engine.loadModelFromUrl(location.origin + '/model.gguf', { useCache: false, n_ctx: 4096, n_gpu_layers: 0 });
    const all = await engine.createNatlangScore({ items });
    const again = await engine.createNatlangScore({ items: [items[1]] });
    return { all: all.results.map(result => result.log_probs ?? result), again: again.results[0].log_probs ?? again.results[0],
      capabilities: engine.capabilities };
  }, items);
} finally { await browser.close(); http.close(); rmSync(profile, { recursive: true, force: true }); }

const rank = scores => scores.map((_, index) => index).sort((a, b) => scores[b] - scores[a]).join();
const failures = [];
expected.forEach((scores, index) => {
  const mine = page.all[index];
  if (!Array.isArray(mine)) return failures.push(`item ${index}: ${JSON.stringify(mine)}`);
  if (rank(mine) !== rank(scores)) failures.push(`item ${index}: page ranks ${rank(mine)}, native ${rank(scores)}`);
  const gap = Math.max(...mine.map((score, option) => Math.abs(score - scores[option])));
  if (gap > tolerance) failures.push(`item ${index}: page differs by ${gap.toFixed(3)} nats`);
});
if (JSON.stringify(page.again) !== JSON.stringify(page.all[1])) failures.push('a repeated item scored differently');
console.log(JSON.stringify({ native: expected, page: page.all, repeated: page.again, capabilities: page.capabilities, failures }));
process.exit(failures.length ? 1 : 0);
