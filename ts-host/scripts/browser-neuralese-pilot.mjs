#!/usr/bin/env node
/**
 * Headless Chromium check of the browser runtime with Neuralese (test/browser-neuralese.html): the WebAssembly service
 * in a Web Worker, driven by the runtime. Serves ts-host and the given GGUF files; prints the page's report.
 *
 *   node scripts/browser-neuralese-pilot.mjs --model model.gguf --heads neuralese.gguf [--gpu] [--cache] [--chromium-flags '...']
 *
 * --cache loads the files through the OPFS model-file cache (and opens them a second time from it, timed).
 *
 * --gpu forces the WebGPU build (Chromium with --enable-unsafe-webgpu and Vulkan; it needs an adapter with shader-f16, else
 * it runs on one CPU thread). On Linux, headless Chromium only
 * offers the SwiftShader (software) adapter; --headed (under `xvfb-run -a` on a server) gets the hardware one.
 */
import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { extname, resolve, sep } from 'node:path';
import { parseArgs } from 'node:util';
import { chromium } from 'playwright-core';
import { chromiumPath } from './chromium-path.mjs';

const { values } = parseArgs({ options: { model: { type: 'string' }, heads: { type: 'string' }, timeout: { type: 'string', default: '900' },
  cache: { type: 'boolean', default: false }, gpu: { type: 'boolean', default: false }, headed: { type: 'boolean', default: false }, 'chromium-flags': { type: 'string', default: '' } } });
const root = resolve(import.meta.dirname, '..');
const files = { '/files/model.gguf': resolve(values.model), '/files/heads.gguf': resolve(values.heads) };
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm' };
const server = createServer((request, response) => {
  const path = new URL(request.url, 'http://x').pathname;
  const file = files[path] ?? resolve(root, '.' + decodeURIComponent(path));
  if (!files[path] && !file.startsWith(root + sep)) { response.writeHead(403); response.end(); return; }
  try {
    const size = statSync(file).size;
    // Cross-origin isolation, as threaded wasm will need.
    response.writeHead(200, { 'content-type': types[extname(file)] ?? 'application/octet-stream', 'content-length': size,
      'cross-origin-opener-policy': 'same-origin', 'cross-origin-embedder-policy': 'require-corp' });
    createReadStream(file).pipe(response);
  } catch { response.writeHead(404); response.end(); }
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const url = `http://127.0.0.1:${server.address().port}`;
// vulkan_enable_f16_on_nvidia: Dawn hides shader-f16 on NVIDIA's Vulkan driver unless told otherwise, and
// ggml-webgpu needs it (it only changes NVIDIA adapters).
const args = [...(values.gpu ? ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist',
  '--enable-dawn-features=vulkan_enable_f16_on_nvidia'] : []),
  ...values['chromium-flags'].split(' ').filter(Boolean)];
const browser = await chromium.launch({ headless: !values.headed, args, executablePath: chromiumPath() });
try {
  const page = await browser.newPage();
  page.on('console', message => process.stderr.write(`[page] ${message.text()}\n`));
  await page.goto(`${url}/test/browser-neuralese.html?model=/files/model.gguf&heads=/files/heads.gguf${values.gpu ? '&gpu=force' : ''}${values.cache ? '&cache=1' : ''}`);
  await page.waitForFunction(() => window.__neuraleseReport, null, { timeout: Number(values.timeout) * 1000 });
  const report = await page.evaluate(() => window.__neuraleseReport);
  console.log(JSON.stringify(report, null, 1));
  process.exitCode = report.ok ? 0 : 1;
} finally {
  await browser.close();
  server.close();
}
