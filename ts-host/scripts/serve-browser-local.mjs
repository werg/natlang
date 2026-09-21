#!/usr/bin/env node
/** Serve the browser pilot with WASM isolation and optional hardware WebGPU Chrome. */
import { createServer } from 'node:http';
import { createReadStream, existsSync, readdirSync } from 'node:fs';
import { stat } from 'node:fs/promises';
import { homedir } from 'node:os';
import { extname, join, resolve, sep } from 'node:path';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';
import { createPlaygroundJobs } from './playground-jobs.mjs';

const root = resolve(import.meta.dirname, '../..');
const handleWorkbench = createPlaygroundJobs(root);
const port = Number(process.argv.find(arg => arg.startsWith('--port='))?.slice(7) ?? 8765);
if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error('invalid --port');
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.wasm': 'application/wasm', '.json': 'application/json', '.jinja': 'text/plain; charset=utf-8' };
const server = createServer(async (request, response) => {
  try {
    if (await handleWorkbench(request, response)) return;
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (!pathname.startsWith('/ts-host/') && !pathname.startsWith('/models/'))
      throw new Error('path is not a pilot asset');
    if (pathname === '/models/browser-catalog.json' &&
        !existsSync(resolve(root, 'models/browser-catalog.json'))) {
      const { BROWSER_MODEL_CATALOG } = await import('../dist/browser/models.js');
      const available = BROWSER_MODEL_CATALOG.filter(model =>
        existsSync(resolve(root, '.' + model.url)) && existsSync(resolve(root, '.' + model.templateUrl)));
      const catalog = { schema: 'natlang.browser-model-catalog/1',
        defaultId: available[0]?.id ?? '', models: available };
      const body = JSON.stringify(catalog);
      response.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8',
        'Content-Length': Buffer.byteLength(body), 'Cache-Control': 'no-store',
        'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' });
      response.end(body); return;
    }
    const file = resolve(root, '.' + decodeURIComponent(pathname), pathname.endsWith('/') ? 'index.html' : '');
    if (!file.startsWith(root + sep)) throw new Error('path outside root');
    const size = (await stat(file)).size;
    const range = /^bytes=(\d+)-(\d*)$/.exec(request.headers.range ?? '');
    const start = range ? Number(range[1]) : 0;
    const end = range && range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
    if (start > end || end >= size) {
      response.writeHead(416, { 'Content-Range': `bytes */${size}` }); response.end(); return;
    }
    response.writeHead(range ? 206 : 200, {
      'Content-Type': types[extname(file)] ?? 'application/octet-stream',
      'Content-Length': end - start + 1, 'Accept-Ranges': 'bytes',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      ...(range ? { 'Content-Range': `bytes ${start}-${end}/${size}` } : {}),
    });
    if (request.method === 'HEAD') response.end();
    else createReadStream(file, { start, end }).pipe(response);
  } catch (error) { response.writeHead(404); response.end(String(error)); }
});

function chromiumPath() {
  if (process.env.NATLANG_CHROMIUM) return process.env.NATLANG_CHROMIUM;
  if (existsSync(chromium.executablePath())) return chromium.executablePath();
  const cache = join(homedir(), '.cache/ms-playwright');
  if (existsSync(cache)) {
    const versions = readdirSync(cache).filter(name => /^chromium-\d+$/.test(name)).sort().reverse();
    for (const version of versions) {
      const candidate = join(cache, version, 'chrome-linux64/chrome');
      if (existsSync(candidate)) return candidate;
    }
  }
  throw new Error('Chromium is missing; install it with npx playwright install chromium');
}

await new Promise(resolve => server.listen(port, '127.0.0.1', resolve));
const page = process.argv.includes('--playground') ? 'playground' : 'examples/browser-local';
const url = `http://127.0.0.1:${server.address().port}/ts-host/${page}/`;
console.log(`natlang browser ${page === 'playground' ? 'playground' : 'pilot'}: ${url}`);
if (process.argv.includes('--open-gpu') || process.argv.includes('--open')) {
  const gpu = process.argv.includes('--open-gpu');
  const flags = gpu && process.platform === 'linux' ? [
    '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan',
    '--enable-dawn-features=vulkan_enable_f16_on_nvidia',
  ] : [];
  const profile = join(homedir(), '.cache/natlang-chromium-gpu');
  const browser = spawn(chromiumPath(), [`--user-data-dir=${profile}`, ...flags, url], { stdio: 'inherit' });
  browser.once('error', error => console.error(`Cannot open Chromium: ${error.message}`));
}
