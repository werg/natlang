#!/usr/bin/env node
/** Actual Chromium smoke, with an optional real-model pilot. */
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat, writeFile } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { chromium } from 'playwright-core';

const root = resolve(import.meta.dirname, '../..');
const liveModel = process.argv.includes('--model');
const cpu = process.argv.includes('--cpu');
const broad = process.argv.includes('--broad');
const output = process.argv.find(arg => arg.startsWith('--output='))?.slice('--output='.length);
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm', '.json': 'application/json' };
const server = createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (!pathname.startsWith('/ts-host/') && !pathname.startsWith('/models/'))
      throw new Error('path is not a pilot asset');
    const file = resolve(root, '.' + decodeURIComponent(pathname), pathname.endsWith('/') ? 'index.html' : '');
    if (!file.startsWith(root + sep)) throw new Error('path outside root');
    const size = (await stat(file)).size;
    const range = /^bytes=(\d+)-(\d*)$/.exec(request.headers.range ?? '');
    const start = range ? Number(range[1]) : 0;
    const end = range && range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
    if (start > end || end >= size) { response.writeHead(416, { 'Content-Range': `bytes */${size}` }); response.end(); return; }
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
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ headless: true,
  executablePath: process.env.NATLANG_CHROMIUM || chromium.executablePath(),
  args: ['--no-sandbox', '--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(`${url}/ts-host/test/browser-smoke.html`);
  await page.getByText('PASS browser interpreter').waitFor({ timeout: 30000 });
  console.log('PASS actual Chromium browser interpreter smoke');
  if (errors.length) throw new Error(errors.join('\n'));
  if (liveModel) {
    await page.goto(`${url}/ts-host/examples/browser-local/`);
    if (broad) await page.locator('#schema').selectOption('broad');
    if (cpu) await page.locator('#gpuLayers').fill('0');
    await page.locator('#context').fill('2048');
    await page.locator('#load').click();
    await page.waitForFunction(() => /Model ready|Load failed/.test(document.querySelector('#status').textContent),
      null, { timeout: 600000 });
    const loadStatus = await page.locator('#status').textContent();
    console.log(loadStatus);
    if (!loadStatus.startsWith('Model ready')) throw new Error(loadStatus);
    await page.locator('#run').click();
    await page.waitForFunction(() => /correct|Run failed|Load failed/.test(document.querySelector('#status').textContent),
      null, { timeout: 240000 });
    const result = { status: await page.locator('#status').textContent(),
      diagnostics: JSON.parse(await page.locator('#diagnostics').textContent()),
      task: JSON.parse(await page.locator('#result').textContent()) };
    if (output) await writeFile(output, JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify({ status: result.status, task: result.task }, null, 2));
    if (!result.task[0]?.correct) process.exitCode = 1;
  }
} finally { await browser.close(); server.close(); }
