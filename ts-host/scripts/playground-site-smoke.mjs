#!/usr/bin/env node
/**
 * Real-browser smoke for the public playground (build-playground-site.mjs), served as GitHub Pages serves it: static
 * files under a subpath, without cross-origin isolation headers or the local pipeline API.
 *
 *   node scripts/playground-site-smoke.mjs [--site SITE] [--network]
 *
 * Builds the site into a temporary directory unless --site names a built one. --network also checks that the page can
 * read the catalog's model from its host (CORS and range requests).
 */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createReadStream, mkdtempSync, rmSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { extname, join, resolve, sep } from 'node:path';
import { chromium } from 'playwright-core';
import { chromiumPath } from './chromium-path.mjs';

const given = process.argv.indexOf('--site');
const temporary = given < 0 ? mkdtempSync(join(tmpdir(), 'natlang-playground-site-')) : null;
const site = resolve(temporary ?? process.argv[given + 1]);
if (temporary) {
  const built = spawnSync(process.execPath, [new URL('./build-playground-site.mjs', import.meta.url).pathname, '--out', site], { stdio: 'inherit' });
  if (built.status !== 0) process.exit(built.status ?? 1);
}
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.wasm': 'application/wasm' };
const requested = [];
const server = createServer((request, response) => {
  const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
  requested.push(pathname);
  const file = resolve(site, '.' + pathname.replace(/^\/natlang/, ''), pathname.endsWith('/') ? 'index.html' : '');
  const size = file.startsWith(site + sep) ? statSync(file, { throwIfNoEntry: false })?.size : undefined;
  if (size === undefined || !pathname.startsWith('/natlang/')) { response.writeHead(404); response.end(); return; }
  response.writeHead(200, { 'Content-Type': types[extname(file)] ?? 'application/octet-stream', 'Content-Length': size });
  createReadStream(file).pipe(response);
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const base = `http://127.0.0.1:${server.address().port}/natlang/play/`;
let browser;
try {
  browser = await chromium.launch({ headless: true, executablePath: chromiumPath(), args: ['--no-sandbox', '--enable-unsafe-swiftshader'] });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#projectSelect')?.options.length > 0);
  assert.equal(await page.evaluate(() => crossOriginIsolated), false, 'served like GitHub Pages');
  assert.match(await page.locator('#editor').inputValue(), /find the person's full name and email address/);
  // The case and training workbench is not part of the public edition.
  for (const selector of ['#advancedButton', '#captureCase', '[data-panel=cases]', '[data-panel=jobs]', '#casesPanel', '#jobsPanel'])
    assert.equal(await page.locator(selector).isVisible(), false, `${selector} is hidden`);
  // The gallery opens its examples as editable projects; natural-language ones ask for the model.
  await page.locator('#libraryNav').click();
  assert.equal(await page.locator('.example-card').count(), 14);
  await page.locator('#exampleSearch').fill('contact');
  await page.locator('.example-card').click();
  await page.waitForFunction(() => document.querySelector('#projectName').textContent === 'Extract contact');
  assert.match(await page.locator('#runButton').textContent(), /Load model to run/);
  await page.locator('#runButton').click();
  assert.equal(await page.locator('#modelDialog').isVisible(), true);
  assert.match(await page.locator('#modelSelect').textContent(), /LFM3 1\.7B · Q4_K_M/);
  await page.locator('#modelDialog button[value=cancel]').last().click();
  // The runtime executes in the page: a TypeScript fixture runs to its expected value and records a run.
  await page.evaluate(async () => {
    const [{ crispExamples }, { storage }, { newPlaygroundProject }] = await Promise.all([
      import('./examples/crisp.mjs'), import('./storage.mjs'), import('../dist/browser/natlang.js')]);
    const template = crispExamples.find(item => item.id === 'structured-calculation');
    const project = newPlaygroundProject(template.name, template.root, template.files, template.inputs, template.expected);
    await storage.put('projects', project); localStorage.setItem('natlang-project', project.id);
  });
  await page.reload();
  await page.waitForFunction(() => window.natlangPlayground?.project?.name && document.querySelector('#runButton:not([disabled])'));
  await page.locator('#runButton').click();
  await page.getByText('Expected value matched').waitFor();
  assert.match(await page.locator('#resultValue').textContent(), /"sum": 8/);
  const catalog = await (await page.request.get(new URL('models/browser-catalog.json', base).href)).json();
  const [model] = catalog.models;
  assert.equal((await page.request.head(new URL(model.templateUrl, base).href)).status(), 200);
  if (process.argv.includes('--network')) {
    const magic = await page.evaluate(async url => {
      const response = await fetch(url, { headers: { Range: 'bytes=0-3' } });
      return `${response.status} ${new TextDecoder().decode(await response.arrayBuffer())}`;
    }, model.url);
    assert.equal(magic, '206 GGUF', `the page can range-read ${model.url}`);
  }
  assert.equal(requested.some(path => path.includes('/api/')), false, 'no local pipeline API requests');
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  assert.deepEqual(errors, []);
  console.log(`PASS public playground: gallery, model catalog (${model.url}), TypeScript run, no workbench${process.argv.includes('--network') ? ', model readable' : ''}`);
} finally {
  await browser?.close();
  server.close();
  if (temporary) rmSync(temporary, { recursive: true, force: true });
}
