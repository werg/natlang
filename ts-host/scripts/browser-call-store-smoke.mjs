#!/usr/bin/env node
/**
 * Chromium check of the browser call store (src/browser/call-store*.ts): a page's runtime records a call into the
 * origin's OPFS database, `query` reads it back, the record survives a reload, and a second tab, which cannot open the
 * held database, runs its calls without recording. Served without cross-origin isolation headers on purpose.
 *   node scripts/browser-call-store-smoke.mjs [--headed]
 */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { chromium } from 'playwright-core';
import { chromiumPath } from './chromium-path.mjs';

const root = resolve(import.meta.dirname, '..');
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.wasm': 'application/wasm' };
const PAGE = `<!doctype html><meta charset="utf-8"><script type="module">
import { createNatlangRuntime, loadVirtualNatlang } from '/dist/browser/natlang.js';
const price = loadVirtualNatlang({ 'price.nl': '---\\nargs: { id: string }\\nreturns: number\\n---\\nLook up order id and return its total.\\n' }, 'price.nl');
const model = async () => ({ calls: [['return_result', { status: 'success', value: 30 }]] });
window.smoke = async id => {
  const runtime = createNatlangRuntime({ model });
  const value = await runtime.run(() => price(id));
  const store = runtime.callStore();
  let calls = null, record = null, error = null;
  try {
    await store.ready;
    calls = await store.query('calls', { limit: 50 });
    record = await store.query('call', calls[0].call_id);
  } catch (caught) { error = String(caught.message ?? caught); }
  return { value, calls: calls?.map(call => [call.definition_name, call.outcome]) ?? null,
    input: record ? await store.query('value', record.inputs.id) : null, error };
};
</script>`;

const server = createServer(async (request, response) => {
  const pathname = new URL(request.url, 'http://localhost').pathname;
  if (pathname === '/smoke.html') { response.writeHead(200, { 'Content-Type': types['.html'] }); response.end(PAGE); return; }
  const file = resolve(root, '.' + decodeURIComponent(pathname));
  if (!file.startsWith(resolve(root, 'dist') + sep)) { response.writeHead(404); response.end(); return; }
  try { const body = await readFile(file); response.writeHead(200, { 'Content-Type': types[extname(file)] ?? 'application/octet-stream' }); response.end(body); }
  catch { response.writeHead(404); response.end(); }
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const url = `http://127.0.0.1:${server.address().port}/smoke.html`;
const browser = await chromium.launch({ headless: !process.argv.includes('--headed'), executablePath: chromiumPath() });
let failed = false;
const check = (label, ok, detail) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? `: ${JSON.stringify(detail)}` : ''}`); if (!ok) failed = true; };
try {
  const context = await browser.newContext();
  const page = await context.newPage();
  page.on('console', message => { if (message.type() === 'warning' || message.type() === 'error') console.log(`  [page] ${message.text()}`); });
  await page.goto(url);
  await page.waitForFunction(() => typeof window.smoke === 'function');
  const first = await page.evaluate(() => window.smoke('abc'));
  check('first call is recorded and readable', first.value === 30 && first.calls?.length === 1 && first.input === 'abc', first);
  await page.reload();
  await page.waitForFunction(() => typeof window.smoke === 'function');
  const second = await page.evaluate(() => window.smoke('abcd'));
  check('records survive a reload', second.calls?.length === 2 && second.calls.every(([name, outcome]) => name === 'price' && outcome === 'done'), second);
  const other = await context.newPage();
  other.on('console', message => console.log(`  [tab 2] ${message.text()}`));
  await other.goto(url);
  await other.waitForFunction(() => typeof window.smoke === 'function');
  const held = await other.evaluate(() => window.smoke('x'));
  check('a second tab runs without recording while the first holds the store', held.value === 30 && held.calls === null && !!held.error, held);
} finally { await browser.close(); server.close(); }
process.exit(failed ? 1 : 0);
