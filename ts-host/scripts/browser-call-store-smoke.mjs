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
import { exitUnlessFreshBrowserBuild } from './browser-build-freshness.mjs';

exitUnlessFreshBrowserBuild('scripts/browser-call-store-smoke.mjs');

const root = resolve(import.meta.dirname, '..');
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.wasm': 'application/wasm' };
const PAGE = `<!doctype html><meta charset="utf-8"><script type="module">
import { createNatlangRuntime, loadVirtualNatlang } from '/dist/browser/natlang.js';
const price = loadVirtualNatlang({ 'price.nl': '---\\nargs: { id: string }\\nreturns: number\\n---\\nLook up order id and return its total.\\n' }, 'price.nl');
const model = async () => ({ calls: [['return_result', { status: 'success', value: 30 }]] });
window.smoke = async id => {
  const runtime = createNatlangRuntime({ model });
  const store = runtime.callStore();
  // The smoke's temporary origin must not evict its fixtures due to unrelated host disk pressure.
  try { await store.ready; await store.query('writeSettings',{minFreeBytes:0}); } catch {}
  const value = await runtime.run(() => price(id));
  let calls = null, record = null, error = null;
  try {
    await store.ready;
    calls = await store.query('calls', { limit: 50 });
    record = await store.query('call', calls[0].call_id);
  } catch (caught) { error = String(caught.message ?? caught); }
  return { value, calls: calls?.map(call => [call.definition_name, call.outcome]) ?? null,
    input: record ? await store.query('value', record.inputs.id) : null, error };
};
window.checkpointSmoke = async () => {
  const store=createNatlangRuntime({model}).callStore(); await store.ready;
  const id='unfinished-native-checkpoint';
  const text=JSON.stringify({receipt:'observed native bytes',revision:7});
  const bytes=new TextEncoder().encode(text);
  const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(value=>value.toString(16).padStart(2,'0')).join('');
  const ref={complete:true,hash,bytes:bytes.length};
  store.begin({callId:id,parentCallId:null,parentActionIndex:null,taskId:'checkpoint',programId:null,programRoot:null,
    definition:{id:'native-checkpoint',name:'native-checkpoint',source:null,key:'key',interface:'interface',site:'named'},
    modelId:null,startedAt:new Date().toISOString(),auditOf:null});
  await store.checkpoint(id,{version:'natlang.call-checkpoint/1',call_id:id,inputs:{native:ref},effects:[{result:ref}]},new Map([[hash,text]]));
  await store.close();
};
window.readCheckpointSmoke = async () => {
  const store=createNatlangRuntime({model}).callStore();await store.ready;
  const rows=await store.query('annotations','unfinished-native-checkpoint');
  const state=rows.find(row=>row.kind==='call-checkpoint')?.value;
  return {value:state?await store.query('value',state.effects[0].result):null,
    completed:!!(await store.query('call','unfinished-native-checkpoint'))};
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
  await page.evaluate(()=>window.checkpointSmoke());
  await page.reload();
  await page.waitForFunction(()=>typeof window.readCheckpointSmoke==='function');
  const unfinished=await page.evaluate(()=>window.readCheckpointSmoke());
  check('unfinished native evidence survives OPFS close/reload without a completed record',unfinished.value?.receipt==='observed native bytes'&&unfinished.value?.revision===7&&!unfinished.completed,unfinished);
  const other = await context.newPage();
  other.on('console', message => console.log(`  [tab 2] ${message.text()}`));
  await other.goto(url);
  await other.waitForFunction(() => typeof window.smoke === 'function');
  const held = await other.evaluate(() => window.smoke('x'));
  check('a second tab runs without recording while the first holds the store', held.value === 30 && held.calls === null && !!held.error, held);
} finally { await browser.close(); server.close(); }
process.exit(failed ? 1 : 0);
