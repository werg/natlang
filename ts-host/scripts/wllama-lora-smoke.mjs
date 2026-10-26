#!/usr/bin/env node
/**
 * Real in-browser LoRA check for a wllama build with LoRA blob loading (docs/TUNING.md "wllama").
 *
 *   node scripts/wllama-lora-smoke.mjs --wllama <built wllama package dir> --base base.gguf --lora a.gguf --lora b.gguf
 *
 * Headless Chromium loads the base and adapters as blobs (scale 0), then greedily completes one prompt with no
 * adapter, with each adapter alone, back to the base (explicitly and with no `lora` field, after an adapter request),
 * and each adapter again. It checks that every adapter changes the output, that returning to the base reproduces the
 * base exactly (no prefix cached under another adapter serves it), and that each stack is reproducible.
 */
import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { chromium } from 'playwright-core';
import { chromiumPath } from './chromium-path.mjs';

const { values } = parseArgs({ options: { wllama: { type: 'string' }, base: { type: 'string' }, lora: { type: 'string', multiple: true },
  prompt: { type: 'string', default: 'The three most important things about a lighthouse are' }, tokens: { type: 'string', default: '24' } } });
if (!values.wllama || !values.base || (values.lora?.length ?? 0) < 2) {
  console.error('usage: wllama-lora-smoke.mjs --wllama DIR --base BASE.gguf --lora A.gguf --lora B.gguf');
  process.exit(2);
}
const files = { '/base.gguf': resolve(values.base), ...Object.fromEntries(values.lora.map((path, i) => [`/lora-${i}.gguf`, resolve(path)])) };
const wllamaDir = resolve(values.wllama);
const page = `<!doctype html><meta charset="utf-8"><script type="module">
import { Wllama } from '/wllama/esm/index.js';
window.run = async ({ loras, prompt, tokens }) => {
  const wllama = new Wllama({ default: '/wllama/esm/wasm/wllama.wasm' }, { allowOffline: true });
  if (!wllama.capabilities?.loraBlobs) throw new Error('this wllama build cannot load LoRA blobs');
  const blob = async path => (await fetch(path)).blob();
  await wllama.loadModel([await blob('/base.gguf')], { n_ctx: 512, n_gpu_layers: 0, n_parallel: 1,
    lora_adapters: await Promise.all(loras.map(async path => ({ blob: await blob(path), scale: 0 }))), lora_init_without_apply: true });
  const off = loras.map((_, id) => ({ id, scale: 0 }));
  const complete = async lora => {
    const response = await wllama.createCompletion({ prompt, n_predict: tokens, temperature: 0, top_k: 1, cache_prompt: true,
      ...(lora ? { lora } : {}) });
    return response.choices?.[0]?.text ?? response.content;
  };
  const on = id => off.map(item => item.id === id ? { id, scale: 1 } : item);
  const out = {};
  out.base = await complete(off);
  for (let id = 0; id < loras.length; id++) out['lora' + id] = await complete(on(id));
  out.baseExplicitAfter = await complete(off);
  out.lora0Again = await complete(on(0));
  out.baseDefaultAfter = await complete(undefined);
  out.lora1Again = await complete(on(1));
  await wllama.exit();
  return out;
};
window.ready = true;
</script>`;
const server = createServer((request, response) => {
  const path = new URL(request.url, 'http://x').pathname;
  // Cross-origin isolation enables wllama's threads.
  const headers = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' };
  if (path === '/') { response.writeHead(200, { ...headers, 'content-type': 'text/html' }); response.end(page); return; }
  const file = files[path] ?? (path.startsWith('/wllama/') ? resolve(wllamaDir, '.' + path.slice('/wllama'.length)) : null);
  if (!file || (path.startsWith('/wllama/') && !file.startsWith(wllamaDir))) { response.writeHead(404); response.end(); return; }
  try { statSync(file); } catch { response.writeHead(404); response.end(); return; }
  const type = file.endsWith('.js') ? 'text/javascript' : file.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream';
  response.writeHead(200, { ...headers, 'content-type': type, 'content-length': statSync(file).size });
  createReadStream(file).pipe(response);
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const browser = await chromium.launch({ executablePath: chromiumPath(), args: ['--enable-features=SharedArrayBuffer'] });
let failed = false;
try {
  const tab = await browser.newPage();
  tab.on('console', message => { if (message.type() === 'error') console.error('[page]', message.text()); });
  await tab.goto(`http://127.0.0.1:${server.address().port}/`);
  await tab.waitForFunction(() => window.ready);
  const started = Date.now();
  const out = await tab.evaluate(options => window.run(options),
    { loras: values.lora.map((_, i) => `/lora-${i}.gguf`), prompt: values.prompt, tokens: Number(values.tokens) });
  console.log(JSON.stringify({ seconds: (Date.now() - started) / 1000, out }, null, 1));
  const checks = {
    'adapter 0 changes the output': out.lora0 !== out.base,
    'adapter 1 changes the output': out.lora1 !== out.base,
    'the adapters differ from each other': out.lora0 !== out.lora1,
    'explicit base after an adapter equals the base': out.baseExplicitAfter === out.base,
    'base with no lora field after an adapter equals the base': out.baseDefaultAfter === out.base,
    'adapter 0 is reproducible after other stacks': out.lora0Again === out.lora0,
    'adapter 1 is reproducible after the base': out.lora1Again === out.lora1,
  };
  for (const [name, ok] of Object.entries(checks)) { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}`); failed ||= !ok; }
} finally { await browser.close(); server.close(); }
process.exit(failed ? 1 : 0);
