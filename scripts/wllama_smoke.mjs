#!/usr/bin/env node
/**
 * The in-page engine against the managed llama-server on the same model: decision scores of a few items, once plain and
 * once with `cache_messages`, in headless Chrome (JSPI build). Run it after `update_llama_runtime.mjs --fork` vendors a
 * new release (docs/ENGINE_PARITY.md).
 *
 *   node scripts/wllama_smoke.mjs --model MODEL.gguf --server LLAMA_SERVER [--chrome PATH] [--tolerance NATS] [--wllama DIR]
 *
 * Needs playwright (npx playwright, or a wllama checkout's node_modules via NODE_PATH) and Chrome. Exits 1 when the
 * page renders or tokenizes an item differently from the native server (its prompt and reply token counts), ranks its
 * options differently, differs by more than the tolerance, or ranks a repeated item differently or moves it by more than
 * the tolerance. The default tolerance, 2.5 nats, is three standard deviations of the difference between two Q8_0 engines:
 * a Q8_0 matmul quantizes its activations to int8 blocks, so any change upstream (a build's kernels, flash attention, the
 * KV type, WebAssembly vs NEON) redraws that rounding error. Measured on LFM3-1.7B Q8_0 against the same weights
 * dequantized to F32 (where flash attention and the KV type move a score by at most 0.03 nats), six engine
 * configurations, the page among them, deviate by 0.58 nats RMS per score; pairs differed by 0.5-1.75 nats at most over
 * the 9 scores (docs/ENGINE_PARITY.md, numerics). A repeat is not bit-identical in the page: its KV cache is unified, so where a sequence's cells sit
 * depends on what ran before, attention sums in that cell order, and the int8 activation rounding of the Q8 matmuls turns
 * the last-bit differences into up to a few tenths of a nat (docs/ENGINE_PARITY.md, numerics).
 *
 * With a natlang GGUF export (native identity and producer adapter) it also checks native states across the two engines
 * (docs/ENGINE_PARITY.md, "Native state: one format"): a state the server produced is put into the page after load
 * (`putNativeState`, exact consumption) and a decision consuming it scores like the server's; a state the page produces
 * (a grammar-forced producer span) reads back as stored (`getNativeState`), decodes with natlang's codec to its ID, holds
 * the server's tokens and text, and the server accepts it (PUT) and consumes it exactly.
 */
import { spawn } from 'node:child_process';
import { createReadStream, mkdtempSync, rmSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const rank = scores => scores.map((_, index) => index).sort((a, b) => scores[b] - scores[a]).join();
const failures = [];

const args = process.argv.slice(2), value = name => args.includes(name) ? args[args.indexOf(name) + 1] : undefined;
const model = value('--model'), server = value('--server'), chrome = value('--chrome') ?? '/usr/bin/google-chrome';
const tolerance = Number(value('--tolerance') ?? 2.5);
if (!model || !server) throw new Error('usage: node scripts/wllama_smoke.mjs --model MODEL.gguf --server LLAMA_SERVER [--chrome PATH]');
// `--wllama DIR`: a candidate build (a release's llama-natlang-wllama.tar.gz unpacked) before it is vendored
const vendor = resolve(value('--wllama') ?? resolve(dirname(fileURLToPath(import.meta.url)), '../ts-host/vendor/wllama'));
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

const port = 18000 + Math.floor(Math.random() * 1000), base = `http://127.0.0.1:${port}`;
const native = spawn(server, ['-m', model, '--port', String(port), '-c', '4096', '-np', '2', '--swa-full'], { stdio: 'ignore' });
const post = async (path, body, init = {}) => fetch(base + path, { method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body), ...init });
const score = async scored => (await (await post('/v1/natlang/score', { items: scored })).json()).results.map(result => result.log_probs ?? result);
let expected, nativeResults, states = null;
try {
  for (let attempt = 0; attempt < 300 && !nativeResults; attempt++) {
    await new Promise(done => setTimeout(done, 1000));
    try {
      const response = await post('/v1/natlang/score', { items });
      if (response.ok) nativeResults = (await response.json()).results;
    } catch { /* still loading */ }
  }
  if (!nativeResults) throw new Error('the native server did not answer');
  expected = nativeResults.map(result => result.log_probs);
  const info = await (await fetch(base + '/v1/neuralese/info')).json();
  if (info.native_state?.enabled && info.native_state.production) {
    const [open, close] = info.native_state.role_marker_ids;
    const produce = { messages: [{ role: 'user', content: 'Summarize the ticket.' }], max_tokens: 40, temperature: 0,
      grammar: `root ::= "Billing: " <[${open}]> " refund request for order 4411" <[${close}]> " done"` };
    const reply = await (await post('/v1/chat/completions', produce)).json();
    const id = reply.native_state.blocks[0].id;
    const wire = new Uint8Array(await (await fetch(`${base}/v1/neuralese/native-states/${id}`)).arrayBuffer());
    const consume = { messages: [system, { role: 'user', content: [{ type: 'text', text: 'Ticket summary: ' }, { type: 'native_state', id }] }],
      continuations: options };
    states = { produce, id, wire: Buffer.from(wire).toString('base64'), consume, expected: (await score([consume]))[0] };
  }
  await runPage();
  if (states && page.states) await crossCheck();
} finally { native.kill(); }

var page;
async function runPage() {
  const profile = mkdtempSync(join(tmpdir(), 'natlang-wllama-smoke-'));
  const browser = await chromium.launchPersistentContext(profile, { executablePath: chrome,
    args: ['--enable-features=WebAssemblyJSPI', '--js-flags=--experimental-wasm-jspi'] });
  try {
    const tab = await browser.newPage();
    await tab.goto(`http://127.0.0.1:${http.address().port}/`);
    page = await tab.evaluate(async ({ items, states }) => {
      const { Wllama } = await import('/esm/index.js');
      const engine = new Wllama({ default: '/esm/wasm/wllama.wasm' });
      await engine.loadModelFromUrl(location.origin + '/model.gguf', { useCache: false, n_ctx: 4096, n_gpu_layers: 0 });
      const all = await engine.createNatlangScore({ items });
      const again = await engine.createNatlangScore({ items: [items[1]] });
      const result = { shapes: all.results.map(result => ({ tokens: result.tokens, prompt_tokens: result.prompt_tokens })),
        all: all.results.map(result => result.log_probs ?? result), again: again.results[0].log_probs ?? again.results[0],
        capabilities: engine.capabilities };
      if (!states || !engine.capabilities.nativeStateStore) return result;
      // the server's state, put after load
      const wire = Uint8Array.from(atob(states.wire), char => char.charCodeAt(0));
      const put = await engine.putNativeState(wire, states.id);
      const consumed = (await engine.createNatlangScore({ items: [states.consume] })).results[0];
      // a state the page produces, read back as stored
      const reply = await engine.createChatCompletion(states.produce);
      const id = reply.native_state?.blocks?.[0]?.id;
      const got = id ? await engine.getNativeState(id) : null;
      let binary = ''; for (const byte of got?.state ?? []) binary += String.fromCharCode(byte);
      return { ...result, states: { put, consumed: consumed.log_probs ?? consumed, id: id ?? null, meta: got?.meta ?? null,
        wire: got ? btoa(binary) : null, reply: id ? null : reply } };
    }, { items, states });
  } finally { await browser.close(); http.close(); rmSync(profile, { recursive: true, force: true }); }
}

async function crossCheck() {
  const { decodeNativeState } = await import(pathToFileURL(resolve(dirname(fileURLToPath(import.meta.url)),
    '../ts-host/dist/native/neuralese-store.js')).href);
  const got = page.states;
  if (got.put.mode !== 'exact-native-state') failures.push(`the page consumes the server's state as ${got.put.mode}`);
  if (!Array.isArray(got.consumed) || rank(got.consumed) !== rank(states.expected) ||
      Math.max(...got.consumed.map((score, option) => Math.abs(score - states.expected[option]))) > tolerance)
    failures.push(`a decision consuming the server's state: page ${JSON.stringify(got.consumed)}, server ${JSON.stringify(states.expected)}`);
  if (!got.wire) return failures.push(`the page produced no state: ${JSON.stringify(got.reply).slice(0, 400)}`);
  const wire = Buffer.from(got.wire, 'base64'), mine = decodeNativeState(new Uint8Array(wire), got.id);
  const theirs = decodeNativeState(new Uint8Array(Buffer.from(states.wire, 'base64')), states.id);
  // The same text; its tokens may differ: a grammar-forced span takes, of the tokenizations the grammar allows, the one
  // each engine's numerics rank first, and near ties fall differently in WebAssembly and native code.
  if (mine.metadata.text !== theirs.metadata.text)
    failures.push(`the page-produced state holds other text than the server's: ${JSON.stringify(mine.metadata.text)}`);
  const tokens = state => new BigInt64Array(state.tensors.token_ids.data.slice().buffer).join(' ');
  const sameTokens = tokens(mine) === tokens(theirs);
  for (const key of ['architecture', 'checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256'])
    if (mine.metadata[key] !== theirs.metadata[key]) failures.push(`the page-produced state's ${key} differs`);
  const stored = await fetch(`${base}/v1/neuralese/native-states/${got.id}`, { method: 'PUT',
    headers: { 'content-type': 'application/octet-stream' }, body: wire });
  if (stored.status !== 201) failures.push(`the server refused the page's state: ${stored.status} ${await stored.text()}`);
  const reply = await (await post('/v1/chat/completions', { messages: [{ role: 'user', content: [{ type: 'text', text: 'Which order? ' },
    { type: 'native_state', id: got.id }] }], max_tokens: 4, temperature: 0 })).json();
  if (!reply.native_state?.imports?.some(item => item.id === got.id && item.mode === 'exact-native-state'))
    failures.push(`the server does not consume the page's state exactly: ${JSON.stringify(reply).slice(0, 400)}`);
  states.page = { id: got.id, bytes: wire.length, same_id_as_server: got.id === states.id, same_tokens_as_server: sameTokens };
}
expected.forEach((scores, index) => {
  const mine = page.all[index];
  if (!Array.isArray(mine)) return failures.push(`item ${index}: ${JSON.stringify(mine)}`);
  const shape = result => JSON.stringify([result.tokens, result.prompt_tokens]);
  if (shape(page.shapes[index]) !== shape(nativeResults[index]))
    failures.push(`item ${index}: page renders or tokenizes differently: ${shape(page.shapes[index])}, native ${shape(nativeResults[index])}`);
  if (rank(mine) !== rank(scores)) failures.push(`item ${index}: page ranks ${rank(mine)}, native ${rank(scores)}`);
  const gap = Math.max(...mine.map((score, option) => Math.abs(score - scores[option])));
  if (gap > tolerance) failures.push(`item ${index}: page differs by ${gap.toFixed(3)} nats`);
});
if (rank(page.again) !== rank(page.all[1])) failures.push('a repeated item ranks its options differently');
const drift = Math.max(...page.again.map((score, option) => Math.abs(score - page.all[1][option])));
if (drift > tolerance) failures.push(`a repeated item moved by ${drift.toFixed(3)} nats`);
if (states && !page.states) failures.push('the natlang GGUF\'s native states were not checked: the page build has no nativeStateStore');
console.log(JSON.stringify({ native: expected, page: page.all, repeated: page.again, capabilities: page.capabilities,
  ...(states ? { native_states: { server: { id: states.id, consumed: states.expected }, page: { ...states.page, consumed: page.states?.consumed } } } : {}),
  failures }));
process.exit(failures.length ? 1 : 0);
