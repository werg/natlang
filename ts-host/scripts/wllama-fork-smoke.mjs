#!/usr/bin/env node
/**
 * Real in-browser check of the llama.cpp fork's native state spans and decision readout in a wllama build
 * (docs/TUNING.md "wllama").
 *
 *   node scripts/wllama-fork-smoke.mjs --wllama <built wllama package dir> --base base.gguf \
 *     --state state.gguf --control control.gguf [--server <llama-server of the fork>]
 *
 * `state` is a native state span that text cannot reproduce (scripts/llama_state_span_parity.py writes a perturbed
 * one as state.gguf); `control` is the unperturbed state of the same text (state-text.gguf there). Headless Chromium
 * loads the base and both states as blobs and greedily answers one chat prompt that consumes each state through a
 * `{type: "native_state", id}` content part, the text in plain text, and the state again after other requests, with
 * the log-probability of every answer token. With `--server`, the fork's native llama-server answers the same
 * requests (`--native-state`); the browser must match it token for token and within `--tolerance` in log-probability.
 * Then both score the same replies (`createNatlangScore`, `POST /v1/natlang/score`), with and without a state.
 */
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { createReadStream, readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { chromium } from 'playwright-core';
import { chromiumPath } from './chromium-path.mjs';

const { values } = parseArgs({ options: { wllama: { type: 'string' }, base: { type: 'string' }, state: { type: 'string' },
  control: { type: 'string' }, server: { type: 'string' }, text: { type: 'string',
    default: 'Answer only with the capital city of the country named in the question, in lowercase.' },
  question: { type: 'string', default: 'France?' }, tokens: { type: 'string', default: '12' },
  tolerance: { type: 'string', default: '0.05' } } });
if (!values.wllama || !values.base || !values.state || !values.control) {
  console.error('usage: wllama-fork-smoke.mjs --wllama DIR --base BASE.gguf --state STATE.gguf --control CONTROL.gguf [--server LLAMA_SERVER]');
  process.exit(2);
}
// The state ID is the GGUF's natlang.state.id string (common/native-state.h).
const stateId = path => {
  const bytes = readFileSync(path), key = Buffer.from('natlang.state.id');
  const at = bytes.indexOf(key);
  if (at < 0) throw new Error(`${path} has no natlang.state.id`);
  const offset = at + key.length + 4; // value type (u32 string)
  const length = Number(bytes.readBigUInt64LE(offset));
  return bytes.subarray(offset + 8, offset + 8 + length).toString('utf8');
};
const ids = { state: stateId(values.state), control: stateId(values.control) };
const messages = part => [{ role: 'user', content: [{ type: 'text', text: 'Instructions: ' }, part, { type: 'text', text: `\nQuestion: ${values.question}` }] }];
const requests = {
  state: messages({ type: 'native_state', id: ids.state }),
  control: messages({ type: 'native_state', id: ids.control }),
  text: [{ role: 'user', content: `Instructions: ${values.text}\nQuestion: ${values.question}` }],
};
const order = ['state', 'control', 'text', 'stateAgain', 'missing'];
const scoring = { items: [{ messages: requests.text, continuations: ['paris', 'Paris', 'London'] },
  { messages: requests.state, continuations: ['paris', 'France is a country located in Western Europe.'] },
  { messages: 'not messages', continuations: ['x'] }] };
const body = name => ({ messages: name === 'stateAgain' ? requests.state : name === 'missing'
  ? messages({ type: 'native_state', id: 'nzs1_not_loaded' }) : requests[name], max_tokens: Number(values.tokens), temperature: 0, top_k: 1,
  logprobs: true });
// The answer text and the log-probability of each answer token.
const answer = response => ({ text: response.choices[0].message.content,
  logprobs: (response.choices[0].logprobs?.content ?? []).map(item => item.logprob) });

const files = { '/base.gguf': resolve(values.base), '/state.gguf': resolve(values.state), '/control.gguf': resolve(values.control) };
const wllamaDir = resolve(values.wllama);
const page = `<!doctype html><meta charset="utf-8"><script type="module">
import { Wllama } from '/wllama/esm/index.js';
window.run = async ({ bodies, scoring }) => {
  const wllama = new Wllama({ default: '/wllama/esm/wasm/wllama.wasm' }, { allowOffline: true });
  if (!wllama.capabilities?.nativeStates) throw new Error('this wllama build cannot load native states');
  const blob = async path => (await fetch(path)).blob();
  await wllama.loadModel([await blob('/base.gguf')], { n_ctx: 512, n_gpu_layers: 0, n_parallel: 1, cache_type_k: 'f32', cache_type_v: 'f32',
    native_states: [{ blob: await blob('/state.gguf') }, { blob: await blob('/control.gguf') }] });
  const out = {};
  console.log('loaded');
  for (const [name, body] of bodies) {
    console.log('request ' + name);
    try {
      const response = await wllama.createChatCompletion({ ...body, cache_prompt: true });
      out[name] = { text: response.choices[0].message.content,
        logprobs: (response.choices[0].logprobs?.content ?? []).map(item => item.logprob) };
    } catch (error) { out[name] = { error: String(error.message ?? error) }; }
  }
  console.log('scoring');
  out.scores = (await wllama.createNatlangScore(scoring)).results;
  await wllama.exit();
  return out;
};
window.ready = true;
</script>`;
const server = createServer((request, response) => {
  const path = new URL(request.url, 'http://x').pathname;
  const headers = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' };
  if (path === '/') { response.writeHead(200, { ...headers, 'content-type': 'text/html' }); response.end(page); return; }
  const file = files[path] ?? (path.startsWith('/wllama/') ? resolve(wllamaDir, '.' + path.slice('/wllama'.length)) : null);
  if (!file || (path.startsWith('/wllama/') && !file.startsWith(wllamaDir))) { response.writeHead(404); response.end(); return; }
  try { statSync(file); } catch { response.writeHead(404); response.end(); return; }
  const type = file.endsWith('.js') ? 'text/javascript' : file.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream';
  response.writeHead(200, { ...headers, 'content-type': type, 'content-length': statSync(file).size });
  createReadStream(file).pipe(response);
});

/** The same requests on the fork's native llama-server. */
async function native() {
  const port = 18000 + Math.floor(Math.random() * 1000);
  const child = spawn(values.server, ['-m', values.base, '--port', String(port), '-c', '512', '-np', '1', '-ctk', 'f32', '-ctv', 'f32',
    '--jinja', '--native-state', values.state, '--native-state', values.control], { stdio: ['ignore', 'ignore', 'pipe'] });
  let log = ''; child.stderr.on('data', chunk => { log += chunk; });
  try {
    for (let i = 0; ; i++) {
      if (i > 600) throw new Error('llama-server did not start:\n' + log.slice(-2000));
      try { if ((await fetch(`http://127.0.0.1:${port}/health`)).ok) break; } catch { /* starting */ }
      await new Promise(done => setTimeout(done, 200));
    }
    const out = {};
    for (const name of order) {
      const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body(name)) });
      const json = await response.json();
      out[name] = response.ok ? answer(json) : { error: json.error?.message ?? response.status };
    }
    const scored = await fetch(`http://127.0.0.1:${port}/v1/natlang/score`, { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify(scoring) });
    out.scores = (await scored.json()).results;
    return out;
  } finally { child.kill('SIGTERM'); }
}

await new Promise(done => server.listen(0, '127.0.0.1', done));
const browser = await chromium.launch({ executablePath: chromiumPath(), args: ['--enable-features=SharedArrayBuffer'] });
let failed = false;
try {
  const tab = await browser.newPage();
  tab.on('console', message => { if (message.type() === 'error' || message.text().match(/^(loaded|request|scoring)/))
    console.error('[page]', message.text()); });
  await tab.goto(`http://127.0.0.1:${server.address().port}/`);
  await tab.waitForFunction(() => window.ready);
  const started = Date.now();
  const out = await tab.evaluate(options => window.run(options), { bodies: order.map(name => [name, body(name)]), scoring });
  const reference = values.server ? await native() : undefined;
  console.log(JSON.stringify({ seconds: (Date.now() - started) / 1000, ids, browser: out, ...(reference ? { native: reference } : {}) }, null, 1));
  const answered = value => typeof value?.text === 'string' && value.logprobs.length > 0;
  // The largest log-probability difference over the tokens two greedy answers share.
  const gap = (a, b) => Math.max(...a.logprobs.slice(0, b.logprobs.length).map((value, i) => Math.abs(value - b.logprobs[i])));
  const tolerance = Number(values.tolerance);
  const checks = {
    'the state is consumed': answered(out.state) && answered(out.control) && answered(out.text),
    'the perturbed state is read differently from its text state': answered(out.state) && gap(out.state, out.control) > tolerance,
    'the state is reproducible after other requests': answered(out.stateAgain) && out.stateAgain.text === out.state.text &&
      gap(out.stateAgain, out.state) <= tolerance,
    'a state that is not loaded is refused': !answered(out.missing) && /not loaded/.test(out.missing?.error),
    'decision readout scores each reply': out.scores?.[0]?.log_probs?.length === 3 && out.scores[1]?.log_probs?.length === 2 &&
      out.scores[0].log_probs.every(value => Number.isFinite(value) && value < 0),
    'decision readout reads the state': out.scores?.[1]?.log_probs?.every(Number.isFinite),
    'an item that cannot be rendered fails alone': typeof out.scores?.[2]?.error === 'string',
    ...(reference ? Object.fromEntries(order.slice(0, 4).map(name => [`${name}: the browser equals the native server`,
      answered(out[name]) && out[name].text === reference[name]?.text && gap(out[name], reference[name]) <= tolerance])) : {}),
    ...(reference ? { 'decision readout: the browser equals the native server': [0, 1].every(item =>
      out.scores?.[item]?.tokens?.join() === reference.scores?.[item]?.tokens?.join() &&
      out.scores[item].log_probs.every((value, k) => Math.abs(value - reference.scores[item].log_probs[k]) <= tolerance)) } : {}),
  };
  for (const [name, ok] of Object.entries(checks)) { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}`); failed ||= !ok; }
} finally { await browser.close(); server.close(); }
process.exit(failed ? 1 : 0);
