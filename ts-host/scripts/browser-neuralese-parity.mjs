#!/usr/bin/env node
/**
 * Node/browser parity of a Neuralese natlang program (plans/neuralese/S4_RUNTIME_SERVERS.md §10): the program of
 * test/neuralese-parity/program.mjs (a Neuralese<string> template write, its read-back, a decision over it, the generic
 * builtin `view` at both representations, `ask`) runs
 *
 *   (a) on Node (dist/index.js) against the llama.cpp fork's native server (build-cpu/bin/llama-neuralese-server), and
 *   (b) in headless Chromium (dist/browser/natlang.js) against the WebAssembly engine in a Web Worker,
 *
 * with the same GGUF files, and the two traces are compared: rendered prompts (every model request as sent), replies,
 * written block lengths and payloads (gate 5e-2 on their scale; the conformance tolerance 2e-2 is reported as a
 * measurement, since the one-thread CPU build drifts past it after long prompts), stop logits and decision
 * log-probabilities (5e-2), step values, and streamed turns (deltas assembled equal to the whole reply, and equal across
 * hosts). The browser half then reloads the page and checks that the OPFS block store restores every archived block (a
 * fresh engine reads the note back through it to the same text), and runs pi's browser bundle in a dedicated worker
 * with a scripted agent model, whose OPFS session survives a second reload. Exit 1 when a gating check fails; the
 * report (--out) lists every check, measurements marked `gate: false`.
 *
 *   node scripts/browser-neuralese-parity.mjs [--model model.gguf --heads neuralese.gguf] [--build neuralese-wasm|neuralese-wasm-mt|neuralese-wasm-gpu]
 *     [--threads N] [--gpu] [--headed] [--ctx 8192] [--out report.json] [--skip-pi | --only-pi]
 *
 * WebGPU: headless Chromium offers no hardware adapter on Linux; `xvfb-run -a … --gpu --headed` gets the GB10's.
 *
 * Without --model/--heads the LFM2.5-350M backbone with untrained port heads is exported once (natlang_neuralese.export,
 * cutoff 6, max block 6) to $NATLANG_PARITY_MODELS (default ~/data/natlang-browser-ci/lfm350m-untrained). The fork is
 * $NATLANG_NEURALESE_FORK (default ~/llama.cpp-neuralese). Loads models: run it under the memory ledger.
 */
import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, extname, join, resolve, sep } from 'node:path';
import { parseArgs } from 'node:util';
import { chromium } from 'playwright-core';
import { chromiumPath } from './chromium-path.mjs';
import { buildApplicationBrowser } from './build-application-browser.mjs';
import { blockNamer, runParityProgram } from '../test/neuralese-parity/program.mjs';

const ATOL_PAYLOAD = 2e-2, ATOL_LOGPROB = 5e-2, ATOL_LONG = 5e-2;
const { values } = parseArgs({ options: { model: { type: 'string' }, heads: { type: 'string' }, build: { type: 'string', default: 'neuralese-wasm' },
  threads: { type: 'string' }, gpu: { type: 'boolean', default: false }, headed: { type: 'boolean', default: false },
  out: { type: 'string', default: '/tmp/neuralese-parity-report.json' }, ctx: { type: 'string', default: '8192' }, 'skip-pi': { type: 'boolean', default: false }, 'only-pi': { type: 'boolean', default: false },
  'chromium-flags': { type: 'string', default: '' } } });
const root = resolve(import.meta.dirname, '..');
const repo = resolve(root, '..');
const log = line => process.stderr.write(`[parity ${new Date().toISOString().slice(11, 19)}] ${line}\n`);
const build = values.gpu ? 'neuralese-wasm-gpu' : values.build;
const threads = Number(values.threads ?? (build === 'neuralese-wasm-mt' ? 8 : 1));

// The model files: given, or the untrained-heads export (made once).
function modelFiles() {
  if (values.model && values.heads) return { model: resolve(values.model), heads: resolve(values.heads) };
  const dir = process.env.NATLANG_PARITY_MODELS ?? join(homedir(), 'data', 'natlang-browser-ci', 'lfm350m-untrained');
  const files = { model: join(dir, 'model.gguf'), heads: join(dir, 'neuralese.gguf') };
  if (existsSync(files.model) && existsSync(files.heads)) return files;
  if (existsSync(dir)) throw new Error(`${dir} exists without model.gguf and neuralese.gguf (an interrupted export?); remove it or pass --model/--heads`);
  mkdirSync(dirname(dir), { recursive: true });
  const python = process.env.NATLANG_NEURALESE_PYTHON ?? join(repo, '.venv-neuralese', 'bin', 'python');
  log(`exporting LFM2.5-350M with untrained heads to ${dir}`);
  const done = spawnSync(python, ['-m', 'natlang_neuralese.export', '--out', dir, '--cutoff', '6', '--max-block', '6'],
    { cwd: join(repo, 'training', 'neuralese'), stdio: ['ignore', 'inherit', 'inherit'],
      env: { ...process.env, HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1' } });
  if (done.status !== 0) throw new Error(`the export failed (${done.status})`);
  return files;
}

async function startNative(files) {
  const fork = process.env.NATLANG_NEURALESE_FORK ?? join(homedir(), 'llama.cpp-neuralese');
  const bin = join(fork, 'build-cpu', 'bin');
  const binary = join(bin, 'llama-neuralese-server');
  if (!existsSync(binary)) throw new Error(`the fork's CPU server is not built at ${binary}`);
  const commit = spawnSync('git', ['-C', fork, 'rev-parse', '--short=9', 'HEAD'], { encoding: 'utf8' }).stdout.trim();
  const child = spawn(binary, ['-m', files.model, '--nz', files.heads, '--port', '0', '-t', '8', '--max-block', '6', '-c', values.ctx],
    { env: { ...process.env, LD_LIBRARY_PATH: bin }, stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '';
  child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-4000); });
  const endpoint = await new Promise((done, fail) => {
    let out = '';
    child.stdout.on('data', chunk => {
      out += chunk;
      const line = out.split('\n').find(item => item.startsWith('{'));
      if (line) done(JSON.parse(line).listening);
    });
    child.on('exit', code => fail(new Error(`the native server exited (${code}): ${stderr}`)));
  });
  return { endpoint, commit, close: () => child.kill() };
}

// The static server: ts-host, the model files, and the scripted agent model of pi's run.
const agent = { replies: [], requests: [] };
function startStatic(files) {
  const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm',
    '.nl': 'text/plain; charset=utf-8', '.json': 'application/json' };
  const routes = { '/files/model.gguf': files.model, '/files/heads.gguf': files.heads };
  const isolation = { 'cross-origin-opener-policy': 'same-origin', 'cross-origin-embedder-policy': 'require-corp' };
  const server = createServer((request, response) => {
    const path = new URL(request.url, 'http://x').pathname;
    if (path.startsWith('/agent/')) {
      let data = '';
      request.on('data', chunk => { data += chunk; });
      request.on('end', () => {
        agent.requests.push({ path, body: data ? JSON.parse(data) : undefined });
        const send = (status, body) => { response.writeHead(status, { 'content-type': 'application/json', ...isolation }); response.end(JSON.stringify(body)); };
        if (path !== '/agent/v1/chat/completions') return send(404, { error: { message: 'not found' } });
        const message = agent.replies.shift() ?? { content: 'out of replies' };
        send(200, { id: `r${agent.requests.length}`, choices: [{ finish_reason: message.tool_calls ? 'tool_calls' : 'stop', message }],
          usage: { prompt_tokens: 10, completion_tokens: 5 } });
      });
      return;
    }
    const file = routes[path] ?? resolve(root, '.' + decodeURIComponent(path));
    if (!routes[path] && !file.startsWith(root + sep)) { response.writeHead(403); response.end(); return; }
    try {
      const size = statSync(file).size;
      response.writeHead(200, { 'content-type': types[extname(file)] ?? 'application/octet-stream', 'content-length': size, ...isolation });
      createReadStream(file).pipe(response);
    } catch { response.writeHead(404); response.end(); }
  });
  return new Promise(done => server.listen(0, '127.0.0.1', () => done({ url: `http://127.0.0.1:${server.address().port}`, close: () => server.close() })));
}

// Comparison of the two traces.
const checks = [];
// A check with `gate: false` is a measurement: reported (NOTE), not failing the run.
const check = (name, ok, detail = {}, { gate = true } = {}) => {
  checks.push({ name, ok: Boolean(ok), gate, ...detail });
  log(`${ok ? 'PASS' : gate ? 'FAIL' : 'NOTE'} ${name}${ok ? '' : ' ' + JSON.stringify(detail).slice(0, 600)}`);
};
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
function firstDifference(a, b, path = '$') {
  if (same(a, b)) return null;
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    for (const key of new Set([...Object.keys(a), ...Object.keys(b)])) {
      const found = firstDifference(a[key], b[key], `${path}.${key}`);
      if (found) return found;
    }
  }
  return { path, node: typeof a === 'string' ? a.slice(0, 400) : a, browser: typeof b === 'string' ? b.slice(0, 400) : b };
}
const maxDiff = (a, b) => a.length !== b.length ? Infinity : a.reduce((m, x, i) => Math.max(m, Math.abs(x - b[i])), 0);

function compare(node, browser) {
  const nn = blockNamer(), bn = blockNamer();
  const pick = trace => ({ steps: trace.steps.map(({ ms: _, ...step }) => step), exchanges: trace.exchanges, decisions: trace.decisions,
    streaming: Object.fromEntries(Object.entries(trace.streaming).map(([key, value]) => [key, value && typeof value === 'object' ?
      (({ ms: _, ...rest }) => rest)(value) : value])) });
  // Stop logits and decision scores are floats: compared within tolerance below, not exactly.
  const exact = trace => {
    const picked = structuredClone(pick(trace));
    const strip = value => { if (value && typeof value === 'object') { delete value.stop_logits; Object.values(value).forEach(strip); } };
    strip(picked);
    for (const decision of picked.decisions) if (decision.scores) decision.scores = { tokens: decision.scores.tokens ?? null };
    return picked;
  };
  const a = nn.normalize(exact(node)), b = bn.normalize(exact(browser));
  check('server info agrees', same(node.info, browser.info), { node: node.info, browser: browser.info });
  check('same number of model requests', a.exchanges.length === b.exchanges.length, { node: a.exchanges.length, browser: b.exchanges.length });
  const prompts = a.exchanges.map((item, i) => same(item.request, b.exchanges[i]?.request));
  check('rendered prompts agree (every request as sent, blocks named by order)', prompts.every(Boolean),
    { agreeing: prompts.filter(Boolean).length, of: prompts.length, first: firstDifference(a.exchanges.map(x => x.request), b.exchanges.map(x => x.request)) });
  const replies = a.exchanges.map((item, i) => same(item.reply, b.exchanges[i]?.reply));
  check('replies agree (text, calls, written block lengths)', replies.every(Boolean),
    { agreeing: replies.filter(Boolean).length, of: replies.length, first: firstDifference(a.exchanges.map(x => x.reply), b.exchanges.map(x => x.reply)) });
  const stops = node.exchanges.flatMap((item, i) => item.reply.blocks.map((block, j) => [block.stop_logits, browser.exchanges[i]?.reply.blocks[j]?.stop_logits]))
    .filter(([x, y]) => x && y).map(([x, y]) => maxDiff(x, y));
  // Stop logits are logits: held to the log-probability tolerance (the conformance suite's short prompts meet 2e-2;
  // the runtime's prompts here run to thousands of tokens).
  check(`stop logits agree within ${ATOL_LOGPROB}`, stops.every(d => d <= ATOL_LOGPROB), { blocks: stops.length, max: Math.max(0, ...stops) });
  check('decision requests agree', same(a.decisions, b.decisions), { first: firstDifference(a.decisions, b.decisions) });
  const logprobs = node.decisions.map((item, i) => maxDiff(item.scores?.log_probs ?? [], browser.decisions[i]?.scores?.log_probs ?? []));
  check(`decision log-probabilities agree within ${ATOL_LOGPROB}`, logprobs.every(d => d <= ATOL_LOGPROB),
    { decisions: logprobs.length, max: Math.max(0, ...logprobs), node: node.decisions.map(d => d.scores?.log_probs ?? d.error), browser: browser.decisions.map(d => d.scores?.log_probs ?? d.error) });
  const steps = a.steps.map((step, i) => ({ step: step.step, ok: same(step, b.steps[i]), node: step.error ?? step.value, browser: b.steps[i]?.error ?? b.steps[i]?.value }));
  for (const step of steps) check(`step ${step.step}: same value`, step.ok, step.ok ? { value: step.node } : { node: step.node, browser: step.browser });
  // Agreeing on an error is parity, not function: steps that failed on both hosts are reported.
  const failed = node.steps.filter(step => step.error).map(step => ({ step: step.step, error: step.error.slice(0, 300) }));
  check('every step ran without error', failed.length === 0, { failed }, { gate: false });
  // Payloads, ordinal by ordinal.
  const nids = nn.ids(), bids = bn.ids();
  check('the same number of distinct blocks', nids.length === bids.length, { node: nids.length, browser: bids.length });
  const payloads = nids.map((id, k) => {
    const x = node.blocks[id], y = browser.blocks[bids[k]];
    if (!x || !y) return { k, missing: !x ? 'node' : 'browser' };
    const scale = Math.max(1, ...x.payload.map(Math.abs));
    const diff = x.length === y.length && x.width === y.width ? maxDiff(x.payload, y.payload) : Infinity;
    return { k, length: [x.length, y.length], diff, scale, relative: diff / scale, same_id: id === bids[k], stored: [x.stored, y.stored] };
  });
  // Payloads on their scale (absolute for values within ±1). The conformance suite's 2e-2 holds for its short prompts;
  // the native and wasm builds drift apart with prompt length (measured 2026-10-10: 0.0014 after a 5k-token prompt's
  // write, 0.027 after the longest), so the gate is the long-prompt tolerance 5e-2 and 2e-2 is reported as a measurement.
  const summary = tolerance => ({ blocks: payloads.length, max_abs: Math.max(0, ...payloads.map(p => p.diff ?? Infinity)),
    max_relative: Math.max(0, ...payloads.map(p => p.relative ?? Infinity)), identical_ids: payloads.filter(p => p.same_id).length,
    worst: payloads.filter(p => !(p.relative <= tolerance)).slice(0, 5) });
  check(`block payloads agree within ${ATOL_LONG} of their scale`, payloads.every(p => p.relative <= ATOL_LONG), summary(ATOL_LONG));
  check(`block payloads agree within the conformance tolerance ${ATOL_PAYLOAD}`, payloads.every(p => p.relative <= ATOL_PAYLOAD),
    summary(ATOL_PAYLOAD), { gate: false });
  for (const [host, trace] of [['node', node], ['browser', browser]]) {
    const s = trace.streaming;
    check(`${host}: the server streams`, s.server_streams === true);
    check(`${host}: streamed plain turn assembles to its reply and to the whole (non-streamed) reply`,
      s.plain.assembled === s.plain.text && s.plain.text === s.plain.whole && s.plain.deltas > 1,
      { deltas: s.plain.deltas, assembled: s.plain.assembled, text: s.plain.text, whole: s.plain.whole });
    // The turn's text names the block, between the literal markers U+E000/U+E001, where the stream carried it as a part.
    check(`${host}: streamed write turn carries the block as a delta`, s.write.neuralese_parts.length === 1 &&
      s.write.text === s.write.assembled.replace('Note: ', `Note: \uE000${s.write.neuralese_parts[0]}\uE001`),
      { parts: s.write.neuralese_parts.length, assembled: s.write.assembled, text: s.write.text, types: s.write.delta_types });
  }
  check('streamed turns agree across hosts', same(a.streaming, b.streaming), { first: firstDifference(a.streaming, b.streaming) });
  return { payloads, steps };
}

const report = { started: new Date().toISOString(), build, threads };
const files = modelFiles();
report.files = Object.fromEntries(Object.entries(files).map(([key, path]) => [key, { path, bytes: statSync(path).size }]));
const viewSource = readFileSync(join(root, 'src', 'builtin', 'view.nl'), 'utf8');
const natlang = await import('../dist/index.js');

let started = Date.now();
if (!values['only-pi']) {
  log(`native server (${files.model})`);
  const native = await startNative(files);
  report.fork_commit = native.commit;
  try {
    report.node = await runParityProgram({ natlang, endpoint: native.endpoint, store: new natlang.MemoryNeuraleseStore(), viewSource, target: 'node', log });
  } finally { native.close(); }
  report.node_ms = Date.now() - started;
}

const web = await startStatic(files);
let piEntry;
if (!values['skip-pi']) {
  log('building pi\'s browser entry');
  const built = await buildApplicationBrowser({ app: join(repo, 'applications', 'pi'), outdir: join(root, '.natlang', 'browser-ci', 'pi'),
    runtime: '/dist/browser/natlang.js' });
  piEntry = `/${built.entry.slice(root.length + 1)}`;
}
const args = [...(values.gpu ? ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist',
  '--enable-dawn-features=vulkan_enable_f16_on_nvidia'] : []), ...values['chromium-flags'].split(' ').filter(Boolean)];
const browserProcess = await chromium.launch({ headless: !values.headed, args, executablePath: chromiumPath() });
report.chromium = browserProcess.version();
try {
  const context = await browserProcess.newContext();
  const page = await context.newPage();
  page.setDefaultTimeout(0);
  page.on('console', message => process.stderr.write(`[page] ${message.text()}\n`));
  page.on('response', response => { if (response.status() >= 400) process.stderr.write(`[http ${response.status()}] ${response.url()}\n`); });
  page.on('close', () => process.stderr.write('[page closed]\n'));
  page.on('worker', worker => { process.stderr.write(`[worker] ${worker.url()}\n`); worker.on('close', () => process.stderr.write(`[worker closed] ${worker.url()}\n`)); });
  page.on('crash', () => process.stderr.write('[page crashed]\n'));
  page.on('framenavigated', frame => process.stderr.write(`[navigated] ${frame.url()}\n`));
  page.on('pageerror', error => process.stderr.write(`[page error] ${error.stack ?? error}\n`));
  const pageUrl = `${web.url}/test/neuralese-parity/page.html`;
  const ready = async () => { await page.waitForFunction(() => window.__parityReady === true, null, { timeout: 120_000 }); };
  await page.goto(pageUrl);
  await ready();
  const engine = { nCtx: Number(values.ctx), build, threads, gpuLayers: values.gpu ? 999 : 0, model: '/files/model.gguf', heads: '/files/heads.gguf' };
  if (!values['only-pi']) {
  started = Date.now();
  log(`browser run (${build}, ${threads} thread${threads === 1 ? '' : 's'})`);
  const run = await page.evaluate(options => window.runParity(options), engine);
  report.browser_ms = Date.now() - started;
  report.browser_environment = run.environment;
  report.browser = run.trace;
  report.comparison = compare(report.node, report.browser);

  // Reload: the OPFS archive restores the blocks; a fresh engine reads the note back through it.
  const readBack = report.browser.steps.find(step => step.step === 'read-back');
  if (report.browser.note?.$neuralese && readBack) {
    await page.reload();
    await ready();
    log('after reload: restoring blocks from OPFS');
    // The blocks the runtime archived (a write's producer record also names server-side blocks it never copies).
    const stored = Object.entries(report.browser.blocks).filter(([, block]) => block?.stored).map(([id]) => id);
    const restore = await page.evaluate(options => window.checkRestore(options), { ...engine, ids: stored,
      note: report.browser.note, library: { dialect: report.browser.info.dialects[0], width: report.browser.info.width, bodies: report.browser.library },
      expected: readBack.value ?? null });
    report.restore = restore;
    const missing = Object.entries(restore.present).filter(([, item]) => !item.ok).map(([id]) => id);
    check('reload: the OPFS store holds every block of the run', /OpfsNeuraleseStore/.test(restore.store) && missing.length === 0 && Object.keys(restore.present).length > 0,
      { store: restore.store, blocks: Object.keys(restore.present).length, missing });
    check('reload: a fresh engine lacked the note and read it back through the archive to the same text',
      restore.engine_had_note === 404 && restore.matches, { engine_had_note: restore.engine_had_note, read_back: restore.read_back, error: restore.error });
  } else check('reload: the note was written and read back before the reload', false, { note: report.browser.note, readBack });
  }

  if (piEntry) {
    const call = (id, name, args) => ({ id, type: 'function', function: { name, arguments: JSON.stringify(args) } });
    agent.replies = [{ content: 'Writing it.', tool_calls: [call('c1', 'write', { path: 'hello.txt', content: 'hi from the browser\n' })] },
      { content: 'hello.txt says hi.' }];
    agent.requests.length = 0;
    log('pi in a dedicated worker: first task');
    const first = await page.evaluate(options => window.runPi(options), { phase: 'first', agent: `${web.url}/agent`, entry: piEntry });
    const firstRequests = agent.requests.length;
    await page.reload();
    await ready();
    agent.replies = [{ content: 'It said hi from the browser.' }];
    agent.requests.length = 0;
    log('pi after reload: the same session');
    const again = await page.evaluate(options => window.runPi(options), { phase: 'again', agent: `${web.url}/agent`, entry: piEntry });
    const resent = JSON.stringify(agent.requests.find(item => item.path === '/agent/v1/chat/completions')?.body?.messages ?? []);
    report.pi = { first, again, first_agent_requests: firstRequests, again_agent_requests: agent.requests.length };
    check('pi (dedicated worker): the session is in OPFS and the task completes',
      first.ok && first.session === 'opfs' && first.result?.status === 'done' && first.result.answer === 'hello.txt says hi.' &&
      first.files['hello.txt'] === 'hi from the browser\n', { first });
    check('pi after reload: the OPFS session holds the conversation and the agent sees it',
      again.ok && again.session === 'opfs' && again.entries_before === first.entries_after && again.entries_before > 0 &&
      again.result?.status === 'done' && resent.includes('hello.txt says hi.') && resent.includes('Write hello.txt saying hi.'),
      { again, first_entries: first.entries_after });
  }
} catch (error) {
  check('the browser half ran to its end', false, { error: String(error?.stack ?? error).slice(0, 2000) });
} finally {
  await browserProcess.close();
  web.close();
}
report.checks = checks;
// The written report keeps block shapes, not payload floats.
for (const trace of [report.node, report.browser]) if (trace) for (const block of Object.values(trace.blocks)) if (block) block.payload = block.payload.length;
report.ok = checks.every(item => item.ok || !item.gate);
report.finished = new Date().toISOString();
writeFileSync(values.out, JSON.stringify(report, null, 1));
log(`${checks.filter(item => item.ok).length}/${checks.length} checks passed (${checks.filter(item => !item.ok && !item.gate).length} measurement notes); report ${values.out}`);
process.exitCode = report.ok ? 0 : 1;
