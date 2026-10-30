/**
 * Embedding contract of the dashboard pages (dashboard/types.d.ts): every page mounts into a bare host element with a fake
 * DashboardApi and an inProcessRunHost over a scripted runtime, draws, and disposes without leaving an interval, a pending
 * timer, a listener on the document, a run watcher or an open dialog behind. A host that omits a capability is not offered
 * the pages that need it, and the dashboard's CSS neither leaks out of its root nor beats the host's own styles.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFileSync, existsSync } from 'node:fs';
import { extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { chromiumPath } from './chromium-path.mjs';

const ASSETS = fileURLToPath(new URL('../dashboard/', import.meta.url));
const TYPES = { '.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html' };
const HARNESS = `<!doctype html><meta charset="utf-8"><title>embed</title>
<link rel="icon" href="data:,"><link rel="stylesheet" href="/style.css">
<style>
  /* The host's own, unlayered styles: they must beat the dashboard's layered ones and be untouched by them. */
  h1 { font-size: 31px; }
  .nl-dash .panel { padding: 3px; }
</style>
<h1 id="outside">Host heading</h1><button id="outside-button">Host button</button><div id="slot"></div>`;
const server = createServer((request, response) => {
  const path = new URL(request.url, 'http://x').pathname;
  if (path === '/embed.html') { response.writeHead(200, { 'content-type': 'text/html' }); response.end(HARNESS); return; }
  const file = normalize(join(ASSETS, path));
  if (!file.startsWith(ASSETS.replace(/[\\/]$/, '') + sep) || !existsSync(file)) { response.writeHead(404); response.end(); return; }
  response.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream' }); response.end(readFileSync(file));
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
let browser;
try {
  browser = await chromium.launch({ executablePath: chromiumPath(), headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const problems = [];
  page.on('pageerror', error => problems.push(`pageerror: ${error.message}`));
  page.on('console', message => { if (message.type() === 'error') problems.push(`console: ${message.text()}`); });
  // Count what outlives a page: intervals and pending timeouts, and listeners on the document, window and host.
  await page.addInitScript(() => {
    const intervals = new Set(), timeouts = new Set(), listeners = new Map();
    const { setInterval: si, clearInterval: ci, setTimeout: st, clearTimeout: ct } = window;
    window.setInterval = (...args) => { const id = si(...args); intervals.add(id); return id; };
    window.clearInterval = id => { intervals.delete(id); ci(id); };
    window.setTimeout = (fn, ...rest) => { const id = st(() => { timeouts.delete(id); fn(); }, ...rest); timeouts.add(id); return id; };
    window.clearTimeout = id => { timeouts.delete(id); ct(id); };
    const { addEventListener: add, removeEventListener: remove } = EventTarget.prototype;
    const long = target => target === window || target === document || target === document.body || target === document.documentElement;
    EventTarget.prototype.addEventListener = function (type, fn, options) {
      // Only listeners the dashboard's own modules add (Playwright adds its own to the window).
      if ((long(this) || this.id === 'slot') && new Error().stack.includes(`${location.origin}/`)) { const key = `${this === window ? 'window' : this.nodeName ?? 'target'}:${type}`; (listeners.get(fn) ?? listeners.set(fn, new Set()).get(fn)).add(key); }
      return add.call(this, type, fn, options);
    };
    EventTarget.prototype.removeEventListener = function (type, fn, options) {
      const keys = listeners.get(fn); keys?.delete(`${this === window ? 'window' : this.nodeName ?? 'target'}:${type}`);
      return remove.call(this, type, fn, options);
    };
    window.__leaks = () => ({ intervals: intervals.size, timeouts: timeouts.size, listeners: [...listeners.values()].flatMap(keys => [...keys]) });
  });
  await page.goto(`${origin}/embed.html`);
  const hostStyle = () => page.evaluate(() => { const style = getComputedStyle(document.getElementById('outside-button')); return [style.borderRadius, style.backgroundColor, style.padding, style.fontSize]; });
  const outsideBefore = await hostStyle();
  await page.evaluate(async origin => {
    const [{ availablePages, inProcessRunHost }, { createH }] = await Promise.all([import(`${origin}/pages.js`), import(`${origin}/common.js`)]);
    const now = new Date().toISOString();
    const prim = name => ({ kind: 'prim', name });
    const argument = { name: 'input', type: 'string', text: 'string', required: true, tree: prim('string') };
    const stock = { id: 'lfm3-1.7b', label: 'LFM3 1.7B', bytes: 1.7e9, installed: true, default: true, path: '/m', download: null, source: { repository: 'repo', revision: 'abcdef0123456789' } };
    const ok = value => async () => structuredClone(value);
    // A DashboardApi over fixtures: reads return data; writes record what was asked.
    const writes = [];
    const write = name => async (...args) => { writes.push([name, ...args]); return { ok: true }; };
    const api = {
      status: { get: ok({ version: 'x', node: 'v24', root: '/r', configPath: '/c', model: { profile: 'default', kind: 'managed-local', model: 'lfm3-1.7b', release: stock },
        calls: { root: '/store', calls: 3, bytes: 2048, settings: { specialization: 'off' } },
        runtime: { selected: { source: 'fork', version: 'b1', build: 1, path: '/llama', fork: true }, recommended: { version: 'b1', build: 1 }, fork: { repository: 'r', commit: '0123456789abcdef' }, candidates: [] },
        machine: { cpus: 4, cpu: 'x', memory: 8e9, free: 4e9, platform: 'linux-x64' }, jobs: { running: 0, total: 0 } }) },
      models: {
        config: ok({ path: '/c', defaultProfile: null, profiles: { fast: { profile: { provider: 'anthropic', model: 'haiku' }, kind: 'pi-provider', model: 'haiku' } }, credentials: [], stock: [stock], builtinDefault: { kind: 'managed-local', model: 'lfm3-1.7b' } }),
        providers: ok([{ id: 'anthropic', name: 'Anthropic', stored: null, connected: null, problem: null, apiKey: { name: 'ANTHROPIC_API_KEY', prompts: true },
          oauth: { name: 'Claude Pro (browser)', subscription: true, label: null }, models: 1, recommended: { haiku: 'haiku', model: 'haiku' } }]),
        providerModels: ok([{ id: 'haiku', name: 'Haiku', reasoning: null, contextWindow: 200000, recommended: true, family: 'haiku' }]),
        modelServer: ok({ running: false }), stock: ok([stock]),
        saveProfile: write('saveProfile'), deleteProfile: write('deleteProfile'), setDefault: write('setDefault'), downloadStock: write('downloadStock'),
      },
      programs: {
        list: ok([{ name: 'hello', directory: 'hello', description: 'Says hello', targets: ['main'], functions: 1, warm: null, builtAt: null }]),
        detail: ok({ directory: 'hello', manifest: { name: 'hello', version: '0.1.0', description: 'Says hello', targets: { main: { entry: 'main.ts', arguments: [{ name: 'input', type: 'string' }] } }, warmup: [] },
          problem: null, functions: [{ name: 'answer', source: 'answer.nl', description: '', params: ['input: string'], returns: 'string', fields: [argument], result: { ...argument, name: 'expected result' } }],
          lastRuns: {}, lastCalls: {}, examples: 0, examplesProblem: null, lock: null, tuning: null, readme: { name: 'README.md', text: '# Hello\n\nA program.' }, recent: [],
          forms: { main: { declared: true, arguments: [argument] } } }),
        model: ok({ model: 'lfm3-1.7b', kind: 'managed-local', detail: 'model downloaded; runtime installed.' }),
        checkArguments: ok({ problems: [] }), files: ok({ dir: '.', entries: [] }), create: ok({ directory: 'made' }),
      },
      traces: {
        list: ok([{ call_id: 'c1', definition_name: 'answer', definition_source: 'answer.nl', outcome: 'done', executor: 'model', started_at: now, wall_ms: 10, tokens_in: 5, tokens_out: 3, parent_call_id: null }]),
        hot: ok([{ definition_name: 'answer', definition_source: 'answer.nl', calls: 3, crisp_calls: 0, tokens: 20 }]),
        get: ok({ record: { call_id: 'c1', definition: { name: 'answer', source: 'answer.nl' }, outcome: 'done', executor: { kind: 'model', model_id: 'm' },
          cost: { wall_ms: 10, model_requests: 1, tokens_in: 5, tokens_out: 3, evals: 1 }, started_at: now, inputs: { input: 'x' }, output: 'y', effects: [] }, annotations: [], children: [], cases: [] }),
        events: ok([]), feedback: write('feedback'),
      },
      system: { warm: ok([]), clearWarm: write('clearWarm'), saveStoreSettings: write('saveStoreSettings') },
    };
    // A runtime that scripts what the natlang runtime reports to onLive for one function call, then runs the work.
    let release = () => {};
    const runtime = { async run(work, { onLive, signal }) {
      const call = { taskId: 't', callId: 't/1', parentCallId: null, definitionId: 'answer.nl', name: 'answer' };
      onLive({ type: 'call_start', ...call });
      onLive({ type: 'trace', ...call, event: { kind: 'model_request', phase: 'start', turn: 1 } });
      onLive({ type: 'model_delta', ...call, turn: 1, delta: { type: 'text', text: 'Working ' } });
      onLive({ type: 'model_delta', ...call, turn: 1, delta: { type: 'text', text: 'on it.' } });
      const value = await work();
      if (signal.aborted) throw new Error('aborted');
      onLive({ type: 'call_end', ...call, outcome: 'done', detail: '' });
      return value;
    } };
    const executors = { call: async (spec, io) => { io.stderr('log line\n'); await new Promise(resolve => { release = resolve; if (spec.inputs?.immediate) resolve(); }); return { answer: 'done' }; },
      ask: async () => 'asked', run: async () => 'ran' };
    const runs = inProcessRunHost(runtime, executors);
    let watchers = 0;
    const counted = { ...runs, watch: (id, observer) => { watchers++; const stop = runs.watch(id, observer); return () => { watchers--; stop(); }; } };
    const toasts = [], navigations = [];
    const host = (extra = {}) => ({ api, runs: counted, toast: (text, error) => toasts.push([String(text), error]), navigate: route => navigations.push(route), ...extra });
    const slot = document.getElementById('slot');
    window.embed = {
      availablePages, host, slot, api, runs: counted, toasts, navigations, writes, release: () => release(), watchers: () => watchers,
      mount(id, arg = '', extra = {}) {
        const page = availablePages(host(extra)).find(item => item.id === id);
        const element = createH(document)('div'); slot.append(element);
        return { element, dispose: page.mount(element, host({ arg, ...extra })), page };
      },
      hostWithout(...omitted) { const copy = { ...api }; for (const name of omitted) delete copy[name]; return { api: copy, runs: omitted.includes('runs') ? undefined : counted }; },
    };
  }, origin);

  const pageIds = await page.evaluate(() => window.embed.availablePages(window.embed.host()).map(item => item.id));
  assert.deepEqual(pageIds, ['overview', 'programs', 'runs', 'traces', 'playground', 'models', 'system']);

  // Every page draws from the fixtures, then disposes completely.
  const baseline = await page.evaluate(() => window.__leaks());
  const arg = { programs: 'hello', traces: 'call-c1' };
  for (const id of pageIds) {
    const view = await page.evaluateHandle(([id, arg]) => window.embed.mount(id, arg ?? ''), [id, arg[id]]);
    const element = await view.evaluateHandle(value => value.element);
    await page.waitForFunction(node => node.querySelector('.view') && !/^Loading…$/.test(node.querySelector('.view').textContent.trim()), element);
    assert.equal(await element.evaluate(node => node.querySelector('.callout.warn:not([hidden])')?.textContent ?? ''), '', `${id} drew an error: ${await element.evaluate(node => node.innerText.slice(0, 600))}`);
    assert.ok(await element.evaluate(node => node.querySelector('h1')), `${id} drew its heading`);
    // Control: the instrumentation sees a mounted page's timer (the Models page polls stock downloads).
    if (id === 'models') assert.equal((await page.evaluate(() => window.__leaks())).intervals, 1);
    await view.evaluate(value => value.dispose());
    assert.equal(await element.evaluate(node => node.childElementCount), 0, `${id} leaves nothing in its element`);
    await element.evaluate(node => node.remove());
    const leaks = await page.evaluate(() => window.__leaks());
    assert.deepEqual(leaks, baseline, `${id} leaked timers or listeners`);
  }
  assert.equal(await page.evaluate(() => window.embed.watchers()), 0);

  // The Runs page follows an in-process run: lambdas and output arrive from the run host's live events; disposing mid-run unwatches it.
  const run = await page.evaluate(async () => (await window.embed.runs.start({ kind: 'call', function: 'hello/answer.nl', inputs: {}, program: 'hello' })).id);
  const runs = await page.evaluateHandle(id => window.embed.mount('runs', id), run);
  const runsElement = await runs.evaluateHandle(value => value.element);
  await runsElement.evaluate(node => new Promise(resolve => { const check = () => node.querySelector('.lambda[data-call-id="t/1"]') ? resolve() : setTimeout(check, 20); check(); }));
  assert.equal(await page.evaluate(() => window.embed.watchers()), 1);
  assert.match(await runsElement.evaluate(node => node.querySelector('.execution-panel').innerText), /answer/);
  await page.evaluate(() => window.embed.release());
  await runsElement.evaluate(node => new Promise(resolve => { const check = () => node.querySelector('.pill[data-status=done]') ? resolve() : setTimeout(check, 20); check(); }));
  assert.match(await runsElement.evaluate(node => node.querySelector('.program-output').innerText), /Answer\s+done/);
  await page.getByRole('button', { name: 'Log', exact: true }).click();
  assert.match(await runsElement.evaluate(node => node.innerText), /log line/);
  await runs.evaluate(value => value.dispose());
  assert.equal(await page.evaluate(() => window.embed.watchers()), 0);
  const mid = await page.evaluate(async () => (await window.embed.runs.start({ kind: 'call', function: 'hello/answer.nl', inputs: {} })).id);
  const midView = await page.evaluateHandle(id => window.embed.mount('runs', id), mid);
  await page.waitForFunction(() => window.embed.watchers() === 1);
  await midView.evaluate(value => value.dispose());
  assert.equal(await page.evaluate(() => window.embed.watchers()), 0, 'disposing a page mid-run stops following the run');
  await page.evaluate(() => window.embed.release());
  await page.evaluate(() => window.embed.slot.replaceChildren());

  // An action goes through the injected host: starting a run announces it and opens it on the Runs page.
  const playground = await page.evaluateHandle(() => window.embed.mount('playground'));
  const playgroundElement = await playground.evaluateHandle(value => value.element);
  await playgroundElement.evaluate(node => node.querySelector('button.primary').click());
  await page.waitForFunction(() => window.embed.navigations.some(route => /^runs-run-/.test(route)));
  assert.ok(await page.evaluate(() => window.embed.toasts.some(([text]) => /^Started: Playground experiment/.test(text))));
  await playground.evaluate(value => value.dispose());

  // A dialog a page opened closes with the page.
  const programs = await page.evaluateHandle(() => window.embed.mount('programs', 'hello'));
  const programsElement = await programs.evaluateHandle(value => value.element);
  await page.waitForFunction(node => node.querySelector('.run-target'), programsElement);
  await programsElement.evaluate(node => [...node.querySelectorAll('button')].find(button => button.textContent === 'Create program').click());
  assert.equal(await page.evaluate(() => document.querySelectorAll('dialog[open]').length), 1);
  await programs.evaluate(value => value.dispose());
  assert.equal(await page.evaluate(() => document.querySelectorAll('dialog').length), 0);
  await page.evaluate(() => window.embed.slot.replaceChildren());

  // Capabilities: a host that omits a group is offered only the pages that do not need it, and a page refuses a host without it.
  assert.deepEqual(await page.evaluate(() => window.embed.availablePages(window.embed.hostWithout('models', 'system', 'traces', 'programs', 'runs')).map(item => item.id)), ['overview']);
  assert.deepEqual(await page.evaluate(() => window.embed.availablePages(window.embed.hostWithout('models', 'system')).map(item => item.id)), ['overview', 'programs', 'runs', 'traces', 'playground']);
  const refused = await page.evaluate(async () => {
    const { pages } = await import(`${location.origin}/pages.js`);
    try { pages.find(item => item.id === 'models').mount(window.embed.slot, { ...window.embed.hostWithout('models'), navigate() {}, toast() {} }); return 'mounted'; } catch (error) { return error.message; }
  });
  assert.match(refused, /models page needs models/);
  // Overview without models, traces and runs still draws (the panels that need them are left out).
  await page.evaluate(() => { const { api, runs } = window.embed.hostWithout('models', 'traces', 'runs'); window.embed.partial = { api, runs }; });
  const partial = await page.evaluateHandle(() => {
    const page = window.embed.availablePages(window.embed.partial).find(item => item.id === 'overview');
    const element = document.createElement('div'); window.embed.slot.append(element);
    return { element, dispose: page.mount(element, { ...window.embed.partial, navigate() {}, toast() {} }) };
  });
  const partialElement = await partial.evaluateHandle(value => value.element);
  await page.waitForFunction(node => node.querySelector('.tile'), partialElement);
  assert.equal(await partialElement.evaluate(node => node.textContent.includes('Recent runs') || node.textContent.includes('Busiest functions') || node.textContent.includes('Keep the model loaded')), false);
  await partial.evaluate(value => value.dispose());

  // Style scoping: the dashboard's rules do not reach the host's own elements, and a host's unlayered rule beats the dashboard's layered one.
  assert.deepEqual(await hostStyle(), outsideBefore);
  const overview = await page.evaluateHandle(() => window.embed.mount('overview'));
  const overviewElement = await overview.evaluateHandle(value => value.element);
  await page.waitForFunction(node => node.querySelector('.panel'), overviewElement);
  assert.equal(await overviewElement.evaluate(node => getComputedStyle(node.querySelector('.panel')).paddingTop), '3px', 'a host rule beats the natlang layer');
  assert.equal(await overviewElement.evaluate(node => getComputedStyle(node.querySelector('.panel')).borderTopLeftRadius), '7px', 'inside the root the dashboard styles apply');
  assert.equal(await overviewElement.evaluate(node => node.classList.contains('nl-dash')), true, 'mounting into an unstyled element makes it the dashboard root');
  await overview.evaluate(value => value.dispose());
  assert.equal(await overviewElement.evaluate(node => node.classList.contains('nl-dash')), false);

  assert.deepEqual(problems, []);
  process.stdout.write('Dashboard embed smoke passed: seven pages mount, draw and dispose cleanly; runs follow an in-process host; capabilities gate pages; styles are scoped.\n');
} finally {
  await browser?.close(); await new Promise(resolve => server.close(resolve));
}
