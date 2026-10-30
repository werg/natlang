// natlang dashboard: one page, one view per section (location.hash), all state from the local API (src/dashboard/server.ts).
import { createExecutionModel as liveModel, createExecutionView } from './execution.js';
import { argumentForm } from './arguments.js';
import { connection, providersPanel, signIn } from './providers.js';
const TOKEN = document.querySelector('meta[name="natlang-token"]').content;
// Each route renders into its own element: a view still loading when the route changes writes into a detached one.
const root = document.getElementById('main');

// ---- small helpers -------------------------------------------------------------------------------------------------
async function api(method, path, body, signal) {
  const response = await fetch(path, { method, headers: { 'x-natlang-token': TOKEN, ...(body === undefined ? {} : { 'content-type': 'application/json' }) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }), ...(signal ? { signal } : {}) });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
const get = (path, signal) => api('GET', path, undefined, signal);
function h(tag, attrs = {}, ...children) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith('on')) element.addEventListener(key.slice(2), value);
    else if (key === 'class') element.className = value;
    else if (key === 'html') element.innerHTML = value;
    else element.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat(Infinity)) if (child !== null && child !== undefined && child !== false)
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  return element;
}
/** Replace an element's children, skipping absent ones (null, undefined, false). */
const fill = (element, ...children) => element.replaceChildren(...children.flat(Infinity).filter(child => child !== null && child !== undefined && child !== false));
let toastTimer;
function toast(text, error = false) {
  const element = document.getElementById('toast');
  element.textContent = text; element.className = error ? 'error' : ''; element.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { element.hidden = true; }, error ? 7000 : 3500);
}
const attempt = fn => async (...args) => { try { return await fn(...args); } catch (error) { toast(error.message, true); } };
const bytes = n => n == null ? '–' : n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(1)} GB` : n >= 2 ** 20 ? `${(n / 2 ** 20).toFixed(0)} MB` : `${Math.round(n / 1024)} KB`;
const ms = n => n == null ? '–' : n >= 60000 ? `${Math.floor(n / 60000)}m ${Math.round(n / 1000) % 60}s` : n >= 1000 ? `${(n / 1000).toFixed(1)} s` : `${n} ms`;
const tokens = n => n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n ?? 0);
const ago = iso => { if (!iso) return '–'; const s = (Date.now() - Date.parse(iso)) / 1000;
  return s < 60 ? 'just now' : s < 3600 ? `${Math.floor(s / 60)} min ago` : s < 86400 ? `${Math.floor(s / 3600)} h ago` : new Date(iso).toLocaleString(); };
const OUTCOME = { done: 'ok', running: 'info', failed: 'bad', quiesced: 'warn', stopped: 'idle', lost: 'warn', interrupted: 'warn', blocked: 'warn' };
const pill = (text, kind = OUTCOME[text] ?? 'idle') => h('span', { class: `pill ${kind}` }, text);
const short = (value, limit = 160) => { const text = typeof value === 'string' ? value : JSON.stringify(value); return text === undefined ? '' : text.length > limit ? text.slice(0, limit) + '…' : text; };
const pretty = value => typeof value === 'string' ? value : JSON.stringify(value, null, 2);
function programDialog(options = {}) {
  const name = h('input', { placeholder: 'my-program', value: 'my-program', required: true, pattern: '[A-Za-z0-9][A-Za-z0-9._-]{0,63}' });
  const errorBox = h('div', { class: 'callout warn', hidden: true });
  const dialog = h('dialog', {}, h('form', { class: 'stack', onsubmit: async event => {
    event.preventDefault(); const button = event.target.querySelector('button[type=submit]'); button.disabled = true;
    try { const created = await api('POST', '/api/programs', { name: name.value, ...options }); dialog.close(); go(`programs-${encodeURIComponent(created.directory)}`); }
    catch (error) { errorBox.textContent = error.message; errorBox.hidden = false; }
    finally { button.disabled = false; }
  } }, h('h2', {}, options.instruction ? 'Save as a program' : 'Create a program'),
  h('p', {}, 'Creates a folder under the dashboard root, with instructions, sample input and a runnable entry point.'),
  h('label', { class: 'field' }, 'Directory name', name), errorBox,
  h('div', { class: 'row' }, h('button', { type: 'submit', class: 'primary' }, 'Create program'), h('button', { type: 'button', onclick: () => dialog.close() }, 'Cancel'))));
  dialog.addEventListener('close', () => dialog.remove()); document.body.append(dialog); dialog.showModal(); name.select();
}
function openProgramDialog() {
  const path = h('input', { placeholder: 'relative/path/to/program', required: true });
  const errorBox = h('div', { class: 'callout warn', hidden: true });
  const dialog = h('dialog', {}, h('form', { class: 'stack', onsubmit: async event => {
    event.preventDefault();
    try { const program = await get(`/api/programs/detail?path=${encodeURIComponent(path.value)}`); dialog.close(); go(`programs-${encodeURIComponent(program.directory)}`); }
    catch (error) { errorBox.textContent = error.message; errorBox.hidden = false; }
  } }, h('h2', {}, 'Open a program'), h('p', {}, 'Enter a directory under this dashboard root. To open another workspace, start natlang dashboard --root PATH --open.'),
  path, errorBox, h('div', { class: 'row' }, h('button', { type: 'submit' }, 'Open'), h('button', { type: 'button', onclick: () => dialog.close() }, 'Cancel'))));
  dialog.addEventListener('close', () => dialog.remove()); document.body.append(dialog); dialog.showModal(); path.focus();
}
const header = (title, text, ...actions) => h('div', { class: 'head' }, h('div', {}, h('h1', {}, title), text ? h('p', {}, text) : null), h('div', { class: 'row' }, actions));
const kv = pairs => h('dl', { class: 'kv' }, pairs.filter(Boolean).flatMap(([key, value]) => [h('dt', {}, key), h('dd', {}, value ?? '–')]));
let profilesCache = null;
async function profileSelect(id, value = '') {
  profilesCache ??= await get('/api/config');
  // The last entry leads to Models to connect a hosted provider; choosing it does not change the selection.
  const select = h('select', { id, onchange: () => { if (select.value === '+connect') { select.value = value; go('models'); } } },
    h('option', { value: '' }, `automatic (the program's model, else ${profilesCache.defaultProfile ?? 'built-in LFM3 1.7B'})`),
    Object.keys(profilesCache.profiles).map(name => h('option', { value: name, selected: name === value }, name)),
    h('option', { value: '+connect' }, 'Connect a hosted model (Claude, GPT, …)…'));
  return select;
}

// ---- routing --------------------------------------------------------------------------------------------------------
const views = { overview, programs, runs, traces, playground, models, system };
let cleanup = [];
async function route() {
  for (const fn of cleanup) fn(); cleanup = [];
  const [name, ...rest] = (location.hash.slice(1) || 'overview').split('-');
  const view = views[name] ?? overview;
  for (const link of document.querySelectorAll('#nav a')) link.classList.toggle('active', link.dataset.view === (views[name] ? name : 'overview'));
  const target = h('div', { class: 'view' }, h('div', { class: 'empty' }, 'Loading…'));
  fill(root, target);
  try { await view(rest.join('-'), target); } catch (error) { fill(target, h('div', { class: 'callout warn' }, error.message)); }
}
window.addEventListener('hashchange', route);
const go = hash => { if (location.hash === '#' + hash) route(); else location.hash = hash; };
const every = (fn, period) => { const timer = setInterval(fn, period); cleanup.push(() => clearInterval(timer)); };

async function refreshRail() {
  try {
    const status = await get('/api/status');
    const model = status.model;
    document.getElementById('rail-model').textContent = model.error ? `Model profile problem: ${model.error}` :
      `${model.profile} · ${model.release?.label ?? model.model}${model.provider ? ` (${model.provider})` : ''}`;
    const count = document.getElementById('running-count');
    count.hidden = !status.jobs.running; count.textContent = status.jobs.running;
  } catch (error) { document.getElementById('rail-model').textContent = error.message; }
}

// ---- overview -------------------------------------------------------------------------------------------------------
async function overview(_arg, target) {
  const [status, jobs, hot] = await Promise.all([get('/api/status'), get('/api/jobs'), get('/api/traces/hot').catch(() => [])]);
  const model = status.model, release = model.release;
  const runtime = status.runtime.selected;
  const problems = [];
  if (!runtime && model.kind === 'managed-local') problems.push(['The llama.cpp runtime is not installed.', 'Install runtime and model', () => startJob({ kind: 'setup' })]);
  const downloading = release && !release.installed && release.download && !release.download.done && !release.download.error;
  const percent = download => Math.floor(100 * download.received / download.total);
  const progress = downloading ? h('div', { class: 'callout row' }, h('span', {}, `Downloading ${release.label} (${bytes(release.bytes)})…`),
    h('div', { class: 'bar', style: 'flex: 1' }, h('span', { style: `width: ${percent(release.download)}%` })), h('span', { class: 'small muted' }, `${percent(release.download)}%`)) : null;
  if (downloading) every(attempt(async () => {
    const next = (await get('/api/status')).model.release;
    if (next?.installed || !next?.download || next.download.done || next.download.error) return route();
    progress.querySelector('.bar span').style.width = `${percent(next.download)}%`; progress.lastChild.textContent = `${percent(next.download)}%`;
  }), 1500);
  else if (release && !release.installed) problems.push([`${release.label} (${bytes(release.bytes)}) is not downloaded yet; the first run would download it.`, 'Download now',
    attempt(async () => { await api('POST', `/api/stock/${release.id}/download`); toast('Download started'); go('models'); })]);
  fill(target, 
    header('Overview', `Programs under ${status.root}. Everything here is the same machine state the natlang CLI uses.`,
      h('button', { onclick: () => programDialog() }, 'Create program'), h('button', { onclick: () => go('playground') }, 'Ask the model'), h('button', { class: 'primary', onclick: () => go('programs') }, 'Run a program')),
    ...problems.map(([text, label, action]) => h('div', { class: 'callout warn row' }, h('span', {}, text), h('button', { onclick: action }, label),
      model.kind === 'managed-local' ? h('button', { onclick: () => go('models') }, 'Use a hosted model instead') : null)), progress,
    h('div', { class: 'grid' },
      tile('Model', model.error ? 'Profile problem' : release?.label ?? model.model,
        model.error ?? `profile ${model.profile} · ${model.kind === 'managed-local' ? 'managed llama.cpp' : model.kind}${release ? (release.installed ? ' · downloaded' : downloading ? ' · downloading' : ' · not downloaded') : ''}`),
      tile('Runtime', runtime ? `llama.cpp ${runtime.version}` : 'not installed', runtime ? (runtime.source === 'fork' ? `natlang fork ${status.runtime.fork.commit.slice(0, 10)}` : `build ${runtime.build} · ${runtime.source}`) : `pinned ${status.runtime.recommended.version} (b${status.runtime.recommended.build})`),
      tile('Traces', status.calls ? `${status.calls.calls} calls` : 'recording off', status.calls ? `${bytes(status.calls.bytes)} · specialization ${status.calls.settings.specialization}` : ''),
      tile('Machine', `${status.machine.cpus} CPUs · ${bytes(status.machine.memory)}`, `${bytes(status.machine.free)} free · ${status.machine.platform}`)),
    await keptModelPanel(),
    h('div', { class: 'grid', style: 'grid-template-columns: repeat(auto-fit, minmax(360px, 1fr))' },
      h('section', { class: 'panel' }, h('div', { class: 'row', style: 'justify-content: space-between' }, h('h2', {}, 'Recent runs'), h('a', { href: '#runs' }, 'All runs')),
        jobs.length ? jobTable(jobs.slice(0, 8)) : h('div', { class: 'empty' }, 'No runs yet. Start one from Programs or the Playground.')),
      h('section', { class: 'panel' }, h('h2', {}, 'Busiest functions'),
        hot.length ? h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Function'), h('th', { class: 'num' }, 'Calls'), h('th', { class: 'num' }, 'Crisp'), h('th', { class: 'num' }, 'Tokens'))),
          h('tbody', {}, hot.slice(0, 8).map(row => h('tr', { class: 'click', onclick: () => go(`traces-${encodeURIComponent(row.definition_name)}`) },
            h('td', {}, row.definition_name, h('div', { class: 'small muted' }, row.definition_source ?? '')), h('td', { class: 'num' }, row.calls),
            h('td', { class: 'num' }, row.crisp_calls), h('td', { class: 'num' }, tokens(row.tokens))))))) :
          h('div', { class: 'empty' }, 'No recorded calls yet.'))));
}
/** The model the dashboard keeps loaded for every natlang command on this machine (one server, warm slots, no load time). */
async function keptModelPanel() {
  const [state, config] = await Promise.all([get('/api/model-server'), profilesCache ? Promise.resolve(profilesCache) : get('/api/config')]);
  profilesCache = config;
  const local = Object.entries(config.profiles).filter(([, entry]) => entry.kind === 'managed-local').map(([name]) => name);
  const choose = h('select', { id: 'kept-profile' }, h('option', { value: '' }, 'built-in LFM3 1.7B'),
    local.map(name => h('option', { value: name, selected: name === state.profile }, name)));
  const panel = h('section', { class: 'panel' });
  const busy = label => attempt(async action => { fill(panel, h('div', { class: 'muted' }, label)); await action(); });
  const start = () => busy('Loading the model…')(async () => { await api('POST', '/api/model-server', { profile: choose.value || undefined }); toast('Model kept loaded'); route(); });
  const stop = () => busy('Stopping…')(async () => { await api('DELETE', '/api/model-server'); toast('Model unloaded'); route(); });
  const server = state.server;
  fill(panel, h('div', { class: 'row', style: 'justify-content: space-between' }, h('h2', {}, 'Keep the model loaded'),
      state.running ? h('div', { class: 'row' }, pill('loaded', 'ok'), h('button', { onclick: stop }, 'Unload')) :
        h('div', { class: 'row' }, choose, h('button', { class: 'primary', onclick: start }, 'Load'))),
    h('div', { class: 'small muted' }, 'One llama.cpp server for every run, call and terminal command on this machine whose model matches: no load time per run, warm slots, and one copy of the model in memory.'),
    server ? kv([['Model', `${server.model} (profile ${server.profile})`], ['Server', `${server.endpoint} · ${server.slots} slots of ${tokens(server.contextPerSlot)} tokens`],
      ['Since', ago(server.startedAt)]]) : state.other ? h('div', { class: 'small' }, `Another natlang process (pid ${state.other.pid}) keeps ${state.other.model} loaded at ${state.other.endpoint}.`) : null,
    state.log?.length ? h('pre', { class: 'block small' }, state.log.slice(-4).join('\n')) : null);
  return panel;
}
const tile = (label, value, sub) => h('div', { class: 'panel tile' }, h('div', { class: 'label' }, label), h('div', { class: 'value' }, value), h('div', { class: 'sub' }, sub));
function jobTable(jobs) {
  return h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Run'), h('th', {}, 'State'), h('th', {}, 'Started'), h('th', { class: 'num' }, 'Time'))),
    h('tbody', {}, jobs.map(job => h('tr', { class: 'click', onclick: () => go(`runs-${job.id}`) },
      h('td', { class: 'mono' }, short(job.title, 70)), h('td', {}, pill(job.status)), h('td', {}, ago(job.startedAt)),
      h('td', { class: 'num' }, ms((job.endedAt ? Date.parse(job.endedAt) : Date.now()) - Date.parse(job.startedAt))))))));
}

// ---- jobs: start and supervise ---------------------------------------------------------------------------------------
async function startJob(spec) {
  try { const job = await api('POST', '/api/jobs', spec); toast(`Started: ${job.title}`); go(`runs-${job.id}`); return job; }
  catch (error) { toast(error.message, true); }
}

// Generation speed as the server measured it, and the prefill time when the prompt took noticeably long.
const rate = turn => turn.generationMs > 0 ? ` · ${(turn.completion * 1000 / turn.generationMs).toFixed(1)} tokens/s${turn.prefillMs > 1000 ? ` after ${ms(turn.prefillMs)} prefill` : ''}` : '';
function renderCall(call, all = false) {
  const turns = [...call.turns.values()].sort((a, b) => a.number - b.number);
  const totals = turns.reduce((sum, turn) => [sum[0] + turn.prompt, sum[1] + turn.completion], [0, 0]);
  const kind = call.status === 'done' ? 'done' : call.status === 'running' ? '' : 'failed';
  return h('div', { class: `call ${kind}` },
    h('div', { class: 'row', style: 'justify-content: space-between' }, h('strong', {}, call.name), pill(call.status)),
    h('div', { class: 'meta' }, `${turns.length} turn${turns.length === 1 ? '' : 's'} · ${tokens(totals[0])} prompt + ${tokens(totals[1])} generated tokens`,
      call.ended && call.started ? ` · ${ms(call.ended - call.started)}` : ''),
    call.detail && call.status !== 'done' ? h('div', { class: 'turn' }, h('div', { class: 'fail' }, call.detail)) : null,
    (all ? turns : turns.slice(-4)).map(turn => h('div', { class: 'turn' },
      h('div', { class: 'meta' }, `turn ${turn.number}${turn.working ? ` · ${turn.reasoning ? 'thinking' : turn.text || turn.tools.length ? 'writing' : 'reading the prompt'}… ` : turn.ms ? ` · ${ms(turn.ms)} · ${tokens(turn.prompt)} prompt (${tokens(turn.cached ?? 0)} cached) + ${tokens(turn.completion)} generated${rate(turn)}` : ''}`,
        turn.working && turn.since && !all ? h('span', { 'data-since': turn.since }, ms(Date.now() - turn.since)) : null),
      turn.reasoning ? h('div', { class: 'reason' }, turn.reasoning.slice(-1500)) : null,
      turn.text ? h('div', {}, turn.text.slice(-1500)) : null,
      turn.tools.map((tool, index) => [h('div', { class: 'code' }, toolText(tool)),
        turn.results[index] ? h('div', { class: 'result' }, short(turn.results[index], all ? 2400 : 600)) : null]),
      turn.failures.filter(text => !turn.results.some(result => result.includes(String(text).slice(0, 80))))
        .map(text => h('div', { class: 'fail' }, short(text, 600))))),
    call.children.length ? h('div', { class: 'children' }, call.children.map(child => renderCall(child, all))) : null);
}
function toolText(tool) {
  try { const args = JSON.parse(tool.arguments || '{}');
    return tool.name === 'eval' && typeof args.code === 'string' ? args.code : `${tool.name}(${JSON.stringify(args)})`; }
  catch { return `${tool.name}(${tool.arguments})`; }
}

async function runs(id, target) {
  const jobs = await get('/api/jobs');
  const selected = id || jobs[0]?.id;
  const list = h('div', { class: 'list scroll' });
  const drawList = rows => fill(list, rows.length ? rows.map(job => h('button', { class: `item${job.id === selected ? ' selected' : ''}`, onclick: () => go(`runs-${job.id}`) },
    h('div', { class: 'title' }, h('span', { class: 'mono' }, short(job.title, 46)), pill(job.status)),
    h('div', { class: 'desc' }, `${job.kind} · ${ago(job.startedAt)}`))) : h('div', { class: 'empty' }, 'No runs yet. Start one from Programs or the Playground.'));
  drawList(jobs);
  every(attempt(async () => drawList(await get('/api/jobs'))), 4000);
  const detail = h('div', { class: 'stack' });
  fill(target, header('Runs', 'Every program run, call, ask, build and tuning job started from here, with its live calls and output.'),
    h('div', { class: 'split' }, h('section', { class: 'panel', style: 'padding: 6px 10px' }, list), detail));
  if (selected) await jobDetail(selected, detail);
}

async function jobDetail(id, container) {
  const controller = new AbortController(); cleanup.push(() => controller.abort());
  const state = { stdout: 0, stderr: 0, live: 0, out: '', err: '', model: liveModel(), tab: 'output', job: null, drained: false };
  const body = h('div', { class: 'stack' }), headLine = h('div', { class: 'row', style: 'justify-content: space-between' });
  const tabs = h('div', { class: 'tabs' }), summary = h('div', {}), execution = createExecutionView(h);
  const failureTitle = h('strong'), failureMessage = h('pre', { class: 'block run-failure-message' });
  const failureBox = h('div', { class: 'callout warn', hidden: true }, failureTitle, failureMessage,
    h('p', {}, 'See Output, expand a lambda that needs attention, or open Log for details.'));
  summary.append(failureBox);
  const output = h('pre', { class: 'console program-output', tabindex: '0', 'aria-label': 'Program output' });
  const log = h('pre', { class: 'console', tabindex: '0', 'aria-label': 'Program log' });
  const field = h('input', { id: 'job-input', 'aria-label': 'Program input', placeholder: 'Type input for the program and press Enter', style: 'flex: 1' });
  const input = h('form', { class: 'row', onsubmit: attempt(async event => {
    event.preventDefault(); await api('POST', `/api/jobs/${id}/input`, { text: field.value }); field.value = '';
  }) }, field, h('button', { type: 'submit' }, 'Send'));
  let followOutput = true, outputPending = false, headerKey = '', mountedTab = '', busy = false, failures = 0, nextPoll = 0, lastRail = 0;
  const follow = h('button', { type: 'button', hidden: true, onclick: () => { followOutput = true; output.scrollTop = output.scrollHeight; follow.hidden = true; } }, 'Follow output');
  output.addEventListener('scroll', () => {
    if (outputPending) return;
    followOutput = output.scrollHeight - output.scrollTop - output.clientHeight < 28;
    follow.hidden = followOutput;
  });
  const outputPane = h('section', { class: 'panel output-panel' }, h('div', { class: 'row', style: 'justify-content: space-between' }, h('h2', {}, 'Program output'), follow), output, input);
  const workspace = h('div', { class: 'run-workspace' }, execution.element, outputPane);
  fill(container, h('section', { class: 'panel' }, headLine, summary, tabs), body);
  const buttons = new Map();
  for (const [key, label] of [['output', 'Output'], ['live', 'Lambdas'], ['log', 'Log']]) {
    const button = h('button', { type: 'button', onclick: () => { state.tab = key; draw(); } }, label); buttons.set(key, button); tabs.append(button);
  }
  const updateText = (node, value, follow = false) => {
    if (node.textContent === value) return;
    const previous = node.scrollTop, old = node.textContent;
    if (old && value.startsWith(old) && node.childNodes.length === 1 && node.firstChild.nodeType === 3) node.firstChild.appendData(value.slice(old.length));
    else node.textContent = value;
    node.scrollTop = follow ? node.scrollHeight : previous;
  };
  const draw = () => {
    const job = state.job; if (!job) return;
    const key = `${job.id}:${job.status}:${job.exitCode}`;
    if (key !== headerKey) {
      headerKey = key;
      const commandOpen = headLine.querySelector('.run-command')?.open ?? false;
      const command = `natlang ${job.argv.map(arg => /\s/.test(arg) ? JSON.stringify(arg) : arg).join(' ')}`;
      fill(headLine, h('div', { class: 'stack', style: 'gap: 2px' }, h('h2', {}, job.title),
        h('details', { class: 'run-command', open: commandOpen }, h('summary', { class: 'small muted' }, 'Command'), h('code', {}, command))),
        h('div', { class: 'row' }, pill(job.status), job.exitCode !== null ? h('span', { class: 'small muted' }, `exit ${job.exitCode}`) : null,
          job.meta?.program || job.kind === 'run' ? h('button', { onclick: () => go(`programs-${encodeURIComponent(job.meta?.program ?? job.meta?.path ?? '.')}`) }, 'Program and source') : null,
          job.status === 'running' ? h('button', { class: 'danger', onclick: attempt(async () => { state.job = await api('POST', `/api/jobs/${id}/stop`); draw(); }) }, 'Stop') :
            h('button', { onclick: attempt(async () => { await api('DELETE', `/api/jobs/${id}`); go('runs'); }) }, 'Remove'),
          ['run', 'call', 'ask'].includes(job.kind) ? h('button', { onclick: () => rerun(job) }, 'Run again') : null));
    }
    failureBox.hidden = !['failed', 'lost'].includes(job.status);
    if (!failureBox.hidden) {
      updateText(failureTitle, job.status === 'lost' ? 'The run was interrupted.' : 'The run failed.');
      const message = job.kind === 'test' ? state.out.trim() || state.err.trim() : state.err.trim() || state.out.trim();
      updateText(failureMessage, message.split('\n').slice(-8).join('\n') || 'See Log for details.');
    }
    if (mountedTab !== state.tab) {
      mountedTab = state.tab;
      if (state.tab === 'output') { workspace.append(execution.element, outputPane); fill(body, workspace); }
      else if (state.tab === 'live') fill(body, execution.element);
      else fill(body, log);
    }
    for (const [key, button] of buttons) { button.classList.toggle('active', key === state.tab); button.setAttribute('aria-pressed', String(key === state.tab)); }
    buttons.get('live').textContent = `Lambdas${state.model.calls.size ? ` (${state.model.calls.size})` : ''}`;
    input.hidden = !job.interactive || job.status !== 'running';
    outputPending = true;
    updateText(output, state.out || (job.status === 'running' ? 'Waiting for program output. Live execution shows what is happening alongside it.' : '(no program output)'), followOutput);
    requestAnimationFrame(() => { outputPending = false; });
    updateText(log, state.err || '(nothing logged)', log.scrollHeight - log.scrollTop - log.clientHeight < 28);
    execution.update(state.model, { ...job, drained: state.drained }, state.err);
  };
  const poll = async () => {
    const value = await get(`/api/jobs/${id}?stdout=${state.stdout}&stderr=${state.stderr}&live=${state.live}`, AbortSignal.any([controller.signal, AbortSignal.timeout(8000)]));
    state.job = value.job; state.stdout = value.stdout.offset; state.stderr = value.stderr.offset; state.live = value.live.offset;
    state.out += value.stdout.text; state.err += value.stderr.text;
    for (const line of value.live.text.split('\n')) if (line) { try { state.model.add(JSON.parse(line)); } catch { /* only complete event lines are supplied */ } }
    state.drained = value.job.status !== 'running' && Boolean(value.job.endedAt) && !value.live.text && !value.stdout.text && !value.stderr.text;
    failures = 0; execution.connection(true); draw();
    if (Date.now() - lastRail > 4000) { lastRail = Date.now(); refreshRail(); }
  };
  await poll();
  every(async () => {
    if (busy || state.drained || Date.now() < nextPoll) return;
    busy = true;
    try { await poll(); }
    catch { failures++; execution.connection(false); nextPoll = Date.now() + Math.min(4000, 350 * 2 ** Math.min(failures, 4)); }
    finally { busy = false; }
  }, 350);
}
function rerun(job) {
  const meta = job.meta ?? {};
  if (job.kind === 'call') return startJob({ kind: 'call', function: meta.path, inputs: meta.inputs ?? {}, profile: meta.profile, program: meta.program });
  if (job.kind === 'ask') return startJob({ kind: 'ask', instruction: meta.instruction, input: meta.input, returns: meta.returns, profile: meta.profile });
  return startJob({ kind: 'run', path: meta.path ?? '.', target: meta.target, args: meta.args ?? [], profile: meta.profile });
}

// ---- programs -------------------------------------------------------------------------------------------------------
async function programs(path, target) {
  const list = await get('/api/programs');
  const selected = path ? decodeURIComponent(path) : list[0]?.directory;
  const detail = h('div', { class: 'stack' });
  const search = h('input', { id: 'program-search', type: 'search', placeholder: `Filter ${list.length} programs`, style: 'width: 100%' });
  const listBox = h('div', { class: 'list scroll' });
  fill(target, header('Programs', 'Choose a program and run it with sample input, or create your own.',
    h('button', { onclick: openProgramDialog }, 'Open program'), h('button', { class: 'primary', onclick: () => programDialog() }, 'Create program')),
    h('div', { class: 'split' }, h('section', { class: 'panel', style: 'padding: 10px' }, search, listBox), detail));
  const draw = () => {
    const words = search.value.toLowerCase().split(/\s+/).filter(Boolean);
    const shown = list.filter(item => words.every(word => `${item.name} ${item.directory} ${item.description}`.toLowerCase().includes(word)));
    fill(listBox, shown.length ? shown.map(item => h('button', { class: `item${item.directory === selected ? ' selected' : ''}`, onclick: () => go(`programs-${encodeURIComponent(item.directory)}`) },
      h('div', { class: 'title' }, h('span', {}, item.name), item.warm ? pill('warm', 'ok') : item.targets.length ? null : pill('functions', 'idle')),
      h('div', { class: 'small muted mono' }, item.directory), h('div', { class: 'desc' }, item.description))) :
      h('div', { class: 'empty' }, list.length ? 'No program matches.' : 'No programs yet. Create a program above, or open a directory containing natlang.json.'));
  };
  search.addEventListener('input', draw); draw();
  if (selected) await programDetail(selected, detail);
}

// A small Markdown reader for program READMEs, built from elements (never HTML from the file): headings, paragraphs
// with soft line breaks, lists, fenced code, tables as written, inline code, emphasis and web links.
function markdown(text) {
  const inline = line => {
    const out = [], pattern = /`([^`]+)`|\*\*([^*]+)\*\*|\[([^\]]+)\]\(([^)\s]+)\)/g;
    let last = 0, match;
    while ((match = pattern.exec(line))) {
      out.push(line.slice(last, match.index)); last = pattern.lastIndex;
      if (match[1] !== undefined) out.push(h('code', {}, match[1]));
      else if (match[2] !== undefined) out.push(h('strong', {}, match[2]));
      else out.push(/^https?:\/\//.test(match[4]) ? h('a', { href: match[4], target: '_blank', rel: 'noreferrer' }, match[3]) : h('span', { title: match[4] }, match[3]));
    }
    out.push(line.slice(last)); return out;
  };
  const blocks = [], lines = text.replace(/\r/g, '').split('\n');
  for (let index = 0; index < lines.length;) {
    const line = lines[index];
    if (/^\s*(```|~~~)/.test(line)) {
      const fence = line.trim().slice(0, 3), body = [];
      for (index++; index < lines.length && !lines[index].trim().startsWith(fence); index++) body.push(lines[index]);
      index++; blocks.push(h('pre', { class: 'block' }, body.join('\n'))); continue;
    }
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) { blocks.push(h(heading[1].length <= 2 ? 'h3' : 'h4', {}, inline(heading[2]))); index++; continue; }
    if (!line.trim()) { index++; continue; }
    const group = [];
    const kind = /^\s*([-*]|\d+\.)\s+/.test(line) ? 'list' : /^\s*\|/.test(line) ? 'table' : 'text';
    while (index < lines.length && lines[index].trim() && !/^(#{1,6})\s|^\s*(```|~~~)/.test(lines[index])) {
      const current = lines[index];
      if (kind === 'list' && /^\s*([-*]|\d+\.)\s+/.test(current)) group.push(current.replace(/^\s*([-*]|\d+\.)\s+/, ''));
      else if (kind === 'list' && group.length) group[group.length - 1] += ' ' + current.trim();
      else group.push(current);
      index++;
    }
    if (kind === 'list') blocks.push(h('ul', {}, group.map(item => h('li', {}, inline(item)))));
    else if (kind === 'table') blocks.push(h('pre', { class: 'block' }, group.join('\n')));
    else blocks.push(h('p', {}, inline(group.map(item => item.trim()).join(' '))));
  }
  return blocks;
}
// What a profile changes beyond its model: sampling, reasoning budget and server slots.
function profileSettings(profile) {
  const request = profile.request ?? {}, local = profile.local ?? {}, parts = [];
  if (request.temperature !== undefined) parts.push(`temperature ${request.temperature}`);
  if (request.thinking_budget_tokens !== undefined) parts.push(request.thinking_budget_tokens === 0 ? 'no thinking' : `thinking ≤ ${tokens(request.thinking_budget_tokens)} tokens`);
  if (local.contextTokens) parts.push(`${tokens(local.contextTokens)} context`);
  if (local.parallel) parts.push(`${local.parallel} slots`);
  if (local.server) parts.push('own llama-server');
  if (profile.escalate) parts.push(`escalates to ${profile.escalate}`);
  return parts.join(' · ') || '–';
}
async function programDetail(path, container) {
  const program = await get(`/api/programs/detail?path=${encodeURIComponent(path)}`);
  const manifest = program.manifest, lock = program.lock;
  const targets = Object.entries(manifest.targets ?? {});
  const runPanel = h('section', { class: 'panel' }, h('h2', {}, 'Run'),
    targets.length ? targets.map(([name, target]) => {
      const previous = program.lastRuns?.[name];
      const run = attempt(async () => startJob({ kind: 'run', path: program.directory, target: name,
        args: await form.args(), profile: document.getElementById(`profile-${name}`)?.value || undefined }));
      // The typed form of the target's declared arguments (dashboard/arguments.js); free-form words otherwise.
      const form = argumentForm({ h, fill, get, api, toast }, { program: program.directory, target: name, form: program.forms?.[name],
        previous: previous?.args, onSubmit: run });
      const profile = h('span', {});
      const readiness = h('div', { class: 'small muted', role: 'status' }, 'Resolving the model…');
      profileSelect(`profile-${name}`, previous?.profile).then(select => {
        fill(profile, select);
        let revision = 0;
        const refresh = async () => {
          const current = ++revision;
          try { const model = await get(`/api/programs/model?path=${encodeURIComponent(program.directory)}&profile=${encodeURIComponent(select.value)}`);
            if (revision === current) readiness.textContent = `${model.model} · ${model.kind} · ${model.detail}`;
          } catch (error) { if (revision === current) readiness.textContent = error.message; }
        };
        select.addEventListener('change', refresh); refresh();
      });
      return h('div', { class: 'stack run-target' }, h('div', {}, h('strong', { class: 'mono' }, name), target.description ? h('span', { class: 'muted' }, ` · ${target.description}`) : null),
        form.element,
        previous ? h('div', { class: 'small muted' }, 'Inputs restored from the last successful run.') : null,
        h('div', { class: 'row' }, profile, h('button', { class: 'primary', title: 'Run (Ctrl+Enter)', onclick: run }, 'Run'),
          h('button', { onclick: () => form.reset() }, 'Reset inputs')), readiness);
    }) : h('div', { class: 'muted' }, 'This package has no targets; call its functions below.'));
  const functions = h('section', { class: 'panel' }, h('h2', {}, 'Natural-language functions'),
    program.functions.length ? h('div', { class: 'stack' }, program.functions.map(fn => functionCard(program, fn))) : h('div', { class: 'muted' }, 'No .nl files.'));
  const warm = lock?.warm;
  const build = h('section', { class: 'panel' }, h('div', { class: 'row', style: 'justify-content: space-between' }, h('h2', {}, 'Compiled program'),
    h('div', { class: 'row' }, h('button', { onclick: () => startJob({ kind: 'build', path: program.directory }) }, 'Build'),
      h('button', { class: 'primary', onclick: () => startJob({ kind: 'build', path: program.directory, warm: true }),
        title: 'Runs the natlang.json warmup calls with tracing and prefills their prompt prefixes' }, 'Build warm'))),
    lock ? kv([['Built', ago(lock.builtAt)], ['Model', lock.model.release ? `${lock.model.release.id} · ${lock.model.release.source}@${lock.model.release.revision.slice(0, 8)}` : lock.model.model],
      ['Backend', `${lock.model.kind}${lock.model.runtime ? ` · llama.cpp b${lock.model.runtime.build}` : ''} · profile ${lock.model.profile}`],
      ['Tuning', lock.tuning ? `release ${lock.tuning.release.slice(0, 12)} on checkpoint ${lock.tuning.base.slice(0, 12)}` : 'none'],
      ['Warm', warm ? `${warm.cases.filter(item => item.outcome === 'done').length}/${warm.cases.length} warm-up calls passed · ${warm.prefixes.length} prompt prefix${warm.prefixes.length === 1 ? '' : 'es'} prefilled (${warm.prefixes.map(item => `${tokens(item.tokens)} tokens`).join(', ') || 'none'}) · ${ago(warm.at)}` : 'not warmed']]) :
      h('div', { class: 'muted' }, 'Not built yet. Build writes the lock: the model, runtime, tuning and functions this program runs with.'),
    warm?.cases.length ? h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Warm-up call'), h('th', {}, 'Outcome'), h('th', { class: 'num' }, 'Time'), h('th', { class: 'num' }, 'Tokens'))),
      h('tbody', {}, warm.cases.map(item => h('tr', { class: item.call_id ? 'click' : '', onclick: () => item.call_id && go(`traces-call-${encodeURIComponent(item.call_id)}`) },
        h('td', {}, item.name, item.detail ? h('div', { class: 'small muted' }, short(item.detail, 200)) : null), h('td', {}, pill(item.outcome)),
        h('td', { class: 'num' }, ms(item.wall_ms)), h('td', { class: 'num' }, `${tokens(item.tokens_in)} + ${tokens(item.tokens_out)}`)))))) : null,
    !(manifest.warmup ?? []).length && program.functions.length ? h('div', { class: 'small muted' }, `Add "warmup": [{ "function": "${program.functions[0].source}", "inputs": { … } }] to natlang.json to trace and prefill this program ahead of time.`) : null);
  fill(container, 
    h('section', { class: 'panel' }, h('div', { class: 'row', style: 'justify-content: space-between' }, h('h2', {}, manifest.name, h('span', { class: 'muted' }, ` ${manifest.version}`)),
      h('a', { href: `#traces-program-${encodeURIComponent(program.directory)}` }, 'Traces of this program')),
      manifest.description ? h('div', {}, manifest.description) : null,
      program.problem ? h('div', { class: 'callout warn' }, `natlang.json: ${program.problem}`) : null,
      manifest.model ? h('div', { class: 'small' }, 'Declared model: ', h('code', {}, JSON.stringify(manifest.model))) : h('div', { class: 'small muted' }, 'Runs on the machine default model unless natlang.json declares "model".')),
    runPanel, functions,
    h('section', { class: 'panel' }, h('h2', {}, 'Saved tests'), h('p', {}, program.examplesProblem ?? `${program.examples} saved input/expected-output example${program.examples === 1 ? '' : 's'}. Tests execute real model calls and compare exact results.`),
      h('button', { onclick: () => startJob({ kind: 'test', path: program.directory }) }, 'Run tests')),
    h('details', { class: 'panel' }, h('summary', {}, 'Build and warm-up details'), build),
    program.readme ? h('section', { class: 'panel' }, h('h3', { class: 'muted small' }, program.readme.name), h('div', { class: 'prose' }, markdown(program.readme.text))) : null);
}
function functionCard(program, fn) {
  const formId = encodeURIComponent(fn.source);
  const fields = fn.params.map(param => { const [name, ...type] = param.split(':'); return { name: name.replace('?', '').trim(), type: type.join(':').trim() }; });
  const warmCase = (program.manifest.warmup ?? []).find(item => item.function === fn.source);
  const last = program.lastCalls?.[`${program.directory === '.' ? '' : program.directory + '/'}${fn.source}`];
  const sample = last?.inputs ?? warmCase?.inputs ?? {};
  const inputs = fields.map(field => h('label', { class: 'field' }, `${field.name}: ${field.type}`,
    h('textarea', { id: `in-${formId}-${field.name}`, rows: 2, placeholder: field.type === 'string' ? 'text' : 'JSON' },
      sample[field.name] !== undefined ? (field.type === 'string' && typeof sample[field.name] === 'string' ? sample[field.name] : JSON.stringify(sample[field.name])) : '')));
  const sourceBox = h('div', { class: 'stack', hidden: true });
  const profile = h('span', {});
  profileSelect(`call-profile-${formId}`, last?.profile).then(select => fill(profile, select)).catch(error => fill(profile, error.message));
  const valuesFromFields = () => {
    const values = {};
    for (const field of fields) {
      const raw = document.getElementById(`in-${formId}-${field.name}`).value;
      if (raw === '' && field.type !== 'string') continue;
      if (field.type === 'string') values[field.name] = raw;
      else try { values[field.name] = JSON.parse(raw); } catch { throw new Error(`${field.name} must be JSON (${field.type})`); }
    }
    return values;
  };
  const call = attempt(async () => startJob({ kind: 'call', function: `${program.directory}/${fn.source}`, inputs: valuesFromFields(), program: program.directory,
    profile: document.getElementById(`call-profile-${formId}`)?.value || undefined }));
  const expected = h('textarea', { rows: 2, placeholder: 'Expected JSON result, e.g. "billing" or {"count": 2}' });
  return h('details', { class: 'panel', style: 'padding: 10px 12px; gap: 8px' },
    h('summary', { style: 'cursor: pointer' }, h('strong', { class: 'mono' }, fn.name), h('span', { class: 'muted mono small' }, ` (${fn.params.join(', ')}) → ${fn.returns}`),
      fn.description ? h('div', { class: 'small muted' }, fn.description) : null),
    h('div', { class: 'stack', style: 'margin-top: 8px' }, h('code', { class: 'small muted' }, fn.source), inputs, h('div', { class: 'row' }, profile, h('button', { class: 'primary', onclick: call }, 'Call'),
      h('a', { href: `#traces-${encodeURIComponent(fn.name)}` }, 'Recorded calls'),
      h('button', { onclick: attempt(async () => { const source = await get(`/api/programs/source?path=${encodeURIComponent(`${program.directory}/${fn.source}`)}`);
        const editor = h('textarea', { rows: 14, class: 'source-editor', 'aria-label': `Source of ${fn.name}` }, source.text);
        let previous = source.text;
        fill(sourceBox, h('code', {}, source.path), editor, h('button', { onclick: attempt(async () => {
          await api('PUT', '/api/programs/source', { path: source.path, text: editor.value, previous }); previous = editor.value; toast('Saved. Call the function or run the program to try the change.');
        }) }, 'Save source'), h('p', { class: 'small muted' }, 'Arguments and returns define the contract. Instructions can use these arguments and helpers in the companion folder. Static checking checks the contract; saved examples check behavior.'));
        sourceBox.hidden = false; }) }, 'Edit source')), sourceBox,
      h('details', {}, h('summary', {}, 'Save an expected result as a test'), h('label', { class: 'field' }, 'Expected result (JSON)', expected),
        h('button', { onclick: attempt(async () => {
          let result; try { result = JSON.parse(expected.value); } catch { throw new Error('Enter the expected result as JSON. Put quotes around a string.'); }
          await api('POST', '/api/programs/example', { path: program.directory, example: { function: fn.source, inputs: valuesFromFields(), expected: result } }); toast('Test saved to examples.json. Use Run tests to verify it.');
        }) }, 'Save test'))));
}

// ---- traces ---------------------------------------------------------------------------------------------------------
async function traces(arg, target) {
  let definition = '', program = '', callId = '';
  if (arg.startsWith('call-')) callId = decodeURIComponent(arg.slice(5));
  else if (arg.startsWith('program-')) program = decodeURIComponent(arg.slice(8));
  else if (arg) definition = decodeURIComponent(arg);
  const filters = { definition: h('input', { id: 'f-definition', placeholder: 'function', value: definition }),
    outcome: h('select', { id: 'f-outcome' }, ['', 'done', 'failed', 'quiesced', 'running', 'interrupted'].map(value => h('option', { value }, value || 'any outcome'))),
    program: h('input', { id: 'f-program', placeholder: 'program directory', value: program }) };
  const table = h('div', { class: 'table-wrap' }), detail = h('div', { class: 'stack' });
  const load = attempt(async () => {
    const query = new URLSearchParams(Object.entries({ definition: filters.definition.value, outcome: filters.outcome.value, program: filters.program.value, limit: '150' }).filter(([, v]) => v));
    const rows = await get(`/api/traces?${query}`);
    fill(table, rows.length ? h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Function'), h('th', {}, 'Outcome'), h('th', {}, 'Executor'), h('th', {}, 'Started'), h('th', { class: 'num' }, 'Time'), h('th', { class: 'num' }, 'Tokens'))),
      h('tbody', {}, rows.map(row => h('tr', { class: `click${row.call_id === callId ? ' selected' : ''}`, onclick: () => { callId = row.call_id; traceDetail(row.call_id, detail); } },
        h('td', {}, row.definition_name, row.parent_call_id ? h('span', { class: 'small muted' }, ' · child') : null, h('div', { class: 'small muted' }, row.definition_source ?? '')),
        h('td', {}, pill(row.outcome)), h('td', {}, row.executor), h('td', {}, ago(row.started_at)), h('td', { class: 'num' }, ms(row.wall_ms)),
        h('td', { class: 'num' }, `${tokens(row.tokens_in)} + ${tokens(row.tokens_out)}`))))) : h('div', { class: 'empty' }, 'No recorded calls match.'));
  });
  for (const field of Object.values(filters)) field.addEventListener('change', load);
  fill(target, header('Traces', 'Every natural-language call this machine ran: exact inputs, output, model turns, evals, service calls and cost. Mark calls good or fix them to grow the program\'s tuning examples.'),
    h('div', { class: 'row' }, filters.definition, filters.outcome, filters.program, h('button', { onclick: load }, 'Filter')),
    h('div', { class: 'split', style: 'grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr)' }, h('section', { class: 'panel' }, table), detail));
  await load();
  if (callId) await traceDetail(callId, detail);
  else fill(detail, h('div', { class: 'panel empty' }, 'Select a call to see its execution.'));
}

async function traceDetail(id, container) {
  fill(container, h('div', { class: 'panel empty' }, 'Loading…'));
  const value = await get(`/api/traces/${encodeURIComponent(id)}`);
  if (!value.record) {
    fill(container, h('section', { class: 'panel' }, h('h2', {}, value.running.definition_name), pill(value.running.outcome),
      h('div', { class: 'muted' }, `Started ${ago(value.running.started_at)}. The record is complete when the call ends.`)));
    return;
  }
  const record = value.record, cost = record.cost ?? {};
  const expected = h('textarea', { id: 'fix-expected', rows: 3, placeholder: 'The right result, as JSON (a string in quotes)' });
  const feedback = verdict => attempt(async () => {
    const result = await api('POST', `/api/traces/${encodeURIComponent(id)}/feedback`, { verdict, ...(expected.value.trim() ? { expected: expected.value.trim() } : {}) });
    toast(result.message);
  });
  const events = h('div', {});
  // The recorded turns, drawn as the live view draws a running call.
  const execution = h('div', { class: 'calltree' }, h('div', { class: 'muted small' }, 'Loading the turns…'));
  get(`/api/traces/${encodeURIComponent(id)}/events`).then(list => {
    const model = liveModel(), call = { callId: record.call_id, name: record.definition.name, parentCallId: null };
    model.add({ type: 'call_start', ...call, at: Date.parse(record.started_at) });
    for (const event of list) if (event.kind !== 'state' && event.kind !== 'reduction') model.add({ type: 'trace', ...call, event });
    model.add({ type: 'call_end', ...call, outcome: record.outcome, detail: record.detail ?? '', at: Date.parse(record.started_at) + (cost.wall_ms ?? 0) });
    fill(execution, model.roots.map(root => renderCall(root, true)));
  }, error => fill(execution, h('div', { class: 'muted small' }, error.message)));
  fill(container, h('section', { class: 'panel' },
    h('div', { class: 'row', style: 'justify-content: space-between' }, h('h2', {}, record.definition.name), pill(record.outcome)),
    record.detail && record.outcome !== 'done' ? h('div', { class: 'callout' }, record.detail) : null,
    kv([['Call', h('span', { class: 'mono small' }, record.call_id)], ['Source', record.definition.source ?? record.definition.site],
      ['Executor', `${record.executor.kind}${record.executor.model_id ? ` · ${record.executor.model_id}` : ''}`],
      ['Cost', `${ms(cost.wall_ms)} · ${cost.model_requests ?? 0} model requests · ${tokens(cost.tokens_in)} + ${tokens(cost.tokens_out)} tokens · ${cost.evals ?? 0} evals`],
      ['Started', new Date(record.started_at).toLocaleString()], record.parent_call_id ? ['Caller', h('a', { href: `#traces-call-${encodeURIComponent(record.parent_call_id)}` }, record.parent_call_id)] : null]),
    h('h3', {}, 'Inputs'), h('pre', { class: 'block' }, pretty(record.inputs)),
    h('h3', {}, 'Output'), h('pre', { class: 'block' }, pretty(record.output)),
    h('h3', {}, 'Execution'), execution,
    record.effects?.length ? [h('h3', {}, 'Service calls'), h('pre', { class: 'block' }, record.effects.map(effect => `${effect.service}.${effect.method}(${short(effect.args, 200)}) ${effect.error ? `failed: ${effect.error}` : `→ ${short(effect.result, 200)}`}`).join('\n'))] : null,
    value.children.length ? [h('h3', {}, 'Calls it made'), h('div', { class: 'list' }, value.children.map(child => h('button', { class: 'item', onclick: () => traceDetail(child.call_id, container) },
      h('div', { class: 'title' }, h('span', {}, child.definition_name), pill(child.outcome)))))] : null,
    value.annotations.length ? [h('h3', {}, 'Annotations'), h('pre', { class: 'block' }, value.annotations.map(item => `${item.kind}: ${short(item.value, 300)} (${item.source ?? ''}, ${ago(item.created_at)})`).join('\n'))] : null,
    h('h3', {}, 'Feedback'),
    h('div', { class: 'row' }, h('button', { onclick: feedback('good') }, 'Good: add as example'), h('button', { onclick: feedback('bad') }, 'Bad'),
      h('button', { class: 'primary', onclick: feedback('fix') }, 'Fix with expected result')), expected,
    h('div', { class: 'row' }, h('button', { onclick: attempt(async () => { const list = await get(`/api/traces/${encodeURIComponent(id)}/events`);
      fill(events, h('pre', { class: 'block', style: 'max-height: 600px' }, list.filter(event => event.kind !== 'state').map(event =>
        `${String(event.elapsed_ms ?? '').padStart(7)} ms  ${event.kind}${event.phase ? `/${event.phase}` : ''}  ${short(Object.fromEntries(Object.entries(event).filter(([key]) => !['version', 'seq', 'kind', 'phase', 'observed_at', 'elapsed_ms', 'call_id'].includes(key))), 260)}`).join('\n'))); }) }, 'Show the full event trace')),
    events));
}

// ---- playground -----------------------------------------------------------------------------------------------------
async function playground(_arg, target) {
  const instruction = h('textarea', { id: 'pg-instruction', rows: 3 }, 'Summarize the input in one sentence, then list any action items.');
  const input = h('textarea', { id: 'pg-input', rows: 7 }, 'Hi team, the March invoice run failed for 14 customers because the payment provider rotated its API key. Ops rotated our copy at 10:40. Please re-run billing for those accounts today and email them an apology.');
  const returns = h('input', { id: 'pg-returns', placeholder: 'string', value: '{ summary: string; actions: string[] }' });
  const profile = await profileSelect('pg-profile');
  fill(target, header('Playground', 'Run one natural-language call (natlang ask) on any configured model and watch it work. The run appears under Runs and its trace under Traces.'),
    h('section', { class: 'panel' },
      h('label', { class: 'field' }, 'Instruction', instruction), h('label', { class: 'field' }, 'Input (passed as the call\'s input)', input),
      h('div', { class: 'row' }, h('label', { class: 'field', style: 'flex: 1' }, 'Result type (TypeScript)', returns), h('label', { class: 'field' }, 'Model profile', profile)),
      h('div', { class: 'row' }, h('button', { class: 'primary', onclick: () => startJob({ kind: 'ask', instruction: instruction.value, input: input.value,
        returns: returns.value.trim() || undefined, profile: profile.value || undefined }) }, 'Run'),
        h('button', { onclick: () => programDialog({ instruction: instruction.value, input: input.value, returns: returns.value.trim() || 'string', profile: profile.value || undefined }) }, 'Save as program'))));
}

// ---- models ---------------------------------------------------------------------------------------------------------
async function models(_arg, target) {
  profilesCache = null;
  const [config, providers] = await Promise.all([get('/api/config'), get('/api/providers').catch(() => [])]);
  profilesCache = config;
  const stockRows = h('tbody', {});
  const drawStock = list => fill(stockRows, ...list.map(model => h('tr', {},
    h('td', {}, h('strong', {}, model.label), model.default ? h('span', { class: 'small muted' }, ' · built-in default') : null, h('div', { class: 'small muted mono' }, `${model.source.repository}@${model.source.revision.slice(0, 8)}`)),
    h('td', { class: 'num' }, bytes(model.bytes)),
    h('td', {}, model.installed ? pill('downloaded', 'ok') : model.download && !model.download.done && !model.download.error ?
      h('div', { class: 'row' }, h('div', { class: 'bar' }, h('span', { style: `width: ${Math.round(100 * model.download.received / model.download.total)}%` })), `${Math.round(100 * model.download.received / model.download.total)}%`) :
      model.download?.error ? pill('failed', 'bad') : pill('not downloaded', 'idle')),
    h('td', {}, h('div', { class: 'row' }, !model.installed ? h('button', { onclick: attempt(async () => { await api('POST', `/api/stock/${model.id}/download`); }) }, 'Download') : null,
      h('button', { onclick: () => editProfile(model.id.replace(/[^a-z0-9]+/g, '-'), { local: model.default ? undefined : { stock: model.id } }) }, 'Use in a profile'))))));
  drawStock(config.stock);
  every(attempt(async () => { if (config.stock.some(model => model.download && !model.download.done)) drawStock(config.stock = await get('/api/stock')); }), 1500);
  const profileRows = Object.entries(config.profiles);
  const editor = h('div', { id: 'profile-editor' });
  const providerUi = { h, fill, api, get, toast, attempt, startJob, refreshRail, editProfile };
  const hosted = providersPanel(providerUi, providers, config, route);
  const profilesPanel = h('section', { class: 'panel' }, h('div', { class: 'row', style: 'justify-content: space-between' }, h('h2', {}, 'Profiles'),
      h('label', { class: 'row small muted' }, 'Default', h('select', { id: 'default-profile', onchange: attempt(async event => { await api('POST', '/api/default', { profile: event.target.value || null }); toast('Default profile saved'); refreshRail(); route(); }) },
        h('option', { value: '' }, 'built-in (stock LFM3 1.7B, local)'), profileRows.map(([name]) => h('option', { value: name, selected: name === config.defaultProfile }, name))))),
      profileRows.length ? h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Profile'), h('th', {}, 'Backend'), h('th', {}, 'Model'), h('th', {}, 'Settings'), h('th', {}))),
        h('tbody', {}, profileRows.map(([name, entry]) => h('tr', {}, h('td', {}, h('strong', {}, name), name === config.defaultProfile ? h('span', { class: 'small muted' }, ' · default') : null),
          h('td', {}, entry.error ? pill('invalid', 'bad') : entry.kind), h('td', { class: 'mono small' }, entry.error ?? entry.model),
          h('td', { class: 'small muted' }, profileSettings(entry.profile)),
          h('td', {}, h('div', { class: 'row' }, h('button', { onclick: () => editProfile(name, entry.profile) }, 'Edit'),
            h('button', { onclick: () => startJob({ kind: 'ask', instruction: 'Reply with the single word: ready', input: '', profile: name }) }, 'Test'),
            h('button', { class: 'danger', onclick: attempt(async () => { await api('DELETE', `/api/profiles/${encodeURIComponent(name)}`); toast(`Deleted ${name}`); route(); }) }, 'Delete')))))))) :
        h('div', { class: 'muted' }, 'No profiles yet, so every command runs the built-in local model (stock LFM3 1.7B on the managed llama.cpp server). Connecting a hosted model above creates one; New profile sets up anything else.'));
  fill(target, header('Models', 'Choose the model natlang runs: a hosted provider or a model on this machine. Every natlang command, program and this dashboard use the same profiles (~/.config/natlang/config.json).',
      h('button', { onclick: () => editProfile('', {}) }, 'New profile')),
    // On a first run the hosted-model choices come first; once profiles exist, the profile list leads.
    ...(profileRows.length ? [profilesPanel, hosted] : [hosted, profilesPanel]), editor, await keptModelPanel(),
    h('section', { class: 'panel' }, h('h2', {}, 'Local models'), h('div', { class: 'muted small' }, 'Stock GGUF files natlang serves on its managed llama.cpp server, pinned to one repository revision and checked by SHA-256.'),
      h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Model'), h('th', { class: 'num' }, 'Size'), h('th', {}, 'State'), h('th', {}))), stockRows))));

  function editProfile(name, profile) {
    const kind = profile.provider ? 'provider' : profile.endpoint ? 'endpoint' : profile.native ? 'native' : 'local';
    const nameField = h('input', { id: 'p-name', value: name, placeholder: 'e.g. claude, local-fast' });
    const kindField = h('select', { id: 'p-kind' }, [['local', 'Local model (managed llama.cpp)'], ['provider', 'API provider'], ['endpoint', 'OpenAI-compatible server'], ...(kind === 'native' ? [['native', 'Native checkpoint (natlang setup --native)']] : [])]
      .map(([value, label]) => h('option', { value, selected: value === kind }, label)));
    const fields = h('div', { class: 'stack' });
    const makeDefault = h('input', { type: 'checkbox', id: 'p-default' });
    const drawFields = async () => {
      const value = kindField.value;
      if (value === 'local') {
        fill(fields, h('div', { class: 'row' },
          h('label', { class: 'field' }, 'Model', h('select', { id: 'p-stock' }, config.stock.map(model => h('option', { value: model.id, selected: (profile.local?.stock ?? config.builtinDefault.model) === model.id }, `${model.label}${model.installed ? '' : ` (${bytes(model.bytes)} download)`}`)))),
          h('label', { class: 'field' }, 'Context per slot', h('input', { id: 'p-context', type: 'number', min: 1024, step: 1024, value: profile.local?.contextTokens ?? '', placeholder: '32768' })),
          h('label', { class: 'field' }, 'Parallel slots', h('input', { id: 'p-parallel', type: 'number', min: 1, value: profile.local?.parallel ?? '', placeholder: 'auto' })),
          h('label', { class: 'field' }, 'GPU layers', h('input', { id: 'p-gpu', type: 'number', min: 0, value: profile.local?.gpuLayers ?? '', placeholder: '99 (all)' }))),
          h('label', { class: 'field' }, 'llama-server binary (optional, e.g. a CUDA build)', h('input', { id: 'p-server', value: profile.local?.server ?? '', placeholder: 'managed CPU/Metal build' })));
      } else if (value === 'provider') {
        // A new provider profile starts on a connected provider (else Anthropic) and its recommended model, listed first.
        const initial = profile.provider ?? providers.find(item => item.connected)?.id ?? 'anthropic';
        const providerField = h('select', { id: 'p-provider' }, providers.map(item => h('option', { value: item.id, selected: item.id === initial }, `${item.name}${item.connected ? ` · ${connection(item)}` : ''}`)));
        const modelField = h('select', { id: 'p-model' });
        const keyField = h('input', { id: 'p-key', type: 'password', placeholder: 'leave empty to keep the saved key or use the environment' });
        const signInRow = h('div', { class: 'row small' });
        const loadModels = attempt(async (refresh = false) => {
          const provider = providers.find(item => item.id === providerField.value);
          const list = await get(`/api/providers/${encodeURIComponent(providerField.value)}/models${refresh === true ? '?refresh=1' : ''}`);
          const keep = provider?.id === profile.provider ? profile.model : undefined;
          fill(modelField, ...list.map(item => h('option', { value: item.id, selected: item.id === keep }, `${item.name}${item.family ? ` · newest ${item.family === 'luna' ? 'Luna' : 'Haiku'}` : ''}${item.recommended ? ' (recommended)' : ''}${item.contextWindow ? ` · ${tokens(item.contextWindow)} context` : ''}`)));
          fill(signInRow, h('span', { class: 'muted' }, provider?.connected ? `Connected: ${connection(provider)}.` : 'Not connected yet.'),
            provider?.oauth ? h('button', { type: 'button', onclick: async () => { if (await signIn(providerUi, provider, 'oauth')) loadModels(); } }, provider.oauth.label ?? 'Sign in with subscription') : null,
            h('button', { type: 'button', title: 'Fetch the provider\'s current model list', onclick: () => loadModels(true) }, 'Refresh models'));
        });
        providerField.addEventListener('change', () => loadModels());
        fill(fields, h('div', { class: 'row' }, h('label', { class: 'field' }, 'Provider', providerField), h('label', { class: 'field', style: 'flex: 1' }, 'Model', modelField)),
          h('label', { class: 'field' }, 'API key (saved to natlang\'s credential store for this provider)', keyField),
          signInRow);
        if (providers.length) await loadModels();
        if (profile.provider && !name) nameField.value = profile.provider;
      } else if (value === 'endpoint') {
        fill(fields, h('div', { class: 'row' },
          h('label', { class: 'field', style: 'flex: 2' }, 'Server URL', h('input', { id: 'p-endpoint', value: profile.endpoint ?? '', placeholder: 'http://127.0.0.1:8080' })),
          h('label', { class: 'field', style: 'flex: 1' }, 'Model name', h('input', { id: 'p-model-name', value: profile.model ?? '', placeholder: 'as the server expects' }))),
          h('div', { class: 'row' }, h('label', { class: 'field' }, 'API key variable', h('input', { id: 'p-key-env', value: profile.apiKeyEnv ?? 'NATLANG_API_KEY' })),
            h('label', { class: 'field', style: 'flex: 1' }, 'API key (optional; saved for that variable)', h('input', { id: 'p-key', type: 'password' }))));
      } else fill(fields, h('pre', { class: 'block' }, JSON.stringify(profile, null, 2)), h('div', { class: 'small muted' }, 'Native profiles are written by natlang setup --native; edit them there.'));
    };
    kindField.addEventListener('change', drawFields);
    const temperature = h('input', { id: 'p-temperature', type: 'number', min: 0, max: 2, step: 0.1, value: profile.request?.temperature ?? '', placeholder: 'model default' });
    // llama.cpp's per-request reasoning budget: unlimited is how the stock model was evaluated; a budget trades some
    // accuracy for speed, mostly on a CPU. Pi providers set reasoning through their own options.
    const budget = profile.request?.thinking_budget_tokens;
    const thinking = h('select', { id: 'p-thinking', title: 'Reasoning tokens per model turn' },
      [['', 'unlimited'], ['2048', '2,048 tokens'], ['1024', '1,024 tokens'], ['512', '512 tokens'], ['256', '256 tokens'], ['0', 'off']]
        .map(([value, label]) => h('option', { value, selected: String(budget ?? '') === value }, label)));
    const save = attempt(async () => {
      const profileName = nameField.value.trim();
      if (!profileName) throw new Error('Give the profile a name');
      const value = kindField.value, number = id => { const raw = document.getElementById(id)?.value; return raw === '' || raw === undefined ? undefined : Number(raw); };
      let next;
      if (value === 'local') {
        const stock = document.getElementById('p-stock').value, server = document.getElementById('p-server').value.trim();
        const local = Object.fromEntries(Object.entries({ stock: stock === config.builtinDefault.model ? undefined : stock, contextTokens: number('p-context'),
          parallel: number('p-parallel'), gpuLayers: number('p-gpu'), server: server || undefined }).filter(([, v]) => v !== undefined));
        next = { ...(Object.keys(local).length ? { local } : {}) };
      } else if (value === 'provider') {
        const provider = document.getElementById('p-provider').value, key = document.getElementById('p-key').value.trim();
        if (key) await api('PUT', `/api/credentials/${encodeURIComponent(provider)}`, { key });
        next = { provider, model: document.getElementById('p-model').value };
      } else if (value === 'endpoint') {
        const apiKeyEnv = document.getElementById('p-key-env').value.trim() || 'NATLANG_API_KEY', key = document.getElementById('p-key').value.trim();
        if (key) await api('PUT', `/api/credentials/${encodeURIComponent(`env:${apiKeyEnv}`)}`, { key });
        next = { endpoint: document.getElementById('p-endpoint').value.trim(), ...(document.getElementById('p-model-name').value.trim() ? { model: document.getElementById('p-model-name').value.trim() } : {}), apiKeyEnv };
      } else next = profile;
      if (value !== 'provider') {
        const request = { ...(profile.request ?? {}) }, t = number('p-temperature'), think = number('p-thinking');
        if (t === undefined) delete request.temperature; else request.temperature = t;
        if (think === undefined) delete request.thinking_budget_tokens; else request.thinking_budget_tokens = think;
        if (Object.keys(request).length) next.request = request; else delete next.request;
      }
      if (profile.escalate) next.escalate = profile.escalate;
      await api('PUT', `/api/profiles/${encodeURIComponent(profileName)}`, { profile: next, makeDefault: makeDefault.checked });
      toast(`Saved profile ${profileName}`); refreshRail(); route();
    });
    fill(editor, h('section', { class: 'panel' }, h('h2', {}, name ? `Edit ${name}` : 'New profile'),
      h('div', { class: 'row' }, h('label', { class: 'field' }, 'Name', nameField), h('label', { class: 'field' }, 'Backend', kindField),
        h('label', { class: 'field' }, 'Temperature', temperature), h('label', { class: 'field' }, 'Thinking', thinking)),
      fields, h('label', { class: 'row small' }, makeDefault, 'Make this the default profile'),
      h('div', { class: 'row' }, h('button', { class: 'primary', onclick: save }, 'Save profile'), h('button', { onclick: () => fill(editor, ) }, 'Cancel'))));
    drawFields(); editor.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// ---- system ---------------------------------------------------------------------------------------------------------
async function system(_arg, target) {
  const [status, warm] = await Promise.all([get('/api/status'), get('/api/warm')]);
  const runtime = status.runtime;
  const warmPanel = h('section', { class: 'panel' }, h('div', { class: 'row', style: 'justify-content: space-between' }, h('h2', {}, 'Warm prompt openings'),
      warm.length ? h('button', { onclick: attempt(async () => { await api('DELETE', '/api/warm'); toast('Cleared; the next calls prefill again'); route(); }) }, 'Clear') : null),
    h('div', { class: 'small muted' }, 'Every call starts with the same long system prompt and tools. The managed server prefills each distinct opening once, saves it here and restores it into every slot, so later calls and runs skip that prefill.'),
    warm.length ? h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'State'), h('th', { class: 'num' }, 'Tokens'), h('th', { class: 'num' }, 'Size'), h('th', {}, 'Model · build · slots'), h('th', {}, 'Last used'))),
      h('tbody', {}, warm.map(item => h('tr', {}, h('td', { class: 'mono small' }, item.file.slice(0, 16)), h('td', { class: 'num' }, item.tokens ?? '–'),
        h('td', { class: 'num' }, bytes(item.bytes)), h('td', { class: 'mono small' }, `${item.model ?? '?'} · b${item.build ?? '?'} · ${item.layout ? item.layout.replace('x', ' × ') : '?'}`), h('td', {}, ago(item.usedAt))))))) :
      h('div', { class: 'muted' }, 'None saved yet; the first call of each kind saves one.'));
  const mode = h('select', { id: 'spec-mode', onchange: attempt(async event => { await api('PUT', '/api/store/settings', { specialization: event.target.value }); toast('Saved'); }) },
    ['off', 'shadow', 'on'].map(value => h('option', { value, selected: status.calls?.settings.specialization === value }, value)));
  fill(target, header('System', 'The model runtime, the call record store and where natlang keeps its state.'),
    h('section', { class: 'panel' }, h('div', { class: 'row', style: 'justify-content: space-between' }, h('h2', {}, 'llama.cpp runtime'),
      h('div', { class: 'row' }, runtime.selected?.fork ? null : h('button', { title: `Compile ${runtime.fork.repository} at ${runtime.fork.commit.slice(0, 10)} (needs git, cmake and a C++ compiler; CUDA when nvcc is present)`,
        onclick: () => startJob({ kind: 'runtime-build' }) }, 'Build the natlang fork'),
        h('button', { class: 'primary', onclick: () => startJob({ kind: 'setup' }) }, runtime.selected ? 'Check and download the default model' : 'Install runtime and default model'))),
      kv([['Selected', runtime.selected ? `${runtime.selected.source} · ${runtime.selected.version} ${runtime.selected.source === 'fork' ? `(${runtime.fork.repository} at ${runtime.fork.commit.slice(0, 12)})` : `(build ${runtime.selected.build})`}` : 'none'],
        runtime.selected ? ['Build', runtime.selected.fork ? 'natlang fork: decision scoring, guided generation and native states' :
          'upstream llama.cpp: decisions fall back to model turns, no guided generation or tuned soft states'] : null,
        ['Path', runtime.selected?.path], ['Pinned', `${runtime.recommended.version} (build ${runtime.recommended.build})`],
        runtime.selected?.reason ? ['Note', runtime.selected.reason] : null]),
      runtime.candidates.length ? h('div', { class: 'small muted', style: 'white-space: pre-line' }, runtime.candidates.map(item => `${item.source}: ${item.path}${item.compatible ? '' : ` (${item.reason})`}`).join('\n')) : null),
    h('section', { class: 'panel' }, h('div', { class: 'row', style: 'justify-content: space-between' }, h('h2', {}, 'Call records and specialization'),
      h('button', { onclick: () => startJob({ kind: 'specialize' }) }, 'Run the specializer')),
      status.calls ? kv([['Store', status.calls.root], ['Calls', status.calls.calls], ['Size', bytes(status.calls.bytes)],
        ['Specialization', h('span', {}, mode, h('span', { class: 'small muted' }, ' compiled crisp cases: off, shadow (checked, not served) or on'))]]) :
        h('div', { class: 'muted' }, 'Recording is off (NATLANG_CALL_STORE=off).')),
    warmPanel,
    h('section', { class: 'panel' }, h('h2', {}, 'Paths'), kv([['Project root', status.root], ['Model profiles', status.configPath], ['natlang', `${status.version} on Node ${status.node}`]])));
}

refreshRail(); setInterval(refreshRail, 10000);
// Elapsed times of work in progress tick between redraws (a CPU prefill can take a minute with nothing else to show).
setInterval(() => { for (const element of document.querySelectorAll('[data-since]')) element.textContent = ms(Date.now() - Number(element.dataset.since)); }, 1000);
route();
