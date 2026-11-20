import { num, fmt, gb, pct, signed, dur, ago, clock, dayClock, shortUnit, health, improving, latestHeld, metricSeries, metricNames, memorySegments, outcomeParts, matches, PRESETS, SNAP_STALE_MS } from './model.mjs';
import { h, lineChart, valueTable, spark, segBar, slotVar } from './charts.mjs';
import { dataView, link as dl } from './data.mjs';
import { editRef } from '../ops-protocol.mjs';

let src = null;      // the data source (session or preview), set by start()
const $ = id => document.getElementById(id);
const store = (k, v) => { try { if (v === undefined) return JSON.parse(localStorage.getItem(k)); localStorage.setItem(k, JSON.stringify(v)); } catch { return null; } };
const now = () => Date.now();

/* ---------- state ---------- */
const S = {
  snap: null, live: new Map(), hist: new Map(), runs: new Map(), events: [], notices: new Map(),
  acked: store('ops.acked') || {}, slots: store('ops.slots') || {}, tab: 'overview', sel: new Set(), selTouched: false,
  smooth: store('ops.smooth') ?? 0.3, logy: false, extra: store('ops.extra') || '', tables: new Set(), open: new Set(), showAcked: false,
  q: '', procMachine: 'all', procState: 'live', evAll: false, hold: false, sent: 0, route: '', gate: null,
  dui: { forms: {}, taskTab: {}, file: {}, traj: { reasoning: false, context: false, results: false, ...(store('ops.traj') || {}) }, tokens: {}, cmp: [] },
};
const runLink = id => h('a', { class: 'btn sm', href: dl.run(id), title: 'Steps, selected groups, episodes and held evals of this run' }, 'Training data');
const TABS = [['overview', 'Overview'], ['training', 'Training'], ['data', 'Training data'], ['held', 'Held-out'], ['units', 'Units'], ['machines', 'Machines'], ['teachers', 'Teachers'], ['backlog', 'Queue & fallow'], ['investigations', 'Investigations'], ['inbox', 'Inbox']];
const SMOOTH = [['Raw', 0], ['Light', 0.5], ['Medium', 0.3], ['Heavy', 0.1]];
const LEDGER_FLOOR_GB = 15;
const live = () => (S.snap?.processes || []).filter(p => p.state === 'running');

/* ---------- data intake ---------- */
let paintTimer = null, lastPaint = 0;
function schedule() {
  if (paintTimer) return;
  paintTimer = setTimeout(() => { paintTimer = null; if (S.hold) { schedule(); return; } lastPaint = now(); if (S.tab === 'data') { freshness(); return; } render(); }, Math.max(250, 2000 - (now() - lastPaint)));
}
function notice(key, text) { text ? S.notices.set(key, text) : S.notices.delete(key); $('notices').textContent = [...S.notices.values()].join(' · '); }

function takeSnapshot(rec) {
  if (!rec?.data || rec.data.v !== 3) return;
  if (S.snap && rec.data.ts < S.snap.ts) return;
  S.snap = rec.data;
  if (!S.selTouched) defaultSelection();
  schedule();
}
function takeMetric(rec) {
  const d = rec.data;
  if (!d?.run_id || !num(d.step)) return;
  if (rec.type === 'train') {
    if (!S.runs.has(d.run_id)) S.runs.set(d.run_id, new Map());
    S.runs.get(d.run_id).set(d.step, { ...d, ts: rec.ts });
    if (!S.selTouched) defaultSelection();
  }
  schedule();
}
function takeMachine(rec) {
  const d = rec.data; if (!d?.machine) return;
  const list = S.hist.get(d.machine) || S.hist.set(d.machine, []).get(d.machine);
  list.push({ ts: rec.ts, ...d }); if (list.length > 1500) list.shift();
  schedule();
}
function takeEvent(rec) { S.events.push({ ts: rec.ts, ...rec.data }); if (S.events.length > 400) S.events.shift(); schedule(); }

/* Exporter state arrives through the session's on-demand queries; a tick names the files that changed. */
const offsets = { metrics: 0, machines: 0, events: 0 }, seen = {};
let pulling = false;
async function pull(tick) {
  if (pulling || !src) return;
  pulling = true;
  try {
    if (!S.snap || tick?.snapshot !== seen.snapshot) {
      const snap = await src.query('ops.snapshot');
      seen.snapshot = tick?.snapshot; takeSnapshot({ data: snap }); notice('snap', '');
    }
    for (const [key, fn] of [['metrics', takeMetric], ['machines', takeMachine], ['events', takeEvent]]) {
      if (offsets[key] && tick?.[key] === seen[key]) continue;
      const r = await src.query(`ops.${key}`, { offset: offsets[key] });
      for (const rec of r.records) fn(rec);
      offsets[key] = r.offset; seen[key] = tick?.[key];
    }
  } catch (e) { notice('snap', `Could not read the ops state: ${e.message}`); }
  finally { pulling = false; }
}

/* ---------- run selection and colours: a run keeps its colour slot for good (colour follows the entity) ---------- */
function slotOf(id) {
  if (S.slots[id] == null) { S.slots[id] = Object.keys(S.slots).length % 8; store('ops.slots', S.slots); }
  return S.slots[id];
}
function defaultSelection() {
  const rows = (S.snap?.runs || []).filter(r => r.kind === 'grpo' || r.kind === 'other');
  if (!rows.length) return;
  S.sel = new Set(rows.slice(0, 4).map(r => r.id));
}
function toggleRun(id) { S.selTouched = true; S.sel.has(id) ? S.sel.delete(id) : S.sel.size < 8 && S.sel.add(id); render(); }

/* ---------- ask the supervising agent ---------- */
const dlg = $('ask');
let askCtx = null;
function openAsk(ctx) {
  askCtx = ctx;
  const stop = ctx.mode === 'stop';
  $('ask-title').textContent = stop ? 'Request to stop a unit' : 'Ask the supervising agent';
  $('ask-sub').textContent = ctx.subject.title || ctx.subject.id;
  $('ask-label').textContent = stop ? 'Why should it stop?' : 'Your question or request';
  $('ask-note').textContent = stop ? 'This records a request for the supervising agent. Nothing is stopped by this page.' : 'Recorded as a signed operation; the supervising agent reads it from its inbox and its answer appears in the Inbox tab.';
  $('ask-status').textContent = ''; $('ask-text').value = ''; $('ask-send').disabled = false;
  dlg.showModal(); $('ask-text').focus();
}
$('ask-cancel').onclick = () => dlg.close();
$('ask-form').addEventListener('submit', async e => {
  e.preventDefault();
  const text = $('ask-text').value.trim(); if (!text || !askCtx) return;
  $('ask-send').disabled = true; $('ask-status').textContent = 'Sending…';
  try {
    const r = await src.command('ask', { kind: askCtx.mode === 'stop' ? 'request_stop' : 'ask', subject: askCtx.subject, text });
    $('ask-status').textContent = r.status === 'applied' ? (r.outcome === 'accepted' ? 'Recorded. The supervising agent reads it from its inbox; answers appear under Inbox.' : `Rejected: ${r.rejection || r.outcome}`) : 'Saved on this device; it is applied when the native participant is reachable.';
    setTimeout(() => dlg.close(), 1200);
  } catch (err) { $('ask-status').textContent = `Could not send: ${err.message || err}`; $('ask-send').disabled = false; }
});
const askBtn = (kind, id, title, label = 'Ask') => h('button', { class: 'btn sm', type: 'button', onclick: () => openAsk({ mode: 'ask', subject: { kind, id: String(id), title: String(title).slice(0, 160) } }) }, label);
async function ackAlert(id) {
  S.acked[id] = now(); store('ops.acked', S.acked); render();
  try { await src.command('ask', { kind: 'ack_alert', subject: { kind: 'alert', id: String(id).slice(0, 300) }, text: 'acknowledged on the dashboard' }); } catch { /* local acknowledgement stands */ }
}

/* ---------- small view helpers ---------- */
const machineChip = m => h('span', { class: `chip ${m}` }, m === 'dgx' ? 'DGX' : m === 'pop' ? 'Pop' : m || '');
const levelIcon = l => l === 'crit' ? '✖' : l === 'warn' ? '▲' : '✔';
const meter = (frac, tone = '') => h('div', { class: `meter ${tone}`, role: 'img', 'aria-label': `${Math.round(100 * frac)}%` }, h('b', { style: `width:${Math.min(100, Math.max(0, 100 * frac)).toFixed(1)}%` }));
const section = (title, note, ...right) => h('div', { class: 'sect' }, h('h2', {}, title), note ? h('span', { class: 'muted' }, note) : null, ...right);
const stat = (value, label) => h('div', { class: 'stat' }, h('b', {}, value), h('span', {}, label));
const dgxHost = () => S.snap?.machines?.dgx;

function progressView(p, state) {
  if (!p) return h('span', { class: 'muted small' }, state === 'running' ? 'no progress signal' : '');
  const frac = p.total ? p.done / p.total : null;
  return h('div', {}, frac != null ? meter(frac, 'good') : null,
    h('div', { class: 'small muted' }, `${fmt(p.done)}${p.total ? ` / ${fmt(p.total)}` : ''} ${p.unit}${frac != null ? ` (${(100 * frac).toFixed(frac < 0.1 ? 1 : 0)}%)` : ''}`));
}
const etaText = p => num(p.eta_s) ? `ETA ${dur(p.eta_s)} (${clock(now() + p.eta_s * 1000)})` : p.state === 'running' ? 'ETA unknown' : '';

/* ---------- alerts ---------- */
function alertsView(hv) {
  const open = hv.items.filter(i => !S.acked[i.id]), acked = hv.items.filter(i => S.acked[i.id]);
  if (!open.length && !acked.length) return null;
  const box = h('div', { class: 'banner', role: 'region', 'aria-label': 'Alerts' });
  for (const a of open) box.append(h('div', { class: `alert ${a.level}` }, h('span', { class: 'ic', 'aria-hidden': 'true' }, levelIcon(a.level)),
    h('div', { class: 'what' }, h('b', {}, a.level === 'crit' ? 'Critical: ' : 'Warning: '), a.text, ' ', a.machine ? machineChip(a.machine) : null),
    h('div', { class: 'acts' }, askBtn('alert', a.id, a.text), a.system ? null : h('button', { class: 'btn sm', type: 'button', onclick: () => ackAlert(a.id) }, 'Acknowledge'))));
  if (acked.length) box.append(h('div', { class: 'small muted' }, `${acked.length} acknowledged `, h('button', { class: 'btn sm', type: 'button', onclick: () => { S.showAcked = !S.showAcked; render(); } }, S.showAcked ? 'Hide' : 'Show'),
    S.showAcked ? h('ul', {}, acked.map(a => h('li', {}, a.text))) : null));
  return box;
}

/* ---------- overview ---------- */
function activeRun() {
  const runs = S.snap?.runs || [];
  return runs.find(r => r.active && r.kind !== 'bench') || runs.find(r => r.kind === 'grpo') || null;
}
function heroHealth(hv) {
  const snap = S.snap, dgx = dgxHost();
  const el = h('div', { class: `card hero ${hv.level}` }, h('div', { class: 'q' }, 'Healthy?'),
    h('div', { class: `big tone-${hv.level}` }, h('span', { 'aria-hidden': 'true' }, levelIcon(hv.level)), hv.open.length ? `${hv.open.length} issue${hv.open.length === 1 ? '' : 's'}` : hv.items.length ? 'Healthy (acknowledged)' : 'Healthy'));
  if (hv.open.length) el.append(h('div', { class: 'sub' }, hv.open.slice(0, 2).map(a => h('div', {}, a.text))));
  else el.append(h('div', { class: 'sub' }, 'No open alerts. Exporter, units and disks are within their limits.'));
  if (dgx) {
    const disks = dgx.disks || [], head = dgx.ledger?.headroom_gb;
    el.append(h('div', { class: 'rows small' },
      h('div', { class: 'mini' }, h('span', { class: 't' }, 'DGX ledger headroom'), h('span', { class: `r ${num(head) && head < LEDGER_FLOOR_GB ? 'tone-crit' : ''}` }, `${gb(head)} (floor ${LEDGER_FLOOR_GB})`)),
      ...disks.map(d => h('div', { class: 'mini' }, h('span', { class: 't' }, `DGX ${d.mount}`), h('span', { class: `r ${num(d.floor_gb) && d.free_gb < d.floor_gb ? 'tone-crit' : ''}` }, `${d.free_gb} GB free (floor ${d.floor_gb})`)))));
  }
  return el;
}
function heroRunning() {
  const ps = live(), by = m => ps.filter(p => p.machine === m).length;
  const el = h('div', { class: 'card hero flat' }, h('div', { class: 'q' }, 'What is running?'),
    h('div', { class: 'big' }, `${ps.length} unit${ps.length === 1 ? '' : 's'}`), h('div', { class: 'sub' }, `${by('dgx')} on DGX · ${by('pop')} on Pop`));
  const rows = h('div', { class: 'rows' });
  for (const p of ps.filter(p => p.kind !== 'service').slice(0, 4)) rows.append(h('div', {}, h('div', { class: 'mini' }, h('span', { class: 't', title: p.title }, p.title), h('span', { class: 'r' }, etaText(p))), progressView(p.progress, p.state)));
  if (!rows.childNodes.length) rows.append(h('div', { class: 'muted' }, 'Nothing but plumbing is running.'));
  el.append(rows, h('div', { class: 'row', style: 'margin-top:8px' }, h('button', { class: 'btn sm', onclick: () => go('units') }, 'All units')));
  return el;
}
function heroTraining() {
  const run = activeRun(), v = improving(run);
  const el = h('div', { class: `card hero ${v.tone}` }, h('div', { class: 'q' }, 'Is training improving?'));
  if (!run) return el.append(h('div', { class: 'big' }, 'No run yet')), el;
  el.append(h('div', { class: `big tone-${v.tone}` }, v.headline), h('div', { class: 'sub' }, v.sub ? `${v.sub} · ${v.detail}` : v.detail),
    h('div', { class: 'small muted', style: 'margin-top:4px' }, `${run.id} · ${run.steps} steps · ${run.active ? 'running' : 'idle'}${run.last_ts ? ` · last step ${ago(run.last_ts, now())}` : ''}`));
  const sp = run.spark || {};
  el.append(h('div', { class: 'grid', style: 'grid-template-columns:1fr 1fr;gap:12px;margin-top:10px' },
    h('div', {}, h('div', { class: 'small muted' }, `reward ${fmt(sp.reward_mean?.at(-1))}`), spark(sp.reward_mean || [], { slot: slotOf(run.id), label: 'reward' })),
    h('div', {}, h('div', { class: 'small muted' }, `KL ${fmt(sp.kl_mean?.at(-1))}`), spark(sp.kl_mean || [], { slot: slotOf(run.id), label: 'KL' }))));
  el.append(h('div', { class: 'row', style: 'margin-top:8px' }, h('button', { class: 'btn sm', onclick: () => go('training') }, 'Charts'), h('button', { class: 'btn sm', onclick: () => go('held') }, 'Held-out results'), runLink(run.id)));
  return el;
}

function memoryBlock(m, { big = false } = {}) {
  const ms = memorySegments(m), colors = (_, i) => slotVar(i);
  const claims = ms.segs.filter(s => s.kind === 'claim');
  const parts = [...claims.map((s, i) => ({ label: `${s.label} (ledger claim)`, value: s.gb, color: slotVar(i) })), ...ms.segs.filter(s => s.kind === 'other').map(s => ({ label: s.label, value: s.gb, color: 'var(--muted-fill)' }))];
  const box = h('div', {}, h('div', { class: 'row', style: 'justify-content:space-between' }, h('span', {}, h('b', {}, gb(ms.used)), h('span', { class: 'muted' }, ` used of ${gb(ms.total, 0)}`)), h('span', { class: 'muted small' }, `${gb(m.mem_available_gb)} available`)),
    segBar(parts, { total: ms.total, height: big ? 18 : 14, label: `Memory: ${gb(ms.used)} used of ${gb(ms.total, 0)}` }));
  const key = h('div', { class: 'legend', style: 'margin-top:6px' }, parts.map(p => h('span', { class: 'key' }, h('i', { style: `background:${p.color};height:10px;width:10px` }), `${p.label.replace(' (ledger claim)', '')} ${gb(p.value)}`)));
  box.append(key);
  return box;
}
function diskRows(m) {
  return (m.disks || []).map(d => {
    const floor = d.floor_gb, low = num(floor) && d.free_gb < floor, near = num(floor) && d.free_gb < floor + 20;
    return h('div', { class: 'disk' }, h('div', { class: 'row', style: 'justify-content:space-between' }, h('span', {}, `${d.mount}`), h('span', { class: low ? 'tone-crit' : near ? 'tone-warn' : 'muted' }, `${d.free_gb} GB free of ${d.size_gb}${num(floor) ? ` · floor ${floor} GB` : ''}${low ? ' ✖' : near ? ' ▲' : ''}`)),
      meter((d.size_gb - d.free_gb) / d.size_gb, low ? 'crit' : near ? 'warn' : ''));
  });
}
function machineCard(name, m, { detail = false } = {}) {
  const title = name === 'dgx' ? 'DGX (GB10, unified memory)' : 'Pop (RTX 4060 8 GB)';
  if (!m) return h('div', { class: 'card' }, h('header', {}, h('h2', {}, title)), h('div', { class: 'alert crit' }, h('span', { class: 'ic' }, '✖'), h('div', { class: 'what' }, 'No data from this machine. The exporter could not reach it.')));
  const hist = (S.hist.get(name) || []).filter(r => now() - r.ts < 12 * 3600e3);
  const head = m.ledger?.headroom_gb, lowHead = name === 'dgx' && num(head) && head < LEDGER_FLOOR_GB;
  const liveRec = name === 'dgx' ? S.live.get('dgx')?.data : null;
  const el = h('div', { class: 'card' }, h('header', {}, h('h2', {}, title), h('span', { class: 'muted small' }, `load ${fmt(m.load1, 2)}${m.swap_used_gb > 0.5 ? ` · swap ${gb(m.swap_used_gb)}` : ''}`)),
    memoryBlock(m, { big: detail }),
    h('div', { class: 'grid', style: 'grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-top:12px' },
      stat(pct(m.cpu_percent), 'CPU'), stat(pct(m.gpu_util_percent), 'GPU util'),
      stat(name === 'dgx' ? 'shared' : gb(m.gpu_mem_used_gb), name === 'dgx' ? 'GPU memory is system RAM' : 'GPU memory used'),
      stat(gb(head), `${name === 'dgx' ? 'ledger headroom' : 'ledger headroom (Pop)'}${lowHead ? ' ▲' : ''}`)));
  if (liveRec) el.append(h('div', { class: 'small muted', style: 'margin-top:6px' }, `Live sample ${ago(S.live.get('dgx').ts, now())}: CPU ${pct(liveRec.cpu_percent ?? liveRec.cpu?.percent)}, GPU ${pct(liveRec.gpus?.[0]?.utilization_percent ?? liveRec.gpu?.utilization_percent)}${liveRec.gpus?.[0]?.unavailable?.memory_used_bytes ? `, GPU memory ${liveRec.gpus[0].unavailable.memory_used_bytes.replace(/_/g, ' ')} (shared with RAM)` : ''}.`));
  el.append(...diskRows(m));
  if (hist.length > 2) el.append(h('div', { class: 'grid', style: 'grid-template-columns:1fr 1fr;gap:12px;margin-top:10px' },
    h('div', {}, h('div', { class: 'small muted' }, 'memory used, 12 h'), spark(hist.map(r => r.mem_used_gb), { label: 'memory used' })),
    h('div', {}, h('div', { class: 'small muted' }, 'GPU util, 12 h'), spark(hist.map(r => r.gpu_util_percent), { label: 'GPU util', slot: 2 }))));
  return el;
}
function eventsView(limit = 8) {
  const evs = [...S.events].sort((a, b) => b.ts - a.ts).slice(0, S.evAll ? 80 : limit);
  if (!evs.length) return h('div', { class: 'muted' }, 'No events recorded yet.');
  const evLink = e => e.kind === 'flag' && e.run_id && num(e.step) ? dl.step('grpo:' + e.run_id.replace(/\./g, '/'), e.step) : e.kind === 'held' && e.run_id ? dl.ds('grpo:' + e.run_id.replace(/\./g, '/'), { part: 'held' }) : null;
  return h('div', { class: 'timeline' }, evs.map(e => h('div', { class: `ev ${e.level || ''}` }, h('time', { title: dayClock(e.ts) }, ago(e.ts, now())), h('span', { class: 't' }, evLink(e) ? h('a', { href: evLink(e), title: e.kind === 'flag' ? 'What this step trained on' : 'Held episodes' }, e.text || e.kind) : (e.text || e.kind)))));
}
function teachersMini() {
  const tt = S.snap?.teachers;
  if (!tt) return h('div', { class: 'card' }, h('header', {}, h('h2', {}, 'Teacher generation')), h('div', { class: 'muted' }, 'No pool configured.'));
  const live_ = live().find(p => p.unit.startsWith('tgen-'));
  return h('div', { class: 'card' }, h('header', {}, h('h2', {}, `Teacher generation · ${tt.pool}`), h('button', { class: 'btn sm', onclick: () => go('teachers') }, 'Details')),
    h('div', { class: 'grid', style: 'grid-template-columns:repeat(3,1fr);gap:10px' }, stat(`${fmt(tt.tasks_done)}`, `of ${fmt(tt.eligible)} tasks`), stat(fmt(tt.success), 'verified successes'), stat(`${fmt(tt.success_last_hour)}/h`, 'successes, last hour')),
    tt.eligible ? h('div', { style: 'margin-top:10px' }, meter(tt.tasks_done / tt.eligible), h('div', { class: 'small muted' }, `${(100 * tt.tasks_done / tt.eligible).toFixed(1)}% of the pool${live_ && num(live_.eta_s) ? ` · ETA ${dur(live_.eta_s)}` : ''} · last activity ${ago(tt.last_activity_ts, now())}`)) : null);
}

function overview(hv) {
  const v = h('div', {});
  v.append(h('div', { class: 'grid g3' }, heroHealth(hv), heroRunning(), heroTraining()));
  v.append(section('Machines', 'live resources'));
  v.append(h('div', { class: 'grid g2' }, machineCard('dgx', dgxHost()), machineCard('pop', S.snap?.machines?.pop)));
  v.append(section('Teacher generation and recent events'));
  v.append(h('div', { class: 'grid g2' }, teachersMini(), h('div', { class: 'card' }, h('header', {}, h('h2', {}, 'Recent events')), eventsView())));
  return v;
}

/* ---------- training ---------- */
function runChips() {
  const rows = (S.snap?.runs || []);
  const mk = r => h('button', { class: `runchip${r.active ? ' active' : ''}`, type: 'button', 'aria-pressed': S.sel.has(r.id), onclick: () => toggleRun(r.id), title: `${r.steps} steps` }, h('i', { style: S.sel.has(r.id) ? `background:${slotVar(slotOf(r.id))};height:4px` : '' }), r.id);
  const main = rows.filter(r => r.kind !== 'bench'), bench = rows.filter(r => r.kind === 'bench');
  return h('div', { class: 'row', role: 'group', 'aria-label': 'Runs shown' }, main.map(mk), bench.length ? h('details', { open: S.open.has('bench'), ontoggle: e => e.target.open ? S.open.add('bench') : S.open.delete('bench') }, h('summary', {}, `${bench.length} bench / hunt arms`), h('div', { class: 'row', style: 'margin-top:6px' }, bench.map(mk))) : null);
}
function chartPanel({ key, title, note, series, yFmt, extra = {}, xLabel = 'step', xFmt, xKind }) {
  const ch = lineChart({ series, smooth: S.smooth, logy: S.logy, yLabel: title, xLabel, xFmt, xKind, yFmt, ...extra });
  const showTable = S.tables.has(key);
  return h('div', { class: 'card' }, h('header', {}, h('h2', {}, title), h('button', { class: 'btn sm', type: 'button', 'aria-pressed': showTable, onclick: () => { showTable ? S.tables.delete(key) : S.tables.add(key); render(); } }, showTable ? 'Chart' : 'Table')),
    showTable ? valueTable(series, { xLabel, xFmt: xFmt || (v => v), yFmt }) : ch.el, note ? h('div', { class: 'panel-note' }, note) : null);
}
function training() {
  const v = h('div', {});
  const ids = [...S.sel], names = metricNames(S.runs);
  v.append(h('div', { class: 'toolbar' }, runChips(), h('span', { class: 'row', style: 'margin-left:auto' },
    h('span', { class: 'muted small' }, 'Smoothing'), h('span', { class: 'seg-ctl', role: 'group', 'aria-label': 'Smoothing' }, SMOOTH.map(([label, a]) => h('button', { type: 'button', 'aria-pressed': S.smooth === a, onclick: () => { S.smooth = a; store('ops.smooth', a); render(); } }, label))),
    h('button', { class: 'btn', type: 'button', 'aria-pressed': S.logy, onclick: () => { S.logy = !S.logy; render(); } }, 'Log y'))));
  if (!S.runs.size) return v.append(h('div', { class: 'notice' }, 'No training steps retained yet: the metrics stream is empty or still loading.')), v;
  const flagged = metric => ids.flatMap(id => [...(S.runs.get(id) || [])].filter(([, r]) => r.metrics?.grad_spike || r.metrics?.behaviour_shift || r.metrics?.reasoning_shift).map(([step, r]) => ({ x: step, label: ['grad_spike', 'behaviour_shift', 'reasoning_shift'].filter(f => r.metrics[f]).map(f => f.split('_')[0]).join('+') })));
  const grid = h('div', { class: 'grid g2' });
  const panels = PRESETS.map(([title, metric, note]) => ({ title, metric, note }));
  if (S.extra && !panels.some(p => p.metric === S.extra)) panels.push({ title: S.extra, metric: S.extra, note: 'Custom metric.' });
  for (const p of panels) {
    const series = metricSeries(S.runs, ids, p.metric).map(s => ({ ...s, slot: slotOf(s.id) }));
    if (!series.length && !PRESETS.some(([, m]) => m === p.metric)) continue;
    if (!series.length) continue;
    const extra = { height: 230, vmarks: ['grad_norm', 'behaviour.close_ratio'].includes(p.metric) ? flagged(p.metric) : [], hlines: p.metric === 'behaviour.close_ratio' ? [{ y: 1, label: 'unchanged' }] : [] };
    grid.append(chartPanel({ key: `train:${p.metric}`, title: p.title, note: p.note, series, extra }));
  }
  v.append(grid);
  const flaggedSteps = ids.flatMap(id => [...(S.runs.get(id) || [])].filter(([, r]) => r.metrics?.grad_spike || r.metrics?.behaviour_shift || r.metrics?.reasoning_shift).map(([step, r]) => ({ id, step, r })));
  v.append(h('div', { class: 'card', style: 'margin-top:12px' }, h('header', {}, h('h2', {}, 'Training data of the selected runs'), h('span', { class: 'muted small' }, 'what each step selected and trained on')),
    h('div', { class: 'row' }, ids.map(id => h('a', { class: 'btn sm', href: dl.run(id) }, h('span', { class: 'swatch', style: `background:${slotVar(slotOf(id))}` }), id))),
    flaggedSteps.length ? h('div', { style: 'margin-top:8px' }, h('b', { class: 'small' }, 'Sentinel steps: '), flaggedSteps.slice(-20).map(f => h('a', { class: 'chip warn', style: 'margin:2px', href: dl.step('grpo:' + f.id.replace(/\./g, '/'), f.step), title: ['grad_spike', 'behaviour_shift', 'reasoning_shift'].filter(k => f.r.metrics[k]).join(', ') }, `${f.id.replace(/^grpo-/, '')} step ${f.step}`))) : null));
  const sel = h('select', { 'aria-label': 'Add a metric chart', onchange: e => { S.extra = e.target.value; store('ops.extra', S.extra); render(); } }, h('option', { value: '' }, 'Add a metric chart…'), names.map(n => h('option', { value: n, selected: n === S.extra }, n)));
  v.append(h('div', { class: 'toolbar', style: 'margin-top:12px' }, sel, h('span', { class: 'muted small' }, `${names.length} metrics retained per step`)));
  return v;
}

/* ---------- held-out ---------- */
function held() {
  const v = h('div', {}), runs = (S.snap?.runs || []).filter(r => (r.held || []).length);
  v.append(h('div', { class: 'notice' }, 'Held-out results are paired against the untouched base policy on the same 256 held-v3 tasks. The statistic is the mean per-task pass-rate change; p is a within-task permutation test, the interval a task bootstrap. Only these results count as training progress; dev16 and training reward do not.'));
  if (!runs.length) return v.append(h('div', { class: 'card muted' }, 'No held comparison has finished yet.')), v;
  const series = runs.map(r => ({ id: r.id, label: r.id, slot: slotOf(r.id), points: r.held.filter(x => num(x.delta_pts)).map(x => ({ x: x.step, y: x.delta_pts, lo: x.ci_lo != null ? 100 * x.ci_lo : null, hi: x.ci_hi != null ? 100 * x.ci_hi : null })).sort((a, b) => a.x - b.x), dots: true }));
  v.append(chartPanel({ key: 'held', title: 'Held-out change vs base (percentage points, 95% task bootstrap)', series, yFmt: x => signed(x, 1), extra: { height: 260, hlines: [{ y: 0, label: 'base' }], smooth: 0 }, note: 'Each point is one paired evaluation of a checkpoint (x = training step). Whiskers are the 95% interval; a whisker that crosses zero is not a significant change.' }));
  const rows = h('tbody', {});
  for (const r of runs) for (const x of [...r.held].sort((a, b) => (b.step ?? 0) - (a.step ?? 0))) {
    const sig = x.p != null && x.p < 0.05, cls = sig ? (x.delta_pts > 0 ? 'sig-good' : 'sig-bad') : '';
    rows.append(h('tr', { class: cls }, h('td', {}, h('span', { class: 'swatch', style: `background:${slotVar(slotOf(r.id))}` }), h('a', { href: dl.ds('grpo:' + r.id.replace(/\./g, '/'), { part: 'held' }) }, r.id)), h('td', { class: 'n' }, x.held_dir ? h('a', { href: dl.ds(`held-eval:${r.id.replace(/\./g, '/')}:${x.held_dir}`), title: 'Every held episode of this evaluation (view only)' }, x.step) : x.step), h('td', { class: 'n' }, h('b', {}, signed(x.delta_pts, 1)), ' pts'),
      h('td', { class: 'n' }, x.ci_lo != null ? `[${signed(100 * x.ci_lo, 1)}, ${signed(100 * x.ci_hi, 1)}]` : '—'), h('td', { class: 'n' }, fmt(x.p, 2)),
      h('td', { class: 'n' }, num(x.first_rate) ? `${(100 * x.first_rate).toFixed(1)}% → ${(100 * x.second_rate).toFixed(1)}%` : '—'), h('td', { class: 'n' }, x.held_dir ? [h('a', { href: dl.ds(`held-eval:${r.id.replace(/\./g, '/')}:${x.held_dir}`, { 'f.direction': 'improved' }), title: 'Tasks that passed more often than on the start policy' }, `${x.tasks_up ?? '—'} up`), ' / ',
        h('a', { href: dl.ds(`held-eval:${r.id.replace(/\./g, '/')}:${x.held_dir}`, { 'f.direction': 'regressed' }), title: 'Tasks that passed less often: open their episodes' }, `${x.tasks_down ?? '—'} down`)] : `${x.tasks_up ?? '—'} up / ${x.tasks_down ?? '—'} down`),
      h('td', { class: 'n' }, fmt(x.mcnemar_p, 2)), h('td', { class: 'n' }, num(x.loop_rate_delta) ? signed(100 * x.loop_rate_delta, 1, ' pts') : '—'), h('td', { class: 'n' }, `${x.ran ?? '?'}/${x.of ?? '?'}${x.complete ? '' : ' (partial)'}`), h('td', {}, sig ? h('span', { class: `chip ${x.delta_pts > 0 ? 'ok' : 'crit'}` }, x.delta_pts > 0 ? '✔ gain' : '✖ regression') : h('span', { class: 'chip' }, 'not significant'),
        x.held_dir && x.delta_pts < 0 ? h('a', { class: 'btn sm', style: 'margin-left:6px', href: dl.ds(`held-eval:${r.id.replace(/\./g, '/')}:${x.held_dir}`, { 'f.direction': 'regressed' }) }, 'Regressed episodes') : null)));
  }
  v.append(h('div', { class: 'card', style: 'margin-top:14px' }, h('header', {}, h('h2', {}, 'Every paired held evaluation')), h('div', { class: 'scroll', style: 'max-height:none' }, h('table', {}, h('thead', {}, h('tr', {}, ['run', 'step', 'change', '95% CI', 'p', 'pass rate base → policy', 'tasks', 'McNemar p', 'ran-out Δ', 'ran / of', 'verdict'].map((c, i) => h('th', { class: i > 0 && i < 10 ? 'n' : '' }, c)))), rows))));
  return v;
}

/* ---------- units ---------- */
function procCard(p) {
  const res = p.resources || {}, run = p.state === 'running';
  const card = h('div', { class: 'card proc' }, h('div', { class: 'phead row' }, h('h3', {}, p.title), machineChip(p.machine), h('span', { class: `chip ${run ? 'ok' : p.state === 'failed' ? 'crit' : ''}` }, `${run ? '●' : p.state === 'failed' ? '✖' : '○'} ${p.state}`), h('span', { class: 'muted small mono' }, shortUnit(p.unit))),
    h('div', {}, p.description), p.rationale ? h('div', { class: 'why' }, h('b', {}, 'Why: '), p.rationale) : null,
    progressView(p.progress, p.state));
  const facts = h('dl', {}, h('dt', {}, 'Elapsed'), h('dd', {}, dur(p.elapsed_s)), run ? [h('dt', {}, 'ETA'), h('dd', {}, etaText(p).replace(/^ETA /, '') || 'unknown')] : null, p.started_ts ? [h('dt', {}, 'Started'), h('dd', {}, dayClock(p.started_ts))] : null);
  card.append(facts);
  if (num(res.budget_gb)) card.append(h('div', { class: 'res' }, h('div', {}, h('b', {}, gb(res.used_gb)), h('span', {}, 'memory used')), h('div', {}, h('b', {}, gb(res.peak_gb)), h('span', {}, 'peak')), h('div', {}, h('b', {}, gb(res.budget_gb, res.budget_gb < 10 ? 1 : 0)), h('span', {}, 'ledger budget')), h('div', {}, h('b', {}, res.gpu ? 'GPU' : 'CPU'), h('span', {}, num(res.cpu_percent) ? `${res.cpu_percent}% CPU` : 'resource class'))));
  if (p.last_log) card.append(h('div', { class: 'log mono', title: p.last_log }, p.last_log));
  card.append(h('div', { class: 'row' }, askBtn('unit', p.unit, p.title), run ? h('button', { class: 'btn sm danger', type: 'button', onclick: () => openAsk({ mode: 'stop', subject: { kind: 'unit', id: p.unit, title: p.title } }) }, 'Request stop') : null,
    p.investigation ? h('button', { class: 'btn sm', onclick: () => go('investigations') }, `Investigation: ${p.investigation}`) : null, p.run ? runLink(p.run) : null));
  return card;
}
function units() {
  const v = h('div', {}), all = S.snap?.processes || [];
  const seg = (key, opts, set) => h('span', { class: 'seg-ctl' }, opts.map(([val, label]) => h('button', { type: 'button', 'aria-pressed': S[key] === val, onclick: () => { S[key] = val; render(); } }, label)));
  v.append(h('div', { class: 'toolbar' }, seg('procMachine', [['all', 'Both'], ['dgx', 'DGX'], ['pop', 'Pop']]), seg('procState', [['live', 'Running'], ['done', 'Failed / finished'], ['all', 'All']]), h('span', { class: 'muted small' }, 'Resources come from the memory ledger. Service plumbing is listed last.')));
  const keep = p => (S.procMachine === 'all' || p.machine === S.procMachine) && (S.procState === 'all' || (S.procState === 'live') === (p.state === 'running'));
  const rows = all.filter(keep).sort((a, b) => (a.kind === 'service') - (b.kind === 'service'));
  v.append(rows.length ? h('div', { class: 'grid g2' }, rows.map(procCard)) : h('div', { class: 'card muted' }, 'No matching units.'));
  v.append(section('Memory ledger claims', 'budget vs use, by machine'));
  for (const [name, m] of Object.entries(S.snap?.machines || {})) {
    const claims = m.ledger?.claim_list || [];
    const row = c => h('tr', {}, h('td', { class: 'mono' }, shortUnit(c.unit)), h('td', {}, c.class), h('td', {}, c.priority), h('td', { class: 'n' }, gb(c.used_gb)), h('td', { class: 'n' }, gb(c.peak_gb)), h('td', { class: 'n' }, gb(c.budget_gb, c.budget_gb < 10 ? 1 : 0)),
      h('td', { style: 'min-width:120px' }, c.budget_gb ? meter((c.used_gb || 0) / c.budget_gb) : ''));
    const head = h('tr', {}, ['unit', 'class', 'priority', 'used', 'peak', 'budget', 'use of budget'].map((c, i) => h('th', { class: i > 2 && i < 6 ? 'n' : '' }, c)));
    v.append(h('div', { class: 'card', style: 'margin-bottom:12px' },
      h('header', {}, h('h2', {}, name === 'dgx' ? 'DGX ledger' : 'Pop ledger'), h('span', { class: 'muted small' }, `${gb(m.ledger?.available_gb)} available · headroom ${gb(m.ledger?.headroom_gb)} · ${m.ledger?.waiting || 0} waiting`)),
      claims.length ? h('div', { class: 'scroll', style: 'max-height:none' }, h('table', {}, h('thead', {}, head), h('tbody', {}, claims.map(row)))) : h('div', { class: 'muted' }, 'No claims.')));
  }
  return v;
}

/* ---------- machines ---------- */
function machineHistory(name) {
  const hist = (S.hist.get(name) || []).filter(r => now() - r.ts < 24 * 3600e3);
  const xFmt = v => clock(v), mk = (key, title, yFmt, slot, extra = {}) => chartPanel({ key: `m:${name}:${key}`, title, series: [{ id: key, label: title, slot, points: hist.map(r => ({ x: r.ts, y: key === 'headroom' ? r.ledger?.headroom_gb : key === 'claims' ? r.ledger?.claims : r[key] })).filter(p => num(p.y)) }], yFmt, xKind: 'time', xFmt, xLabel: '', extra: { height: 170, gap: 600e3, smooth: 0, ...extra } });
  return h('div', { class: 'grid g3' }, mk('mem_used_gb', 'Memory used (GB)', v => fmt(v, 3), 0, { yMin: 0, yMax: S.snap?.machines?.[name]?.mem_total_gb }),
    mk('gpu_util_percent', 'GPU utilisation (%)', v => `${Math.round(v)}`, 2, { yMin: 0, yMax: 100 }), mk('cpu_percent', 'CPU (%)', v => `${Math.round(v)}`, 1, { yMin: 0, yMax: 100 }),
    ...(name === 'dgx' ? [mk('headroom', 'Ledger headroom (GB)', v => fmt(v, 3), 3, { hlines: [{ y: LEDGER_FLOOR_GB, label: 'floor 15 GB' }], yMin: 0 })] : []));
}
function machines() {
  const v = h('div', {}), m = S.snap?.machines || {};
  v.append(h('div', { class: 'notice' }, 'The GB10 shares one memory pool between CPU and GPU: "GPU memory" is system RAM, so nvidia-smi reports it as not available and the bars show ledger claims against total RAM. Ledger headroom is available memory minus what running claims still reserve.'));
  for (const name of ['dgx', 'pop']) {
    v.append(section(name === 'dgx' ? 'DGX' : 'Pop', '24 h history at 2-minute resolution'), machineCard(name, m[name], { detail: true }), h('div', { style: 'height:12px' }), machineHistory(name));
  }
  return v;
}

/* ---------- teachers ---------- */
function teachers() {
  const tt = S.snap?.teachers, v = h('div', {});
  if (!tt) return v.append(h('div', { class: 'card muted' }, 'No teacher pool configured.')), v;
  v.append(h('div', { class: 'grid g4' }, ...[[fmt(tt.tasks_done), `of ${fmt(tt.eligible)} eligible tasks done`], [fmt(tt.success), 'verified successes'], [fmt(tt.deferred), 'deferred (retry) rows'], [`${tt.families_with_success}/${tt.families_total}`, 'families with a success'], [`${fmt(tt.success_last_hour)}/h`, 'successes in the last hour']].map(([a, b]) => h('div', { class: 'card' }, stat(a, b)))));
  const colors = { task_success: 'var(--s3)', semantic_mismatch: 'var(--s4)', budget_exhaustion: 'var(--s8)', other: 'var(--muted-fill)' };
  const tds = tt.pool_rel ? `teacher-gen:${tt.pool_rel}` : null;
  v.append(section('Per teacher', tt.pool, tds ? h('a', { class: 'btn sm', href: dl.ds(tds) }, 'Browse the tasks and teacher episodes') : null));
  v.append(h('div', { class: 'grid g2' }, tt.teachers.map(te => {
    const parts = outcomeParts(te).map(p => ({ label: p.label, value: p.n, color: colors[p.key] }));
    return h('div', { class: 'card' }, h('header', {}, h('h2', { class: 'mono' }, te.id), te.success_rate != null ? h('span', { class: 'chip ok' }, `${Math.round(100 * te.success_rate)}% success`) : h('span', { class: 'chip' }, 'no episodes yet')),
      segBar(parts, { height: 16, label: `${te.id} outcomes` }), h('div', { class: 'legend', style: 'margin-top:6px' }, parts.filter(p => p.value).map(p => h('span', { class: 'key' }, h('i', { style: `background:${p.color};height:10px;width:10px` }), `${p.label} ${p.value}`))),
      h('div', { class: 'kv', style: 'margin-top:8px' }, h('span', {}, 'tasks / episodes'), h('span', {}, `${te.tasks} / ${te.episodes}`), h('span', {}, 'mean seconds per task'), h('span', {}, fmt(te.seconds_mean, 4)), h('span', {}, 'mean completion tokens'), h('span', {}, fmt(te.tokens_mean)),
        h('span', {}, 'deferred'), h('span', {}, Object.entries(te.deferred || {}).map(([k, n]) => `${n} ${k.replace(/_/g, ' ')}`).join(', ') || 'none')));
  })));
  const hist = tt.history || [];
  if (hist.length > 2) v.append(section('Progress over time'), chartPanel({ key: 'teach', title: 'Tasks done and verified successes', series: [{ id: 'done', label: 'tasks done', slot: 0, points: hist.map(r => ({ x: r[0], y: r[1] })) }, { id: 'success', label: 'successes', slot: 2, points: hist.map(r => ({ x: r[0], y: r[2] })) }], xKind: 'time', xFmt: clock, xLabel: '', yFmt: v => fmt(v, 4), extra: { smooth: 0, height: 220, gap: 3 * 3600e3 } }));
  v.append(section('Top families', `${tt.families_total} families seen`));
  v.append(h('div', { class: 'card' }, h('div', { class: 'scroll' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'family'), h('th', { class: 'n' }, 'tasks'), h('th', { class: 'n' }, 'successes'))), h('tbody', {}, tt.families.map(f => h('tr', {}, h('td', { class: 'mono' }, tds ? h('a', { href: dl.ds(tds, { 'f.family': f.family }) }, f.family) : f.family), h('td', { class: 'n' }, f.tasks), h('td', { class: 'n' }, f.success))))))));
  return v;
}

/* ---------- queue and fallow ---------- */
function backlog() {
  const v = h('div', {}), snap = S.snap || {};
  v.append(section('Queue', 'what is waiting and what it waits for'));
  v.append((snap.queue || []).length ? h('div', { class: 'grid g2' }, snap.queue.map(q => h('div', { class: 'card proc' }, h('div', { class: 'phead row' }, h('h3', {}, q.title), q.machine ? machineChip(q.machine) : null), h('div', {}, q.description), q.rationale ? h('div', { class: 'why' }, h('b', {}, 'Why: '), q.rationale) : null,
    h('div', { class: 'row' }, h('span', { class: 'chip warn' }, `▲ blocked on: ${q.blocked_on || '—'}`), num(q.eta_s) ? h('span', { class: 'chip' }, `ETA ${dur(q.eta_s)}`) : null, askBtn('queue', q.title, q.title))))) : h('div', { class: 'card muted' }, 'Nothing queued.'));
  v.append(section('Fallow ledger', 'built capabilities not in use (plans/neuralese/FALLOW_REGISTER.md)'));
  const rows = (snap.fallow || []).filter(r => matches(r, S.q));
  v.append(h('div', { class: 'toolbar' }, h('input', { type: 'search', placeholder: 'Filter the fallow ledger…', value: S.q, 'aria-label': 'Filter fallow rows', oninput: e => { S.q = e.target.value; S.hold = false; clearTimeout(S.qt); S.qt = setTimeout(render, 250); } }), h('span', { class: 'muted small' }, `${rows.length} of ${(snap.fallow || []).length} rows`)));
  let sec = null; const body = h('tbody', {});
  for (const r of rows) {
    if (r.section !== sec) { sec = r.section; body.append(h('tr', {}, h('th', { colspan: 5, style: 'background:var(--sunken)' }, sec))); }
    body.append(h('tr', {}, h('td', { class: 'mono' }, r.id), h('td', {}, r.piece), h('td', {}, r.code_now), h('td', {}, r.blocker), h('td', {}, r.next_action, ' ', askBtn('fallow', r.id, r.piece))));
  }
  v.append(h('div', { class: 'card' }, h('div', { class: 'scroll', style: 'max-height:none' }, h('table', {}, h('thead', {}, h('tr', {}, ['id', 'capability', 'code now', 'blocker', 'next action'].map(c => h('th', {}, c)))), body))));
  return v;
}

/* ---------- investigations ---------- */
function investigations() {
  const v = h('div', {}), invs = S.snap?.investigations || [];
  if (!invs.length) return v.append(h('div', { class: 'card muted' }, 'No open investigations.')), v;
  for (const i of invs) {
    const card = h('div', { class: 'card investigation', style: 'margin-bottom:14px' }, h('header', {}, h('h2', {}, i.title), h('span', { class: `chip ${i.status === 'open' ? 'open' : 'ok'}` }, i.status), ),
      h('p', { style: 'font-weight:600' }, i.question), h('p', {}, i.summary), h('div', { class: 'muted small' }, `updated ${ago(i.updated_ts, now())}`));
    card.append(h('details', { open: S.open.has(`f:${i.id}`) || undefined, ontoggle: e => e.target.open ? S.open.add(`f:${i.id}`) : S.open.delete(`f:${i.id}`) }, h('summary', {}, `${(i.findings || []).length} findings`),
      (i.findings || []).map(f => h('div', { class: 'finding' }, h('span', { class: `chip ${f.status}` }, f.status === 'ruled_out' ? '✔ ruled out' : f.status === 'open' ? '… open' : 'evidence'), h('span', {}, f.text)))));
    if ((i.next || []).length) card.append(h('div', { class: 'small', style: 'margin-top:6px' }, h('b', {}, 'Next: '), i.next.join(' · ')));
    const charts = (i.charts || []).map(c => ({ ...c, series: metricSeries(S.runs, c.runs || [], c.metric).map(s => ({ ...s, label: s.id.replace(/^grpo-v\d\.(hunt\.)?/, ''), slot: slotOf(s.id) })) })).filter(c => c.series.length);
    if (charts.length) card.append(h('div', { class: 'grid g2', style: 'margin-top:10px' }, charts.map(c => chartPanel({ key: `inv:${i.id}:${c.title}`, title: c.title, series: c.series, extra: { height: 200 } }))));
    card.append(h('div', { class: 'row', style: 'margin-top:8px' }, askBtn('investigation', i.id, i.title), ...[...new Set([...(i.runs || []), ...(i.charts || []).flatMap(c => c.runs || [])])].slice(0, 6).map(r => h('a', { class: 'btn sm', href: dl.run(r) }, `data: ${r.replace(/^grpo-v\d\.(hunt\.)?/, '')}`))));
    v.append(card);
  }
  return v;
}

/* ---------- shell ---------- */
function data() {
  if (!src) return h('div', { class: 'card muted' }, 'Connect to read training data.');
  return dataView({ src, route: S.route, render: softRender, ui: S.dui });
}
function inbox() {
  const ib = src?.inbox.get() || { asks: [], answers: [] }, v = h('div', {});
  v.append(h('div', { class: 'notice' }, 'Questions and requests you sent from this page, with the supervising agent\'s answers. Each is a signed operation in the application journal; the native participant mirrors them to ', h('span', { class: 'mono' }, 'ops/dashboard/inbox.jsonl'), ' for the agent.'));
  if (!ib.asks.length) return v.append(h('div', { class: 'card muted' }, 'Nothing asked yet. Every alert, unit, queue item, investigation and fallow row has an Ask button.')), v;
  for (const a of [...ib.asks].reverse()) {
    const ans = ib.answers.filter(x => x.ask === editRef(a.operationId));
    v.append(h('div', { class: 'card', style: 'margin-bottom:10px' }, h('header', {}, h('h2', {}, a.subject.title || a.subject.id), h('span', { class: `chip ${a.kind === 'request_stop' ? 'crit' : a.kind === 'ack_alert' ? '' : 'evidence'}` }, a.kind.replace('_', ' '))),
      h('div', {}, a.text), ans.length ? ans.map(x => h('div', { class: 'answer' }, h('b', {}, 'Agent: '), x.text)) : h('div', { class: 'muted small' }, a.kind === 'ack_alert' ? '' : 'Not answered yet.')));
  }
  return v;
}
const VIEWS = { overview, training, data, held, units, machines, teachers, backlog, investigations, inbox };
function go(tab, route = '') { const hash = `#${tab}${route ? '/' + route : ''}`; if (location.hash !== hash) location.hash = hash; else { S.tab = tab; S.route = route; render(); } }
function tabs(hv) {
  const nav = $('tabs'); nav.replaceChildren();
  for (const [id, label] of TABS) nav.append(h('button', { class: 'tab', role: 'tab', type: 'button', 'aria-selected': S.tab === id, onclick: () => go(id) }, label, id === 'overview' && hv.open.length ? h('span', { class: 'n', title: 'open alerts' }, hv.open.length) : id === 'inbox' && unanswered() ? h('span', { class: 'n info', title: 'unanswered questions' }, unanswered()) : null));
}
/** Re-render without stealing an input the person is typing in: deferred until focus leaves it. */
let softPending = false;
function softRender() {
  const a = document.activeElement;
  if (a && $('view').contains(a) && a.matches?.('textarea, input[type=search], input[type=text], select')) { softPending = true; return; }
  render();
}
$('view').addEventListener('focusout', () => { if (softPending) { softPending = false; setTimeout(softRender, 0); } });
function unanswered() { const ib = src?.inbox.get(); return ib ? ib.asks.filter(a => a.kind !== 'ack_alert' && !ib.answers.some(x => x.ask === editRef(a.operationId))).length : 0; }
function chrome(hv) {
  const pill = $('health'), ic = pill.querySelector('.ic');
  pill.hidden = !src;
  pill.className = `pill ${hv.level}`; ic.textContent = hv.level === 'ok' ? '✔' : hv.level === 'crit' ? '✖' : '▲'; $('health-text').textContent = hv.title;
  pill.onclick = () => go('overview');
  freshness();
}
function freshness() {
  const el = $('fresh'), age = S.snap ? now() - S.snap.ts : null;
  el.className = `live ${age == null ? 'dead' : age < 90e3 ? 'fresh' : age < SNAP_STALE_MS * 2 ? 'stale' : 'dead'}`;
  $('fresh-text').textContent = !src ? 'not connected' : age == null ? 'connected, waiting for data' : `data ${dur(age / 1000)} old`;
}
function render() {
  const hv = health(S.snap, now(), S.acked);
  document.title = `${hv.level === 'ok' ? '✔' : hv.level === 'crit' ? '✖' : '▲'} natlang ops`;
  chrome(hv); tabs(hv);
  $('tabs').hidden = !!S.gate;
  document.documentElement.style.setProperty('--hdr', `${document.querySelector('.top').offsetHeight}px`);
  const view = $('view'), keepScroll = window.scrollY;
  const detKey = (() => { const seen = new Map(); return d => { const t = (d.querySelector(':scope > summary')?.textContent || '').slice(0, 80); const n = (seen.get(t) || 0) + 1; seen.set(t, n); return `${t}#${n}`; }; });
  const openState = new Map(), k1 = detKey();
  view.querySelectorAll('details').forEach(d => openState.set(k1(d), d.open));
  const next = h('div', {});
  const banner = S.gate || !src ? null : S.tab === 'data' ? (hv.open.length ? h('div', { class: 'alertline' }, h('span', { class: `tone-${hv.level}` }, levelIcon(hv.level), ` ${hv.open.length} open alert${hv.open.length > 1 ? 's' : ''}: `), hv.open[0].text, ' ', h('a', { href: '#overview' }, 'Overview')) : null) : alertsView(hv);
  if (banner) next.append(banner);
  if (S.gate) { next.append(S.gate); view.replaceChildren(next); return; }
  if (!S.snap && S.tab !== 'data') next.append(h('div', { class: 'notice' }, 'Connected. The first snapshot comes from the DGX within about 30 seconds. If nothing arrives in a minute, the exporter or the native ops participant on the DGX is not running.'));
  try { next.append((VIEWS[S.tab] || overview)(hv)); } catch (e) { next.append(h('div', { class: 'alert crit' }, `This section failed to draw: ${e.message}`)); console.error(e); }
  view.replaceChildren(next);
  if (restoreTo != null && document.documentElement.scrollHeight > restoreTo + innerHeight * 0.5) { window.scrollTo({ top: restoreTo }); restoreTo = null; }
  const k2 = detKey();
  next.querySelectorAll('details').forEach(d => { const k = k2(d); if (openState.has(k)) d.open = openState.get(k); });
  if (restoreTo == null) window.scrollTo({ top: keepScroll });
}

/* ---------- boot ---------- */
$('theme').onclick = () => {
  const cur = document.documentElement.dataset.theme || 'auto', nextTheme = cur === 'auto' ? 'light' : cur === 'light' ? 'dark' : 'auto';
  nextTheme === 'auto' ? delete document.documentElement.dataset.theme : (document.documentElement.dataset.theme = nextTheme);
  store('ops.theme', nextTheme); $('theme').textContent = nextTheme[0].toUpperCase() + nextTheme.slice(1);
};
$('theme').textContent = ({ light: 'Light', dark: 'Dark' })[store('ops.theme')] || 'Auto';
const view0 = $('view');
view0.addEventListener('pointerover', e => { S.hold = !!e.target.closest?.('.plot'); });
view0.addEventListener('focusin', e => { S.hold = e.target.matches?.('.plot, textarea, input, select'); });
view0.addEventListener('focusout', () => { S.hold = false; });
view0.addEventListener('pointerleave', () => { S.hold = false; });
function fromHash() {
  const [tab, ...rest] = location.hash.slice(1).split('/');
  if (TABS.some(([id]) => id === tab)) { S.tab = tab; S.route = rest.join('/'); }
}
fromHash();
/* Back/forward restore the scroll position a route was left at; a link to a new route starts at the top; a change of
   query only (filters, tabs, sort) keeps the position. Positions are restored once the route's data has drawn. */
const scrollAt = new Map();
let lastHash = location.hash, forward = false, restoreTo = null;
document.addEventListener('click', e => { if (e.target.closest?.('a[href^="#"]')) forward = true; }, true);
addEventListener('scroll', () => { scrollAt.set(location.hash, window.scrollY); }, { passive: true });
addEventListener('hashchange', () => {
  const path = h0 => h0.split('?')[0], samePath = path(location.hash) === path(lastHash);
  lastHash = location.hash; fromHash(); render();
  if (samePath) return;
  if (!forward && scrollAt.has(location.hash)) { restoreTo = scrollAt.get(location.hash); window.scrollTo({ top: restoreTo }); } else { restoreTo = null; window.scrollTo({ top: 0 }); }
  forward = false;
});
/* Keyboard: / search · j/k move through rows or calls · Enter opens the row · [ ] previous/next sample · Esc leaves a field · ? help */
addEventListener('keydown', e => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const inField = e.target.matches?.('input, textarea, select, [contenteditable]');
  if (e.key === 'Escape' && inField) { e.target.blur(); return; }
  if (inField) return;
  if (e.key === '/') { const box = document.querySelector('.catside input[type=search], .datatools input[type=search]'); if (box) { e.preventDefault(); box.focus(); box.select(); } return; }
  if (e.key === '?') { e.preventDefault(); toggleHelp(); return; }
  if (S.tab !== 'data') return;
  const calls = [...document.querySelectorAll('section.call[id^="call-"]')];
  if ((e.key === 'j' || e.key === 'k') && calls.length) {
    const cur = document.querySelector('.callrail a.current')?.dataset.call;
    const i = Math.max(0, Math.min(calls.length - 1, (cur == null ? -1 : Number(cur)) + (e.key === 'j' ? 1 : -1)));
    calls[i].scrollIntoView({ block: 'start' }); document.querySelectorAll('.callrail a').forEach(x => x.classList.toggle('current', x.dataset.call === String(i))); return;
  }
  const rows = [...document.querySelectorAll('.kbrow')];
  if ((e.key === 'j' || e.key === 'k') && rows.length) {
    e.preventDefault();
    let i = rows.findIndex(r => r.classList.contains('kbsel'));
    i = i < 0 ? 0 : Math.max(0, Math.min(rows.length - 1, i + (e.key === 'j' ? 1 : -1)));
    rows.forEach((r, k) => r.classList.toggle('kbsel', k === i)); rows[i].scrollIntoView({ block: 'nearest' }); return;
  }
  if (e.key === 'Enter') { const r = document.querySelector('.kbrow.kbsel'); if (r?.dataset.href) { forward = true; location.hash = r.dataset.href; } return; }
  if (e.key === '[' || e.key === ']') {
    const pills = [...document.querySelectorAll('.samplenav .sbtn')], i = pills.findIndex(x => x.classList.contains('on'));
    const t = pills[i + (e.key === ']' ? 1 : -1)]; if (t) { forward = true; location.hash = t.getAttribute('href'); }
  }
});
function toggleHelp() {
  const old = document.getElementById('kbhelp'); if (old) { old.remove(); return; }
  const k = x => h('kbd', {}, x);
  document.body.append(h('div', { id: 'kbhelp', class: 'card kbhelp', role: 'dialog', 'aria-label': 'Keyboard shortcuts', onclick: e => { if (e.target.id === 'kbhelp' || e.target.tagName === 'BUTTON') toggleHelp(); } },
    h('h3', {}, 'Keyboard'), h('dl', { class: 'kvgrid' }, [[k('/'), 'search or filter'], [[k('j'), k('k')], 'next / previous row, or call in an episode'], [k('Enter'), 'open the selected row'],
      [[k('['), k(']')], 'previous / next sample of a group'], [k('Esc'), 'leave a text field'], [k('?'), 'this help']].map(([a1, b]) => [h('dt', {}, a1), h('dd', {}, b)])), h('button', { class: 'btn sm' }, 'Close')));
}
setInterval(freshness, 5000);
setInterval(() => { if (!S.hold && S.tab !== 'data') render(); }, 60000);

/** Start the dashboard on a connected source (session or preview). Before this, `gate` (the connection panel) is shown. */
export function start(source) {
  src = source; S.gate = null;
  if (src.kind === 'preview') document.querySelector('.top .brand')?.append(h('span', { class: 'chip warn previewchip', title: 'A local preview of the page over the DGX exporter files and data service. Edits made here stay in this preview server and are not applied.' }, 'local preview'));
  src.ticks.on(t => { if (t) void pull(t); });
  for (const c of [src.edits, src.inbox, src.pending]) c.on(() => { if (S.tab === 'data' || S.tab === 'inbox') softRender(); else schedule(); });
  void pull(null);
  setInterval(() => void pull(src.ticks.get()), 60000);      // ticks drive refreshes; this is the fallback
  render();
}
export function showGate(node) { S.gate = node; render(); }
render();
