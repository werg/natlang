import { num, fmt, gb, signed, dur, ago, clock, dayClock, shortUnit, health, metricSeries, SNAP_STALE_MS } from './model.mjs';
import { h, lineChart, valueTable, slotVar } from './charts.mjs';
import { makeViews } from './views.mjs';
import { dataView, link as dl } from './data.mjs';
import { editRef } from '../ops-protocol.mjs';
import { BUILD } from './connect.mjs';

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
const PRIMARY = [['overview', 'Now', '◉'], ['training', 'Rounds', '↻'], ['flow', 'Data', '≋'], ['held', 'Held-out', '✓'], ['machines', 'Machines', '▣'], ['more', 'More', '⋯']];
const SECONDARY = [['units', 'Units'], ['teachers', 'Teachers'], ['data', 'Training data'], ['backlog', 'Open items'], ['investigations', 'Investigations'], ['inbox', 'Inbox'], ['events', 'Events']];
const TABS = [...PRIMARY, ...SECONDARY];

/* ---------- data intake ---------- */
let paintTimer = null, lastPaint = 0;
function schedule() {
  if (paintTimer) return;
  paintTimer = setTimeout(() => { paintTimer = null; if (S.hold) { schedule(); return; } lastPaint = now(); if (S.tab === 'data') { freshness(); return; } render(); }, Math.max(250, 2000 - (now() - lastPaint)));
}
function notice(key, text) { text ? S.notices.set(key, text) : S.notices.delete(key); $('notices').textContent = [...S.notices.values()].join(' · '); }

const report = e => { try { src?.report?.(e); } catch {} };
function takeSnapshot(rec) {
  if (!rec?.data || rec.data.v !== 3) { report({ ev: 'snapshot_rejected', why: !rec?.data ? 'empty' : `version ${rec.data.v}` }); return; }
  if (S.snap && rec.data.ts < S.snap.ts) return;
  if (!S.snap) report({ ev: 'snapshot_first', age_s: Math.round((Date.now() - rec.data.ts) / 1000), runs: (rec.data.runs || []).length, machines: Object.keys(rec.data.machines || {}).length });
  S.snap = rec.data; S.snapCached = false;
  if (now() - (S.cachedAt || 0) > 120e3) {          // the last snapshot paints a reopened page at once (marked with its age)
    S.cachedAt = now();
    try { const text = JSON.stringify(rec.data); if (text.length < 3e6) localStorage.setItem('ops.snap-cache', text); } catch {}
  }
  schedule();
}
function takeMetric(rec) {
  const d = rec.data;
  if (!d?.run_id || !num(d.step)) return;
  if (rec.type === 'train') {
    if (!S.runs.has(d.run_id)) S.runs.set(d.run_id, new Map());
    S.runs.get(d.run_id).set(d.step, { ...d, ts: rec.ts });
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
    const streams = S.tab === 'investigations' ? [['metrics', takeMetric]] : [];      // the snapshot carries everything else; the metric stream only feeds the old investigation charts
    for (const [key, fn] of streams) {
      if (offsets[key] && tick?.[key] === seen[key]) continue;
      const r = await src.query(`ops.${key}`, { offset: offsets[key] });
      for (const rec of r.records) fn(rec);
      offsets[key] = r.offset; seen[key] = tick?.[key];
    }
  } catch (e) { notice('snap', `Could not read the ops state: ${e.message}`); report({ ev: 'pull_failed', error: String(e.message).slice(0, 200) }); }
  finally { pulling = false; }
}

/* ---------- colours: a run keeps its colour slot for good (colour follows the entity) ---------- */
function slotOf(id) {
  if (S.slots[id] == null) { S.slots[id] = Object.keys(S.slots).length % 8; store('ops.slots', S.slots); }
  return S.slots[id];
}

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
function chartPanel({ key, title, note, series, yFmt, extra = {}, xLabel = 'step', xFmt, xKind }) {
  const ch = lineChart({ series, smooth: extra.smooth ?? 0, logy: false, yLabel: title, xLabel, xFmt, xKind, yFmt, ...extra });
  const showTable = S.tables.has(key);
  return h('div', { class: 'card chartcard' }, h('header', {}, h('h2', {}, title), h('button', { class: 'btn sm', type: 'button', 'aria-pressed': showTable, onclick: () => { showTable ? S.tables.delete(key) : S.tables.add(key); render(); } }, showTable ? 'Chart' : 'Table')),
    showTable ? valueTable(series, { xLabel, xFmt: xFmt || (v => v), yFmt }) : ch.el, note ? h('div', { class: 'panel-note' }, note) : null);
}

/* ---------- alerts ---------- */
function alertsView(hv) {
  const open = hv.items.filter(i => !S.acked[i.id]), acked = hv.items.filter(i => S.acked[i.id]);
  if (!open.length && !acked.length) return null;
  const rank = { crit: 0, warn: 1 }, sorted = [...open].sort((a, b) => (rank[a.level] ?? 2) - (rank[b.level] ?? 2));
  const row = a => h('div', { class: `arow ${a.level}` }, h('span', { class: 'ic', 'aria-hidden': 'true' }, levelIcon(a.level)),
    h('div', { class: 'what' }, h('span', { class: 'sr' }, a.level === 'crit' ? 'Critical: ' : 'Warning: '), a.text, ' ', a.machine ? machineChip(a.machine) : null),
    h('div', { class: 'acts' }, askBtn('alert', a.id, a.text), a.system ? null : h('button', { class: 'btn sm', type: 'button', onclick: () => ackAlert(a.id) }, 'Dismiss')));
  const box = h('section', { class: 'card attention', 'aria-label': 'Alerts' });
  if (open.length) {
    const first = sorted.slice(0, 3), rest = sorted.slice(3);
    box.append(h('header', {}, h('h2', {}, 'Needs attention'), h('span', { class: `chip ${sorted[0].level === 'crit' ? 'crit' : 'warn'}` }, `${open.length}`)), ...first.map(row));
    if (rest.length) box.append(h('details', { class: 'morealerts' }, h('summary', {}, `${rest.length} more`), ...rest.map(row)));
  }
  if (acked.length) box.append(h('div', { class: 'small muted' }, `${acked.length} dismissed `, h('button', { class: 'btn sm', type: 'button', onclick: () => { S.showAcked = !S.showAcked; render(); } }, S.showAcked ? 'Hide' : 'Show'),
    S.showAcked ? h('ul', {}, acked.map(a => h('li', {}, a.text))) : null));
  return box;
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
const V = makeViews({ S, go: (...a) => go(...a), askBtn, alertsView, now, chartPanel, machineChip, meter, section, stat, runLink, openAsk, render: () => render() });
const VIEWS = { overview: V.overview, training: V.training, flow: V.flow, held: V.held, machines: V.machines, units: V.units, teachers: V.teachers, backlog: V.backlog, more: V.more, events: V.events, data, investigations, inbox };
function go(tab, route = '') { const hash = `#${tab}${route ? '/' + route : ''}`; if (location.hash !== hash) location.hash = hash; else { S.tab = tab; S.route = route; render(); } }
function tabs(hv) {
  const nav = $('tabs'); nav.replaceChildren();
  const inMore = SECONDARY.some(([id]) => id === S.tab) || S.tab === 'more';
  const badge = id => id === 'overview' && hv.open.length ? h('span', { class: 'n', title: 'open alerts' }, hv.open.length) : id === 'more' && unanswered() ? h('span', { class: 'n info', title: 'unanswered questions' }, unanswered()) : null;
  for (const [id, label, icon] of PRIMARY) nav.append(h('button', { class: `tab${id === 'more' ? ' moretab' : ''}`, role: 'tab', type: 'button', 'aria-selected': id === 'more' ? inMore : S.tab === id, onclick: () => go(id) }, h('span', { class: 'ico', 'aria-hidden': 'true' }, icon), h('span', { class: 'lbl' }, label), badge(id)));
  for (const [id, label] of SECONDARY) nav.append(h('button', { class: 'tab sec', role: 'tab', type: 'button', 'aria-selected': S.tab === id, onclick: () => go(id) }, label, id === 'inbox' && unanswered() ? h('span', { class: 'n info' }, unanswered()) : null));
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
  pill.className = `pill ${hv.level}`; ic.textContent = hv.level === 'ok' ? '✔' : hv.level === 'crit' ? '✖' : '▲'; $('health-text').textContent = hv.level === 'ok' ? 'OK' : `${hv.open.length}`; pill.title = hv.title;
  pill.onclick = () => go('overview');
  freshness();
}
function releaseLine() {
  const el = document.getElementById('release') || document.querySelector('.foot')?.appendChild(h('span', { id: 'release', class: 'release' }));
  if (el) el.textContent = `release ${window.tendril?.identity?.version ?? '–'} · build ${BUILD.build.replace('T', ' ').replace('Z', ' UTC')}${BUILD.commit ? ' · ' + BUILD.commit : ''}`;
}
function freshness() {
  releaseLine();
  const el = $('fresh'), age = S.snap ? now() - S.snap.ts : null;
  el.className = `live ${age == null ? 'dead' : age < 90e3 ? 'fresh' : age < SNAP_STALE_MS * 2 ? 'stale' : 'dead'}`;
  $('fresh-text').textContent = S.snapCached ? `saved ${dur(age / 1000)} · ${src ? 'updating' : 'connecting'}`
    : !src ? 'connecting' : age == null ? 'waiting for data' : age < 90e3 ? 'live' : `${dur(age / 1000)} old`;
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
  const banner = S.gate || !src || S.tab === 'overview' ? null : (hv.open.length ? h('div', { class: 'alertline' }, h('span', { class: `tone-${hv.level}` }, levelIcon(hv.level), ` ${hv.open.length} open alert${hv.open.length > 1 ? 's' : ''}: `), hv.open[0].text, ' ', h('a', { href: '#overview' }, 'Now')) : null);
  if (banner) next.append(banner);
  if (S.gate) { next.append(S.gate); view.replaceChildren(next); return; }
  if (!S.snap && S.tab !== 'data' && src) next.append(h('div', { class: 'notice' }, 'Connected. The first snapshot comes from the DGX within about 30 seconds. If nothing arrives in a minute, the exporter or the native ops participant on the DGX is not running.'));
  const sec = SECONDARY.find(([id]) => id === S.tab);
  if (sec && S.tab !== 'data') next.append(h('div', { class: 'pagehead' }, h('a', { href: '#more' }, '‹ More'), h('b', {}, sec[1])));
  try { next.append((VIEWS[S.tab] || VIEWS.overview)(hv)); } catch (e) { next.append(h('div', { class: 'alert crit' }, `This section failed to draw: ${e.message}`)); console.error(e); report({ ev: 'render_failed', tab: S.tab, error: String(e.message).slice(0, 200), stack: String(e.stack || '').slice(0, 300) }); }
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
  addEventListener('error', e => report({ ev: 'page_error', error: String(e.message).slice(0, 200), at: `${e.filename?.split('/').pop()}:${e.lineno}` }));
  addEventListener('unhandledrejection', e => report({ ev: 'page_rejection', error: String(e.reason?.message || e.reason).slice(0, 200) }));
  document.addEventListener('visibilitychange', () => report({ ev: 'visibility', state: document.visibilityState }));
  // A tab stays pinned to the publication it opened (Tendriller keeps the pin on reload). When the participant announces
  // a newer build, move to the selected release once (the page's base URL); a second announcement of the same build is ignored.
  src.ticks.on(t => {
    const b = t?.page_build;
    if (!b || !BUILD.build || BUILD.build === 'dev' || b <= BUILD.build || sessionStorage.getItem('ops.moved-to') === b) return;
    sessionStorage.setItem('ops.moved-to', b); report({ ev: 'moving_to_latest', from: BUILD.build, to: b });
    setTimeout(() => { location.href = window.tendril?.identity?.base || location.href.split('?')[0]; }, 1500);
  });
  if (src.kind === 'preview') document.querySelector('.top .brand')?.append(h('span', { class: 'chip warn previewchip', title: 'A local preview of the page over the DGX exporter files and data service. Edits made here stay in this preview server and are not applied.' }, 'local preview'));
  src.ticks.on(t => { if (t) void pull(t); });
  // a phone resumes a backgrounded page: read the current state at once rather than at the next tick
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') void pull(src.ticks.get()); });
  for (const c of [src.edits, src.inbox, src.pending]) c.on(() => { if (S.tab === 'data' || S.tab === 'inbox') softRender(); else schedule(); });
  void pull(null);
  setInterval(() => void pull(src.ticks.get()), 60000);      // ticks drive refreshes; this is the fallback
  render();
}
export function showGate(node) { S.gate = node; render(); }
{ const saved = store('ops.snap-cache'); if (saved?.v === 3 && !S.snap) { S.snap = saved; S.snapCached = true; } }
render();
