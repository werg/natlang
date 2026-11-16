import { num, fmt, gb, pct, signed, dur, ago, clock, dayClock, health, improving, latestHeld, metricSeries, metricNames, memorySegments, outcomeParts, matches, trend } from './model.mjs';

const t = window.tendril;
const $ = id => document.getElementById(id);
const store = (k, v) => { try { if (v === undefined) return JSON.parse(localStorage.getItem(k)); localStorage.setItem(k, JSON.stringify(v)); } catch { return null; } };

/* ---------- state ---------- */
const S = {
  snap: null, live: new Map(), hist: new Map(), runs: new Map(), events: [], notices: new Map(),
  acked: store('ops.acked') || {}, tab: 'now', open: new Set(), version: 0, sel: new Set(), slot: new Map(),
  ema: store('ops.ema') ?? 0.3, logy: false, preset: store('ops.preset') || 'outcome', custom: store('ops.custom') || 'phase_s.wait_groups',
  q: '', evFilter: 'all',
};
const SLOTS = 6;

/* ---------- dom helpers ---------- */
function h(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (k === 'style') n.setAttribute('style', v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) n.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return n;
}
const svg = (tag, attrs = {}, ...kids) => {
  const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  n.append(...kids);
  return n;
};
const chip = (text, cls = '') => h('span', { class: `chip ${cls}` }, text);
const now = () => Date.now();

function spark(values, { floor, width = 160, height = 34 } = {}) {
  const v = values.filter(num);
  if (v.length < 2) return h('div', { class: 'muted', style: `height:${height}px;font-size:12px` }, v.length ? 'one value so far' : 'no data yet');
  let lo = Math.min(...v, floor ?? Infinity), hi = Math.max(...v, floor ?? -Infinity);
  if (hi === lo) { hi += 1; lo -= 1; }
  const x = i => 3 + (i / (v.length - 1)) * (width - 6), y = val => height - 4 - ((val - lo) / (hi - lo)) * (height - 8);
  const root = svg('svg', { class: 'spark', viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: 'none', role: 'img', 'aria-label': `trend of ${v.length} values, latest ${fmt(v.at(-1))}` });
  if (floor != null) root.append(svg('line', { class: 'floor', x1: 0, x2: width, y1: y(floor), y2: y(floor) }));
  root.append(svg('path', { d: v.map((val, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(val).toFixed(1)}`).join(''), 'vector-effect': 'non-scaling-stroke' }));
  root.append(svg('circle', { cx: x(v.length - 1), cy: y(v.at(-1)), r: 3 }));
  return root;
}

function bar(parts, { total, thin, mark, label } = {}) {
  const sum = total ?? parts.reduce((a, p) => a + p.value, 0);
  const el = h('div', { class: `bar${thin ? ' thin' : ''}`, role: 'img', 'aria-label': label || parts.map(p => `${p.label} ${fmt(p.value)}`).join(', ') });
  for (const p of parts) if (p.value > 0) el.append(h('i', { class: p.cls, style: `width:${(100 * p.value / sum).toFixed(2)}%${p.color ? `;background:${p.color}` : ''}`, title: `${p.label}: ${fmt(p.value)}` }));
  if (mark != null) el.append(h('span', { class: 'mark', style: `left:${(100 * mark / sum).toFixed(2)}%`, title: 'floor' }));
  return el;
}

/* ---------- data intake ---------- */
const paint = window.tendrilSnippets.streamView.batch(() => render());
const schedule = () => paint.schedule();
function notice(key, text) { text ? S.notices.set(key, text) : S.notices.delete(key); $('notices').textContent = [...S.notices.values()].join(' · '); }

function takeSnapshot(rec) {
  if (!rec?.data || rec.data.v !== 2) return;
  if (S.snap && rec.data.ts < S.snap.ts) return;
  S.snap = rec.data; S.version++;
  if (!S.sel.size) defaultSelection();
  schedule();
}
function takeMetric(rec) {
  const d = rec.data;
  if (rec.type !== 'train' || !d?.run_id || !num(d.step)) return;
  if (!S.runs.has(d.run_id)) S.runs.set(d.run_id, new Map());
  S.runs.get(d.run_id).set(d.step, { ...d, ts: rec.ts });
  S.version++; schedule();
}
function takeMachine(rec) {
  const d = rec.data; if (!d?.machine) return;
  const list = S.hist.get(d.machine) || S.hist.set(d.machine, []).get(d.machine);
  list.push({ ts: rec.ts, ...d }); if (list.length > 1500) list.shift();
  schedule();
}
function takeEvent(rec) { S.events.push({ ts: rec.ts, ...rec.data }); if (S.events.length > 400) S.events.shift(); schedule(); }

const wanted = ['ops.snapshot', 'machine.live', 'ops.machines', 'training.metrics', 'ops.events'];
const subscribed = new Set(); let initializing = false, again = false;
async function init() {
  if (subscribed.size === wanted.length) return;
  if (initializing) { again = true; return; }
  initializing = true;
  try {
    const { streams } = await t.streams.list();
    for (const name of wanted) {
      if (subscribed.has(name)) continue;
      const info = streams.find(s => s.name === name);
      if (!info) { notice(name, `Stream ${name} is not provisioned yet.`); continue; }
      notice(name, '');
      const onGap = hist => notice(`gap:${name}`, `${name}: history before #${hist.missing_through + 1} is no longer retained.`);
      const onError = e => notice(`err:${name}`, `${name}: ${e.message || 'unavailable'}`);
      if (name === 'ops.snapshot') {
        const latest = await t.streams.latest(name);
        if (latest.record) takeSnapshot(latest.record);
        t.streams.subscribe(name, takeSnapshot, { after: latest.cursor ?? 0, onGap, onError });
      } else if (name === 'machine.live') {
        t.streams.subscribe(name, rec => { S.live.set(rec.data?.source, rec); schedule(); }, { onGap, onError });
      } else {
        const fn = { 'ops.machines': takeMachine, 'training.metrics': takeMetric, 'ops.events': takeEvent }[name];
        t.streams.subscribe(name, fn, { after: 0, onGap, onError });
      }
      subscribed.add(name);
    }
  } catch (e) { notice('init', e.message || 'Could not list streams.'); }
  finally { initializing = false; if (again) { again = false; void init(); } }
}

/* ---------- selection of runs for the training charts ---------- */
function defaultSelection() {
  const rows = (S.snap?.runs || []).filter(r => r.kind === 'grpo');
  for (const r of rows.slice(0, 3)) select(r.id, true);
}
function select(id, on) {
  if (on) {
    if (S.sel.size >= SLOTS || S.sel.has(id)) return false;
    const used = new Set(S.slot.values());
    S.slot.set(id, [...Array(SLOTS).keys()].find(i => !used.has(i)));
    S.sel.add(id);
  } else { S.sel.delete(id); S.slot.delete(id); }
  return true;
}

/* ---------- ask / request dialog ---------- */
const dlg = $('ask');
let askCtx = null;
function openAsk(ctx) {
  askCtx = ctx;
  $('ask-title').textContent = ctx.mode === 'stop' ? 'Request to stop a process' : 'Ask the supervising agent';
  $('ask-sub').textContent = ctx.title;
  $('ask-label').firstChild.textContent = ctx.mode === 'stop' ? 'Why should it stop?' : 'Your question or request';
  $('ask-note').textContent = ctx.mode === 'stop' ? 'This sends a request. Nothing is stopped by this page; the agent decides.' : 'The agent sees your question with this item attached and answers in the conversation.';
  $('ask-send').textContent = ctx.mode === 'stop' ? 'Send request' : 'Send';
  $('ask-status').textContent = ''; $('ask-text').value = '';
  dlg.showModal(); $('ask-text').focus();
}
$('ask-cancel').addEventListener('click', () => dlg.close());
$('ask-form').addEventListener('submit', async e => {
  e.preventDefault();
  const text = $('ask-text').value.trim(); if (!text || !askCtx) return;
  $('ask-send').disabled = true; $('ask-status').textContent = 'Sending…';
  try {
    const type = askCtx.mode === 'stop' ? 'ops.request_stop' : 'ops.ask';
    const r = await t.send({ type, subject: { kind: askCtx.kind, id: askCtx.id, title: askCtx.title }, text });
    if (r?.status === 'rejected') throw new Error(r.message || 'rejected');
    $('ask-status').textContent = r?.status === 'queued' ? 'Saved offline; sent on reconnect.' : 'Sent.';
    setTimeout(() => dlg.close(), 700);
  } catch (err) { $('ask-status').textContent = `Not sent: ${err.message || err}`; }
  finally { $('ask-send').disabled = false; }
});
const askBtn = (kind, id, title, label = 'Ask', mode = 'ask') => h('button', { class: `btn small${mode === 'stop' ? ' danger' : ''}`, type: 'button', onclick: () => openAsk({ kind, id, title, mode }) }, label);

async function ackAlert(a) {
  S.acked[a.id] = now(); store('ops.acked', S.acked); schedule();
  try { await t.send({ type: 'ops.ack_alert', alert_id: a.id, text: a.text }, { wake: false }); } catch { /* local acknowledgement still applies */ }
}

/* ---------- shared pieces ---------- */
const machineName = m => m === 'dgx' ? 'DGX' : m === 'pop' ? 'Pop' : m;

function alertsBlock(hl) {
  const open = (hl.open || []);
  if (!open.length) return h('div', { class: 'allclear', role: 'status' }, h('span', { 'aria-hidden': true }, '✓'), 'All checks pass: disks above floors, ledger headroom ≥ 15 GB, no failed units, data fresh.');
  return h('div', { class: 'alerts', id: 'alerts', role: 'alert' }, open.map(a => h('div', { class: `alert ${a.level}` },
    h('span', { class: 'ic', 'aria-hidden': true }, a.level === 'crit' ? '▲' : '◆'),
    h('div', { class: 'msg' }, h('b', {}, a.level === 'crit' ? 'Critical · ' : 'Warning · '), a.machine ? `${machineName(a.machine)}: ` : '', a.text),
    a.system ? null : h('button', { class: 'btn small', type: 'button', onclick: () => ackAlert(a) }, 'Acknowledge'),
    askBtn('alert', a.id, a.text, 'Ask'))));
}

function tile(label, value, sub, { tone = 'flat', extra, title } = {}) {
  return h('div', { class: 'card tile', title }, h('div', { class: 'label' }, label), h('div', { class: `value num tone-${tone}` }, value), h('div', { class: 'sub' }, sub), extra);
}

function headlineTiles(snap) {
  const runs = snap.runs || [], active = runs.filter(r => r.active), primary = active[0] || runs[0];
  const verdict = improving(primary);
  const dgx = snap.machines?.dgx, tg = snap.teachers;
  const rew = primary?.spark?.reward_mean || [];
  const tiles = [
    tile(primary ? `Training · ${primary.id}${primary.active ? '' : ' (idle)'}` : 'Training', verdict.headline, verdict.detail || `${primary?.steps ?? 0} steps`, {
      tone: verdict.tone, title: 'Held-out paired comparison against the base policy is the only measure that counts.',
      extra: h('div', {}, spark(rew), h('div', { class: 'sub' }, rew.length ? `reward_mean ${fmt(rew.at(-1))} · step ${primary.steps}` : 'no steps yet')) }),
  ];
  if (dgx) {
    const head = dgx.ledger?.headroom_gb, low = num(head) && head < 15;
    tiles.push(tile('DGX memory ledger headroom', gb(head), `floor 15 GB · RAM available ${gb(dgx.mem_available_gb)} of ${gb(dgx.mem_total_gb, 0)}`, {
      tone: low ? 'bad' : 'flat', extra: bar([{ value: head || 0, label: 'headroom', color: low ? 'var(--crit)' : 'var(--good)' }], { total: Math.max(40, head || 0), mark: 15, thin: true, label: `headroom ${gb(head)}, floor 15 GB` }) }));
    const disks = dgx.disks || [];
    const worst = disks.map(d => ({ d, m: (d.free_gb ?? 0) - (d.floor_gb ?? 0) })).sort((a, b) => a.m - b.m)[0];
    tiles.push(tile('DGX disk, tightest margin', worst ? `${worst.d.free_gb} GB free` : '—', worst ? `${worst.d.mount} · floor ${worst.d.floor_gb} GB · margin ${worst.m} GB` : '', {
      tone: worst && worst.m < 0 ? 'bad' : worst && worst.m < 20 ? 'warn' : 'flat' }));
  }
  if (tg) tiles.push(tile('Teacher generation', `${tg.success} verified`, `${tg.tasks_done} of ${fmt(tg.eligible)} tasks · ${tg.success_last_hour} in the last hour · ${tg.deferred} deferred`, {
    extra: spark((tg.history || []).map(x => x[2])) }));
  return h('div', { class: 'grid cols-4' }, tiles);
}

/* ---------- machines ---------- */
function liveOf(machine) {
  const rec = S.live.get(machine);
  if (!rec || now() - Math.min(rec.ts, rec.received_at ?? rec.ts) > 20000) return null;
  return rec.data;
}

function machineCard(name, m) {
  const live = name === 'dgx' ? liveOf('dgx') : null;
  const cpu = live?.metrics?.cpu_percent ?? m.cpu_percent;
  const seg = memorySegments(m), cls = ['', '', ''];
  const hist = (S.hist.get(name) || []).filter(x => x.ts > now() - 12 * 3600_000);
  const ledger = m.ledger || {};
  const claimParts = seg.segs.map((s, i) => ({ value: s.gb, label: s.label, color: s.kind === 'other' ? 'var(--seg-other)' : `var(--s${i % SLOTS})` }));
  return h('div', { class: 'card stack' },
    h('div', { class: 'row between' }, h('h3', {}, `${machineName(name)}`, h('span', { class: 'muted' }, name === 'dgx' ? ' · GB10, unified memory' : ' · RTX 4060 8 GB')),
      live ? chip('live 5 s', 'good') : h('span', { class: 'muted', style: 'font-size:12px' }, `sampled ${ago(m.ts ?? S.snap.ts, now())}`)),
    h('div', {}, h('div', { class: 'row between' }, h('span', { class: 'muted' }, 'Memory' + (name === 'dgx' ? ' (GPU memory is system RAM)' : '')), h('span', { class: 'num' }, `${gb(seg.used)} / ${gb(seg.total, 0)}`)),
      bar([...claimParts, { value: seg.free, label: 'free', color: 'transparent' }], { total: seg.total, label: `memory used ${gb(seg.used)} of ${gb(seg.total, 0)}` }),
      h('div', { class: 'legend' }, seg.segs.map((s, i) => h('span', {}, h('i', { style: `background:${s.kind === 'other' ? 'var(--seg-other)' : `var(--s${i % SLOTS})`}` }), `${s.label} ${gb(s.gb)}`)))),
    h('dl', { class: 'kv' },
      h('dt', {}, 'CPU'), h('dd', { class: 'num' }, `${pct(cpu)} · load ${fmt(m.load1)}`),
      h('dt', {}, 'GPU'), h('dd', { class: 'num' }, name === 'dgx' ? `${pct(m.gpu_util_percent)} util · memory n/a (shared)` : `${pct(m.gpu_util_percent)} util · ${gb(m.gpu_mem_used_gb)} VRAM`),
      h('dt', {}, 'Swap'), h('dd', { class: 'num' }, gb(m.swap_used_gb)),
      h('dt', {}, 'Ledger'), h('dd', { class: `num ${num(ledger.headroom_gb) && name === 'dgx' && ledger.headroom_gb < 15 ? 'tone-bad' : ''}` }, `${gb(ledger.headroom_gb)} headroom · ${ledger.claims ?? 0} claims · ${ledger.waiting ?? 0} waiting`)),
    h('div', {}, h('div', { class: 'muted', style: 'font-size:12px;margin-bottom:2px' }, `Ledger headroom, last 12 h${name === 'dgx' ? ' (dashed: 15 GB floor)' : ''}`),
      spark(hist.map(x => x.ledger?.headroom_gb), { floor: name === 'dgx' ? 15 : undefined })),
    h('div', { class: 'stack' }, (m.disks || []).map(d => diskRow(d))));
}

function diskRow(d) {
  const floor = d.floor_gb, margin = floor != null ? d.free_gb - floor : null;
  const tone = margin == null ? 'good' : margin < 0 ? 'crit' : margin < 20 ? 'warn' : 'good';
  const scale = floor != null ? floor + 120 : Math.max(d.size_gb * 0.5, d.free_gb);
  return h('div', {}, h('div', { class: 'row between' }, h('span', { class: 'mono' }, d.mount), h('span', { class: 'num' }, `${d.free_gb} GB free of ${d.size_gb}`, floor != null ? h('span', { class: 'muted' }, ` · floor ${floor}`) : null)),
    bar([{ value: Math.min(d.free_gb, scale), label: 'free', color: `var(--${tone === 'good' ? 'good' : tone})` }], { total: scale, mark: floor, thin: true, label: `${d.free_gb} GB free${floor != null ? `, floor ${floor} GB` : ''}` }));
}

/* ---------- processes ---------- */
function procCard(p) {
  const pr = p.progress, res = p.resources || {};
  const failed = p.state === 'failed';
  const frac = pr?.total ? pr.done / pr.total : null;
  const open = S.open.has(p.unit);
  const d = h('details', { class: 'more', open: open || null, ontoggle: e => e.target.open ? S.open.add(p.unit) : S.open.delete(p.unit) },
    h('summary', {}, 'What it is and why'),
    h('div', {}, h('div', {}, p.description), p.rationale ? h('div', { class: 'why' }, h('b', {}, 'Why: '), p.rationale) : null,
      p.investigation ? h('div', {}, 'Part of investigation ', h('a', { href: '#/investigations' }, p.investigation)) : null,
      p.run ? h('div', { class: 'muted' }, 'Run ', h('span', { class: 'mono' }, p.run)) : null,
      h('div', { class: 'row' }, askBtn('unit', p.unit, p.title, 'Ask about this'), p.state === 'running' ? askBtn('unit', p.unit, p.title, 'Request stop…', 'stop') : null)));
  return h('div', { class: `card proc ${p.state}` },
    h('div', { class: 'row between' }, h('div', {}, h('div', { class: 'title' }, p.title), h('div', { class: 'unit mono' }, p.unit)),
      h('div', { class: 'row' }, chip(machineName(p.machine)), chip(p.kind), chip(failed ? 'failed' : p.state, failed ? 'failed' : p.state === 'running' ? 'running' : ''))),
    p.state === 'running' && pr ? h('div', { class: 'progress' }, h('div', { class: 'row between' },
      h('span', { class: 'num' }, pr.total ? `${pr.done} / ${pr.total} ${pr.unit}` : `${pr.unit.replace(/s$/, '')} ${pr.done}`), h('span', { class: 'muted num' }, p.eta_s ? `ETA ${dur(p.eta_s)}` : 'ETA unknown')),
      frac != null ? bar([{ value: frac, label: 'done', color: 'var(--info)' }], { total: 1, thin: true, label: `${Math.round(100 * frac)}% done` }) : null) : null,
    h('div', { class: 'row muted num', style: 'font-size:12.5px' }, `${p.state === 'running' ? 'running' : p.state} ${dur(p.elapsed_s)}`,
      num(res.used_gb) ? ` · ${gb(res.used_gb)} of ${gb(res.budget_gb, 0)} budget` : null, num(res.cpu_percent) ? ` · CPU ${pct(res.cpu_percent)}` : null, res.gpu ? ' · GPU' : null),
    num(res.used_gb) && res.budget_gb ? bar([{ value: Math.min(res.used_gb, res.budget_gb), label: 'used', color: res.used_gb > res.budget_gb ? 'var(--crit)' : 'var(--s0)' }], { total: res.budget_gb, thin: true, label: `${gb(res.used_gb)} of ${gb(res.budget_gb, 0)}` }) : null,
    p.last_log ? h('div', { class: 'last mono', title: p.last_log }, p.last_log) : null, d);
}

function queueCard(q) {
  return h('div', { class: 'card stack' }, h('div', { class: 'row between' }, h('b', {}, q.title), h('div', { class: 'row' }, q.machine ? chip(machineName(q.machine)) : null, chip(q.source === 'ledger-waiting' ? 'waiting for memory' : 'queued', q.source === 'ledger-waiting' ? 'warn' : ''))),
    h('div', {}, q.description), q.rationale ? h('div', { class: 'muted' }, h('b', {}, 'Why: '), q.rationale) : null,
    q.blocked_on ? h('div', { class: 'muted' }, h('b', {}, 'Blocked on: '), q.blocked_on) : null, h('div', {}, askBtn('queue', q.title, q.title)));
}

/* ---------- views ---------- */
function viewNow() {
  const snap = S.snap, hl = health(snap, now(), S.acked);
  const procs = snap.processes || [], run = procs.filter(p => p.state === 'running'), rest = procs.filter(p => p.state !== 'running');
  const failed = rest.filter(p => p.state === 'failed'), done = rest.filter(p => p.state !== 'failed');
  return h('div', {},
    alertsBlock(hl),
    h('section', {}, headlineTiles(snap)),
    h('section', {}, h('div', { class: 'head' }, h('h2', {}, 'Machines'), h('p', { class: 'muted' }, 'GPU memory on the DGX is system RAM: one pool, split by ledger claims.')),
      h('div', { class: 'grid cols-2', id: 'machines' }, Object.entries(snap.machines || {}).map(([n, m]) => machineCard(n, m)))),
    h('section', {}, h('div', { class: 'head' }, h('h2', {}, `Running now (${run.length})`)),
      run.length ? h('div', { class: 'grid cols-2' }, run.map(procCard)) : h('div', { class: 'card empty' }, 'Nothing is running.')),
    snap.queue?.length ? h('section', {}, h('div', { class: 'head' }, h('h2', {}, `Queued (${snap.queue.length})`)), h('div', { class: 'grid cols-2' }, snap.queue.map(queueCard))) : null,
    rest.length ? h('section', {}, h('div', { class: 'head' }, h('h2', {}, `Finished and failed in the last 6 h (${rest.length})`)),
      h('div', { class: 'grid cols-2' }, [...failed, ...done].map(procCard))) : null);
}

/* training: built once, charts persist across updates */
const PRESETS = {
  outcome: ['reward_mean', 'spread_fraction', 'reasoning_tokens_per_call', 'reply_tokens_per_call'],
  drift: ['kl_mean', 'behaviour.logp_change', 'behaviour.entropy', 'behaviour.close_ratio', 'ref_dlogp_max', 'bf16_changed_vs_start_fraction'],
  optimizer: ['grad_norm', 'lr', 'rho_mean', 'abs_log_rho_mean', 'grad_moment_cosine', 'grad_mass_share.think'],
  throughput: ['learner_tokens_per_s', 'actor_tokens_per_s_collect_phase', 'cuda_peak_gb', 'phase_s.wait_groups'],
};
const TITLES = { reward_mean: 'Reward (mean)', spread_fraction: 'Groups with reward spread', kl_mean: 'KL to start policy', grad_norm: 'Gradient norm', lr: 'Learning rate', 'behaviour.logp_change': 'Behaviour: log-prob change on probe', 'behaviour.entropy': 'Behaviour: entropy', 'behaviour.close_ratio': 'Behaviour: </think> hazard ratio', ref_dlogp_max: 'Max |Δ log-prob| vs start', learner_tokens_per_s: 'Learner tokens/s', actor_tokens_per_s_collect_phase: 'Actor tokens/s (collecting)', cuda_peak_gb: 'CUDA peak (GB)', bf16_changed_vs_start_fraction: 'BF16 weights changed vs start', reasoning_tokens_per_call: 'Reasoning tokens per call', reply_tokens_per_call: 'Reply tokens per call' };
const LOGY = new Set(['kl_mean', 'lr', 'ref_dlogp_max', 'abs_log_rho_mean']);
let T = null;

function buildTraining() {
  const root = h('div', {});
  const controls = h('div', { class: 'card stack' });
  const panelHost = h('div', { class: 'panels' });
  const heldHost = h('div', {});
  root.append(controls, h('section', {}, h('div', { class: 'head' }, h('h2', {}, 'Learner metrics by step'), h('p', { class: 'muted' }, 'Hover or use the slider under a chart to compare all runs at one step; the cursor is shared.')), panelHost),
    h('section', {}, h('div', { class: 'head' }, h('h2', {}, 'Held-out result vs the base policy'), h('p', { class: 'muted' }, 'Paired change in task pass rate, points, with 95% task-bootstrap interval. Whiskers crossing zero are not significant.')), heldHost));
  T = { root, controls, panelHost, heldHost, charts: new Map(), cursorSync: x => T.charts.forEach(c => c.chart.setCursor(x)) };
  return T;
}

function trainingControls() {
  const runs = S.snap?.runs || [];
  const chips = h('div', { class: 'runs', role: 'group', 'aria-label': 'Runs shown' }, runs.map(r => {
    const on = S.sel.has(r.id), slot = S.slot.get(r.id);
    return h('button', { class: 'runchip', type: 'button', 'aria-pressed': String(on), title: `${r.steps} steps · ${r.kind}`, onclick: () => { if (on) select(r.id, false); else if (!select(r.id, true)) notice('sel', `At most ${SLOTS} runs can be compared at once.`); schedule(); } },
      h('i', { style: on ? `background:var(--s${slot})` : '' }), r.id, r.active ? chip('active', 'running') : null);
  }));
  const metrics = metricNames(S.runs);
  const sel = h('select', { id: 'preset', onchange: e => { S.preset = e.target.value; store('ops.preset', S.preset); schedule(); } },
    ...[['outcome', 'Outcome'], ['drift', 'Policy drift'], ['optimizer', 'Optimizer'], ['throughput', 'Throughput'], ['custom', 'One metric…']].map(([v, l]) => h('option', { value: v, selected: S.preset === v || null }, l)));
  const cm = h('select', { id: 'custom', 'aria-label': 'Metric', onchange: e => { S.custom = e.target.value; store('ops.custom', S.custom); schedule(); }, hidden: S.preset !== 'custom' || null },
    ...metrics.map(m => h('option', { value: m, selected: S.custom === m || null }, m)));
  const ema = h('select', { 'aria-label': 'Smoothing', onchange: e => { S.ema = e.target.value === 'off' ? null : Number(e.target.value); store('ops.ema', S.ema); schedule(); } },
    ...[['off', 'No smoothing'], ['0.5', 'Light smoothing'], ['0.3', 'Smoothing'], ['0.1', 'Heavy smoothing']].map(([v, l]) => h('option', { value: v, selected: String(S.ema ?? 'off') === v || null }, l)));
  return [h('div', { class: 'row between' }, h('h2', {}, 'Runs'), h('div', { class: 'row' }, h('label', {}, 'Metrics', sel), h('label', { hidden: S.preset !== 'custom' || null }, 'Metric', cm), h('label', {}, 'Smoothing', ema))), chips,
    h('div', { class: 'muted', style: 'font-size:12px' }, 'Each run keeps its colour while selected. Steps are slow (hours), so early curves are short.')];
}

function updateTraining() {
  if (!S.snap) return;
  if (!T) buildTraining();
  T.controls.replaceChildren(...trainingControls());
  const metrics = S.preset === 'custom' ? [S.custom] : PRESETS[S.preset];
  const runs = new Map([...S.sel].map(id => [id, S.runs.get(id) || new Map()]));
  for (const [m, c] of [...T.charts]) if (!metrics.includes(m)) { c.chart.destroy(); c.el.remove(); T.charts.delete(m); }
  for (const m of metrics) {
    if (!T.charts.has(m)) {
      const host = h('div', { class: 'panel' }, h('div', {}));
      T.panelHost.append(host);
      T.charts.set(m, { el: host, chart: window.tendrilSnippets.chart.mount(host.firstChild, { title: TITLES[m] || m, xLabel: 'Step', yLabel: m, crosshair: true, maxPoints: 2000, onCursor: T.cursorSync, xFormat: v => String(Math.round(v)), yFormat: v => fmt(v, 3) }) });
    }
  }
  for (const m of metrics) {
    const { el, chart } = T.charts.get(m);
    const series = metricSeries(runs, m);
    series.forEach((s, i) => el.style.setProperty(`--pc-${i}`, `var(--s${S.slot.get(s.id)})`));
    chart.update(series, { title: TITLES[m] || m, yLabel: m, emaAlpha: S.ema, yScale: LOGY.has(m) && series.every(s => s.points.every(p => p.y > 0)) ? 'log' : 'linear' });
    el.hidden = false;
  }
  T.panelHost.append(...metrics.map(m => T.charts.get(m).el));
  T.heldHost.replaceChildren(heldView());
}

function heldView() {
  const rows = [];
  for (const r of S.snap.runs || []) for (const hd of r.held || []) if (num(hd.delta_pts) && (S.sel.has(r.id) || !S.sel.size)) rows.push({ run: r.id, ...hd });
  if (!rows.length) return h('div', { class: 'card empty' }, 'No completed held comparison for the selected runs yet.');
  rows.sort((a, b) => a.run.localeCompare(b.run) || a.step - b.step);
  const W = 640, rowH = 28, padL = 190, padR = 24, top = 14, H = top + rows.length * rowH + 26;
  const lim = Math.max(10, ...rows.flatMap(r => [Math.abs(r.ci_lo ?? 0) * 100, Math.abs(r.ci_hi ?? 0) * 100, Math.abs(r.delta_pts)])) * 1.1;
  const x = v => padL + ((v + lim) / (2 * lim)) * (W - padL - padR);
  const root = svg('svg', { class: 'forest', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Held-out change versus base per run and step' });
  for (const tick of [-lim, -lim / 2, 0, lim / 2, lim].map(v => Math.round(v))) {
    root.append(svg('line', { class: tick === 0 ? 'zero' : 'grid', x1: x(tick), x2: x(tick), y1: top - 6, y2: H - 22 }), svg('text', { x: x(tick), y: H - 8, 'text-anchor': 'middle' }, `${tick > 0 ? '+' : ''}${tick}`));
  }
  rows.forEach((r, i) => {
    const y = top + i * rowH + rowH / 2, sig = r.p != null && r.p < 0.05, color = !sig ? 'var(--muted)' : r.delta_pts > 0 ? 'var(--good)' : 'var(--crit)';
    root.append(svg('text', { x: padL - 10, y: y + 4, 'text-anchor': 'end' }, `${r.run} · step ${r.step}${r.complete ? '' : ' (partial)'}`));
    if (num(r.ci_lo) && num(r.ci_hi) && (r.ci_lo || r.ci_hi)) root.append(svg('line', { class: 'ci', x1: x(r.ci_lo * 100), x2: x(r.ci_hi * 100), y1: y, y2: y, stroke: color }));
    root.append(svg('circle', { class: 'pt', cx: x(r.delta_pts), cy: y, r: 5, fill: color }, svg('title', {}, `${signed(r.delta_pts, 1)} pts, p ${fmt(r.p, 2)}`)));
  });
  const table = h('table', {}, h('thead', {}, h('tr', {}, ...['Run', 'Step', 'Change (pts)', 'p', 'Base → policy pass rate', 'Tasks', 'Ran', 'Tasks up / down', 'Loop rate Δ'].map((c, i) => h('th', { class: i > 1 ? 'r' : '' }, c)))),
    h('tbody', {}, rows.map(r => h('tr', {}, h('td', { class: 'mono' }, r.run), h('td', {}, r.step), h('td', { class: `r num ${r.p < 0.05 ? (r.delta_pts > 0 ? 'tone-good' : 'tone-bad') : ''}` }, signed(r.delta_pts, 1)), h('td', { class: 'r num' }, fmt(r.p, 2)),
      h('td', { class: 'r num' }, `${fmt(100 * r.first_rate, 3)}% → ${fmt(100 * r.second_rate, 3)}%`), h('td', { class: 'r num' }, r.tasks), h('td', { class: 'r num' }, `${r.ran}/${r.of}`), h('td', { class: 'r num' }, `${r.tasks_up ?? '—'} / ${r.tasks_down ?? '—'}`), h('td', { class: 'r num' }, signed(100 * (r.loop_rate_delta ?? NaN), 1, ' pts'))))));
  return h('div', { class: 'stack' }, h('div', { class: 'card' }, root), h('div', { class: 'card scroll' }, table));
}

function viewTeachers() {
  const tg = S.snap.teachers;
  if (!tg) return h('div', { class: 'card empty' }, 'No teacher generation pool is configured in the exporter.');
  const outs = [['task_success', 'success', 'seg-task_success'], ['semantic_mismatch', 'wrong answer', 'seg-semantic_mismatch'], ['budget_exhaustion', 'out of budget', 'seg-budget_exhaustion'], ['other', 'other', 'seg-other']];
  const wrap = h('div', {});
  const chartHost = h('div', { class: 'panel' }, h('div', {}));
  const hist = tg.history || [];
  wrap.append(
    h('div', { class: 'grid cols-4' },
      tile('Verified teacher episodes', fmt(tg.success), `${tg.success_last_hour} in the last hour · last activity ${ago(tg.last_activity_ts, now())}`, { tone: 'flat' }),
      tile('Tasks attempted', fmt(tg.tasks_done), `of ${fmt(tg.eligible)} eligible (${(100 * tg.tasks_done / (tg.eligible || 1)).toFixed(1)}%)`),
      tile('Deferred for retry', fmt(tg.deferred), 'unresolved or infrastructure errors; they do not count as teacher failures'),
      tile('Families with a success', `${tg.families_with_success} / ${tg.families_total}`, 'task families covered so far')),
    h('section', {}, h('div', { class: 'head' }, h('h2', {}, 'By teacher')),
      h('div', { class: 'card scroll' }, h('table', {}, h('thead', {}, h('tr', {}, ...['Teacher', 'Outcomes', 'Success rate', 'Tasks', 'Mean seconds', 'Mean tokens', 'Deferred'].map((c, i) => h('th', { class: i > 1 ? 'r' : '' }, c)))),
        h('tbody', {}, tg.teachers.map(tc => {
          const parts = outcomeParts(tc), total = parts.reduce((a, p) => a + p.n, 0) || 1;
          return h('tr', {}, h('td', {}, h('div', { class: 'mono' }, tc.id)), h('td', { class: 'outbar' }, bar(parts.map(p => ({ value: p.n, label: p.label, cls: `seg-${p.key}` })), { total, thin: true, label: parts.map(p => `${p.label} ${p.n}`).join(', ') })),
            h('td', { class: 'r num' }, `${tc.success} / ${tc.episodes} · ${num(tc.success_rate) ? Math.round(100 * tc.success_rate) : '—'}%`), h('td', { class: 'r num' }, tc.tasks), h('td', { class: 'r num' }, fmt(tc.seconds_mean)), h('td', { class: 'r num' }, fmt(tc.tokens_mean)),
            h('td', { class: 'r num' }, Object.entries(tc.deferred || {}).map(([k, v]) => `${v} ${k.replace('_', ' ')}`).join(', ') || '0'));
        }))), h('div', { class: 'legend' }, outs.map(([, label, cls]) => h('span', {}, h('i', { class: cls }), label))))),
    h('section', {}, h('div', { class: 'head' }, h('h2', {}, 'Progress over time'), h('p', { class: 'muted' }, 'Cumulative verified episodes and attempted tasks; a point is added whenever either changes.')), chartHost),
    h('section', {}, h('div', { class: 'head' }, h('h2', {}, `Task families (top ${tg.families.length} by success)`)),
      h('div', { class: 'card scroll' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Family'), h('th', { class: 'r' }, 'Tasks'), h('th', { class: 'r' }, 'With a verified episode'))), h('tbody', {}, tg.families.map(f => h('tr', {}, h('td', { class: 'mono' }, f.family), h('td', { class: 'r num' }, f.tasks), h('td', { class: 'r num' }, f.success))))))));
  queueMicrotask(() => {
    const c = window.tendrilSnippets.chart.mount(chartHost.firstChild, { title: 'Teacher generation', xLabel: 'Time', yLabel: 'count', maxPoints: 300, crosshair: true, xFormat: v => dayClock(v), yFormat: v => fmt(v) });
    chartHost.style.setProperty('--pc-0', 'var(--s2)'); chartHost.style.setProperty('--pc-1', 'var(--s0)');
    c.update([{ id: 'success', label: 'verified episodes', points: hist.map(x => ({ x: x[0], y: x[2] })) }, { id: 'tasks', label: 'tasks attempted', points: hist.map(x => ({ x: x[0], y: x[1] })) }]);
  });
  return wrap;
}

function viewInvestigations() {
  const invs = S.snap.investigations || [];
  if (!invs.length) return h('div', { class: 'card empty' }, 'No open investigations. The supervising agent maintains them in annotations.json.');
  const wrap = h('div', { class: 'invs' });
  for (const inv of invs) {
    const charts = h('div', { class: 'panels' });
    const findings = (inv.findings || []).map(f => h('div', { class: 'finding' }, chip({ evidence: 'evidence', ruled_out: 'ruled out', open: 'open' }[f.status] || f.status, { evidence: 'good', ruled_out: '', open: 'warn' }[f.status]), h('span', {}, f.text)));
    wrap.append(h('div', { class: 'card stack' },
      h('div', { class: 'row between' }, h('h2', {}, inv.title), h('div', { class: 'row' }, chip(inv.status, inv.status === 'open' ? 'warn' : inv.status === 'confirmed' ? 'good' : ''), h('span', { class: 'muted' }, `updated ${ago(inv.updated_ts, now())}`), askBtn('investigation', inv.id, inv.title))),
      h('div', {}, h('b', {}, 'Question: '), inv.question), h('div', {}, inv.summary),
      findings.length ? h('div', { class: 'stack' }, h('h3', {}, 'Findings'), ...findings) : null,
      inv.next?.length ? h('div', {}, h('h3', {}, 'Next'), h('ul', { style: 'margin:4px 0 0;padding-left:20px' }, inv.next.map(n => h('li', {}, n)))) : null,
      inv.runs?.length ? h('div', { class: 'muted' }, 'Runs: ', inv.runs.map(r => h('span', { class: 'mono', style: 'margin-right:8px' }, r))) : null, charts));
    queueMicrotask(() => (inv.charts || []).forEach((cfg, ci) => {
      const host = h('div', { class: 'panel' }, h('div', {})); charts.append(host);
      const c = window.tendrilSnippets.chart.mount(host.firstChild, { title: cfg.title, xLabel: 'Step', yLabel: cfg.metric, crosshair: true, maxPoints: 2000, xFormat: v => String(Math.round(v)), yFormat: v => fmt(v) });
      const series = metricSeries(new Map((cfg.runs || []).slice(0, SLOTS).map(r => [r, S.runs.get(r) || new Map()])), cfg.metric);
      series.forEach((s, i) => host.style.setProperty(`--pc-${i}`, `var(--s${i})`));
      c.update(series, { emaAlpha: null });
      if (!series.length) host.append(h('p', { class: 'muted' }, 'No retained step data for these runs (the stream keeps the newest records only).'));
    }));
  }
  return wrap;
}

function viewFallow() {
  const rows = (S.snap.fallow || []).filter(r => matches({ ...r, ...r.fields }, S.q));
  const sections = [...new Set(rows.map(r => r.section))];
  const search = h('input', { type: 'search', placeholder: 'Filter the register…', value: S.q, 'aria-label': 'Filter the fallow register', oninput: e => { S.q = e.target.value; const pos = e.target.selectionStart; schedule(); queueMicrotask(() => { const i = $('view').querySelector('input[type=search]'); i?.focus(); i?.setSelectionRange(pos, pos); }); } });
  return h('div', {}, h('div', { class: 'row between' }, h('div', {}, h('h2', {}, 'Fallow register'), h('p', { class: 'muted', style: 'margin:2px 0 0' }, 'Built capabilities that are not in use: what blocks each one and what unblocks it.')), search),
    !rows.length ? h('div', { class: 'card empty', style: 'margin-top:14px' }, 'No rows match.') : sections.map(sec => h('section', {}, h('div', { class: 'head' }, h('h3', {}, sec || 'Register')),
      h('div', { class: 'card scroll' }, h('table', {}, h('thead', {}, h('tr', {}, ...['', 'Piece', 'Code now', 'Blocker', 'Next action', ''].map(c => h('th', {}, c)))),
        h('tbody', {}, rows.filter(r => r.section === sec).map(r => h('tr', {}, h('td', { class: 'mono' }, r.id), h('td', {}, h('b', {}, r.piece)), h('td', {}, r.code_now), h('td', {}, r.blocker), h('td', {}, r.next_action),
          h('td', {}, askBtn('fallow', r.id, `${r.id} ${r.piece}`, 'Ask'))))))))));
}

function viewActivity() {
  const ev = [...S.events].reverse().filter(e => S.evFilter === 'all' || e.level === 'warn');
  const f = h('select', { 'aria-label': 'Filter', onchange: e => { S.evFilter = e.target.value; schedule(); } }, h('option', { value: 'all', selected: S.evFilter === 'all' || null }, 'All events'), h('option', { value: 'warn', selected: S.evFilter === 'warn' || null }, 'Warnings only'));
  return h('div', {}, h('div', { class: 'row between' }, h('h2', {}, 'Activity'), f),
    h('div', { class: 'card', style: 'margin-top:12px' }, ev.length ? h('table', {}, h('tbody', {}, ev.map(e => h('tr', {}, h('td', { class: 'num muted', style: 'white-space:nowrap' }, dayClock(e.ts)), h('td', {}, chip(e.kind.replace('unit_', '').replace('_', ' '), e.level === 'warn' ? 'warn' : e.kind === 'unit_started' ? 'running' : e.kind === 'held' ? 'info' : '')), h('td', {}, e.text))))) : h('div', { class: 'empty' }, 'No events yet.')));
}

/* ---------- shell ---------- */
const TABS = [['now', 'Now'], ['training', 'Training'], ['teachers', 'Teacher data'], ['investigations', 'Investigations'], ['fallow', 'Fallow'], ['activity', 'Activity']];
const sigs = {};
function routeFromHash() { const k = location.hash.replace(/^#\/?/, ''); S.tab = TABS.some(([id]) => id === k) ? k : 'now'; }
function drawTabs(hl) {
  const nav = $('tabs'); nav.replaceChildren(...TABS.map(([id, label]) => h('button', { class: 'tab', role: 'tab', type: 'button', 'aria-selected': String(S.tab === id), id: `tab-${id}`,
    onclick: () => { location.hash = `#/${id}`; }, onkeydown: e => { const i = TABS.findIndex(([x]) => x === S.tab), j = e.key === 'ArrowRight' ? i + 1 : e.key === 'ArrowLeft' ? i - 1 : null; if (j != null) { e.preventDefault(); location.hash = `#/${TABS[(j + TABS.length) % TABS.length][0]}`; $(`tab-${TABS[(j + TABS.length) % TABS.length][0]}`)?.focus(); } } },
    label, id === 'now' && hl?.open?.length ? h('span', { class: 'badge alert' }, hl.open.length) : null, id === 'investigations' && S.snap?.investigations?.length ? h('span', { class: 'badge' }, S.snap.investigations.length) : null)));
}

function render() {
  const hl = health(S.snap, now(), S.acked);
  const pill = $('health'), off = !t.connected;
  pill.className = `pill ${off && hl.level === 'ok' ? 'warn' : hl.level}`;
  $('health-text').textContent = off ? 'Offline: showing last data' : hl.title;
  pill.onclick = () => { location.hash = '#/now'; $('alerts')?.scrollIntoView({ block: 'center' }); };
  drawTabs(hl);
  $('updated').textContent = S.snap ? `data ${ago(S.snap.ts, now())}` : '';
  const view = $('view');
  if (!S.snap) { view.replaceChildren(h('div', { class: 'card empty' }, 'Waiting for the first snapshot from the exporter…')); return; }
  if (S.tab === 'training') {
    updateTraining();
    if (view.firstChild !== T.root) view.replaceChildren(T.root);
    return;
  }
  const sig = `${S.tab}:${S.snap.ts}:${S.version}:${S.q}:${S.evFilter}:${S.events.length}:${Object.keys(S.acked).length}:${S.tab === 'now' ? S.live.get('dgx')?.ts : ''}`;
  if (sigs.last === sig && view.firstChild) return;
  sigs.last = sig;
  const keep = { x: scrollX, y: scrollY };
  const builder = { now: viewNow, teachers: viewTeachers, investigations: viewInvestigations, fallow: viewFallow, activity: viewActivity }[S.tab];
  view.replaceChildren(builder());
  if (S.tab === 'fallow' && S.q) $('view').querySelector('input[type=search]')?.focus();
  scrollTo(keep.x, keep.y);
}

$('theme').addEventListener('click', () => {
  const order = ['auto', 'light', 'dark'], cur = store('ops.theme') || 'auto', next = order[(order.indexOf(cur) + 1) % 3];
  store('ops.theme', next); applyTheme();
});
function applyTheme() {
  const v = store('ops.theme') || 'auto';
  if (v === 'auto') document.documentElement.removeAttribute('data-theme'); else document.documentElement.dataset.theme = v;
  $('theme').textContent = v === 'auto' ? 'Auto' : v === 'light' ? 'Light' : 'Dark';
}
applyTheme();
addEventListener('hashchange', () => { routeFromHash(); sigs.last = null; schedule(); window.scrollTo(0, 0); });
routeFromHash();
t.on('connect', () => { void init(); schedule(); }); t.on('disconnect', schedule);
setInterval(() => { if (S.tab === 'now' || S.tab === 'activity') sigs.last = null; schedule(); }, 5000);
void init(); render();
