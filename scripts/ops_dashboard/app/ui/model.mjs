// Pure helpers of the ops dashboard: formatting, health verdicts, series building. No DOM, no tendril: testable in node.
import { editRef } from '../ops-protocol.mjs';

export const num = v => typeof v === 'number' && Number.isFinite(v);

export function fmt(v, digits = 3) {
  if (!num(v)) return '—';
  if (v !== 0 && Math.abs(v) < 1e-3) return v.toExponential(1).replace('e-', 'e−');
  if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString('en-US');
  return String(Number(v.toPrecision(digits)));
}
export const gb = (v, d = 1) => num(v) ? `${v.toFixed(d)} GB` : '—';
export const pct = v => num(v) ? `${Math.round(v)}%` : '—';
export const signed = (v, d = 1, unit = '') => num(v) ? `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(v).toFixed(d)}${unit}` : '—';

export function dur(s) {
  if (!num(s)) return '—';
  s = Math.max(0, Math.round(s));
  if (s < 90) return `${s}s`;
  const m = Math.round(s / 60);
  if (m < 90) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h ${String(m % 60).padStart(2, '0')}m`;
  return `${Math.floor(h / 24)}d ${h % 24}h`;
}
export const ago = (ts, now) => num(ts) ? `${dur((now - ts) / 1000)} ago` : '—';
export const clock = ts => num(ts) ? new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—';
export const dayClock = ts => num(ts) ? new Date(ts).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—';
export const shortUnit = u => String(u || '').replace(/\.service$/, '');
export const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;

/** Direction of a series: mean of the newest k values against the k before them; a short series says so. */
export function trend(values, k = 4) {
  const v = values.filter(num);
  if (v.length < 2 * k) return { state: 'early', n: v.length, text: `${v.length} steps so far` };
  const last = mean(v.slice(-k)), prev = mean(v.slice(-2 * k, -k)), delta = last - prev;
  const scale = Math.max(Math.abs(prev), 1e-9);
  const state = Math.abs(delta) / scale < 0.03 ? 'flat' : delta > 0 ? 'up' : 'down';
  return { state, n: v.length, delta, last, prev, text: `${signed(delta, 3)} over ${2 * k} steps` };
}

/** Sample-weighted EMA in original units; restarts after a gap (null / non-finite) as the present-live-measurements cookbook suggests. */
export function ema(points, alpha) {
  if (!(alpha > 0 && alpha < 1)) return points.map(p => ({ ...p, s: p.y }));
  let prev = null;
  return points.map(p => {
    if (!num(p.y)) { prev = null; return { ...p, s: null }; }
    prev = prev === null ? p.y : alpha * p.y + (1 - alpha) * prev;
    return { ...p, s: prev };
  });
}

export const SNAP_STALE_MS = 150_000;
const RUN_QUIET_MS = 6 * 3600_000;
export const latestHeld = run => (run.held || []).filter(h => num(h.delta_pts)).sort((a, b) => (a.step ?? 0) - (b.step ?? 0)).at(-1);

/** One verdict for the whole system: level ok|warn|crit plus the items that cause it. `acked` ids are listed but do not count. */
export function health(snap, now, acked = {}) {
  const items = [];
  if (!snap) return { level: 'warn', title: 'Waiting for data', items: [{ id: 'nodata', level: 'warn', text: 'No snapshot received yet.' }], open: [] };
  const age = now - snap.ts;
  if (age > SNAP_STALE_MS) items.push({ id: 'stale', level: age > 600_000 ? 'crit' : 'warn', text: `Exporter data is ${dur(age / 1000)} old. The exporter or collector unit may have stopped.`, system: true });
  for (const a of snap.alerts || []) items.push({ id: a.id, level: a.level, text: a.text, machine: a.machine, unit: a.unit });
  const gate = snap.ei?.held?.gate, pair = gate?.pair;
  if (pair && pair.ci_pts[1] < 0) items.push({ id: `held:${gate.candidate}`, level: 'crit', text: `Held regression: ${gate.candidate} is ${signed(pair.delta_pts, 1)} pts against ${gate.reference} (95% interval ${signed(pair.ci_pts[0], 1)} to ${signed(pair.ci_pts[1], 1)}).` });
  for (const r of snap.runs || []) {      // retired GRPO runs still alarm if one is live
    if (!r.active) continue;
    if (r.last_ts && now - r.last_ts > RUN_QUIET_MS) items.push({ id: `quiet:${r.id}`, level: 'warn', text: `${r.id} wrote nothing for ${dur((now - r.last_ts) / 1000)}.`, run: r.id });
  }
  const open = items.filter(i => !acked[i.id]);
  const level = open.some(i => i.level === 'crit') ? 'crit' : open.length ? 'warn' : 'ok';
  const title = level === 'ok' ? 'Healthy' : `${open.length} ${open.length === 1 ? 'issue needs' : 'issues need'} attention`;
  return { level, title, items, open };
}

/** Compact number: 1.2k, 4.5M. */
export const compact = v => !num(v) ? '—' : Math.abs(v) >= 1e9 ? `${(v / 1e9).toFixed(1)}B` : Math.abs(v) >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : Math.abs(v) >= 1e4 ? `${Math.round(v / 1e3)}k` : Math.abs(v) >= 1e3 ? `${(v / 1e3).toFixed(1)}k` : String(Math.round(v));
export const rate = v => num(v) ? `${(100 * v).toFixed(1)}%` : '—';
/** "≈ 09:39 (in 41m)" for a future timestamp, "overdue" past it. */
export const etaText = (ts, now) => !num(ts) ? 'ETA unknown' : ts < now ? 'finishing' : `≈ ${clock(ts)} (in ${dur((ts - now) / 1000)})`;
export const where = w => ({ dgx: 'DGX', pop: 'Pop', colab: 'Colab A100', api: 'API', 'colab-g4a': 'Colab A100', 'colab-g4b': 'Colab L4' })[w] || w;

/** The steps of one round, in order, each done | active | todo with a short detail. `gate` is ei.held.gate. */
export function roundSteps(r, held, now) {
  const st = r.stamps, s = r.sft;
  const sftActive = r.state === 'sft', published = r.state === 'published';
  const pubResult = r.version != null ? (held?.results || []).filter(x => x.version === r.version).at(-1) : null;
  const state = (done, active) => done ? 'done' : active ? 'active' : 'todo';
  const prepActive = r.state === 'preparing';
  return [
    { key: 'frontier', label: 'Frontier', state: state(!!st.frontier, prepActive && !st.frontier), detail: r.frontier?.tasks ? `${r.frontier.tasks} tasks` : '' },
    { key: 'data', label: 'Data', state: state(!!st.data, prepActive && !!st.frontier && !st.data), detail: r.data_progress != null ? `${r.data_progress} filled` : '' },
    { key: 'build', label: 'Build', state: state(!!st.build, prepActive && !!st.data && !st.build), detail: r.mix ? `${compact(r.mix.chains)}` : '' },
    { key: 'sft', label: 'SFT', state: state(published || !!st.sft_end, sftActive && !st.sft_end), detail: s ? `${s.step}/${s.of}` : '' },
    { key: 'publish', label: 'Publish', state: state(published, !published && !!st.sft_end), detail: published ? `v${r.version}` : '' },
    { key: 'held', label: 'Held', state: state(!!pubResult, published && !pubResult), detail: pubResult ? rate(pubResult.rate) : '' },
  ];
}

/** Estimated times for a round: SFT end and publish (merge and export took about five minutes in finished rounds). */
export function roundEta(r) {
  if (r.state === 'published') return { sftEnd: r.stamps.sft_end, publish: r.stamps.published };
  const sftEnd = r.sft?.eta_ts || null;
  return { sftEnd, publish: sftEnd ? sftEnd + 5 * 60_000 : null };
}

/** Stage segments of a round for the timeline: [{key,label,from,to,est}] in ms; unfinished ends use `now` or the estimate. */
export function roundSegments(r, now) {
  const st = r.stamps, e = roundEta(r), out = [];
  const push = (key, label, from, to, est = false) => { if (from && to && to > from) out.push({ key, label, from, to, est }); };
  push('prepare', 'Frontier, data, build', st.start, st.build || (r.state === 'preparing' ? now : null));
  push('sft', 'SFT', st.sft_start, st.sft_end || (r.state === 'sft' ? now : null));
  if (r.state === 'sft' && e.sftEnd && e.sftEnd > now) push('sft', 'SFT (estimate)', now, e.sftEnd, true);
  push('publish', 'Merge and publish', st.sft_end || (r.state === 'sft' ? e.sftEnd : null), st.published || (e.publish && r.state === 'sft' ? e.publish : null), !st.published);
  return out;
}

/** Episodes per hour for the last `hours` hours, a bar per hour (missing hours are zero). */
export function hourlyBars(hourly, now, hours = 12) {
  const top = Math.floor(now / 3600e3) * 3600e3, map = new Map((hourly || []).map(b => [b.t, b]));
  return Array.from({ length: hours }, (_, i) => { const t = top - (hours - 1 - i) * 3600e3, b = map.get(t) || {}; return { t, dgx: b.dgx || 0, colab: b.colab || 0, pop: b.pop || 0 }; });
}

/** Headline of "is training making progress?": the paired held result when there is one, else the newest measured pass rate. */
export function progress(ei) {
  const held = ei?.held;
  if (!held) return { tone: 'flat', headline: '—', sub: 'no held data yet', detail: '' };
  const gate = held.gate, pair = gate?.pair;
  const latest = [...(held.results || [])].filter(r => r.version != null).sort((a, b) => b.version - a.version)[0];
  if (pair) {
    const tone = pair.ci_pts[0] > 0 ? 'good' : pair.ci_pts[1] < 0 ? 'bad' : 'flat';
    return { tone, headline: `${signed(pair.delta_pts, 1)} pts`, sub: `${gate.candidate} vs ${gate.reference}, held-v3 paired`,
      detail: `95% interval ${signed(pair.ci_pts[0], 1)} to ${signed(pair.ci_pts[1], 1)} over ${pair.tasks} tasks: ${tone === 'good' ? 'a gain' : tone === 'bad' ? 'a regression' : 'not distinguishable from noise'}` };
  }
  if (latest) {
    const prog = (held.in_progress || [])[0];
    return { tone: 'flat', headline: rate(latest.rate), sub: `${latest.label} on held-v3, ${latest.scored}/${latest.ran} scored`,
      detail: gate && !gate.reference_same_series ? `Not paired yet: the reference ${gate.reference} is re-measured on this series${prog ? ` (${prog.done}/${prog.of} episodes)` : ''}.` : 'Paired verdict pending.' };
  }
  return { tone: 'flat', headline: '—', sub: 'no measured policy', detail: '' };
}

/** What runs where: the live units grouped by machine and role (one line per role), for the machine cards. */
export function activity(snap) {
  const by = { dgx: [], pop: [], 'colab-g4a': [], 'colab-g4b': [], api: [] };
  const ei = snap?.ei;
  const sftRound = (ei?.rounds || []).find(r => r.sft_running);
  if (sftRound) by[/g4b/.test(sftRound.host || '') ? 'colab-g4b' : /colab/.test(sftRound.host || '') ? 'colab-g4a' : 'dgx'].push({ role: 'sft', text: `SFT of ${sftRound.id}${sftRound.sft ? ` · step ${sftRound.sft.step}/${sftRound.sft.of}` : ''}` });
  for (const a of ei?.collection?.actors || []) by[a.host === 'colab' ? 'colab-g4a' : a.host].push({ role: 'collect', text: `Collecting with v${a.version} · ${a.rate_h}/h` });
  const counts = new Map();
  for (const p of (snap?.processes || []).filter(p => p.state === 'running')) {
    if (['collect', 'sft', 'driver', 'plumbing', 'housekeeping', 'colab', 'test', 'other', 'prepare'].includes(p.role)) continue;
    if (p.role === 'held' && p.machine !== 'pop') continue;        // the DGX-side launcher of a Pop evaluation
    const w = p.role === 'held' ? 'pop' : p.role === 'tuning' ? 'colab-g4b' : p.where === 'colab' ? 'colab-g4a' : p.where;
    const key = `${w}|${p.role === 'fill' ? 'Student fill' : p.title}`;
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  for (const [key, n] of counts) { const [w, text] = key.split('|'); if (by[w]) by[w].push({ role: 'other', text: n > 1 ? `${text} ×${n}` : text }); }
  return by;
}

/** Per-run {step -> record} map to [{id, points:[{x: step, y}]}] in the given run order. */
export function metricSeries(runs, ids, metric) {
  const out = [];
  for (const id of ids) {
    const steps = runs.get(id);
    if (!steps) continue;
    const pts = [];
    for (const [step, rec] of [...steps].sort((a, b) => a[0] - b[0])) {
      const y = rec.metrics?.[metric];
      if (num(y)) pts.push({ x: step, y });
    }
    if (pts.length) out.push({ id, label: id, points: pts });
  }
  return out;
}

export function metricNames(runs) {
  const names = new Set();
  for (const steps of runs.values()) for (const rec of steps.values()) for (const [k, v] of Object.entries(rec.metrics || {})) if (num(v)) names.add(k);
  return [...names].sort();
}

/** Unified-memory bar segments for a machine: one per ledger claim, other use, free. */
export function memorySegments(machine) {
  const total = machine.mem_total_gb, claims = machine.ledger?.claim_list || [];
  const claimed = claims.reduce((a, c) => a + (c.used_gb || 0), 0);
  const used = machine.mem_used_gb ?? 0;
  const segs = claims.filter(c => (c.used_gb || 0) > 0.05).map(c => ({ label: shortUnit(c.unit), gb: c.used_gb, kind: 'claim' }));
  const other = Math.max(0, used - claimed);
  if (other > 0.05) segs.push({ label: 'other (cache, desktop, unclaimed)', gb: other, kind: 'other' });
  return { total, used, segs, free: Math.max(0, total - used) };
}

/** Outcome mix of one teacher, in a fixed order. */
export const OUTCOMES = [['task_success', 'success'], ['semantic_mismatch', 'wrong answer'], ['budget_exhaustion', 'ran out of budget']];
export function outcomeParts(t) {
  const known = OUTCOMES.map(([key, label]) => ({ key, label, n: t.outcomes?.[key] || 0 }));
  const rest = Object.entries(t.outcomes || {}).filter(([k]) => !OUTCOMES.some(([o]) => o === k)).reduce((a, [, n]) => a + n, 0);
  return [...known, { key: 'other', label: 'other', n: rest }];
}

export const matches = (row, q) => !q || Object.values(row).some(v => typeof v === 'string' && v.toLowerCase().includes(q.toLowerCase()));


/* ---------- the supervising agent's inbox: what happened to a question, and whether anyone is listening ---------- */
export const LISTENER_LIVE_MS = 180_000;
const ASKS = ['ask', 'request_stop'];

/** The agent's listener heartbeat as carried by the participant's tick ({ts, listener}: both DGX epoch milliseconds). */
export function listenerStatus(tick, nowMs = null) {
  if (!tick) return { state: 'unknown', text: 'Waiting for the first tick from the DGX.' };
  if (nowMs != null && nowMs - tick.ts > 120_000) return { state: 'unknown', text: `The participant on the DGX has not reported for ${dur((nowMs - tick.ts) / 1000)}, so whether the agent listens is unknown.` };
  if (!num(tick.listener)) return { state: 'none', text: 'Nobody is listening: the agent has not started its inbox listener (inbox.sh wait).' };
  const age = Math.max(0, tick.ts - tick.listener);
  return age < LISTENER_LIVE_MS ? { state: 'live', age, text: `The agent is listening (heard ${dur(age / 1000)} ago).` }
    : { state: 'stale', age, text: `Nobody is listening: the agent's inbox listener was last heard ${dur(age / 1000)} ago.` };
}

/**
 * What became of the newest question asked about one item (kind, id): still being sent from this device, waiting for the agent, or
 * answered. `pending` are this device's unapplied commands, `times` maps ask operation id to {at}, `seen` lists answer operation ids already read.
 */
export function askState(inbox, pending, times, seen, kind, id) {
  const subj = s => s?.kind === kind && s?.id === String(id);
  const known = new Set((inbox?.asks || []).map(a => a.operationId));
  const queued = (pending || []).filter(p => p.name === 'ask' && ASKS.includes(p.input?.kind) && subj(p.input?.subject) && !known.has(p.id)).at(-1);
  if (queued) return { state: 'waiting', confirmed: false, failed: queued.state === 'uncertain', at: queued.ts, text: queued.input.text, op: queued.id };
  const a = (inbox?.asks || []).filter(x => ASKS.includes(x.kind) && subj(x.subject)).at(-1);
  if (!a) return null;
  const ans = (inbox.answers || []).filter(x => x.ask === editRef(a.operationId)).at(-1);
  const at = times?.[a.operationId]?.at ?? null;
  if (!ans) return { state: 'waiting', confirmed: true, at, text: a.text, op: a.operationId };
  return { state: 'answered', at, text: a.text, answer: ans.text, op: a.operationId, answerOp: ans.operationId, unread: !(seen || []).includes(ans.operationId) };
}

/** Counts for the badges: questions waiting for an answer and answers not yet read. */
export function inboxSummary(inbox, pending, seen) {
  const asks = (inbox?.asks || []).filter(a => ASKS.includes(a.kind));
  const answered = new Set((inbox?.answers || []).map(x => x.ask));
  const known = new Set((inbox?.asks || []).map(a => a.operationId));
  const queued = (pending || []).filter(p => p.name === 'ask' && ASKS.includes(p.input?.kind) && !known.has(p.id)).length;
  const unread = (inbox?.answers || []).filter(x => asks.some(a => editRef(a.operationId) === x.ask) && !(seen || []).includes(x.operationId)).length;
  return { waiting: asks.filter(a => !answered.has(editRef(a.operationId))).length + queued, unread, total: asks.length + queued };
}
