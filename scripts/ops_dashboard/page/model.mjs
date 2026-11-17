// Pure helpers of the ops dashboard: formatting, health verdicts, series building. No DOM, no tendril: testable in node.

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
  for (const r of snap.runs || []) {
    if (!r.active) continue;
    if (r.last_ts && now - r.last_ts > RUN_QUIET_MS) items.push({ id: `quiet:${r.id}`, level: 'warn', text: `${r.id} wrote nothing for ${dur((now - r.last_ts) / 1000)}.`, run: r.id });
    const h = latestHeld(r);
    if (h && h.p != null && h.p < 0.05 && h.delta_pts < 0) items.push({ id: `held:${r.id}:${h.step}`, level: 'crit', text: `Held regression on ${r.id} step ${h.step}: ${signed(h.delta_pts, 1)} pts (p ${fmt(h.p, 2)}).`, run: r.id });
  }
  const open = items.filter(i => !acked[i.id]);
  const level = open.some(i => i.level === 'crit') ? 'crit' : open.length ? 'warn' : 'ok';
  const title = level === 'ok' ? 'Healthy' : `${open.length} ${open.length === 1 ? 'issue needs' : 'issues need'} attention`;
  return { level, title, items, open };
}

/** Verdict of "is training improving": the held-out paired comparison first (the only metric that counts), reward trend second. */
export function improving(run) {
  if (!run) return { tone: 'flat', headline: 'No active run', detail: '' };
  const h = latestHeld(run);
  const t = trend(run.spark?.reward_mean || []);
  if (h) {
    const sig = h.p != null && h.p < 0.05;
    const tone = !sig ? 'flat' : h.delta_pts > 0 ? 'good' : 'bad';
    const word = !sig ? 'not significant' : h.delta_pts > 0 ? 'significant gain' : 'significant regression';
    return { tone, headline: `${signed(h.delta_pts, 1)} pts`, sub: 'held-out vs base', detail: `step ${h.step}, p ${fmt(h.p, 2)}: ${word}`, held: h, trend: t };
  }
  return { tone: 'flat', headline: t.state === 'early' ? 'No held eval yet' : `reward ${t.state}`, sub: '', detail: `reward_mean ${t.text}`, trend: t };
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

/** Metric presets for the training tab: [title, metric, note]. A chart is a presentation choice; the metric names are the exporter's. */
export const PRESETS = [
  ['Mean reward', 'reward_mean', 'Binary or graded task reward of the step\'s episodes; the training signal, not the held-out result.'],
  ['KL to start policy', 'kl_mean', 'k2 estimate against the start policy; telemetry, never a stop.'],
  ['Gradient norm', 'grad_norm', 'Pre-clip norm; a spike is more than 5x the run median.'],
  ['Learning rate', 'lr', 'Effective LR after warmup and the groups-with-spread scale.'],
  ['Groups with spread', 'groups_with_spread', 'Groups whose episodes differ in reward (the only ones that carry advantage).'],
  ['Expected </think> closes vs base', 'behaviour.close_ratio', 'Behaviour probe: 1.0 means unchanged, drift away from 1 is the early-close signature.'],
  ['Base-sample logp change', 'behaviour.logp_change', 'Behaviour probe on 48 start-policy calls.'],
  ['Learner tokens per second', 'learner_tokens_per_s', 'Learner throughput.'],
  ['Actor tokens per second', 'actor_tokens_per_s_collect_phase', 'Actor generation rate (collect phase).'],
  ['Learner CUDA peak', 'cuda_peak_gb', 'Peak GPU memory of the step (GB, unified memory).'],
];
